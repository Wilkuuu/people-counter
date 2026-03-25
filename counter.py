from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import time

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


Point = Tuple[int, int]


@dataclass
class CounterConfig:
    model_weights: str
    model_conf: float
    model_iou: float
    model_device: str
    tracker: str
    frame_step: int
    start_frame: int
    max_frames: Optional[int]
    checkpoint_every_frames: int
    min_crossing_gap_frames: int
    show_progress: bool
    line_p1: Point
    line_p2: Point
    save_preview_video: bool
    preview_fps: Optional[float]
    preview_codec: str
    events_csv_name: str
    summary_json_name: str
    preview_video_name: str


@dataclass
class CountingEvent:
    frame_idx: int
    time_seconds: float
    track_id: int
    direction: str


def _point_side(point: Tuple[float, float], line_p1: Point, line_p2: Point) -> float:
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


class LineCrossingCounter:
    def __init__(self, line_p1: Point, line_p2: Point, min_crossing_gap_frames: int) -> None:
        self.line_p1 = line_p1
        self.line_p2 = line_p2
        self.min_crossing_gap_frames = min_crossing_gap_frames
        self.total_in = 0
        self.total_out = 0
        self.events: List[CountingEvent] = []

        self._last_side_by_track: Dict[int, float] = {}
        self._last_counted_frame_by_track: Dict[int, int] = {}

    def update_track(self, track_id: int, center_xy: Tuple[float, float], frame_idx: int, time_seconds: float) -> None:
        current_side = _point_side(center_xy, self.line_p1, self.line_p2)
        if math.isclose(current_side, 0.0, abs_tol=1e-6):
            return

        previous_side = self._last_side_by_track.get(track_id)
        self._last_side_by_track[track_id] = current_side
        if previous_side is None:
            return

        crossed = (previous_side < 0 < current_side) or (previous_side > 0 > current_side)
        if not crossed:
            return

        last_counted_frame = self._last_counted_frame_by_track.get(track_id, -10**12)
        if frame_idx - last_counted_frame < self.min_crossing_gap_frames:
            return

        direction = "IN" if previous_side < 0 < current_side else "OUT"
        if direction == "IN":
            self.total_in += 1
        else:
            self.total_out += 1

        self._last_counted_frame_by_track[track_id] = frame_idx
        self.events.append(
            CountingEvent(
                frame_idx=frame_idx,
                time_seconds=time_seconds,
                track_id=track_id,
                direction=direction,
            )
        )

    def summary(self) -> Dict[str, int]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "net": self.total_in - self.total_out,
        }


class PeopleCounterPipeline:
    def __init__(self, cfg: CounterConfig, checkpoint_file: Optional[Path] = None, resume: bool = False) -> None:
        self.cfg = cfg
        self.checkpoint_file = checkpoint_file
        self.resume = resume
        self.model = YOLO(cfg.model_weights)
        self.counter = LineCrossingCounter(cfg.line_p1, cfg.line_p2, cfg.min_crossing_gap_frames)

    def _resolve_start_frame(self) -> int:
        start = self.cfg.start_frame
        if not self.resume or self.checkpoint_file is None or not self.checkpoint_file.exists():
            return start
        with self.checkpoint_file.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return max(start, int(data.get("last_processed_frame", start)))

    def _save_checkpoint(self, frame_idx: int) -> None:
        if self.checkpoint_file is None:
            return
        payload = {
            "last_processed_frame": frame_idx,
            "updated_at_unix": int(time.time()),
            "summary": self.counter.summary(),
        }
        with self.checkpoint_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)

    @staticmethod
    def _draw_overlay(frame, line_p1: Point, line_p2: Point, summary: Dict[str, int]) -> None:
        cv2.line(frame, line_p1, line_p2, (0, 255, 255), 2)
        cv2.putText(frame, f"IN: {summary['total_in']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 60), 2)
        cv2.putText(frame, f"OUT: {summary['total_out']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 180, 255), 2)
        cv2.putText(frame, f"NET: {summary['net']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def run(self, video_path: Path, output_dir: Path) -> Dict[str, int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        start_frame = self._resolve_start_frame()
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        preview_writer = None
        if self.cfg.save_preview_video:
            preview_path = output_dir / self.cfg.preview_video_name
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.preview_codec)
            preview_fps = self.cfg.preview_fps if self.cfg.preview_fps else src_fps
            preview_writer = cv2.VideoWriter(str(preview_path), fourcc, preview_fps, (width, height))

        processed_frames = 0
        pbar = None
        if self.cfg.show_progress:
            max_possible = max(total_frames - start_frame, 0) if total_frames > 0 else None
            pbar = tqdm(total=max_possible, desc="Processing", unit="frame")

        frame_idx = start_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if self.cfg.max_frames is not None and processed_frames >= self.cfg.max_frames:
                break

            should_process = (frame_idx - start_frame) % self.cfg.frame_step == 0
            if should_process:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.cfg.tracker,
                    classes=[0],  # person class
                    conf=self.cfg.model_conf,
                    iou=self.cfg.model_iou,
                    device=self.cfg.model_device,
                    verbose=False,
                )
                result = results[0]
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().tolist()
                    for track_id, box in zip(ids, xyxy):
                        x1, y1, x2, y2 = box
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        t_sec = frame_idx / src_fps if src_fps > 0 else 0.0
                        self.counter.update_track(track_id, center, frame_idx, t_sec)
                        if preview_writer is not None:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
                            cv2.putText(
                                frame,
                                f"ID {track_id}",
                                (int(x1), max(int(y1) - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 220, 0),
                                2,
                            )

            if preview_writer is not None:
                self._draw_overlay(frame, self.cfg.line_p1, self.cfg.line_p2, self.counter.summary())
                preview_writer.write(frame)

            processed_frames += 1
            if pbar is not None:
                pbar.update(1)

            if (
                self.cfg.checkpoint_every_frames > 0
                and processed_frames % self.cfg.checkpoint_every_frames == 0
            ):
                self._save_checkpoint(frame_idx)

            frame_idx += 1

        cap.release()
        if preview_writer is not None:
            preview_writer.release()
        if pbar is not None:
            pbar.close()

        self._save_checkpoint(frame_idx)

        events_path = output_dir / self.cfg.events_csv_name
        pd.DataFrame([asdict(e) for e in self.counter.events]).to_csv(events_path, index=False)

        summary = self.counter.summary()
        summary_payload = {
            **summary,
            "video_path": str(video_path),
            "processed_frames": processed_frames,
            "start_frame": start_frame,
            "frame_step": self.cfg.frame_step,
            "events_count": len(self.counter.events),
        }
        summary_path = output_dir / self.cfg.summary_json_name
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, ensure_ascii=True, indent=2)

        return summary_payload
