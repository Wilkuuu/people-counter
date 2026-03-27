from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    tracker_config: Dict[str, Any]
    preset: str
    frame_step: int
    start_frame: int
    max_frames: Optional[int]
    checkpoint_every_frames: int
    min_crossing_gap_frames: int
    show_progress: bool
    counting_mode: str
    line_p1: Point
    line_p2: Point
    zone_split_x: Optional[int]
    zone_margin_px: int
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
        self.from_left = 0
        self.from_right = 0
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

        direction = "LEFT_TO_RIGHT" if previous_side < 0 < current_side else "RIGHT_TO_LEFT"
        if direction == "LEFT_TO_RIGHT":
            self.from_left += 1
        else:
            self.from_right += 1

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
            "total_crossings": self.from_left + self.from_right,
            "from_left": self.from_left,
            "from_right": self.from_right,
        }


class ZoneCrossingCounter:
    def __init__(self, split_x: int, margin_px: int, min_crossing_gap_frames: int) -> None:
        self.split_x = split_x
        self.margin_px = max(0, margin_px)
        self.min_crossing_gap_frames = min_crossing_gap_frames
        self.from_left = 0
        self.from_right = 0
        self.events: List[CountingEvent] = []
        self._last_zone_by_track: Dict[int, str] = {}
        self._last_counted_frame_by_track: Dict[int, int] = {}

    def _zone(self, x: float) -> Optional[str]:
        if x < self.split_x - self.margin_px:
            return "left"
        if x > self.split_x + self.margin_px:
            return "right"
        return None

    def update_track(self, track_id: int, center_xy: Tuple[float, float], frame_idx: int, time_seconds: float) -> None:
        zone = self._zone(center_xy[0])
        if zone is None:
            return

        previous_zone = self._last_zone_by_track.get(track_id)
        self._last_zone_by_track[track_id] = zone
        if previous_zone is None or previous_zone == zone:
            return

        last_counted_frame = self._last_counted_frame_by_track.get(track_id, -10**12)
        if frame_idx - last_counted_frame < self.min_crossing_gap_frames:
            return

        direction = "LEFT_TO_RIGHT" if previous_zone == "left" and zone == "right" else "RIGHT_TO_LEFT"
        if direction == "LEFT_TO_RIGHT":
            self.from_left += 1
        else:
            self.from_right += 1
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
            "total_crossings": self.from_left + self.from_right,
            "from_left": self.from_left,
            "from_right": self.from_right,
        }


class PeopleCounterPipeline:
    def __init__(self, cfg: CounterConfig, checkpoint_file: Optional[Path] = None, resume: bool = False) -> None:
        self.cfg = cfg
        self.checkpoint_file = checkpoint_file
        self.resume = resume
        self.model = YOLO(cfg.model_weights)
        self.counter = None

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
    def _draw_overlay(frame, cfg: CounterConfig, summary: Dict[str, int], width: int, height: int) -> None:
        if cfg.counting_mode == "zone":
            split_x = cfg.zone_split_x if cfg.zone_split_x is not None else width // 2
            margin = cfg.zone_margin_px
            cv2.line(frame, (split_x, 0), (split_x, height), (0, 255, 255), 2)
            if margin > 0:
                cv2.line(frame, (split_x - margin, 0), (split_x - margin, height), (100, 100, 255), 1)
                cv2.line(frame, (split_x + margin, 0), (split_x + margin, height), (100, 100, 255), 1)
        else:
            cv2.line(frame, cfg.line_p1, cfg.line_p2, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"TOTAL: {summary['total_crossings']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"L->R: {summary['from_left']}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (60, 220, 60),
            2,
        )
        cv2.putText(
            frame,
            f"R->L: {summary['from_right']}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (50, 180, 255),
            2,
        )

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        started_at = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if self.cfg.counting_mode == "zone":
            split_x = self.cfg.zone_split_x if self.cfg.zone_split_x is not None else (width // 2 if width > 0 else 640)
            self.counter = ZoneCrossingCounter(
                split_x=split_x,
                margin_px=self.cfg.zone_margin_px,
                min_crossing_gap_frames=self.cfg.min_crossing_gap_frames,
            )
        else:
            self.counter = LineCrossingCounter(self.cfg.line_p1, self.cfg.line_p2, self.cfg.min_crossing_gap_frames)

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
                    **self.cfg.tracker_config,
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
                self._draw_overlay(frame, self.cfg, self.counter.summary(), width, height)
                preview_writer.write(frame)

            processed_frames += 1
            if pbar is not None:
                pbar.update(1)
            if progress_callback is not None and (processed_frames == 1 or processed_frames % 10 == 0):
                progress_callback(
                    {
                        "processed_frames": processed_frames,
                        "total_frames": total_frames,
                        "percent": (processed_frames / total_frames * 100.0) if total_frames > 0 else None,
                    }
                )

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
        if progress_callback is not None:
            progress_callback(
                {
                    "processed_frames": processed_frames,
                    "total_frames": total_frames,
                    "percent": 100.0,
                }
            )

        events_path = output_dir / self.cfg.events_csv_name
        pd.DataFrame([asdict(e) for e in self.counter.events]).to_csv(events_path, index=False)

        summary = self.counter.summary()
        elapsed_seconds = max(time.time() - started_at, 0.001)
        effective_fps = processed_frames / elapsed_seconds
        summary_payload = {
            **summary,
            "video_path": str(video_path),
            "processed_frames": processed_frames,
            "start_frame": start_frame,
            "frame_step": self.cfg.frame_step,
            "events_count": len(self.counter.events),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "effective_fps": round(effective_fps, 3),
            "model_weights": self.cfg.model_weights,
            "preset": self.cfg.preset,
        }
        summary_path = output_dir / self.cfg.summary_json_name
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, ensure_ascii=True, indent=2)

        return summary_payload
