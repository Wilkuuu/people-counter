from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from counter import CounterConfig, PeopleCounterPipeline
from reporting import generate_html_report


PRESETS: Dict[str, Dict[str, Any]] = {
    "speed": {
        "model": {"weights": "yolov8n.pt", "conf": 0.45, "iou": 0.50},
        "processing": {"frame_step": 2, "min_crossing_gap_frames": 10},
        "tracker": {"agnostic_nms": True},
    },
    "balanced": {
        "model": {"weights": "yolov8m.pt", "conf": 0.35, "iou": 0.55},
        "processing": {"frame_step": 1, "min_crossing_gap_frames": 14},
        "tracker": {"agnostic_nms": False},
    },
    "accuracy": {
        "model": {"weights": "yolov8l.pt", "conf": 0.25, "iou": 0.60},
        "processing": {"frame_step": 1, "min_crossing_gap_frames": 18},
        "tracker": {"agnostic_nms": False},
    },
}


def _best_cuda_device_index() -> int | None:
    if not torch.cuda.is_available():
        return None
    count = torch.cuda.device_count()
    if count <= 0:
        return None

    best_idx = 0
    best_total_mem = -1
    for idx in range(count):
        try:
            total_mem = int(torch.cuda.get_device_properties(idx).total_memory)
        except Exception:
            total_mem = 0
        if total_mem > best_total_mem:
            best_total_mem = total_mem
            best_idx = idx
    return best_idx


def _warn_cuda_unavailable(requested_device: str) -> None:
    print(
        f"[warn] CUDA device '{requested_device}' requested, but CUDA is unavailable. "
        "Falling back to 'cpu'."
    )


def resolve_device(requested_device: str) -> str:
    normalized = str(requested_device or "").strip().lower()
    if normalized in {"", "auto", "cuda"}:
        best_idx = _best_cuda_device_index()
        if best_idx is None:
            print("[info] CUDA not available. Using 'cpu'.")
            return "cpu"
        selected = str(best_idx)
        print(f"[info] Using CUDA device '{selected}'.")
        return selected

    if normalized == "cpu":
        return "cpu"

    if re.fullmatch(r"\d+(,\d+)*", normalized):
        if not torch.cuda.is_available():
            _warn_cuda_unavailable(requested_device)
            return "cpu"

        count = torch.cuda.device_count()
        requested_ids = [int(part) for part in normalized.split(",")]
        valid_ids = [idx for idx in requested_ids if 0 <= idx < count]
        if not valid_ids:
            best_idx = _best_cuda_device_index()
            fallback = "cpu" if best_idx is None else str(best_idx)
            print(
                f"[warn] Requested CUDA device(s) '{requested_device}' are invalid for this host "
                f"(available: 0..{max(count - 1, 0)}). Falling back to '{fallback}'."
            )
            return fallback
        return ",".join(str(idx) for idx in valid_ids)

    print(f"[warn] Unknown device '{requested_device}'. Falling back to auto-detection.")
    return resolve_device("auto")


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, preset_override: str | None = None) -> CounterConfig:
    with path.open("r", encoding="utf-8") as fp:
        raw: Dict[str, Any] = yaml.safe_load(fp) or {}

    preset_name = str(preset_override or raw.get("preset", "balanced")).lower()
    preset_update = PRESETS.get(preset_name, PRESETS["balanced"])
    raw = _deep_merge(raw, preset_update)

    model = raw["model"]
    processing = raw["processing"]
    counting = raw["counting"]
    output = raw["output"]
    line = counting.get("line", {})
    zone = counting.get("zone", {})
    mode = str(counting.get("mode", "zone")).lower()
    tracker_cfg = raw.get("tracker_config", {})
    if not isinstance(tracker_cfg, dict):
        tracker_cfg = {}

    return CounterConfig(
        model_weights=str(model.get("weights", "yolov8n.pt")),
        model_conf=float(model.get("conf", 0.35)),
        model_iou=float(model.get("iou", 0.50)),
        model_device=resolve_device(str(model.get("device", "auto"))),
        tracker=str(model.get("tracker", "bytetrack.yaml")),
        tracker_config=tracker_cfg,
        preset=preset_name if preset_name in PRESETS else "balanced",
        frame_step=max(int(processing.get("frame_step", 1)), 1),
        start_frame=max(int(processing.get("start_frame", 0)), 0),
        max_frames=processing.get("max_frames", None),
        checkpoint_every_frames=max(int(processing.get("checkpoint_every_frames", 500)), 0),
        min_crossing_gap_frames=max(int(processing.get("min_crossing_gap_frames", 12)), 0),
        show_progress=bool(processing.get("show_progress", True)),
        counting_mode=mode if mode in {"zone", "line"} else "zone",
        line_p1=(int(line.get("x1", 640)), int(line.get("y1", 120))),
        line_p2=(int(line.get("x2", 640)), int(line.get("y2", 680))),
        zone_split_x=(int(zone["split_x"]) if zone.get("split_x") is not None else None),
        zone_margin_px=max(int(zone.get("margin_px", 40)), 0),
        save_preview_video=bool(output.get("save_preview_video", False)),
        preview_fps=output.get("preview_fps", None),
        preview_codec=str(output.get("preview_codec", "mp4v")),
        events_csv_name=str(output.get("events_csv_name", "events.csv")),
        summary_json_name=str(output.get("summary_json_name", "summary.json")),
        preview_video_name=str(output.get("preview_video_name", "preview.mp4")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-based people in/out counter")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default="outputs", help="Directory for reports and optional preview")
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Optional checkpoint JSON path. Defaults to <output-dir>/checkpoint.json",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint frame if available")
    parser.add_argument(
        "--device",
        default=None,
        help='Override config device, e.g. "auto", "0", "0,1", or "cpu"',
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["speed", "balanced", "accuracy"],
        help="Quality/speed preset overriding config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint_file).expanduser().resolve()
        if args.checkpoint_file
        else output_dir / "checkpoint.json"
    )

    cfg = load_config(config_path, preset_override=args.preset)
    if args.device is not None:
        cfg.model_device = resolve_device(args.device)

    pipeline = PeopleCounterPipeline(cfg=cfg, checkpoint_file=checkpoint_path, resume=args.resume)
    summary = pipeline.run(video_path=video_path, output_dir=output_dir)
    generate_html_report(output_dir=output_dir)

    print("=== PEOPLE COUNTER SUMMARY ===")
    print(f"video: {video_path}")
    print(f"preset:    {summary['preset']}")
    print(f"TOTAL:     {summary['total_crossings']}")
    print(f"L->R:      {summary['from_left']}")
    print(f"R->L:      {summary['from_right']}")
    print(f"events: {summary['events_count']}")
    print(f"processed_frames: {summary['processed_frames']}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']}")
    print(f"effective_fps: {summary['effective_fps']}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
