from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from counter import CounterConfig, PeopleCounterPipeline


def _load_config(path: Path) -> CounterConfig:
    with path.open("r", encoding="utf-8") as fp:
        raw: Dict[str, Any] = yaml.safe_load(fp)

    model = raw["model"]
    processing = raw["processing"]
    counting = raw["counting"]
    output = raw["output"]
    line = counting["line"]

    return CounterConfig(
        model_weights=str(model.get("weights", "yolov8n.pt")),
        model_conf=float(model.get("conf", 0.35)),
        model_iou=float(model.get("iou", 0.50)),
        model_device=str(model.get("device", "0")),
        tracker=str(model.get("tracker", "bytetrack.yaml")),
        frame_step=max(int(processing.get("frame_step", 1)), 1),
        start_frame=max(int(processing.get("start_frame", 0)), 0),
        max_frames=processing.get("max_frames", None),
        checkpoint_every_frames=max(int(processing.get("checkpoint_every_frames", 500)), 0),
        min_crossing_gap_frames=max(int(processing.get("min_crossing_gap_frames", 12)), 0),
        show_progress=bool(processing.get("show_progress", True)),
        line_p1=(int(line["x1"]), int(line["y1"])),
        line_p2=(int(line["x2"]), int(line["y2"])),
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
    parser.add_argument("--device", default=None, help='Override config device, e.g. "0" or "cpu"')
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

    cfg = _load_config(config_path)
    if args.device is not None:
        cfg.model_device = args.device

    pipeline = PeopleCounterPipeline(cfg=cfg, checkpoint_file=checkpoint_path, resume=args.resume)
    summary = pipeline.run(video_path=video_path, output_dir=output_dir)

    print("=== PEOPLE COUNTER SUMMARY ===")
    print(f"video: {video_path}")
    print(f"IN:   {summary['total_in']}")
    print(f"OUT:  {summary['total_out']}")
    print(f"NET:  {summary['net']}")
    print(f"events: {summary['events_count']}")
    print(f"processed_frames: {summary['processed_frames']}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
