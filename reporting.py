from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd


def _format_seconds(value: float) -> str:
    total = int(value)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_html_report(output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.csv"
    report_path = output_dir / "report.html"

    with summary_path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)

    if events_path.exists():
        events = pd.read_csv(events_path)
    else:
        events = pd.DataFrame(columns=["frame_idx", "time_seconds", "track_id", "direction"])

    rows = []
    for _, row in events.head(500).iterrows():
        t = float(row["time_seconds"]) if "time_seconds" in row else 0.0
        rows.append(
            "<tr>"
            f"<td>{int(row['frame_idx'])}</td>"
            f"<td>{_format_seconds(t)}</td>"
            f"<td>{int(row['track_id'])}</td>"
            f"<td>{row['direction']}</td>"
            "</tr>"
        )
    events_table = "\n".join(rows) if rows else "<tr><td colspan='4'>Brak zdarzen</td></tr>"

    html = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>People Counter Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #f9fafb; }}
    .label {{ font-size: 12px; color: #6b7280; margin-bottom: 6px; }}
    .value {{ font-size: 22px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 14px; }}
    .meta {{ color: #4b5563; margin-bottom: 10px; }}
  </style>
</head>
<body>
  <h1>Raport zliczania osob</h1>
  <div class="meta">
    <div><strong>Video:</strong> {summary.get("video_path", "-")}</div>
    <div><strong>Preset:</strong> {summary.get("preset", "-")} | <strong>Model:</strong> {summary.get("model_weights", "-")}</div>
    <div><strong>Wygenerowano:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Total crossings</div><div class="value">{summary.get("total_crossings", 0)}</div></div>
    <div class="card"><div class="label">Left to right</div><div class="value">{summary.get("from_left", 0)}</div></div>
    <div class="card"><div class="label">Right to left</div><div class="value">{summary.get("from_right", 0)}</div></div>
    <div class="card"><div class="label">Events</div><div class="value">{summary.get("events_count", 0)}</div></div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Processed frames</div><div class="value">{summary.get("processed_frames", 0)}</div></div>
    <div class="card"><div class="label">Effective FPS</div><div class="value">{summary.get("effective_fps", 0)}</div></div>
    <div class="card"><div class="label">Elapsed seconds</div><div class="value">{summary.get("elapsed_seconds", 0)}</div></div>
    <div class="card"><div class="label">Frame step</div><div class="value">{summary.get("frame_step", 1)}</div></div>
  </div>

  <h2>Zdarzenia</h2>
  <table>
    <thead><tr><th>Frame</th><th>Czas</th><th>Track ID</th><th>Kierunek</th></tr></thead>
    <tbody>
      {events_table}
    </tbody>
  </table>
  <p><a href="events.csv">events.csv</a> | <a href="summary.json">summary.json</a></p>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path
