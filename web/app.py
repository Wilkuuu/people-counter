from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import csv
import json
import subprocess
import shutil
import sys
import threading
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from counter import PeopleCounterPipeline
from main import load_config
from reporting import generate_html_report


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(getattr(sys, "_MEIPASS"))
    RUNTIME_ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent
    RUNTIME_ROOT = ROOT
TEMPLATES = Jinja2Templates(directory=str(ROOT / "web" / "templates"))
OUTPUTS_ROOT = RUNTIME_ROOT / "outputs" / "jobs"
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUNTIME_ROOT / "people-counter.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("people_counter.web")

app = FastAPI(title="People Counter Web")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_ROOT.parent)), name="outputs")

JOBS: dict[str, dict] = {}


def _ensure_web_video(src_video: Path, dst_video: Path) -> None:
    if dst_video.exists():
        return
    if shutil.which("ffmpeg") is None:
        # ffmpeg is optional; skip conversion when unavailable.
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_video),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_video),
    ]
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # Guard for environments where ffmpeg disappears at runtime.
        return


def _run_job(job_id: str, config_path: Path, video_path: Path, output_dir: Path) -> None:
    try:
        def on_progress(data: dict) -> None:
            JOBS[job_id]["progress"] = {
                "processed_frames": int(data.get("processed_frames", 0)),
                "total_frames": int(data.get("total_frames", 0)) if data.get("total_frames") else None,
                "percent": float(data.get("percent", 0.0)) if data.get("percent") is not None else None,
            }

        cfg = load_config(config_path)
        LOGGER.info("Starting job_id=%s preset=%s video=%s", job_id, JOBS[job_id]["preset"], video_path)
        pipeline = PeopleCounterPipeline(cfg=cfg, checkpoint_file=output_dir / "checkpoint.json", resume=False)
        summary = pipeline.run(video_path=video_path, output_dir=output_dir, progress_callback=on_progress)
        _ensure_web_video(video_path, output_dir / "web_preview.mp4")
        generate_html_report(output_dir)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["summary"] = summary
        LOGGER.info("Completed job_id=%s total_crossings=%s", job_id, summary.get("total_crossings"))
    except Exception as exc:  # pragma: no cover
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        LOGGER.exception("Job failed job_id=%s", job_id)


def _read_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    out: list[dict] = []
    with events_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            out.append(row)
    return out


@app.get("/")
def index(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    request: Request,
    video: UploadFile = File(...),
    preset: str = Form("balanced"),
):
    LOGGER.info("Received analyze request preset=%s filename=%s", preset, video.filename or "input.mp4")
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    video_path = output_dir / f"input{suffix}"
    with video_path.open("wb") as fp:
        shutil.copyfileobj(video.file, fp)

    config_template = ROOT / "config.example.yaml"
    config_text = config_template.read_text(encoding="utf-8")
    config_text = config_text.replace('preset: "balanced"', f'preset: "{preset}"')
    config_path = output_dir / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    JOBS[job_id] = {
        "status": "running",
        "output_dir": str(output_dir),
        "preset": preset,
        "progress": {"processed_frames": 0, "total_frames": None, "percent": 0.0},
    }
    thread = threading.Thread(target=_run_job, args=(job_id, config_path, video_path, output_dir), daemon=True)
    thread.start()

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}")
def job_result(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return TEMPLATES.TemplateResponse(
            request,
            "result.html",
            {"request": request, "job_id": job_id, "status": "missing", "events": [], "summary": None},
        )
    output_dir = Path(job["output_dir"])
    summary = None
    events = []
    if (output_dir / "summary.json").exists():
        with (output_dir / "summary.json").open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
    if (output_dir / "events.csv").exists():
        events = _read_events(output_dir / "events.csv")
    preview_candidate = output_dir / "preview.mp4"
    web_candidate = output_dir / "web_preview.mp4"
    input_candidate = output_dir / "input.mp4"
    if preview_candidate.exists():
        video_rel = f"/outputs/jobs/{job_id}/preview.mp4"
    elif web_candidate.exists():
        video_rel = f"/outputs/jobs/{job_id}/web_preview.mp4"
    elif input_candidate.exists():
        video_rel = f"/outputs/jobs/{job_id}/input.mp4"
    else:
        video_rel = None

    return TEMPLATES.TemplateResponse(
        request,
        "result.html",
        {
            "request": request,
            "job_id": job_id,
            "status": job["status"],
            "error": job.get("error"),
            "progress": job.get("progress", {}),
            "summary": summary,
            "events": events,
            "video_url": video_rel,
            "report_url": f"/outputs/jobs/{job_id}/report.html",
            "events_url": f"/outputs/jobs/{job_id}/events.csv",
            "summary_url": f"/outputs/jobs/{job_id}/summary.json",
        },
    )


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "missing"}
    return {
        "status": job.get("status", "running"),
        "progress": job.get("progress", {"processed_frames": 0, "total_frames": None, "percent": 0.0}),
        "error": job.get("error"),
    }
