"""
Web server for testing the CNN-LSTM surveillance model.

Two modes:
  * Live surveillance: the browser streams webcam frames to /api/live/frame
  * Test a clip: upload a recorded video to /api/analyze

Run (from inside this folder):
    uvicorn app:app --port 8000
then open http://localhost:8000
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

import cv2
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so "from pipeline import ..." works from any cwd
# Defaults assume this folder sits next to the notebook's outputs/ folder
# (e.g. /Users/mikie/Downloads/surveillance_webapp). Override with env vars if not.
MODEL_PATH = os.environ.get("MODEL_PATH", str(HERE.parent / "outputs" / "cnn_lstm.pt"))
YOLO_PATH = os.environ.get("YOLO_PATH", str(HERE.parent / "yolov8n.pt"))
if not os.path.exists(YOLO_PATH):
    YOLO_PATH = "yolov8n.pt"  # ultralytics will download it

ALLOWED_EXT = (".mp4", ".mov", ".avi", ".m4v", ".mkv")
MAX_UPLOAD_MB = 200

app = FastAPI(title="Threat detection model tester")
_pipeline = None
_load_error = None


def get_pipeline():
    global _pipeline, _load_error
    if _pipeline is None and _load_error is None:
        try:
            from pipeline import SurveillancePipeline
            _pipeline = SurveillancePipeline(MODEL_PATH, YOLO_PATH)
            print(f"Model loaded from {MODEL_PATH} on {_pipeline.device}")
        except Exception as e:  # surface the reason in the UI instead of crashing
            _load_error = str(e)
            print("MODEL LOAD FAILED:", e)
    return _pipeline


@app.on_event("startup")
def _startup():
    get_pipeline()


def _jpeg_b64(img, width=None):
    if width and img.shape[1] > width:
        h = int(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    from config import ALL_ZONES, CLASSES, RESTRICTED_ZONES
    p = get_pipeline()
    return {
        "ready": p is not None,
        "error": _load_error,
        "device": str(p.device) if p else None,
        "model_path": MODEL_PATH,
        "classes": CLASSES,
        "zones": [{"id": z, "restricted": z in RESTRICTED_ZONES} for z in ALL_ZONES],
    }


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), zone: str = Form("gate_area")):
    p = get_pipeline()
    if p is None:
        raise HTTPException(503, f"Model isn't loaded: {_load_error}")

    suffix = Path(video.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Upload a video file ({', '.join(ALLOWED_EXT)}). Got '{suffix}'.")

    data = await video.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Clip is larger than {MAX_UPLOAD_MB} MB. Trim it and try again.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = await run_in_threadpool(p.analyze, tmp_path, zone)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp_path)

    result["filename"] = video.filename
    result["previews"] = [
        {"crop": _jpeg_b64(pv["crop"], 112), "frame": _jpeg_b64(pv["frame"], 640),
         "conf": round(pv["conf"], 3), "t": round(pv["t"], 2)}
        for pv in result["previews"]
    ]
    return result


# ---- Live surveillance -------------------------------------------------------------
_sessions = {}
SESSION_IDLE_S = 60


def _drop_idle_sessions():
    now = time.time()
    for sid in [k for k, v in _sessions.items() if now - v.last_seen > SESSION_IDLE_S]:
        _sessions.pop(sid, None)


@app.post("/api/live/start")
def live_start():
    from live import LiveSession
    p = get_pipeline()
    if p is None:
        raise HTTPException(503, f"Model isn't loaded: {_load_error}")
    _drop_idle_sessions()
    sid = uuid.uuid4().hex
    _sessions[sid] = LiveSession(p)
    return {"session": sid}


@app.post("/api/live/frame")
async def live_frame(request: Request, session: str, zone: str = "gate_area",
                     threshold: float = 0.7):
    sess = _sessions.get(session)
    if sess is None:
        raise HTTPException(404, "Surveillance session expired. Stop and start surveillance again.")
    data = await request.body()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR, like training
    if frame is None:
        raise HTTPException(400, "Couldn't decode the camera frame.")
    return await run_in_threadpool(sess.process, frame, zone, threshold)


@app.post("/api/live/stop")
def live_stop(session: str):
    _sessions.pop(session, None)
    return {"stopped": True}
