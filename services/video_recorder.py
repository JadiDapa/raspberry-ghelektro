"""
video_recorder.py — records a single MP4 of a Data Collection sweep.

A background thread samples the latest JPEG from the camera's shared frame buffer
(services/camera.py) at the camera's configured FPS, decodes it, and writes frames
to a cv2.VideoWriter. Because it reads the same buffer the MJPEG stream/snapshot use,
it works even in camera stub mode (the stub still fills the buffer), so a real .mp4
is produced for testing without a webcam.

One recording at a time — a dataset session is the only caller and the global
one-session-at-a-time guard (routers/sessions.py) keeps that true.

The file is written under settings.videos_dir. The dataset session service uploads
it to the dashboard and, on upload failure, keeps it on disk (path is logged) so
footage is never lost.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from config import settings
from services import camera as camera_service

_thread: threading.Thread | None = None
_running = False
_state: dict = {}


def _record_loop(path: str, fps: float) -> None:
    global _running
    interval = 1.0 / fps if fps > 0 else 0.1
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    start = time.monotonic()

    # Wait briefly for the first frame so we can size the writer to the real frame.
    first: np.ndarray | None = None
    deadline = time.monotonic() + 5.0
    while _running and time.monotonic() < deadline:
        jpeg = camera_service.latest_jpeg()
        if jpeg is not None:
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                first = decoded
                break
        time.sleep(0.05)

    if first is None:
        print("[recorder] no frame available — recording produced 0 frames")
        _state.update({"path": path, "frame_count": 0, "duration_sec": 0.0})
        return

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[recorder] WARNING: could not open VideoWriter for {path}")
        _state.update({"path": path, "frame_count": 0, "duration_sec": 0.0})
        return

    writer.write(first)
    frame_count = 1
    print(f"[recorder] recording → {path} @ {w}x{h} {fps}fps")

    prev = time.monotonic()
    while _running:
        now = time.monotonic()
        if now - prev < interval:
            time.sleep(0.005)
            continue
        prev = now
        jpeg = camera_service.latest_jpeg()
        if jpeg is None:
            continue
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            continue
        # Guard against a mid-stream resolution change (camera reconnect).
        if decoded.shape[:2] != (h, w):
            decoded = cv2.resize(decoded, (w, h))
        writer.write(decoded)
        frame_count += 1

    writer.release()
    duration = time.monotonic() - start
    _state.update({"path": path, "frame_count": frame_count, "duration_sec": round(duration, 2)})
    print(f"[recorder] stopped → {frame_count} frames, {duration:.1f}s")


def start(session_id: int) -> str:
    """Begin recording in a background thread. Returns the target file path."""
    global _thread, _running, _state
    if _running:
        raise RuntimeError("recorder already running")

    Path(settings.videos_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = str(Path(settings.videos_dir) / f"session_{session_id}_{ts}.mp4")

    _state = {"path": path, "frame_count": 0, "duration_sec": 0.0}
    _running = True
    _thread = threading.Thread(
        target=_record_loop, args=(path, float(settings.camera_fps)), daemon=True, name="video-recorder"
    )
    _thread.start()
    return path


def stop() -> dict:
    """Stop recording and return {path, frame_count, duration_sec}."""
    global _running, _thread
    if not _running:
        return dict(_state)
    _running = False
    if _thread is not None:
        _thread.join(timeout=10.0)
        _thread = None
    return dict(_state)


def is_recording() -> bool:
    return _running
