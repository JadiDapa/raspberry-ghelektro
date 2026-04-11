"""
Camera service — USB webcam on the gantry tool head.

Two responsibilities:
  1. MJPEG live stream  — continuously reads frames, streams to dashboard
  2. Snapshot capture   — grabs a fresh frame after gantry stabilizes, saves to disk

Stub mode:
  If no camera is found on startup, generates a synthetic frame so the
  rest of the system still works during development.
  Controlled by STUB_MODE — the service sets it automatically based on
  whether OpenCV can open the device. You don't need to flip it manually.
"""

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from config import settings

# ─── Config (from settings) ───────────────────────────────────────────────────

CAMERA_INDEX = settings.camera_index
FRAME_WIDTH = settings.camera_width
FRAME_HEIGHT = settings.camera_height
TARGET_FPS = settings.camera_fps
JPEG_QUALITY = settings.camera_jpeg_quality

# Set automatically on startup — True if camera couldn't be opened
STUB_MODE = False

# ─── Shared frame buffer ──────────────────────────────────────────────────────
# One background thread writes here. All stream clients and snapshot calls read
# from this single buffer — only one cv2.VideoCapture ever runs.


class FrameBuffer:
    def __init__(self):
        self._frame: bytes | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def write(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes
        self._event.set()

    def read(self) -> bytes | None:
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout: float = 2.0) -> bytes | None:
        self._event.clear()  # ← clear BEFORE waiting
        self._event.wait(timeout=timeout)
        return self.read()


_buffer = FrameBuffer()
_capture_thread: threading.Thread | None = None
_running = False
_executor = ThreadPoolExecutor(max_workers=1)


# ─── Stub frame generator ─────────────────────────────────────────────────────


def _generate_stub_frame() -> bytes:
    """Dark grey canvas with timestamp — used when no camera is connected."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)

    cv2.putText(
        frame,
        "FarmBot Camera",
        (int(FRAME_WIDTH * 0.28), int(FRAME_HEIGHT * 0.33)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 200, 100),
        2,
    )
    cv2.putText(
        frame,
        "[ STUB MODE ]",
        (int(FRAME_WIDTH * 0.33), int(FRAME_HEIGHT * 0.44)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 80),
        1,
    )
    cv2.putText(
        frame,
        time.strftime("%Y-%m-%d"),
        (int(FRAME_WIDTH * 0.38), int(FRAME_HEIGHT * 0.62)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        frame,
        time.strftime("%H:%M:%S"),
        (int(FRAME_WIDTH * 0.40), int(FRAME_HEIGHT * 0.73)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "No camera connected",
        (int(FRAME_WIDTH * 0.26), int(FRAME_HEIGHT * 0.87)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )

    # Blinking live dot
    if int(time.time()) % 2 == 0:
        cv2.circle(frame, (40, 40), 12, (0, 0, 200), -1)
    cv2.putText(
        frame, "LIVE", (58, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
    )

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes()


# ─── Background capture thread ────────────────────────────────────────────────


def _capture_loop():
    global _running, STUB_MODE

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print(
            f"[camera] WARNING: could not open /dev/video{CAMERA_INDEX} — falling back to stub"
        )
        STUB_MODE = True
        cap.release()
        interval = 1.0 / TARGET_FPS
        while _running:
            _buffer.write(_generate_stub_frame())
            time.sleep(interval)
        return

    # Read actual resolution granted by the driver (may differ from requested)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[camera] opened /dev/video{CAMERA_INDEX} at {actual_w}x{actual_h} {TARGET_FPS}fps"
    )

    interval = 1.0 / TARGET_FPS
    prev = 0.0

    while _running:
        now = time.time()
        if now - prev < interval:
            time.sleep(0.001)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[camera] frame read failed — retrying")
            time.sleep(0.1)
            continue

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        _buffer.write(jpeg.tobytes())
        prev = now

    cap.release()
    print("[camera] capture thread stopped")


# ─── Lifecycle ────────────────────────────────────────────────────────────────


def start():
    """Start background capture thread. Called once on app startup."""
    global _capture_thread, _running
    if _running:
        return
    _running = True
    _capture_thread = threading.Thread(
        target=_capture_loop, daemon=True, name="camera-capture"
    )
    _capture_thread.start()
    print("[camera] capture thread started")


def stop():
    """Stop background capture thread. Called on app shutdown."""
    global _running
    _running = False
    if _capture_thread:
        _capture_thread.join(timeout=3.0)
    print("[camera] stopped")


# ─── MJPEG stream (dashboard live view) ──────────────────────────────────────


async def mjpeg_stream():
    """
    Async generator — yields MJPEG boundary chunks indefinitely.
    Plugs directly into FastAPI StreamingResponse.
    """
    loop = asyncio.get_event_loop()
    while True:
        frame = await loop.run_in_executor(_executor, _buffer.wait_for_frame, 2.0)
        if frame is None:
            await asyncio.sleep(0.1)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
            b"\r\n" + frame + b"\r\n"
        )


# ─── Snapshot capture (used by session scan loop) ─────────────────────────────


async def capture_snapshot(plant_id: int, session_id: str) -> str:
    """
    Capture a single fresh frame and save it to disk.

    Called AFTER the gantry has already stabilized (the caller waits
    camera_stabilize_delay before calling this). We discard the current
    buffer and wait for the very next frame so we never save a stale image
    from while the gantry was still moving.

    Returns: file path relative to project root (e.g. "static/images/abc1/plant_03.jpg")
    Raises:  RuntimeError if camera produces no frame within 5 seconds
    """
    loop = asyncio.get_event_loop()

    # Force a fresh frame — the event was already cleared by the last write,
    # so wait_for_frame will block until the capture thread writes the next one.
    frame = await loop.run_in_executor(_executor, _buffer.wait_for_frame, 5.0)

    if frame is None:
        raise RuntimeError("Camera not ready — no frame received within 5 seconds")

    dir_path = os.path.join(settings.images_dir, session_id)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"plant_{plant_id:02d}.jpg")
    with open(file_path, "wb") as f:
        f.write(frame)

    print(f"[camera] snapshot saved → {file_path}")
    return file_path
