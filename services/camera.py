import asyncio
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from config import settings

# ─── Config (from settings) ───────────────────────────────────────────────────

CAMERA_DEVICE = settings.camera_device
JPEG_QUALITY = settings.camera_jpeg_quality

# Native resolution pushed to the device — always a real camera capture mode.
# The dashboard's requested frame_width/frame_height are the OUTPUT size, produced
# from this native frame by `_process_frame` (center-crop to aspect + resize). The
# C922 sensor is 16:9, so a 4:3 output is cropped here rather than requested of the
# driver, which would just snap 1280×960 back to 1280×720. See config.py.
CAPTURE_WIDTH = settings.camera_capture_width
CAPTURE_HEIGHT = settings.camera_capture_height

# Set automatically on startup — True if camera couldn't be opened
STUB_MODE = False

# ─── Adjustable camera controls ───────────────────────────────────────────────
# Single source of truth for every cv2.CAP_PROP_* we let the dashboard change.
# The capture thread is the only place these are written to the device. A change
# from a request handler updates `_controls` and trips `_reopen_event`; the loop
# then releases and re-opens the device through `_open_and_configure()`, which is
# the same path used on first start and on a read-failure reconnect — so there is
# exactly one place that knows how to push controls to the camera.
#
# `None` means "leave the driver at its own default / don't touch this control".
# The defaults below reproduce the camera's pre-existing behaviour (full auto,
# resolution/fps from config) so a fresh Pi with no dashboard settings behaves
# exactly as before. The dashboard is where a user opts into manual values.
_controls: dict = {
    # OUTPUT dimensions the dashboard receives. Any aspect ratio is allowed; the
    # camera is captured at CAPTURE_WIDTH×CAPTURE_HEIGHT and each frame is
    # cropped+resized to this in `_process_frame`. Selecting 4:3 here really does
    # yield a 4:3 image even though the C922 sensor is natively 16:9.
    "frame_width": settings.camera_width,
    "frame_height": settings.camera_height,
    "fps": settings.camera_fps,
    "auto_exposure": True,  # False → manual, then `exposure` is applied
    "exposure": None,
    "gain": None,
    "auto_wb": True,  # False → manual, then `wb_temperature` is applied
    "wb_temperature": None,
    "autofocus": True,  # False → manual, then `focus` is applied
    "focus": None,
    "brightness": None,
    "contrast": None,
    "saturation": None,
    "sharpness": None,
    "zoom": None,
}

# The keys a caller may set, with their value type for light validation.
_BOOL_KEYS = {"auto_exposure", "auto_wb", "autofocus"}
_INT_KEYS = {
    "frame_width",
    "frame_height",
    "fps",
    "exposure",
    "gain",
    "wb_temperature",
    "focus",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "zoom",
}

_controls_lock = threading.Lock()
_reopen_event = threading.Event()

# Values the driver actually granted after the last open, read straight back from
# the device. Lets the dashboard show what really took effect (drivers clamp or
# ignore unsupported controls). Populated by `_read_actuals()`.
_actuals: dict = {}

# The camera's control values read once at the very first open, BEFORE we apply
# anything — used as a fallback factory baseline for "reset to defaults" when the
# more accurate `v4l2-ctl` query isn't available. See `_open_and_configure`.
_factory_defaults: dict | None = None


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


# ─── Control plumbing ─────────────────────────────────────────────────────────
# UVC webcams (the V4L2 backend OpenCV uses here) follow a few non-obvious
# conventions:
#   • CAP_PROP_AUTO_EXPOSURE: 3 = auto (aperture priority), 1 = manual.  You must
#     switch to manual (1) BEFORE a CAP_PROP_EXPOSURE write will stick.
#   • CAP_PROP_AUTO_WB / CAP_PROP_AUTOFOCUS: 1 = auto, 0 = manual.  Same ordering.
# Unsupported controls just make cap.set() return False; we don't treat that as
# an error since support varies per camera.

_AUTO_EXPOSURE_AUTO = 3
_AUTO_EXPOSURE_MANUAL = 1


def _apply_controls(cap: "cv2.VideoCapture", c: dict) -> None:
    """Push the control dict onto an open VideoCapture. Skips None values."""

    def _set(prop: int, value) -> None:
        if value is not None:
            cap.set(prop, float(value))

    # Native capture resolution / fps first — these may force the driver to
    # renegotiate format. We always request the fixed native mode (a resolution the
    # camera really supports); the dashboard's frame_width/frame_height are the
    # output size and are synthesized later by `_process_frame`, not asked of the
    # driver. Requesting an unsupported 4:3 mode just gets snapped back to 16:9.
    _set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    _set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    _set(cv2.CAP_PROP_FPS, c.get("fps"))

    # Exposure — turn auto off before writing a manual value.
    if c.get("auto_exposure"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_AUTO)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_MANUAL)
        _set(cv2.CAP_PROP_EXPOSURE, c.get("exposure"))

    # White balance.
    if c.get("auto_wb"):
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        _set(cv2.CAP_PROP_WB_TEMPERATURE, c.get("wb_temperature"))

    # Focus.
    if c.get("autofocus"):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        _set(cv2.CAP_PROP_FOCUS, c.get("focus"))

    # Plain image controls — safe to set in any order.
    _set(cv2.CAP_PROP_GAIN, c.get("gain"))
    _set(cv2.CAP_PROP_BRIGHTNESS, c.get("brightness"))
    _set(cv2.CAP_PROP_CONTRAST, c.get("contrast"))
    _set(cv2.CAP_PROP_SATURATION, c.get("saturation"))
    _set(cv2.CAP_PROP_SHARPNESS, c.get("sharpness"))
    _set(cv2.CAP_PROP_ZOOM, c.get("zoom"))


def _read_control_values(cap: "cv2.VideoCapture") -> dict:
    """Read every control we manage straight off the device, in our schema."""
    ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    return {
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "auto_exposure": ae == _AUTO_EXPOSURE_AUTO,
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "auto_wb": bool(cap.get(cv2.CAP_PROP_AUTO_WB)),
        "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "autofocus": bool(cap.get(cv2.CAP_PROP_AUTOFOCUS)),
        "focus": cap.get(cv2.CAP_PROP_FOCUS),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        "sharpness": cap.get(cv2.CAP_PROP_SHARPNESS),
        "zoom": cap.get(cv2.CAP_PROP_ZOOM),
    }


def _read_actuals(cap: "cv2.VideoCapture") -> None:
    """Read back what the driver granted so the dashboard can show real values.

    The device is always driven at the native capture resolution, so the raw
    frame_width/frame_height read off it describe the capture, not the image the
    dashboard gets. We expose those as capture_width/capture_height and overlay
    frame_width/frame_height with the OUTPUT size (what `_process_frame` produces),
    so `actuals.frame_width` matches what the user selected — no phantom mismatch.
    """
    global _actuals
    values = _read_control_values(cap)
    values["capture_width"] = values["frame_width"]
    values["capture_height"] = values["frame_height"]
    with _controls_lock:
        values["frame_width"] = _controls["frame_width"]
        values["frame_height"] = _controls["frame_height"]
        _actuals = values


def _open_and_configure() -> "cv2.VideoCapture":
    """Open the capture device and push the current controls. Single source of
    truth for device setup — first start, reconnect and a settings change all
    funnel through here."""
    global _factory_defaults
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
    # First successful open this session: capture the driver's control values
    # BEFORE we change anything, as a fallback factory baseline for reset.
    if cap.isOpened() and _factory_defaults is None:
        _factory_defaults = _read_control_values(cap)
        print(f"[camera] captured factory baseline → {_factory_defaults}")
    with _controls_lock:
        snapshot = dict(_controls)
    _apply_controls(cap, snapshot)
    if cap.isOpened():
        _read_actuals(cap)
    return cap


def set_controls(updates: dict) -> dict:
    """Merge `updates` into the live controls and ask the capture thread to
    re-open the device so they take effect. Returns the merged control dict.

    Only known keys are accepted; unknown keys are ignored. Bool keys are
    coerced to bool, int keys to int (or None to mean 'driver default').
    """
    with _controls_lock:
        for key, value in updates.items():
            if key in _BOOL_KEYS:
                _controls[key] = bool(value)
            elif key in _INT_KEYS:
                _controls[key] = None if value is None else int(value)
        merged = dict(_controls)
    _reopen_event.set()
    print(f"[camera] controls updated → {merged}")
    return merged


def get_controls() -> dict:
    """Current desired controls plus the values the driver actually granted."""
    with _controls_lock:
        return {"controls": dict(_controls), "actuals": dict(_actuals)}


# ─── Factory reset ────────────────────────────────────────────────────────────
# V4L2 control name → our numeric control key. Only the numeric image controls
# are mapped: the auto toggles are forced to auto on reset (the universal factory
# state), which side-steps the menu-value differences between kernel versions
# (e.g. exposure_auto vs auto_exposure). Names vary across kernels, so several
# aliases map to the same key.
_V4L2_NUMERIC_MAP = {
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "sharpness": "sharpness",
    "gain": "gain",
    "white_balance_temperature": "wb_temperature",
    "exposure_absolute": "exposure",
    "exposure_time_absolute": "exposure",
    "focus_absolute": "focus",
    "zoom_absolute": "zoom",
}


def _query_v4l2_defaults(device: str) -> dict:
    """Parse `v4l2-ctl --list-ctrls` for each control's factory `default=`.

    Returns a dict in our control schema (numeric keys only), or {} when v4l2-ctl
    is missing / outputs nothing. This is the accurate, boot-independent source of
    the camera's true defaults; the caller falls back to the first-open baseline.
    """
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-ctrls"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
        print(f"[camera] v4l2-ctl unavailable ({e}) — using first-open baseline")
        return {}

    defaults: dict = {}
    for line in proc.stdout.splitlines():
        m = re.search(r"^\s*([a-z0-9_]+)\s+0x[0-9a-f]+.*\bdefault=(-?\d+)", line)
        if not m:
            continue
        key = _V4L2_NUMERIC_MAP.get(m.group(1))
        if key:
            defaults[key] = int(m.group(2))
    return defaults


def reset_to_defaults() -> dict:
    """Return the camera to its out-of-box look: all auto modes on, and every
    image control back to the driver's factory default (resolution/fps included).
    Re-opens the device so OpenCV's cached values don't immediately re-apply.

    Numeric defaults come from `v4l2-ctl` when available, else the values captured
    at first open this session. Auto toggles are always set on.
    """
    numeric = _query_v4l2_defaults(CAMERA_DEVICE)
    fd = _factory_defaults or {}
    if not numeric:
        numeric = {
            k: fd.get(k)
            for k in (
                "brightness",
                "contrast",
                "saturation",
                "sharpness",
                "gain",
                "wb_temperature",
                "exposure",
                "focus",
                "zoom",
            )
        }

    with _controls_lock:
        _controls.update(
            {
                # Resolution/fps back to the factory baseline if we have it, else
                # the config defaults already in _controls are left untouched.
                "frame_width": int(fd["frame_width"])
                if fd.get("frame_width")
                else _controls["frame_width"],
                "frame_height": int(fd["frame_height"])
                if fd.get("frame_height")
                else _controls["frame_height"],
                "fps": int(fd["fps"]) if fd.get("fps") else _controls["fps"],
                "auto_exposure": True,
                "auto_wb": True,
                "autofocus": True,
                "exposure": numeric.get("exposure"),
                "gain": numeric.get("gain"),
                "wb_temperature": numeric.get("wb_temperature"),
                "focus": numeric.get("focus"),
                "brightness": numeric.get("brightness"),
                "contrast": numeric.get("contrast"),
                "saturation": numeric.get("saturation"),
                "sharpness": numeric.get("sharpness"),
                "zoom": numeric.get("zoom"),
            }
        )
        merged = dict(_controls)
    _reopen_event.set()
    print(f"[camera] reset to factory defaults → {merged}")
    return merged


# ─── Stub frame generator ─────────────────────────────────────────────────────


def _generate_stub_frame() -> bytes:
    """Dark grey canvas with timestamp — used when no camera is connected."""
    with _controls_lock:
        w = _controls["frame_width"]
        h = _controls["frame_height"]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)

    cv2.putText(
        frame,
        "FarmBot Camera",
        (int(w * 0.28), int(h * 0.33)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 200, 100),
        2,
    )
    cv2.putText(
        frame,
        "[ STUB MODE ]",
        (int(w * 0.33), int(h * 0.44)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 80),
        1,
    )
    cv2.putText(
        frame,
        time.strftime("%Y-%m-%d"),
        (int(w * 0.38), int(h * 0.62)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        frame,
        time.strftime("%H:%M:%S"),
        (int(w * 0.40), int(h * 0.73)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "No camera connected",
        (int(w * 0.26), int(h * 0.87)),
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


# ─── Output framing ───────────────────────────────────────────────────────────
# The camera captures at a fixed native 16:9 mode; the dashboard picks the output
# size/aspect. We reconcile the two here: center-crop the native frame to the
# requested aspect ratio (trimming the longer axis, keeping the middle), then
# resize to the exact requested dimensions. A 4:3 request on the 16:9 C922 trims
# the left/right edges and keeps the full vertical field of view — sharp, not
# stretched or letterboxed. A 16:9 request is a crop-free downscale. This runs on
# every frame before JPEG encoding, so the MJPEG stream, session snapshots, YOLO
# input and recorded video all inherit the same framing from the shared buffer.


def _process_frame(frame: np.ndarray) -> np.ndarray:
    with _controls_lock:
        out_w = _controls["frame_width"]
        out_h = _controls["frame_height"]
    if not out_w or not out_h:
        return frame

    h, w = frame.shape[:2]
    # Center-crop to the target aspect ratio (compare w/h vs out_w/out_h via
    # cross-multiplication to avoid float error).
    if w * out_h > h * out_w:  # frame wider than target → trim width
        crop_w = h * out_w // out_h
        x0 = (w - crop_w) // 2
        frame = frame[:, x0 : x0 + crop_w]
    elif w * out_h < h * out_w:  # frame taller than target → trim height
        crop_h = w * out_h // out_w
        y0 = (h - crop_h) // 2
        frame = frame[y0 : y0 + crop_h, :]

    # Resize to the exact requested output size (INTER_AREA is best for downscale).
    if frame.shape[1] != out_w or frame.shape[0] != out_h:
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return frame


# ─── Background capture thread ────────────────────────────────────────────────


def _current_fps() -> float:
    with _controls_lock:
        return float(_controls.get("fps") or 10)


def _capture_loop():
    global _running, STUB_MODE

    cap = _open_and_configure()

    print(f"[camera] opening {CAMERA_DEVICE} ...")
    print("[camera] opened:", cap.isOpened())

    if cap.isOpened():
        print("[camera] backend:", cap.getBackendName())

    if not cap.isOpened():
        print(
            f"[camera] WARNING: could not open {CAMERA_DEVICE} — falling back to stub"
        )
        STUB_MODE = True
        cap.release()
        while _running:
            _buffer.write(_generate_stub_frame())
            time.sleep(1.0 / _current_fps())
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] opened {CAMERA_DEVICE} at {actual_w}x{actual_h}")

    prev = 0.0

    while _running:
        # A settings change requests a clean re-open so resolution/fps and the
        # auto/manual control modes are renegotiated from scratch.
        if _reopen_event.is_set():
            _reopen_event.clear()
            print("[camera] applying new settings — reopening device")
            cap.release()
            cap = _open_and_configure()
            if not cap.isOpened():
                print("[camera] reopen failed — retrying shortly")
                time.sleep(1)
                continue

        interval = 1.0 / _current_fps()
        now = time.time()
        if now - prev < interval:
            time.sleep(0.001)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[camera] frame read failed — reconnecting...")
            cap.release()
            time.sleep(1)
            cap = _open_and_configure()
            continue

        frame = _process_frame(frame)  # native 16:9 → requested output size/aspect
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
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+
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


async def capture_bytes() -> bytes:
    """
    Capture a single fresh frame and return raw JPEG bytes.
    Never writes to disk. Raises RuntimeError if no frame within 5 seconds.
    """
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+
    frame = await loop.run_in_executor(_executor, _buffer.wait_for_frame, 5.0)
    if frame is None:
        raise RuntimeError("Camera not ready — no frame received within 5 seconds")
    print("[camera] snapshot captured → in-memory bytes")
    return frame


def latest_jpeg() -> bytes | None:
    """Return the most recent JPEG frame from the shared buffer (or None).

    Synchronous, non-blocking read of whatever the background capture thread
    last wrote. Used by the video recorder to sample frames while the gantry
    sweeps; it tolerates None (no frame yet) and just skips that tick.
    """
    return _buffer.read()
