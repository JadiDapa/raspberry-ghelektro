import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from services.camera import (
    mjpeg_stream,
    _buffer,
    _running,
    STUB_MODE,
    CAMERA_DEVICE,
    get_controls,
    set_controls,
)

router = APIRouter(prefix="/camera", tags=["camera"])


# ─── Adjustable camera settings ───────────────────────────────────────────────
# Every field is optional so the dashboard can PATCH just what changed. `None`
# for a manual value means "leave the driver at its default". Bools pick auto vs
# manual mode for exposure / white balance / focus.
class CameraSettings(BaseModel):
    frame_width: int | None = None
    frame_height: int | None = None
    fps: int | None = None
    auto_exposure: bool | None = None
    exposure: int | None = None
    gain: int | None = None
    auto_wb: bool | None = None
    wb_temperature: int | None = None
    autofocus: bool | None = None
    focus: int | None = None
    brightness: int | None = None
    contrast: int | None = None
    saturation: int | None = None
    sharpness: int | None = None


@router.get("/stream")
async def live_stream():
    """MJPEG live stream — embed directly in an <img> tag."""
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/snapshot")
async def snapshot():
    """Returns the current frame as a single JPEG."""
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+
    frame = await loop.run_in_executor(None, _buffer.wait_for_frame, 3.0)
    if frame is None:
        raise HTTPException(503, "Camera not ready — no frame available")
    return Response(
        content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-cache"}
    )


@router.get("/settings")
async def read_settings():
    """Current desired camera controls + the values the driver actually granted."""
    return get_controls()


@router.post("/settings")
async def update_settings(body: CameraSettings):
    """Apply camera controls (exposure, gain, white balance, focus, resolution …).

    Only the fields present in the body are changed; the capture thread re-opens
    the device to apply them. Returns the merged controls and the driver-granted
    actuals so the caller can see what really took effect.
    """
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "No settings provided")
    set_controls(updates)
    # Give the capture thread a moment to re-open and read actuals back.
    await asyncio.sleep(0.6)
    return get_controls()


@router.get("/status")
async def camera_status():
    """Is the camera running and producing frames?"""
    state = get_controls()
    controls = state["controls"]
    return {
        "running": _running,
        "has_frame": _buffer.read() is not None,
        "stub_mode": STUB_MODE,
        "camera_device": CAMERA_DEVICE,
        "resolution": f"{controls['frame_width']}x{controls['frame_height']}",
        "target_fps": controls["fps"],
    }
