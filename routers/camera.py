import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response

from services.camera import (
    mjpeg_stream,
    _buffer,
    _running,
    STUB_MODE,
    CAMERA_DEVICE,
    TARGET_FPS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

router = APIRouter(prefix="/camera", tags=["camera"])


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
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _buffer.wait_for_frame, 3.0)
    if frame is None:
        raise HTTPException(503, "Camera not ready — no frame available")
    return Response(
        content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-cache"}
    )


@router.get("/status")
async def camera_status():
    """Is the camera running and producing frames?"""
    return {
        "running": _running,
        "has_frame": _buffer.read() is not None,
        "stub_mode": STUB_MODE,
        "camera_device": CAMERA_DEVICE,
        "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "target_fps": TARGET_FPS,
    }
