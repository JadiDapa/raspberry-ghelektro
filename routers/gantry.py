from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from routers.sessions import cancel_active_session
from routers.sessions import is_active as session_is_active
from services import gantry as gantry_service

router = APIRouter(prefix="/gantry", tags=["gantry"])


class MoveRequest(BaseModel):
    x: float = Field(..., ge=0, le=6000, description="X position in mm (0–6000)")
    y: float = Field(..., ge=0, le=2000, description="Y position in mm (0–2000)")
    z: float = Field(0.0, ge=0, le=800, description="Z position in mm (0–800)")
    speed: int = Field(150, ge=1, le=200, description="Travel speed in mm/s (capped at 200)")


class RelayRequest(BaseModel):
    channel: str = Field(..., pattern="^(sol|dc)$", description="'sol' = solenoid valve, 'dc' = water pump")
    on: bool = Field(..., description="True to open/enable, False to close/disable")


@router.get("/ping")
async def ping():
    """
    Diagnostic: check if ESP32 is reachable over USB serial.
    Useful for verifying the connection before starting a session.
    Returns {"alive": true} if the ESP32 responds to PING within 10 s.
    """
    alive = await gantry_service.ping()
    return {
        "alive": alive,
        "port": gantry_service._ser.port if gantry_service._ser else None,
    }


@router.get("/limits")
async def limits():
    """Read limit switch states directly from ESP32 (diagnostic)."""
    return await gantry_service.get_limits()


@router.post("/move")
async def move(body: MoveRequest):
    """Move gantry to an absolute X, Y, Z position in mm."""
    if session_is_active():
        raise HTTPException(409, "A session is running — manual control is disabled")
    if gantry_service.get_state()["busy"]:
        raise HTTPException(409, "Gantry is currently busy")
    result = await gantry_service.move_to(body.x, body.y, body.z, body.speed)
    return {"ok": True, "position": result}


@router.post("/home")
async def home():
    """Home all axes — moves gantry to X=0, Y=0, Z=0."""
    if session_is_active():
        raise HTTPException(409, "A session is running — manual control is disabled")
    if gantry_service.get_state()["busy"]:
        raise HTTPException(409, "Gantry is currently busy")
    result = await gantry_service.home()
    return {"ok": True, "position": result}


@router.post("/stop")
async def stop():
    """
    Emergency stop — kill everything.

    Cancels any running session, halts all motion, de-energizes the stepper
    drivers, and closes every relay (solenoid valve + water pump). Each step is
    best-effort so the stop does as much as it can even if part of the machine
    is unresponsive.
    """
    cancelled = await cancel_active_session()
    result = await gantry_service.emergency_stop()
    return {
        "ok": True,
        "stopped": True,
        "cancelled_session": cancelled,
        "position": result,
    }


@router.get("/position")
async def position():
    """Current gantry position, busy state, and whether a session is active."""
    state = gantry_service.get_state()
    return {**state, "session_active": session_is_active()}


@router.post("/relay")
async def relay(body: RelayRequest):
    """
    Toggle solenoid valve ('sol') or water pump ('dc').
    Blocked while a session is running.
    """
    if session_is_active():
        raise HTTPException(409, "A session is running — manual control is disabled")
    await gantry_service.set_relay(body.channel, body.on)
    return {"ok": True, "channel": body.channel, "on": body.on}
