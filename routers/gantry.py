from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services import gantry as gantry_service

router = APIRouter(prefix="/gantry", tags=["gantry"])


class MoveRequest(BaseModel):
    x: float = Field(..., ge=0, le=6000, description="X position in mm (0–6000)")
    y: float = Field(..., ge=0, le=2000, description="Y position in mm (0–2000)")
    z: float = Field(0.0, ge=0, le=200, description="Z position in mm (0–200)")
    speed: int = Field(500, ge=1, le=5000, description="Travel speed in mm/min")


@router.post("/move")
async def move(body: MoveRequest):
    """Move gantry to an absolute X, Y, Z position in mm."""
    if gantry_service.get_state()["busy"]:
        raise HTTPException(409, "Gantry is currently busy")
    result = await gantry_service.move_to(body.x, body.y, body.z, body.speed)
    return {"ok": True, "position": result}


@router.post("/home")
async def home():
    """Home all axes — moves gantry to X=0, Y=0, Z=0."""
    if gantry_service.get_state()["busy"]:
        raise HTTPException(409, "Gantry is currently busy")
    result = await gantry_service.home()
    return {"ok": True, "position": result}


@router.post("/stop")
async def stop():
    """Emergency stop — immediately halts all motors."""
    result = await gantry_service.emergency_stop()
    return {"ok": True, "stopped": True, "position": result}


@router.get("/position")
async def position():
    """Current gantry position and status."""
    return gantry_service.get_state()
