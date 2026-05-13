from fastapi import APIRouter
from pydantic import BaseModel, Field

from services import gantry as gantry_service

router = APIRouter(prefix="/servo", tags=["servo"])


class ServoControlRequest(BaseModel):
    pan: float = Field(..., ge=0, le=180, description="Pan angle in degrees (0–180)")
    tilt: float = Field(..., ge=0, le=180, description="Tilt angle in degrees (0–180)")


@router.get("/angles")
async def get_angles():
    """Current servo pan/tilt angles."""
    return await gantry_service.get_servo_angles()


@router.post("/control")
async def control(body: ServoControlRequest):
    """Set servo pan/tilt angles and return the applied angles."""
    return await gantry_service.set_servo_angles(body.pan, body.tilt)
