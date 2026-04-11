from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db import crud
from models.schemas import serialize_plant_scan

router = APIRouter(prefix="/plants", tags=["plants"])


@router.get("/{session_id}")
async def get_plants(session_id: str, db: AsyncSession = Depends(get_db)):
    """All plant scan results for a session, ordered by plant_id."""
    row = await crud.get_session(db, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    scans = await crud.get_plant_scans(db, session_id)
    return [serialize_plant_scan(s) for s in scans]


@router.get("/{session_id}/{plant_id}")
async def get_plant(session_id: str, plant_id: int, db: AsyncSession = Depends(get_db)):
    """Single plant scan detail — image, detections, height, moisture, watering."""
    scan = await crud.get_plant_scan(db, session_id, plant_id)
    if not scan:
        raise HTTPException(404, f"Plant {plant_id} not found in session {session_id}")
    return serialize_plant_scan(scan)
