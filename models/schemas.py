from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ─── Session ──────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    notes: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_plants: Optional[int] = None
    avg_height_cm: Optional[float] = None
    avg_moisture_pct: Optional[float] = None
    total_water_sec: Optional[float] = None
    ripeness: Optional[dict] = None
    harvest_ready_ids: Optional[list[int]] = None


class PlantScanResponse(BaseModel):
    plant_id: int
    row: int
    col: int
    scanned_at: Optional[datetime] = None
    image_url: Optional[str] = None
    total_fruits: Optional[int] = None
    ripe_count: Optional[int] = None
    turning_count: Optional[int] = None
    unripe_count: Optional[int] = None
    broken_count: Optional[int] = None
    detections: list[dict] = []
    height_cm: Optional[float] = None
    moisture_pct: Optional[float] = None
    valve_duration_sec: Optional[float] = None
    watering_reason: Optional[str] = None


# ─── SSE Events ───────────────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    cls: str   # "ripe" | "turning" | "unripe" | "broken"
    count: int
    confidence: float


class HarvestCandidate(BaseModel):
    plant_id: int
    row: int
    col: int
    ripe_count: int


class SessionSummary(BaseModel):
    total_plants: int
    avg_height_cm: float
    avg_moisture_pct: float
    ripeness_distribution: dict   # {"ripe": N, "turning": N, "unripe": N, "broken": N}
    harvest_ready: list[HarvestCandidate]
    total_water_sec: float


# ─── Serialization helpers (shared by all routers) ────────────────────────────

def serialize_session(row) -> dict:
    return {
        "session_id": row.id,
        "status": row.status,
        "notes": row.notes,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "total_plants": row.total_plants,
        "avg_height_cm": row.avg_height_cm,
        "avg_moisture_pct": row.avg_moisture_pct,
        "total_water_sec": row.total_water_sec,
        "ripeness": {
            "ripe":    row.ripe_count,
            "turning": row.turning_count,
            "unripe":  row.unripe_count,
            "broken":  row.broken_count,
        },
        "harvest_ready_ids": row.harvest_ready_list(),
    }
# Note: ambient sensor data (temp, humidity, light, etc.) comes from a
# standalone ESP32 that talks directly to the dashboard, not through the Pi.


def serialize_plant_scan(scan) -> dict:
    return {
        "plant_id": scan.plant_id,
        "row": scan.row,
        "col": scan.col,
        "scanned_at": scan.scanned_at.isoformat() if scan.scanned_at else None,
        "image_url": scan.image_url,
        "total_fruits": scan.total_fruits,
        "ripe_count": scan.ripe_count,
        "turning_count": scan.turning_count,
        "unripe_count": scan.unripe_count,
        "broken_count": scan.broken_count,
        "detections": scan.detections(),
        "height_cm": scan.height_cm,
        "moisture_pct": scan.moisture_pct,
        "valve_duration_sec": scan.valve_duration_sec,
        "watering_reason": scan.watering_reason,
    }
