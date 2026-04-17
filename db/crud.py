import json
from datetime import datetime, timezone
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.models import Session, PlantScan


def utcnow():
    return datetime.now(timezone.utc)


# ─── Sessions ─────────────────────────────────────────────────────────────────


async def create_session(
    db: AsyncSession, session_id: str, notes: str | None
) -> Session:
    row = Session(id=session_id, status="created", notes=notes, created_at=utcnow())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(selectinload(Session.plant_scans))
    )
    return result.scalar_one_or_none()


async def get_all_sessions(db: AsyncSession) -> list[Session]:
    result = await db.execute(select(Session).order_by(desc(Session.created_at)))
    return list(result.scalars().all())


async def set_session_running(db: AsyncSession, session_id: str) -> Session | None:
    row = await get_session(db, session_id)
    if not row:
        return None
    row.status = "running"
    row.started_at = utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def complete_session(
    db: AsyncSession, session_id: str, summary: dict
) -> Session | None:
    row = await get_session(db, session_id)
    if not row:
        return None

    ripeness = summary.get("ripeness_distribution", {})
    row.status = "complete"
    row.completed_at = utcnow()
    row.total_plants = summary.get("total_plants")
    row.avg_height_cm = summary.get("avg_height_cm")
    row.avg_moisture_pct = summary.get("avg_moisture_pct")
    row.total_water_sec = summary.get("total_water_sec")
    row.ripe_count = ripeness.get("ripe", 0)
    row.turning_count = ripeness.get("turning", 0)
    row.unripe_count = ripeness.get("unripe", 0)
    row.broken_count = ripeness.get("broken", 0)
    row.harvest_ready_ids = json.dumps(
        [p["plant_id"] for p in summary.get("harvest_ready", [])]
    )
    await db.commit()
    await db.refresh(row)
    return row


async def set_session_error(db: AsyncSession, session_id: str, message: str) -> None:
    row = await get_session(db, session_id)
    if row:
        row.status = "error"
        row.notes = f"{row.notes or ''} | error: {message}".strip(" |")
        await db.commit()


async def set_session_stopped(db: AsyncSession, session_id: str) -> None:
    row = await get_session(db, session_id)
    if row and row.status in ("running", "error"):  # ← was only "running"
        row.status = "stopped"
        row.completed_at = utcnow()
        await db.commit()


# ─── Plant scans ──────────────────────────────────────────────────────────────


async def create_plant_scan(
    db: AsyncSession, session_id: str, plant_id: int, row_idx: int, col_idx: int
) -> PlantScan:
    scan = PlantScan(session_id=session_id, plant_id=plant_id, row=row_idx, col=col_idx)
    db.add(scan)
    await db.commit()
    await db.refresh(scan)
    return scan


async def update_plant_scan_vision(
    db: AsyncSession,
    session_id: str,
    plant_id: int,
    image_url: str,
    detections: list[dict],
) -> None:
    result = await db.execute(
        select(PlantScan).where(
            PlantScan.session_id == session_id,
            PlantScan.plant_id == plant_id,
        )
    )
    scan = result.scalar_one_or_none()
    if not scan:
        return

    scan.image_url = image_url
    scan.detections_json = json.dumps(detections)
    scan.total_fruits = sum(d["count"] for d in detections)
    scan.ripe_count = sum(d["count"] for d in detections if d["cls"] == "ripe")
    scan.turning_count = sum(d["count"] for d in detections if d["cls"] == "turning")
    scan.unripe_count = sum(d["count"] for d in detections if d["cls"] == "unripe")
    scan.broken_count = sum(d["count"] for d in detections if d["cls"] == "broken")
    await db.commit()


async def update_plant_scan_sensors(
    db: AsyncSession,
    session_id: str,
    plant_id: int,
    height_cm: float,
    moisture_pct: float,
    valve_duration_sec: float,
    watering_reason: str,
) -> None:
    result = await db.execute(
        select(PlantScan).where(
            PlantScan.session_id == session_id,
            PlantScan.plant_id == plant_id,
        )
    )
    scan = result.scalar_one_or_none()
    if not scan:
        return

    scan.height_cm = height_cm
    scan.moisture_pct = moisture_pct
    scan.valve_duration_sec = valve_duration_sec
    scan.watering_reason = watering_reason
    await db.commit()


async def get_plant_scans(db: AsyncSession, session_id: str) -> list[PlantScan]:
    result = await db.execute(
        select(PlantScan)
        .where(PlantScan.session_id == session_id)
        .order_by(PlantScan.plant_id)
    )
    return list(result.scalars().all())


async def get_plant_scan(
    db: AsyncSession, session_id: str, plant_id: int
) -> PlantScan | None:
    result = await db.execute(
        select(PlantScan).where(
            PlantScan.session_id == session_id,
            PlantScan.plant_id == plant_id,
        )
    )
    return result.scalar_one_or_none()


async def reset_session(db: AsyncSession, session_id: str) -> Session | None:
    """Reset a stopped/error session back to 'created' so it can be started again."""
    row = await get_session(db, session_id)
    if not row:
        return None
    if row.status in ("stopped", "error"):
        row.status = "created"
        row.started_at = None
        row.completed_at = None
        await db.commit()
        await db.refresh(row)
    return row
