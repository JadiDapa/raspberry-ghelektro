"""
pi_client.py — HTTP client for posting real-time scan data to the Next.js dashboard.
One method per endpoint. Exceptions propagate to the caller.
"""

import httpx

from config import settings


def _base() -> str:
    return settings.dashboard_url.rstrip("/")


async def patch_status(session_id: int, status: str) -> None:
    """PATCH /api/sessions/{id}/status — mark session running/stopped/error."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.patch(
            f"{_base()}/api/sessions/{session_id}/status",
            json={"status": status},
        )
        r.raise_for_status()


async def upload_image(session_id: int, plant_index: int, image_bytes: bytes) -> str:
    """POST image bytes as multipart → returns Next.js imageUrl."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/captures/{plant_index}/image",
            files={"file": ("plant.jpg", image_bytes, "image/jpeg")},
        )
        r.raise_for_status()
        return r.json()["imageUrl"]


async def post_vision(
    session_id: int,
    plant_index: int,
    row: int,
    col: int,
    image_url: str,
    detections: list[dict],
) -> None:
    """POST /captures/{plantIndex}/vision — create Capture row with YOLO results."""
    total_fruits = sum(d["count"] for d in detections)
    counts = {cls: 0 for cls in ("ripe", "turning", "unripe", "broken")}
    for d in detections:
        if d["cls"] in counts:
            counts[d["cls"]] += d["count"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/captures/{plant_index}/vision",
            json={
                "row": row,
                "col": col,
                "imageUrl": image_url,
                "totalFruits": total_fruits,
                "ripeCount": counts["ripe"],
                "turningCount": counts["turning"],
                "unripeCount": counts["unripe"],
                "brokenCount": counts["broken"],
            },
        )
        r.raise_for_status()


async def post_sensors(
    session_id: int,
    plant_index: int,
    height_cm: float | None,
    moisture_pct: float | None,
    valve_duration_sec: float,
    watering_reason: str,
) -> None:
    """POST /captures/{plantIndex}/sensors — update Capture with sensor readings."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/captures/{plant_index}/sensors",
            json={
                "heightCm": height_cm,
                "moisturePct": moisture_pct,
                "valveDurationSec": valve_duration_sec,
                "wateringReason": watering_reason,
            },
        )
        r.raise_for_status()


async def post_complete(session_id: int, summary: dict) -> None:
    """POST /complete — finalize session with aggregate summary."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/complete",
            json={"summary": summary},
        )
        r.raise_for_status()


async def post_error(session_id: int) -> None:
    """POST /error — mark session as error in Next.js."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/error",
            json={},
        )
        r.raise_for_status()


async def post_watering_stop(
    session_id: int,
    stop_index: int,
    x_mm: float,
    y_mm: float,
    max_height_cm: float | None,
    valve_duration_sec: float,
) -> None:
    """POST /sessions/{id}/watering-stops — record one column watering stop."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/watering-stops",
            json={
                "stopIndex": stop_index,
                "xMm": x_mm,
                "yMm": y_mm,
                "maxHeightCm": max_height_cm,
                "valveDurationSec": valve_duration_sec,
            },
        )
        r.raise_for_status()


async def post_watering_complete(session_id: int, summary: dict) -> None:
    """POST /sessions/{id}/complete — finalize watering session with summary."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/complete",
            json={"summary": summary},
        )
        r.raise_for_status()
