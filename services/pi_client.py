"""
pi_client.py — HTTP client for posting real-time scan data to the Next.js dashboard.
One method per endpoint. Exceptions propagate to the caller.

Stub mode: active when STUB_MODE=true OR dashboard_url is empty.
All methods log their call and return immediately — no HTTP request is made.
This lets you run the full session pipeline on a Pi (or dev machine) without
a running Next.js instance.
"""

import httpx

from config import settings


def _base() -> str:
    return settings.dashboard_url.rstrip("/")


def _is_stub() -> bool:
    """True when there is no dashboard to talk to."""
    return settings.stub_mode or not settings.dashboard_url.strip()


async def patch_status(session_id: int, status: str) -> None:
    """PATCH /api/sessions/{id}/status — mark session running/stopped/error."""
    if _is_stub():
        print(f"[pi_client:stub] patch_status({session_id}, {status!r})")
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.patch(
            f"{_base()}/api/sessions/{session_id}/status",
            json={"status": status},
        )
        r.raise_for_status()


async def upload_image(session_id: int, plant_index: int, image_bytes: bytes) -> str:
    """POST image bytes as multipart → returns Next.js imageUrl."""
    if _is_stub():
        url = f"stub://session-{session_id}/plant-{plant_index}.jpg"
        print(f"[pi_client:stub] upload_image({session_id}, {plant_index}) → {url}")
        return url
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
    if _is_stub():
        print(f"[pi_client:stub] post_vision({session_id}, plant={plant_index}, detections={detections})")
        return
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
    if _is_stub():
        print(f"[pi_client:stub] post_sensors({session_id}, plant={plant_index}, height={height_cm}, moisture={moisture_pct})")
        return
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
    if _is_stub():
        print(f"[pi_client:stub] post_complete({session_id}, summary={summary})")
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/complete",
            json={"summary": summary},
        )
        r.raise_for_status()


async def post_error(session_id: int) -> None:
    """POST /error — mark session as error in Next.js."""
    if _is_stub():
        print(f"[pi_client:stub] post_error({session_id})")
        return
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
    if _is_stub():
        print(f"[pi_client:stub] post_watering_stop({session_id}, stop={stop_index}, duration={valve_duration_sec}s)")
        return
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
    if _is_stub():
        print(f"[pi_client:stub] post_watering_complete({session_id}, summary={summary})")
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_base()}/api/sessions/{session_id}/complete",
            json={"summary": summary},
        )
        r.raise_for_status()
