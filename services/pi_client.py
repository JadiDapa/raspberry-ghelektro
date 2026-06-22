"""
pi_client.py — HTTP client for posting real-time scan data to the Next.js dashboard.
One method per endpoint.

Resilience: every request goes through `_send()`, which retries transient failures
(transport errors and 5xx responses) with exponential backoff. Client errors (4xx)
are NOT retried — they indicate a bad request, not a flaky network. After retries are
exhausted the original exception propagates so the caller can decide what to do
(the session loop treats real-time posts as non-fatal and falls back to an
end-of-session /sync, see services/session_service.py).

Stub mode: active when STUB_MODE=true OR dashboard_url is empty.
All methods log their call and return immediately — no HTTP request is made.
This lets you run the full session pipeline on a Pi (or dev machine) without
a running Next.js instance.
"""

import asyncio

import httpx

from config import settings


def _base() -> str:
    return settings.dashboard_url.rstrip("/")


def _is_stub() -> bool:
    """True when there is no dashboard to talk to."""
    return settings.stub_mode or not settings.dashboard_url.strip()


async def _send(
    method: str,
    path: str,
    *,
    json: dict | None = None,
    files: dict | None = None,
    timeout: float = 10.0,
) -> httpx.Response:
    """
    Send one request to the dashboard with bounded retry + exponential backoff.

    Retries on transport errors (DNS/connect/read) and 5xx responses. Does NOT
    retry 4xx — those are caller bugs and won't get better by retrying. Raises the
    last exception once `sync_max_retries` attempts are used up.
    """
    url = f"{_base()}{path}"
    attempts = max(1, settings.sync_max_retries)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.request(method, url, json=json, files=files)
            if 400 <= r.status_code < 500:
                r.raise_for_status()  # client error — fail fast, no retry
            if r.status_code >= 500:
                # Treat as retryable server error.
                last_exc = httpx.HTTPStatusError(
                    f"{method} {path} → {r.status_code}",
                    request=r.request,
                    response=r,
                )
                raise last_exc
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500:
                raise  # don't retry client errors
            last_exc = e
        except httpx.HTTPError as e:  # transport / timeout
            last_exc = e

        if attempt < attempts:
            backoff = settings.sync_backoff_base * (2 ** (attempt - 1))
            print(f"[pi_client] {method} {path} failed (attempt {attempt}/{attempts}) — retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)

    assert last_exc is not None
    raise last_exc


async def patch_status(session_id: int, status: str) -> None:
    """PATCH /api/sessions/{id}/status — mark session running/stopped/error."""
    if _is_stub():
        print(f"[pi_client:stub] patch_status({session_id}, {status!r})")
        return
    await _send(
        "PATCH",
        f"/api/sessions/{session_id}/status",
        json={"status": status},
    )


async def upload_image(session_id: int, plant_index: int, image_bytes: bytes) -> str:
    """POST image bytes as multipart → returns Next.js imageUrl."""
    if _is_stub():
        url = f"stub://session-{session_id}/plant-{plant_index}.jpg"
        print(f"[pi_client:stub] upload_image({session_id}, {plant_index}) → {url}")
        return url
    r = await _send(
        "POST",
        f"/api/sessions/{session_id}/captures/{plant_index}/image",
        files={"file": ("plant.jpg", image_bytes, "image/jpeg")},
        timeout=30.0,
    )
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

    await _send(
        "POST",
        f"/api/sessions/{session_id}/captures/{plant_index}/vision",
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
    await _send(
        "POST",
        f"/api/sessions/{session_id}/captures/{plant_index}/sensors",
        json={
            "heightCm": height_cm,
            "moisturePct": moisture_pct,
            "valveDurationSec": valve_duration_sec,
            "wateringReason": watering_reason,
        },
    )


async def post_complete(session_id: int, summary: dict) -> None:
    """POST /complete — finalize session with aggregate summary."""
    if _is_stub():
        print(f"[pi_client:stub] post_complete({session_id}, summary={summary})")
        return
    await _send(
        "POST",
        f"/api/sessions/{session_id}/complete",
        json={"summary": summary},
    )


async def post_error(session_id: int) -> None:
    """POST /error — mark session as error in Next.js."""
    if _is_stub():
        print(f"[pi_client:stub] post_error({session_id})")
        return
    await _send("POST", f"/api/sessions/{session_id}/error", json={})


async def sync_session(payload: dict) -> int | None:
    """
    POST /api/sessions/sync — push a whole session in one shot.

    Used as the resilient fallback when real-time posts failed during a run, and
    replayed from the outbox at startup after an outage. Idempotent on the
    dashboard side (it reconciles by the session's integer id). Returns the
    persisted session id, or None in stub mode.
    """
    if _is_stub():
        print(f"[pi_client:stub] sync_session({payload.get('session_id')})")
        return None
    r = await _send("POST", "/api/sessions/sync", json=payload, timeout=30.0)
    try:
        return r.json().get("session_id")
    except Exception:
        return None


async def fetch_due_sessions(bed_id: int) -> list[dict]:
    """
    POST /api/schedules/tick — ask the dashboard which scheduled sessions are due.

    The dashboard mints the PENDING sessions (session ids are never created here)
    and returns the ready-to-run ones, oldest first. Each item:
        {"session_id": int, "session_type": "SCAN"|"WATERING", "config": {...}|None}
    Returns an empty list in stub mode (no dashboard to poll).
    """
    if _is_stub():
        return []
    r = await _send("POST", f"/api/schedules/tick?bedId={bed_id}")
    try:
        return r.json().get("sessions", [])
    except Exception:
        return []


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
    await _send(
        "POST",
        f"/api/sessions/{session_id}/watering-stops",
        json={
            "stopIndex": stop_index,
            "xMm": x_mm,
            "yMm": y_mm,
            "maxHeightCm": max_height_cm,
            "valveDurationSec": valve_duration_sec,
        },
    )


async def post_watering_complete(session_id: int, summary: dict) -> None:
    """POST /sessions/{id}/complete — finalize watering session with summary."""
    if _is_stub():
        print(f"[pi_client:stub] post_watering_complete({session_id}, summary={summary})")
        return
    await _send(
        "POST",
        f"/api/sessions/{session_id}/complete",
        json={"summary": summary},
    )
