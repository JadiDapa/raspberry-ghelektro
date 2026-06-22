import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.scan_config import ScanConfig as ScanConfigModel
from models.watering_config import WateringConfig as WateringConfigModel
from services import event_bus, session_service, session_state, watering_session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


class StartSessionBody(BaseModel):
    session_type: str = "SCAN"  # "SCAN" | "WATERING"
    scan_config: Optional[ScanConfigModel] = None
    watering_config: Optional[WateringConfigModel] = None

# Registry of running asyncio tasks, keyed by session_id string.
# Needed so stop_session can cancel the scan loop.
_tasks: dict[str, asyncio.Task] = {}

# Global guard — only one session may run at a time.
_active_session_id: str | None = None

def is_active() -> bool:
    """Returns True if a session is currently running."""
    return _active_session_id is not None


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# ─── Start ────────────────────────────────────────────────────────────────────


def launch_session(session_id: str, body: StartSessionBody) -> dict:
    """
    Start a session's async run loop. Shared by the HTTP /start route and the
    scheduler (services/scheduler.py) so both honour the one-session-at-a-time
    guard and orphan marker identically. Raises HTTPException(409) if another
    session is already running.
    """
    global _active_session_id

    if _active_session_id is not None and _active_session_id != session_id:
        stale_task = _tasks.get(_active_session_id)
        if stale_task is None or stale_task.done():
            _active_session_id = None
        else:
            raise HTTPException(
                409, f"Another session ({_active_session_id}) is already running."
            )

    _active_session_id = session_id
    event_bus.create(session_id)

    # Persist a tiny marker so a mid-session crash/power-loss is detected and
    # cleaned up at the next startup (orphan recovery in main.py).
    session_state.set_active(int(session_id), body.session_type)

    if body.session_type == "WATERING":
        wconfig = body.watering_config if body.watering_config is not None else WateringConfigModel()
        task = asyncio.create_task(
            watering_session_service.run_watering_session(int(session_id), wconfig)
        )
    else:
        sconfig = body.scan_config if body.scan_config is not None else ScanConfigModel()
        task = asyncio.create_task(
            session_service.run_session(int(session_id), sconfig)
        )

    _tasks[session_id] = task

    def _on_done(t: asyncio.Task) -> None:
        global _active_session_id
        _tasks.pop(session_id, None)
        # The session loop finished (any outcome) — the process is alive and has
        # handled it, so the orphan marker is no longer needed.
        session_state.clear()
        if _active_session_id == session_id:
            _active_session_id = None

    task.add_done_callback(_on_done)
    return {"session_id": session_id, "status": "running"}


@router.post("/{session_id}/start")
async def start_session(
    session_id: str,
    body: StartSessionBody = Body(default=StartSessionBody()),
):
    return launch_session(session_id, body)


# ─── Stop ─────────────────────────────────────────────────────────────────────


@router.post("/{session_id}/stop")
async def stop_session(session_id: str):
    task = _tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return {"session_id": session_id, "status": "stopped"}


# ─── SSE event stream ─────────────────────────────────────────────────────────


@router.get("/{session_id}/events")
async def session_events(session_id: str):
    async def stream():
        # Wait for the event bus to be created (client may connect before /start)
        for _ in range(50):
            if event_bus.exists(session_id):
                break
            await asyncio.sleep(0.1)
        else:
            error = {"type": "session_error", "message": "scan loop not started"}
            yield f"data: {json.dumps(error)}\n\n"
            return

        bus = event_bus.get(session_id)
        print(f"[sse] client connected → session {session_id}")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(bus.get(), timeout=60.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("session_complete", "session_error"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            print(f"[sse] client disconnected → session {session_id}")

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )
