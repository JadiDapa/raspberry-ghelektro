import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db import crud
from models.schemas import SessionCreate, serialize_session, serialize_plant_scan
from services import event_bus, session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Registry of running asyncio tasks, keyed by session_id.
# Needed so stop_session can cancel the scan loop — without this,
# calling /stop only updates the DB but the gantry keeps moving.
_tasks: dict[str, asyncio.Task] = {}

# Global guard — only one session may run at a time.
# The gantry has one serial port; concurrent sessions corrupt each other's commands.
_active_session_id: str | None = None

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# ─── List all sessions ────────────────────────────────────────────────────────


@router.get("")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    rows = await crud.get_all_sessions(db)
    return [serialize_session(r) for r in rows]


# ─── Create ───────────────────────────────────────────────────────────────────


@router.post("")
async def create_session(
    body: SessionCreate = SessionCreate(),
    db: AsyncSession = Depends(get_db),
):
    session_id = str(uuid.uuid4())[:8]
    row = await crud.create_session(db, session_id, body.notes)
    return serialize_session(row)


# ─── Get session detail ───────────────────────────────────────────────────────


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    row = await crud.get_session(db, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    return serialize_session(row)


# ─── Start ────────────────────────────────────────────────────────────────────


@router.post("/{session_id}/start")
async def start_session(session_id: str, db: AsyncSession = Depends(get_db)):
    global _active_session_id

    row = await crud.get_session(db, session_id)
    if not row:
        raise HTTPException(404, "Session not found")

    # Allow restarting stopped/error sessions by resetting them first
    if row.status in ("stopped", "error"):
        row = await crud.reset_session(db, session_id)

    if row.status != "created":
        raise HTTPException(409, f"Cannot start session with status '{row.status}'")

    # Clear stale active guard — in case a previous session errored without cleanup
    if _active_session_id is not None and _active_session_id != session_id:
        stale_task = _tasks.get(_active_session_id)
        if stale_task is None or stale_task.done():
            _active_session_id = None  # clear stale lock
        else:
            raise HTTPException(
                409, f"Another session ({_active_session_id}) is already running."
            )

    _active_session_id = session_id
    event_bus.create(session_id)
    task = asyncio.create_task(session_service.run_session(session_id))
    _tasks[session_id] = task

    # Auto-clean the registry when the task finishes naturally
    def _on_done(t: asyncio.Task) -> None:
        global _active_session_id
        _tasks.pop(session_id, None)
        if _active_session_id == session_id:
            _active_session_id = None

    task.add_done_callback(_on_done)

    return {"session_id": session_id, "status": "running"}


# ─── Stop ─────────────────────────────────────────────────────────────────────


@router.post("/{session_id}/stop")
async def stop_session(session_id: str, db: AsyncSession = Depends(get_db)):
    row = await crud.get_session(db, session_id)
    if not row:
        raise HTTPException(404, "Session not found")

    # Cancel the running scan task — this triggers CancelledError inside
    # run_session(), which turns the pump off and marks the session stopped.
    task = _tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task  # wait for the CancelledError handler to finish cleanly
        except asyncio.CancelledError:
            pass  # expected — the task was cancelled cleanly
    else:
        # Task already finished (e.g. session completed naturally) —
        # still update DB and clean up in case status wasn't set correctly
        await crud.set_session_stopped(db, session_id)
        event_bus.destroy(session_id)

    return {"session_id": session_id, "status": "stopped"}


# ─── SSE event stream ─────────────────────────────────────────────────────────


@router.get("/{session_id}/events")
async def session_events(session_id: str, db: AsyncSession = Depends(get_db)):
    row = await crud.get_session(db, session_id)
    if not row:
        raise HTTPException(404, "Session not found")

    async def stream():
        # Already finished — send a one-shot reconnect summary and close
        if row.status == "complete":
            scans = await crud.get_plant_scans(db, session_id)
            event = {
                "type": "session_reconnect",
                "session_id": session_id,
                "status": "complete",
                "plant_count": len(scans),
            }
            yield f"data: {json.dumps(event)}\n\n"
            return

        # Wait for the event bus to be created (dashboard may connect before /start)
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


@router.post("/{session_id}/restart")
async def restart_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Reset a stopped/error session and start it again."""
    row = await crud.get_session(db, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    if row.status == "running":
        raise HTTPException(409, "Session is already running — stop it first")
    await crud.reset_session(db, session_id)
    return await start_session(session_id, db)
