"""
scheduler.py — polls the Next.js dashboard for due scheduled sessions and launches them.

The dashboard owns all scheduling logic and mints the session ids (see
dashboard/server/services/schedule.service.ts). This loop is just the always-on
trigger: every `schedule_poll_seconds` it asks the dashboard which scheduled
sessions are ready, and — if the gantry is idle — launches the oldest one via the
same path the HTTP /start route uses (routers.sessions.launch_session). The
one-session-at-a-time guard means a session due while another is running simply
stays queued on the dashboard and is picked up on a later poll.

Errors are non-fatal: a failed poll is logged and the loop keeps running, mirroring
the "real-time dashboard posts are non-fatal" rule in session_service.
"""

import asyncio

from config import settings
from routers import sessions as sessions_router
from routers.sessions import StartSessionBody
from services import pi_client


def _enabled() -> bool:
    """Scheduling needs a dashboard to poll; skip in stub mode / no URL / disabled."""
    return (
        settings.schedule_enabled
        and not settings.stub_mode
        and bool(settings.dashboard_url.strip())
    )


async def _tick_once() -> None:
    try:
        bed_id = int(settings.bed_id)
    except (TypeError, ValueError):
        print(f"[scheduler] invalid bed_id {settings.bed_id!r} — skipping poll")
        return

    due = await pi_client.fetch_due_sessions(bed_id)
    if not due:
        return

    # Launch at most one per tick — the gantry runs one session at a time and the
    # rest stay PENDING on the dashboard until a later poll finds the gantry free.
    if sessions_router.is_active():
        print(f"[scheduler] {len(due)} session(s) due but gantry busy — leaving queued")
        return

    item = due[0]
    session_id = str(item["session_id"])
    session_type = item.get("session_type", "SCAN")
    config = item.get("config")
    body = StartSessionBody(
        session_type=session_type,
        scan_config=config if session_type == "SCAN" else None,
        watering_config=config if session_type == "WATERING" else None,
    )
    try:
        sessions_router.launch_session(session_id, body)
        print(f"[scheduler] launched scheduled {session_type} session {session_id}")
    except Exception as e:
        print(f"[scheduler] could not launch session {session_id} — {e}")


async def run_scheduler_loop() -> None:
    """Background task: poll the dashboard for due sessions until cancelled."""
    if not _enabled():
        print("[scheduler] disabled (stub mode / no dashboard_url / schedule_enabled=false)")
        return

    interval = max(10, settings.schedule_poll_seconds)
    print(f"[scheduler] polling dashboard for due sessions every {interval}s")
    try:
        while True:
            try:
                await _tick_once()
            except Exception as e:  # never let one bad poll kill the loop
                print(f"[scheduler] poll failed — {e}")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        print("[scheduler] stopped")
        raise
