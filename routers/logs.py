"""
Logs router — serves per-session log files to the dashboard.

GET /logs/{session_id}          → full log as plain text
GET /logs/{session_id}/tail     → last N lines  (?lines=100)
GET /logs/                      → list all log files
"""

import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from services.session_logger import SessionLogger

router = APIRouter(prefix="/logs", tags=["logs"])

LOG_DIR = SessionLogger.LOG_DIR


def _log_path(session_id: str) -> str:
    return os.path.join(LOG_DIR, f"session_{session_id}.log")


@router.get("", response_class=PlainTextResponse)
async def list_logs():
    """Return a newline-separated list of all session log filenames."""
    if not os.path.isdir(LOG_DIR):
        return ""
    files = sorted(
        f
        for f in os.listdir(LOG_DIR)
        if f.startswith("session_") and f.endswith(".log")
    )
    return "\n".join(files)


@router.get("/{session_id}", response_class=PlainTextResponse)
async def get_log(session_id: str):
    """Return the full log for a session."""
    path = _log_path(session_id)
    if not os.path.isfile(path):
        raise HTTPException(404, f"No log found for session {session_id!r}")
    with open(path, encoding="utf-8") as f:
        return f.read()


@router.get("/{session_id}/tail", response_class=PlainTextResponse)
async def tail_log(session_id: str, lines: int = Query(default=100, ge=1, le=5000)):
    """Return the last N lines of a session log (default 100)."""
    path = _log_path(session_id)
    if not os.path.isfile(path):
        raise HTTPException(404, f"No log found for session {session_id!r}")
    with open(path, encoding="utf-8") as f:
        all_lines = f.readlines()
    return "".join(all_lines[-lines:])
