"""
session_state.py — tiny persistent marker for the active session.

This is NOT a data store. It holds exactly one fact: "session N (type T) is
currently running." It is written when a session starts and removed when it
ends cleanly. If the file is still present at the next startup, the process
died mid-session (crash / power loss / SIGKILL) — orphan recovery uses it to
mark that session as errored in the dashboard and safe the gantry.

See main.py lifespan (recover_orphan_session) and services/session_service.py.
"""

import json
from pathlib import Path

from config import settings


def _path() -> Path:
    d = Path(settings.runtime_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "active_session.json"


def set_active(session_id: int, session_type: str = "SCAN") -> None:
    """Record that a session is now running."""
    try:
        _path().write_text(
            json.dumps({"session_id": session_id, "session_type": session_type}),
            encoding="utf-8",
        )
    except Exception as e:
        # The marker is best-effort; never let it break a session start.
        print(f"[session_state] could not write active marker — {e}")


def clear() -> None:
    """Remove the active-session marker (called on clean session end)."""
    try:
        p = _path()
        if p.exists():
            p.unlink()
    except Exception as e:
        print(f"[session_state] could not clear active marker — {e}")


def get_active() -> dict | None:
    """Return the active-session marker if one survived a restart, else None."""
    p = _path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
