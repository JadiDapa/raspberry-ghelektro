"""
Event bus — one asyncio.Queue per active session.

The scan loop calls emit() to push events.
The SSE endpoint awaits get() to stream them to the dashboard.
"""

import asyncio
from typing import Dict


_buses: Dict[str, asyncio.Queue] = {}


def create(session_id: str) -> None:
    """Called when a session starts. Creates a fresh queue."""
    _buses[session_id] = asyncio.Queue()
    print(f"[event_bus] created bus for session {session_id}")


def get(session_id: str) -> asyncio.Queue | None:
    return _buses.get(session_id)


def exists(session_id: str) -> bool:
    return session_id in _buses


async def emit(session_id: str, event: dict) -> None:
    """Push an event dict into the session queue."""
    bus = _buses.get(session_id)
    if bus:
        await bus.put(event)
        print(
            f"[event_bus] emit → {event.get('type')} (plant {event.get('plant_id', '-')})"
        )


def destroy(session_id: str) -> None:
    """Called when a session ends. Cleans up the queue."""
    _buses.pop(session_id, None)
    print(f"[event_bus] destroyed bus for session {session_id}")
