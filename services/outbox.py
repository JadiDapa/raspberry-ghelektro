"""
outbox.py — durable write-ahead store for end-of-session syncs.

When the dashboard is unreachable for a whole run, the session's full payload is
appended here (one JSON object per line) instead of being lost. On the next
startup, drain() replays each pending payload via pi_client.sync_session() and
deletes it once the dashboard confirms. This is a transient buffer, not a second
source of truth — entries live only until the dashboard has the data.
"""

import json
from pathlib import Path

from config import settings
from services import pi_client


def _dir() -> Path:
    d = Path(settings.outbox_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def append(payload: dict) -> str:
    """Persist one whole-session payload. Returns the file path written."""
    sid = payload.get("session_id", "unknown")
    path = _dir() / f"session_{sid}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"[outbox] queued session {sid} → {path}")
    return str(path)


async def drain() -> int:
    """
    Replay every pending payload to the dashboard. Entries that sync successfully
    are removed; entries that still fail are kept for the next attempt. Returns
    the number of payloads successfully drained.
    """
    drained = 0
    for path in sorted(_dir().glob("*.jsonl")):
        kept: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[outbox] dropping corrupt line in {path.name}")
                continue
            try:
                await pi_client.sync_session(payload)
                drained += 1
            except Exception as e:
                print(f"[outbox] replay failed for {path.name} — keeping for next startup ({e})")
                kept.append(line)

        if kept:
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    if drained:
        print(f"[outbox] drained {drained} pending session(s)")
    return drained
