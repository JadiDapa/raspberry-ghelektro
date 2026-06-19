# CLAUDE.md — Raspberry Pi FastAPI

Greenhouse UNSRI Chili Harvesting System. This service runs on a Raspberry Pi 4,
controls the CNC gantry robot, and orchestrates full plant scanning and watering
sessions.

It is **stateless with respect to persistence** — there is **no SQLite**. Next.js
(PostgreSQL via Prisma) is the single source of truth. During a session the RPi
writes plant data to Next.js in real time via `services/pi_client.py`. The only
local state is **transient buffers** used purely for crash/outage resilience
(see "Resilience" below), never a second copy of the data.

## Commands

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload   # dev
uvicorn main:app --host 0.0.0.0 --port 8000             # production
# Swagger UI: http://<pi-ip>:8000/docs
```

No test suite. Use `/docs` Swagger UI to test endpoints manually.
All hardware services degrade to stub mode when hardware is absent
(set `STUB_MODE=true` to also bypass the dashboard HTTP calls).

## Critical — read before touching anything

- **Serial ports are blocking** — all serial and YOLO calls use `loop.run_in_executor()`
  to avoid blocking the asyncio event loop. Never add synchronous I/O directly in async functions.
  Use `asyncio.get_running_loop()`, not the deprecated `get_event_loop()`.
- **One session at a time** — enforced via `_active_session_id` in `routers/sessions.py`.
  Do not remove this guard.
- **`proxy.ts` middleware** is on the Next.js side — do not look for it here.
- **YOLO executor** is a dedicated `ThreadPoolExecutor(max_workers=1)` — keeps inference
  serialized. Do not change to default executor.
- **Real-time dashboard posts are non-fatal.** In `services/session_service.py` a failed
  `pi_client` call is logged and the scan continues; it does not crash the session. Keep it
  that way — only gantry/sensor faults should be fatal.
- **Orphan recovery** runs at startup in `main.py` lifespan. A tiny marker
  (`services/session_state.py`) is written when a session starts and removed when it ends.
  If it survives a restart, the process died mid-session → recovery safes the gantry
  (`emergency_stop`) and marks that session `error` in the dashboard. Do not remove.

## Hardware topology

```
Raspberry Pi 4
├── ESP32 #1  /dev/ttyUSB0   — 4× stepper (DRV8825), VL53L0X TOF, solenoid valve, pump relay
├── ESP32 #2  /dev/ttyAMA0   — 3× capacitive moisture sensors (cols 0–2, 3–5, 6–7)
└── USB webcam /dev/video0   — OpenCV MJPEG capture

Standalone ESP32 (not through Pi) — ambient sensors → posts directly to Next.js dashboard
```

**Serial protocol (both ESP32s):**

```
Send:    "CMD arg1=val1 arg2=val2\n"
Receive: "OK <json>\n"        — success
         "ERR <msg>\n"        — failure
         "DONE <json>\n"      — async completion (MOVE only)
```

Both serial drivers (`gantry.py`, `soil_service.py`) attempt **one reconnect** on a
disconnect and then retry; a failed reconnect raises (it never silently returns stub
data) so the session can stop the machine instead of trusting fake readings.

**Plant grid:** defined by `ScanConfig` (`models/scan_config.py`). Default 2 rows × 8 cols.
Gantry coordinates: X = `start_x_mm + col·gap_x_mm`, Y = `start_y_mm + row·gap_y_mm`.
Z=0 = home/top, Z>0 = working depth. Configs are validated against the gantry travel
envelope (X≤6000, Y≤2000, Z≤800 mm) and rejected if a position would exit it.

## Session lifecycle

```
running → complete | stopped | error
```

**Scan** orchestration is in `services/session_service.py:run_session()`.
Per-plant sequence:

```
move gantry → set servo → capture image → upload image to Next.js (non-fatal)
→ YOLO inference → POST vision result to Next.js (non-fatal) → emit SSE event
```

Scan sessions do **not** read TOF/moisture or water — that is the watering flow.

**Watering** orchestration is in `services/watering_session_service.py`: it homes,
sweeps TOF heights across all positions, computes valve duration via fuzzy logic
(`services/fuzzy_watering.py`), then waters each column, posting per-column stops to
Next.js. Both flows share the SSE event bus and the same start/stop endpoints.

## Persistence — real-time first, /sync only as a fallback

Plant data is written to Next.js **in real time** during the session (status patch,
image upload, vision result). These power both history and the live "reconnect to a
running session" view on the dashboard.

The end-of-session whole-session `POST /api/sessions/sync` is a **fallback, not a
routine** — it fires only if a real-time post failed during the run (tracked by a
`sync_dirty` flag). In the normal case nothing is double-written.

## Resilience (transient local buffers only)

- **Retry** — every `pi_client` call goes through `_send()`: bounded retry with
  exponential backoff on transport errors / 5xx; 4xx is not retried.
- **Outbox** (`services/outbox.py`) — if even the `/sync` fallback can't reach Next.js,
  the whole-session payload is appended to `pending_sync/session_<id>.jsonl` and replayed
  on the next startup, then deleted once the dashboard confirms.
- **Image fallback** (`services/image_store.py`) — images normally go straight to Next.js.
  Only when a live upload failed is the JPEG written to `static/images/session_<id>/` and
  served at `/static/...` so the dashboard can fetch it during the `/sync` replay.
- **Active-session marker** (`services/session_state.py`) — see orphan recovery above.

These buffers hold in-flight data only until the dashboard has it. They are not a
queryable store and never the source of truth.

## SSE — owned entirely by this service

Browser connects directly: `GET /sessions/{id}/events`
`services/event_bus.py` holds one `asyncio.Queue` per active session.
The scan/watering loop calls `event_bus.emit()`, the SSE endpoint drains it.
Bus is created at session start, destroyed ~2 s after session end.

**SSE event types** (see `dashboard/lib/pi.ts` for the exact union):

```
Shared : session_started | motors_homed | session_complete | session_error | session_reconnect
Scan   : gantry_moving | gantry_moved | plant_scanned
Water  : tof_sweep_started | tof_position_scanned | tof_sweep_complete |
         moisture_read_before | fuzzy_computed | watering_stop | moisture_read_after
```

Do not add a WebSocket alternative. Do not proxy SSE through Next.js.

## Next.js sync — whole-session payload (fallback path)

`POST {DASHBOARD_URL}/api/sessions/sync`. The dashboard reconciles by the **integer
session id** (the Next.js `Session.id` that was passed to the RPi at `/start`), so a
late sync updates the existing row instead of creating a duplicate. `session_id` is sent
as a string; `bed_id` comes from settings.

```python
{
    "session_id": str,          # the Next.js Session.id this run belongs to
    "bed_id": str,              # which Bed in Next.js DB (settings.bed_id)
    "status": "complete" | "stopped" | "error",
    "started_at": str,          # ISO 8601
    "completed_at": str,
    "plant_scans": [...],       # full PiPlantScan list (mirrors dashboard/lib/pi.ts)
    "summary": {
        "total_plants": int,
        "avg_height_cm": float,
        "avg_moisture_pct": float,
        "total_water_sec": float,
        "ripeness": { "ripe": int, "turning": int, "unripe": int, "broken": int },
        "harvest_ready_ids": [int]   # plant_ids with ripe_count > 5
    }
}
```

If the POST fails it is queued to the outbox and replayed at the next startup.

## Key files

| File                                  | Purpose                                                            |
| ------------------------------------- | ----------------------------------------------------------------- |
| `main.py`                             | Entry point, lifespan (orphan recovery + outbox drain), `/static` mount |
| `config.py`                           | All settings via pydantic-settings, `FRUIT_CLASSES`               |
| `routers/`                            | `sessions`, `camera`, `gantry`, `servo`, `sensors`, `info`, `logs` |
| `models/scan_config.py`               | `ScanConfig` + `CaptureOffset` (travel-validated)                 |
| `models/watering_config.py`           | `WateringConfig` (travel-validated)                               |
| `services/session_service.py`         | Scan-loop orchestration + resilience                              |
| `services/watering_session_service.py`| Watering-loop orchestration                                       |
| `services/fuzzy_watering.py`          | Fuzzy-logic valve-duration computation                            |
| `services/gantry.py`                  | ESP32 #1 serial driver (motion, TOF, valve, pump) + reconnect     |
| `services/soil_service.py`            | ESP32 #2 UART driver (moisture) + reconnect                       |
| `services/camera.py`                  | Background capture thread, MJPEG stream, snapshot                 |
| `services/yolo_service.py`            | YOLOv11n model load + threaded inference                          |
| `services/hardware.py`                | Facade wiring camera/YOLO/sensors/valve for the loops             |
| `services/pi_client.py`               | HTTP client to Next.js (retry + `sync_session`)                   |
| `services/outbox.py`                  | Durable replay queue for failed end-of-session syncs              |
| `services/image_store.py`             | On-disk image fallback (served via `/static`)                     |
| `services/session_state.py`           | Active-session marker for orphan recovery                         |
| `services/event_bus.py`               | Per-session SSE event queues                                      |
| `services/session_logger.py`          | Per-session structured log files                                  |

## Configuration (.env)

All settings have safe defaults in `config.py`; override via `.env`.

```
# Hardware
ESP32_PORT=/dev/ttyUSB0          # ESP32 #1 (motion) USB serial
SOIL_UART_PORT=/dev/ttyAMA0      # ESP32 #2 (moisture) GPIO UART
CAMERA_DEVICE=/dev/video0

# YOLO
YOLO_MODEL_PATH=best.pt          # prefer NCNN export for speed; excluded from git
YOLO_CONFIDENCE=0.4
CAMERA_STABILIZE_DELAY=1.0

# Dashboard sync
DASHBOARD_URL=http://<nextjs-ip>:3000   # Next.js app URL (empty → stub mode)
RPI_BASE_URL=http://<pi-ip>:8000        # used to build fallback image URLs
BED_ID=1                                 # Next.js Bed.id this RPi manages

# Sync resilience
SYNC_MAX_RETRIES=3
SYNC_BACKOFF_BASE=0.5
OUTBOX_DIR=pending_sync
RUNTIME_DIR=runtime

# CORS (empty = allow any origin, credentials off) / dev
CORS_ORIGINS=
STUB_MODE=false
```

YOLO model is excluded from git — download `best.pt` separately or export to NCNN format.
