# CLAUDE.md — Raspberry Pi FastAPI

Greenhouse UNSRI Chili Harvesting System. This service runs on a Raspberry Pi 4,
controls the CNC gantry robot, and orchestrates full plant scanning sessions.
It is **stateless with respect to persistence** — SQLite is used only as a local
cache during and after sessions until Next.js confirms sync.

## Commands

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload   # dev
uvicorn main:app --host 0.0.0.0 --port 8000             # production
# Swagger UI: http://<pi-ip>:8000/docs
```

No test suite. Use `/docs` Swagger UI to test endpoints manually.
All hardware services degrade to stub mode when hardware is absent.

## Critical — read before touching anything

- **Serial ports are blocking** — all serial and YOLO calls use `loop.run_in_executor()`
  to avoid blocking the asyncio event loop. Never add synchronous I/O directly in async functions.
- **One session at a time** — enforced via `_active_session_id` in `routers/sessions.py`.
  Do not remove this guard.
- **`proxy.ts` middleware** is on the Next.js side — do not look for it here.
- **YOLO executor** is a dedicated `ThreadPoolExecutor(max_workers=1)` — keeps inference
  serialized. Do not change to default executor.
- **Orphan recovery** runs at startup in `main.py` lifespan — resets any `running`
  sessions to `error` on restart. Do not remove.

## Hardware topology

```
Raspberry Pi 4
├── ESP32 #1  /dev/ttyUSB0   — 4× stepper (DRV8825), VL53L1X TOF, solenoid valve, pump relay
├── ESP32 #2  /dev/ttyAMA0   — 3× capacitive moisture sensors (cols 0–2, 3–5, 6–7)
└── USB webcam /dev/video0   — OpenCV MJPEG capture

Standalone ESP32 (not through Pi) — ambient sensors → posts directly to Next.js dashboard
```

**Serial protocol (both ESP32s):**

```
Send:    "CMD arg1=val1 arg2=val2\n"
Receive: "OK <json>\n"        — success
         "ERR <msg>\n"        — failure
         "DONE <json>\n"      — async completion (MOVE, HOME only)
```

**Plant grid:** `PLANT_GRID` in `config.py` is canonical (2×8 list).
Gantry coordinates: X = col × 750mm, Y = row × 1000mm. Z=0 = home/top, Z=50mm = working height.

## Session lifecycle

```
created → running → complete | stopped | error
```

Orchestration loop in `services/session_service.py:run_session()`.
Per-plant sequence: move gantry → capture image → YOLO inference → read TOF → read moisture
→ decide watering → open valve → emit SSE events → save to local SQLite.

On session end (complete/stopped/error): POST full results to Next.js
`POST {DASHBOARD_URL}/api/sessions/sync` — this is the persistence handoff.

## SSE — owned entirely by this service

Browser connects directly: `GET /sessions/{id}/events`
`services/event_bus.py` holds one `asyncio.Queue` per active session.
Scan loop calls `event_bus.emit()`, SSE endpoint drains it.
Bus is created at session start, destroyed ~2s after session end.

**SSE event types emitted:**
`session_started` | `motors_homed` | `gantry_moving` | `gantry_moved` | `plant_scanned` |
`sensor_read` | `plant_watered` | `session_complete` | `session_error` | `session_reconnect`

Do not add a WebSocket alternative. Do not proxy SSE through Next.js.

## Watering logic (in `services/hardware.py`)

| Moisture % | Valve duration    |
| ---------- | ----------------- |
| 0–24       | 8s (very dry)     |
| 25–44      | 5s (dry)          |
| 45–64      | 2s (slightly dry) |
| ≥65        | skip              |

Harvest-ready threshold: `ripe_count > 5` → included in `harvest_ready_ids` summary.

## Next.js sync — what to POST after session ends

```python
# POST to {DASHBOARD_URL}/api/sessions/sync
{
    "session_id": str,          # RPi's UUID — stored as externalId in Next.js
    "bed_id": str,              # which Bed in Next.js DB
    "status": "complete" | "stopped" | "error",
    "started_at": str,          # ISO 8601
    "completed_at": str,
    "plant_scans": [...],       # full PiPlantScan list
    "summary": {
        "total_plants": int,
        "avg_height_cm": float,
        "avg_moisture_pct": float,
        "total_water_sec": float,
        "ripeness": { "ripe": int, "turning": int, "unripe": int, "broken": int },
        "harvest_ready_ids": [int]
    }
}
```

If POST fails (network down), log the error and keep data in local SQLite.
Add a retry endpoint `POST /sessions/{id}/sync` so Next.js can pull on demand.

## Key files

| File                          | Purpose                                                                |
| ----------------------------- | ---------------------------------------------------------------------- |
| `main.py`                     | Entry point, lifespan hooks, router registration                       |
| `config.py`                   | All settings via pydantic-settings, `PLANT_GRID`, `FRUIT_CLASSES`      |
| `services/session_service.py` | Full scan-loop orchestration                                           |
| `services/gantry.py`          | ESP32 #1 serial driver                                                 |
| `services/soil_service.py`    | ESP32 #2 serial driver                                                 |
| `services/camera.py`          | Background capture thread, MJPEG stream, snapshot                      |
| `services/yolo_service.py`    | YOLOv11n model load + threaded inference (ONNX preferred)              |
| `services/event_bus.py`       | Per-session SSE event queues                                           |
| `services/hardware.py`        | Facade wiring camera/YOLO/sensors/valve for scan loop + watering logic |
| `db/database.py`              | SQLite async engine (`farmbot.db`)                                     |
| `db/models.py`                | `Session` and `PlantScan` ORM models                                   |
| `db/crud.py`                  | All DB read/write operations                                           |
| `models/schemas.py`           | Pydantic request/response schemas                                      |

## Configuration (.env)

```
ESP32_PORT=/dev/ttyUSB0
SOIL_UART_PORT=/dev/ttyAMA0
CAMERA_INDEX=0
YOLO_MODEL_PATH=best.pt   # prefer NCNN over .pt for speed
YOLO_CONFIDENCE=0.4
CAMERA_STABILIZE_DELAY=1.0
DASHBOARD_URL=http://<nextjs-ip>:3000  # Next.js app URL for sync POST
```

YOLO model is excluded from git — download `best.pt` separately or export to NCNN format.
