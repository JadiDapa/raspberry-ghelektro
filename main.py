import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from routers import sessions, camera, gantry, info, logs, servo, sensors
from services import camera as camera_service
from services.camera import set_controls as camera_set_controls
from services import gantry as gantry_service
from services import yolo_service
from services import soil_service
from services import pi_client, outbox, session_state, scheduler


async def _recover_orphan_session() -> None:
    """
    If a session marker survived a restart, the process died mid-session. Safe the
    gantry and mark that session errored in the dashboard so it doesn't hang in
    RUNNING forever (and motors aren't left energized).
    """
    orphan = session_state.get_active()
    if not orphan:
        return
    sid = orphan.get("session_id")
    print(f"[orphan] unfinished session {sid} from a previous run — safing gantry + marking error")
    try:
        await gantry_service.emergency_stop()
    except Exception as e:
        print(f"[orphan] emergency_stop failed — {e}")
    try:
        await pi_client.post_error(int(sid))
    except Exception as e:
        print(f"[orphan] could not mark session {sid} as error in dashboard — {e}")
    session_state.clear()


async def _restore_camera_settings() -> None:
    """Load this bed's saved camera controls from the dashboard and apply them so
    manual exposure/WB/focus survive a reboot. Non-fatal: any failure just leaves
    the camera on its defaults."""
    try:
        saved = await pi_client.fetch_camera_settings(int(settings.bed_id))
    except Exception as e:
        print(f"[camera] could not load saved settings from dashboard — {e}")
        return
    if saved:
        camera_set_controls(saved)
        print("[camera] restored saved settings from dashboard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_service.start()
    await _restore_camera_settings()
    gantry_service.connect()
    soil_service.connect()
    yolo_service.load_model()
    await _recover_orphan_session()
    await outbox.drain()  # replay any sessions queued during a past outage
    scheduler_task = asyncio.create_task(scheduler.run_scheduler_loop())
    print("[main] FarmBot API ready")
    yield
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    gantry_service.disconnect()
    soil_service.disconnect()
    camera_service.stop()
    print("[main] FarmBot API shut down")


app = FastAPI(
    title="FarmBot API",
    description="CNC gantry plant monitoring and watering system",
    version="0.1.0",
    lifespan=lifespan,
)

# The dashboard sends no cookies/credentials to the Pi (see dashboard/lib/pi.ts),
# so we don't need allow_credentials. With credentials off, a wildcard origin is
# valid and lets the dashboard reach the Pi from any host — public IP, Tailscale
# IP, localhost, Vercel preview — without maintaining an allow list.
# Optionally lock this down by setting CORS_ORIGINS in .env (comma-separated).
_explicit_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_explicit_origins or ["*"],
    allow_credentials=bool(_explicit_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve fallback-saved capture images so the dashboard can fetch them during a
# /sync after an outage (see services/image_store.py). Mounting the "static" root
# exposes <images_dir> at /static/images/...
Path(settings.images_dir).mkdir(parents=True, exist_ok=True)
Path(settings.videos_dir).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(sessions.router)
app.include_router(camera.router)
app.include_router(gantry.router)
app.include_router(servo.router)
app.include_router(sensors.router)
app.include_router(info.router)
app.include_router(logs.router)


@app.get("/health")
async def health():
    return {"status": "ok", "system": "farmbot-api"}


@app.get("/")
async def root():
    return {"message": "FarmBot API is running", "docs": "/docs", "health": "/health"}
