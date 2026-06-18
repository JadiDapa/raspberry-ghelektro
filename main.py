from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import sessions, camera, gantry, info, logs, servo, sensors
from services import camera as camera_service
from services import gantry as gantry_service
from services import yolo_service
from services import soil_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_service.start()
    gantry_service.connect()
    soil_service.connect()
    yolo_service.load_model()
    print("[main] FarmBot API ready")
    yield
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
