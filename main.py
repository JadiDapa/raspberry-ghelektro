import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from db.database import init_db
from routers import sessions, camera, gantry, info, plants
from services import camera as camera_service
from services import gantry as gantry_service
from services import yolo_service
from services import soil_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    camera_service.start()
    gantry_service.connect()  # ESP32 #1 — /dev/ttyUSB0
    soil_service.connect()  # ESP32 #2 — /dev/ttyAMA0
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.images_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(sessions.router)
app.include_router(camera.router)
app.include_router(gantry.router)
app.include_router(info.router)
app.include_router(plants.router)


@app.get("/health")
async def health():
    return {"status": "ok", "system": "farmbot-api"}


@app.get("/")
async def root():
    return {"message": "FarmBot API is running", "docs": "/docs", "health": "/health"}
