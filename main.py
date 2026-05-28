from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
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
