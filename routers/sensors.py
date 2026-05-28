import random

from fastapi import APIRouter

from services import soil_service

router = APIRouter(prefix="/sensors", tags=["sensors"])


@router.get("/soil")
async def get_soil():
    """
    Read all 3 soil moisture sensors.
    Returns {"sensors": [{"id": 1, "label": "Sensor 1", "moisture_pct": 62.3}, ...]}

    Note: ambient sensors (environment, light) are on a standalone ESP32 that
    posts directly to the Next.js dashboard — they are not accessible from the RPi.
    """
    s0 = await soil_service.read_sensor(0)
    s1 = await soil_service.read_sensor(1)
    s2 = await soil_service.read_sensor(2)
    return {
        "sensors": [
            {"id": 1, "label": "Sensor 1", "moisture_pct": round(s0, 1)},
            {"id": 2, "label": "Sensor 2", "moisture_pct": round(s1, 1)},
            {"id": 3, "label": "Sensor 3", "moisture_pct": round(s2, 1)},
        ]
    }


@router.get("/environment")
async def get_environment():
    """
    Ambient temperature, humidity, and exhaust fan speed.
    These sensors are on a standalone ESP32 that talks directly to the dashboard.
    This stub endpoint satisfies dashboard polling while that ESP32 is offline.
    """
    return {
        "temperature_c": round(random.uniform(26.0, 32.0), 1),
        "humidity_pct": round(random.uniform(55.0, 75.0), 1),
        "exhaust_fan_speed_pct": 0.0,
    }


@router.get("/light")
async def get_light():
    """
    Ambient light level (lux).
    Same standalone ESP32 as /sensors/environment — stub data while offline.
    """
    return {"lux": round(random.uniform(800.0, 2500.0), 1)}
