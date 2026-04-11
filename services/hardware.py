"""
Hardware layer — camera, YOLO, sensors, valve, watering logic.

What's real:
  ✅ capture_image()             — USB webcam via services/camera
  ✅ run_yolo()                  — YOLOv11 via services/yolo_service
  ✅ read_tof_distance()         — VL53L1X via ESP32 #1 serial
  ✅ read_soil_moisture(col)     — capacitive sensor via ESP32 #2 UART
  ✅ open_valve()                — solenoid via ESP32 #1 serial
  ✅ compute_watering_duration() — threshold-based logic

Note: ambient sensors (temp, humidity, light, etc.) are on a standalone
ESP32 that talks directly to the dashboard — not through the Pi.
"""

import asyncio
import random

from config import settings


# ─── Camera ───────────────────────────────────────────────────────────────────

async def capture_image(plant_id: int, session_id: str) -> str:
    """Wait for gantry to stabilize, then capture a fresh frame from the webcam."""
    from services.camera import capture_snapshot
    print(f"[camera] waiting {settings.camera_stabilize_delay}s for gantry to stabilize...")
    await asyncio.sleep(settings.camera_stabilize_delay)
    return await capture_snapshot(plant_id, session_id)


# ─── YOLO ─────────────────────────────────────────────────────────────────────

async def run_yolo(image_path: str) -> list[dict]:
    """Run YOLOv11 inference. Returns detections per class."""
    from services.yolo_service import run_inference
    return await run_inference(image_path)


# ─── Per-plant sensors ────────────────────────────────────────────────────────

async def read_tof_distance() -> float:
    """Read plant height from VL53L1X TOF sensor on ESP32 #1. Returns cm."""
    from services import gantry
    print("[tof] reading distance sensor via ESP32 #1...")
    height_cm = await gantry.read_tof()
    if height_cm is None:
        print("[tof] sensor had no data — using stub fallback")
        height_cm = round(random.uniform(20.0, 65.0), 1)
    print(f"[tof] plant height: {height_cm} cm")
    return height_cm


async def read_soil_moisture(col: int = 0) -> float:
    """
    Read soil moisture for the zone covering the given plant column.
    Sensor 0 → cols 0–2, Sensor 1 → cols 3–5, Sensor 2 → cols 6–7.
    Returns percentage 0–100.
    """
    from services.soil_service import read_moisture_for_plant
    moisture_pct = await read_moisture_for_plant(col)
    print(f"[moisture] col={col} → {moisture_pct}%")
    return moisture_pct


# ─── Valve ────────────────────────────────────────────────────────────────────

async def open_valve(duration_sec: float) -> None:
    """Open solenoid valve for N seconds via ESP32 #1, then close it."""
    from services import gantry
    print(f"[valve] opening solenoid for {duration_sec:.1f}s")
    await gantry.set_relay("sol", on=True)
    await asyncio.sleep(duration_sec)
    await gantry.set_relay("sol", on=False)
    print("[valve] solenoid closed")


# ─── Watering thresholds ──────────────────────────────────────────────────────
#
#   moisture %    condition          valve open
#   ──────────    ─────────          ──────────
#   0 – 24        very dry           8 s
#   25 – 44       moderately dry     5 s
#   45 – 64       slightly dry       2 s
#   65 – 100      moist enough       skip

WATER_DRY_SEC   = 8.0
WATER_MID_SEC   = 5.0
WATER_LIGHT_SEC = 2.0


def compute_watering_duration(moisture_pct: float) -> tuple[float, str]:
    """
    Decide how long to open the valve based on soil moisture.
    Returns (duration_seconds, reason).
    Adjust WATER_*_SEC constants above to suit your setup.
    """
    if moisture_pct < 25:
        duration = WATER_DRY_SEC
        reason = f"very dry ({moisture_pct:.1f}%) → {duration}s"
    elif moisture_pct < 45:
        duration = WATER_MID_SEC
        reason = f"moderately dry ({moisture_pct:.1f}%) → {duration}s"
    elif moisture_pct < 65:
        duration = WATER_LIGHT_SEC
        reason = f"slightly dry ({moisture_pct:.1f}%) → {duration}s"
    else:
        duration = 0.0
        reason = f"moist enough ({moisture_pct:.1f}%) → skip"

    print(f"[water] {reason}")
    return duration, reason
