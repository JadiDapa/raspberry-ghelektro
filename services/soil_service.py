"""
Soil moisture service — communicates with ESP32 #2 over UART.

ESP32 #2 is connected to Raspberry Pi GPIO 14/15 (ttyAMA0).
It reads 3 capacitive soil moisture sensors covering the 2×8 plant grid:

  Sensor 0 → cols 0–2  (plants in the left third of the bed)
  Sensor 1 → cols 3–5  (plants in the middle third)
  Sensor 2 → cols 6–7  (plants in the right third)

Usage:
    from services.soil_service import read_moisture_for_plant
    pct = await read_moisture_for_plant(col=3)  # returns sensor 1 reading
"""

import asyncio
import json
import threading
import serial

from config import settings

# ─── Serial connection ────────────────────────────────────────────────────────

_ser: serial.Serial | None = None
_lock = threading.Lock()

# Sensor zone mapping — which sensor index covers which columns
# cols 0–2 → sensor 0, cols 3–5 → sensor 1, cols 6–7 → sensor 2
_COL_TO_SENSOR = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2,
}


def connect():
    """Open UART connection to ESP32 #2. Called once on startup."""
    global _ser
    try:
        _ser = serial.Serial(
            port=settings.soil_uart_port,
            baudrate=settings.soil_uart_baudrate,
            timeout=5.0,
        )
        import time; time.sleep(2.0)
        _ser.reset_input_buffer()
        print(f"[soil] connected → {settings.soil_uart_port} @ {settings.soil_uart_baudrate}")
    except Exception as e:
        print(f"[soil] WARNING: could not open UART — {e}")
        print("[soil] running in stub mode")
        _ser = None


def disconnect():
    """Close UART connection. Called on shutdown."""
    global _ser
    if _ser and _ser.is_open:
        _ser.close()
    print("[soil] disconnected")


# ─── Low-level send/receive ───────────────────────────────────────────────────

def _send(command: str) -> dict:
    """Send a command and wait for OK/ERR. Runs in a thread."""
    if _ser is None or not _ser.is_open:
        import random
        print(f"[soil:stub] {command}")
        return {"stub": True, "pct": round(random.uniform(30.0, 80.0), 1)}

    with _lock:
        _ser.reset_input_buffer()
        _ser.write((command.strip() + "\n").encode())
        print(f"[soil] → {command}")

        while True:
            raw = _ser.readline().decode(errors="replace").strip()
            if not raw:
                continue
            print(f"[soil] ← {raw}")

            if raw.startswith("OK "):
                return json.loads(raw[3:])
            elif raw.startswith("ERR "):
                raise RuntimeError(f"ESP32 #2 error: {raw[4:]}")
            # Skip debug lines (e.g. "[CMD] READ sensor=1")


async def _run(fn, *args):
    """Run blocking serial call in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ─── Public API ───────────────────────────────────────────────────────────────

def col_to_sensor(col: int) -> int:
    """Return the sensor index (0–2) that covers a given plant column."""
    return _COL_TO_SENSOR.get(col, 0)


async def read_sensor(sensor_index: int) -> float:
    """
    Read a specific sensor by index (0, 1, or 2).
    Returns moisture percentage 0–100.
    """
    result = await _run(_send, f"READ sensor={sensor_index}")
    return result.get("pct", 0.0)


async def read_all_sensors() -> dict:
    """
    Read all 3 sensors in one command.
    Returns {"s0": 62.3, "s1": 71.5, "s2": 51.2}
    """
    return await _run(_send, "READ sensor=all")


async def read_moisture_for_plant(col: int) -> float:
    """
    Read the soil moisture for a plant at a given column.
    Automatically picks the correct sensor based on column position.
    Returns moisture percentage 0–100.
    """
    sensor_index = col_to_sensor(col)
    print(f"[soil] reading sensor {sensor_index} for col {col}")
    return await read_sensor(sensor_index)


async def ping() -> bool:
    """Check if ESP32 #2 is alive."""
    try:
        await _run(_send, "PING")
        return True
    except Exception:
        return False
