"""
Gantry service — communicates with ESP32 over USB Serial.

The ESP32 firmware uses this protocol:
  Pi  → ESP32 : "CMD arg1=val1 arg2=val2\n"
  ESP32 → Pi  : "OK <json>\n"    success
                "ERR <message>\n" failure
                "DONE <json>\n"   when a MOVE finishes (async)

This module is the only place that talks to the serial port.
All other code (session_service, routers) calls these async functions.
"""

import asyncio
import json
import threading
import serial
from config import settings

# ─── Serial connection ────────────────────────────────────────────────────────

_ser: serial.Serial | None = None
_lock = threading.Lock()  # one command at a time

# In-memory state — updated after every successful command
_state = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "busy": False,
    "homed": False,
}


def connect():
    """Open serial port. Call once on startup."""
    global _ser
    try:
        _ser = serial.Serial(
            port=settings.esp32_port,
            baudrate=settings.esp32_baudrate,
            timeout=5.0,
        )
        # Flush any boot messages from the ESP32
        import time; time.sleep(2.0)
        _ser.reset_input_buffer()
        print(f"[gantry] connected → {settings.esp32_port} @ {settings.esp32_baudrate}")
    except Exception as e:
        print(f"[gantry] WARNING: could not open serial port — {e}")
        print("[gantry] running in stub mode")
        _ser = None


def disconnect():
    """Close serial port. Call on shutdown."""
    global _ser
    if _ser and _ser.is_open:
        _ser.close()
    print("[gantry] disconnected")


# ─── Low-level send/receive ───────────────────────────────────────────────────

def _send(command: str) -> dict:
    """
    Send a command to ESP32 and wait for OK/ERR response.
    Runs in a thread (called via run_in_executor).
    Returns parsed JSON payload or raises RuntimeError.
    """
    if _ser is None or not _ser.is_open:
        # Stub fallback — pretend it worked
        print(f"[gantry:stub] {command}")
        return {"ok": True, "stub": True}

    with _lock:
        _ser.reset_input_buffer()
        _ser.write((command.strip() + "\n").encode())
        print(f"[gantry] → {command}")

        while True:
            raw = _ser.readline().decode(errors="replace").strip()
            if not raw:
                continue
            print(f"[gantry] ← {raw}")

            if raw.startswith("OK "):
                return json.loads(raw[3:])
            elif raw.startswith("ERR "):
                raise RuntimeError(f"ESP32 error: {raw[4:]}")
            # Skip debug lines that start with [ (like "[boot]", "[CMD]")
            # and keep reading until we get OK or ERR


def _send_and_wait_done(command: str) -> dict:
    """
    Send a MOVE command and wait for both OK (accepted) and DONE (finished).
    Runs in a thread (called via run_in_executor).
    """
    if _ser is None or not _ser.is_open:
        print(f"[gantry:stub] {command}")
        import time; time.sleep(settings.gantry_move_delay)
        return {"ok": True, "stub": True}

    with _lock:
        _ser.reset_input_buffer()
        _ser.write((command.strip() + "\n").encode())
        print(f"[gantry] → {command}")

        got_ok = False
        # Wait for OK first, then DONE
        while True:
            raw = _ser.readline().decode(errors="replace").strip()
            if not raw:
                continue
            print(f"[gantry] ← {raw}")

            if raw.startswith("ERR "):
                raise RuntimeError(f"ESP32 error: {raw[4:]}")

            if not got_ok and raw.startswith("OK "):
                got_ok = True
                continue  # keep reading until DONE

            if got_ok and raw.startswith("DONE "):
                return json.loads(raw[5:])


# ─── Public async API ─────────────────────────────────────────────────────────

def get_state() -> dict:
    return dict(_state)


async def _run(fn, *args):
    """Run a blocking serial call in a thread pool so asyncio stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def ping() -> bool:
    """Check if ESP32 is alive. Returns True if responsive."""
    try:
        await _run(_send, "PING")
        return True
    except Exception:
        return False


async def home(axis: str = "all") -> dict:
    """
    Home one or all axes. Blocking — waits until homing completes.
    axis: "all" | "x" | "y" | "z"
    """
    _state["busy"] = True
    print(f"[gantry] homing axis={axis}")
    try:
        result = await _run(_send, f"HOME axis={axis}")
        _state.update({"x": 0.0, "y": 0.0, "z": 0.0, "busy": False, "homed": True})
        return get_state()
    except Exception as e:
        _state["busy"] = False
        raise RuntimeError(f"Homing failed: {e}")


async def move_to(x: float, y: float, z: float, speed: int = 150) -> dict:
    """
    Move gantry to absolute XYZ position in mm.
    Waits until motion completes (DONE response).
    """
    _state["busy"] = True
    print(f"[gantry] moving → X={x} Y={y} Z={z} speed={speed}")
    try:
        result = await _run(_send_and_wait_done, f"MOVE x={x} y={y} z={z} speed={speed}")
        _state.update({"x": x, "y": y, "z": z, "busy": False})
        return get_state()
    except Exception as e:
        _state["busy"] = False
        raise RuntimeError(f"Move failed: {e}")


async def move_to_plant(row: int, col: int) -> dict:
    """
    Move above a plant grid position.
    Raises Z first (safe clearance), then moves XY, then lowers to working height.
    """
    x_mm = col * 750.0   # 6000mm / 8 cols = 750mm spacing
    y_mm = row * 1000.0  # 2000mm / 2 rows = 1000mm spacing
    z_mm = 50.0          # working height above plant canopy

    # Raise Z first before XY travel (safe move — avoid hitting plants)
    await move_to(_state["x"], _state["y"], 0.0)
    await move_to(x_mm, y_mm, z_mm)
    return get_state()


async def emergency_stop() -> dict:
    """Immediately halt all motors."""
    _state["busy"] = False
    print("[gantry] EMERGENCY STOP")
    try:
        await _run(_send, "STOP")
    except Exception:
        pass  # stop is best-effort
    return get_state()


async def get_position() -> dict:
    """Read current XYZ position from ESP32."""
    result = await _run(_send, "POS")
    _state.update({"x": result.get("x", 0), "y": result.get("y", 0), "z": result.get("z", 0)})
    return get_state()


async def get_limits() -> dict:
    """Read limit switch states from ESP32."""
    return await _run(_send, "LIMITS")


async def set_relay(channel: str, on: bool) -> dict:
    """
    Control solenoid valve or water pump.
    channel: "sol" (solenoid valve) | "dc" (water pump)
    """
    return await _run(_send, f"RELAY ch={channel} on={1 if on else 0}")


async def read_tof() -> float | None:
    """
    Read plant height from VL53L1X TOF sensor via ESP32.
    Returns distance in cm, or None if sensor has no data.
    """
    result = await _run(_send, "TOF")
    mm = result.get("mm")
    if mm is None:
        return None
    return round(mm / 10.0, 1)  # mm → cm
