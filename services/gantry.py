"""
Gantry service — communicates with ESP32 over USB Serial.

The ESP32 firmware uses this protocol:
  Pi  → ESP32 : "CMD arg1=val1 arg2=val2\n"
  ESP32 → Pi  : "OK <json>\n"    success
                "ERR <message>\n" failure
                "DONE <json>\n"   when a MOVE finishes (async)

This module is the only place that talks to the serial port.
All other code (session_service, routers) calls these async functions.

Timeout design
──────────────
readline() uses a SHORT per-call timeout (SERIAL_READ_TIMEOUT = 1 s).
When the ESP32 doesn't answer (or sends debug lines only), readline()
returns "" every second. The _send() loop tracks a wall-clock deadline
and raises RuntimeError("timeout") when COMMAND_TIMEOUT seconds pass
without receiving an OK or ERR. This prevents the session from hanging
silently forever.

  COMMAND_TIMEOUT   = 10 s  — PING, EN, RELAY, POS, LIMITS, TOF
  MOVE_TIMEOUT      = 300 s — MOVE / HOME (may take minutes on a 6m bed)
"""

import asyncio
import json
import threading
import time
import serial
from config import settings

# ─── Timeouts ─────────────────────────────────────────────────────────────────

SERIAL_READ_TIMEOUT = 1.0  # per readline() call (s) — short so deadline loop ticks fast
COMMAND_TIMEOUT = 10.0  # max wait for OK/ERR on quick commands (s)
MOVE_TIMEOUT = 300.0  # max wait for DONE on MOVE/HOME (s) — covers a full 6 m traverse

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
        # FIX: Open with dsrdtr/rtscts disabled, then deassert DTR as a property.
        # pyserial does not accept dtr= in the constructor — it must be set after open.
        # DTR toggling on port-open resets the ESP32 via its EN pin, causing it
        # to miss the first command sent right after connect().
        _ser = serial.Serial(
            port=settings.esp32_port,
            baudrate=settings.esp32_baudrate,
            timeout=SERIAL_READ_TIMEOUT,
            dsrdtr=False,
            rtscts=False,
        )
        _ser.dtr = False  # deassert DTR — prevents ESP32 reset on port open
        # Wait for ESP32 to finish booting. 3 s covers a cold-start.
        time.sleep(3.0)
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


def _send(command: str, timeout_s: float = COMMAND_TIMEOUT) -> dict:
    """
    Send a command to ESP32 and wait for OK/ERR response.
    Runs in a thread (called via run_in_executor).
    Returns parsed JSON payload or raises RuntimeError on error or timeout.
    """
    if _ser is None or not _ser.is_open:
        print(f"[gantry:stub] {command}")
        return {"ok": True, "stub": True}

    with _lock:
        _ser.reset_input_buffer()
        _ser.write((command.strip() + "\n").encode())
        print(f"[gantry] → {command}")

        deadline = time.monotonic() + timeout_s

        while True:
            # Deadline check — raises instead of looping forever
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"timeout ({timeout_s:.0f}s) waiting for response to: {command!r}"
                )

            raw = _ser.readline().decode(errors="replace").strip()

            if not raw:
                # readline() timed out (SERIAL_READ_TIMEOUT) — loop and check deadline
                continue

            print(f"[gantry] ← {raw}")

            if raw.startswith("OK "):
                try:
                    return json.loads(raw[3:])
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"bad JSON in OK response: {raw!r}") from e

            if raw.startswith("ERR "):
                raise RuntimeError(f"ESP32 error: {raw[4:]}")

            # Any other line (e.g. "[CMD] EN on=1" debug echo) — skip and keep waiting


def _send_and_wait_done(command: str, timeout_s: float = MOVE_TIMEOUT) -> dict:
    """
    Send a MOVE command and wait for both OK (accepted) and DONE (finished).
    Runs in a thread (called via run_in_executor).
    """
    if _ser is None or not _ser.is_open:
        print(f"[gantry:stub] {command}")
        time.sleep(settings.gantry_move_delay)
        return {"ok": True, "stub": True}

    with _lock:
        _ser.reset_input_buffer()
        _ser.write((command.strip() + "\n").encode())
        print(f"[gantry] → {command}")

        deadline = time.monotonic() + timeout_s
        got_ok = False

        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"timeout ({timeout_s:.0f}s) waiting for DONE after: {command!r}"
                )

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
                try:
                    return json.loads(raw[5:])
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"bad JSON in DONE response: {raw!r}") from e


# ─── Public async API ─────────────────────────────────────────────────────────


def get_state() -> dict:
    return dict(_state)


async def _run(fn, *args):
    """Run a blocking serial call in a thread pool so asyncio stays free."""
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+
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
    Uses MOVE_TIMEOUT because full homing can take several minutes.
    """
    _state["busy"] = True
    print(f"[gantry] homing axis={axis}")
    try:
        await _run(_send, f"HOME axis={axis}", MOVE_TIMEOUT)
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
        await _run(_send_and_wait_done, f"MOVE x={x} y={y} z={z} speed={speed}")
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
    x_mm = col * 750.0  # 6000mm / 8 cols = 750mm spacing
    y_mm = row * 1000.0  # 2000mm / 2 rows = 1000mm spacing
    z_mm = 50.0  # working height above plant canopy

    # Raise Z to safe height first, then travel XY, then lower
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
    _state.update(
        {"x": result.get("x", 0), "y": result.get("y", 0), "z": result.get("z", 0)}
    )
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


async def enable_motors() -> dict:
    """
    Enable all stepper drivers (DRV8825 EN pin LOW).
    Call at session start — before HOME or any MOVE.

    FIX: Ping first with retries to confirm ESP32 is responsive before
    sending EN. If the ESP32 just booted (e.g. after a reconnect) it may
    still be printing boot messages. Retry up to 5 times with 1 s gaps.
    """
    for attempt in range(1, 6):
        alive = await ping()
        if alive:
            break
        print(f"[gantry] PING attempt {attempt}/5 failed — waiting 1 s")
        await asyncio.sleep(1.0)
    else:
        raise RuntimeError(
            "ESP32 is not responding to PING — check USB connection and firmware"
        )

    result = await _run(_send, "EN on=1")
    print("[gantry] stepper drivers ENABLED")
    return result


async def disable_motors() -> dict:
    """
    Disable all stepper drivers (DRV8825 EN pin HIGH).
    Call at session end / stop / error — motors go unpowered and free-spin.
    """
    result = await _run(_send, "EN on=0")
    print("[gantry] stepper drivers DISABLED")
    return result


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
