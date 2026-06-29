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
import time
import serial

from config import settings

# ─── Serial connection ────────────────────────────────────────────────────────

_ser: serial.Serial | None = None
_lock = threading.Lock()

# On a cold power-up ESP32 #2 may still be booting when this service starts.
# connect() retries open + PING for up to CONNECT_TIMEOUT before falling back to
# stub mode, so a power cycle doesn't require a manual `sudo reboot`.
CONNECT_TIMEOUT = 30.0  # total time to keep retrying the connection on startup (s)
CONNECT_RETRY_INTERVAL = 2.0  # wait between connection attempts (s)

# Sensor zone mapping — which sensor index covers which columns
# cols 0–2 → sensor 0, cols 3–5 → sensor 1, cols 6–7 → sensor 2
_COL_TO_SENSOR = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
}


def _open_port() -> serial.Serial:
    """Open the UART port and return it (raises on failure)."""
    return serial.Serial(
        port=settings.soil_uart_port,
        baudrate=settings.soil_uart_baudrate,
        timeout=5.0,
    )


def connect():
    """
    Open UART to ESP32 #2 and verify it answers a PING. Called once on startup.

    ttyAMA0 is a hardware UART so the node always exists, but on a cold power-up the
    ESP32 may still be booting when we open it. We retry open + PING for up to
    CONNECT_TIMEOUT — an open port alone isn't proof the firmware is up — and only
    then fall back to stub mode. Avoids needing a `sudo reboot` after a power cycle.
    """
    global _ser
    deadline = time.monotonic() + CONNECT_TIMEOUT
    attempt = 0
    while True:
        attempt += 1
        try:
            _ser = _open_port()
            time.sleep(2.0)  # let the ESP32 finish booting
            _ser.reset_input_buffer()
            # An open port isn't proof the firmware is up — confirm with a PING.
            _send_once("PING")
            print(
                f"[soil] connected → {settings.soil_uart_port} "
                f"@ {settings.soil_uart_baudrate} (attempt {attempt})"
            )
            return
        except Exception as e:
            try:
                if _ser is not None and _ser.is_open:
                    _ser.close()
            except Exception:
                pass
            _ser = None
            if time.monotonic() >= deadline:
                print(
                    f"[soil] WARNING: could not connect to ESP32 #2 after "
                    f"{attempt} attempts ({CONNECT_TIMEOUT:.0f}s) — {e}"
                )
                print("[soil] running in stub mode")
                return
            print(
                f"[soil] connect attempt {attempt} failed ({e}) — "
                f"retrying in {CONNECT_RETRY_INTERVAL:.0f}s..."
            )
            time.sleep(CONNECT_RETRY_INTERVAL)


def disconnect():
    """Close UART connection. Called on shutdown."""
    global _ser
    if _ser and _ser.is_open:
        _ser.close()
    print("[soil] disconnected")


# ─── Low-level send/receive ───────────────────────────────────────────────────


def _reconnect() -> bool:
    """Reopen the UART after a disconnect. Returns True if open again."""
    global _ser
    try:
        if _ser is not None and _ser.is_open:
            _ser.close()
    except Exception:
        pass
    _ser = None
    connect()
    return _ser is not None and _ser.is_open


def _send(command: str, timeout_s: float = 10.0) -> dict:
    """
    Send a command and wait for OK/ERR. On a UART disconnect, attempt one
    reconnect and retry; a failed reconnect raises (never silently stubs).
    """
    try:
        return _send_once(command, timeout_s)
    except serial.SerialException as e:
        print(f"[soil] UART disconnect on {command!r} ({e}) — attempting reconnect")
        if _reconnect():
            print("[soil] reconnected — retrying command")
            return _send_once(command, timeout_s)
        raise RuntimeError(f"UART lost and reconnect failed: {e}")


def _send_once(command: str, timeout_s: float = 10.0) -> dict:
    """Send a command and wait for OK/ERR. Runs in a thread."""
    if _ser is None or not _ser.is_open:
        print(f"[soil:stub] {command}")
        return {"stub": True, "pct": 0.0}  # lowest moisture → max valve duration

    with _lock:
        _ser.reset_input_buffer()
        _ser.write((command.strip() + "\n").encode())
        print(f"[soil] → {command}")

        deadline = time.monotonic() + timeout_s
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"timeout ({timeout_s:.0f}s) waiting for response to soil command: {command!r}"
                )
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
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+
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
