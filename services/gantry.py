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
import os
import threading
import time
import serial
from config import settings
from models.scan_config import ScanConfig

# ─── Timeouts ─────────────────────────────────────────────────────────────────

SERIAL_READ_TIMEOUT = 1.0  # per readline() call (s) — short so deadline loop ticks fast
COMMAND_TIMEOUT = 10.0  # max wait for OK/ERR on quick commands (s)
MOVE_TIMEOUT = 300.0  # max wait for DONE on MOVE/HOME (s) — covers a full 6 m traverse

# On a COLD power-up the Pi boots before the CH340/CP2102 USB chip has enumerated,
# so /dev/ttyUSB0 may not exist yet (and the ESP32 firmware may still be booting).
# connect() retries open + PING for up to CONNECT_TIMEOUT before giving up, so the
# service comes up healthy without a manual `sudo reboot`.
CONNECT_TIMEOUT = 30.0  # total time to keep retrying the connection on startup (s)
CONNECT_RETRY_INTERVAL = 2.0  # wait between connection attempts (s)

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


def _open_port() -> serial.Serial:
    """
    Open the serial port and return it (raises on failure).

    Uses the deferred-open pattern to prevent a DTR pulse on port open. On Linux,
    opening a USB serial port briefly asserts DTR even if you immediately set
    dtr=False afterward — the pulse is enough to reset the ESP32. By constructing
    Serial with port=None first, setting dtr=False before open(), then calling
    open(), we guarantee DTR is never asserted.
    """
    ser = serial.Serial()
    ser.port = settings.esp32_port
    ser.baudrate = settings.esp32_baudrate
    ser.timeout = SERIAL_READ_TIMEOUT
    ser.dsrdtr = False
    ser.rtscts = False
    ser.dtr = False  # LOW = no reset pulse on open (active-low reset on ESP32)
    ser.rts = False  # FIX: RTS also resets ESP32 on many CH340/CP2102 boards.
    #                       Must be False BEFORE open(), same reason as dtr=False.
    ser.open()
    return ser


def connect():
    """
    Open serial port to ESP32 #1 and verify it answers a PING. Call once on startup.

    On a cold power-up the USB-serial chip may not have enumerated yet (so the port
    node doesn't exist) and the ESP32 may still be booting. We retry open + PING for
    up to CONNECT_TIMEOUT instead of giving up on the first failure — an open port
    alone isn't "ready", so we confirm the firmware actually responds. Only after the
    window elapses do we fall back to stub mode. This removes the need to `sudo reboot`
    the Pi after a power cycle just to get the ESP32 recognized.
    """
    global _ser
    deadline = time.monotonic() + CONNECT_TIMEOUT
    attempt = 0
    while True:
        attempt += 1
        try:
            if not os.path.exists(settings.esp32_port):
                # Port node not present yet — USB hasn't enumerated after a cold boot.
                raise serial.SerialException(f"{settings.esp32_port} does not exist yet")
            _ser = _open_port()
            # Give the ESP32 time to finish booting.
            # 5 s covers: CH340 enumeration (~0.5 s) + ESP32 ROM boot (~0.5 s) +
            # Arduino setup() including VL53L1X init (~1-2 s) + margin.
            time.sleep(5.0)
            _ser.reset_input_buffer()
            # An open port isn't proof the firmware is up — confirm with a PING.
            _send_once("PING")
            print(
                f"[gantry] connected → {settings.esp32_port} "
                f"@ {settings.esp32_baudrate} (attempt {attempt})"
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
                    f"[gantry] WARNING: could not connect to ESP32 after "
                    f"{attempt} attempts ({CONNECT_TIMEOUT:.0f}s) — {e}"
                )
                print("[gantry] running in stub mode")
                return
            print(
                f"[gantry] connect attempt {attempt} failed ({e}) — "
                f"retrying in {CONNECT_RETRY_INTERVAL:.0f}s..."
            )
            time.sleep(CONNECT_RETRY_INTERVAL)


def disconnect():
    """Close serial port. Call on shutdown."""
    global _ser
    if _ser and _ser.is_open:
        _ser.close()
    print("[gantry] disconnected")


# ─── Low-level send/receive ───────────────────────────────────────────────────


def _reconnect() -> bool:
    """
    Reopen the serial port after a disconnect (e.g. USB unplugged mid-session).
    Returns True if the port is open again. Runs inside the executor thread.
    """
    global _ser
    try:
        if _ser is not None and _ser.is_open:
            _ser.close()
    except Exception:
        pass
    _ser = None
    connect()  # deferred-open + boot wait; sets _ser to None on failure
    return _ser is not None and _ser.is_open


def _send(command: str, timeout_s: float = COMMAND_TIMEOUT) -> dict:
    """
    Send a command and wait for OK/ERR. On a serial disconnect, attempt one
    reconnect and retry. A failed reconnect RAISES (never silently degrades to
    stub) so the session loop can stop the machine instead of trusting fake data.
    """
    try:
        return _send_once(command, timeout_s)
    except serial.SerialException as e:
        print(f"[gantry] serial disconnect on {command!r} ({e}) — attempting reconnect")
        if _reconnect():
            print("[gantry] reconnected — retrying command")
            return _send_once(command, timeout_s)
        raise RuntimeError(f"serial port lost and reconnect failed: {e}")


def _send_once(command: str, timeout_s: float = COMMAND_TIMEOUT) -> dict:
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
    Send a MOVE/HOME and wait for OK + DONE. On a serial disconnect mid-move,
    attempt one reconnect and retry; a failed reconnect RAISES so the session
    aborts and the gantry is safed rather than continuing blind.
    """
    try:
        return _send_and_wait_done_once(command, timeout_s)
    except serial.SerialException as e:
        print(f"[gantry] serial disconnect during move {command!r} ({e}) — attempting reconnect")
        if _reconnect():
            print("[gantry] reconnected — retrying move")
            return _send_and_wait_done_once(command, timeout_s)
        raise RuntimeError(f"serial port lost and reconnect failed: {e}")


def _send_and_wait_done_once(command: str, timeout_s: float = MOVE_TIMEOUT) -> dict:
    """
    Send a MOVE command and wait for both OK (accepted) and DONE (finished).
    Runs in a thread (called via run_in_executor).
    """
    if _ser is None or not _ser.is_open:
        print(f"[gantry:stub] {command}")
        time.sleep(settings.stub_gantry_delay if settings.stub_mode else settings.gantry_move_delay)
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
                # FIX: ESP32 sends OK with note="already at target" when no motion
                # is needed — it will never send DONE in that case. Treat it as done.
                try:
                    payload = json.loads(raw[3:])
                    if payload.get("note") == "already at target":
                        return payload
                except json.JSONDecodeError:
                    pass
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
    Retries once on failure — limit switches can occasionally miss on first pass.
    """
    _state["busy"] = True
    print(f"[gantry] homing axis={axis}")
    last_err = None
    for attempt in range(1, 3):  # try twice
        try:
            result = await _run(_send, f"HOME axis={axis}", MOVE_TIMEOUT)
            _state.update({"x": 0.0, "y": 0.0, "z": 0.0, "busy": False, "homed": True})
            return get_state()
        except Exception as e:
            last_err = e
            print(f"[gantry] homing attempt {attempt}/2 failed: {e}")
            if attempt < 2:
                print("[gantry] retrying homing in 2 s...")
                await asyncio.sleep(2.0)
    _state["busy"] = False
    raise RuntimeError(
        f"Homing failed after 2 attempts. Last error: {last_err}\n"
        "Check: (1) limit switch wiring on failing axis, "
        "(2) 'LIMITS' command shows 0 when switch pressed manually, "
        "(3) motor direction moves toward the switch."
    )


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


# Stub height used when the TOF is unavailable, mirroring hardware.read_tof_distance()
# so a run without a connected sensor still produces a plausible sweep for testing.
_STUB_TOF_CM = 100.0


def _synthetic_tof_samples(
    start_x: float, x_target: float, speed: int, sample_interval_s: float, *, pace: bool
) -> list[dict]:
    """Linear set of stub {x, tof_cm} samples along the X path.

    Used when there is no ESP32 at all (`pace=True` — sleep to simulate the
    real sweep duration) or when the ESP32 is present but the TOF sensor is
    missing (`pace=False` — the move already happened in real time).
    """
    span = abs(x_target - start_x)
    duration = span / speed if speed else 0.0
    n = max(1, int(duration / sample_interval_s)) if sample_interval_s > 0 else 1
    out: list[dict] = []
    for i in range(n + 1):
        frac = i / n if n else 1.0
        xi = start_x + (x_target - start_x) * frac
        out.append({"x": round(xi, 1), "tof_cm": _STUB_TOF_CM})
        if pace and not settings.stub_mode:
            time.sleep(sample_interval_s)
    return out


async def sweep_tof(
    x_target: float,
    y: float,
    z: float,
    speed: int = 150,
    sample_interval_s: float = 0.2,
) -> list[dict]:
    """
    Continuous height sweep: fire ONE MOVE to (x_target, y, z) and, while the
    gantry travels, poll the TOF sensor roughly every `sample_interval_s` s
    without ever stopping. Returns a list of ``{"x": mm, "tof_cm": cm}`` samples
    (out-of-range/no-data readings are dropped).

    Unlike move_to(), this holds the serial lock for the whole move and drives
    the MOVE↔TOF protocol itself, because a normal _send would reset the input
    buffer between commands and discard the pending DONE. The ESP32 loop() polls
    serial every iteration and MOVE is async, so it answers TOF mid-move.
    """
    _state["busy"] = True
    print(f"[gantry] sweep_tof → X={x_target} Y={y} Z={z} speed={speed} interval={sample_interval_s}s")
    try:
        samples = await _run(_sweep_tof_once, x_target, y, z, speed, sample_interval_s)
        _state.update({"x": x_target, "y": y, "z": z, "busy": False})
        return samples
    except Exception as e:
        _state["busy"] = False
        raise RuntimeError(f"TOF sweep failed: {e}")


def _sweep_tof_once(
    x_target: float, y: float, z: float, speed: int, sample_interval_s: float
) -> list[dict]:
    """
    Blocking body of sweep_tof — runs in the serial executor thread.

    Protocol: write MOVE, wait for the accept OK, then loop: emit a TOF request
    every `sample_interval_s` (only one outstanding at a time so it self-throttles
    if the sensor is slower), read responses, and stop when DONE arrives.

    Error handling mirrors the machine's fatality model: a motion fault
    (`ERR limit_triggered`) raises so the session aborts and safes the gantry,
    but a TOF fault (`ERR TOF sensor not ready` — sensor unplugged) is NOT fatal.
    Those samples are dropped, and if the sensor is missing for the whole sweep
    the result falls back to stub heights so a run without a TOF still completes.
    """
    cmd = f"MOVE x={x_target} y={y} z={z} speed={speed}"
    start_x = _state["x"]

    if _ser is None or not _ser.is_open:
        # No ESP32 at all — full stub: simulate the sweep in real time.
        print(f"[gantry:stub] sweep {cmd}")
        return _synthetic_tof_samples(start_x, x_target, speed, sample_interval_s, pace=True)

    samples = []
    tof_absent = False  # True once the ESP32 reports the TOF sensor is unavailable
    with _lock:
        _ser.reset_input_buffer()
        prev_timeout = _ser.timeout
        # Short readline timeout so the loop ticks fast enough to pace TOF
        # requests and notice DONE promptly (instead of blocking up to 1 s).
        _ser.timeout = min(0.05, sample_interval_s)
        try:
            _ser.write((cmd + "\n").encode())
            print(f"[gantry] → {cmd}")

            deadline = time.monotonic() + MOVE_TIMEOUT
            done = False
            move_started = False

            # 1. Wait for the MOVE to be accepted (OK), or finish immediately.
            while not move_started and not done:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"timeout waiting for MOVE ack: {cmd!r}")
                raw = _ser.readline().decode(errors="replace").strip()
                if not raw:
                    continue
                print(f"[gantry] ← {raw}")
                if raw.startswith("ERR "):
                    raise RuntimeError(f"ESP32 error: {raw[4:]}")
                if raw.startswith("DONE "):
                    done = True  # zero-length move — nothing to sample
                elif raw.startswith("OK "):
                    try:
                        payload = json.loads(raw[3:])
                    except json.JSONDecodeError:
                        continue
                    if payload.get("note") == "already at target":
                        done = True
                    else:
                        move_started = True
                # any other line is a debug echo — keep waiting

            # 2. Sample TOF while the gantry travels, until DONE.
            next_sample = time.monotonic()
            awaiting_tof = False
            while not done:
                now = time.monotonic()
                if now > deadline:
                    raise RuntimeError(f"timeout ({MOVE_TIMEOUT:.0f}s) waiting for DONE: {cmd!r}")

                if not awaiting_tof and now >= next_sample:
                    _ser.write(b"TOF\n")
                    awaiting_tof = True
                    next_sample += sample_interval_s
                    if next_sample < now:  # fell behind — don't spiral
                        next_sample = now + sample_interval_s

                raw = _ser.readline().decode(errors="replace").strip()
                if not raw:
                    continue
                if raw.startswith("ERR "):
                    msg = raw[4:]
                    if "limit" in msg:
                        # Motion fault — fatal, abort so the gantry is safed.
                        raise RuntimeError(f"ESP32 error: {msg}")
                    # TOF fault (e.g. sensor not ready/unplugged) — non-fatal:
                    # drop this sample and keep sweeping. Stubbed after the move.
                    tof_absent = True
                    awaiting_tof = False
                    continue
                if raw.startswith("DONE "):
                    print(f"[gantry] ← {raw}")
                    done = True
                    break
                if raw.startswith("OK "):
                    awaiting_tof = False
                    try:
                        payload = json.loads(raw[3:])
                    except json.JSONDecodeError:
                        continue
                    mm = payload.get("mm")
                    if mm is not None:
                        xi = payload.get("x", _state["x"])
                        samples.append({"x": float(xi), "tof_cm": round(mm / 10.0, 1)})
                # else: debug echo — ignore

            # Sensor was unavailable for the whole sweep (unplugged / not ready):
            # fall back to stub heights so the watering flow still completes.
            if tof_absent and not samples:
                print("[gantry] TOF sensor unavailable — using stub heights for sweep")
                return _synthetic_tof_samples(start_x, x_target, speed, sample_interval_s, pace=False)

            return samples
        finally:
            _ser.timeout = prev_timeout


async def move_to_plant(row: int, col: int) -> dict:
    """Move above a plant grid position using hardcoded defaults."""
    return await move_to_plant_with_config(row, col, ScanConfig())


async def move_to_plant_with_config(row: int, col: int, config) -> dict:
    """
    Move above a plant grid position using values from a ScanConfig.

    Z axis convention (matches ESP32 firmware):
      Z=0   → home / top of travel (limit switch end)
      Z>0   → mm below home (working depth)
    """
    x_mm, y_mm = config.plant_position_mm(row, col)
    z_working = config.offset.z_mm
    z_clear = 10.0

    # Raise to clearance first to avoid clipping plants while travelling XY
    if _state["z"] > z_clear:
        await move_to(_state["x"], _state["y"], z_clear)

    await move_to(x_mm, y_mm, z_working)
    return get_state()


async def emergency_stop() -> dict:
    """
    Immediately halt all motors, de-energize the stepper drivers, AND close
    every relay (solenoid valve + water pump).

    `STOP` halts motion but leaves the DRV8825 EN pins enabled (motors stay
    powered/held). We follow it with `EN on=0` so the drivers are disabled at
    click time — the motors go slack and no longer hold torque. Finally we cut
    both relays so nothing keeps watering. Every step is best-effort and
    independent so an emergency stop always does as much as it can even if the
    ESP32 is mid-command.
    """
    _state["busy"] = False
    print("[gantry] EMERGENCY STOP")
    try:
        await _run(_send, "STOP")
    except Exception:
        pass  # stop is best-effort
    try:
        await _run(_send, "EN on=0")
        print("[gantry] stepper drivers DISABLED (emergency stop)")
    except Exception:
        pass  # EN cut is best-effort
    for channel in ("sol", "dc"):
        try:
            await _run(_send, f"RELAY ch={channel} on=0")
            print(f"[gantry] relay '{channel}' OFF (emergency stop)")
        except Exception:
            pass  # each relay cut is best-effort
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
    Sends PING first to confirm the ESP32 is alive.

    Retry policy: 3 attempts × 5 s apart.
    Rationale: the ESP32 takes up to ~5 s to finish booting (ROM + setup() +
    VL53L1X init). If the server starts at the same moment the ESP32 is
    powered, connect()'s 5 s sleep may barely cover it. Retrying here makes
    session start resilient without increasing the startup delay further.
    """
    for attempt in range(1, 4):  # 3 attempts
        alive = await ping()
        if alive:
            break
        if attempt < 3:
            print(
                f"[gantry] PING attempt {attempt}/3 failed — "
                f"ESP32 may still be booting, retrying in 5 s..."
            )
            await asyncio.sleep(5.0)
    else:
        raise RuntimeError(
            "ESP32 not responding to PING after 3 attempts — "
            "check USB cable and firmware. "
            "Try: (1) replug USB, (2) reflash firmware, "
            "(3) verify /dev/ttyUSB0 with: ls -l /dev/ttyUSB*"
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


# ─── Servo control ────────────────────────────────────────────────────────────

_servo_state = {"pan": 90, "tilt": 90}


def get_servo_state() -> dict:
    return dict(_servo_state)


async def get_servo_angles() -> dict:
    """Return last-known servo angles (no serial round-trip needed)."""
    return get_servo_state()


async def set_servo_angles(pan: float, tilt: float) -> dict:
    """Move servos to the given pan/tilt angles (0–180°)."""
    pan_i  = max(0, min(180, int(round(pan))))
    tilt_i = max(0, min(180, int(round(tilt))))
    result = await _run(_send, f"SERVO pan={pan_i} tilt={tilt_i}")
    _servo_state.update({
        "pan":  result.get("pan",  pan_i),
        "tilt": result.get("tilt", tilt_i),
    })
    return get_servo_state()
