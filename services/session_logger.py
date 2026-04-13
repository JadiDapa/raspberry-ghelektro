"""
Session logger — writes structured, timestamped logs to a per-session file.

Each session gets its own file:  logs/session_<id>.log
Every line is also echoed to stdout so uvicorn captures it too.

Usage inside session_service.py:
    log = SessionLogger(session_id)
    log.info("some message")
    log.step("CAMERA", "capturing plant 3")
    log.ok("YOLO", "3 ripe, 1 unripe detected")
    log.warn("SENSOR", "TOF returned no data — using fallback")
    log.error("GANTRY", "move failed: ERR limit_triggered axis=x")
    log.separator()           # thick rule between plants
    log.section("Plant 4/16")  # labelled section header
"""

import os
import sys
from datetime import datetime, timezone


# ─── Log levels ───────────────────────────────────────────────────────────────

_LEVELS = {
    "INFO": " ",
    "STEP": "→",
    "OK": "✓",
    "WARN": "!",
    "ERROR": "✗",
}


class SessionLogger:
    """
    One instance per session. Call close() when done (or use as context manager).
    The log file is created on first write.
    """

    LOG_DIR = "logs"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._start = datetime.now(timezone.utc)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self._path = os.path.join(self.LOG_DIR, f"session_{session_id}.log")
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")  # line-buffered
        self._write_header()

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    @property
    def path(self) -> str:
        return self._path

    # ── Core write method ─────────────────────────────────────────────────────

    def _write(self, level: str, tag: str, msg: str):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms precision
        elapsed = now - self._start
        elapsed_s = elapsed.total_seconds()
        mins, secs = divmod(int(elapsed_s), 60)
        elapsed_str = f"{mins:02d}:{secs:02d}"

        marker = _LEVELS.get(level, " ")
        tag_field = f"[{tag}]" if tag else ""
        line = f"{ts}  +{elapsed_str}  {marker} {level:<5}  {tag_field:<12} {msg}"

        # Write to file
        self._fh.write(line + "\n")

        # Echo to stdout (uvicorn captures this)
        print(line, file=sys.stdout, flush=True)

    def _write_raw(self, raw: str):
        """Write a raw line (headers, separators) without the timestamp prefix."""
        self._fh.write(raw + "\n")
        print(raw, file=sys.stdout, flush=True)

    # ── Header / structure helpers ────────────────────────────────────────────

    def _write_header(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._write_raw("=" * 72)
        self._write_raw(f"  SESSION {self.session_id}   started {now}")
        self._write_raw("=" * 72)

    def separator(self):
        """Thin rule — used between plants."""
        self._write_raw("─" * 72)

    def section(self, title: str):
        """Labelled section header — used for each plant."""
        self._write_raw(f"\n{'─' * 72}")
        self._write_raw(f"  {title}")
        self._write_raw("─" * 72)

    def footer(self, status: str):
        """Closing rule with final status."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        elapsed = datetime.now(timezone.utc) - self._start
        mins, secs = divmod(int(elapsed.total_seconds()), 60)
        self._write_raw("=" * 72)
        self._write_raw(
            f"  SESSION {self.session_id}   {status.upper()}"
            f"   ended {now}   duration {mins:02d}:{secs:02d}"
        )
        self._write_raw("=" * 72 + "\n")

    # ── Log level helpers ─────────────────────────────────────────────────────

    def info(self, msg: str, tag: str = ""):
        self._write("INFO", tag, msg)

    def step(self, tag: str, msg: str):
        """Starting an action (before it completes)."""
        self._write("STEP", tag, msg)

    def ok(self, tag: str, msg: str):
        """Action completed successfully."""
        self._write("OK", tag, msg)

    def warn(self, tag: str, msg: str):
        """Non-fatal issue — execution continues."""
        self._write("WARN", tag, msg)

    def error(self, tag: str, msg: str):
        """Fatal or significant error."""
        self._write("ERROR", tag, msg)

    # ── Domain-specific helpers (called from session_service) ─────────────────

    def log_session_start(self, total_plants: int):
        self.info(f"session initialised  total_plants={total_plants}", tag="SESSION")

    def log_motors_enabled(self):
        self.ok("MOTORS", "stepper drivers enabled (EN=LOW)")

    def log_motors_disabled(self):
        self.ok("MOTORS", "stepper drivers disabled (EN=HIGH)")

    def log_pump_on(self):
        self.ok("PUMP", "water pump relay ON (dc)")

    def log_pump_off(self, reason: str = ""):
        tag_msg = f"water pump relay OFF (dc){f'  reason={reason}' if reason else ''}"
        self.ok("PUMP", tag_msg)

    def log_homing_start(self):
        self.step("GANTRY", "homing all axes  (Z → Y → X)")

    def log_homing_done(self, position: dict):
        self.ok(
            "GANTRY",
            f"homed  x={position['x']:.1f}mm  y={position['y']:.1f}mm  z={position['z']:.1f}mm",
        )

    def log_plant_start(self, plant_id: int, total: int, row: int, col: int):
        self.section(f"Plant {plant_id}/{total}  (row={row}, col={col})")
        self.info(
            f"starting scan  plant_id={plant_id}  row={row}  col={col}", tag="PLANT"
        )

    def log_gantry_move_start(self, plant_id: int, row: int, col: int):
        self.step("GANTRY", f"moving to plant {plant_id}  row={row}  col={col}")

    def log_gantry_move_done(self, x: float, y: float, z: float):
        self.ok("GANTRY", f"arrived  x={x:.1f}mm  y={y:.1f}mm  z={z:.1f}mm")

    def log_camera_capture_start(self, plant_id: int):
        self.step(
            "CAMERA",
            f"waiting for gantry to stabilise then capturing  plant={plant_id}",
        )

    def log_camera_capture_done(self, image_path: str):
        self.ok("CAMERA", f"snapshot saved  path={image_path}")

    def log_yolo_start(self, image_path: str):
        self.step("YOLO", f"running inference  image={image_path}")

    def log_yolo_done(self, detections: list[dict], total_fruits: int):
        if not detections:
            self.ok("YOLO", f"no fruits detected  total=0")
            return
        breakdown = "  ".join(
            f"{d['cls']}={d['count']}({d['confidence']:.2f})" for d in detections
        )
        self.ok("YOLO", f"total_fruits={total_fruits}  {breakdown}")

    def log_tof_start(self):
        self.step("TOF", "reading distance sensor")

    def log_tof_done(self, height_cm: float, fallback: bool = False):
        note = "  [stub fallback]" if fallback else ""
        self.ok("TOF", f"height={height_cm:.1f}cm{note}")

    def log_moisture_start(self, col: int, sensor_index: int):
        self.step("MOISTURE", f"reading sensor {sensor_index}  (col={col})")

    def log_moisture_done(self, moisture_pct: float, col: int):
        self.ok("MOISTURE", f"moisture={moisture_pct:.1f}%  col={col}")

    def log_watering_decision(self, moisture_pct: float, duration: float, reason: str):
        if duration > 0:
            self.step(
                "VALVE", f"opening solenoid  duration={duration:.1f}s  reason={reason}"
            )
        else:
            self.info(f"watering skipped  {reason}", tag="VALVE")

    def log_valve_done(self, duration: float):
        self.ok("VALVE", f"solenoid closed after {duration:.1f}s")

    def log_plant_done(self, plant_id: int):
        self.info(f"plant {plant_id} scan complete", tag="PLANT")

    def log_summary(self, summary: dict):
        self.separator()
        self.info("session summary", tag="SUMMARY")
        self.info(f"  total_plants     = {summary.get('total_plants')}", tag="SUMMARY")
        self.info(f"  avg_height_cm    = {summary.get('avg_height_cm')}", tag="SUMMARY")
        self.info(
            f"  avg_moisture_pct = {summary.get('avg_moisture_pct')}", tag="SUMMARY"
        )
        self.info(
            f"  total_water_sec  = {summary.get('total_water_sec')}", tag="SUMMARY"
        )
        r = summary.get("ripeness_distribution", {})
        self.info(
            f"  ripeness         = ripe:{r.get('ripe',0)}  turning:{r.get('turning',0)}"
            f"  unripe:{r.get('unripe',0)}  broken:{r.get('broken',0)}",
            tag="SUMMARY",
        )
        hr = summary.get("harvest_ready", [])
        if hr:
            ids = ", ".join(str(p["plant_id"]) for p in hr)
            self.ok("SUMMARY", f"harvest ready  plants=[{ids}]  count={len(hr)}")
        else:
            self.info("harvest ready  none", tag="SUMMARY")

    def log_session_complete(self):
        self.ok("SESSION", "all plants scanned  status=complete")
        self.footer("complete")

    def log_session_stopped(self):
        self.warn("SESSION", "session cancelled by user  status=stopped")
        self.footer("stopped")

    def log_session_error(self, message: str):
        self.error("SESSION", f"unhandled error  status=error  message={message}")
        self.footer("error")
