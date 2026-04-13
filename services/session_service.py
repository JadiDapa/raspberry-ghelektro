"""
Session service — orchestrates the full scan loop.
Persists every step to the database, emits SSE events,
and logs every action to logs/session_<id>.log via SessionLogger.
"""

import asyncio
import os
from datetime import datetime, timezone

from config import settings, PLANT_GRID
from services import event_bus, hardware
from services import gantry as gantry_service
from services.session_logger import SessionLogger
from services.soil_service import col_to_sensor
from db.database import AsyncSessionLocal
from db import crud


async def run_session(session_id: str) -> None:
    log = SessionLogger(session_id)
    os.makedirs(os.path.join(settings.images_dir, session_id), exist_ok=True)
    plant_results = []

    async with AsyncSessionLocal() as db:
        try:
            # ── Session start ──────────────────────────────────────────────
            await crud.set_session_running(db, session_id)
            log.log_session_start(total_plants=len(PLANT_GRID))
            log.info(
                f"image dir created  path=static/images/{session_id}", tag="SESSION"
            )
            log.info(f"log file           path={log.path}", tag="SESSION")

            await event_bus.emit(
                session_id,
                {
                    "type": "session_started",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_plants": len(PLANT_GRID),
                },
            )
            await asyncio.sleep(0.1)

            # ── Hardware init ──────────────────────────────────────────────
            log.step("MOTORS", "enabling stepper drivers")
            await gantry_service.enable_motors()
            log.log_motors_enabled()

            # ── Limit switch sanity check ──────────────────────────────────
            # Read limit switches before homing. If any are already triggered
            # (LOW = active) the gantry is sitting on a switch — homing will
            # fail immediately. Log the states so the user can diagnose wiring.
            log.step("LIMITS", "reading limit switch states before homing")
            limits = await gantry_service.get_limits()
            log.info(f"limit switch states: {limits}", tag="LIMITS")
            already_triggered = {k: v for k, v in limits.items() if v == 0}
            if already_triggered:
                log.warn(
                    "LIMITS",
                    f"switches already LOW at start: {already_triggered} — gantry may be at home already or switch is wired incorrectly",
                )

            # ── Homing ────────────────────────────────────────────────────
            log.log_homing_start()
            position = await gantry_service.home()
            log.log_homing_done(position)
            await event_bus.emit(
                session_id,
                {
                    "type": "motors_homed",
                    "session_id": session_id,
                    "position": {
                        "x": position["x"],
                        "y": position["y"],
                        "z": position["z"],
                    },
                },
            )

            # FIX: Turn pump on AFTER homing — it was before, which meant
            # the pump ran during the entire homing sequence unnecessarily.
            log.step("PUMP", "turning water pump ON")
            await gantry_service.set_relay("dc", on=True)
            log.log_pump_on()

            # ── Plant scan loop ───────────────────────────────────────────
            for plant_id, (row, col) in enumerate(PLANT_GRID, start=1):
                log.log_plant_start(plant_id, len(PLANT_GRID), row, col)
                await crud.create_plant_scan(db, session_id, plant_id, row, col)

                # — Gantry move —
                log.log_gantry_move_start(plant_id, row, col)
                await gantry_service.move_to_plant(row, col)
                state = gantry_service.get_state()
                log.log_gantry_move_done(state["x"], state["y"], state["z"])
                await event_bus.emit(
                    session_id,
                    {
                        "type": "gantry_moved",
                        "session_id": session_id,
                        "plant_id": plant_id,
                        "row": row,
                        "col": col,
                        "x": state["x"],
                        "y": state["y"],
                        "z": state["z"],
                    },
                )

                # — Camera capture —
                log.log_camera_capture_start(plant_id)
                image_path = await hardware.capture_image(plant_id, session_id)
                image_url = f"/static/images/{session_id}/plant_{plant_id:02d}.jpg"
                log.log_camera_capture_done(image_path)

                # — YOLO inference —
                log.log_yolo_start(image_path)
                detections = await hardware.run_yolo(image_path)
                total_fruits = sum(d["count"] for d in detections)
                log.log_yolo_done(detections, total_fruits)
                await crud.update_plant_scan_vision(
                    db, session_id, plant_id, image_url, detections
                )
                await event_bus.emit(
                    session_id,
                    {
                        "type": "plant_scanned",
                        "session_id": session_id,
                        "plant_id": plant_id,
                        "image_url": image_url,
                        "detections": detections,
                        "total_fruits": total_fruits,
                    },
                )

                # — TOF height sensor —
                log.log_tof_start()
                height_cm = await hardware.read_tof_distance()
                log.log_tof_done(height_cm)

                # — Soil moisture sensor —
                sensor_idx = col_to_sensor(col)
                log.log_moisture_start(col, sensor_idx)
                moisture_pct = await hardware.read_soil_moisture(col)
                log.log_moisture_done(moisture_pct, col)
                await event_bus.emit(
                    session_id,
                    {
                        "type": "sensor_read",
                        "session_id": session_id,
                        "plant_id": plant_id,
                        "height_cm": height_cm,
                        "moisture_pct": moisture_pct,
                    },
                )

                # — Watering decision —
                valve_duration, reason = hardware.compute_watering_duration(
                    moisture_pct
                )
                log.log_watering_decision(moisture_pct, valve_duration, reason)
                if valve_duration > 0:
                    await hardware.open_valve(valve_duration)
                    log.log_valve_done(valve_duration)
                await crud.update_plant_scan_sensors(
                    db,
                    session_id,
                    plant_id,
                    height_cm,
                    moisture_pct,
                    valve_duration,
                    reason,
                )
                await event_bus.emit(
                    session_id,
                    {
                        "type": "plant_watered",
                        "session_id": session_id,
                        "plant_id": plant_id,
                        "valve_duration_sec": valve_duration,
                        "reason": reason,
                    },
                )

                plant_results.append(
                    {
                        "plant_id": plant_id,
                        "row": row,
                        "col": col,
                        "detections": detections,
                        "height_cm": height_cm,
                        "moisture_pct": moisture_pct,
                        "valve_duration_sec": valve_duration,
                    }
                )
                log.log_plant_done(plant_id)

            # ── Session complete ───────────────────────────────────────────
            summary = _build_summary(plant_results)
            log.log_summary(summary)
            await crud.complete_session(db, session_id, summary)
            await event_bus.emit(
                session_id,
                {
                    "type": "session_complete",
                    "session_id": session_id,
                    "summary": summary,
                },
            )

            log.step("PUMP", "turning water pump OFF")
            await gantry_service.set_relay("dc", on=False)
            log.log_pump_off(reason="session complete")

            log.step("MOTORS", "disabling stepper drivers")
            await gantry_service.disable_motors()
            log.log_motors_disabled()

            log.log_session_complete()

        except asyncio.CancelledError:
            log.warn("SESSION", "CancelledError received — stopping cleanly")

            try:
                await gantry_service.set_relay("dc", on=False)
                log.log_pump_off(reason="cancelled")
            except Exception as cleanup_err:
                log.warn("SESSION", f"cleanup: could not turn off pump — {cleanup_err}")

            try:
                await gantry_service.disable_motors()
                log.log_motors_disabled()
            except Exception as cleanup_err:
                log.warn(
                    "SESSION", f"cleanup: could not disable motors — {cleanup_err}"
                )

            await crud.set_session_stopped(db, session_id)
            await event_bus.emit(
                session_id,
                {
                    "type": "session_error",
                    "session_id": session_id,
                    "message": "cancelled",
                },
            )
            log.log_session_stopped()

        except Exception as e:
            log.error("SESSION", f"unhandled exception: {e}")

            # BUG FIX: wrap every cleanup call in its own try/except.
            # If the primary failure was a serial timeout (e.g. ESP32 not responding),
            # these calls would ALSO time out and raise a second unhandled exception
            # that showed up as "Task exception was never retrieved" in the logs.
            try:
                await gantry_service.set_relay("dc", on=False)
                log.log_pump_off(reason="error")
            except Exception as cleanup_err:
                log.warn("SESSION", f"cleanup: could not turn off pump — {cleanup_err}")

            try:
                await gantry_service.disable_motors()
                log.log_motors_disabled()
            except Exception as cleanup_err:
                log.warn(
                    "SESSION", f"cleanup: could not disable motors — {cleanup_err}"
                )

            await crud.set_session_error(db, session_id, str(e))
            await event_bus.emit(
                session_id,
                {
                    "type": "session_error",
                    "session_id": session_id,
                    "message": str(e),
                },
            )
            log.log_session_error(str(e))

        finally:
            await asyncio.sleep(2)
            event_bus.destroy(session_id)
            log.close()


def _build_summary(plant_results: list[dict]) -> dict:
    if not plant_results:
        return {}

    heights = [p["height_cm"] for p in plant_results]
    moisture = [p["moisture_pct"] for p in plant_results]

    ripeness = {"ripe": 0, "turning": 0, "unripe": 0, "broken": 0}
    for p in plant_results:
        for d in p["detections"]:
            if d["cls"] in ripeness:
                ripeness[d["cls"]] += d["count"]

    harvest_ready = [
        {
            "plant_id": p["plant_id"],
            "row": p["row"],
            "col": p["col"],
            "ripe_count": sum(
                d["count"] for d in p["detections"] if d["cls"] == "ripe"
            ),
        }
        for p in plant_results
        if sum(d["count"] for d in p["detections"] if d["cls"] == "ripe") > 5
    ]

    return {
        "total_plants": len(plant_results),
        "avg_height_cm": round(sum(heights) / len(heights), 1),
        "avg_moisture_pct": round(sum(moisture) / len(moisture), 1),
        "ripeness_distribution": ripeness,
        "harvest_ready": harvest_ready,
        "total_water_sec": round(
            sum(p["valve_duration_sec"] for p in plant_results), 1
        ),
    }
