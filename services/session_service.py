"""
Session service — orchestrates the full scan loop.
All data is written to Next.js Postgres in real time via pi_client.
No local SQLite. No db session. Exceptions propagate and stop the session.
"""

import asyncio
from datetime import datetime, timezone

from models.scan_config import ScanConfig
from services import event_bus, hardware
from services import gantry as gantry_service
from services import pi_client
from services.session_logger import SessionLogger
from services.soil_service import col_to_sensor


async def run_session(session_id: int, config: ScanConfig | None = None) -> None:
    if config is None:
        config = ScanConfig()

    log = SessionLogger(str(session_id))
    plant_grid = config.plant_grid()
    plant_results = []

    try:
        # ── Session start ──────────────────────────────────────────────
        await pi_client.patch_status(session_id, "running")
        log.log_session_start(total_plants=len(plant_grid))
        log.info(f"log file  path={log.path}", tag="SESSION")

        await event_bus.emit(
            str(session_id),
            {
                "type": "session_started",
                "session_id": str(session_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_plants": len(plant_grid),
            },
        )
        await asyncio.sleep(0.1)

        # ── Hardware init ──────────────────────────────────────────────
        log.step("MOTORS", "enabling stepper drivers")
        await gantry_service.enable_motors()
        log.log_motors_enabled()

        # ── Limit switch sanity check ──────────────────────────────────
        log.step("LIMITS", "reading limit switch states before homing")
        limits = await gantry_service.get_limits()
        log.info(f"limit switch states: {limits}", tag="LIMITS")
        already_triggered = {k: v for k, v in limits.items() if v == 0}
        if already_triggered:
            log.warn(
                "LIMITS",
                f"switches already LOW at start: {already_triggered}",
            )

        # ── Homing ────────────────────────────────────────────────────
        log.log_homing_start()
        position = await gantry_service.home()
        log.log_homing_done(position)
        await event_bus.emit(
            str(session_id),
            {
                "type": "motors_homed",
                "session_id": str(session_id),
                "position": {
                    "x": position["x"],
                    "y": position["y"],
                    "z": position["z"],
                },
            },
        )

        log.step("PUMP", "turning water pump ON")
        await gantry_service.set_relay("dc", on=True)
        log.log_pump_on()

        # ── Plant scan loop ───────────────────────────────────────────
        for plant_id, (row, col) in enumerate(plant_grid, start=1):
            log.log_plant_start(plant_id, len(plant_grid), row, col)

            # — Gantry move —
            log.log_gantry_move_start(plant_id, row, col)
            await gantry_service.move_to_plant_with_config(row, col, config)
            await gantry_service.set_servo_angles(
                config.offset.servo_pan, config.offset.servo_tilt
            )
            state = gantry_service.get_state()
            log.log_gantry_move_done(state["x"], state["y"], state["z"])
            await event_bus.emit(
                str(session_id),
                {
                    "type": "gantry_moved",
                    "session_id": str(session_id),
                    "plant_id": plant_id,
                    "row": row,
                    "col": col,
                    "x": state["x"],
                    "y": state["y"],
                    "z": state["z"],
                },
            )

            # — Camera capture → upload to Next.js —
            log.log_camera_capture_start(plant_id)
            image_bytes = await hardware.capture_image()
            image_url = await pi_client.upload_image(session_id, plant_id, image_bytes)
            log.log_camera_capture_done(image_url)

            # — YOLO inference —
            log.log_yolo_start(image_url)
            detections = await hardware.run_yolo(image_bytes)
            total_fruits = sum(d["count"] for d in detections)
            log.log_yolo_done(detections, total_fruits)

            # — Write vision result to Next.js —
            await pi_client.post_vision(
                session_id, plant_id, row, col, image_url, detections
            )
            await event_bus.emit(
                str(session_id),
                {
                    "type": "plant_scanned",
                    "session_id": str(session_id),
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
                str(session_id),
                {
                    "type": "sensor_read",
                    "session_id": str(session_id),
                    "plant_id": plant_id,
                    "height_cm": height_cm,
                    "moisture_pct": moisture_pct,
                },
            )

            # — Watering decision —
            valve_duration, reason = hardware.compute_watering_duration(moisture_pct)
            log.log_watering_decision(moisture_pct, valve_duration, reason)
            if valve_duration > 0:
                await hardware.open_valve(valve_duration)
                log.log_valve_done(valve_duration)

            # — Write sensor + watering data to Next.js —
            await pi_client.post_sensors(
                session_id, plant_id, height_cm, moisture_pct, valve_duration, reason
            )
            await event_bus.emit(
                str(session_id),
                {
                    "type": "plant_watered",
                    "session_id": str(session_id),
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
        await pi_client.post_complete(session_id, summary)
        await event_bus.emit(
            str(session_id),
            {
                "type": "session_complete",
                "session_id": str(session_id),
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
        except Exception as e:
            log.warn("SESSION", f"cleanup: could not turn off pump — {e}")
        try:
            await gantry_service.disable_motors()
            log.log_motors_disabled()
        except Exception as e:
            log.warn("SESSION", f"cleanup: could not disable motors — {e}")
        await pi_client.patch_status(session_id, "stopped")
        await event_bus.emit(
            str(session_id),
            {
                "type": "session_error",
                "session_id": str(session_id),
                "message": "cancelled",
            },
        )
        log.log_session_stopped()

    except Exception as e:
        log.error("SESSION", f"unhandled exception: {e}")
        try:
            await gantry_service.set_relay("dc", on=False)
            log.log_pump_off(reason="error")
        except Exception as ce:
            log.warn("SESSION", f"cleanup: could not turn off pump — {ce}")
        try:
            await gantry_service.disable_motors()
            log.log_motors_disabled()
        except Exception as ce:
            log.warn("SESSION", f"cleanup: could not disable motors — {ce}")
        await pi_client.post_error(session_id)
        await event_bus.emit(
            str(session_id),
            {
                "type": "session_error",
                "session_id": str(session_id),
                "message": str(e),
            },
        )
        log.log_session_error(str(e))

    finally:
        await asyncio.sleep(2)
        event_bus.destroy(str(session_id))
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

    harvest_ready_ids = [
        p["plant_id"]
        for p in plant_results
        if sum(d["count"] for d in p["detections"] if d["cls"] == "ripe") > 5
    ]

    return {
        "totalPlants": len(plant_results),
        "avgHeightCm": round(sum(heights) / len(heights), 1),
        "avgMoisturePct": round(sum(moisture) / len(moisture), 1),
        "ripeness": ripeness,
        "harvestReadyIds": harvest_ready_ids,
        "totalWaterSec": round(sum(p["valve_duration_sec"] for p in plant_results), 1),
    }
