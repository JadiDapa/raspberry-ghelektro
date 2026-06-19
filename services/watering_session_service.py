"""
Watering session service — orchestrates:
  1. TOF height sweep across all plant positions (at z_max, discrete stops)
  2. Soil moisture read (before)
  3. Fuzzy logic → single global watering duration
  4. Per-column solenoid valve control (8 stops, sprinklers cover both rows)
  5. Soil moisture read (after)
  6. Summary POST to Next.js dashboard

Structure mirrors session_service.py exactly — same try/except/finally cleanup.
"""

import asyncio
from datetime import datetime, timezone

from models.watering_config import WateringConfig
from services import event_bus, hardware
from services import gantry as gantry_service
from services import pi_client
from services.fuzzy_watering import compute_watering_duration
from services.session_logger import SessionLogger


async def run_watering_session(
    session_id: int,
    config: WateringConfig | None = None,
) -> None:
    if config is None:
        config = WateringConfig()

    log = SessionLogger(str(session_id))
    positions = config.plant_positions()   # all (row, col) pairs
    center_y = config.center_y_mm()

    # Per-column minimum TOF reading (cm). Lower = closer = taller plant.
    col_min_tof: dict[int, float] = {}
    watering_stops_count = 0

    try:
        # ── Session start ──────────────────────────────────────────────
        await pi_client.patch_status(session_id, "running")
        log.log_session_start(total_plants=len(positions))
        log.info(f"log file path={log.path}", tag="SESSION")

        await event_bus.emit(
            str(session_id),
            {
                "type": "session_started",
                "session_id": str(session_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_plants": len(positions),
            },
        )
        await asyncio.sleep(0.1)

        # ── Hardware init ──────────────────────────────────────────────
        log.step("MOTORS", "enabling stepper drivers")
        await gantry_service.enable_motors()
        log.log_motors_enabled()

        log.log_homing_start()
        position = await gantry_service.home()
        log.log_homing_done(position)
        await event_bus.emit(
            str(session_id),
            {
                "type": "motors_homed",
                "session_id": str(session_id),
                "position": {"x": position["x"], "y": position["y"], "z": position["z"]},
            },
        )

        # ── Step 1: Raise Z to max for TOF sweep ──────────────────────
        log.step("GANTRY", f"raising Z to z_max={config.z_max_mm}mm for TOF sweep")
        await gantry_service.move_to(0.0, 0.0, config.z_max_mm)

        # ── Step 2: TOF height sweep ───────────────────────────────────
        await event_bus.emit(
            str(session_id),
            {
                "type": "tof_sweep_started",
                "session_id": str(session_id),
                "total_positions": len(positions),
            },
        )

        for pos_idx, (row, col) in enumerate(positions, start=1):
            x_mm = config.col_x_mm(col)
            y_mm = config.row_y_mm(row)

            log.step("TOF", f"moving to row={row} col={col} ({x_mm}mm, {y_mm}mm)")
            await gantry_service.move_to(
                x_mm, y_mm, config.z_max_mm,
                speed=int(config.sweep_speed_mm_sec),
            )

            # Take N TOF readings; keep the minimum (closest surface = tallest plant)
            readings: list[float] = []
            for _ in range(config.tof_samples):
                tof_cm = await hardware.read_tof_distance()
                readings.append(tof_cm)
                await asyncio.sleep(0.1)

            min_tof_cm = min(readings)
            # height above bed floor: max Z in cm minus TOF reading
            height_cm = round((config.z_max_mm / 10.0) - min_tof_cm, 1)

            # Track per-column minimum TOF (tallest plant in each column)
            if col not in col_min_tof or min_tof_cm < col_min_tof[col]:
                col_min_tof[col] = min_tof_cm

            log.ok("TOF", f"row={row} col={col} tof={min_tof_cm:.1f}cm height={height_cm:.1f}cm")
            await event_bus.emit(
                str(session_id),
                {
                    "type": "tof_position_scanned",
                    "session_id": str(session_id),
                    "row": row,
                    "col": col,
                    "height_cm": height_cm,
                    "position": pos_idx,
                    "total": len(positions),
                },
            )

        # Global max height: z_max minus the globally smallest TOF reading
        global_min_tof = min(col_min_tof.values()) if col_min_tof else 0.0
        max_height_cm = round((config.z_max_mm / 10.0) - global_min_tof, 1)
        log.ok("TOF", f"sweep complete — global max_height={max_height_cm:.1f}cm")
        await event_bus.emit(
            str(session_id),
            {
                "type": "tof_sweep_complete",
                "session_id": str(session_id),
                "max_height_cm": max_height_cm,
            },
        )

        # ── Step 3: Soil moisture — before watering ────────────────────
        log.step("MOISTURE", "reading all 3 soil sensors before watering")
        sensors_before = await hardware.read_all_soil_sensors()
        moisture_before_avg = round(
            (sensors_before["s0"] + sensors_before["s1"] + sensors_before["s2"]) / 3.0, 1
        )
        log.ok(
            "MOISTURE",
            f"before: s0={sensors_before['s0']}% s1={sensors_before['s1']}%"
            f" s2={sensors_before['s2']}% avg={moisture_before_avg}%",
        )
        await event_bus.emit(
            str(session_id),
            {
                "type": "moisture_read_before",
                "session_id": str(session_id),
                "sensors": [sensors_before["s0"], sensors_before["s1"], sensors_before["s2"]],
                "avg_pct": moisture_before_avg,
            },
        )

        # ── Step 4: Fuzzy logic ────────────────────────────────────────
        fuzzy_duration = compute_watering_duration(max_height_cm, moisture_before_avg)
        log.ok(
            "FUZZY",
            f"height={max_height_cm}cm moisture={moisture_before_avg}%"
            f" → duration={fuzzy_duration}s",
        )
        await event_bus.emit(
            str(session_id),
            {
                "type": "fuzzy_computed",
                "session_id": str(session_id),
                "max_height_cm": max_height_cm,
                "avg_moisture_pct": moisture_before_avg,
                "duration_sec": fuzzy_duration,
            },
        )

        # ── Step 5: Watering loop ──────────────────────────────────────
        log.step("PUMP", "turning water pump ON")
        await gantry_service.set_relay("dc", on=True)
        log.log_pump_on()

        for col in range(config.cols):
            x_mm = config.col_x_mm(col)
            col_min = col_min_tof.get(col, global_min_tof)
            col_height_cm = round((config.z_max_mm / 10.0) - col_min, 1)

            log.step("WATERING", f"moving to col={col} x={x_mm:.0f}mm y={center_y:.0f}mm")
            await gantry_service.move_to(
                x_mm, center_y, config.z_water_mm,
                speed=int(config.water_speed_mm_sec),
            )

            if fuzzy_duration > 0:
                log.step("VALVE", f"opening solenoid for {fuzzy_duration}s at col={col}")
                await hardware.open_valve(fuzzy_duration)
                log.log_valve_done(fuzzy_duration)

            await pi_client.post_watering_stop(
                session_id, col, x_mm, center_y, col_height_cm, fuzzy_duration
            )
            await event_bus.emit(
                str(session_id),
                {
                    "type": "watering_stop",
                    "session_id": str(session_id),
                    "stop_index": col,
                    "x_mm": x_mm,
                    "y_mm": center_y,
                    "duration_sec": fuzzy_duration,
                },
            )
            watering_stops_count += 1

        log.step("PUMP", "turning water pump OFF")
        await gantry_service.set_relay("dc", on=False)
        log.log_pump_off(reason="watering complete")

        # ── Step 6: Soil moisture — after watering ─────────────────────
        log.step("MOISTURE", "reading all 3 soil sensors after watering")
        sensors_after = await hardware.read_all_soil_sensors()
        moisture_after_avg = round(
            (sensors_after["s0"] + sensors_after["s1"] + sensors_after["s2"]) / 3.0, 1
        )
        log.ok(
            "MOISTURE",
            f"after: s0={sensors_after['s0']}% s1={sensors_after['s1']}%"
            f" s2={sensors_after['s2']}% avg={moisture_after_avg}%",
        )
        await event_bus.emit(
            str(session_id),
            {
                "type": "moisture_read_after",
                "session_id": str(session_id),
                "sensors": [sensors_after["s0"], sensors_after["s1"], sensors_after["s2"]],
                "avg_pct": moisture_after_avg,
            },
        )

        # ── Step 7: Post summary + complete ───────────────────────────
        summary = {
            "max_height_cm": max_height_cm,
            "fuzzy_duration_sec": fuzzy_duration,
            "stops_watered": watering_stops_count,
            "moisture_before_avg": moisture_before_avg,
            "moisture_after_avg": moisture_after_avg,
        }
        log.info(f"summary: {summary}", tag="SESSION")
        await pi_client.post_watering_complete(session_id, summary)
        await event_bus.emit(
            str(session_id),
            {
                "type": "session_complete",
                "session_id": str(session_id),
                "summary": summary,
            },
        )

        # ── Home before parking ───────────────────────────────────────
        # Return the gantry to a known safe position after a successful run.
        # Best-effort: a homing hiccup must not flip a completed run to error.
        log.step("GANTRY", "homing gantry after successful watering")
        try:
            await gantry_service.home()
        except Exception as e:
            log.warn("SESSION", f"home after complete failed — {e}")

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
