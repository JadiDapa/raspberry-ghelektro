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

        # ── Step 1 & 2: Continuous serpentine TOF height sweep ─────────
        # No stop-and-scan: the gantry sweeps each row in one uninterrupted
        # motion while the TOF is polled ~config.tof_sample_hz times a second.
        # Each sample is bucketed by X into its nearest column, so the per-column
        # and global max heights match the old discrete results — only the
        # acquisition method changed. Downstream (fuzzy + watering) is unchanged.
        await event_bus.emit(
            str(session_id),
            {
                "type": "tof_sweep_started",
                "session_id": str(session_id),
                "total_positions": len(positions),
            },
        )

        sample_interval = 1.0 / config.tof_sample_hz
        # (row, col) → minimum TOF seen (cm); lower = closer surface = taller plant
        pos_min_tof: dict[tuple[int, int], float] = {}
        position_index = 0

        for row, x_start, x_end, y_mm in config.sweep_segments():
            # Move to the row's start corner first (this also raises Z to z_max
            # on row 0; later rows just step Y since serpentine ends at the far
            # corner, which is the next row's start).
            log.step("TOF", f"row={row}: positioning to x={x_start:.0f}mm y={y_mm:.0f}mm z_max={config.z_max_mm}mm")
            await gantry_service.move_to(
                x_start, y_mm, config.z_max_mm,
                speed=int(config.sweep_speed_mm_sec),
            )

            log.step(
                "TOF",
                f"row={row}: continuous sweep to x={x_end:.0f}mm "
                f"@ {config.sweep_speed_mm_sec:.0f}mm/s, {config.tof_sample_hz:g}Hz",
            )
            samples = await gantry_service.sweep_tof(
                x_end, y_mm, config.z_max_mm,
                speed=int(config.sweep_speed_mm_sec),
                sample_interval_s=sample_interval,
            )
            log.ok("TOF", f"row={row}: captured {len(samples)} height samples")

            # Bucket each sample into its nearest column; keep the closest (min TOF).
            for s in samples:
                key = (row, config.nearest_col(s["x"]))
                if key not in pos_min_tof or s["tof_cm"] < pos_min_tof[key]:
                    pos_min_tof[key] = s["tof_cm"]

            # Emit one position event per column in this row (ascending col) so
            # the dashboard progress/heights stay identical to stop-and-scan.
            for col in range(config.cols):
                position_index += 1
                min_tof_cm = pos_min_tof.get((row, col))
                if min_tof_cm is None:
                    # No valid sample over this column (sensor miss) — leave it
                    # out of col_min_tof so watering falls back to the global min.
                    height_cm = 0.0
                else:
                    height_cm = config.height_cm(min_tof_cm)
                    if col not in col_min_tof or min_tof_cm < col_min_tof[col]:
                        col_min_tof[col] = min_tof_cm

                log.ok(
                    "TOF",
                    f"row={row} col={col} "
                    f"tof={'n/a' if min_tof_cm is None else f'{min_tof_cm:.1f}cm'} "
                    f"height={height_cm:.1f}cm",
                )
                await event_bus.emit(
                    str(session_id),
                    {
                        "type": "tof_position_scanned",
                        "session_id": str(session_id),
                        "row": row,
                        "col": col,
                        "height_cm": height_cm,
                        "position": position_index,
                        "total": len(positions),
                    },
                )

        # Global max height from the globally smallest TOF reading. With no valid
        # samples at all (total sensor failure) report 0 rather than inventing a
        # full-height reading, so a dead TOF can't trigger max watering.
        global_min_tof = min(col_min_tof.values()) if col_min_tof else None
        max_height_cm = config.height_cm(global_min_tof) if global_min_tof is not None else 0.0
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
            x_mm = config.water_col_x_mm(col)   # plant column X + nozzle offset
            col_min = col_min_tof.get(col, global_min_tof)
            col_height_cm = config.height_cm(col_min) if col_min is not None else 0.0

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
