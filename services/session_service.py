"""
Session service — orchestrates the full scan loop.

Persistence model:
  - Plant data is written to Next.js in real time via pi_client (live UI + history).
  - Those real-time posts are NON-FATAL: a network blip is logged and the scan
    continues instead of crashing the run. If any post failed, the whole session
    is reconciled once at the end via pi_client.sync_session() (retried); if even
    that can't reach Next.js, the payload is queued to the outbox and replayed at
    the next startup.
  - Gantry/sensor failures remain FATAL — a hardware fault must stop the machine.

Dashboard stays the single source of truth; the RPi only buffers in-flight data
when Next.js is unreachable.
"""

import asyncio
from datetime import datetime, timezone

from config import settings
from models.scan_config import ScanConfig
from services import event_bus, hardware, image_store, outbox
from services import gantry as gantry_service
from services import pi_client
from services.session_logger import SessionLogger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _counts(detections: list[dict]) -> dict:
    counts = {"ripe": 0, "turning": 0, "unripe": 0, "broken": 0}
    for d in detections:
        if d["cls"] in counts:
            counts[d["cls"]] += d["count"]
    return counts


async def run_session(session_id: int, config: ScanConfig | None = None) -> None:
    if config is None:
        config = ScanConfig()

    log = SessionLogger(str(session_id))
    plant_grid = config.plant_grid()
    plant_scans: list[dict] = []  # PiPlantScan dicts (carry "_image_bytes" for fallback)
    state = {"sync_dirty": False}
    started_at = _now()

    async def safe_post(label: str, coro) -> None:
        """Await a real-time pi_client post; on failure log, mark dirty, continue."""
        try:
            await coro
        except Exception as e:
            log.warn("SYNC", f"{label} failed (non-fatal) — {e}")
            state["sync_dirty"] = True

    try:
        # ── Session start ──────────────────────────────────────────────
        await safe_post("patch_status running", pi_client.patch_status(session_id, "running"))
        log.log_session_start(total_plants=len(plant_grid))
        log.info(f"log file  path={log.path}", tag="SESSION")

        await event_bus.emit(
            str(session_id),
            {
                "type": "session_started",
                "session_id": str(session_id),
                "timestamp": _now(),
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
            log.warn("LIMITS", f"switches already LOW at start: {already_triggered}")

        # ── Homing ────────────────────────────────────────────────────
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

        # ── Plant scan loop ───────────────────────────────────────────
        for plant_id, (row, col) in enumerate(plant_grid, start=1):
            log.log_plant_start(plant_id, len(plant_grid), row, col)

            # — Gantry move (FATAL on failure) —
            log.log_gantry_move_start(plant_id, row, col)
            await event_bus.emit(
                str(session_id),
                {
                    "type": "gantry_moving",
                    "session_id": str(session_id),
                    "plant_id": plant_id,
                    "row": row,
                    "col": col,
                },
            )
            await gantry_service.move_to_plant_with_config(row, col, config)
            await gantry_service.set_servo_angles(
                config.offset.servo_pan, config.offset.servo_tilt
            )
            gstate = gantry_service.get_state()
            log.log_gantry_move_done(gstate["x"], gstate["y"], gstate["z"])
            await event_bus.emit(
                str(session_id),
                {
                    "type": "gantry_moved",
                    "session_id": str(session_id),
                    "plant_id": plant_id,
                    "row": row,
                    "col": col,
                    "x": gstate["x"],
                    "y": gstate["y"],
                    "z": gstate["z"],
                },
            )

            # — Camera capture (FATAL on failure) —
            log.log_camera_capture_start(plant_id)
            image_bytes = await hardware.capture_image()

            # — Upload to Next.js (NON-FATAL) —
            image_url: str | None = None
            try:
                image_url = await pi_client.upload_image(session_id, plant_id, image_bytes)
            except Exception as e:
                log.warn("SYNC", f"upload_image plant {plant_id} failed (non-fatal) — {e}")
                state["sync_dirty"] = True
            log.log_camera_capture_done(image_url or "(not uploaded)")

            # — YOLO inference (FATAL on failure) —
            log.log_yolo_start(image_url or "(local)")
            detections, annotated_bytes = await hardware.run_yolo(
                image_bytes, roi=(config.roi_w_pct, config.roi_h_pct)
            )
            total_fruits = sum(d["count"] for d in detections)
            counts = _counts(detections)
            log.log_yolo_done(detections, total_fruits)

            # — Upload annotated frame to Next.js (NON-FATAL) —
            # The boxes-drawn image is what the dashboard shows by default; the raw
            # capture stays available behind the card toggle. None when YOLO ran in
            # stub mode or rendering failed → dashboard falls back to the raw image.
            annotated_url: str | None = None
            if annotated_bytes is not None:
                try:
                    annotated_url = await pi_client.upload_image(
                        session_id, plant_id, annotated_bytes, kind="annotated"
                    )
                except Exception as e:
                    log.warn("SYNC", f"upload annotated plant {plant_id} failed (non-fatal) — {e}")
                    state["sync_dirty"] = True

            # — Write vision result to Next.js (NON-FATAL) —
            await safe_post(
                f"post_vision plant {plant_id}",
                pi_client.post_vision(
                    session_id, plant_id, row, col, image_url or "", detections, annotated_url
                ),
            )
            await event_bus.emit(
                str(session_id),
                {
                    "type": "plant_scanned",
                    "session_id": str(session_id),
                    "plant_id": plant_id,
                    "image_url": image_url,
                    "annotated_image_url": annotated_url,
                    "detections": detections,
                    "total_fruits": total_fruits,
                },
            )

            plant_scans.append(
                {
                    "plant_id": plant_id,
                    "row": row,
                    "col": col,
                    "scanned_at": _now(),
                    "image_url": image_url,
                    "annotated_image_url": annotated_url,
                    "total_fruits": total_fruits,
                    "ripe_count": counts["ripe"],
                    "turning_count": counts["turning"],
                    "unripe_count": counts["unripe"],
                    "broken_count": counts["broken"],
                    "detections": detections,
                    "height_cm": None,
                    "moisture_pct": None,
                    "valve_duration_sec": None,
                    "watering_reason": None,
                    # Keep bytes only if the live upload failed — needed for the
                    # offline /sync fallback so the image isn't lost.
                    "_image_bytes": image_bytes if image_url is None else None,
                    "_annotated_bytes": (
                        annotated_bytes if annotated_url is None else None
                    ),
                }
            )
            log.log_plant_done(plant_id)

        # ── Session complete ───────────────────────────────────────────
        summary = _build_summary(plant_scans)
        log.log_summary(summary)
        await safe_post("post_complete", pi_client.post_complete(session_id, summary))

        # Only reconcile via /sync if a real-time post actually failed — no
        # redundant double-write in the normal case.
        if state["sync_dirty"]:
            await _sync_fallback(session_id, "complete", started_at, plant_scans, log)

        await event_bus.emit(
            str(session_id),
            {
                "type": "session_complete",
                "session_id": str(session_id),
                "summary": summary,
            },
        )

        # ── Home before parking ───────────────────────────────────────
        log.step("GANTRY", "homing gantry after successful scan")
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
        await _safe_the_gantry(log)
        await safe_post("patch_status stopped", pi_client.patch_status(session_id, "stopped"))
        if state["sync_dirty"] and plant_scans:
            await _sync_fallback(session_id, "stopped", started_at, plant_scans, log)
        await event_bus.emit(
            str(session_id),
            {"type": "session_error", "session_id": str(session_id), "message": "cancelled"},
        )
        log.log_session_stopped()

    except Exception as e:
        log.error("SESSION", f"unhandled exception: {e}")
        await _safe_the_gantry(log)
        await safe_post("post_error", pi_client.post_error(session_id))
        if state["sync_dirty"] and plant_scans:
            await _sync_fallback(session_id, "error", started_at, plant_scans, log)
        await event_bus.emit(
            str(session_id),
            {"type": "session_error", "session_id": str(session_id), "message": str(e)},
        )
        log.log_session_error(str(e))

    finally:
        await asyncio.sleep(2)
        event_bus.destroy(str(session_id))
        log.close()


async def _safe_the_gantry(log: SessionLogger) -> None:
    """Halt motion and cut motor power, best-effort. Used on stop/error paths."""
    try:
        await gantry_service.emergency_stop()
        log.log_motors_disabled()
    except Exception as e:
        log.warn("SESSION", f"cleanup: emergency_stop failed — {e}")


async def _sync_fallback(
    session_id: int,
    status: str,
    started_at: str,
    plant_scans: list[dict],
    log: SessionLogger,
) -> None:
    """
    Reconcile the whole session in one POST when real-time posts failed. Persists
    any not-yet-uploaded images to disk so the dashboard can fetch them, then calls
    /sync (retried). If the dashboard is still unreachable, queues to the outbox.
    """
    log.warn("SYNC", "real-time posts failed during run — sending whole-session /sync fallback")

    payload_scans: list[dict] = []
    for scan in plant_scans:
        scan = dict(scan)
        img_bytes = scan.pop("_image_bytes", None)
        annotated_bytes = scan.pop("_annotated_bytes", None)
        if scan.get("image_url") is None and img_bytes is not None:
            try:
                scan["image_url"] = image_store.save(session_id, scan["plant_id"], img_bytes)
            except Exception as e:
                log.warn("SYNC", f"could not persist image for plant {scan['plant_id']} — {e}")
        if scan.get("annotated_image_url") is None and annotated_bytes is not None:
            try:
                scan["annotated_image_url"] = image_store.save(
                    session_id, scan["plant_id"], annotated_bytes, kind="annotated"
                )
            except Exception as e:
                log.warn("SYNC", f"could not persist annotated image for plant {scan['plant_id']} — {e}")
        payload_scans.append(scan)

    payload = {
        "session_id": str(session_id),
        "status": status,
        "bed_id": settings.bed_id,
        "started_at": started_at,
        "completed_at": _now(),
        "plant_scans": payload_scans,
        "summary": _build_sync_summary(plant_scans),
    }

    try:
        await pi_client.sync_session(payload)
        log.info("whole-session /sync fallback succeeded", tag="SYNC")
    except Exception as e:
        log.warn("SYNC", f"/sync fallback unreachable — queuing to outbox ({e})")
        outbox.append(payload)


def _build_summary(plant_scans: list[dict]) -> dict:
    """Summary shape consumed by the real-time /complete endpoint (camelCase)."""
    if not plant_scans:
        return {}

    ripeness = {"ripe": 0, "turning": 0, "unripe": 0, "broken": 0}
    for ps in plant_scans:
        ripeness["ripe"] += ps["ripe_count"]
        ripeness["turning"] += ps["turning_count"]
        ripeness["unripe"] += ps["unripe_count"]
        ripeness["broken"] += ps["broken_count"]

    harvest_ready_ids = [ps["plant_id"] for ps in plant_scans if ps["ripe_count"] > 5]

    return {
        "totalPlants": len(plant_scans),
        "ripeness": ripeness,
        "harvestReadyIds": harvest_ready_ids,
    }


def _build_sync_summary(plant_scans: list[dict]) -> dict:
    """Summary shape consumed by /api/sessions/sync (snake_case)."""
    ripeness = {"ripe": 0, "turning": 0, "unripe": 0, "broken": 0}
    for ps in plant_scans:
        ripeness["ripe"] += ps["ripe_count"]
        ripeness["turning"] += ps["turning_count"]
        ripeness["unripe"] += ps["unripe_count"]
        ripeness["broken"] += ps["broken_count"]

    heights = [ps["height_cm"] for ps in plant_scans if ps.get("height_cm") is not None]
    moistures = [ps["moisture_pct"] for ps in plant_scans if ps.get("moisture_pct") is not None]
    total_water = sum(ps.get("valve_duration_sec") or 0 for ps in plant_scans)
    harvest_ready_ids = [ps["plant_id"] for ps in plant_scans if ps["ripe_count"] > 5]

    return {
        "total_plants": len(plant_scans),
        "avg_height_cm": (sum(heights) / len(heights)) if heights else 0.0,
        "avg_moisture_pct": (sum(moistures) / len(moistures)) if moistures else 0.0,
        "total_water_sec": total_water,
        "ripeness": ripeness,
        "harvest_ready_ids": harvest_ready_ids,
    }
