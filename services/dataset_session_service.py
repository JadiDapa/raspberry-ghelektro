"""
Dataset session service — orchestrates a continuous video-collection sweep.

Unlike the scan loop, this session does NOT stop at each plant, capture stills,
or run YOLO. It homes, then sweeps the whole bed in one continuous serpentine
pass at the configured speed while recording a single video. When the sweep ends
it uploads the video to Next.js and stores the URL on the session.

Fatality model (mirrors session_service):
  - Gantry/home failures are FATAL — a hardware fault must stop the machine.
  - The video upload is FATAL too (the video is the whole point of the session),
    but the recorded file is always kept on disk and its path logged so footage
    is never lost. There is no /sync or outbox fallback for video (too large).
  - The status patch to Next.js is non-fatal (logged, run continues).
"""

import asyncio
import os
from datetime import datetime, timezone

from models.dataset_config import DatasetConfig
from services import event_bus, video_recorder
from services import gantry as gantry_service
from services import pi_client
from services.session_logger import SessionLogger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_dataset_session(session_id: int, config: DatasetConfig | None = None) -> None:
    if config is None:
        config = DatasetConfig()

    log = SessionLogger(str(session_id))
    waypoints = config.serpentine_waypoints()
    total_rows = config.rows
    recording_path: str | None = None

    async def safe_post(label: str, coro) -> None:
        """Await a non-fatal pi_client post; on failure log and continue."""
        try:
            await coro
        except Exception as e:
            log.warn("SYNC", f"{label} failed (non-fatal) — {e}")

    try:
        # ── Session start ──────────────────────────────────────────────
        await safe_post("patch_status running", pi_client.patch_status(session_id, "running"))
        log.info(f"dataset session start  rows={total_rows} cols={config.cols} "
                 f"speed={config.speed_mm_sec} z={config.z_mm}", tag="SESSION")
        log.info(f"log file  path={log.path}", tag="SESSION")

        await event_bus.emit(
            str(session_id),
            {
                "type": "session_started",
                "session_id": str(session_id),
                "timestamp": _now(),
                "total_plants": config.rows * config.cols,
                "total_rows": total_rows,
            },
        )
        await asyncio.sleep(0.1)

        # ── Hardware init + homing ─────────────────────────────────────
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

        # ── Start recording ────────────────────────────────────────────
        recording_path = video_recorder.start(session_id)
        log.info(f"recording started  path={recording_path}", tag="VIDEO")
        await event_bus.emit(
            str(session_id),
            {
                "type": "recording_started",
                "session_id": str(session_id),
                "total_rows": total_rows,
            },
        )

        # ── Continuous serpentine sweep ────────────────────────────────
        speed = int(round(config.speed_mm_sec))
        rows_swept = 0
        last_row = -1
        for (row, x_mm, y_mm) in waypoints:
            log.info(f"sweep → row={row} x={x_mm:.0f} y={y_mm:.0f}", tag="GANTRY")
            await event_bus.emit(
                str(session_id),
                {
                    "type": "gantry_moving",
                    "session_id": str(session_id),
                    "row": row,
                    "x": x_mm,
                    "y": y_mm,
                },
            )
            await gantry_service.move_to(x_mm, y_mm, config.z_mm, speed=speed)
            # Each row contributes two waypoints (its X extremes); count the row
            # as swept once we reach its far end (the second time we see it).
            if row != last_row:
                last_row = row
            else:
                rows_swept += 1
                await event_bus.emit(
                    str(session_id),
                    {
                        "type": "pass_progress",
                        "session_id": str(session_id),
                        "row": row,
                        "rows_swept": rows_swept,
                        "total_rows": total_rows,
                    },
                )

        # ── Stop recording ─────────────────────────────────────────────
        rec = video_recorder.stop()
        recording_path = rec.get("path", recording_path)
        log.info(f"recording stopped  frames={rec.get('frame_count')} "
                 f"duration={rec.get('duration_sec')}s", tag="VIDEO")

        # ── Upload video (FATAL on failure, but keep the local file) ────
        # Tell the browser the upload has begun so it can show a loading state.
        # This can take a while for a large sweep over a slow link, and it happens
        # before session_complete, so without this the UI would look stalled.
        log.step("VIDEO", "uploading recording to dashboard")
        try:
            size_mb = round(os.path.getsize(recording_path) / 1_048_576, 1)
        except OSError:
            size_mb = 0.0
        await event_bus.emit(
            str(session_id),
            {
                "type": "video_uploading",
                "session_id": str(session_id),
                "size_mb": size_mb,
            },
        )
        video_url = await pi_client.upload_video(session_id, recording_path)
        log.info(f"video uploaded → {video_url}", tag="VIDEO")

        summary = {
            "videoUrl": video_url,
            "durationSec": rec.get("duration_sec", 0.0),
            "frameCount": rec.get("frame_count", 0),
            "rowsSwept": rows_swept,
        }
        await safe_post("post_dataset_complete", pi_client.post_dataset_complete(session_id, summary))

        await event_bus.emit(
            str(session_id),
            {
                "type": "session_complete",
                "session_id": str(session_id),
                "summary": {"video_url": video_url, **summary},
            },
        )

        # ── Home + park ────────────────────────────────────────────────
        log.step("GANTRY", "homing gantry after sweep")
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
        _stop_recorder_quietly(log)
        await _safe_the_gantry(log)
        await safe_post("patch_status stopped", pi_client.patch_status(session_id, "stopped"))
        await event_bus.emit(
            str(session_id),
            {"type": "session_error", "session_id": str(session_id), "message": "cancelled"},
        )
        log.log_session_stopped()

    except Exception as e:
        log.error("SESSION", f"unhandled exception: {e}")
        _stop_recorder_quietly(log)
        if recording_path:
            log.warn("VIDEO", f"recording kept on disk for recovery → {recording_path}")
        await _safe_the_gantry(log)
        await safe_post("post_error", pi_client.post_error(session_id))
        await event_bus.emit(
            str(session_id),
            {"type": "session_error", "session_id": str(session_id), "message": str(e)},
        )
        log.log_session_error(str(e))

    finally:
        await asyncio.sleep(2)
        event_bus.destroy(str(session_id))
        log.close()


def _stop_recorder_quietly(log: SessionLogger) -> None:
    """Best-effort stop of the recorder on the cancel/error paths."""
    try:
        if video_recorder.is_recording():
            video_recorder.stop()
    except Exception as e:
        log.warn("VIDEO", f"cleanup: recorder stop failed — {e}")


async def _safe_the_gantry(log: SessionLogger) -> None:
    """Halt motion and cut motor power, best-effort. Used on stop/error paths."""
    try:
        await gantry_service.emergency_stop()
        log.log_motors_disabled()
    except Exception as e:
        log.warn("SESSION", f"cleanup: emergency_stop failed — {e}")
