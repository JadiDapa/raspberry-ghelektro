"""
Session service — orchestrates the full scan loop.
Persists every step to the database and emits SSE events.
"""

import asyncio
import os
from datetime import datetime, timezone

from config import settings, PLANT_GRID
from services import event_bus, hardware
from services import gantry as gantry_service
from db.database import AsyncSessionLocal
from db import crud


async def run_session(session_id: str) -> None:
    print(f"\n{'='*50}")
    print(f"[session] starting {session_id}")
    print(f"{'='*50}\n")

    os.makedirs(os.path.join(settings.images_dir, session_id), exist_ok=True)
    plant_results = []

    async with AsyncSessionLocal() as db:
        try:
            await crud.set_session_running(db, session_id)
            await event_bus.emit(session_id, {
                "type": "session_started",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_plants": len(PLANT_GRID),
            })
            await asyncio.sleep(0.1)

            # Turn water pump on for the entire session
            await gantry_service.set_relay("dc", on=True)
            print("[session] water pump ON")

            # Home gantry
            position = await gantry_service.home()
            await event_bus.emit(session_id, {
                "type": "motors_homed",
                "session_id": session_id,
                "position": {"x": position["x"], "y": position["y"], "z": position["z"]},
            })

            for plant_id, (row, col) in enumerate(PLANT_GRID, start=1):
                print(f"\n--- Plant {plant_id}/{len(PLANT_GRID)} (row={row}, col={col}) ---")
                await crud.create_plant_scan(db, session_id, plant_id, row, col)

                # Move gantry above plant
                await gantry_service.move_to_plant(row, col)
                state = gantry_service.get_state()
                await event_bus.emit(session_id, {
                    "type": "gantry_moved",
                    "session_id": session_id,
                    "plant_id": plant_id,
                    "row": row, "col": col,
                    "x": state["x"], "y": state["y"], "z": state["z"],
                })

                # Camera + YOLO
                image_path = await hardware.capture_image(plant_id, session_id)
                image_url = f"/static/images/{session_id}/plant_{plant_id:02d}.jpg"
                detections = await hardware.run_yolo(image_path)
                total_fruits = sum(d["count"] for d in detections)
                await crud.update_plant_scan_vision(db, session_id, plant_id, image_url, detections)
                await event_bus.emit(session_id, {
                    "type": "plant_scanned",
                    "session_id": session_id,
                    "plant_id": plant_id,
                    "image_url": image_url,
                    "detections": detections,
                    "total_fruits": total_fruits,
                })

                # Sensors
                height_cm = await hardware.read_tof_distance()
                moisture_pct = await hardware.read_soil_moisture(col)
                await event_bus.emit(session_id, {
                    "type": "sensor_read",
                    "session_id": session_id,
                    "plant_id": plant_id,
                    "height_cm": height_cm,
                    "moisture_pct": moisture_pct,
                })

                # Fuzzy + valve
                valve_duration, reason = hardware.compute_watering_duration(moisture_pct)
                if valve_duration > 0:
                    await hardware.open_valve(valve_duration)
                await crud.update_plant_scan_sensors(db, session_id, plant_id, height_cm, moisture_pct, valve_duration, reason)
                await event_bus.emit(session_id, {
                    "type": "plant_watered",
                    "session_id": session_id,
                    "plant_id": plant_id,
                    "valve_duration_sec": valve_duration,
                    "reason": reason,
                })

                plant_results.append({
                    "plant_id": plant_id,
                    "row": row, "col": col,
                    "detections": detections,
                    "height_cm": height_cm,
                    "moisture_pct": moisture_pct,
                    "valve_duration_sec": valve_duration,
                })

            summary = _build_summary(plant_results)
            await crud.complete_session(db, session_id, summary)
            await event_bus.emit(session_id, {
                "type": "session_complete",
                "session_id": session_id,
                "summary": summary,
            })

            # Turn water pump off — session finished
            await gantry_service.set_relay("dc", on=False)
            print("[session] water pump OFF")
            print(f"\n[session] {session_id} complete")

        except asyncio.CancelledError:
            await gantry_service.set_relay("dc", on=False)
            print("[session] water pump OFF (cancelled)")
            await crud.set_session_stopped(db, session_id)
            await event_bus.emit(session_id, {
                "type": "session_error",
                "session_id": session_id,
                "message": "cancelled",
            })

        except Exception as e:
            print(f"[session] ERROR: {e}")
            await gantry_service.set_relay("dc", on=False)
            print("[session] water pump OFF (error)")
            await crud.set_session_error(db, session_id, str(e))
            await event_bus.emit(session_id, {
                "type": "session_error",
                "session_id": session_id,
                "message": str(e),
            })

        finally:
            await asyncio.sleep(2)
            event_bus.destroy(session_id)


def _build_summary(plant_results: list[dict]) -> dict:
    if not plant_results:
        return {}

    heights = [p["height_cm"] for p in plant_results]
    moisture = [p["moisture_pct"] for p in plant_results]

    # Count all detections across all plants per class
    ripeness = {"ripe": 0, "turning": 0, "unripe": 0, "broken": 0}
    for p in plant_results:
        for d in p["detections"]:
            if d["cls"] in ripeness:
                ripeness[d["cls"]] += d["count"]

    # Harvest ready = plants with more than 5 ripe fruits
    harvest_ready = [
        {
            "plant_id": p["plant_id"],
            "row": p["row"],
            "col": p["col"],
            "ripe_count": sum(d["count"] for d in p["detections"] if d["cls"] == "ripe"),
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
        "total_water_sec": round(sum(p["valve_duration_sec"] for p in plant_results), 1),
    }
