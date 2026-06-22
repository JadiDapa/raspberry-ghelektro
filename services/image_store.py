"""
image_store.py — persist captured frames to disk for the offline fallback.

In the happy path images go straight to the dashboard (pi_client.upload_image)
and nothing is written here. Only when a real-time upload fails do we save the
JPEG locally so the end-of-session /sync payload can carry a URL the dashboard
can still fetch (served via the /static mount in main.py). These files are a
temporary fallback, not a permanent gallery — they can be cleaned up once the
dashboard has pulled them.
"""

from pathlib import Path

from config import settings


def _session_dir(session_id: int) -> Path:
    d = Path(settings.images_dir) / f"session_{session_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(session_id: int, plant_id: int, image_bytes: bytes, kind: str = "raw") -> str:
    """
    Write the JPEG to disk and return a dashboard-fetchable RPi URL.

    The path mirrors the /static mount: <images_dir>/session_<id>/plant_<n>.jpg
    is served at {rpi_base_url}/static/images/session_<id>/plant_<n>.jpg.
    `kind` ("raw" | "annotated") keeps the raw capture and the YOLO-annotated
    frame in separate files so a fallback /sync can carry both.
    """
    suffix = "" if kind == "raw" else f"_{kind}"
    fname = f"plant_{plant_id}{suffix}.jpg"
    (_session_dir(session_id) / fname).write_bytes(image_bytes)
    base = settings.rpi_base_url.rstrip("/")
    return f"{base}/static/images/session_{session_id}/{fname}"
