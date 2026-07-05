"""
YOLO service — loads the model once at startup and runs inference.

Why a separate file:
  Loading a YOLO model takes ~2–5 seconds on a Pi. We do it once here
  at startup, not on every scan. All calls go through run_inference().

Pi CPU note:
  YOLO11n (nano) is the fastest model in the family. On a Pi 4 CPU,
  expect ~3–8 seconds per inference at 640px. This is fine because
  the gantry is stationary while we scan each plant anyway.

  If you need it faster later, export to NCNN format:
    yolo export model=best.pt format=ncnn
  Then set yolo_model_path = "yolo11n_ncnn_model" in .env
"""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from config import settings, FRUIT_CLASSES

# One thread for YOLO — inference is CPU-bound, must not block the event loop
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
_model = None  # built-in fallback, loaded once on startup

# Per-session models selected in the dashboard, cached by checksum so a model is
# downloaded + loaded once and reused across sessions. See prepare_session_model.
_model_cache: dict[str, object] = {}


# ─── Per-session model download + load ────────────────────────────────────────

def _models_dir() -> Path:
    d = Path(settings.models_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _download_model(file_url: str, checksum: str) -> str:
    """Download weights from the dashboard to models_dir, cached by checksum.

    `file_url` is the dashboard-relative URL (/api/uploads/<name>); we prefix
    settings.dashboard_url. Skips the download when a file for this checksum
    already exists on disk. Verifies the sha256 when a checksum is provided.
    Returns the local path.
    """
    ext = Path(file_url).suffix or ".pt"
    dest = _models_dir() / f"{(checksum or 'model')[:16]}{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[yolo] model cache hit → {dest}")
        return str(dest)

    base = settings.dashboard_url.rstrip("/")
    url = file_url if file_url.startswith("http") else f"{base}{file_url}"
    print(f"[yolo] downloading model {url} → {dest}")
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.content

    if checksum:
        actual = hashlib.sha256(data).hexdigest()
        if actual != checksum:
            raise ValueError(
                f"model checksum mismatch (expected {checksum[:12]}…, got {actual[:12]}…)"
            )
    dest.write_bytes(data)
    print(f"[yolo] model saved ({len(data) / 1_048_576:.1f} MB)")
    return str(dest)


def _load_model_sync(path: str):
    from ultralytics import YOLO

    m = YOLO(path)
    return m


async def prepare_session_model(model_cfg: dict | None) -> dict | None:
    """Resolve the model a scan session should run.

    `model_cfg` is the snapshot's `model` sub-object (dict) or None. Returns a
    handle {model, imgsz, conf, iou, max_det} passed to run_inference, or None to
    fall back to the built-in startup model. Non-fatal: any download/load failure
    is logged and returns None so the session still runs on the built-in model.
    """
    if not model_cfg or not model_cfg.get("file_url"):
        return None
    try:
        checksum = model_cfg.get("checksum") or ""
        cache_key = checksum or model_cfg["file_url"]
        model = _model_cache.get(cache_key)
        if model is None:
            path = await _download_model(model_cfg["file_url"], checksum)
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(_executor, _load_model_sync, path)
            _model_cache[cache_key] = model
            print(f"[yolo] session model ready → {model_cfg.get('name') or cache_key}")
        return {
            "model": model,
            "imgsz": int(model_cfg.get("imgsz") or settings.yolo_imgsz),
            "conf": float(model_cfg.get("confidence") or settings.yolo_confidence),
            "iou": float(model_cfg.get("iou_nms") or 0.7),
            "max_det": int(model_cfg.get("max_det") or 300),
        }
    except Exception as e:
        print(f"[yolo] could not prepare session model ({e}) — using built-in model")
        return None


def load_model():
    """Load YOLO model into memory. Called once from main.py lifespan."""
    global _model
    try:
        from ultralytics import YOLO

        _model = YOLO(settings.yolo_model_path)
        # Warm up — run a blank inference so the first real scan isn't slow
        import numpy as np

        dummy = np.zeros((settings.yolo_imgsz, settings.yolo_imgsz, 3), dtype="uint8")
        _model.predict(
            dummy,
            imgsz=settings.yolo_imgsz,
            conf=settings.yolo_confidence,
            verbose=False,
        )
        print(f"[yolo] model loaded and warmed up → {settings.yolo_model_path}")
    except Exception as e:
        print(f"[yolo] WARNING: could not load model — {e}")
        print("[yolo] run_inference() will return stub data")
        _model = None


def _stub_detections() -> list[dict]:
    import random

    return [
        {
            "cls": "ripe",
            "count": random.randint(0, 6),
            "confidence": round(random.uniform(0.80, 0.97), 2),
        },
        {
            "cls": "unripe",
            "count": random.randint(0, 5),
            "confidence": round(random.uniform(0.75, 0.95), 2),
        },
        {
            "cls": "turning",
            "count": random.randint(0, 3),
            "confidence": round(random.uniform(0.70, 0.93), 2),
        },
        {
            "cls": "broken",
            "count": random.randint(0, 2),
            "confidence": round(random.uniform(0.65, 0.90), 2),
        },
    ]


def _roi_box(arr, roi) -> tuple[float, float, float, float] | None:
    """Return the centered ROI rect (x0, y0, x1, y1) in pixels, or None.

    `roi` is (w_pct, h_pct) of the frame; None or 100×100 means whole frame, so
    we return None to signal "no filtering / no rectangle to draw".
    """
    if roi is None:
        return None
    w_pct, h_pct = roi
    if w_pct >= 100.0 and h_pct >= 100.0:
        return None
    h, w = arr.shape[:2]
    bw = w * w_pct / 100.0
    bh = h * h_pct / 100.0
    x0 = (w - bw) / 2.0
    y0 = (h - bh) / 2.0
    return x0, y0, x0 + bw, y0 + bh


def _predict_array(arr, roi=None, handle: dict | None = None) -> tuple[list[dict], bytes | None]:
    """
    Run inference on a numpy BGR array.

    `roi` is an optional (w_pct, h_pct) centered region: a detection is counted
    only when its bounding-box center falls inside that box, so fruit on
    neighboring plants visible at the frame edges is ignored.

    `handle` selects a per-session model + its inference settings (see
    prepare_session_model); None uses the built-in startup model and the global
    settings.yolo_* values.

    Returns (detections, annotated_jpeg_bytes). The annotated frame is the input
    image with YOLO boxes/labels drawn on it (via Ultralytics' result.plot()) plus
    the ROI rectangle, re-encoded as JPEG so it can be uploaded alongside the raw
    capture. It is None in stub mode (no model) or if rendering/encoding fails —
    callers fall back to the raw image in that case.
    """
    model = handle["model"] if handle else _model
    imgsz = handle["imgsz"] if handle else settings.yolo_imgsz
    conf = handle["conf"] if handle else settings.yolo_confidence
    iou = handle["iou"] if handle else 0.7
    max_det = handle["max_det"] if handle else 300

    if model is None:
        return _stub_detections(), None

    results = model.predict(
        source=arr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )

    box = _roi_box(arr, roi)

    counts: dict[str, int] = {cls: 0 for cls in FRUIT_CLASSES}
    best_conf: dict[str, float] = {cls: 0.0 for cls in FRUIT_CLASSES}

    for result in results:
        for det in result.boxes:
            if box is not None:
                cx, cy = (float(v) for v in det.xywh[0][:2])
                x0, y0, x1, y1 = box
                if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                    continue  # detection center outside ROI — belongs to a neighbor
            cls_index = int(det.cls[0])
            cls_name = result.names.get(cls_index, "").lower()
            if cls_name in counts:
                counts[cls_name] += 1
                conf = float(det.conf[0])
                if conf > best_conf[cls_name]:
                    best_conf[cls_name] = conf

    detections = [
        {"cls": cls, "count": counts[cls], "confidence": round(best_conf[cls], 2)}
        for cls in FRUIT_CLASSES
        if counts[cls] > 0
    ]
    print(f"[yolo] detections: {detections}")

    annotated_bytes = _render_annotated(results, box)
    return detections, annotated_bytes


def _render_annotated(results, roi_box=None) -> bytes | None:
    """Draw boxes (+ ROI rect) on the frame and JPEG-encode it. Non-fatal — None on failure."""
    try:
        import cv2

        annotated = results[0].plot()  # BGR ndarray with boxes + labels drawn
        if roi_box is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in roi_box)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", annotated)
        if ok:
            return buf.tobytes()
        print("[yolo] WARNING: could not JPEG-encode annotated frame")
    except Exception as e:
        print(f"[yolo] WARNING: could not render annotated image — {e}")
    return None


def _predict_from_bytes(
    image_bytes: bytes, roi=None, handle: dict | None = None
) -> tuple[list[dict], bytes | None]:
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return _predict_array(arr, roi, handle)


async def run_inference_from_bytes(
    image_bytes: bytes, roi=None, handle: dict | None = None
) -> tuple[list[dict], bytes | None]:
    """Async wrapper — offloads blocking inference to thread pool.

    `roi` is an optional (w_pct, h_pct) centered counting region; see _predict_array.
    `handle` selects a per-session model (see prepare_session_model); None uses the
    built-in startup model. Returns (detections, annotated_jpeg_bytes).
    """
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+
    return await loop.run_in_executor(
        _executor, _predict_from_bytes, image_bytes, roi, handle
    )
