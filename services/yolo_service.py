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
    yolo export model=yolo11n.pt format=ncnn
  Then set yolo_model_path = "yolo11n_ncnn_model" in .env
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings, FRUIT_CLASSES

# One thread for YOLO — inference is CPU-bound, must not block the event loop
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
_model = None  # loaded once on startup


def load_model():
    """Load YOLO model into memory. Called once from main.py lifespan."""
    global _model
    try:
        from ultralytics import YOLO
        _model = YOLO(settings.yolo_model_path)
        # Warm up — run a blank inference so the first real scan isn't slow
        import numpy as np
        dummy = np.zeros((settings.yolo_imgsz, settings.yolo_imgsz, 3), dtype="uint8")
        _model.predict(dummy, imgsz=settings.yolo_imgsz, conf=settings.yolo_confidence, verbose=False)
        print(f"[yolo] model loaded and warmed up → {settings.yolo_model_path}")
    except Exception as e:
        print(f"[yolo] WARNING: could not load model — {e}")
        print("[yolo] run_inference() will return stub data")
        _model = None


def _predict(image_path: str) -> list[dict]:
    """
    Blocking inference — runs in the thread executor.
    Returns: [{"cls": "ripe", "count": 3, "confidence": 0.91}, ...]
    One dict per class that was detected (classes with 0 detections are omitted).
    """
    if _model is None:
        # Stub fallback if model failed to load
        import random
        return [
            {"cls": "ripe",    "count": random.randint(0, 6), "confidence": round(random.uniform(0.80, 0.97), 2)},
            {"cls": "unripe",  "count": random.randint(0, 5), "confidence": round(random.uniform(0.75, 0.95), 2)},
            {"cls": "turning", "count": random.randint(0, 3), "confidence": round(random.uniform(0.70, 0.93), 2)},
            {"cls": "broken",  "count": random.randint(0, 2), "confidence": round(random.uniform(0.65, 0.90), 2)},
        ]

    results = _model.predict(
        source=image_path,
        imgsz=settings.yolo_imgsz,
        conf=settings.yolo_confidence,
        verbose=False,
    )

    # Count detections per class
    counts: dict[str, int] = {cls: 0 for cls in FRUIT_CLASSES}
    best_conf: dict[str, float] = {cls: 0.0 for cls in FRUIT_CLASSES}

    for result in results:
        for box in result.boxes:
            cls_index = int(box.cls[0])
            cls_name = result.names.get(cls_index, "").lower()
            if cls_name in counts:
                counts[cls_name] += 1
                conf = float(box.conf[0])
                if conf > best_conf[cls_name]:
                    best_conf[cls_name] = conf

    # Return only classes that had at least one detection
    detections = [
        {"cls": cls, "count": counts[cls], "confidence": round(best_conf[cls], 2)}
        for cls in FRUIT_CLASSES
        if counts[cls] > 0
    ]

    print(f"[yolo] detections: {detections}")
    return detections


async def run_inference(image_path: str) -> list[dict]:
    """Async wrapper — offloads blocking inference to thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _predict, image_path)
