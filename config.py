from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── Pydantic config ───────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # 🔥 prevents crash from leftover env vars
    )

    # ─── Paths ─────────────────────────────────────────────────────────
    images_dir: str = "static/images"
    yolo_model_path: str = "best.pt"

    # ─── ESP32 #1 — Motion (USB serial) ────────────────────────────────
    esp32_port: str = "/dev/ttyUSB0"
    esp32_baudrate: int = 115200

    # ─── ESP32 #2 — Soil moisture (UART GPIO 14/15) ────────────────────
    soil_uart_port: str = "/dev/ttyAMA0"
    soil_uart_baudrate: int = 115200

    # ─── Camera (Raspberry Pi optimized) ───────────────────────────────
    camera_device: str = "/dev/video0"

    # 🔥 SAFE defaults for Raspberry Pi (you can increase later)
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10

    camera_jpeg_quality: int = 85
    camera_stabilize_delay: float = 1.0

    # ─── YOLO ──────────────────────────────────────────────────────────
    yolo_confidence: float = 0.4
    yolo_imgsz: int = 640

    # ─── Timing stubs ──────────────────────────────────────────────────
    sensor_read_delay: float = 0.3
    gantry_move_delay: float = 2.0
    stub_gantry_delay: float = 0.05  # fake move delay used when stub_mode=true

    # ─── CORS ───────────────────────────────────────────────────────────
    # Comma-separated list of allowed browser origins (exact match), e.g.
    #   CORS_ORIGINS=https://dashboard.example.com,http://203.0.113.5:3000
    # localhost is always allowed for dev. Vercel previews are matched by regex.
    cors_origins: str = ""

    # ─── Dashboard sync ─────────────────────────────────────────────────
    dashboard_url: str = ""  # e.g. http://192.168.1.10:3000
    rpi_base_url: str = "http://localhost:8000"  # used to build absolute image URLs
    bed_id: str = "1"  # Next.js Bed.id this RPi manages

    # ─── Sync resilience ────────────────────────────────────────────────
    # Real-time posts to Next.js retry on transient failures (transport errors
    # and 5xx) with exponential backoff; 4xx are not retried. If the whole
    # end-of-session sync still can't reach Next.js, the full payload is written
    # to outbox_dir and replayed on the next startup (see services/outbox.py).
    sync_max_retries: int = 3
    sync_backoff_base: float = 0.5  # seconds; doubles each attempt
    outbox_dir: str = "pending_sync"

    # Tiny persistent marker for the currently-running session. Written at session
    # start, removed at clean end. If it survives a restart, the process died mid
    # session → orphan recovery marks that session errored and safes the gantry.
    runtime_dir: str = "runtime"

    # ─── Dev / debug ────────────────────────────────────────────────────
    # Set STUB_MODE=true to run the full session pipeline without any hardware.
    # All serial/camera calls already fall back to stubs when hardware is absent;
    # this flag additionally bypasses dashboard HTTP calls so sessions complete
    # end-to-end without a running Next.js instance.
    stub_mode: bool = False


settings = Settings()

# ─── Constants ─────────────────────────────────────────────────────────

FRUIT_CLASSES = ["unripe", "turning", "ripe", "broken"]
