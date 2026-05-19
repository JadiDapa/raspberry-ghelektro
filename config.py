from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── Pydantic config ───────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # 🔥 prevents crash from leftover env vars
    )

    # ─── Paths ─────────────────────────────────────────────────────────
    images_dir: str = "static/images"
    yolo_model_path: str = "yolo11n.pt"

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
    gantry_move_delay: float = 2.0  # ← add this

    # ─── Dashboard sync ─────────────────────────────────────────────────
    dashboard_url: str = ""                      # e.g. http://192.168.1.10:3000
    rpi_base_url: str = "http://localhost:8000"  # used to build absolute image URLs
    bed_id: str = "1"                            # Next.js Bed.id this RPi manages


settings = Settings()

# ─── Constants ─────────────────────────────────────────────────────────

FRUIT_CLASSES = ["unripe", "turning", "ripe", "broken"]
