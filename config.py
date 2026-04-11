from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── Pydantic config ───────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # 🔥 prevents crash from leftover env vars
    )

    # ─── Grid ──────────────────────────────────────────────────────────
    grid_rows: int = 2
    grid_cols: int = 8

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


settings = Settings()

# ─── Constants ─────────────────────────────────────────────────────────

FRUIT_CLASSES = ["unripe", "turning", "ripe", "broken"]

PLANT_GRID = [
    (row, col) for row in range(settings.grid_rows) for col in range(settings.grid_cols)
]
