from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Grid
    grid_rows: int = 2
    grid_cols: int = 8

    # Paths
    images_dir: str = "static/images"
    yolo_model_path: str = "yolo11n.pt"

    # ESP32 #1 — Motion (USB serial)
    esp32_port: str = "/dev/ttyUSB0"
    esp32_baudrate: int = 115200

    # ESP32 #2 — Soil moisture (UART GPIO 14/15)
    soil_uart_port: str = "/dev/ttyAMA0"
    soil_uart_baudrate: int = 115200

    # Camera
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 15
    camera_jpeg_quality: int = 85
    camera_stabilize_delay: float = 1.0

    # YOLO
    yolo_confidence: float = 0.4
    yolo_imgsz: int = 640

    # Timing stubs — remove each once real hardware replaces it
    sensor_read_delay: float = 0.3

    class Config:
        env_file = ".env"


settings = Settings()

# Fruit detection classes — must match your model's training labels exactly
FRUIT_CLASSES = ["unripe", "turning", "ripe", "broken"]

# Plant grid — (row, col) positions for plant_id 1..16
PLANT_GRID = [
    (row, col) for row in range(settings.grid_rows) for col in range(settings.grid_cols)
]
