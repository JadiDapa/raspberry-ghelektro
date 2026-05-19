from pydantic import BaseModel, Field


class CaptureOffset(BaseModel):
    z_mm: float = 50.0
    x_offset_mm: float = 0.0
    y_offset_mm: float = 0.0
    servo_pan: float = 90.0
    servo_tilt: float = 90.0


class ScanConfig(BaseModel):
    cols: int = 8
    rows: int = 2
    gap_x_mm: float = 750.0
    gap_y_mm: float = 1000.0
    padding_x_mm: float = 0.0
    padding_y_mm: float = 0.0
    capture_offsets: list[CaptureOffset] = Field(
        default_factory=lambda: [CaptureOffset()],
        min_length=1,
    )

    def plant_grid(self) -> list[tuple[int, int]]:
        """Return ordered (row, col) pairs for all plants in this config."""
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
        ]

    def plant_position_mm(self, row: int, col: int) -> tuple[float, float]:
        """Return absolute (x_mm, y_mm) for a given grid position."""
        x = self.padding_x_mm + col * self.gap_x_mm
        y = self.padding_y_mm + row * self.gap_y_mm
        return x, y

    @property
    def offset(self) -> CaptureOffset:
        """The single active capture offset (first entry)."""
        return self.capture_offsets[0]
