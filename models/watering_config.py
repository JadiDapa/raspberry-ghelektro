from pydantic import BaseModel


class WateringConfig(BaseModel):
    cols: int = 8
    rows: int = 2
    gap_x_mm: float = 750.0
    gap_y_mm: float = 1000.0
    padding_x_mm: float = 0.0
    padding_y_mm: float = 0.0
    z_max_mm: float = 0.0       # Z raised to this for TOF sweep
    z_water_mm: float = 50.0   # Z working height during valve open
    tof_samples: int = 5        # TOF readings taken per plant position
    sweep_speed_mm_sec: float = 150.0
    water_speed_mm_sec: float = 100.0

    def col_x_mm(self, col: int) -> float:
        return self.padding_x_mm + col * self.gap_x_mm

    def row_y_mm(self, row: int) -> float:
        return self.padding_y_mm + row * self.gap_y_mm

    def center_y_mm(self) -> float:
        """Y midpoint between all rows — sprinkler arm covers the full width."""
        return self.padding_y_mm + ((self.rows - 1) * self.gap_y_mm) / 2.0

    def plant_positions(self) -> list[tuple[int, int]]:
        """All (row, col) pairs for TOF sweep, row-major order."""
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
        ]
