from pydantic import BaseModel, Field, model_validator

# Gantry working envelope (mm) — see models/scan_config.py.
X_MAX_MM = 6000.0
Y_MAX_MM = 2000.0
Z_MAX_MM = 800.0


class WateringConfig(BaseModel):
    cols: int = Field(default=8, ge=1, le=16)
    rows: int = Field(default=2, ge=1, le=8)
    gap_x_mm: float = Field(default=750.0, ge=0.0, le=X_MAX_MM)
    gap_y_mm: float = Field(default=1000.0, ge=0.0, le=Y_MAX_MM)
    start_x_mm: float = Field(default=0.0, ge=0.0, le=X_MAX_MM)
    start_y_mm: float = Field(default=0.0, ge=0.0, le=Y_MAX_MM)
    z_max_mm: float = Field(default=0.0, ge=0.0, le=Z_MAX_MM)       # Z raised to this for TOF sweep
    z_water_mm: float = Field(default=50.0, ge=0.0, le=Z_MAX_MM)    # Z working height during valve open
    tof_samples: int = Field(default=5, ge=1, le=50)                # TOF readings per plant position
    sweep_speed_mm_sec: float = Field(default=150.0, gt=0.0, le=5000.0)
    water_speed_mm_sec: float = Field(default=100.0, gt=0.0, le=5000.0)

    @model_validator(mode="after")
    def _within_travel(self) -> "WateringConfig":
        """Reject grids whose extreme column/row position exits the envelope."""
        far_x = self.start_x_mm + (self.cols - 1) * self.gap_x_mm
        far_y = self.start_y_mm + (self.rows - 1) * self.gap_y_mm
        if self.start_x_mm < 0 or far_x > X_MAX_MM:
            raise ValueError(
                f"watering X range [0,{far_x:.0f}]mm outside gantry travel [0,{X_MAX_MM:.0f}]"
            )
        if self.start_y_mm < 0 or far_y > Y_MAX_MM:
            raise ValueError(
                f"watering Y range [0,{far_y:.0f}]mm outside gantry travel [0,{Y_MAX_MM:.0f}]"
            )
        return self

    def col_x_mm(self, col: int) -> float:
        return self.start_x_mm + col * self.gap_x_mm

    def row_y_mm(self, row: int) -> float:
        return self.start_y_mm + row * self.gap_y_mm

    def center_y_mm(self) -> float:
        """Y midpoint between all rows — sprinkler arm covers the full width."""
        return self.start_y_mm + ((self.rows - 1) * self.gap_y_mm) / 2.0

    def plant_positions(self) -> list[tuple[int, int]]:
        """All (row, col) pairs for TOF sweep, row-major order."""
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
        ]
