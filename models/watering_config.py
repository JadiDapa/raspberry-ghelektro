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
    # Physical calibration: TOF-sensor-to-floor distance with the head fully
    # raised (Z=0). Plant height = this reference minus the live TOF reading.
    # This is a rig constant, NOT a gantry coordinate — it is unrelated to
    # z_max_mm and can exceed the Z travel envelope.
    tof_floor_ref_mm: float = Field(default=1200.0, gt=0.0, le=5000.0)
    tof_samples: int = Field(default=5, ge=1, le=50)                # legacy (stop-and-scan); kept for payload compat
    tof_sample_hz: float = Field(default=5.0, gt=0.0, le=50.0)      # TOF polls/sec during the continuous sweep
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

    def height_cm(self, tof_cm: float) -> float:
        """Plant height (cm) from a TOF reading taken during the sweep.

        The reference is the sensor→floor distance at the sweep height: with the
        head fully raised the sensor is `tof_floor_ref_mm` above the floor, and
        lowering it to the sweep coordinate `z_max_mm` brings it that much
        closer, so the effective reference is their difference. A taller plant
        returns a smaller TOF reading, hence a larger height.
        """
        ref_cm = (self.tof_floor_ref_mm - self.z_max_mm) / 10.0
        return round(ref_cm - tof_cm, 1)

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

    def sweep_segments(self) -> list[tuple[int, float, float, float]]:
        """Continuous serpentine height-sweep segments, one per row.

        Returns (row, x_start, x_end, y) in travel order. Even rows sweep
        col 0 → last, odd rows last → col 0, so consecutive rows connect with a
        short Y step instead of a long return — one uninterrupted pass over all
        plants while the TOF is polled the whole way.
        """
        first_x = self.col_x_mm(0)
        last_x = self.col_x_mm(self.cols - 1)
        segments: list[tuple[int, float, float, float]] = []
        for row in range(self.rows):
            y = self.row_y_mm(row)
            if row % 2 == 0:
                segments.append((row, first_x, last_x, y))
            else:
                segments.append((row, last_x, first_x, y))
        return segments

    def nearest_col(self, x_mm: float) -> int:
        """Column whose center is closest to `x_mm` — buckets a mid-sweep sample."""
        if self.gap_x_mm <= 0:
            return 0
        col = round((x_mm - self.start_x_mm) / self.gap_x_mm)
        return max(0, min(self.cols - 1, int(col)))
