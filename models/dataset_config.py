from pydantic import BaseModel, Field, model_validator

# Gantry working envelope (mm) — see models/scan_config.py.
X_MAX_MM = 6000.0
Y_MAX_MM = 2000.0
Z_MAX_MM = 800.0


class DatasetConfig(BaseModel):
    """Per-session config for a Data Collection sweep.

    A dataset session does NOT stop at each plant — it sweeps the bed in one
    continuous serpentine pass at `speed_mm_sec` while recording a video. The
    grid fields share the same meaning/defaults as ScanConfig so coordinates
    line up with the rest of the system; `z_mm` is the single height held for
    the whole pass and `speed_mm_sec` controls how fast the gantry travels.
    """

    cols: int = Field(default=8, ge=1, le=16)
    rows: int = Field(default=2, ge=1, le=8)
    gap_x_mm: float = Field(default=750.0, ge=0.0, le=X_MAX_MM)
    gap_y_mm: float = Field(default=1000.0, ge=0.0, le=Y_MAX_MM)
    start_x_mm: float = Field(default=0.0, ge=0.0, le=X_MAX_MM)
    start_y_mm: float = Field(default=0.0, ge=0.0, le=Y_MAX_MM)
    z_mm: float = Field(default=50.0, ge=0.0, le=Z_MAX_MM)  # held constant during the sweep
    speed_mm_sec: float = Field(default=100.0, gt=0.0, le=5000.0)  # gantry travel speed

    @model_validator(mode="after")
    def _within_travel(self) -> "DatasetConfig":
        """Reject grids whose extreme column/row position exits the envelope."""
        far_x = self.start_x_mm + (self.cols - 1) * self.gap_x_mm
        far_y = self.start_y_mm + (self.rows - 1) * self.gap_y_mm
        if self.start_x_mm < 0 or far_x > X_MAX_MM:
            raise ValueError(
                f"dataset X range [0,{far_x:.0f}]mm outside gantry travel [0,{X_MAX_MM:.0f}]"
            )
        if self.start_y_mm < 0 or far_y > Y_MAX_MM:
            raise ValueError(
                f"dataset Y range [0,{far_y:.0f}]mm outside gantry travel [0,{Y_MAX_MM:.0f}]"
            )
        return self

    def col_x_mm(self, col: int) -> float:
        return self.start_x_mm + col * self.gap_x_mm

    def row_y_mm(self, row: int) -> float:
        return self.start_y_mm + row * self.gap_y_mm

    def serpentine_waypoints(self) -> list[tuple[int, float, float]]:
        """Row-corner endpoints for a continuous boustrophedon sweep.

        Each row contributes its two X extremes (the plants in between are passed
        over without stopping). Even rows go left→right, odd rows right→left, so
        consecutive rows connect with a short Y step instead of a long return.

        Returns a list of (row, x_mm, y_mm) in travel order.
        """
        first_col = 0
        last_col = self.cols - 1
        waypoints: list[tuple[int, float, float]] = []
        for row in range(self.rows):
            y = self.row_y_mm(row)
            if row % 2 == 0:
                cols = (first_col, last_col)
            else:
                cols = (last_col, first_col)
            for col in cols:
                waypoints.append((row, self.col_x_mm(col), y))
        return waypoints
