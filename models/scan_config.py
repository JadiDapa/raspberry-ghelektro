from pydantic import BaseModel, Field, model_validator

# Gantry working envelope (mm). Matches the MoveRequest bounds in routers/gantry.py
# and stays inside the firmware homing travel. A config that would command the
# gantry outside this box is rejected before any motion starts.
X_MAX_MM = 6000.0
Y_MAX_MM = 2000.0
Z_MAX_MM = 800.0


class CaptureOffset(BaseModel):
    z_mm: float = Field(default=50.0, ge=0.0, le=Z_MAX_MM)
    x_offset_mm: float = Field(default=0.0, ge=-500.0, le=500.0)
    y_offset_mm: float = Field(default=0.0, ge=-500.0, le=500.0)
    servo_pan: float = Field(default=90.0, ge=0.0, le=180.0)
    servo_tilt: float = Field(default=90.0, ge=0.0, le=180.0)


class ModelConfig(BaseModel):
    """A specific uploaded YOLO model this session should run.

    The dashboard bakes this into the scan config snapshot. `file_url` is the
    dashboard-relative weights URL (/api/uploads/<name>); the RPi prefixes its
    dashboard_url to download it, caches it by `checksum`, and runs inference
    with the settings below. When the scan config has no model, this is absent
    and the RPi falls back to its built-in yolo_model_path.
    """

    file_url: str
    checksum: str = ""
    name: str = ""
    imgsz: int = Field(default=640, ge=32, le=4096)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_nms: float = Field(default=0.7, ge=0.0, le=1.0)
    max_det: int = Field(default=300, ge=1, le=10000)
    class_names: list[str] = Field(default_factory=list)


class ScanConfig(BaseModel):
    cols: int = Field(default=8, ge=1, le=16)
    rows: int = Field(default=2, ge=1, le=8)
    gap_x_mm: float = Field(default=750.0, ge=0.0, le=X_MAX_MM)
    gap_y_mm: float = Field(default=1000.0, ge=0.0, le=Y_MAX_MM)
    start_x_mm: float = Field(default=0.0, ge=0.0, le=X_MAX_MM)
    start_y_mm: float = Field(default=0.0, ge=0.0, le=Y_MAX_MM)
    capture_offsets: list[CaptureOffset] = Field(
        default_factory=lambda: [CaptureOffset()],
        min_length=1,
    )

    # Region of interest for YOLO counting, as a centered box sized by percentage
    # of the frame. Detections whose box center falls outside this box are dropped
    # so neighboring plants visible at the frame edges don't inflate the count.
    # Default 100×100 = whole frame = no filtering (behaves as before).
    roi_w_pct: float = Field(default=100.0, gt=0.0, le=100.0)
    roi_h_pct: float = Field(default=100.0, gt=0.0, le=100.0)

    # Which uploaded model to run. None → use the built-in yolo_model_path.
    model: ModelConfig | None = None

    @model_validator(mode="after")
    def _within_travel(self) -> "ScanConfig":
        """Reject grids whose extreme position (incl. offsets) exits the envelope."""
        far_x = self.start_x_mm + (self.cols - 1) * self.gap_x_mm
        far_y = self.start_y_mm + (self.rows - 1) * self.gap_y_mm
        x_offsets = [o.x_offset_mm for o in self.capture_offsets]
        y_offsets = [o.y_offset_mm for o in self.capture_offsets]
        lo_x = self.start_x_mm + min(x_offsets, default=0.0)
        hi_x = far_x + max(x_offsets, default=0.0)
        lo_y = self.start_y_mm + min(y_offsets, default=0.0)
        hi_y = far_y + max(y_offsets, default=0.0)
        if lo_x < 0 or hi_x > X_MAX_MM:
            raise ValueError(
                f"scan X range [{lo_x:.0f},{hi_x:.0f}]mm outside gantry travel [0,{X_MAX_MM:.0f}]"
            )
        if lo_y < 0 or hi_y > Y_MAX_MM:
            raise ValueError(
                f"scan Y range [{lo_y:.0f},{hi_y:.0f}]mm outside gantry travel [0,{Y_MAX_MM:.0f}]"
            )
        return self

    def plant_grid(self) -> list[tuple[int, int]]:
        """Return ordered (row, col) pairs for all plants in this config."""
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
        ]

    def plant_position_mm(self, row: int, col: int) -> tuple[float, float]:
        """Return absolute (x_mm, y_mm) for a given grid position.

        The first plant (row 0, col 0) sits at (start_x_mm, start_y_mm); every
        other plant is stepped from there by the per-axis gap.
        """
        x = self.start_x_mm + col * self.gap_x_mm
        y = self.start_y_mm + row * self.gap_y_mm
        return x, y

    @property
    def offset(self) -> CaptureOffset:
        """The single active capture offset (first entry)."""
        return self.capture_offsets[0]
