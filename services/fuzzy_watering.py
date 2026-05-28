"""
Fuzzy watering controller — Mamdani rules, singleton (Takagi-Sugeno) outputs.

Inputs
  max_height_cm   : tallest plant height in the bed (0–100 cm)
  avg_moisture_pct: average soil moisture across all 3 sensors (0–100 %)

Output
  watering_duration_sec : how long to open the valve (seconds)

Membership functions (triangular):
  Height  — Low:    trimf(0,  0,  40)
           — Medium: trimf(20, 50, 80)
           — High:   trimf(60, 100, 100)
  Moisture— Dry:    trimf(0,  0,  40)
           — Moist:  trimf(20, 50, 80)
           — Wet:    trimf(60, 100, 100)

Rule table (AND = min operator):
  High   ∧ Dry   → 8 s
  High   ∧ Moist → 5 s
  High   ∧ Wet   → 2 s
  Medium ∧ Dry   → 5 s
  Medium ∧ Moist → 2 s
  Medium ∧ Wet   → 0 s
  Low    ∧ Dry   → 2 s
  Low    ∧ Moist → 0 s
  Low    ∧ Wet   → 0 s

Defuzzification: weighted average of singleton outputs
  duration = Σ(weight_i × singleton_i) / Σ(weight_i)
"""


def _trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular MF: rises a→b, falls b→c. Returns value in [0, 1]."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if (b - a) != 0 else 1.0
    return (c - x) / (c - b) if (c - b) != 0 else 1.0


def _height_mf(h: float) -> dict[str, float]:
    return {
        "Low":    _trimf(h,  0,   0,  40),
        "Medium": _trimf(h, 20,  50,  80),
        "High":   _trimf(h, 60, 100, 100),
    }


def _moisture_mf(m: float) -> dict[str, float]:
    return {
        "Dry":   _trimf(m,  0,   0,  40),
        "Moist": _trimf(m, 20,  50,  80),
        "Wet":   _trimf(m, 60, 100, 100),
    }


# (height_label, moisture_label) → singleton output in seconds
_RULES: dict[tuple[str, str], float] = {
    ("High",   "Dry"):   8.0,
    ("High",   "Moist"): 5.0,
    ("High",   "Wet"):   2.0,
    ("Medium", "Dry"):   5.0,
    ("Medium", "Moist"): 2.0,
    ("Medium", "Wet"):   0.0,
    ("Low",    "Dry"):   2.0,
    ("Low",    "Moist"): 0.0,
    ("Low",    "Wet"):   0.0,
}


def compute_watering_duration(
    max_height_cm: float,
    avg_moisture_pct: float,
) -> float:
    """
    Return valve-open duration in seconds, rounded to 2 decimal places.
    Falls back to 2.0 s if inputs fall entirely outside defined MF ranges.
    """
    h_mf = _height_mf(max_height_cm)
    m_mf = _moisture_mf(avg_moisture_pct)

    numerator = 0.0
    denominator = 0.0

    for (h_label, m_label), singleton in _RULES.items():
        weight = min(h_mf[h_label], m_mf[m_label])
        numerator += weight * singleton
        denominator += weight

    if denominator == 0.0:
        return 2.0  # safe default when inputs are out of range

    return round(numerator / denominator, 2)
