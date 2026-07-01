"""
Interval Type-2 (IT2) fuzzy watering controller — Sugeno singleton consequents,
Karnik-Mendel type reduction.

Inputs
  max_height_cm   : tallest plant height in the bed (cm)
  avg_moisture_pct: average soil moisture across all 3 sensors (%)

Output
  watering_duration_sec : how long to open the valve (seconds, 0–120)

Each linguistic set is an interval type-2 set defined by a Lower Membership
Function (LMF) and an Upper Membership Function (UMF), both triangular. A crisp
input therefore produces a firing *interval* [lower, upper] per rule, which is
collapsed back to a crisp number by Karnik-Mendel type reduction.

Plant height sets (growth stage) — trimf(a, b, c):
                    LMF (a,b,c)        UMF (a,b,c)
  Semai            (0,  0,  12)       (0,  0,  15)
  Vegetatif Awal   (8,  18, 30)       (5,  18, 33)
  Vegetatif Aktif  (22, 38, 55)       (19, 38, 58)
  Berbunga         (47, 60, 75)       (44, 60, 78)
  Produksi         (65, 78, 95)       (62, 78, 98)

Soil moisture sets — trimf(a, b, c):
                    LMF (a,b,c)        UMF (a,b,c)
  Sangat Kering    (0,  0,  27)       (0,  0,  30)
  Kering           (18, 30, 55)       (15, 30, 58)
  Optimal          (48, 65, 78)       (45, 65, 81)
  Basah            (70, 80, 92)       (67, 80, 95)
  Sangat Basah     (85, 93, 100)      (82, 93, 100)

Output singletons (Durasi Irigasi, seconds):
  Tidak = 0   Singkat = 30   Normal = 60   Lama = 90   Sangat Lama = 120

Rule base (AND = min operator over height ∧ moisture → duration in seconds):
                 SangatKering  Kering  Optimal  Basah  SangatBasah
  Semai               30         30       0       0         0
  Vegetatif Awal      60         30       0       0         0
  Vegetatif Aktif     90         60      30       0         0
  Berbunga           120         90      60      30         0
  Produksi           120         90      60      30         0

Type reduction: Karnik-Mendel (per rule the firing is [f_lo, f_hi]; KM finds the
left/right endpoints y_l, y_r of the type-reduced interval and the defuzzified
duration is their midpoint).
"""

# Watering duration used when no rule fires (inputs fall entirely outside every
# membership function, e.g. a glitched out-of-range sensor reading). Kept at 0 s
# so a bad reading never triggers watering it cannot justify.
_FALLBACK_DURATION_SEC = 0.0


def _trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular MF: rises a→b, falls b→c. Returns a value in [0, 1].

    Handles shoulders where a == b (left shoulder, e.g. Semai/Sangat Kering) or
    b == c (right shoulder): at the peak of such a shoulder the membership is 1,
    not 0.
    """
    if x <= a:
        return 1.0 if x == a == b else 0.0
    if x >= c:
        return 1.0 if x == c == b else 0.0
    if x <= b:  # a < x <= b, so b > a → denominator non-zero
        return (x - a) / (b - a)
    return (c - x) / (c - b)  # b < x < c, so c > b → denominator non-zero


# label → (LMF params, UMF params), each (a, b, c)
_HEIGHT_SETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "Semai":           ((0,  0,  12), (0,  0,  15)),
    "Vegetatif Awal":  ((8,  18, 30), (5,  18, 33)),
    "Vegetatif Aktif": ((22, 38, 55), (19, 38, 58)),
    "Berbunga":        ((47, 60, 75), (44, 60, 78)),
    "Produksi":        ((65, 78, 95), (62, 78, 98)),
}

_MOISTURE_SETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "Sangat Kering": ((0,  0,  27),  (0,  0,  30)),
    "Kering":        ((18, 30, 55),  (15, 30, 58)),
    "Optimal":       ((48, 65, 78),  (45, 65, 81)),
    "Basah":         ((70, 80, 92),  (67, 80, 95)),
    "Sangat Basah":  ((85, 93, 100), (82, 93, 100)),
}


def _it2_mf(
    x: float,
    sets: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> dict[str, tuple[float, float]]:
    """For each label return (lower, upper) membership of x. lower ≤ upper."""
    out: dict[str, tuple[float, float]] = {}
    for label, (lmf, umf) in sets.items():
        lo = _trimf(x, *lmf)
        hi = _trimf(x, *umf)
        # LMF must never exceed UMF; clamp against numeric edge cases.
        out[label] = (min(lo, hi), hi)
    return out


# (height_label, moisture_label) → singleton output duration in seconds.
_RULES: dict[tuple[str, str], float] = {
    ("Semai",           "Sangat Kering"): 30.0,
    ("Semai",           "Kering"):        30.0,
    ("Semai",           "Optimal"):        0.0,
    ("Semai",           "Basah"):          0.0,
    ("Semai",           "Sangat Basah"):   0.0,
    ("Vegetatif Awal",  "Sangat Kering"): 60.0,
    ("Vegetatif Awal",  "Kering"):        30.0,
    ("Vegetatif Awal",  "Optimal"):        0.0,
    ("Vegetatif Awal",  "Basah"):          0.0,
    ("Vegetatif Awal",  "Sangat Basah"):   0.0,
    ("Vegetatif Aktif", "Sangat Kering"): 90.0,
    ("Vegetatif Aktif", "Kering"):        60.0,
    ("Vegetatif Aktif", "Optimal"):       30.0,
    ("Vegetatif Aktif", "Basah"):          0.0,
    ("Vegetatif Aktif", "Sangat Basah"):   0.0,
    ("Berbunga",        "Sangat Kering"): 120.0,
    ("Berbunga",        "Kering"):        90.0,
    ("Berbunga",        "Optimal"):       60.0,
    ("Berbunga",        "Basah"):         30.0,
    ("Berbunga",        "Sangat Basah"):   0.0,
    ("Produksi",        "Sangat Kering"): 120.0,
    ("Produksi",        "Kering"):        90.0,
    ("Produksi",        "Optimal"):       60.0,
    ("Produksi",        "Basah"):         30.0,
    ("Produksi",        "Sangat Basah"):   0.0,
}


def _weighted_avg(cs: list[float], fs: list[float]) -> float:
    denom = sum(fs)
    if denom == 0.0:
        return 0.0
    return sum(c * f for c, f in zip(cs, fs)) / denom


def _switch_point(cs: list[float], y: float) -> int:
    """Index k with cs[k] <= y <= cs[k+1]; clamped to [0, n-2]. cs sorted asc."""
    n = len(cs)
    if y <= cs[0]:
        return 0
    if y >= cs[-1]:
        return n - 2
    for k in range(n - 1):
        if cs[k] <= y <= cs[k + 1]:
            return k
    return n - 2


def _km_endpoint(cs: list[float], flo: list[float], fhi: list[float], left: bool) -> float:
    """
    Karnik-Mendel endpoint. `left=True` returns y_l (minimum of the type-reduced
    interval), `left=False` returns y_r (maximum). cs must be sorted ascending
    and aligned with flo/fhi.

    For y_l the upper firing is applied to the low-consequent side (i <= k) to
    minimise the average; for y_r the upper firing is applied to the
    high-consequent side (i > k) to maximise it.
    """
    n = len(cs)
    f = [(flo[i] + fhi[i]) / 2.0 for i in range(n)]
    y = _weighted_avg(cs, f)
    k_prev = -1
    for _ in range(1000):  # KM converges in a handful of steps; bound for safety
        k = _switch_point(cs, y)
        if k == k_prev:
            break
        for i in range(n):
            if left:
                f[i] = fhi[i] if i <= k else flo[i]
            else:
                f[i] = flo[i] if i <= k else fhi[i]
        y = _weighted_avg(cs, f)
        k_prev = k
    return y


def compute_watering_duration(
    max_height_cm: float,
    avg_moisture_pct: float,
) -> float:
    """
    Return valve-open duration in seconds (0–120), rounded to 2 decimal places.
    Falls back to `_FALLBACK_DURATION_SEC` when no rule fires (inputs entirely
    outside every membership function).
    """
    h_mf = _it2_mf(max_height_cm, _HEIGHT_SETS)
    m_mf = _it2_mf(avg_moisture_pct, _MOISTURE_SETS)

    # Aggregate firing intervals by consequent value. Rules that share an output
    # singleton are summed together — this is exact for KM (equal-consequent
    # points are always assigned the same firing endpoint at the optimum) and
    # keeps the type reducer working over just the distinct output levels.
    agg_lo: dict[float, float] = {}
    agg_hi: dict[float, float] = {}
    for (h_label, m_label), singleton in _RULES.items():
        h_lo, h_hi = h_mf[h_label]
        m_lo, m_hi = m_mf[m_label]
        agg_lo[singleton] = agg_lo.get(singleton, 0.0) + min(h_lo, m_lo)
        agg_hi[singleton] = agg_hi.get(singleton, 0.0) + min(h_hi, m_hi)

    if sum(agg_hi.values()) == 0.0:
        return _FALLBACK_DURATION_SEC

    cs = sorted(agg_lo.keys())
    flo = [agg_lo[c] for c in cs]
    fhi = [agg_hi[c] for c in cs]

    y_l = _km_endpoint(cs, flo, fhi, left=True)
    y_r = _km_endpoint(cs, flo, fhi, left=False)

    return round((y_l + y_r) / 2.0, 2)
