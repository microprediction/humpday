"""
Seasonal crop irrigation scheduling under a water budget.

Over a twelve-week season we choose how much to irrigate each week, subject to a total
water budget. Soil moisture rises with irrigation and rain and falls with
evapotranspiration, bounded by the soil's holding capacity. When moisture drops below a
threshold the crop is water-stressed, and the yield lost depends on how sensitive that
week is (flowering weeks matter most). The good schedule concentrates water on the
sensitive weeks rather than spreading it evenly.

The HumpDay objective takes a 12-D point in [0,1]^12, normalises it to the water budget,
runs the soil-moisture model, and returns the negative yield (to minimise).
"""

from __future__ import annotations

WEEKS = 12
N_DIM = WEEKS

RAIN = (20, 15, 10, 8, 5, 4, 3, 5, 8, 12, 18, 22)  # mm/week
ET = (15, 18, 22, 28, 32, 35, 34, 30, 25, 20, 16, 14)  # evapotranspiration mm/week
SENS = (0.3, 0.4, 0.6, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2)  # yield sensitivity
BUDGET = 90.0  # total irrigation water (mm); below the seasonal deficit, so
# the schedule must prioritise the sensitive weeks
CAPACITY = 60.0  # soil holding capacity (mm)
STRESS_THRESHOLD = 30.0


def decode(u):
    s = sum(max(0.0, v) for v in u) or 1.0
    return [BUDGET * max(0.0, v) / s for v in u]


def objective(u):
    irr = decode(u)
    moisture = 40.0
    yield_factor = 1.0
    for w in range(WEEKS):
        moisture = min(CAPACITY, moisture + irr[w] + RAIN[w] - ET[w])
        moisture = max(0.0, moisture)
        stress = max(0.0, 1.0 - moisture / STRESS_THRESHOLD)
        yield_factor *= 1.0 - 0.5 * SENS[w] * stress
    return -yield_factor
