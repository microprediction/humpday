"""
Rocket-landing objective: land a booster softly with fuel to spare.

A pure-Python 1-D vertical landing sim. A booster falls under gravity with a fuel
tank good for ~4 seconds of full burn. The HumpDay objective takes a 12-D point
in [0,1]^12 — a piecewise-constant throttle schedule — simulates the descent, and
scores a soft touchdown (a perfect v ≈ 0 scores 100, falling to 0 at v = 80 px/s)
plus up to +10 for fuel left in the tank. It returns the negative score.

The landscape has several distinct local optima: a gradual descent that runs the
tank dry before touchdown, versus a late "suicide burn" that waits and then dumps
thrust. Burning too early wastes fuel and the rocket falls the rest of the way;
burning too late and it slams in. A schedule that never reaches the ground gets
partial credit for being low and slow.

Mirrors the browser demo docs/applications/rocket-landing.html.
"""

from __future__ import annotations

N_SEGS = 12
N_DIM = N_SEGS
GRAVITY = 10.0
THRUST_MAX = 48.0  # hover throttle ≈ 10/48 ≈ 0.21
FUEL_BURN = 0.25  # 1.0 fuel = 4 s of full burn
H0 = 200.0
V0 = -15.0  # downward
SIM_T = 10.0
DT = 0.04
N_STEPS = round(SIM_T / DT)
V_LIMIT = 80.0  # touchdown speed where landing score hits 0


def decode(u):
    return [max(0.0, min(1.0, v)) for v in u]


def simulate(schedule):
    """Return (touch_v, touch_fuel, final_h, final_v); touch_v is None if airborne."""
    h, v, fuel = H0, V0, 1.0
    touch_v = touch_fuel = None
    final_h, final_v = h, v
    for k in range(N_STEPS):
        t = k * DT
        seg = min(N_SEGS - 1, int((t / SIM_T) * N_SEGS))
        throttle = schedule[seg]
        if fuel <= 0:
            throttle = 0.0
        acc = -GRAVITY + THRUST_MAX * throttle
        new_h = h + v * DT + 0.5 * acc * DT * DT
        new_v = v + acc * DT
        fuel = max(0.0, fuel - FUEL_BURN * throttle * DT)
        if new_h <= 0 and touch_v is None:
            frac = h / (h - new_h)
            touch_v = v + acc * DT * frac
            touch_fuel = fuel
            final_h, final_v = 0.0, 0.0
            break
        h, v = new_h, new_v
        final_h, final_v = h, v
    return touch_v, touch_fuel, final_h, final_v


def evaluate_schedule(u):
    """Return (score, landed, touchdown_speed) for a throttle schedule in [0,1]^12."""
    touch_v, touch_fuel, final_h, final_v = simulate(decode(u))
    if touch_v is not None:
        speed = abs(touch_v)
        score = max(0.0, 100 * (1 - speed / V_LIMIT)) + touch_fuel * 10
        return score, True, speed
    eff = abs(final_v) + final_h * 0.5
    return 50 - eff * 0.6, False, abs(final_v)


def objective(u):
    """HumpDay objective: negative landing score (minimise)."""
    return -evaluate_schedule(u)[0]
