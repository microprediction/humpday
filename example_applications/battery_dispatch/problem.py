"""
Battery-dispatch objective: arbitrage a day of electricity prices.

A pure-Python grid-battery model. Over 24 hours a battery can charge (buy) or
discharge (sell) up to a power limit, bounded by its state-of-charge and with
round-trip efficiency losses. The HumpDay objective takes a 24-D point in
[0,1]^24 — the hourly charge/discharge power — clips each hour to what the
state-of-charge and power limits allow, and returns the negative **revenue**
($/day) earned against a fixed price curve.

This is an expected-cost / dispatch-optimisation problem with linear dynamics and
operational limits: buy low overnight, sell into the evening peak, but never
overfill or drain the pack, and pay the efficiency tax on every cycle. The naive
one-cycle preset earns ~$13.7k; a good schedule does better by exploiting the
midday dip and the sharp evening peak.

Mirrors the browser demo docs/applications/battery-dispatch.html.
"""

from __future__ import annotations

import math

HOURS = 24
N_DIM = HOURS
P_MAX = 25.0  # MW charge/discharge limit
E_MAX = 100.0  # MWh nameplate
SOC_LO = 10.0  # MWh
SOC_HI = 90.0  # MWh
SOC_INIT = 50.0  # MWh
ETA_C = 0.92
ETA_D = 0.92
PRICE = (
    30,
    25,
    22,
    20,
    22,
    35,
    55,
    75,
    90,
    100,
    90,
    75,
    55,
    40,
    38,
    45,
    80,
    150,
    240,
    220,
    160,
    100,
    60,
    40,
)


def decode(u):
    return [(u[h] - 0.5) * 2 * P_MAX for h in range(HOURS)]


def simulate(schedule):
    """Return (revenue, final_soc, cycles) for an hourly power schedule (MW)."""
    soc = SOC_INIT
    revenue = 0.0
    throughput = 0.0
    for h in range(HOURS):
        p = schedule[h]
        if p < 0:  # charging
            max_charge = (SOC_HI - soc) / ETA_C
            if -p > max_charge:
                p = -max_charge
        elif p > 0:  # discharging
            max_discharge = (soc - SOC_LO) * ETA_D
            if p > max_discharge:
                p = max_discharge
        revenue += p * PRICE[h]  # $/h = MW · $/MWh
        if p < 0:
            soc += -p * ETA_C
        else:
            soc -= p / ETA_D
        throughput += abs(p)
    return revenue, soc, throughput / (2 * E_MAX)


def evaluate_schedule(u):
    """Return (revenue, final_soc, cycles) for a schedule in [0,1]^24."""
    return simulate(decode(u))


def objective(u):
    """HumpDay objective: negative daily revenue (minimise)."""
    return -simulate(decode(u))[0]


# --- Faithful high-dimensional variants -------------------------------------
# Scaling knob: horizon length in hours = n_dim (must be a multiple of 24, i.e.
# whole days). The battery dynamics, power/SOC limits and round-trip efficiency
# are identical; the state-of-charge is carried across the full multi-day
# horizon, and each day's price curve is the base 24h shape scaled by a smooth
# +/-10% day-to-day factor (so successive days differ as real markets do rather
# than being trivial repeats). It is the same arbitrage problem over a longer
# horizon — strictly faithful, just higher-dimensional.
SCALABLE_DIMS = [48, 72, 96]


def make_objective(n_dim):
    """Return a HumpDay objective over `n_dim` hours (n_dim // 24 days)."""
    hours = int(n_dim)
    days = hours // 24
    price = []
    for d in range(days):
        mult = 1.0 + 0.10 * math.sin(2 * math.pi * d / max(days, 1))
        price.extend(p * mult for p in PRICE)

    def objective_scaled(u):
        soc = SOC_INIT
        revenue = 0.0
        for h in range(hours):
            p = (u[h] - 0.5) * 2 * P_MAX
            if p < 0:  # charging
                max_charge = (SOC_HI - soc) / ETA_C
                if -p > max_charge:
                    p = -max_charge
            elif p > 0:  # discharging
                max_discharge = (soc - SOC_LO) * ETA_D
                if p > max_discharge:
                    p = max_discharge
            revenue += p * price[h]
            if p < 0:
                soc += -p * ETA_C
            else:
                soc -= p / ETA_D
        return -revenue

    return objective_scaled
