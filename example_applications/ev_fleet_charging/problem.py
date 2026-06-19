"""
EV fleet charging schedule with deadlines and a grid cap.

A depot charges its electric fleet over 24 hours. Each hour we choose the aggregate
charging power, bounded by the site's grid connection. The fleet must receive enough
energy by noon (the vehicles that leave on the morning shift) and a larger total by end
of day, while electricity price varies. The good schedule front-loads just enough cheap
overnight energy to clear the noon deadline and tops up the rest in the cheapest
remaining hours.

The HumpDay objective takes a 24-D point in [0,1]^24 (hourly charging power, mapped to
0..cap) and returns price-weighted energy cost plus penalties for missing the noon and
end-of-day energy requirements.
"""
from __future__ import annotations

HOURS = 24
N_DIM = HOURS

PRICE = (0.30, 0.25, 0.22, 0.20, 0.22, 0.30, 0.45, 0.60, 0.70, 0.65, 0.60, 0.58,
         0.55, 0.55, 0.60, 0.70, 0.85, 1.00, 0.95, 0.80, 0.65, 0.50, 0.40, 0.35)
CAP = 10.0              # grid connection limit (per hour)
NEED_BY_NOON = 45.0     # energy required by hour 12
NEED_TOTAL = 110.0      # energy required by end of day


def decode(u):
    return [CAP * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    p = decode(u)
    cost = sum(p[h] * PRICE[h] for h in range(HOURS))
    delivered_noon = sum(p[:12])
    delivered_total = sum(p)
    penalty = 50.0 * max(0.0, NEED_BY_NOON - delivered_noon)
    penalty += 50.0 * max(0.0, NEED_TOTAL - delivered_total)
    return cost + penalty
