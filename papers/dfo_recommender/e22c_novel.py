"""E22c — Novel-family + widened holdout for the forecasting star.

Four series families that appeared in NO selection round (the untouched-
problems analog), seeds 30-34, plus the original six families at virgin
seeds 25-29. Star vs laplace paired per instance.
"""

import math
import random
import sys
import zlib
from math import comb

sys.path.insert(0, ".")
import forecast_sim as fs  # noqa: E402
from e22_forecast_simplex import compile_forecaster  # noqa: E402


def series_seasonal_break(rng, n):
    out = []
    while len(out) < n:
        period = rng.randint(5, 40)
        amp = rng.uniform(0.5, 6)
        s = rng.uniform(0.3, 1.5)
        base = rng.uniform(-5, 5)
        for i in range(rng.randint(300, 800)):
            out.append(
                base + amp * math.sin(2 * math.pi * i / period) + rng.gauss(0, s)
            )
    return out[:n]


def series_multiplicative(rng, n):
    y, out = rng.uniform(20, 100), []
    mu = rng.uniform(-0.001, 0.002)
    s = rng.uniform(0.005, 0.03)
    for _ in range(n):
        y *= math.exp(mu + rng.gauss(0, s))
        out.append(y)
    return out


def series_quantized(rng, n):
    grid = rng.choice([0.25, 0.5, 1.0])
    y, out = 0.0, []
    s = rng.uniform(0.3, 1.5)
    for _ in range(n):
        y += rng.gauss(0, s)
        out.append(round(y / grid) * grid)
    return out


def series_ou(rng, n):
    theta = rng.uniform(0.02, 0.2)
    mu = rng.uniform(-10, 10)
    s = rng.uniform(0.5, 2.0)
    y, out = mu, []
    for _ in range(n):
        y += theta * (mu - y) + rng.gauss(0, s)
        out.append(y)
    return out


NOVEL = {
    "seasonal_break": series_seasonal_break,
    "multiplicative": series_multiplicative,
    "quantized": series_quantized,
    "ou": series_ou,
}


def make_novel(family, seed):
    rng = random.Random(zlib.crc32(f"novel:{family}:{seed}".encode()))
    raw = NOVEL[family](rng, fs.SERIES_LEN)
    scale = rng.uniform(0.2, 50.0) * rng.choice([-1.0, 1.0])
    offset = rng.uniform(-1000.0, 1000.0)
    return [scale * y + offset for y in raw]


star = compile_forecaster(open("runs/e22_adaptive_code/pure_RegimeWatcher.py").read())

groups = {
    "novel families (seeds 30-34)": [
        make_novel(f, s) for f in NOVEL for s in range(30, 35)
    ],
    "original families (seeds 25-29)": [
        fs.make_instance(f, s) for f in fs.FAMILIES for s in range(25, 30)
    ],
}

grand_w = grand_l = 0
for gname, insts in groups.items():
    w = l = 0
    sd = ld = 0.0
    for series in insts:
        a = fs.mean_nll(star, series)
        b = fs.mean_nll(fs.LaplaceRef, series)
        sd += a
        ld += b
        if a < b:
            w += 1
        elif a > b:
            l += 1
    n = w + l
    k = min(w, l)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2**n)
    print(
        f"{gname}: star {w}/{n} = {round(100 * w / n)}pct p={p:.1e} "
        f"(mean NLL star {sd / len(insts):.3f} vs laplace {ld / len(insts):.3f})"
    )
    grand_w += w
    grand_l += l

n = grand_w + grand_l
k = min(grand_w, grand_l)
p = min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2**n)
print(
    f"\nALL fresh instances incl. earlier holdout basis: star {grand_w}/{n} "
    f"= {round(100 * grand_w / n)}pct p={p:.1e}"
)
