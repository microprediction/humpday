"""Distributional-forecasting benchmark for the third inspiration-simplex
example.

Series come from six seeded generator families; the disguise analog
randomizes each family's parameters and applies a seeded affine map (scale,
offset, optional sign flip), so a generated forecaster cannot succeed by
memorising levels or magic constants.

Contract for candidates:

    class Forecaster:
        def __init__(self): ...
        def logpdf(self, y) -> float   # log density of the NEXT value under
                                       # the current one-step predictive
        def update(self, y) -> None    # then observe it

Score: mean negative log-likelihood per tick after a warmup, normalised as
regret against the panel on the same series (0 best panel member, 1 worst).
The skaters package's `laplace` is the reference champion, reported but not
in the panel.
"""

from __future__ import annotations

import math
import random
import sys

sys.path.insert(0, "/Users/petercotton/github/skaters/src")

WARMUP = 100
LOGPDF_FLOOR = -30.0  # one absurd density shouldn't dominate a series
LOGPDF_CAP = 8.0  # spikes on quantized data shouldn't either: the game is
# ill-posed for continuous densities at exact repeats, so
# per-tick log-density is capped symmetrically


# ---------------------------------------------------------------- panel
class RWGauss:
    """Random walk: predict N(last, EWMA scale of diffs)."""

    def __init__(self):
        self.last = 0.0
        self.var = 1.0
        self.n = 0

    def logpdf(self, y):
        if self.n == 0:
            return LOGPDF_FLOOR
        v = max(self.var, 1e-12)
        return -0.5 * math.log(2 * math.pi * v) - 0.5 * (y - self.last) ** 2 / v

    def update(self, y):
        if self.n:
            d = y - self.last
            self.var = 0.95 * self.var + 0.05 * d * d if self.n > 1 else d * d + 1e-8
        self.last = y
        self.n += 1


class LevelEWMA:
    """Exponentially smoothed level with EWMA residual scale."""

    def __init__(self, alpha=0.2):
        self.a = alpha
        self.level = 0.0
        self.var = 1.0
        self.n = 0

    def logpdf(self, y):
        if self.n == 0:
            return LOGPDF_FLOOR
        v = max(self.var, 1e-12)
        return -0.5 * math.log(2 * math.pi * v) - 0.5 * (y - self.level) ** 2 / v

    def update(self, y):
        if self.n == 0:
            self.level = y
            self.var = 1e-6
        else:
            r = y - self.level
            self.var = 0.95 * self.var + 0.05 * r * r
            self.level += self.a * r
        self.n += 1


class AR1Gauss:
    """Online AR(1) on diffs from a slow level, Gaussian errors."""

    def __init__(self):
        self.prev = None
        self.prev2 = None
        self.cov = 0.0
        self.var_x = 1e-8
        self.var_e = 1.0
        self.mean = 0.0
        self.n = 0

    def _phi(self):
        return max(-0.98, min(0.98, self.cov / max(self.var_x, 1e-12)))

    def logpdf(self, y):
        if self.n < 2:
            return LOGPDF_FLOOR
        pred = self.prev + self._phi() * (self.prev - self.prev2)
        v = max(self.var_e, 1e-12)
        return -0.5 * math.log(2 * math.pi * v) - 0.5 * (y - pred) ** 2 / v

    def update(self, y):
        if self.n >= 2:
            dx = self.prev - self.prev2
            dy = y - self.prev
            self.cov = 0.97 * self.cov + 0.03 * dx * dy
            self.var_x = 0.97 * self.var_x + 0.03 * dx * dx
            pred = self.prev + self._phi() * dx
            e = y - pred
            self.var_e = 0.97 * self.var_e + 0.03 * e * e
        elif self.n == 1:
            self.var_e = (y - self.prev) ** 2 + 1e-8
        self.prev2 = self.prev
        self.prev = y
        self.n += 1


class GarchLite:
    """Random-walk mean with GARCH(1,1)-style conditional variance."""

    def __init__(self, a=0.08, b=0.9):
        self.a = a
        self.b = b
        self.last = 0.0
        self.uncond = 1.0
        self.h = 1.0
        self.n = 0

    def logpdf(self, y):
        if self.n == 0:
            return LOGPDF_FLOOR
        v = max(self.h, 1e-12)
        return -0.5 * math.log(2 * math.pi * v) - 0.5 * (y - self.last) ** 2 / v

    def update(self, y):
        if self.n:
            e2 = (y - self.last) ** 2
            self.uncond = 0.99 * self.uncond + 0.01 * e2 if self.n > 1 else e2 + 1e-8
            omega = (1 - self.a - self.b) * self.uncond
            self.h = omega + self.a * e2 + self.b * self.h if self.n > 1 else e2 + 1e-8
        self.last = y
        self.n += 1


class TWalk:
    """Random walk with Student-t(4) innovations, EWMA scale."""

    NU = 4.0

    def __init__(self):
        self.last = 0.0
        self.var = 1.0
        self.n = 0
        nu = self.NU
        self.log_c = (
            math.lgamma((nu + 1) / 2)
            - math.lgamma(nu / 2)
            - 0.5 * math.log(nu * math.pi)
        )

    def logpdf(self, y):
        if self.n == 0:
            return LOGPDF_FLOOR
        nu = self.NU
        s = math.sqrt(max(self.var, 1e-12) * (nu - 2) / nu)
        z = (y - self.last) / s
        return self.log_c - math.log(s) - (nu + 1) / 2 * math.log(1 + z * z / nu)

    def update(self, y):
        if self.n:
            d = y - self.last
            self.var = 0.95 * self.var + 0.05 * d * d if self.n > 1 else d * d + 1e-8
        self.last = y
        self.n += 1


PANEL = {
    "RWGauss": RWGauss,
    "LevelEWMA": LevelEWMA,
    "AR1Gauss": AR1Gauss,
    "GarchLite": GarchLite,
    "TWalk": TWalk,
}


class LaplaceRef:
    """Adapter: skaters laplace under the Forecaster contract."""

    def __init__(self):
        from skaters.api import laplace

        self.f = laplace(k=1)
        self.state = None
        self.pending = None

    def logpdf(self, y):
        if self.pending is None:
            return LOGPDF_FLOOR
        try:
            return max(LOGPDF_FLOOR, self.pending.logpdf(y))
        except Exception:  # noqa: BLE001
            return LOGPDF_FLOOR

    def update(self, y):
        dists, self.state = self.f(y, self.state)
        self.pending = dists[0]


# ---------------------------------------------------------------- series
def series_rw_drift(rng, n):
    mu = rng.uniform(-0.05, 0.05)
    s = rng.uniform(0.5, 2.0)
    y, out = 0.0, []
    for _ in range(n):
        y += mu + rng.gauss(0, s)
        out.append(y)
    return out


def series_trend_seasonal(rng, n):
    period = rng.randint(7, 30)
    amp = rng.uniform(1, 5)
    slope = rng.uniform(-0.05, 0.05)
    s = rng.uniform(0.3, 1.5)
    return [
        slope * i + amp * math.sin(2 * math.pi * i / period) + rng.gauss(0, s)
        for i in range(n)
    ]


def series_regime_var(rng, n):
    out, y = [], 0.0
    while len(out) < n:
        s = rng.choice([0.3, 1.0, 3.0]) * rng.uniform(0.7, 1.4)
        for _ in range(rng.randint(150, 500)):
            y += rng.gauss(0, s)
            out.append(y)
    return out[:n]


def series_garch(rng, n):
    a = rng.uniform(0.05, 0.15)
    b = rng.uniform(0.78, 0.9)
    if a + b > 0.96:  # keep the process stationary; preserves the RNG stream
        b = 0.96 - a
    omega = (1 - a - b) * rng.uniform(0.5, 2.0)
    h, out, y = 1.0, [], 0.0
    for _ in range(n):
        e = math.sqrt(max(h, 1e-10)) * rng.gauss(0, 1)
        y += e
        out.append(y)
        h = omega + a * e * e + b * h
    return out


def series_ar2(rng, n):
    p1 = rng.uniform(0.3, 1.2)
    p2 = rng.uniform(-0.5, min(0.3, 0.95 - p1))
    s = rng.uniform(0.5, 2.0)
    x1 = x2 = 0.0
    out = []
    for _ in range(n):
        x = p1 * x1 + p2 * x2 + rng.gauss(0, s)
        out.append(x)
        x2, x1 = x1, x
    return out


def series_heavy(rng, n):
    s = rng.uniform(0.5, 2.0)
    y, out = 0.0, []
    for _ in range(n):
        u = rng.gauss(0, 1)
        w = rng.gauss(0, 1) ** 2 + rng.gauss(0, 1) ** 2 + rng.gauss(0, 1) ** 2
        y += s * u / math.sqrt(w / 3)
        out.append(y)
    return out


FAMILIES = {
    "rw_drift": series_rw_drift,
    "trend_seasonal": series_trend_seasonal,
    "regime_var": series_regime_var,
    "garch": series_garch,
    "ar2": series_ar2,
    "heavy": series_heavy,
}

SERIES_LEN = 1500


def make_instance(family, seed):
    import zlib

    rng = random.Random(zlib.crc32(f"{family}:{seed}".encode()))
    raw = FAMILIES[family](rng, SERIES_LEN)
    scale = rng.uniform(0.2, 50.0) * rng.choice([-1.0, 1.0])
    offset = rng.uniform(-1000.0, 1000.0)
    return [scale * y + offset for y in raw]


def mean_nll(forecaster_cls, series):
    f = forecaster_cls()
    total, count = 0.0, 0
    for i, y in enumerate(series):
        if i >= WARMUP:
            total += -min(LOGPDF_CAP, max(LOGPDF_FLOOR, f.logpdf(y)))
            count += 1
        f.update(y)
    return total / count


def build_panel_cache(instances):
    return [[mean_nll(c, s) for c in PANEL.values()] for s in instances]


def score_forecaster(cls, instances, panel_cache=None):
    regrets = []
    for i, series in enumerate(instances):
        panel = (
            panel_cache[i]
            if panel_cache is not None
            else [mean_nll(c, series) for c in PANEL.values()]
        )
        try:
            cand = mean_nll(cls, series)
        except Exception:  # noqa: BLE001
            regrets.append(1.0)
            continue
        vals = [cand] + panel
        mn, mx = min(vals), max(vals)
        regrets.append(0.0 if mx <= mn else (cand - mn) / (mx - mn))
    return sum(regrets) / len(regrets)


if __name__ == "__main__":
    insts = [make_instance(f, s) for f in FAMILIES for s in (0, 1)]
    cache = build_panel_cache(insts)
    for name, cls in list(PANEL.items()) + [("laplace", LaplaceRef)]:
        print(f"{name:10s} regret {score_forecaster(cls, insts, cache):.4f}")
