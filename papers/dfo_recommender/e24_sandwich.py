"""E24 — The star inside skaters' machinery, and beside laplace.

Arms, on virgin seeds 35-39 across all ten families (50 fresh instances):
  star      : the generated forecaster, native Student-t density
  laplace   : the reference champion
  mixture   : decayed log-score softmax mixture of star and laplace
              predictive densities (marginal-contribution test)
  sandwich  : the star as a Gaussian-body skater wrapped in skaters'
              gpdtails (conform-last supplies the tails the body drops)
"""

import math
import sys
from math import comb

sys.path.insert(0, ".")
import forecast_sim as fs  # noqa: E402
from e22_forecast_simplex import compile_forecaster  # noqa: E402
from e22c_novel import NOVEL, make_novel  # noqa: E402

sys.path.insert(0, "/Users/petercotton/github/skaters/src")
from skaters.dist import Dist  # noqa: E402
from skaters.tails import gpdtails  # noqa: E402

STAR_CODE = open("runs/e22_adaptive_code/pure_RegimeWatcher.py").read()
StarCls = compile_forecaster(STAR_CODE)


def star_skater_factory():
    """The star under the skater convention, Gaussian body from (mu, var)."""

    def skater(y, state):
        f = state["f"] if state else StarCls()
        f.update(y)
        pr = f._predict()
        d = Dist.gaussian(pr["mu"], math.sqrt(max(pr["var"], 1e-12)))
        return [d], {"f": f}

    return skater


class Sandwich:
    def __init__(self):
        self.f = gpdtails(star_skater_factory(), k=1)
        self.state = None
        self.pending = None

    def logpdf(self, y):
        if self.pending is None:
            return fs.LOGPDF_FLOOR
        try:
            return max(fs.LOGPDF_FLOOR, self.pending.logpdf(y))
        except Exception:  # noqa: BLE001
            return fs.LOGPDF_FLOOR

    def update(self, y):
        dists, self.state = self.f(y, self.state)
        self.pending = dists[0]


class Mixture:
    DECAY = 0.98

    def __init__(self):
        self.a = StarCls()
        self.b = fs.LaplaceRef()
        self.ca = 0.0
        self.cb = 0.0

    def _w(self):
        m = max(self.ca, self.cb)
        ea = math.exp(min(50.0, self.ca - m))
        eb = math.exp(min(50.0, self.cb - m))
        return ea / (ea + eb)

    def logpdf(self, y):
        la = max(fs.LOGPDF_FLOOR, self.a.logpdf(y))
        lb = max(fs.LOGPDF_FLOOR, self.b.logpdf(y))
        w = self._w()
        m = max(la, lb)
        return m + math.log(w * math.exp(la - m) + (1 - w) * math.exp(lb - m))

    def update(self, y):
        la = max(fs.LOGPDF_FLOOR, self.a.logpdf(y))
        lb = max(fs.LOGPDF_FLOOR, self.b.logpdf(y))
        self.ca = self.DECAY * self.ca + la
        self.cb = self.DECAY * self.cb + lb
        m = max(self.ca, self.cb)
        self.ca -= m
        self.cb -= m
        self.a.update(y)
        self.b.update(y)


SEEDS = range(35, 40)
instances = [fs.make_instance(f, s) for f in fs.FAMILIES for s in SEEDS] + [
    make_novel(f, s) for f in NOVEL for s in SEEDS
]
print(f"{len(instances)} fresh instances (seeds 35-39, all ten families)", flush=True)

ARMS = {
    "star": StarCls,
    "laplace": fs.LaplaceRef,
    "mixture": Mixture,
    "sandwich": Sandwich,
}
scores = {name: [] for name in ARMS}
for i, series in enumerate(instances):
    for name, cls in ARMS.items():
        scores[name].append(fs.mean_nll(cls, series))
    if (i + 1) % 10 == 0:
        print(f"  [{i + 1}/{len(instances)}]", flush=True)

print("\nmean NLL per tick:")
for name in ARMS:
    print(f"  {name:9s} {sum(scores[name]) / len(instances):.4f}")


def pair(a, b):
    w = sum(1 for x, y in zip(scores[a], scores[b]) if x < y)
    l = sum(1 for x, y in zip(scores[a], scores[b]) if x > y)
    n = w + l
    k = min(w, l)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2**n)
    print(f"  {a} beats {b}: {w}/{n} = {round(100 * w / n)}pct p={p:.1e}")


print("\npaired:")
pair("star", "laplace")
pair("mixture", "laplace")
pair("mixture", "star")
pair("sandwich", "star")
