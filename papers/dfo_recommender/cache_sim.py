"""Cache-eviction benchmark for the second inspiration-simplex example.

Traces are seeded generators; the disguise analog randomizes each family's
parameters and permutes key labels per seed, so a generated policy cannot
succeed by memorising keys or magic constants tied to one workload.

Reference policies: LRU, LFU, FIFO, CLOCK, RANDOM (the panel) and ARC (the
famous hand-designed interior point, reported but not part of the
normalisation panel).

Score: miss-rate regret normalised against the panel on the same instance,
0 matching the best panel member, 1 the worst.
"""

from __future__ import annotations

import random
from collections import OrderedDict, deque


# ---------------------------------------------------------------- policies
class LRU:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()

    def access(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            return True
        if len(self.d) >= self.cap:
            self.d.popitem(last=False)
        self.d[key] = True
        return False


class LFU:
    def __init__(self, capacity):
        self.cap = capacity
        self.count = {}
        self.tick = 0
        self.last = {}

    def access(self, key):
        self.tick += 1
        if key in self.count:
            self.count[key] += 1
            self.last[key] = self.tick
            return True
        if len(self.count) >= self.cap:
            victim = min(self.count, key=lambda k: (self.count[k], self.last[k]))
            del self.count[victim]
            del self.last[victim]
        self.count[key] = 1
        self.last[key] = self.tick
        return False


class FIFO:
    def __init__(self, capacity):
        self.cap = capacity
        self.q = deque()
        self.s = set()

    def access(self, key):
        if key in self.s:
            return True
        if len(self.q) >= self.cap:
            self.s.discard(self.q.popleft())
        self.q.append(key)
        self.s.add(key)
        return False


class CLOCK:
    def __init__(self, capacity):
        self.cap = capacity
        self.keys = []
        self.ref = {}
        self.hand = 0

    def access(self, key):
        if key in self.ref:
            self.ref[key] = 1
            return True
        if len(self.keys) >= self.cap:
            while True:
                k = self.keys[self.hand]
                if self.ref[k]:
                    self.ref[k] = 0
                    self.hand = (self.hand + 1) % len(self.keys)
                else:
                    del self.ref[k]
                    self.keys[self.hand] = key
                    self.ref[key] = 1
                    self.hand = (self.hand + 1) % len(self.keys)
                    return False
        self.keys.append(key)
        self.ref[key] = 1
        return False


class RANDOM:
    def __init__(self, capacity, seed=0):
        self.cap = capacity
        self.s = set()
        self.rng = random.Random(seed)

    def access(self, key):
        if key in self.s:
            return True
        if len(self.s) >= self.cap:
            self.s.discard(self.rng.choice(list(self.s)))
        self.s.add(key)
        return False


class ARC:
    """Adaptive Replacement Cache (Megiddo & Modha, 2003)."""

    def __init__(self, capacity):
        self.c = capacity
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()

    def _replace(self, in_b2):
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False)
            self.b1[k] = True
        else:
            k, _ = self.t2.popitem(last=False)
            self.b2[k] = True

    def access(self, key):
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = True
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True
        if key in self.b1:
            self.p = min(self.c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            self._replace(False)
            del self.b1[key]
            self.t2[key] = True
            return False
        if key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            self._replace(True)
            del self.b2[key]
            self.t2[key] = True
            return False
        if len(self.t1) + len(self.b1) == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
                self._replace(False)
            else:
                self.t1.popitem(last=False)
        elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= self.c:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * self.c:
                if self.b2:
                    self.b2.popitem(last=False)
                elif self.b1:
                    self.b1.popitem(last=False)
            if len(self.t1) + len(self.t2) >= self.c:
                self._replace(False)
        self.t1[key] = True
        return False


PANEL = {"LRU": LRU, "LFU": LFU, "FIFO": FIFO, "CLOCK": CLOCK, "RANDOM": RANDOM}


# ------------------------------------------------------------------ traces
def _zipf_sampler(rng, n_keys, alpha):
    weights = [1.0 / (i + 1) ** alpha for i in range(n_keys)]
    total = sum(weights)
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)

    def draw():
        u = rng.random()
        lo, hi = 0, n_keys - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return draw


def trace_zipf(rng, length):
    draw = _zipf_sampler(rng, rng.randint(400, 1200), rng.uniform(0.7, 1.2))
    return [draw() for _ in range(length)]


def trace_loop(rng, length):
    loop = rng.randint(150, 400)
    return [i % loop for i in range(length)]


def trace_scan_mix(rng, length):
    draw = _zipf_sampler(rng, rng.randint(300, 800), rng.uniform(0.8, 1.2))
    out = []
    nxt = 10_000
    while len(out) < length:
        out.extend(draw() for _ in range(rng.randint(200, 500)))
        scan = rng.randint(150, 400)
        out.extend(range(nxt, nxt + scan))
        nxt += scan
    return out[:length]


def trace_phase(rng, length):
    out = []
    while len(out) < length:
        ws = rng.sample(range(100_000), rng.randint(80, 250))
        out.extend(rng.choice(ws) for _ in range(rng.randint(500, 1500)))
    return out[:length]


def trace_burst(rng, length):
    out = []
    recent = deque(maxlen=50)
    draw = _zipf_sampler(rng, 2000, 0.6)
    for _ in range(length):
        if recent and rng.random() < rng.uniform(0.4, 0.7):
            k = rng.choice(recent)
        else:
            k = draw()
        recent.append(k)
        out.append(k)
    return out


def trace_mixture(rng, length):
    a = trace_zipf(random.Random(rng.random()), length)
    b = trace_loop(random.Random(rng.random()), length)
    return [(a[i] if rng.random() < 0.6 else b[i] + 1_000_000) for i in range(length)]


FAMILIES = {
    "zipf": trace_zipf,
    "loop": trace_loop,
    "scan_mix": trace_scan_mix,
    "phase": trace_phase,
    "burst": trace_burst,
    "mixture": trace_mixture,
}

TRACE_LEN = 20_000


def make_instance(family, seed):
    """Seeded, parameter-randomized, key-permuted instance."""
    import zlib

    rng = random.Random(zlib.crc32(f"{family}:{seed}".encode()))
    raw = FAMILIES[family](rng, TRACE_LEN)
    perm = {}
    prng = random.Random(seed * 7919 + 13)
    trace = []
    for k in raw:
        if k not in perm:
            perm[k] = prng.randrange(10_000_000)
        trace.append(perm[k])
    capacity = rng.randint(64, 192)
    return trace, capacity


def hit_rate(policy_cls, trace, capacity):
    p = policy_cls(capacity)
    hits = sum(1 for k in trace if p.access(k))
    return hits / len(trace)


def score_policy(policy_cls, instances, panel_cache=None):
    """Mean panel-normalised miss regret across instances (0 best, 1 worst)."""
    regrets = []
    for i, (trace, cap) in enumerate(instances):
        if panel_cache is not None:
            panel_miss = panel_cache[i]
        else:
            panel_miss = [1.0 - hit_rate(c, trace, cap) for c in PANEL.values()]
        try:
            cand = 1.0 - hit_rate(policy_cls, trace, cap)
        except Exception:  # noqa: BLE001
            regrets.append(1.0)
            continue
        vals = [cand] + panel_miss
        mn, mx = min(vals), max(vals)
        regrets.append(0.0 if mx <= mn else (cand - mn) / (mx - mn))
    return sum(regrets) / len(regrets)


def build_panel_cache(instances):
    return [
        [1.0 - hit_rate(c, trace, cap) for c in PANEL.values()]
        for (trace, cap) in instances
    ]


if __name__ == "__main__":
    insts = [make_instance(f, s) for f in FAMILIES for s in (0, 1)]
    cache = build_panel_cache(insts)
    for name, cls in list(PANEL.items()) + [("ARC", ARC)]:
        print(f"{name:8s} regret {score_policy(cls, insts, cache):.4f}")
