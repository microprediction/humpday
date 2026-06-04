"""
Landscape probes for the recommender's § 5 / § 7 follow-up experiment.

For each demo objective, draw k random samples on the unit cube and
compute a handful of cheap, well-defined features that try to capture
"landscape character" — smoothness, range, axis alignment — without
running an optimizer. The features feed the LOO classifier in
`analysis.py::probe_experiment`.

No optimizer API change. The probe is pure overhead on top of the
existing `pure_optimize(objective, algo, n_trials, n_dim)` interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProbeFeatures:
    """Features extracted from k random probe samples on [0,1]^n_dim.

    All features are scale-aware (most are normalised by the range of
    observed f-values) so they are comparable across demos with wildly
    different objective scales.
    """

    k: int
    n_dim: int
    range_f: float
    std_f: float
    rel_std: float
    local_slope_mean: float
    local_slope_cv: float
    axis_advantage: float
    n_extra_evals: int

    def as_vector(self) -> list[float]:
        """Feature vector for the classifier. We omit `range_f` and `std_f`
        (absolute scale) — only the scale-free features go to the classifier."""
        return [
            self.rel_std,
            self.local_slope_mean,
            self.local_slope_cv,
            self.axis_advantage,
        ]


def probe_landscape(
    objective: Callable[[list[float]], float],
    n_dim: int,
    k: int = 20,
    seed: int = 0,
) -> ProbeFeatures:
    """Run a k-sample landscape probe and return the feature vector.

    Total evaluation budget: k + (2 axis perturbations) + (2 random
    perturbations) — currently 24 evals at default k=20.
    """
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    samples = [[rng.random() for _ in range(n_dim)] for _ in range(k)]
    fvals = [float(objective(x)) for x in samples]

    fmin = min(fvals)
    fmax = max(fvals)
    range_f = fmax - fmin
    std_f = float(np.std(fvals))
    mean_abs = float(np.mean(np.abs(fvals)))
    rel_std = std_f / mean_abs if mean_abs > 0 else 0.0

    # Local slope: average of |Δf| / ||Δx||_2 across all C(k,2) pairs.
    # The CV (std/mean) captures whether the surface has a consistent
    # slope (low CV, smooth) or many different slopes (high CV, rough /
    # multimodal / discontinuous penalty).
    slopes: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            dx = np.array(samples[i]) - np.array(samples[j])
            d = float(np.linalg.norm(dx))
            if d > 0 and range_f > 0:
                slopes.append(abs(fvals[i] - fvals[j]) / d / range_f)
    slope_mean = float(np.mean(slopes)) if slopes else 0.0
    slope_std = float(np.std(slopes)) if slopes else 0.0
    slope_cv = slope_std / slope_mean if slope_mean > 0 else 0.0

    # Axis vs random advantage. From the best-found probe sample, take
    # 2 axis-aligned steps and 2 random-direction steps of the same L2
    # length, and ratio the best improvement. > 1 means axis moves are
    # more productive — coordinate descent / pattern search territory.
    step = 0.05
    best_idx = fvals.index(fmin)
    anchor = list(samples[best_idx])
    f_anchor = fvals[best_idx]

    axis_best_improve = 0.0
    n_extra = 0
    for _ in range(2):
        axis = rng.randrange(n_dim)
        sign = rng.choice([-1.0, 1.0])
        x = list(anchor)
        x[axis] = float(np.clip(x[axis] + sign * step, 0.0, 1.0))
        f_new = float(objective(x))
        n_extra += 1
        axis_best_improve = max(axis_best_improve, f_anchor - f_new)

    random_best_improve = 0.0
    for _ in range(2):
        direction = nprng.normal(size=n_dim)
        direction /= np.linalg.norm(direction) or 1.0
        x = np.clip(np.array(anchor) + step * direction, 0.0, 1.0).tolist()
        f_new = float(objective(x))
        n_extra += 1
        random_best_improve = max(random_best_improve, f_anchor - f_new)

    # Ratio: axis vs random. If both ≤ 0 (we got worse), call it 1.0 (neutral).
    if axis_best_improve <= 0 and random_best_improve <= 0:
        axis_advantage = 1.0
    elif random_best_improve <= 0:
        axis_advantage = 10.0  # axis wins decisively
    else:
        axis_advantage = max(axis_best_improve, 0.0) / random_best_improve

    return ProbeFeatures(
        k=k,
        n_dim=n_dim,
        range_f=range_f,
        std_f=std_f,
        rel_std=rel_std,
        local_slope_mean=slope_mean,
        local_slope_cv=slope_cv,
        axis_advantage=axis_advantage,
        n_extra_evals=n_extra,
    )
