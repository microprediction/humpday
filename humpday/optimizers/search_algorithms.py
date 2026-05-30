"""
Search algorithm implementations.

These algorithms perform systematic or guided search through the solution space,
including methods like coordinate descent, pattern search, and adaptive random search.
They are particularly effective for local optimization and structured exploration.
"""

from humpday import _array as _A

from .base import BaseOptimizer


class Rechenberg(BaseOptimizer):
    """Rechenberg's (1+1)-Evolution Strategy with the 1/5-success-rule.

    The classic adaptive random search algorithm: one parent, one
    Gaussian-perturbed offspring per generation; offspring accepted if
    it beats the parent. The step size σ is adapted by Rechenberg's
    1/5-rule — grow σ by 1.5× when more than 1/5 of recent trials
    succeed, shrink by 1.5⁻¹ otherwise. Reference: Rechenberg (1973),
    "Evolutionsstrategie: Optimierung technischer Systeme nach
    Prinzipien der biologischen Evolution".

    Pure-Python via the `humpday._array` shim — no direct numpy use.

    Was previously named ``AdaptiveRandomSearch``; the rename matches
    the canonical literature name and the reference adapter used by
    ``test_reference_alignment.py``. ``AdaptiveRandomSearch`` is kept
    as a module-level alias below for backwards compatibility.
    """

    def optimize(self):
        x = _A.random_uniform(self.n_dim)
        f = self.evaluate(x)
        # Step bounds: 1e-12 floor lets the algorithm refine to machine
        # precision on smooth basins (the previous 0.01 floor was the
        # entire reason this port was ~5.7e+08× off the reference on
        # the sphere benchmark — the algorithm worked, the floor just
        # capped its precision at ~1e-4). Upper cap matches the unit
        # cube diameter.
        step_size = 0.1
        step_min = 1e-12
        step_max = 1.0
        # Rolling window of recent successes for the 1/5-rule.
        window = []
        window_size = 10

        while self.evaluations < self.n_trials:
            # Random unit-direction step.
            direction = _A.random_normal(self.n_dim)
            direction = direction / (_A.norm(direction) + 1e-10)

            x_new = _A.clip(x + step_size * direction, 0, 1)

            if self.evaluations >= self.n_trials:
                break
            f_new = self.evaluate(x_new)

            accepted = f_new < f
            if accepted:
                x = x_new
                f = f_new
            window.append(accepted)
            if len(window) > window_size:
                window.pop(0)

            # Strict 1/5-rule on the rolling window — 1.5×/1.5⁻¹
            # adaptation, no smoothing.
            if len(window) >= window_size:
                rate = sum(window) / window_size
                if rate > 1 / 5:
                    step_size = min(step_max, step_size * 1.5)
                elif rate < 1 / 5:
                    step_size = max(step_min, step_size / 1.5)

        return self.best_value, self.best_x


# Backwards-compatibility alias — the class was renamed in this commit.
# Anyone importing `AdaptiveRandomSearch` (HumpDay registry, docstrings,
# downstream code) keeps working; the registry below pins both names to
# the same class.
AdaptiveRandomSearch = Rechenberg


class CoordinateDescent(BaseOptimizer):
    """Coordinate Descent optimization.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Cycles through axes; for each, takes a step in both directions and
    keeps the better. Step size halves when a full sweep makes no progress;
    resets when it gets too small.
    """

    def optimize(self):
        x = _A.random_uniform(self.n_dim)
        f = self.evaluate(x)
        step_size = 0.1

        while self.evaluations < self.n_trials:
            improved = False

            # Cycle through coordinates.
            for i in range(self.n_dim):
                if self.evaluations >= self.n_trials:
                    break

                best_xi = x[i]
                best_f = f

                # Try steps in both directions along axis `i`.
                for direction in [-1, 1]:
                    x_trial = x.copy()
                    # Clip the single perturbed coordinate to [0, 1].
                    xi_new = x[i] + direction * step_size
                    x_trial[i] = (
                        0.0 if xi_new < 0.0 else (1.0 if xi_new > 1.0 else xi_new)
                    )

                    if self.evaluations < self.n_trials:
                        f_trial = self.evaluate(x_trial)
                        if f_trial < best_f:
                            best_xi = x_trial[i]
                            best_f = f_trial
                            improved = True

                x[i] = best_xi
                f = best_f

            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    step_size = 0.05  # Reset

        return self.best_value, self.best_x


class PatternSearch(BaseOptimizer):
    """Pattern Search algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Probes both signs of each coordinate axis at the current step size;
    grows the step when it succeeds, halves it when no probe improves;
    restarts when the step gets too small.
    """

    def optimize(self):
        x = _A.random_uniform(self.n_dim)
        f = self.evaluate(x)
        step_size = 0.1

        while self.evaluations < self.n_trials:
            improved = False

            # Pattern directions: +/- each coordinate axis.
            directions = []
            for i in range(self.n_dim):
                direction = _A.zeros(self.n_dim)
                direction[i] = 1
                directions.append(direction)
                directions.append(-direction)

            # First-improvement: take the first direction that helps.
            for direction in directions:
                if self.evaluations >= self.n_trials:
                    break

                x_trial = _A.clip(x + step_size * direction, 0, 1)
                f_trial = self.evaluate(x_trial)

                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    break

            if improved:
                step_size = min(0.3, step_size * 1.2)
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    # Random restart.
                    x = _A.random_uniform(self.n_dim)
                    if self.evaluations < self.n_trials:
                        f = self.evaluate(x)
                    step_size = 0.1

        return self.best_value, self.best_x
