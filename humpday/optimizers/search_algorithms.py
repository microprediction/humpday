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
    """Coordinate Descent with an adaptive expanding line search per axis.

    Pure-Python via the `humpday._array` shim — no direct numpy use.

    For each coordinate `i`, take a step `± step_size`; if it improves,
    keep stepping in the same direction until it stops improving
    (greedy expansion — the same shape Powell uses). After a full
    sweep over all coordinates, halve `step_size` until it drops below
    1e-12.

    The previous implementation shrank `step_size *= 0.8` per failed
    sweep (so even after 30 sweeps it was only at ~0.001) and reset
    to 0.05 once `step_size < 1e-6` — a humpday-ism that explicitly
    prevented convergence below ~1e-3. That's why the snapshot showed
    a ~5.6e+07× gap vs scipy Powell with `direc=I` on the sphere.
    """

    def optimize(self):
        n = self.n_dim
        x = _A.random_uniform(n)
        f = self.evaluate(x)

        step = 0.1
        step_min = 1e-12

        while self.evaluations < self.n_trials and step > step_min:
            improved_anywhere = False

            for i in range(n):
                if self.evaluations >= self.n_trials:
                    break

                # Try both signs along axis i; on the first improving
                # step, greedily expand in that direction.
                for sign in (1, -1):
                    if self.evaluations >= self.n_trials:
                        break

                    xi_new = max(0.0, min(1.0, float(x[i]) + sign * step))
                    if abs(xi_new - float(x[i])) < 1e-15:
                        continue  # already at the bound in this direction
                    x_trial = x.copy()
                    x_trial[i] = xi_new
                    f_trial = self.evaluate(x_trial)
                    if f_trial >= f:
                        continue

                    x = x_trial
                    f = f_trial
                    improved_anywhere = True

                    # Greedy expansion in the same direction.
                    while self.evaluations < self.n_trials:
                        xi_next = max(0.0, min(1.0, float(x[i]) + sign * step))
                        if abs(xi_next - float(x[i])) < 1e-15:
                            break
                        x_trial2 = x.copy()
                        x_trial2[i] = xi_next
                        f_trial2 = self.evaluate(x_trial2)
                        if f_trial2 >= f:
                            break
                        x = x_trial2
                        f = f_trial2

                    break  # don't try the other sign on this coordinate

            if not improved_anywhere:
                step *= 0.5

        return self.best_value, self.best_x


class PatternSearch(BaseOptimizer):
    """Hooke-Jeeves pattern search with exploratory + pattern moves.

    Pure-Python via the `humpday._array` shim — no direct numpy use.

    The classic Hooke-Jeeves algorithm (1961):
      1. Exploratory move from `base`: try `±step` along each axis,
         keep improvements (single-shot per coordinate).
      2. If the exploratory move improved on `base`, do a *pattern
         move* — extrapolate from `base` through the new point:
         `new_base = x + (x - base)`. This is the speed-up that
         lets Hooke-Jeeves race down valleys like Rosenbrock that
         pure coordinate descent zigzags through.
      3. Do another exploratory move from `new_base`; accept it
         (and the pattern move) if it beats the bare exploratory.
      4. If no exploratory improvement after a full sweep, halve
         `step` until it shrinks below 1e-12.

    The previous implementation was "first-improvement among the 2n
    axis directions" with a `step *= 0.5` shrink and a random restart
    when `step < 1e-6` — neither the pattern-move acceleration nor
    a precision floor below ~1e-6, which is why the snapshot showed
    a ~1.2e+10× gap on Ackley vs scipy DIRECT.
    """

    def optimize(self):
        base = _A.random_uniform(self.n_dim)
        f_base = self.evaluate(base)
        step = 0.1
        step_min = 1e-12

        while self.evaluations < self.n_trials and step > step_min:
            # 1. Exploratory move from base.
            x, f = self._explore(base.copy(), f_base, step)

            if f < f_base:
                if self.evaluations < self.n_trials:
                    # 2. Pattern move: extrapolate from base through x.
                    new_base = _A.clip(x + (x - base), 0, 1)
                    f_new_base = self.evaluate(new_base)
                    # 3. Exploratory move from the pattern point.
                    x2, f2 = self._explore(new_base.copy(), f_new_base, step)
                    if f2 < f:
                        base, f_base = x2, f2
                    else:
                        base, f_base = x, f
                else:
                    base, f_base = x, f
            else:
                # 4. No exploratory progress at this step: halve.
                step *= 0.5

        return self.best_value, self.best_x

    def _explore(self, x, f, step):
        """Single exploratory sweep — try ±step on each axis in order,
        keep improvements. First-improvement per coordinate (the +
        direction wins immediately if it helps; otherwise try −)."""
        for i in range(self.n_dim):
            if self.evaluations >= self.n_trials:
                break
            for sign in (1, -1):
                if self.evaluations >= self.n_trials:
                    break
                xi_new = max(0.0, min(1.0, float(x[i]) + sign * step))
                if abs(xi_new - float(x[i])) < 1e-15:
                    continue
                x_trial = x.copy()
                x_trial[i] = xi_new
                f_trial = self.evaluate(x_trial)
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    break  # don't try the other sign on this coord
        return x, f
