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
