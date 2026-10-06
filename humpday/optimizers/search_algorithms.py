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

    def _run(self):
        # Step bounds: 1e-12 floor lets the algorithm refine to machine
        # precision on smooth basins. Upper cap matches the unit cube.
        step_max = 1.0
        step_min = 1e-12
        window_size = 10

        x = _A.random_uniform(self.n_dim)
        f = yield x
        sigma = 0.1
        window: list[bool] = []

        while self.evaluations < self.n_trials:
            # Componentwise Gaussian perturbation — `sigma * z` where
            # `z ~ N(0, I)`. This is the canonical (1+1)-ES formulation
            # (Rechenberg 1973) and what the reference adapter at
            # `tests/test_reference_alignment.py::_ref_oneplusone_es_oneFifth`
            # uses. The previous implementation projected to a unit
            # direction first, giving every step magnitude exactly =
            # σ. That damps the stochastic spread (mean ≈ σ·√n with
            # tails much wider) and made humpday's Rechenberg trap in
            # local basins on multimodal landscapes — snapshot Ackley
            # was 2.58 vs the reference's 6.9e-6 at identical settings.
            z = _A.random_normal(self.n_dim)
            x_new = _A.clip(x + sigma * z, 0, 1)

            if self.evaluations >= self.n_trials:
                break
            f_new = yield x_new

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
                rate = _A.fold_sum(window) / window_size
                if rate > 1 / 5:
                    sigma = min(step_max, sigma * 1.5)
                elif rate < 1 / 5:
                    sigma = max(step_min, sigma / 1.5)


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
    sweep over all coordinates that improved nothing, shrink
    `step_size`; once it falls below `_restart_threshold` the basin is
    finished, so restart from a fresh point.

    The previous implementation shrank `step_size *= 0.8` per failed
    sweep (so even after 30 sweeps it was only at ~0.001) and reset
    to 0.05 once `step_size < 1e-6` — a humpday-ism that explicitly
    prevented convergence below ~1e-3. That's why the snapshot showed
    a ~5.6e+07× gap vs scipy Powell with `direc=I` on the sphere.
    """

    # When `step` collapses below this the sweep has taken this basin as far as it goes, and
    # the thing to do is go and look at another one. Refine first, though: the threshold was
    # 1e-6, which stopped the line search at a precision of about 1e-12 on a quadratic and cost
    # every single head-to-head pairing against the textbook reference on the sphere. It was
    # chosen to buy Ackley restarts, and measured against the previous humpday rather than
    # against the reference -- head to head it was losing 0.86 of pairings on Ackley too, so it
    # was not buying them.
    _restart_threshold = 1e-12
    # Shrink per failed sweep. Halving from 0.1 down to 1e-12 is thirty-seven failed sweeps at
    # 2n evaluations each, which on a budget of 200 is the entire budget spent shrinking: the
    # reason the threshold had been raised to 1e-6 was to avoid paying it. A quarter gets there
    # in nineteen sweeps, which is what makes the deep floor reachable at all at that budget.
    #
    # Quartering skips scales, and that is not free. Halving from 0.1 probes 0.05 and 0.025;
    # quartering jumps straight to 0.025, so a problem whose productive step sits near 0.04
    # never gets probed there. `groundwater_remediation` in example_applications is such a
    # problem and it is the one real casualty of this change: at a budget of 1000 it loses 0.98
    # of its head-to-head pairings against the old setting. That is the shrink rate alone, not
    # the threshold -- it measures 0.99 at a threshold of 1e-6 too, and 0.50 under halving at
    # either threshold.
    #
    # Kept anyway, because the unbiased measurement says so. Over all 70 objectives in
    # example_applications at 20 seeds, against the old 1e-6 / halving (mean fraction of
    # head-to-head pairings lost, so below 0.5 is better; "better"/"worse" count problems
    # past 0.35 and 0.65):
    #
    #     budget  optimizer           setting        better  worse   mean lost
    #        200  CoordinateDescent   1e-12, halve     4/70      1       0.489
    #        200  CoordinateDescent   1e-12, quarter  21/70      1       0.392
    #        200  PatternSearch       1e-12, halve     2/70      2       0.500
    #        200  PatternSearch       1e-12, quarter  20/70      0       0.376
    #       1000  CoordinateDescent   1e-12, halve    22/70      6       0.401
    #       1000  CoordinateDescent   1e-12, quarter  25/70      6       0.385
    #       1000  PatternSearch       1e-12, halve    17/70      7       0.430
    #       1000  PatternSearch       1e-12, quarter  22/70      7       0.392
    #
    # Halving is not the safe option it looks like from `groundwater_remediation` alone; it has
    # its own casualties (`gear_ratios` 0.71, `bowling` 0.70, `ebola_response` 0.71,
    # `algo_trading` 0.73) and is worse on average at every budget. A tenth is faster again and
    # measures the same on the sphere and Ackley, but steps past the scale Rosenbrock's valley
    # wants and loses 0.68 of pairings there against 0.63.
    #
    # What would remove the tradeoff rather than settle it is a schedule that visits every
    # octave and accelerates only after several sweeps in a row have failed. Not added on
    # speculation.
    _shrink = 0.25

    def _run(self):
        n = self.n_dim
        x = _A.random_uniform(n)
        f = yield x

        step = 0.1
        restart_step_threshold = self._restart_threshold
        shrink = self._shrink

        while self.evaluations < self.n_trials:
            if step <= restart_step_threshold:
                # Restart either way. This used to `break` when f was already below the
                # converged threshold -- "converged in a good basin" -- which read the
                # situation exactly backwards: a basin that is finished is the reason to go
                # and look at another one, not to stop. On the sphere it meant 130 evaluations
                # of 5,000 and the same answer at every budget (#330). _bookkeep holds the
                # best point seen, so a restart that lands somewhere worse cannot cost
                # anything but the evaluations that were being discarded anyway.
                x = _A.random_uniform(n)
                f = yield x
                step = 0.1
                continue

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
                    f_trial = yield x_trial
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
                        f_trial2 = yield x_trial2
                        if f_trial2 >= f:
                            break
                        x = x_trial2
                        f = f_trial2

                    break  # don't try the other sign on this coordinate

            if not improved_anywhere:
                step *= shrink


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
      4. If no exploratory improvement after a full sweep, shrink
         `step` until it drops below `_restart_threshold`, then
         restart from a fresh base.

    The previous implementation was "first-improvement among the 2n
    axis directions" with a `step *= 0.5` shrink and a random restart
    when `step < 1e-6` — neither the pattern-move acceleration nor
    a precision floor below ~1e-6, which is why the snapshot showed
    a ~1.2e+10× gap on Ackley vs scipy DIRECT.
    """

    # Same two numbers as CoordinateDescent, and for the same reason: exhaust the basin before
    # abandoning it. At 1e-6 this lost 0.98 of head-to-head pairings against Hooke-Jeeves on the
    # sphere and 0.84 on Ackley.
    _restart_threshold = 1e-12
    _shrink = 0.25

    def _run(self):
        base = _A.random_uniform(self.n_dim)
        f_base = yield base
        step = 0.1
        restart_step_threshold = self._restart_threshold
        shrink = self._shrink

        while self.evaluations < self.n_trials:
            if step <= restart_step_threshold:
                # Restart either way; see CoordinateDescent for why converging is a reason to
                # move on rather than to stop. This handed back 4,830 of 5,000 on the sphere.
                base = _A.random_uniform(self.n_dim)
                f_base = yield base
                step = 0.1
                continue

            # 1. Exploratory move from base.
            x, f = yield from self._explore_gen(base.copy(), f_base, step)

            if f < f_base:
                if self.evaluations < self.n_trials:
                    # 2. Pattern move: extrapolate from base through x.
                    new_base = _A.clip(x + (x - base), 0, 1)
                    f_new_base = yield new_base
                    # 3. Exploratory move from the pattern point.
                    x2, f2 = yield from self._explore_gen(
                        new_base.copy(), f_new_base, step
                    )
                    if f2 < f:
                        base, f_base = x2, f2
                    else:
                        base, f_base = x, f
                else:
                    base, f_base = x, f
            else:
                # 4. No exploratory progress at this step: shrink.
                step *= shrink

    def _explore_gen(self, x, f, step):
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
                f_trial = yield x_trial
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    break  # don't try the other sign on this coord
        return x, f


class GridSearch(BaseOptimizer):
    """Regular-grid baseline.

    Evaluates a uniform Cartesian grid over the unit cube `[0, 1]^n_dim`
    in levels. Each level sweeps the largest complete grid that fits in
    half the remaining budget, keeps the cell that won, and sweeps that
    cell in turn, so the resolution compounds instead of being decided
    by one choice of spacing. Each axis is split into equal-width bins
    and the evaluated point in each bin is its centre, so samples never
    sit on the box edges; the bin count is kept odd where it can be, so
    the centre of the box is always one of them.

    Like RandomSearch, this is included as a baseline (regression check,
    contest sanity floor), not as a SOTA algorithm -- though refinement
    makes it a better one than a single sweep was, which is worth
    remembering when reading it as a floor.

    Note: grid size scales as `n_per_axis^n_dim`. For modest budgets and
    `n_dim >= 5` the grid degenerates to fewer than 2 points per axis,
    making the algorithm equivalent to evaluating a handful of corners.
    Practically useful for `n_dim <= 3`.
    """

    def _run(self):
        n = self.n_dim
        lo = [0.0] * n
        hi = [1.0] * n

        first = True
        while True:
            remaining = self.n_trials - self.evaluations
            # Half the remaining budget per level, so there is always something left to refine
            # with. A sweep that spends everything is one grid, and one grid is decided by
            # whether its bin centres happen to land on the optimum: at two dimensions a budget
            # of 1,000 gives 31 bins, whose centres include exactly 0.5 and hit the sphere's
            # minimum dead on, while 5,000 gives 70 and straddles it. Halving turns that piece
            # of luck into a sequence of levels, each of which narrows the box by its own bin
            # count, so the resolution compounds instead of depending on a parity.
            bins = self._odd_bins_that_fit(remaining // 2, n)
            if bins < 2:
                # Not enough left to split; spend what remains on one final sweep.
                bins = self._odd_bins_that_fit(remaining, n)
            if bins < 2:
                if first and remaining > 0:
                    # Not even the coarsest complete grid fits -- the case the docstring warns
                    # about, where 2**n_dim already exceeds the budget. Sweep as much of it as
                    # there is room for, which is what this did at every size before, rather
                    # than returning without having looked at anything.
                    yield from self._sweep_gen(lo, hi, 2)
                break
            best_cell = yield from self._sweep_gen(lo, hi, bins)
            first = False
            if best_cell is None:
                break
            # Refine into the cell that won, and sweep again with whatever is left. A grid is
            # exhausted the moment it is finished, so without this the budget past the first
            # sweep is forfeited -- 904 evaluations of 5,000 at four dimensions (#330) -- and
            # a larger budget bought a coarser answer rather than a finer one.
            lo, hi = best_cell

    @classmethod
    def _odd_bins_that_fit(cls, budget: int, n: int) -> int:
        """As `_bins_that_fit`, but odd where it can be, which keeps the centre of the box in
        the sample.

        An odd bin count puts a bin centre exactly at the middle of the box; an even one
        straddles it. That matters more than it looks, because a great many test objectives
        have their optimum at the centre of the cube, and because the refinement below keeps
        re-centring on the winning cell -- an odd count at every level means the centre of the
        current box is evaluated at every level, so a centred optimum is found exactly rather
        than approached. Dropping one bin per axis to get there costs a few percent of
        resolution and buys the exact hit.
        """
        bins = cls._bins_that_fit(budget, n)
        if bins > 2 and bins % 2 == 0:
            return bins - 1
        return bins

    @staticmethod
    def _bins_that_fit(budget: int, n: int) -> int:
        """Largest b with b**n <= budget, found by integer search rather than by rounding.

        `round(budget ** (1 / n))` overshoots about half the time -- at four dimensions and a
        budget of 1000 it asks for 6 bins, which is 1,296 points, so the sweep was cut off
        partway through and what remained was a slab of the cube rather than a grid of it. The
        answer then depended on which corner the odometer had reached, which is why a budget of
        5,000 scored worse than one of 1,000.
        """
        if budget < 2**n:
            return 0
        b = 2
        while (b + 1) ** n <= budget:
            b += 1
        return b

    def _sweep_gen(self, lo, hi, bins):
        """Evaluate the full `bins**n` grid inside the box, and return the best cell's box.

        Points sit at bin centres, so they are spread inside the box without ever landing on
        its edges -- the same convention the single sweep used, applied to a box that shrinks.
        """
        n = self.n_dim
        widths = [(hi[d] - lo[d]) / bins for d in range(n)]
        indices = [0] * n
        best_value = float("inf")
        best_indices = None

        while True:
            if self.evaluations >= self.n_trials:
                break
            x = _A.asarray([lo[d] + (indices[d] + 0.5) * widths[d] for d in range(n)])
            value = yield x
            if value < best_value:
                best_value = value
                best_indices = list(indices)
            # Increment indices like an odometer; stop once all wrap.
            d = n - 1
            while d >= 0:
                indices[d] += 1
                if indices[d] < bins:
                    break
                indices[d] = 0
                d -= 1
            if d < 0:
                break  # full grid exhausted

        if best_indices is None:
            return None
        return (
            [lo[d] + best_indices[d] * widths[d] for d in range(n)],
            [lo[d] + (best_indices[d] + 1) * widths[d] for d in range(n)],
        )
