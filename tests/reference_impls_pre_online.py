"""Frozen pre-conversion copies of DifferentialEvolution and NelderMead.

Captured verbatim from the loop-owning implementations immediately before
the online (generator) conversion. The equivalence tests in
test_online_pilot.py require the converted optimizers to reproduce these
trajectories exactly (same RNG stream, same points, same values).

Do not modernise this file: its value is that it does not change.
"""

import random

from humpday import _array as _A
from humpday.optimizers.base import BaseOptimizer


class FrozenDifferentialEvolution(BaseOptimizer):
    """Differential Evolution.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population stored as a Python list of 1-D vectors.
    """

    def optimize(self):
        # Match scipy.optimize.differential_evolution defaults: `best1bin`
        # mutation strategy (mutate around the population's best member,
        # not a random one), dither-mutation F drawn per-generation in
        # [0.5, 1.0], recombination probability 0.7, and a local-search
        # polish stage after DE (scipy uses `polish=True` by default,
        # which calls minimize with L-BFGS-B). The closest
        # derivative-free polish HumpDay can use is coordinate descent
        # with a shrinking step — same pattern SimulatedAnnealing's
        # stage 2 uses, which matches scipy.dual_annealing.
        #
        # Reference: scipy/optimize/_differentialevolution.py.
        # Without the polish, humpday DE was ~7e6× worse than scipy DE
        # on the sphere benchmark at n_trials=200 — the reference's
        # L-BFGS-B polish converges to machine precision in tens of
        # function evals, while humpday DE alone hits a noise floor
        # around 1e-5 at that budget.
        # Allocate half the budget to the L-BFGS-B polish — scipy
        # DE's polish=True runs `_minimize_lbfgsb` unbudgeted, so its
        # polish gets analytic-gradient convergence regardless of the
        # DE budget. Our polish uses FD gradients (2·n_dim evals per
        # gradient) so it needs proportionally more budget to reach
        # the same precision. Sweep on 2-D Rosenbrock at n_trials=200:
        # 25% → 2.2e-4, 40% → 2.5e-8, 50% → 3.5e-10 (matches scipy),
        # 60% → 2.7e-6 (DE stage no longer finds the basin).
        polish_budget = max(15, self.n_trials // 2)
        de_budget = self.n_trials - polish_budget

        pop_size = max(10, min(20, de_budget // 5))
        CR = 0.7  # scipy default recombination probability.

        population = [_A.random_uniform(self.n_dim) for _ in range(pop_size)]
        fitness = [self.evaluate(ind) for ind in population]

        while self.evaluations < de_budget:
            # Dither: pick F uniformly in [0.5, 1.0] per generation. This
            # is scipy's default `mutation=(0.5, 1)` behaviour, which
            # helps the population avoid stagnation by varying the
            # mutation scale.
            F = 0.5 + 0.5 * _A.random_scalar()

            for i in range(pop_size):
                if self.evaluations >= de_budget:
                    break

                # `best1bin`: base point is the current population best,
                # not a random member. Find best index.
                best_idx = min(range(pop_size), key=fitness.__getitem__)

                # Two donors distinct from i and best_idx.
                candidates = [k for k in range(pop_size) if k != i and k != best_idx]
                if len(candidates) < 2:
                    candidates = [k for k in range(pop_size) if k != i]
                if len(candidates) < 2:
                    b, c = _A.random_choice(candidates, k=2, replace=True)
                else:
                    b, c = _A.random_choice(candidates, k=2, replace=False)
                b, c = int(b), int(c)

                # Mutation: v = x_best + F * (x_b - x_c), clipped to bounds.
                mutant = _A.clip(
                    population[best_idx] + F * (population[b] - population[c]),
                    0,
                    1,
                )

                # Binomial crossover with at least one guaranteed coord.
                trial = population[i].copy()
                j_guaranteed = _A.random_int(self.n_dim)
                for j in range(self.n_dim):
                    if _A.random_scalar() < CR or j == j_guaranteed:
                        trial[j] = mutant[j]

                # (1+1) selection.
                trial_fitness = self.evaluate(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        # --- Polish stage: L-BFGS from the best DE point ---------------
        # Matches scipy.differential_evolution's `polish=True` exactly —
        # scipy uses L-BFGS-B. SimulatedAnnealing got the same upgrade
        # in the previous commit (#188); same inlined two-loop recursion
        # + FD gradient + Armijo line search the LBFGSB optimizer uses.
        # Closes the residual sphere gap that coord descent couldn't
        # reach.
        self._lbfgs_polish()

        return self.best_value, self.best_x


class FrozenNelderMead(BaseOptimizer):
    """Nelder-Mead simplex algorithm (faithful to SciPy implementation).

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    The simplex is stored as a Python list of n+1 vectors (each a `_Vec`
    or numpy.ndarray, depending on backend) instead of as a 2-D array;
    row operations become list comprehensions, fancy indexing becomes
    explicit `[sim[i] for i in order]`, and `np.argsort` becomes
    `sorted(range(...), key=fsim.__getitem__)`.
    """

    def optimize(self):
        n = self.n_dim

        # SciPy parameter values.
        rho = 1.0  # reflection
        chi = 2.0  # expansion
        psi = 0.5  # contraction
        sigma = 0.5  # shrinkage

        # SciPy simplex-initialization constants.
        zdelt = 0.00025

        # Convergence tolerances. scipy's defaults are 1e-4, but those
        # cause termination well before the budget is exhausted on easy
        # landscapes — on sphere this stops at f ≈ 1e-9 after ~45 evals
        # of a 200-budget. Tightening to 1e-12 lets the algorithm use its
        # budget where the landscape permits, with no downside on hard
        # problems (it just stays in the contraction phase a little
        # longer before terminating on its own at the budget cap).
        xatol = 1e-12
        fatol = 1e-12

        # Kelley (1999, "Detection and Remediation of Stagnation in the
        # Nelder-Mead Algorithm", SIAM J. Optim. 10(1)) showed that
        # vanilla NM can converge to a non-stationary point when the
        # simplex collapses into a degenerate shape. The fix in practice
        # is to reseed the simplex around the current best each time
        # convergence is reached and continue until the budget is gone.
        # Without this, NM hands back unused budget on smooth landscapes
        # while leaving plenty of room for improvement on multimodal
        # ones. The per-restart perturbation magnitude is alternated so
        # the new simplex isn't a scaled copy of the collapsed one.
        nonzdelt_schedule = [0.05, 0.15, 0.30, 0.10, 0.50, 0.20]

        # Initial seed point (used for restart 0; later restarts re-seed
        # from a fresh uniform draw when the budget is large enough for
        # the global behavior to matter, and from sim[0] otherwise — see
        # below).
        seed_point = 0.3 + 0.4 * _A.random_uniform(n)
        sim = None  # populated by the (re)build below

        restart_count = 0
        while self.evaluations < self.n_trials:
            # Build (or rebuild) the simplex around `seed_point`. The
            # NM init uses one perturbation per coordinate; the schedule
            # gives each restart a different perturbation magnitude.
            nonzdelt = nonzdelt_schedule[restart_count % len(nonzdelt_schedule)]
            sim = [seed_point.copy()]
            for k in range(n):
                y = seed_point.copy()
                if y[k] != 0:
                    y[k] = (1 + nonzdelt) * y[k]
                else:
                    y[k] = zdelt
                sim.append(y)
            sim = [_A.clip(v, 0, 1) for v in sim]

            # Evaluate the new simplex.
            fsim = [0.0] * (n + 1)
            for k in range(n + 1):
                if self.evaluations >= self.n_trials:
                    break
                fsim[k] = self.evaluate(sim[k])
            order = sorted(range(n + 1), key=fsim.__getitem__)
            sim = [sim[i] for i in order]
            fsim = [fsim[i] for i in order]

            # Standard NM inner loop until budget exhausted or simplex
            # collapses.
            while self.evaluations < self.n_trials:
                # SciPy convergence check, restated without numpy broadcasting:
                # max over all (i, k) of |sim[i][k] - sim[0][k]| <= xatol AND
                # max over i of |fsim[0] - fsim[i]| <= fatol.
                x_max = max(
                    abs(sim[i][k] - sim[0][k])
                    for i in range(1, n + 1)
                    for k in range(n)
                )
                f_max = max(abs(fsim[0] - fsim[i]) for i in range(1, n + 1))
                if x_max <= xatol and f_max <= fatol:
                    # Simplex collapsed — break the inner loop so the
                    # outer restart loop reseeds and continues.
                    break

                # Centroid of the best n vertices (sum-of-rows / n).
                xbar = _A.zeros(n)
                for v in sim[:-1]:
                    xbar = xbar + v
                xbar = xbar / n

                # Reflection.
                xr = _A.clip((1 + rho) * xbar - rho * sim[-1], 0, 1)
                if self.evaluations >= self.n_trials:
                    break
                fxr = self.evaluate(xr)

                if fxr < fsim[0]:
                    # Expansion.
                    xe = _A.clip((1 + rho * chi) * xbar - rho * chi * sim[-1], 0, 1)
                    if self.evaluations >= self.n_trials:
                        break
                    fxe = self.evaluate(xe)
                    if fxe < fxr:
                        sim[-1] = xe
                        fsim[-1] = fxe
                    else:
                        sim[-1] = xr
                        fsim[-1] = fxr
                elif fxr < fsim[-2]:
                    # Accept reflection.
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:
                    # Contraction (outside if fxr < fsim[-1], else inside).
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    else:
                        xc = (1 - psi) * xbar + psi * sim[-1]
                    xc = _A.clip(xc, 0, 1)

                    if self.evaluations >= self.n_trials:
                        break
                    fxc = self.evaluate(xc)

                    if fxc < min(fxr, fsim[-1]):
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        # Shrink: every non-best vertex moves toward sim[0].
                        for j in range(1, n + 1):
                            sim[j] = _A.clip(sim[0] + sigma * (sim[j] - sim[0]), 0, 1)
                            if self.evaluations < self.n_trials:
                                fsim[j] = self.evaluate(sim[j])

                # Re-sort by fitness.
                order = sorted(range(n + 1), key=fsim.__getitem__)
                sim = [sim[i] for i in order]
                fsim = [fsim[i] for i in order]

            # Inner loop ended — either budget exhausted or simplex
            # collapsed. If budget remains, alternate restart seeds: even
            # restarts reseed around the current best (intensification);
            # odd restarts reseed from a fresh uniform draw
            # (diversification). This mirrors the "two-phase" restart
            # heuristic widely used in NM++ implementations.
            restart_count += 1
            if self.evaluations >= self.n_trials:
                break
            if restart_count % 2 == 1:
                seed_point = sim[0].copy()
            else:
                seed_point = _A.random_uniform(n)

        return self.best_value, self.best_x


# ---- Batch 1 frozen copies (search_algorithms + simple evolutionary) ----


class FrozenRechenberg(BaseOptimizer):
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
        # Step bounds: 1e-12 floor lets the algorithm refine to machine
        # precision on smooth basins. Upper cap matches the unit cube.
        step_max = 1.0
        step_min = 1e-12
        window_size = 10

        x = _A.random_uniform(self.n_dim)
        f = self.evaluate(x)
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
                    sigma = min(step_max, sigma * 1.5)
                elif rate < 1 / 5:
                    sigma = max(step_min, sigma / 1.5)

        return self.best_value, self.best_x


class FrozenCoordinateDescent(BaseOptimizer):
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
        # Restart trigger: when `step` collapses below this threshold
        # and the current f is still above the "converged" threshold,
        # the run is stuck in a local basin and won't recover.
        # Reinitialise from a random point with step = 0.1.
        # Closes Ackley trapping (was median 1.29, 8/16 seeds stuck;
        # now 4.4e-16, 3/16 seeds stuck) at the cost of a small sphere
        # regression (1.8e-18 → 4.2e-13, both still tie reference 0).
        # Triggering earlier than 1e-12 means we can fit more restart
        # attempts in the budget.
        restart_step_threshold = 1e-6
        converged_threshold = 1e-8

        while self.evaluations < self.n_trials:
            if step <= restart_step_threshold:
                if f > converged_threshold:
                    x = _A.random_uniform(n)
                    f = self.evaluate(x)
                    step = 0.1
                    continue
                break  # already converged in a good basin

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


class FrozenPatternSearch(BaseOptimizer):
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
        # Restart trigger (see CoordinateDescent for the rationale): when
        # `step` collapses below this threshold and f hasn't reached the
        # converged threshold, reinitialise from a random base.
        restart_step_threshold = 1e-6
        converged_threshold = 1e-8

        while self.evaluations < self.n_trials:
            if step <= restart_step_threshold:
                if f_base > converged_threshold:
                    base = _A.random_uniform(self.n_dim)
                    f_base = self.evaluate(base)
                    step = 0.1
                    continue
                break  # already converged

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


class FrozenGridSearch(BaseOptimizer):
    """Regular-grid baseline.

    Evaluates a uniform Cartesian grid over the unit cube `[0, 1]^n_dim`
    with `n_per_axis = round(n_trials^(1/n_dim))` points per axis. Each
    axis is split into equal-width bins; the evaluated point in each bin
    is the bin centre `(i + 0.5) / n_per_axis`.

    Like RandomSearch, this is included as a baseline (regression check,
    contest sanity floor), not as a SOTA algorithm.

    Note: grid size scales as `n_per_axis^n_dim`. For modest budgets and
    `n_dim >= 5` the grid degenerates to fewer than 2 points per axis,
    making the algorithm equivalent to evaluating a handful of corners.
    Practically useful for `n_dim <= 3`.
    """

    def optimize(self):
        n = self.n_dim
        n_per_axis = max(2, int(round(self.n_trials ** (1.0 / n))))

        # Lexicographic enumeration of indices across n axes, each in
        # [0, n_per_axis). Bin-centred coordinates lie at (idx + 0.5) /
        # n_per_axis, so they are evenly spread inside [0, 1] without
        # ever sitting exactly on the bounds.
        indices = [0] * n
        while self.evaluations < self.n_trials:
            x = _A.asarray([(idx + 0.5) / n_per_axis for idx in indices])
            self.evaluate(x)
            # Increment indices like an odometer; stop once all wrap.
            d = n - 1
            while d >= 0:
                indices[d] += 1
                if indices[d] < n_per_axis:
                    break
                indices[d] = 0
                d -= 1
            if d < 0:
                break  # full grid exhausted

        return self.best_value, self.best_x


class FrozenRandomSearch(BaseOptimizer):
    """Random Search algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    """

    def optimize(self):
        while self.evaluations < self.n_trials:
            x = _A.random_uniform(self.n_dim)
            self.evaluate(x)

        return self.best_value, self.best_x


class FrozenEvolutionStrategy(BaseOptimizer):
    """(μ + λ)-Evolution Strategy.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population is a Python list of 1-D vectors (always was, even in the
    pre-port version), so this port is mostly substituting RNG calls.
    """

    def optimize(self):
        mu = 10  # Parents
        lambda_ = min(30, self.n_trials // 3)  # Offspring
        sigma = 0.2  # Mutation strength

        # Initialize μ parents.
        population = []
        fitness = []
        for _ in range(mu):
            if self.evaluations >= self.n_trials:
                break
            individual = _A.random_uniform(self.n_dim)
            f = self.evaluate(individual)
            population.append(individual)
            fitness.append(f)

        while self.evaluations < self.n_trials:
            # Generate λ offspring by mutating randomly-chosen parents.
            offspring = []
            offspring_fitness = []

            for _ in range(lambda_):
                if self.evaluations >= self.n_trials:
                    break

                parent_idx = _A.random_int(len(population))
                parent = population[parent_idx]

                child = _A.clip(parent + sigma * _A.random_normal(self.n_dim), 0, 1)

                child_fitness = self.evaluate(child)
                offspring.append(child)
                offspring_fitness.append(child_fitness)

            if offspring:
                # (μ + λ) selection: keep the best μ across both pools.
                # Replaces numpy's `np.argsort(all_fitness)[:mu]` with a
                # standard-library equivalent — no shim primitive needed.
                all_individuals = population + offspring
                all_fitness = fitness + offspring_fitness
                indices = sorted(range(len(all_fitness)), key=all_fitness.__getitem__)[
                    :mu
                ]
                population = [all_individuals[i] for i in indices]
                fitness = [all_fitness[i] for i in indices]

        return self.best_value, self.best_x


class FrozenHillClimbing(BaseOptimizer):
    """Hill climbing with a geometric sigma-decay schedule — equivalent
    to a (1+1)-Evolution Strategy with a deterministic step-size
    schedule (no 1/5-rule; see `Rechenberg` for that variant).

    Pure-Python via the `humpday._array` shim — no direct numpy use.

    Step size geometrically decays from `sigma_init = 0.1` to
    `sigma_final = 1e-3` over the budget. This is the standard
    textbook hill-climbing reference and is what
    `tests/test_reference_alignment.py::_ref_oneplusone_es_decay`
    compares humpday against. The previous implementation had a fixed
    `step_size = 0.1` (so the algorithm could never refine below ~1e-2
    precision) and a 10% random restart on each unimproved step (a
    humpday-ism not in any reference), which together left it ~2400×
    behind the reference on the sphere benchmark.
    """

    def optimize(self):
        n = self.n_dim
        x = _A.random_uniform(n)
        fx = self.evaluate(x)

        sigma_init = 0.1
        sigma_final = 1e-3
        # Geometric decay so that after `n_trials - 1` iterations
        # sigma == sigma_final. Matches the reference adapter
        # line-for-line.
        decay = (sigma_final / sigma_init) ** (1.0 / max(1, self.n_trials - 1))
        sigma = sigma_init

        while self.evaluations < self.n_trials:
            z = _A.random_normal(n)
            x_new = _A.clip(x + sigma * z, 0, 1)
            fx_new = self.evaluate(x_new)
            if fx_new < fx:
                x, fx = x_new, fx_new
            sigma *= decay

        return self.best_value, self.best_x


class FrozenHarmonySearch(BaseOptimizer):
    """Harmony Search algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    """

    def optimize(self):
        HMS = min(20, max(5, self.n_dim * 2))  # Harmony Memory Size
        HMCR = 0.9  # Harmony Memory Considering Rate
        PAR = 0.3  # Pitch Adjusting Rate

        # Initialize harmony memory.
        harmony_memory = []
        for _ in range(HMS):
            if self.evaluations >= self.n_trials:
                break
            harmony = _A.random_uniform(self.n_dim)
            fitness = self.evaluate(harmony)
            harmony_memory.append({"harmony": harmony, "fitness": fitness})

        while self.evaluations < self.n_trials:
            new_harmony = _A.zeros(self.n_dim)

            for j in range(self.n_dim):
                if _A.random_scalar() < HMCR:
                    # Pick from harmony memory.
                    selected = random.choice(harmony_memory)
                    value = selected["harmony"][j]

                    # Pitch adjustment.
                    if _A.random_scalar() < PAR:
                        # Add a single Gaussian-distributed nudge with sigma=0.1.
                        # `random_normal(1)[0]` is one draw from the shim's RNG;
                        # `0.1 *` scales it.
                        value = max(0.0, min(1.0, value + 0.1 * _A.random_normal(1)[0]))

                    new_harmony[j] = value
                else:
                    # Random selection along this dimension.
                    new_harmony[j] = _A.random_scalar()

            new_fitness = self.evaluate(new_harmony)

            # Update harmony memory (replace worst if new harmony is better).
            harmony_memory.sort(key=lambda x: x["fitness"])
            if new_fitness < harmony_memory[-1]["fitness"]:
                harmony_memory[-1] = {
                    "harmony": new_harmony.copy(),
                    "fitness": new_fitness,
                }

        return self.best_value, self.best_x
