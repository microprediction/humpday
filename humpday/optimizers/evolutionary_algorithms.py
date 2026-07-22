"""
Evolutionary algorithm implementations.

These algorithms are inspired by natural evolution processes and include
methods like Differential Evolution, Genetic Algorithms, and Evolution Strategies.
They excel at global optimization and handling multimodal landscapes.
"""

import math
import random

from humpday import _array as _A

from .base import BaseOptimizer, Batch


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population stored as a Python list of 1-D vectors.
    """

    def _run(self):
        # Online (generator) form: yields candidate points, receives their
        # values; the BaseOptimizer driver owns clipping and bookkeeping.
        # Statement order matches the pre-conversion optimize() exactly,
        # so seeded trajectories are identical (see
        # tests/test_online_pilot.py against the frozen reference).
        #
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
        fitness = []
        for ind in population:
            fitness.append((yield ind))

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
                trial_fitness = yield trial
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
        yield from self._lbfgs_polish_gen()


class ParticleSwarm(BaseOptimizer):
    """Particle Swarm Optimization.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Swarm stored as Python lists of 1-D vectors (FireflyAlgorithm pattern).
    """

    def _run(self):
        # Reserve budget for the L-BFGS-B polish stage. Same rationale
        # as DE/SA/BayesianOpt: PSO converges to the basin but doesn't
        # refine well past the noise floor of its inertial dynamics.
        # The polish closes the residual on smooth basins. Reserve
        # 20·n_dim evals capped at half the budget.
        polish_reserve = min(20 * self.n_dim, self.n_trials // 2)
        pso_budget = max(self.evaluations, self.n_trials - polish_reserve)

        swarm_size = min(40, max(15, self.n_dim * 3))

        # Initialize swarm — list-of-vectors instead of 2-D arrays.
        positions = [_A.random_uniform(self.n_dim) for _ in range(swarm_size)]
        velocities = [
            (_A.random_uniform(self.n_dim) - 0.5) * 0.2 for _ in range(swarm_size)
        ]
        personal_best_pos = [p.copy() for p in positions]
        personal_best_fit = []
        for p in positions:
            personal_best_fit.append((yield p))

        max_iterations = max(1, pso_budget // swarm_size)

        # SPSO-2011 style stagnation detection. A canonical PSO has no
        # restart and is known to converge prematurely on multimodal
        # surfaces — once the global best stops improving, particles
        # collapse onto it and there is no mechanism to escape. Track the
        # global-best value across iterations and, when it has stalled
        # for `stagnation_window` iterations, reseed the worst half of
        # the swarm with fresh uniform positions and small random
        # velocities. The personal-best memory of the *kept* half is
        # preserved so prior progress isn't wasted.
        stagnation_window = max(10, max_iterations // 5)
        stagnation_counter = 0
        last_global_best = self.best_value
        # Tolerate tiny floating-point noise as "no improvement".
        improvement_atol = 1e-12

        for iteration in range(max_iterations):
            if self.evaluations >= pso_budget:
                break

            # Adaptive coefficients (anneal inertia / explore-exploit balance).
            w = 0.9 - 0.5 * (iteration / max_iterations)  # Inertia weight
            c1 = 2.5 - 1.0 * (iteration / max_iterations)  # Cognitive
            c2 = 1.5 + 1.0 * (iteration / max_iterations)  # Social

            for i in range(swarm_size):
                if self.evaluations >= pso_budget:
                    break

                # Update velocity (elementwise: r1, r2 are length-n_dim
                # uniform vectors; _Vec/ndarray both support these ops).
                r1 = _A.random_uniform(self.n_dim)
                r2 = _A.random_uniform(self.n_dim)
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best_pos[i] - positions[i])
                    + c2 * r2 * (self.best_x - positions[i])
                )

                # Velocity clamping to [-vmax, +vmax].
                vmax = 0.2 * (1 - 0.5 * iteration / max_iterations)
                velocities[i] = _A.clip(velocities[i], -vmax, vmax)

                # Update position with bounds clipping.
                positions[i] = _A.clip(positions[i] + velocities[i], 0, 1)

                fitness = yield positions[i]

                # Personal-best bookkeeping.
                if fitness < personal_best_fit[i]:
                    personal_best_fit[i] = fitness
                    personal_best_pos[i] = positions[i].copy()

            # Stagnation check — measured against self.best_value because
            # that's the global best across all evals (including any
            # better point hit during the inner sweep).
            if last_global_best - self.best_value > improvement_atol:
                stagnation_counter = 0
                last_global_best = self.best_value
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_window:
                # Reseed the worst half. Rank particles by personal_best_fit
                # (ascending) and replace the bottom half with fresh
                # uniform draws + small random velocities. The top half
                # keeps its memory, so the swarm continues from the same
                # global best but with re-energized exploration.
                ranked = sorted(range(swarm_size), key=personal_best_fit.__getitem__)
                worst = ranked[swarm_size // 2 :]
                for j in worst:
                    positions[j] = _A.random_uniform(self.n_dim)
                    velocities[j] = (_A.random_uniform(self.n_dim) - 0.5) * 0.2
                    if self.evaluations >= pso_budget:
                        break
                    f_new = yield positions[j]
                    personal_best_pos[j] = positions[j].copy()
                    personal_best_fit[j] = f_new
                stagnation_counter = 0
                last_global_best = self.best_value

        # Polish stage: L-BFGS-B from the swarm best.
        yield from self._lbfgs_polish_gen()


class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing with multi-restart + coordinate-descent polish.

    Two-stage algorithm matching the spirit of scipy.optimize.dual_annealing:

      1. Multi-restart Metropolis SA explores the parameter space
         globally with a geometric cooling schedule.
      2. A coordinate-descent polish from the best SA point refines
         to high precision.

    scipy's dual_annealing uses L-BFGS-B for stage 2; the closest
    derivative-free equivalent is a coordinate descent with shrinking
    step. Without the polish stage HumpDay's SA was 1e9-1e11× off scipy
    on sphere and Rosenbrock; the global SA can't reach machine
    precision because its proposals are noisy.
    """

    def _run(self):
        # Reserve ~30% of the budget for the polish phase.
        # Allocate half the budget to the L-BFGS-B polish, matching the
        # DE rationale (#197 + this PR). 50% sweet spot: SA rosenbrock
        # 2.8e-5 → 2.8e-8 (1000× better), sphere stays at machine prec.
        polish_budget = max(20, self.n_trials // 2)
        sa_budget = self.n_trials - polish_budget

        # --- Stage 1: multi-restart SA ---------------------------------
        num_restarts = max(3, sa_budget // 30)
        trials_per_restart = max(1, sa_budget // num_restarts)

        for restart in range(num_restarts):
            if self.evaluations >= sa_budget:
                break

            if restart == 0:
                x = 0.5 + (_A.random_uniform(self.n_dim) - 0.5) * 0.4
            else:
                x = _A.random_uniform(self.n_dim)

            fx = yield x

            # Fixed initial temperature, geometric cooling. Reaches
            # final_temp by the end of the restart's iteration count.
            initial_temp = 1.0
            final_temp = 1e-6
            cooling = (final_temp / initial_temp) ** (1.0 / max(1, trials_per_restart))
            temp = initial_temp

            for _iteration in range(trials_per_restart):
                if self.evaluations >= sa_budget:
                    break

                # Neighbour proposal: step scales with current temp.
                step_size = 0.4 * temp
                new_x = _A.clip(
                    x + (_A.random_uniform(self.n_dim) - 0.5) * 2 * step_size,
                    0,
                    1,
                )
                new_fx = yield new_x

                # Metropolis criterion.
                delta = new_fx - fx
                if delta < 0 or _A.random_scalar() < _A.exp(-delta / max(temp, 1e-12)):
                    x, fx = new_x, new_fx

                temp *= cooling

        # --- Stage 2: L-BFGS polish from best SA point -----------------
        # Matches scipy.dual_annealing exactly — scipy uses L-BFGS-B for
        # its local-search refinement. Two-loop recursion with FD
        # gradient (2·n_dim evals per iter) and Armijo line search; same
        # algorithm the `LBFGSB` optimizer uses. Replaces the previous
        # coordinate-descent polish, which couldn't handle curved
        # valleys like Rosenbrock and stalled around 1e-9 on the sphere
        # at small budgets. With LBFGS the polish reaches machine
        # precision on smooth basins in ~10 iterations.
        yield from self._lbfgs_polish_gen()


class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population stored as a Python list of 1-D vectors. Selection is
    tournament-of-3; crossover is one-point; mutation is per-coordinate
    Bernoulli with uniform noise.
    """

    def _run(self):
        pop_size = min(50, max(20, self.n_dim * 4))
        mutation_rate = 0.1
        crossover_rate = 0.8

        # Initialize population.
        population = [_A.random_uniform(self.n_dim) for _ in range(pop_size)]
        fitness = []
        for ind in population:
            fitness.append((yield ind))

        generations = self.n_trials // pop_size

        for _gen in range(generations):
            if self.evaluations >= self.n_trials:
                break

            new_population = []
            new_fitness = []

            for _i in range(pop_size):
                if self.evaluations >= self.n_trials:
                    break

                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)

                child = parent1.copy()

                # One-point crossover.
                if _A.random_scalar() < crossover_rate:
                    cross_point = _A.random_int(self.n_dim)
                    for j in range(cross_point, self.n_dim):
                        child[j] = parent2[j]

                # Per-coordinate mutation with uniform [-0.1, 0.1] noise.
                # Rewritten from numpy boolean-indexing (`child[mask] += ...`)
                # to an explicit loop for backend independence.
                for j in range(self.n_dim):
                    if _A.random_scalar() < mutation_rate:
                        child[j] = max(
                            0.0,
                            min(1.0, child[j] + (_A.random_scalar() - 0.5) * 0.2),
                        )

                fitness_val = yield child
                new_population.append(child)
                new_fitness.append(fitness_val)

            population = new_population
            fitness = new_fitness

    def tournament_selection(self, population, fitness):
        """Tournament-of-3: pick 3 distinct individuals, return a copy of
        the one with the lowest fitness."""
        tournament_size = 3
        competitors = _A.random_choice(
            len(population), k=tournament_size, replace=False
        )
        competitors = [int(c) for c in competitors]
        best_idx = min(competitors, key=lambda c: fitness[c])
        return population[best_idx].copy()


class RandomSearch(BaseOptimizer):
    """Random Search algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    """

    def _run(self):
        while self.evaluations < self.n_trials:
            x = _A.random_uniform(self.n_dim)
            yield x


class BayesianOpt(BaseOptimizer):
    """Bayesian Optimization with a Gaussian-Process surrogate and the
    Expected-Improvement acquisition.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    The kernel matrix and predictive computations, originally written
    in numpy with row-of-row broadcasting, are rewritten with explicit
    nested loops here. Each row of `X_observed` is a `_Vec` / ndarray;
    `X_observed` itself is a Python list of those rows. The numerical
    fallback path uses a small diagonal jitter (instead of pinv, which
    is not in the linalg shim) to keep the kernel positive-definite.
    """

    def __init__(self, objective, n_trials, n_dim):
        super().__init__(objective, n_trials, n_dim)
        self.X_observed = []  # list of 1-D vectors
        self.y_observed = []  # list of floats
        self.length_scale = 0.2
        self.signal_variance = 1.0
        self.noise_variance = 1e-6

    def _run(self):
        n_initial = min(5, max(2, self.n_dim))

        for _ in range(n_initial):
            if self.evaluations >= self.n_trials:
                break
            x = _A.random_uniform(self.n_dim)
            y = yield x
            self.X_observed.append(x)
            self.y_observed.append(float(y))

        # Reserve budget for the L-BFGS-B polish stage. Reference:
        # scikit-optimize's `gp_minimize` finishes with a
        # `minimize(method='L-BFGS-B')` polish on the best observation;
        # this is the same pattern.
        #
        # The polish takes 2·n_dim evals per gradient + a few per line
        # search; 5-10 L-BFGS iterations on a smooth basin are enough to
        # blow through the GP-EI noise floor (~1e-4) down to machine
        # precision. Reserve 20·n_dim evals (≈ 10 polish iterations),
        # capped at half the budget so very small budgets still get a
        # real GP-EI phase.
        polish_reserve = min(20 * self.n_dim, self.n_trials // 2)
        loop_budget = max(self.evaluations, self.n_trials - polish_reserve)

        # Bayesian optimization loop.
        while self.evaluations < loop_budget:
            try:
                x_next = self._optimize_acquisition()
            except Exception:
                # Fallback: any failure in the GP machinery falls back
                # to a random sample. Keeps the budget tight.
                x_next = _A.random_uniform(self.n_dim)
            y_next = yield x_next
            self.X_observed.append(x_next)
            self.y_observed.append(float(y_next))

        # Polish: L-BFGS-B from the GP-EI best. Closes the residual ~5
        # orders of magnitude on sphere by escaping the GP's RBF
        # smoothing floor.
        yield from self._lbfgs_polish_gen()

    # ---- GP machinery (no numpy, no broadcasting) ----

    def _kernel_matrix(self, X1_rows, X2_rows):
        """RBF kernel for every pair (X1_rows[i], X2_rows[j]).

        Two implementation paths:

        - Under the numpy backend, the kernel is built with numpy's
          broadcasting and vectorised `exp`. This is essential for
          BayesianOpt to keep CI fast — without it the test suite
          balloons from ~40 s to several minutes because BayesianOpt is
          invoked many times across the smoke-test sweeps.
        - Under the pure backend, the kernel is built with explicit
          nested loops. Correctness over speed; pure-backend BayesianOpt
          is intentionally an outlier on performance.

        The backend check happens once per call; numpy is only imported
        on the path where it's known available (the shim's `BACKEND`
        is set at import time).
        """
        scale_sq = self.length_scale * self.length_scale
        sig = self.signal_variance

        if _A.BACKEND == "numpy":
            # numpy backend: classic broadcasting form.
            import numpy as _np

            X1 = _np.asarray([list(r) for r in X1_rows], dtype=float)
            X2 = _np.asarray([list(r) for r in X2_rows], dtype=float)
            n1_sq = (X1 * X1).sum(axis=1).reshape(-1, 1)
            n2_sq = (X2 * X2).sum(axis=1).reshape(1, -1)
            sqdist = n1_sq + n2_sq - 2.0 * X1 @ X2.T
            return sig * _np.exp(-0.5 * sqdist / scale_sq)

        # Pure-Python backend: explicit O(n1 n2 d) loops via the shim's
        # matmul. Slow but no numpy required.
        norms1 = [sum(float(v) * float(v) for v in row) for row in X1_rows]
        norms2 = [sum(float(v) * float(v) for v in row) for row in X2_rows]
        X1_2d = [list(r) for r in X1_rows]
        X2_2d = [list(r) for r in X2_rows]
        X2T = _A.linalg.transpose(X2_2d)
        cross = _A.linalg.matmul(X1_2d, X2T)

        n1, n2 = len(X1_rows), len(X2_rows)
        out = []
        for i in range(n1):
            row = []
            cross_i = cross[i]
            n1_i = norms1[i]
            for j in range(n2):
                sq = n1_i + norms2[j] - 2.0 * float(cross_i[j])
                row.append(sig * math.exp(-0.5 * sq / scale_sq))
            out.append(row)
        return out

    def _gp_predict(self, x_query):
        """Posterior mean and std for a single query point `x_query`."""
        n_obs = len(self.X_observed)

        # K = k(X, X) + noise * I
        K = self._kernel_matrix(self.X_observed, self.X_observed)
        for i in range(n_obs):
            K[i][i] += self.noise_variance

        # K_s = k(X, x_query) as a column vector of length n_obs.
        K_s_col = [
            self._kernel_matrix([x_obs], [x_query])[0][0] for x_obs in self.X_observed
        ]

        # K_ss = k(x_query, x_query) — a single scalar.
        K_ss = self._kernel_matrix([x_query], [x_query])[0][0]

        # Solve K alpha = y via cholesky, with a jitter retry if SPD fails.
        jitter = 0.0
        L = None
        for attempt in range(4):
            try:
                L = _A.linalg.cholesky(K)
                break
            except Exception:
                jitter = max(1e-8, jitter * 10) if jitter > 0 else 1e-8
                for i in range(n_obs):
                    K[i][i] += jitter
        if L is None:
            # Pathological kernel — fall back to a flat prior.
            mu = sum(self.y_observed) / max(1, n_obs)
            var = 0.0
            for y in self.y_observed:
                var += (y - mu) ** 2
            var /= max(1, n_obs)
            return mu, math.sqrt(max(var, 1e-8))

        # alpha = K^-1 y, computed as L^-T (L^-1 y) via two triangular solves.
        # Our shim only exposes a general `solve`; that's still correct, just
        # not as fast.
        alpha = _A.linalg.solve(L, self.y_observed)
        Lt = _A.linalg.transpose(L)
        alpha = _A.linalg.solve(Lt, alpha)

        # Mean: mu = K_s . alpha.
        mu = sum(float(K_s_col[i]) * float(alpha[i]) for i in range(n_obs))

        # Variance: var = K_ss - K_s^T K^-1 K_s.
        # With L L^T = K, K^-1 K_s = L^-T (L^-1 K_s).
        v = _A.linalg.solve(L, K_s_col)
        v_dot_v = sum(float(vi) * float(vi) for vi in v)
        var = max(K_ss - v_dot_v, 1e-8)

        return mu, math.sqrt(var)

    # ---- Acquisition ----

    def _expected_improvement(self, x):
        mu, sigma = self._gp_predict(x)
        f_best = min(self.y_observed)
        improvement = f_best - mu - 0.01
        if sigma <= 0:
            return 0.0
        Z = improvement / sigma
        return improvement * _normal_cdf(Z) + sigma * _normal_pdf(Z)

    def _optimize_acquisition(self):
        """Optimize the acquisition function with random starts."""
        best_x = None
        best_ei = -float("inf")
        for _ in range(min(10, max(5, 2 * self.n_dim))):
            x = _A.random_uniform(self.n_dim)
            ei = self._expected_improvement(x)
            if ei > best_ei:
                best_ei = ei
                best_x = x
        if best_x is None:
            return _A.random_uniform(self.n_dim)
        return _A.clip(best_x, 0, 1)


# ---- Standard-normal CDF / PDF used by BayesianOpt's EI -----------------
#
# Module-level helpers — these are plain scalar math, kept outside the
# class so they're easy to inspect and don't accidentally pick up `self`.


def _normal_cdf(x):
    """Standard-normal CDF, scalar input. Uses the same Abramowitz-style
    approximation as the original numpy implementation."""
    sign = 1.0 if x >= 0 else -1.0
    return 0.5 * (1.0 + sign * math.sqrt(1.0 - math.exp(-2.0 * x * x / math.pi)))


def _normal_pdf(x):
    """Standard-normal PDF, scalar input."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class CMAEvolutionStrategy(BaseOptimizer):
    """CMA-ES with evolution paths and step-size adaptation.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Implementation follows Hansen's standard CMA-ES; the only deviation
    is that `np.random.multivariate_normal(0, C)` is replaced by the
    Cholesky-based sampling `z = cholesky(C) @ random_normal(n)`, which
    is well-known to be equivalent (and is how numpy implements it
    internally).
    """

    def _run(self):
        import math

        n = self.n_dim

        # Reserve budget for the L-BFGS-B polish at the end. CMA-ES
        # converges to a basin geometrically but its stochastic
        # proposals plateau in a noise floor governed by σ. The polish
        # closes the residual.
        #
        # Sweep at n_trials=200, n_dim=2 (8 seeds, median):
        #   polish_factor=0   sphere=5.7e-10  rb=0.147  ackley=9.2e-4
        #   polish_factor=10  sphere=0        rb=0.21   ackley=5.3e-4
        #   polish_factor=20  sphere=0        rb=0.067  ackley=6.7e-4
        # 20·n is the sweet spot: enough polish iterations to refine
        # past the σ noise floor while leaving CMA-ES enough generations
        # to find the basin. Matches the pattern in DE/SA/PSO/Firefly.
        polish_reserve = min(20 * n, self.n_trials // 2)
        cmaes_budget = max(self.evaluations, self.n_trials - polish_reserve)

        # IPOP-CMA-ES (Auger & Hansen 2005, "A Restart CMA Evolution
        # Strategy with Increasing Population Size", CEC 2005). Vanilla
        # CMA-ES converges to a single basin and has no mechanism to
        # escape — on multimodal landscapes it routinely returns local
        # optima. IPOP wraps the main CMA loop with: (a) standard
        # termination checks (σ floor, condition number, TolFun stagnation)
        # and (b) a restart that doubles λ and resets all state. Larger
        # populations explore more aggressively, so successive restarts
        # are progressively better at jumping basins.
        IPOP_INCPOPSIZE = 2.0
        IPOP_TOLFUN = 1e-12
        IPOP_TOLX_FACTOR = 1e-12
        IPOP_CONDITION_COV = 1e14
        IPOP_TOLFUN_HISTORY = 10  # plus 30*n/lambda; bounded below

        base_lambda = min(50, 4 + int(3 * math.log(n)))
        restart_count = 0

        # Carry the best across restarts via self.best_x / self.best_value
        # (which BaseOptimizer.evaluate updates automatically). No need
        # to thread it through manually.

        # Outer IPOP loop: keep restarting (with growing λ) until budget
        # is exhausted.
        while self.evaluations < cmaes_budget:
            # Hansen's recommended parameters at the current population size.
            lambda_ = min(
                cmaes_budget - self.evaluations,
                int(base_lambda * (IPOP_INCPOPSIZE**restart_count)),
            )
            lambda_ = max(lambda_, 4)
            mu = lambda_ // 2  # number of parents
            if mu < 1:
                break

            # Recombination weights: w_i = log(mu + 0.5) - log(i), normalised.
            weights = _A.asarray(
                [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
            )
            weights = weights / _A.sum(weights)
            mueff = 1.0 / _A.sum(weights**2)

            # Adaptation constants.
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            # Fresh state per restart. Initial mean is a random interior
            # point in [0.3, 0.7]^n — same distribution the
            # reference-alignment harness draws from via `_draw_x0`. The
            # previous fixed-centre `0.5 * ones(n)` was a deterministic
            # starting point that disadvantaged Rosenbrock (optimum at
            # 0.75 ones, so distance 0.25) vs the reference's average of
            # ~0.05. Also `sigma=0.2` to match the reference cmaes
            # library's chosen initial step size (HumpDay was 0.3).
            mean = 0.3 + 0.4 * _A.random_uniform(n)
            sigma = 0.2
            C = _A.linalg.eye(n)
            pc = _A.zeros(n)
            ps = _A.zeros(n)
            invsqrtC = _A.linalg.eye(n)

            # TolFun window: history of best-of-generation values for the
            # most recent K generations. Length grows with the restart's
            # population size per IPOP convention.
            tolfun_window = max(IPOP_TOLFUN_HISTORY, int(30 * n / lambda_))
            fbest_history: list[float] = []

            generation = 0
            # Cap iterations by budget directly. The previous
            # `min(100, n_trials // lambda_)` capped at 100 generations
            # even when the user's n_trials budget allowed many more —
            # at lambda_ ≈ 6 and budget 1000, only the first ~600 evals
            # would be spent. Reference pycma has no such cap; the inner
            # `evaluations < n_trials` guard is sufficient, this just
            # protects against pathological infinite loops.
            max_generations = self.n_trials
            converged = False

            while (
                self.evaluations < cmaes_budget
                and generation < max_generations
                and not converged
            ):
                generation += 1

                # Sample λ offspring from N(mean, sigma^2 * C). Use a
                # Cholesky factor L of C so x = mean + sigma * (L @ N(0, I)),
                # equivalent to numpy's `multivariate_normal(0, C)`.
                try:
                    L_C = _A.linalg.cholesky(C)
                except Exception:
                    # If C drifted non-SPD, fall back to identity sampling
                    # for this generation; the eigh-based recovery below
                    # will repair C before the next iteration.
                    L_C = _A.linalg.eye(n)

                # Synchronous generation: every offspring is sampled from the SAME
                # (mean, sigma, C), so the whole generation can be evaluated as one
                # batch with no change in behaviour. Under ask/tell this surfaces the
                # generation via suggest_batch() for parallel evaluation; in a direct
                # optimize() run evaluate_batch() is exactly a per-point loop. We
                # build all offspring first (same RNG order, same budget cutoff) then
                # evaluate them together.
                n_off = max(0, min(lambda_, cmaes_budget - self.evaluations))
                xs, zs = [], []
                for _ in range(n_off):
                    std_z = _A.random_normal(n)
                    z = _A.linalg.matvec(L_C, std_z)
                    x = _A.clip(mean + sigma * z, 0, 1)
                    xs.append(x)
                    zs.append(z)
                fs = (yield Batch(xs)) if xs else []
                population = [(xs[i], zs[i], fs[i]) for i in range(len(xs))]

                if not population:
                    break
                # If the budget ran out mid-sampling and we have fewer
                # than μ offspring, the partial generation can't do a
                # meaningful recombination — stop the inner loop so the
                # restart layer (or polish stage) gets the remaining
                # budget rather than letting noise pollute the next
                # iteration's mean/sigma.
                if len(population) < mu:
                    break

                # Sort offspring by fitness ascending.
                population.sort(key=lambda p: p[2])

                # Recombination: new mean is the weighted average of the
                # μ best offspring.
                old_mean = mean.copy()
                mean = _A.zeros(n)
                for i in range(mu):
                    mean = mean + weights[i] * population[i][0]

                # Evolution paths.
                y = (mean - old_mean) / sigma

                ps = (1 - cs) * ps + math.sqrt(
                    cs * (2 - cs) * mueff
                ) * _A.linalg.matvec(invsqrtC, y)

                hsig = (
                    1
                    if _A.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * generation))
                    < 1.4 + 2 / (n + 1)
                    else 0
                )

                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y

                # Adapt covariance matrix C.
                if len(population) >= mu:
                    # Rank-μ update: sum of weighted outer products.
                    weighted_diffs = _A.linalg.matrix_zeros(n, n)
                    for i in range(mu):
                        diff = (population[i][0] - old_mean) / sigma
                        w_outer = _A.linalg.outer(diff, diff)
                        for r in range(n):
                            for c in range(n):
                                weighted_diffs[r][c] += weights[i] * w_outer[r][c]

                    pc_outer = _A.linalg.outer(pc, pc)
                    new_C = _A.linalg.matrix_zeros(n, n)
                    base = 1 - c1 - cmu
                    for r in range(n):
                        for c in range(n):
                            new_C[r][c] = (
                                base * C[r][c]
                                + c1 * pc_outer[r][c]
                                + cmu * weighted_diffs[r][c]
                            )
                    C = new_C

                    # Ensure C stays positive definite — bump up by the
                    # smallest eigenvalue if needed.
                    try:
                        eigvals, _ = _A.linalg.eigh(C)
                        min_eig = min(eigvals)
                        if min_eig < 1e-14:
                            shift = 1e-14 - min_eig
                            for k in range(n):
                                C[k][k] += shift
                    except Exception:
                        pass

                # Refresh invsqrtC for the next iteration via eigendecomp:
                # C = B diag(D) B^T  =>  invsqrtC = B diag(1/sqrt(D)) B^T.
                # Also use the eigenvalues for IPOP's ConditionCov check.
                try:
                    D, B = _A.linalg.eigh(C)
                    D_inv_sqrt = [1.0 / math.sqrt(max(d, 1e-14)) for d in D]
                    D_diag = _A.linalg.diag(D_inv_sqrt)
                    Bt = _A.linalg.transpose(B)
                    tmp = _A.linalg.matmul(B, D_diag)
                    invsqrtC = _A.linalg.matmul(tmp, Bt)
                    eig_max = max(D)
                    eig_min = max(min(D), 1e-30)
                    cond_C = eig_max / eig_min
                except Exception:
                    invsqrtC = _A.linalg.eye(n)
                    eig_max = 1.0
                    cond_C = 1.0

                # Step-size update. Do NOT floor at 1e-6 — that artificial
                # floor was preventing convergence on smooth basins
                # (Rosenbrock was 4.28× off the cmaes reference because
                # sigma got pinned at 1e-6 rather than shrinking further).
                # And do NOT cap at 0.5: reference pycma has no upper
                # bound on sigma; oversized proposals are handled by the
                # `_A.clip(..., 0, 1)` already applied to each x sample.
                sigma = sigma * math.exp(
                    (cs / damps) * (_A.norm(ps) / math.sqrt(n) - 1)
                )

                # ---- IPOP termination checks ----
                # Maintain a rolling window of best-of-generation values
                # for the TolFun stagnation test.
                fbest_history.append(population[0][2])
                if len(fbest_history) > tolfun_window:
                    fbest_history.pop(0)

                # TolFun: the f-value range over the last `tolfun_window`
                # generations has collapsed to noise.
                if (
                    len(fbest_history) >= tolfun_window
                    and max(fbest_history) - min(fbest_history) < IPOP_TOLFUN
                ):
                    converged = True
                    continue

                # TolX: step-size combined with the largest principal
                # direction has fallen below numerical resolution.
                if sigma * math.sqrt(eig_max) < IPOP_TOLX_FACTOR:
                    converged = True
                    continue

                # ConditionCov: the search distribution has elongated to
                # the point where further updates are numerically unsafe.
                if cond_C > IPOP_CONDITION_COV:
                    converged = True
                    continue

            # Inner loop exited — either budget exhausted, partial
            # generation, or an IPOP termination check fired. If we
            # still have budget, restart with a larger population.
            restart_count += 1
            if not converged:
                # Budget-driven exit, not convergence. No more restarts
                # would help — fall through to the polish stage.
                break

        # Polish stage: L-BFGS-B from the best-found point across all
        # restarts (shared on base; self.best_x carries the global best).
        yield from self._lbfgs_polish_gen()


class FireflyAlgorithm(BaseOptimizer):
    """Firefly Algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population stored as a Python list of 1-D vectors (numpy.ndarray or
    `_Vec` depending on the active backend); all pairwise operations are
    elementwise.
    """

    def _run(self):
        # Reserve budget for the L-BFGS-B polish stage (same pattern as
        # DE/SA/PSO/BayesianOpt). Firefly's stochastic dynamics converge
        # to the basin but stall in a noise floor — polish drives the
        # last few orders of magnitude.
        polish_reserve = min(20 * self.n_dim, self.n_trials // 2)
        firefly_budget = max(self.evaluations, self.n_trials - polish_reserve)

        n_fireflies = min(15, max(2, firefly_budget // 5))
        alpha0 = 0.2  # Initial randomness coefficient.
        beta0 = 1.0  # Attractiveness at zero distance.
        gamma = 1.0  # Light-absorption coefficient.
        # Geometric damping of the randomness coefficient — matches
        # mealpy's FFA `alpha_damp` (default 0.99). The original Yang
        # 2009 paper anneals α to focus exploration early and
        # exploitation late; without damping the algorithm keeps
        # injecting large random jitter even after fireflies cluster
        # around the optimum, which is the snapshot's Ackley failure
        # mode (median 2.58, 3/8 seeds stuck in a wrong basin because
        # the constant α=0.2 kept proposing big steps away).
        alpha_damp = 0.99
        alpha = alpha0

        # Initialize fireflies — list-of-vectors, NOT a 2-D array.
        fireflies = [_A.random_uniform(self.n_dim) for _ in range(n_fireflies)]
        intensities = []
        for fly in fireflies:
            intensities.append((yield fly))

        while self.evaluations < firefly_budget:
            evals_at_sweep_start = self.evaluations
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if self.evaluations >= firefly_budget:
                        break

                    if intensities[j] < intensities[i]:  # j is brighter
                        r = _A.norm(fireflies[i] - fireflies[j])
                        beta = beta0 * _A.exp(-gamma * r * r)

                        # Move firefly i toward the brighter firefly j,
                        # with a small random jitter.
                        fireflies[i] = _A.clip(
                            fireflies[i]
                            + beta * (fireflies[j] - fireflies[i])
                            + alpha * _A.random_normal(self.n_dim),
                            0,
                            1,
                        )

                        if self.evaluations < firefly_budget:
                            intensities[i] = yield fireflies[i]
            # Anneal α at the end of each outer (i, j) sweep, matching
            # mealpy FFA's `dyn_alpha = alpha_damp * alpha`.
            alpha *= alpha_damp

            # Termination guard. evaluate() is reached only when some firefly is
            # strictly brighter than another; if a whole sweep makes no call, all
            # intensities are equal — the swarm has collapsed onto one point (a
            # common end state, since fireflies attract and clip to shared cube
            # corners) or the region is flat. No future sweep can differ, so the
            # loop would spin forever without consuming budget. Stop and polish.
            if self.evaluations == evals_at_sweep_start:
                break

        # Polish stage: L-BFGS-B from the firefly best.
        yield from self._lbfgs_polish_gen()


class AntColonyOpt(BaseOptimizer):
    """ACOR — Ant Colony Optimization for continuous domains (Socha &
    Dorigo, 2008).

    Pure-Python via the `humpday._array` shim — no direct numpy use.

    Maintains an *archive* of the `k` best solutions found so far,
    sorted by fitness. Each generation samples `n_ants` new candidates
    from a mixture of Gaussian kernels centred on the archive points;
    the kernel weight w_i is a Gaussian on rank i (so better-ranked
    archive entries are sampled-from more often), and the per-dimension
    width sigma_d is `xi` times the mean absolute deviation of the
    archive in that dimension. Better candidates replace the worst
    archive entries.

    Replaces humpday's previous discrete-bin "continuous via
    discretization" ACO — a humpday-ism that capped precision at
    ~1/n_nodes per dimension and was ~457× off the mealpy reference
    on the sphere benchmark.

    Reference: Socha, K. & Dorigo, M. (2008). "Ant colony optimization
    for continuous domains." European Journal of Operational Research
    185(3): 1155–1173. Matches the mealpy `swarm_based.ACOR.OriginalACOR`
    adapter used by `tests/test_reference_alignment.py`.
    """

    def _run(self):
        n = self.n_dim
        # Hansen-mealpy-style defaults.
        k = min(50, max(10, self.n_trials // 10))  # archive size
        n_ants = min(25, max(5, self.n_trials // 20))  # samples per gen
        q = 0.5  # selection-pressure parameter (smaller → greedier)
        xi = 1.0  # standard-deviation amplification

        # Pre-compute rank weights: w_i ∝ Gaussian on rank i.
        # w_i = (1 / (q k √(2π))) · exp(−(i)² / (2 q² k²)) for i = 0..k−1
        weights = []
        for i in range(k):
            ex = math.exp(-(i**2) / (2.0 * q * q * k * k))
            weights.append(ex / (q * k * math.sqrt(2.0 * math.pi)))
        wsum = sum(weights)
        weights = [w / wsum for w in weights]

        # Initial archive: k uniform samples (or as many as budget allows).
        archive = []  # list of (x, f), sorted ascending by f
        for _ in range(k):
            if self.evaluations >= self.n_trials:
                break
            x = _A.random_uniform(n)
            archive.append((x, (yield x)))
        if not archive:
            return
        archive.sort(key=lambda t: t[1])

        while self.evaluations < self.n_trials:
            # Per-dimension standard deviation = xi · mean |x_l[d] − x_i[d]|
            # for the chosen kernel i. Precompute the absolute-deviation
            # matrix once per generation (used for every sample).
            sigmas_by_kernel = []
            for i in range(len(archive)):
                xi_vec = archive[i][0]
                sigma = [0.0] * n
                for l in range(len(archive)):
                    if l == i:
                        continue
                    xl_vec = archive[l][0]
                    for d in range(n):
                        sigma[d] += abs(float(xl_vec[d]) - float(xi_vec[d]))
                denom = max(1, len(archive) - 1)
                for d in range(n):
                    sigma[d] = xi * sigma[d] / denom
                sigmas_by_kernel.append(sigma)

            new_solutions = []
            for _ in range(n_ants):
                if self.evaluations >= self.n_trials:
                    break
                # Roulette-pick a kernel (= an archive index) by weights.
                r = _A.random_scalar()
                cum = 0.0
                kernel_idx = len(archive) - 1
                for i, w in enumerate(weights[: len(archive)]):
                    cum += w
                    if r <= cum:
                        kernel_idx = i
                        break

                center = archive[kernel_idx][0]
                sigma = sigmas_by_kernel[kernel_idx]
                x_new = _A.zeros(n)
                for d in range(n):
                    s = max(sigma[d], 1e-12)
                    # Box-Muller via the shim's random_normal.
                    z = float(_A.random_normal(1)[0])
                    x_new[d] = max(0.0, min(1.0, float(center[d]) + s * z))
                f_new = yield x_new
                new_solutions.append((x_new, f_new))

            # Merge: keep the k best across (archive ∪ new_solutions).
            archive.extend(new_solutions)
            archive.sort(key=lambda t: t[1])
            archive = archive[:k]


class EvolutionStrategy(BaseOptimizer):
    """(μ + λ)-Evolution Strategy.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population is a Python list of 1-D vectors (always was, even in the
    pre-port version), so this port is mostly substituting RNG calls.
    """

    def _run(self):
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
            f = yield individual
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

                child_fitness = yield child
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


class HillClimbing(BaseOptimizer):
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

    def _run(self):
        n = self.n_dim
        x = _A.random_uniform(n)
        fx = yield x

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
            fx_new = yield x_new
            if fx_new < fx:
                x, fx = x_new, fx_new
            sigma *= decay


class HarmonySearch(BaseOptimizer):
    """Harmony Search algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    """

    def _run(self):
        HMS = min(20, max(5, self.n_dim * 2))  # Harmony Memory Size
        HMCR = 0.9  # Harmony Memory Considering Rate
        PAR = 0.3  # Pitch Adjusting Rate

        # Initialize harmony memory.
        harmony_memory = []
        for _ in range(HMS):
            if self.evaluations >= self.n_trials:
                break
            harmony = _A.random_uniform(self.n_dim)
            fitness = yield harmony
            harmony_memory.append({"harmony": harmony, "fitness": fitness})

        while self.evaluations < self.n_trials:
            new_harmony = _A.zeros(self.n_dim)

            for j in range(self.n_dim):
                if _A.random_scalar() < HMCR:
                    # Pick from harmony memory.
                    selected = _A.rng_choice(harmony_memory)
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

            new_fitness = yield new_harmony

            # Update harmony memory (replace worst if new harmony is better).
            harmony_memory.sort(key=lambda x: x["fitness"])
            if new_fitness < harmony_memory[-1]["fitness"]:
                harmony_memory[-1] = {
                    "harmony": new_harmony.copy(),
                    "fitness": new_fitness,
                }
