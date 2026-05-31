"""
Evolutionary algorithm implementations.

These algorithms are inspired by natural evolution processes and include
methods like Differential Evolution, Genetic Algorithms, and Evolution Strategies.
They excel at global optimization and handling multimodal landscapes.
"""

import math
import random

from humpday import _array as _A

from .base import BaseOptimizer


class DifferentialEvolution(BaseOptimizer):
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


class ParticleSwarm(BaseOptimizer):
    """Particle Swarm Optimization.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Swarm stored as Python lists of 1-D vectors (FireflyAlgorithm pattern).
    """

    def optimize(self):
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
        personal_best_fit = [self.evaluate(p) for p in positions]

        max_iterations = max(1, pso_budget // swarm_size)

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

                fitness = self.evaluate(positions[i])

                # Personal-best bookkeeping.
                if fitness < personal_best_fit[i]:
                    personal_best_fit[i] = fitness
                    personal_best_pos[i] = positions[i].copy()

        # Polish stage: L-BFGS-B from the swarm best.
        self._lbfgs_polish()

        return self.best_value, self.best_x


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

    def optimize(self):
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

            fx = self.evaluate(x)

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
                new_fx = self.evaluate(new_x)

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
        self._lbfgs_polish()

        return self.best_value, self.best_x


class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population stored as a Python list of 1-D vectors. Selection is
    tournament-of-3; crossover is one-point; mutation is per-coordinate
    Bernoulli with uniform noise.
    """

    def optimize(self):
        pop_size = min(50, max(20, self.n_dim * 4))
        mutation_rate = 0.1
        crossover_rate = 0.8

        # Initialize population.
        population = [_A.random_uniform(self.n_dim) for _ in range(pop_size)]
        fitness = [self.evaluate(ind) for ind in population]

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

                fitness_val = self.evaluate(child)
                new_population.append(child)
                new_fitness.append(fitness_val)

            population = new_population
            fitness = new_fitness

        return self.best_value, self.best_x

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

    def optimize(self):
        while self.evaluations < self.n_trials:
            x = _A.random_uniform(self.n_dim)
            self.evaluate(x)

        return self.best_value, self.best_x


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

    def optimize(self):
        n_initial = min(5, max(2, self.n_dim))

        for _ in range(n_initial):
            if self.evaluations >= self.n_trials:
                break
            x = _A.random_uniform(self.n_dim)
            y = self.evaluate(x)
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
                y_next = self.evaluate(x_next)
            except Exception:
                # Fallback: any failure in the GP machinery falls back
                # to a random sample. Keeps the budget tight.
                x_next = _A.random_uniform(self.n_dim)
                y_next = self.evaluate(x_next)
            self.X_observed.append(x_next)
            self.y_observed.append(float(y_next))

        # Polish: L-BFGS-B from the GP-EI best. Closes the residual ~5
        # orders of magnitude on sphere by escaping the GP's RBF
        # smoothing floor.
        self._lbfgs_polish()

        return self.best_value, self.best_x

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

    def optimize(self):
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

        # Hansen's recommended parameters.
        lambda_ = min(50, 4 + int(3 * math.log(n)))  # population size
        mu = lambda_ // 2  # number of parents

        # Recombination weights: w_i = log(mu + 0.5) - log(i), normalised.
        weights = _A.asarray([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        weights = weights / _A.sum(weights)
        mueff = 1.0 / _A.sum(weights**2)

        # Adaptation constants.
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # State. Initial mean is a random interior point in [0.3, 0.7]^n
        # — same distribution the reference-alignment harness draws from
        # via `_draw_x0`. The previous fixed-centre `0.5 * ones(n)` was a
        # deterministic starting point that disadvantaged Rosenbrock
        # (optimum at 0.75 ones, so distance 0.25) vs the reference's
        # average of ~0.05. Also `sigma=0.2` to match the reference
        # cmaes library's chosen initial step size (HumpDay was 0.3).
        mean = 0.3 + 0.4 * _A.random_uniform(n)
        sigma = 0.2
        C = _A.linalg.eye(n)
        pc = _A.zeros(n)
        ps = _A.zeros(n)
        invsqrtC = _A.linalg.eye(n)

        generation = 0
        # Cap iterations by budget directly. The previous
        # `min(100, n_trials // lambda_)` capped at 100 generations even
        # when the user's n_trials budget allowed many more — at
        # lambda_ ≈ 6 and budget 1000, only the first ~600 evals would
        # be spent. Reference pycma has no such cap; the inner
        # `evaluations < n_trials` guard is sufficient, this just
        # protects against pathological infinite loops.
        max_generations = self.n_trials

        while self.evaluations < cmaes_budget and generation < max_generations:
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

            population = []
            for _ in range(lambda_):
                if self.evaluations >= cmaes_budget:
                    break
                std_z = _A.random_normal(n)
                z = _A.linalg.matvec(L_C, std_z)
                x = _A.clip(mean + sigma * z, 0, 1)
                f = self.evaluate(x)
                population.append((x, z, f))

            if not population:
                break
            # If the budget ran out mid-sampling and we have fewer than μ
            # offspring, we can't do a meaningful recombination — stop
            # here so the partial generation doesn't pollute the next
            # iteration's mean/sigma. Before #155 the
            # `min(100, n_trials // lambda_)` cap on the outer loop made
            # this case unreachable; now that the cap is gone we need to
            # guard explicitly.
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

            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * _A.linalg.matvec(
                invsqrtC, y
            )

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
            try:
                D, B = _A.linalg.eigh(C)
                D_inv_sqrt = [1.0 / math.sqrt(max(d, 1e-14)) for d in D]
                # invsqrtC = B @ diag(D_inv_sqrt) @ B.T
                D_diag = _A.linalg.diag(D_inv_sqrt)
                Bt = _A.linalg.transpose(B)
                tmp = _A.linalg.matmul(B, D_diag)
                invsqrtC = _A.linalg.matmul(tmp, Bt)
            except Exception:
                invsqrtC = _A.linalg.eye(n)

            # Step-size update. Do NOT floor at 1e-6 — that artificial
            # floor was preventing convergence on smooth basins
            # (Rosenbrock was 4.28× off the cmaes reference because
            # sigma got pinned at 1e-6 rather than shrinking further).
            # And do NOT cap at 0.5: reference pycma has no upper bound
            # on sigma; oversized proposals are handled by the
            # `_A.clip(..., 0, 1)` already applied to each x sample, so
            # the cap was throttling exploration without changing the
            # feasible search space.
            sigma = sigma * math.exp((cs / damps) * (_A.norm(ps) / math.sqrt(n) - 1))

        # Polish stage: L-BFGS-B from the CMA-ES best (shared on base).
        self._lbfgs_polish()

        return self.best_value, self.best_x


class FireflyAlgorithm(BaseOptimizer):
    """Firefly Algorithm.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population stored as a Python list of 1-D vectors (numpy.ndarray or
    `_Vec` depending on the active backend); all pairwise operations are
    elementwise.
    """

    def optimize(self):
        # Reserve budget for the L-BFGS-B polish stage (same pattern as
        # DE/SA/PSO/BayesianOpt). Firefly's stochastic dynamics converge
        # to the basin but stall in a noise floor — polish drives the
        # last few orders of magnitude.
        polish_reserve = min(20 * self.n_dim, self.n_trials // 2)
        firefly_budget = max(self.evaluations, self.n_trials - polish_reserve)

        n_fireflies = min(15, max(2, firefly_budget // 5))
        alpha = 0.2  # Randomness coefficient.
        beta0 = 1.0  # Attractiveness at zero distance.
        gamma = 1.0  # Light-absorption coefficient.

        # Initialize fireflies — list-of-vectors, NOT a 2-D array.
        fireflies = [_A.random_uniform(self.n_dim) for _ in range(n_fireflies)]
        intensities = [self.evaluate(f) for f in fireflies]

        while self.evaluations < firefly_budget:
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
                            intensities[i] = self.evaluate(fireflies[i])

        # Polish stage: L-BFGS-B from the firefly best.
        self._lbfgs_polish()

        return self.best_value, self.best_x


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

    def optimize(self):
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
            archive.append((x, self.evaluate(x)))
        if not archive:
            return self.best_value, self.best_x
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
                f_new = self.evaluate(x_new)
                new_solutions.append((x_new, f_new))

            # Merge: keep the k best across (archive ∪ new_solutions).
            archive.extend(new_solutions)
            archive.sort(key=lambda t: t[1])
            archive = archive[:k]

        return self.best_value, self.best_x


class EvolutionStrategy(BaseOptimizer):
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


class HarmonySearch(BaseOptimizer):
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
