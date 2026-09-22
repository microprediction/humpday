"""
Evolutionary algorithm implementations.

These algorithms are inspired by natural evolution processes and include
methods like Differential Evolution, Genetic Algorithms, and Evolution Strategies.
They excel at global optimization and handling multimodal landscapes.
"""

import math
import random

from humpday import _array as _A
from humpday._prng import portable_exp, portable_log

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

        while True:
            before = self.evaluations

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
                    candidates = [
                        k for k in range(pop_size) if k != i and k != best_idx
                    ]
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

            # --- Polish stage: L-BFGS from the best DE point -----------
            # Matches scipy.differential_evolution's `polish=True` exactly —
            # scipy uses L-BFGS-B. SimulatedAnnealing got the same upgrade
            # in the previous commit (#188); same inlined two-loop recursion
            # + FD gradient + Armijo line search the LBFGSB optimizer uses.
            # Closes the residual sphere gap that coord descent couldn't
            # reach.
            yield from self._lbfgs_polish_gen()

            # The reserve is sized for the worst case, but the polish converges in eight to
            # eighteen evaluations and the remainder used to be forfeited: measured, DE spent 2,508
            # of a 5,000 budget and returned. Because the reserve is proportional to n_trials it
            # leaked about half at every budget.
            #
            # Re-split what is left and go round again. The first round is unchanged, so the split
            # tuned on 2-D Rosenbrock at n_trials=200 is reproduced exactly; later rounds only use
            # evaluations that were previously thrown away. `BaseOptimizer` keeps the best point
            # seen, so another round can fail to help but cannot make the answer worse.
            if self.evaluations >= self.n_trials or self.evaluations == before:
                break
            de_budget = self.n_trials - max(15, (self.n_trials - self.evaluations) // 2)
            if self.evaluations >= de_budget:
                break


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


# Generalized Simulated Annealing constants (Tsallis & Stariolo 1996; Xiang, Sun, Fan & Gong
# 1997), matching scipy.optimize.dual_annealing's defaults.
#
# The two factors below depend only on the visiting parameter, and one of them needs a log-gamma
# and a sine. Both are computed once here as literals rather than at run time, so the JavaScript
# twin can hold the same numbers without needing lgamma -- SimulatedAnnealing is in JS_EXACT and
# has to agree bit for bit.
_GSA_QV = 2.62  # visiting parameter; heavier tail as it rises, range (1, 3]
_GSA_QA = -5.0  # acceptance parameter
_GSA_T0 = 5230.0  # initial temperature
_GSA_RESTART_T = 0.1  # re-anneal below this
_GSA_TAIL_LIMIT = 1.0e8
_GSA_MIN_VISIT_BOUND = 1.0e-10
_GSA_FACTOR4P = 11.833986526687411  # sqrt(pi) * f2 / (f3 * (3 - qv))
_GSA_FACTOR6 = 8.054035404548971  # pi (1-f5) / sin(pi (1-f5)) / exp(lgamma(2-f5))
_GSA_T1 = 2.0737503625760247  # exp((qv-1) log 2) - 1


class SimulatedAnnealing(BaseOptimizer):
    """Generalized Simulated Annealing with an L-BFGS-B local search.

    The algorithm scipy.optimize.dual_annealing runs, not the spirit of it: a heavy-tailed
    Tsallis visiting distribution whose scale follows the temperature, generalised Metropolis
    acceptance, the T(i) = T0 (2^(qv-1) - 1) / ((i+2)^(qv-1) - 1) schedule, re-annealing when the
    temperature falls below the restart threshold, and a local search from the chain's best point.

    What was here before was classic Metropolis with uniform proposals and geometric cooling,
    which is a different algorithm: its proposals cannot make the long jumps a heavy tail gives,
    so it explores a neighbourhood rather than a space. Measured against scipy's dual_annealing
    on Rosenbrock it was 3,195,457 times behind (#407's reference gate).

    The local search is the real L-BFGS-B, which is what scipy uses for the same purpose.
    """

    def _run(self):
        n = self.n_dim
        lo = [0.0] * n
        hi = [1.0] * n
        span = [hi[i] - lo[i] for i in range(n)]

        # The stages alternate, re-splitting what is left each round (#338): a local search that
        # converges early must not forfeit the remainder.
        while self.evaluations < self.n_trials:
            before = self.evaluations
            chain_budget = self.n_trials - max(
                20, (self.n_trials - self.evaluations) // 2
            )

            x = _A.random_uniform(n)
            e = yield x
            iteration = 0

            while self.evaluations < chain_budget:
                s_step = float(iteration) + 2.0
                t2 = portable_exp((_GSA_QV - 1.0) * portable_log(s_step)) - 1.0
                temperature = _GSA_T0 * _GSA_T1 / t2
                iteration += 1

                if temperature < _GSA_RESTART_T:
                    # Re-anneal: the schedule has run its course, so start again rather than
                    # keep proposing steps the acceptance rule will almost always refuse.
                    x = _A.random_uniform(n)
                    e = yield x
                    iteration = 0
                    continue

                temperature_step = temperature / float(iteration)
                best_before_chain = self.best_value

                # The strategy chain: 2n steps, the first n moving every coordinate and the rest
                # one coordinate each, as in scipy.
                for j in range(2 * n):
                    if self.evaluations >= chain_budget:
                        break
                    x_visit = yield from self._gsa_visit(
                        x, j, temperature, lo, hi, span
                    )
                    e_new = yield x_visit
                    if e_new < e:
                        x, e = x_visit, e_new
                    else:
                        r = _A.rng_random()
                        pqv_temp = 1.0 - (
                            (1.0 - _GSA_QA) * (e_new - e) / temperature_step
                        )
                        if pqv_temp <= 0.0:
                            pqv = 0.0
                        else:
                            pqv = portable_exp(portable_log(pqv_temp) / (1.0 - _GSA_QA))
                        if r <= pqv:
                            x, e = x_visit, e_new

                # The local search runs after a chain that improved on the best point, which is
                # the "dual" in dual_annealing: the annealing proposes a basin and the local
                # method descends it, every chain rather than once at the end. Running it only
                # once per block is what left this six orders behind scipy on Rosenbrock even
                # with the right visiting distribution -- the annealing was finding the valley
                # and nothing was walking down it.
                if (
                    self.best_value < best_before_chain
                    and self.evaluations < chain_budget
                ):
                    yield from self._lbfgs_polish_gen()

            # And once more from the best point seen, with whatever is left.
            yield from self._lbfgs_polish_gen()

            if self.evaluations == before:
                break

    def _gsa_visit(self, x, step, temperature, lo, hi, span):
        """One draw from the Tsallis visiting distribution (Visita, reference [2] p. 405).

        A generator because it consumes no objective evaluations but has to sit inside one:
        `yield from` keeps the RNG draws in the same order as the JavaScript twin.
        """
        n = len(x)
        if False:  # pragma: no cover - makes this a generator without yielding a point
            yield None

        factor1 = portable_exp(portable_log(temperature) / (_GSA_QV - 1.0))
        factor4 = _GSA_FACTOR4P * factor1
        sigmax = portable_exp(
            -(_GSA_QV - 1.0) * portable_log(_GSA_FACTOR6 / factor4) / (3.0 - _GSA_QV)
        )

        def one_visit():
            a = _A.rng_gauss()
            b = _A.rng_gauss()
            den = portable_exp((_GSA_QV - 1.0) * portable_log(abs(b)) / (3.0 - _GSA_QV))
            return sigmax * a / den

        def wrap(value, i):
            a = value - lo[i]
            b = math.fmod(a, span[i]) + span[i]
            out = math.fmod(b, span[i]) + lo[i]
            if abs(out - lo[i]) < _GSA_MIN_VISIT_BOUND:
                out += _GSA_MIN_VISIT_BOUND
            return out

        if step < n:
            visits = [one_visit() for _ in range(n)]
            upper_sample = _A.rng_random()
            lower_sample = _A.rng_random()
            capped = []
            for v in visits:
                if v > _GSA_TAIL_LIMIT:
                    capped.append(_GSA_TAIL_LIMIT * upper_sample)
                elif v < -_GSA_TAIL_LIMIT:
                    capped.append(-_GSA_TAIL_LIMIT * lower_sample)
                else:
                    capped.append(v)
            return _A.asarray([wrap(capped[i] + float(x[i]), i) for i in range(n)])

        out = [float(v) for v in x]
        visit = one_visit()
        if visit > _GSA_TAIL_LIMIT:
            visit = _GSA_TAIL_LIMIT * _A.rng_random()
        elif visit < -_GSA_TAIL_LIMIT:
            visit = -_GSA_TAIL_LIMIT * _A.rng_random()
        index = step - n
        out[index] = wrap(visit + out[index], index)
        return _A.asarray(out)


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
        self._posterior_stamp = None
        self._posterior = None

    # How many observations the Gaussian process is allowed to condition on.
    #
    # The kernel solve is cubic in the number of points and happens once per iteration, so an
    # uncapped set makes the run quartic in the budget overall: measured at d=2, a budget of 200
    # took 0.7 seconds, 400 took 3.9 and 800 took 25.0 (#330). In the tournament that is not a
    # slow optimizer, it is a disqualified one -- the recorder's allowance runs out and the cell
    # records a timeout instead of a rating.
    #
    # 128 keeps the solve bounded while leaving the surrogate more points than a GP on a smooth
    # low-dimensional problem can usefully distinguish. What it keeps matters more than how many:
    # the best points, because that is where the optimum is, and the most recent, because that is
    # what the acquisition has just been told.
    _GP_MAX_OBSERVATIONS = 128

    def _conditioning_set(self):
        """The observations the GP conditions on: the best half and the newest half."""
        n = len(self.X_observed)
        if n <= self._GP_MAX_OBSERVATIONS:
            return self.X_observed, self.y_observed

        half = self._GP_MAX_OBSERVATIONS // 2
        by_value = sorted(range(n), key=self.y_observed.__getitem__)[:half]
        keep = set(by_value) | set(range(n - half, n))
        order = sorted(keep)
        return (
            [self.X_observed[i] for i in order],
            [self.y_observed[i] for i in order],
        )

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
        norms1 = [_A.fold_sum(float(v) * float(v) for v in row) for row in X1_rows]
        norms2 = [_A.fold_sum(float(v) * float(v) for v in row) for row in X2_rows]
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

    def _gp_posterior(self):
        """Factorise the kernel once for the current observation set, and cache it.

        The Cholesky factor of `k(X, X) + noise I` depends on the observations and nothing
        else, so it is the same for every point the acquisition asks about. It used to be
        rebuilt and refactorised inside `_gp_predict`, once per query -- an O(n^3) solve to
        answer an O(n^2) question, repeated for every candidate.

        That is why `_optimize_acquisition` could only afford ten candidates, and ten uniform
        samples do not find the narrow ridges of an expected-improvement surface. The cost of
        the cap was the algorithm: BayesianOpt sat at 2.58 on Ackley, which is where a run
        trapped on the ring sits, against gp_minimize's 0.032 (#81).

        Returns `(X_obs, L, alpha)`, or None when the kernel will not factorise.
        """
        X_obs, y_obs = self._conditioning_set()
        n_obs = len(X_obs)
        stamp = (n_obs, len(self.X_observed), self.length_scale)
        if self._posterior_stamp == stamp:
            return self._posterior

        K = self._kernel_matrix(X_obs, X_obs)
        for i in range(n_obs):
            K[i][i] += self.noise_variance

        jitter = 0.0
        L = None
        for _ in range(4):
            try:
                L = _A.linalg.cholesky(K)
                break
            except Exception:
                jitter = max(1e-8, jitter * 10) if jitter > 0 else 1e-8
                for i in range(n_obs):
                    K[i][i] += jitter

        if L is None:
            posterior = None
        else:
            # alpha = K^-1 y as L^-T (L^-1 y). The shim exposes only a general `solve`;
            # still correct, just not as fast as a triangular one.
            alpha = _A.linalg.solve(L, y_obs)
            alpha = _A.linalg.solve(_A.linalg.transpose(L), alpha)
            posterior = (X_obs, L, alpha)

        self._posterior_stamp = stamp
        self._posterior = posterior
        return posterior

    def _gp_predict(self, x_query):
        """Posterior mean and std for one query point, against the cached factor."""
        posterior = self._gp_posterior()
        if posterior is None:
            # Pathological kernel -- fall back to a flat prior over what has been seen.
            _, y_obs = self._conditioning_set()
            n_obs = max(1, len(y_obs))
            mu = _A.fold_sum(y_obs) / n_obs
            var = _A.fold_sum((y - mu) ** 2 for y in y_obs) / n_obs
            return mu, math.sqrt(max(var, 1e-8))

        X_obs, L, alpha = posterior
        n_obs = len(X_obs)

        # k(X, x_query) as a column of length n_obs; k(x_query, x_query) is the signal
        # variance, since the RBF kernel of a point with itself is exp(0).
        K_s_col = [row[0] for row in self._kernel_matrix(X_obs, [x_query])]
        K_ss = self.signal_variance

        mu = _A.fold_sum(float(K_s_col[i]) * float(alpha[i]) for i in range(n_obs))

        # var = K_ss - K_s^T K^-1 K_s, and with L L^T = K that is K_ss - |L^-1 K_s|^2.
        v = _A.linalg.solve(L, K_s_col)
        var = max(K_ss - _A.fold_sum(float(vi) * float(vi) for vi in v), 1e-8)

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

    # How hard the acquisition function is searched before a point is proposed.
    #
    # This was ten uniform samples, which is a random search of the acquisition surface rather
    # than an optimisation of it. Expected improvement concentrates on narrow ridges between
    # the observations, and ten draws in the cube miss them; on Ackley that left the port at
    # 2.58 against gp_minimize's 0.032, losing 0.90 of head-to-head pairings.
    #
    # scikit-optimize samples 10,000 candidates and then runs L-BFGS-B from the best five: a
    # broad look followed by a local one. The same shape is affordable here now that the kernel
    # is factorised once per iteration rather than once per candidate, and these two numbers are
    # the knee of it, measured over 41 seeds on Rosenbrock and Ackley as the summed fraction of
    # head-to-head pairings lost to gp_minimize (lower is better):
    #
    #     candidates  refinements   rosenbrock  ackley   sum   secs
    #             10            0         0.76    0.78  1.54    0.6   <- what this used to be
    #             16            3         0.64    0.53  1.16    1.5
    #             64            3         0.58    0.62  1.20    3.9
    #            128            3         0.57    0.49  1.06    7.0
    #            256            3         0.52    0.63  1.15   13.5
    #            256            6         0.56    0.46  1.02   14.2
    #             64            6         0.52    0.47  0.98    4.4
    #
    # Everything from 64 upward is the same within the noise of a rate on 41 samples; what the
    # table really shows is that escaping ten mattered and that refining is worth more than
    # sampling wider, which is the same lesson as scikit-optimize's L-BFGS-B step. So take the
    # cheapest of the indistinguishable ones rather than the widest: 64 candidates costs a
    # third of 256 and measures no worse.
    _ACQ_CANDIDATES = 64
    _ACQ_REFINEMENTS = 6

    def _optimize_acquisition(self):
        """Maximise expected improvement: a broad sample, then refine the best of it.

        Every evaluation here is of the surrogate, not the objective, so none of it is charged
        to the budget. The only cost is time, and the factorisation cache is what makes it
        cheap: a candidate is now O(n_obs^2) against the O(n_obs^3) it used to be.
        """
        best_x, best_ei = None, -float("inf")
        for _ in range(self._ACQ_CANDIDATES):
            x = _A.random_uniform(self.n_dim)
            ei = self._expected_improvement(x)
            if ei > best_ei:
                best_ei, best_x = ei, x

        if best_x is None:
            return _A.random_uniform(self.n_dim)

        # Local refinement: a coordinate pattern search on the surrogate, halving the step.
        # L-BFGS-B on the acquisition would need its gradient, and the acquisition's gradient
        # through a Cholesky solve is not something the array shim exposes.
        step = 0.25
        for _ in range(self._ACQ_REFINEMENTS):
            improved = False
            for i in range(self.n_dim):
                for sign in (1.0, -1.0):
                    trial = [float(v) for v in best_x]
                    trial[i] = min(1.0, max(0.0, trial[i] + sign * step))
                    ei = self._expected_improvement(trial)
                    if ei > best_ei:
                        best_ei, best_x, improved = ei, trial, True
            if not improved:
                step *= 0.5

        return _A.clip(_A.asarray(best_x), 0, 1)


# ---- Standard-normal CDF / PDF used by BayesianOpt's EI -----------------
#
# Module-level helpers — these are plain scalar math, kept outside the
# class so they're easy to inspect and don't accidentally pick up `self`.


def _normal_cdf(x):
    """Standard-normal CDF, scalar input.

    `math.erf` rather than the Abramowitz-style approximation this used to carry, whose worst
    error is about 1.4e-2 -- three digits of a quantity expected improvement then multiplies
    by. The reference, scikit-optimize, calls `scipy.stats.norm.cdf`. There was never a reason
    to approximate: `erf` is in the standard library and is exact to the last ulp.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
            # E||N(0, I_n)||, the scale every evolution-path test compares
            # against. Without it the hsig gate and the step-size update
            # read a normal path as excessive once n is more than a few.
            chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

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

                ps_norm = _A.norm(ps)
                hsig = (
                    1
                    if ps_norm / math.sqrt(1 - (1 - cs) ** (2 * generation))
                    < (1.4 + 2 / (n + 1)) * chi_n
                    else 0
                )

                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y

                # Adapt covariance matrix C.
                base = None
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
                    # When the path update was gated off (hsig = 0) the rank-one
                    # term is missing its variance; the standard recurrence
                    # compensates by keeping that much of the old C.
                    base = 1 - c1 - cmu + c1 * (1 - hsig) * cc * (2 - cc)
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
                sigma_before = sigma
                sigma = sigma * math.exp((cs / damps) * (ps_norm / chi_n - 1))
                # One generation's update, exposed so tests can check it
                # against the standard equations rather than against the
                # JavaScript twin, which could share the same mistake.
                self._cma_trace = {
                    "generation": generation,
                    "n": n,
                    "cs": cs,
                    "cc": cc,
                    "c1": c1,
                    "cmu": cmu,
                    "damps": damps,
                    "chi_n": chi_n,
                    "ps_norm": ps_norm,
                    "hsig": hsig,
                    "sigma_before": sigma_before,
                    "sigma_after": sigma,
                    "cov_base": base,
                }

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
                        # portable_exp, not libm exp: V8's Math.exp is
                        # fdlibm-derived and diverges from platform libm
                        # in the last ulp.
                        beta = beta0 * portable_exp(-gamma * r * r)

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
        wsum = _A.fold_sum(weights)
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


# Rechenberg's 1/5 rule, shared with FrozenEvolutionStrategy and the JavaScript twin so the three
# stay bit-exact. Schwefel's 0.817 factor; bounds strictly wide of the 0.2 starting value.
_ES_TARGET_SUCCESS = 0.2
_ES_ADAPT = 0.817
_ES_SIGMA_MIN = 1e-12
_ES_SIGMA_MAX = 0.5


class EvolutionStrategy(BaseOptimizer):
    """(μ + λ)-Evolution Strategy.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Population is a Python list of 1-D vectors (always was, even in the
    pre-port version), so this port is mostly substituting RNG calls.
    """

    def _run(self):
        mu = 10  # Parents
        lambda_ = min(30, self.n_trials // 3)  # Offspring

        # Mutation strength, adapted by Rechenberg's 1/5 success rule. It used to be assigned here
        # and never touched again, which made this a fixed-radius sampler rather than an evolution
        # strategy: step-size adaptation is the thing that distinguishes the two. Measured on a
        # sphere at d=4 it stalled at 1.8e-03 with 5,000 evaluations where NelderMead reached
        # 1.3e-25, and the extra budget bought three decimal places and then nothing.
        #
        # The rule: if more than a fifth of offspring beat their parent the search is making easy
        # progress and the step should grow; if fewer, it is overshooting and the step should
        # shrink. 0.817 is Schwefel's recommended factor. The bounds are deliberately wide of the
        # initial value on both sides -- a clamp at the value a variable starts from is how
        # PRIMA_UOBYQA's trust region came to be pinned for an entire run (#327).
        sigma = 0.2

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

            successes = 0

            for _ in range(lambda_):
                if self.evaluations >= self.n_trials:
                    break

                parent_idx = _A.random_int(len(population))
                parent = population[parent_idx]

                child = _A.clip(parent + sigma * _A.random_normal(self.n_dim), 0, 1)

                child_fitness = yield child
                if child_fitness < fitness[parent_idx]:
                    successes += 1
                offspring.append(child)
                offspring_fitness.append(child_fitness)

            if offspring:
                # Adapt before selection, so the rate refers to the parents that produced these
                # offspring. No RNG is drawn here, which keeps the call sequence identical to the
                # JavaScript twin.
                rate = successes / len(offspring)
                if rate > _ES_TARGET_SUCCESS:
                    sigma = min(sigma / _ES_ADAPT, _ES_SIGMA_MAX)
                elif rate < _ES_TARGET_SUCCESS:
                    sigma = max(sigma * _ES_ADAPT, _ES_SIGMA_MIN)

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
        # portable_exp/log, not ** : libm pow differs across platforms in
        # the last ulp (V8-on-Linux vs CPython caught by vector replay).
        decay = portable_exp(
            (1.0 / max(1, self.n_trials - 1)) * portable_log(sigma_final / sigma_init)
        )
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
