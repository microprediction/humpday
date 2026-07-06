"""A hand-written optimizer family parameterized by simplex weights.

This is the "it's just hyperparameters" null hypothesis, made as strong as
possible: ONE fixed architecture (taken from the winning LLM centroid
program, so it inherits the discovered glue) whose behaviour is driven by
the same slot shares the LLM prompts use.

  - move-generation probabilities over {NM, DE, CMA-ish, PatternSearch}
    come from weights_to_spec(w)["move_generation"];
  - the simulated-annealing acceptance temperature and restart eagerness
    scale with SA's raw weight (w_SA -> 0 gives greedy acceptance);
  - initialization is always a DE population (DE solely owns that slot);
  - each move's adaptation is tied to the move itself, as in the centroid.

If derivative-free search over w in THIS family matches the LLM blends, the
language model adds nothing beyond the shares. If the LLM blends win, the
semantic layer (per-point architecture and glue) carries real information.
"""

from __future__ import annotations

import math
import random


def make_portfolio(weights, spec):
    """Return optimize(objective, n_trials, n_dim) implementing the slot
    semantics of `spec` (from simplex_blend.weights_to_spec) with fixed
    architecture."""
    move_share = dict(spec["move_generation"])
    p_nm = move_share.get("NelderMead", 0.0)
    p_de = move_share.get("DifferentialEvolution", 0.0)
    p_cma = move_share.get("CMAEvolutionStrategy", 0.0)
    p_ps = move_share.get("PatternSearch", 0.0)
    tot = (p_nm + p_de + p_cma + p_ps) or 1.0
    p_nm, p_de, p_cma = p_nm / tot, p_de / tot, p_cma / tot

    w_sa = spec["inspiration"].get("SimulatedAnnealing", 0.0)

    def optimize(objective, n_trials, n_dim):
        if n_dim <= 0:
            return (objective([]) if n_trials > 0 else float("inf"), [])

        calls = 0
        best_val = float("inf")
        best_pt = [0.5] * n_dim

        def clip(x):
            return [0.0 if xi < 0.0 else (1.0 if xi > 1.0 else xi) for xi in x]

        def feval(x):
            nonlocal calls, best_val, best_pt
            xc = clip(x)
            v = objective(xc)
            calls += 1
            if v < best_val:
                best_val = v
                best_pt = xc[:]
            return v

        pop_size = max(n_dim + 1, min(8 + 2 * n_dim, max(5, n_trials // 6)))
        pop, pop_f = [], []
        for _ in range(pop_size):
            if calls >= n_trials:
                break
            x = [random.random() for _ in range(n_dim)]
            pop.append(x)
            pop_f.append(feval(x))
        if not pop:
            return (best_val, best_pt)

        order = sorted(range(len(pop)), key=lambda i: pop_f[i])
        simplex = [pop[i][:] for i in order[: n_dim + 1]]
        simplex_f = [pop_f[i] for i in order[: n_dim + 1]]
        while len(simplex) < n_dim + 1 and calls < n_trials:
            b = simplex[0][:]
            b[(len(simplex) - 1) % n_dim] = min(1.0, b[(len(simplex) - 1) % n_dim] + 0.1)
            simplex.append(b)
            simplex_f.append(feval(b))

        step, sigma = 0.25, 0.2
        cov_diag = [0.04] * n_dim
        F, CR = 0.6, 0.9

        # SA share drives temperature and restart eagerness. w_sa=0.2 (the
        # centroid) reproduces the centroid program's constants.
        f_lo, f_hi = min(simplex_f), max(simplex_f)
        T0 = max(1e-9, ((f_hi - f_lo) + 1e-3) * (w_sa / 0.2))
        T = T0
        patience = max(12, 4 * n_dim)

        def order_simplex():
            idx = sorted(range(len(simplex)), key=lambda i: simplex_f[i])
            return [simplex[i][:] for i in idx], [simplex_f[i] for i in idx]

        def centroid_of(pts, exclude):
            c = [0.0] * n_dim
            m = 0
            for i, p in enumerate(pts):
                if i == exclude:
                    continue
                for d in range(n_dim):
                    c[d] += p[d]
                m += 1
            return [ci / m for ci in c]

        def accept(f_new, f_old):
            if f_new <= f_old:
                return True
            if T <= 1e-12:
                return False
            try:
                return random.random() < math.exp(-(f_new - f_old) / T)
            except OverflowError:
                return False

        stagnation = 0
        while calls < n_trials:
            simplex, simplex_f = order_simplex()
            for i in range(len(simplex)):
                if simplex_f[i] < best_val:
                    best_val = simplex_f[i]
                    best_pt = simplex[i][:]
            worst_i = len(simplex) - 1
            best_x, worst_x, worst_f = simplex[0], simplex[worst_i], simplex_f[worst_i]
            cen = centroid_of(simplex, worst_i)

            r = random.random()
            improved = False
            if r < p_nm:
                refl = [cen[d] + (cen[d] - worst_x[d]) for d in range(n_dim)]
                if calls >= n_trials:
                    break
                fr = feval(refl)
                if fr < simplex_f[0]:
                    exp = [cen[d] + 2.0 * (cen[d] - worst_x[d]) for d in range(n_dim)]
                    if calls < n_trials:
                        fe = feval(exp)
                        cand, cf = (exp, fe) if fe < fr else (refl, fr)
                    else:
                        cand, cf = refl, fr
                elif fr < worst_f:
                    cand, cf = refl, fr
                else:
                    con = [cen[d] + 0.5 * (worst_x[d] - cen[d]) for d in range(n_dim)]
                    if calls < n_trials:
                        fc = feval(con)
                        if fc < worst_f:
                            cand, cf = con, fc
                        else:
                            for i in range(1, len(simplex)):
                                if calls >= n_trials:
                                    break
                                simplex[i] = [
                                    best_x[d] + 0.5 * (simplex[i][d] - best_x[d])
                                    for d in range(n_dim)
                                ]
                                simplex_f[i] = feval(simplex[i])
                            cand, cf = None, None
                    else:
                        cand, cf = None, None
                if cand is not None and accept(cf, worst_f):
                    simplex[worst_i] = clip(cand)
                    simplex_f[worst_i] = cf
                    improved = cf < worst_f
            elif r < p_nm + p_de:
                idxs = list(range(len(simplex)))
                random.shuffle(idxs)
                a, b, c = idxs[0], idxs[1 % len(idxs)], idxs[2 % len(idxs)]
                if random.random() < 0.5:
                    mut = [simplex[a][d] + F * (simplex[b][d] - simplex[c][d])
                           for d in range(n_dim)]
                else:
                    mut = [worst_x[d] + F * (best_x[d] - worst_x[d])
                           + F * (simplex[b][d] - simplex[c][d]) for d in range(n_dim)]
                jr = random.randrange(n_dim)
                trial = [mut[d] if (random.random() < CR or d == jr) else worst_x[d]
                         for d in range(n_dim)]
                if calls >= n_trials:
                    break
                ft = feval(trial)
                if accept(ft, worst_f):
                    simplex[worst_i] = clip(trial)
                    simplex_f[worst_i] = ft
                    improved = ft < worst_f
            elif r < p_nm + p_de + p_cma:
                cand = [best_x[d] + sigma * random.gauss(0.0, math.sqrt(cov_diag[d]))
                        for d in range(n_dim)]
                if calls >= n_trials:
                    break
                fc = feval(cand)
                if accept(fc, worst_f):
                    if fc < worst_f:
                        improved = True
                        for d in range(n_dim):
                            delta = cand[d] - best_x[d]
                            cov_diag[d] = 0.8 * cov_diag[d] + 0.2 * (delta * delta + 1e-8)
                    simplex[worst_i] = clip(cand)
                    simplex_f[worst_i] = fc
            else:
                cur, cur_f = best_x[:], simplex_f[0]
                base_f = cur_f
                for d in range(n_dim):
                    if calls >= n_trials:
                        break
                    t = cur[:]
                    t[d] += step
                    ft = feval(t)
                    if ft < cur_f:
                        cur, cur_f = clip(t), ft
                    else:
                        if calls >= n_trials:
                            break
                        t = cur[:]
                        t[d] -= step
                        ft = feval(t)
                        if ft < cur_f:
                            cur, cur_f = clip(t), ft
                if cur_f < base_f:
                    improved = True
                    if accept(cur_f, worst_f):
                        simplex[worst_i] = cur
                        simplex_f[worst_i] = cur_f
                else:
                    step *= 0.5
                    if step < 1e-6:
                        step = 0.25

            if improved:
                stagnation = 0
                sigma = min(0.5, sigma * 1.05)
            else:
                stagnation += 1
                sigma = max(1e-3, sigma * 0.97)
            T *= 0.995
            if T < 1e-9:
                T = 1e-9
            if stagnation > patience and calls < n_trials:
                stagnation = 0
                T = max(T0 * 0.5, T * 5.0)
                step, sigma = 0.25, 0.2
                keep = best_pt[:]
                simplex = [keep[:]]
                simplex_f = [best_val]
                for _ in range(n_dim):
                    if calls >= n_trials:
                        break
                    p = [min(1.0, max(0.0, keep[d] + random.uniform(-0.3, 0.3)))
                         for d in range(n_dim)]
                    simplex.append(p)
                    simplex_f.append(feval(p))
                while len(simplex) < n_dim + 1 and calls < n_trials:
                    p = [random.random() for _ in range(n_dim)]
                    simplex.append(p)
                    simplex_f.append(feval(p))
        return (best_val, best_pt)

    return optimize
