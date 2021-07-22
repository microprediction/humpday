import warnings
import numpy as np

# https://github.com/numericalalgorithmsgroup/pybobyqa/blob/master/pybobyqa/solver.py

try:
    from pybobyqa import solve
    using_bobyqa = True
except ImportError:
    using_bobyqa = False

if using_bobyqa:

    def bobyqa_cube_factory(objective, n_trials, n_dim, with_count, **kwargs):
        global feval_count
        feval_count = 0

        lb = np.array([0. for _ in range(n_dim)])
        ub = np.array([1. for _ in range(n_dim)])
        x0 = np.array([0.5]*n_dim)

        def _objective(u) -> float:
            global feval_count
            feval_count += 1
            return objective(u)

        soln = solve(_objective, x0, bounds=(lb, ub), maxfun=n_trials, do_logging=False)

        best_x, best_val = list(soln.x), soln.f

        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)

    def bobyqa_default_cube(objective, n_trials, n_dim, with_count):
        return bobyqa_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count)


    def bobyqa_noise_cube(objective, n_trials, n_dim, with_count):
        return bobyqa_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim,
                                   with_count=with_count, objfun_has_noise=True)


    BOBYQA_OPTIMIZERS = [ bobyqa_default_cube, bobyqa_noise_cube ]
else:
    BOBYQA_OPTIMIZERS = []


if __name__ == '__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES

    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in BOBYQA_OPTIMIZERS:
            print(optimizer(objective, n_trials=50, n_dim=34, with_count=True))