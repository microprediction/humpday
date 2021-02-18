import warnings
from humpday.transforms.zcurves import curl_factory

# http://dlib.net/optimization.html#find_min_global
# This library also provides global_function_search which is pretty darn cool

try:
    from dlib import find_min_global
    using_dlib = True
except ImportError:
    warnings.warn('dlib may not be properly supported on this operating system. Maybe try pip install --upgrade cmake')
    using_dlib = False

if using_dlib:

    def dlib_default_cube(objective ,n_trials, n_dim, with_count):
        global feval_count
        feval_count = 0

        lb = [0. for _ in range(n_dim)]
        ub = [1. for _ in range(n_dim)]

        def _objective(*args) -> float:
            global feval_count
            feval_count += 1
            return objective(list(args))

        best_x, best_val = find_min_global(_objective, lb, ub, n_trials)

        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)

    def dlib_cube(objective ,n_trials, n_dim, with_count):
        return dlib_default_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count)  # It is useful to have a clone of one of the better algos

    def dlib_curl2_cube(objective ,n_trials, n_dim, with_count):
        # Meh
        return curl_factory(optimizer=dlib_default_cube,objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, d=2)

    DLIB_OPTIMIZERS = [dlib_cube, dlib_default_cube ]
else:
    DLIB_OPTIMIZERS = []


if __name__ == '__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES

    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in DLIB_OPTIMIZERS:
            print(optimizer(objective, n_trials=500, n_dim=34, with_count=True))