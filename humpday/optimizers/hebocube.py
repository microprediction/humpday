from humpday.objectives.classic import CLASSIC_OBJECTIVES
import logging
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import pandas as pd
import numpy  as np
import math

logging.getLogger('hebo').setLevel(logging.ERROR)


def hebo_cube_factory(objective, n_trials, n_dim, with_count,n_suggestions=5):
    global feval_count
    feval_count = 0

    variables = [{'name': 'u' + str(i), 'type': 'num', 'lb': 0., 'ub': 1.} for i in range(n_dim)]
    space = DesignSpace().parse(variables)
    opt = HEBO(space)

    def _objective(params: pd.DataFrame) -> np.ndarray:
        global feval_count
        feval_count += len(params.index)
        return np.array([ objective(ui) for ui in params.values ])

    n_batches = int(math.floor(n_trials/n_suggestions))
    n_remainder = n_trials - n_suggestions*n_batches
    for i in range(n_batches):
        rec = opt.suggest(n_suggestions=n_suggestions) # <-- don't change this
        opt.observe(rec, _objective(rec))
    for i in range(n_remainder):
        rec = opt.suggest(n_suggestions=1)  # <-- don't change this
        opt.observe(rec, _objective(rec))

    best_val = opt.y.min()
    best_ndx = np.argmin([y[0] for y in opt.y])  # I mean seriously, why make the user do this?
    best_x = list(opt.X.values[best_ndx])
    return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


def hebo_cube_sequential(objective, n_trials, n_dim, with_count):
    return hebo_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, n_suggestions=1)


def hebo_cube_batch(objective, n_trials, n_dim, with_count):
    return hebo_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, n_suggestions=10)


HEBO_OPTIMIZERS = [ hebo_cube_sequential, hebo_cube_batch ]

if __name__=='__main__':
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        import time
        for optimizer in HEBO_OPTIMIZERS:
            print(optimizer.__name__+'...')
            st = time.time()
            print((optimizer(objective, n_trials=12, n_dim=2, with_count=True)))
            print('   ... took '+str(time.time()-st)+' seconds.')