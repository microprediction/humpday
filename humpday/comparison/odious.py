from humpday.optimizers.alltopoptimizers import TOP_OPTIMIZERS
from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.objectives.classic import CLASSIC_OBJECTIVES
from pprint import pprint
from typing import List
import numpy as np
from humpday.comparison.naming import optimizer_name

# See also https://github.com/microprediction/humpday-testing/tree/main/optimizer_elo_ratings/leaderboards


def compare(objectives:List=None, n_dim=5, n_trials=100):
    """  Run all optimizers and return ranked list
    :param objectives:  A list of functions taking a single argument representing [0,1]^n_dim
    :param n_trials:    Maximum number of function evaluations
    :returns sorted list of minima found and actual number of function evaluations
    """
    if objectives is None:
        objectives = CLASSIC_OBJECTIVES[:2]
    return sorted([(optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True),
                          optimizer_name(optimizer),objective.__name__)
                         for optimizer in OPTIMIZERS for objective in objectives])


def points_race(objectives:List=None, optimizers=None, n_dim=5, n_trials=100, n_top=3):
    """  Run all optimizers and return ranked list, printing as we go
    :param objectives:  A list of functions taking a single argument representing [0,1]^n_dim
    :param n_trials:    Maximum number of function evaluations
    :param n_top:       Determines how many get points awarded, for each obj function (.e.g. 3,2,1 points)
    :returns sorted list of minima found and actual number of function evaluations
    """
    if objectives is None:
        objectives = CLASSIC_OBJECTIVES

    if optimizers is None:
        optimizers = TOP_OPTIMIZERS

    from collections import Counter
    overall_points = None
    for objective in objectives:
        print('Optimizing the '+ objective.__name__.replace('_on_cube','')+' function ...')
        obj_results = list()
        for optimizer in optimizers:
            name = optimizer_name(optimizer)
            best_val, _, feval_count = optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
            obj_results.append( (best_val+0.00001*np.random.randn(), name) ) # Tie breaker as I am lazy
        points = Counter( dict([ (name, n_top-j) for j, (val,name) in enumerate( sorted( obj_results )[:n_top] ) ]) )
        if overall_points is None:
            overall_points = points
        else:
            overall_points.update(points)
        print('Best so far ...')
        pprint(overall_points)
    return overall_points






if __name__=='__main__':
    pprint(points_race())




