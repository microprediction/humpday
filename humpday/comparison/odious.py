from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.objectives.classic import CLASSIC_OBJECTIVES
from pprint import pprint
from typing import List
from humpday.comparison.naming import optimizer_name

# See also https://github.com/microprediction/humpday-testing/tree/main/optimizer_elo_ratings/leaderboards


def comparison(objectives:List=None, n_dim=5, n_trials=100):
    """  Run all optimizers and return ranked list
    :param objectives:  A list of functions taking a single argument representing [0,1]^n_dim
    :param n_trials:    Maximum number of function evaluations
    :returns sorted list of minima found and actual number of function evaluations
    """
    if objectives is None:
        objectives = CLASSIC_OBJECTIVES
    return sorted([(optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True),
                          optimizer_name(optimizer),objective.__name__)
                         for optimizer in OPTIMIZERS for objective in objectives])


if __name__=='__main__':
    pprint( comparison() )




