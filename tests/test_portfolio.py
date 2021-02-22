from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.objectives.portfolio import PORTFOLIO_OBJECTIVES

import random


def test_portfolio():
    n_trials = 10
    for n_dim in [2,3]:
        for optimizer in OPTIMIZERS:
            objective = random.choice(PORTFOLIO_OBJECTIVES)
            try:
                optimizer(objective, n_trials=n_trials,n_dim=n_dim,with_count=True)
            except Exception as e:
                print(e)
                raise Exception(optimizer.__name__ + ' fails on ' + objective.__name__)


if __name__=='__main__':
    test_portfolio()