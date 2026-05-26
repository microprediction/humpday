import random

from humpday.objectives.portfolio import PORTFOLIO_OBJECTIVES
from humpday.optimizers.alloptimizers import OPTIMIZERS


def test_portfolio():
    """Smoke-test each optimizer on a random portfolio objective.

    `random.seed(0)` keeps the objective-per-optimizer pairing
    reproducible. The "probabilities are not non-negative" bug in
    `AntColonyOpt` that previously needed a special-case bypass here
    was fixed in the same PR that ported AntColonyOpt to the shim
    (see `humpday/optimizers/evolutionary_algorithms.py`).
    """
    random.seed(0)
    n_trials = 10
    for n_dim in [2, 3]:
        for optimizer in OPTIMIZERS:
            objective = random.choice(PORTFOLIO_OBJECTIVES)
            try:
                optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
            except Exception as e:
                print(e)
                raise Exception(optimizer.__name__ + " fails on " + objective.__name__)


if __name__ == "__main__":
    from humpday.optimizers.scipycube import scipy_nelder_cube

    from humpday.objectives.portfolio import markowitz_skew_on_cube

    v, x = scipy_nelder_cube(markowitz_skew_on_cube, n_dim=4, n_trials=100)
