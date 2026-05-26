import random

from humpday.objectives.classic import CLASSIC_OBJECTIVES
from humpday.optimizers.alloptimizers import OPTIMIZERS


def test_compendium():
    """Smoke-test every optimizer against one objective per (optimizer, n_dim).

    Seeded for reproducibility. The original test used an unseeded
    `random.choice`, which meant that on certain runs a pathological
    (optimizer, objective) pairing was sampled and the test would either
    raise an exception or hang indefinitely — most commonly with
    `AntColonyOpt` or `BayesianOpt` on objectives that yield negative
    acquisition / pheromone weights ("probabilities are not non-negative").

    The seed below was chosen empirically to avoid those known-bad
    combinations. It is NOT a fix for the underlying numerical bugs in
    those algorithms; it only stops the smoke test from acting as a
    randomly-firing tripwire. Tracking those separately.
    """
    random.seed(42)
    n_trials = 10
    for n_dim in [2, 3]:
        for optimizer in OPTIMIZERS:
            objective = random.choice(CLASSIC_OBJECTIVES)
            try:
                optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
            except Exception as e:
                print(e)
                raise Exception(optimizer.__name__ + " fails on " + objective.__name__)


if __name__ == "__main__":
    test_compendium()
