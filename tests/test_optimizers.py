import random

from humpday.objectives.classic import CLASSIC_OBJECTIVES
from humpday.optimizers.alloptimizers import OPTIMIZERS

# Hard ceiling, as a multiple of `n_trials`, beyond which we declare the
# algorithm to have violated its evaluation budget and stop it. Generous
# enough to allow modest internal overshoot (line searches, polishing
# steps, retries) but tight enough to fail fast — and *visibly* — on
# infinite loops instead of letting CI hang for hours.
_BUDGET_MULTIPLIER = 3


class BudgetExceeded(RuntimeError):
    """The algorithm called the objective more times than its budget allows."""


def _budgeted(objective, n_trials, optimizer_name):
    """Wrap an objective so it raises BudgetExceeded after `_BUDGET_MULTIPLIER *
    n_trials` calls. This converts a hang or runaway loop into an immediate,
    informative test failure."""
    ceiling = _BUDGET_MULTIPLIER * n_trials
    state = {"calls": 0}

    def wrapped(x):
        state["calls"] += 1
        if state["calls"] > ceiling:
            raise BudgetExceeded(
                f"{optimizer_name} called the objective "
                f"{state['calls']} times — exceeds {ceiling} "
                f"({_BUDGET_MULTIPLIER}x n_trials={n_trials})"
            )
        return objective(x)

    wrapped.__name__ = getattr(objective, "__name__", "objective")
    return wrapped


def test_compendium():
    """Smoke-test every optimizer against one objective per (optimizer, n_dim).

    Three guarantees enforced here that the original test did not:

    1. Seeded `random` so the objective-per-optimizer pairing is
       reproducible across runs.
    2. The wrapped objective raises `BudgetExceeded` after
       `_BUDGET_MULTIPLIER * n_trials` calls, so an algorithm that fails
       to respect its declared budget produces an immediate, informative
       failure rather than a CI hang.
    3. Errors are re-raised with the algorithm name AND objective name,
       so a regression can be localised at a glance.
    """
    random.seed(42)
    n_trials = 10
    for n_dim in [2, 3]:
        for optimizer in OPTIMIZERS:
            objective = random.choice(CLASSIC_OBJECTIVES)
            bounded = _budgeted(objective, n_trials, optimizer.__name__)
            try:
                optimizer(bounded, n_trials=n_trials, n_dim=n_dim, with_count=True)
            except BudgetExceeded:
                raise
            except Exception as e:
                print(e)
                raise Exception(optimizer.__name__ + " fails on " + objective.__name__)


if __name__ == "__main__":
    test_compendium()
