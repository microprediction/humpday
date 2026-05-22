import pytest

# Skip entire module if dependencies not available
try:
    from humpday.inclusion.pybobyqainclusion import using_bobyqa
    from humpday.inclusion.winninginclusion import using_winning
except ImportError:
    # If inclusion modules don't exist, skip this test module
    pytest.skip("inclusion modules not available", allow_module_level=True)

# See https://medium.com/@microprediction/a-new-family-of-diffeomorphisms-from-the-simplex-to-the-cube-with-application-to-global-6d358714f429


if using_winning and using_bobyqa:
    import random

    from humpday.optimizers.bobyqacube import BOBYQA_OPTIMIZERS

    from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
    from humpday.transforms.cubetosimplex import minimize_optimizer_on_simplex

    def test_cube_to_simplex():

        for _ in range(3):
            opt = random.choice(BOBYQA_OPTIMIZERS)
            objective = random.choice(CHATGPT_OBJECTIVES)
            n_dim = random.choice([3, 5, 8])
            f_best, x_best = minimize_optimizer_on_simplex(
                optimizer=opt, objective=objective, n_dim=n_dim, n_trials=100
            )
