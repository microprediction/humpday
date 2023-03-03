from humpday.inclusion.winninginclusion import using_winning
from humpday.inclusion.heboinclusion import using_hebo

# See https://medium.com/@microprediction/a-new-family-of-diffeomorphisms-from-the-simplex-to-the-cube-with-application-to-global-6d358714f429


if using_winning and using_hebo:

    from humpday.transforms.cubetosimplex import minimize_optimizer_on_simplex
    from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
    from humpday.optimizers.hebocube import HEBO_OPTIMIZERS
    import random

    def test_cube_to_simplex():
        import random
        for _ in range(3):
            opt = random.choice(HEBO_OPTIMIZERS)
            objective = random.choice(CHATGPT_OBJECTIVES)
            n_dim = random.choice([3,5,8])
            f_best, x_best = minimize_optimizer_on_simplex(optimizer=opt, objective=objective, n_dim=n_dim, n_trials=100)
