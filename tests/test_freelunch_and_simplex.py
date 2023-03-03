
from humpday.inclusion.winninginclusion import using_winning
from humpday.inclusion.freelunchinclusion import using_freelunch

if using_winning and using_freelunch:

    from humpday.transforms.cubetosimplex import minimize_optimizer_on_simplex
    from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
    from humpday.optimizers.freelunchcube import FREELUNCH_OPTIMIZERS
    import random

    def test_cube_to_simplex():
        from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
        from humpday.optimizers.freelunchcube import FREELUNCH_OPTIMIZERS
        import random
        for _ in range(3):
            opt = random.choice(FREELUNCH_OPTIMIZERS)
            objective = random.choice(CHATGPT_OBJECTIVES)
            n_dim = random.choice([3,5,8])
            f_best, x_best = minimize_optimizer_on_simplex(optimizer=opt, objective=objective, n_dim=n_dim, n_trials=100)
