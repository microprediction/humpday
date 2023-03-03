import math
import numpy as np


try:
    import freelunch
    using_freelunch = True
except ImportError:
    using_freelunch = False

if using_freelunch:

    def freelunch_factory(objective , method, n_trials, n_dim,  with_count=False, n_pop=None):
        global feval_count
        feval_count = 0

        def _objective(x) -> float:
            global feval_count
            feval_count += 1
            return objective(np.array(x))

        bounds = np.array( [ np.array([0, 1])] * n_dim )
        optimizer = getattr(freelunch,method)(obj=_objective, bounds=bounds)

        if method.lower() in ['de','sade','pso','krillherd']:
            n_gens = int(math.ceil(n_trials / n_pop))
            optimizer.hypers['N'] = n_pop   # population size
            optimizer.hypers['G'] = n_gens  # number of generations
        if method.lower() in 'sa':
            n_gens = int(math.ceil(n_trials / n_pop))
            optimizer.hypers['N'] = n_pop  # population size
            optimizer.hypers['K'] = n_gens  # number of generations

        runs = optimizer(n_runs=1, full_output=True)  # instance and run
        best_val = runs['scores'][0]  # all obj scores are sorted low to high
        best_x = runs['solutions'][0]  # corresponding inputs
        feval_count_comparison = runs['nfe']  # function evaluations

        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


    def freelunch_de_3_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 n_pop=3, method='DE')  # It is useful to have a clone of one of the better algos

    def freelunch_de_8_cube(objective ,n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 n_pop=8, method='DE')  # It is useful to have a clone of one of the better algos


    def freelunch_de_21_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 n_pop=21, method='DE')  # It is useful to have a clone of one of the better algos


    def freelunch_sa_3_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='SA', n_pop=3)  # It is useful to have a clone of one of the better algos


    def freelunch_sa_8_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='SA', n_pop=8)  # It is useful to have a clone of one of the better algos


    def freelunch_sa_21_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='SA', n_pop=21)  # It is useful to have a clone of one of the better algos


    def freelunch_sade_3_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='SADE',n_pop=3)  # It is useful to have a clone of one of the better algos


    def freelunch_sade_8_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='SADE', n_pop=8)  # It is useful to have a clone of one of the better algos


    def freelunch_sade_21_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='SADE', n_pop=21)  # It is useful to have a clone of one of the better algos


    def freelunch_pso_3_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='PSO', n_pop=3)  # It is useful to have a clone of one of the better algos


    def freelunch_pso_8_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='PSO', n_pop=8)  # It is useful to have a clone of one of the better algos


    def freelunch_pso_21_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='PSO', n_pop=21)  # It is useful to have a clone of one of the better algos


    def freelunch_krillherd_3_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='KrillHerd', n_pop=3)  # It is useful to have a clone of one of the better algos


    def freelunch_krillherd_8_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='KrillHerd', n_pop=8)  # It is useful to have a clone of one of the better algos


    def freelunch_krillherd_21_cube(objective, n_trials, n_dim, with_count):
        return freelunch_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='KrillHerd', n_pop=21)  # It is useful to have a clone of one of the better algos


    FREELUNCH_OPTIMIZERS = [ freelunch_sade_3_cube, freelunch_sade_8_cube, freelunch_sade_21_cube,
                             freelunch_sa_3_cube, freelunch_sa_8_cube, freelunch_sa_21_cube,
                             freelunch_de_3_cube, freelunch_de_8_cube, freelunch_de_21_cube,
                             freelunch_pso_3_cube, freelunch_pso_8_cube, freelunch_pso_21_cube,
                             freelunch_krillherd_3_cube, freelunch_krillherd_8_cube]
    FREELUNCH_TOP_OPTIMIZERS = [freelunch_sade_8_cube, freelunch_sa_21_cube]
else:
    FREELUNCH_OPTIMIZERS = []
    FREELUNCH_TOP_OPTIMIZERS = []


if __name__ == '__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES

    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in FREELUNCH_OPTIMIZERS:
            print(optimizer(objective, n_trials=50, n_dim=3, with_count=True))
