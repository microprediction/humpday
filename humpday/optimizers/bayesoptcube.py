from bayes_opt import BayesianOptimization
from humpday.objectives.classic import CLASSIC_OBJECTIVES


BAYESOPT_METHODS = ['ucb', 'ei', 'poi']

def bayesopt_cube_factory(objective, n_trials, n_dim, with_count,method):

    assert method in BAYESOPT_METHODS

    global feval_count
    feval_count = 0

    pbounds = dict([('u' + str(i), (0., 1.)) for i in range(n_dim)])

    def _neg_objective(**kwargs) -> float:
        global feval_count
        feval_count += 1
        u = [kwargs['u' + str(i)] for i in range(n_dim)]
        return -objective(u)

    optimizer = BayesianOptimization(
        f=_neg_objective,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        acq=method,
        n_iter=n_trials - 5,
    )

    best_val = -optimizer.max['target']
    best_x = [optimizer.max['params']['u' + str(i)] for i in range(n_dim)]

    return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


def bayesopt_ucb_cube(objective, n_trials, n_dim, with_count):
    return bayesopt_cube_factory(objective=objective,n_trials=n_trials,
                                 n_dim=n_dim, with_count=with_count,method='ucb')


def bayesopt_ei_cube(objective, n_trials, n_dim, with_count):
    return bayesopt_cube_factory(objective=objective,n_trials=n_trials,
                                 n_dim=n_dim, with_count=with_count,method='ei')


def bayesopt_poi_cube(objective, n_trials, n_dim, with_count):
    return bayesopt_cube_factory(objective=objective,n_trials=n_trials,
                                 n_dim=n_dim, with_count=with_count,method='poi')


BAYESOPT_OPTIMIZERS = [ bayesopt_ucb_cube, bayesopt_ei_cube, bayesopt_poi_cube ]


if __name__=='__main__':
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in BAYESOPT_OPTIMIZERS:
            print((optimizer(objective, n_trials=25, n_dim=3, with_count=True)))