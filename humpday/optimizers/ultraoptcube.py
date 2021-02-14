from humpday.objectives.classic import CLASSIC_OBJECTIVES
import warnings
# On some systems install is unreliable due to ConfigSpace on some systems,
# so this is not included in setup.

try:
    from ultraopt import fmin
    using_ultraopt = True
except ImportError:
    warnings.warn('If you wish to include ultraopt you must pip install ultraopt')
    using_ultraopt = False


if using_ultraopt:

    def ultraopt_cube_factory(objective, n_trials, n_dim, with_count, method):
        global feval_count
        feval_count = 0

        HDL = dict([('u' + str(i), {"_type": "uniform", "_value": [0., 1.]}) for i in range(n_dim)])

        def _objective(config: dict) -> float:
            global feval_count
            feval_count += 1
            u = [config['u' + str(i)] for i in range(n_dim)]
            return objective(u)

        result = fmin(eval_func=_objective, config_space=HDL, optimizer=method, n_iterations=n_trials,
                      n_jobs=1, show_progressbar=False, parallel_strategy="Serial" )
        best_x = [result.best_config['u' + str(i)] for i in range(n_dim)]
        best_val = result.best_loss
        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


    def ultraopt_etpe_cube(objective, n_trials, n_dim, with_count):
        return ultraopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='ETPE')


    def ultraopt_random_cube(objective, n_trials, n_dim, with_count):
        return ultraopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='Random')


    ULTRAOPT_OPTIMIZERS = [ultraopt_random_cube, ultraopt_etpe_cube ]
else:
    ULTRAOPT_OPTIMIZERS = []


if __name__=='__main__':
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in ULTRAOPT_OPTIMIZERS:
            print((optimizer(objective, n_trials=250, n_dim=6, with_count=True)))
