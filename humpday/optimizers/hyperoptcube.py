import logging
import warnings
from humpday.transforms.zcurves import curl_factory

try:
    from hyperopt import fmin, hp, tpe, Trials
    from hyperopt.tpe import suggest as tpe_suggest
    from hyperopt.rand import suggest as rand_suggest
    from hyperopt.atpe import suggest as atpe_suggest
    using_ultraopt = True
except ImportError:
    warnings.warn('If you wish to include hyperopt you must pip install hyperopt')
    using_ultraopt = False

if using_ultraopt:
    logging.getLogger('hyperopt').setLevel(logging.ERROR)

    def hyperopt_cube(objective, n_trials, n_dim, with_count=False, algo=None):
        """ Minimize a function on the cube using HyperOpt, and audit # of function calls
           :param objective:    function on (0,1)^n_dim
           :param n_trials:     Guideline for function evaluations
           :param n_dim:
           :param with_count:
           :return:
        """
        logging.getLogger('hyperopt').setLevel(logging.ERROR)

        assert algo is not None, 'provide algo'
        hp_space = dict([('u' + str(i), hp.uniform('u' + str(i), 0, 1)) for i in range(n_dim)])

        global feval_count
        feval_count = 0

        def _objective(hps):
            global feval_count
            feval_count += 1
            us = [hps['u' + str(i)] for i in range(n_dim)]
            return objective(us)

        trls = Trials()
        res = fmin(_objective, space=hp_space, algo=tpe.suggest, trials=trls, max_evals=n_trials, show_progressbar=False)
        best_x = [trls.best_trial['misc']['vals']['u' + str(i)][0] for i in range(n_dim)]
        best_val = trls.best_trial['result']['loss']
        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


    def hyperopt_atpe_cube(objective, n_trials, n_dim, with_count=False):
        return hyperopt_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo=atpe_suggest)


    def hyperopt_tpe_cube(objective, n_trials, n_dim, with_count=False):
        return hyperopt_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo=tpe_suggest)


    def hyperopt_rand_cube(objective, n_trials, n_dim, with_count=False):
        return hyperopt_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo=rand_suggest)


    def hyperopt_atpe_curl2_cube(objective, n_trials, n_dim, with_count=False):
        # Probably not the best idea
        return curl_factory(optimizer=hyperopt_atpe_cube, objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,d=2)


    def hyperopt_tpe_curl2_cube(objective, n_trials, n_dim, with_count=False):
        # Probably not the best idea
        return curl_factory(optimizer=hyperopt_tpe_cube, objective=objective, n_trials=n_trials, n_dim=n_dim,
                            with_count=with_count, d=2)


    HYPEROPT_OPTIMIZERS = [ hyperopt_tpe_cube, hyperopt_atpe_cube, hyperopt_rand_cube ]
else:
    HYPEROPT_OPTIMIZERS = []


if __name__ == '__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES

    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in HYPEROPT_OPTIMIZERS:
            print(optimizer(objective, n_trials=250, n_dim=12, with_count=True))
