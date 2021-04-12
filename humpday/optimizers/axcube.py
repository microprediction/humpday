from funcy import print_durations

try:
    from ax import optimize
    using_ax = True
except ImportError:
    using_ax = False

if using_ax:

    from logging import CRITICAL
    from ax.utils.common.logger import get_logger
    rt = get_logger('ax')
    rt.setLevel(CRITICAL)
    import warnings
    warnings.filterwarnings("ignore")


    def ax_cube(objective, n_trials, n_dim, with_count=False, method=None):
        global feval_count
        feval_count = 0
        rt = get_logger('ax')
        rt.setLevel(CRITICAL)

        def evaluation_func(prms):
            global feval_count
            feval_count += 1
            return objective([prms["u" + str(i)] for i in range(n_dim)])

        parameters = [{
            "name": "u" + str(i),
            "type": "range",
            "bounds": [0.0, 1.0],
        } for i in range(n_dim)]
        best_parameters, best_values, experiment, model = optimize(parameters=parameters,
                                                                   evaluation_function=evaluation_func,
                                                                   minimize=True,
                                                                   total_trials=n_trials)
        best_x = [ best_parameters['u'+str(i)] for i in range(n_dim) ]
        best_val = best_values[0]['objective']
        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


    def ax_default_cube(objective, n_trials, n_dim, with_count=False):
        return ax_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count)


    AX_OPTIMIZERS = [ ax_default_cube ]
else:
    AX_OPTIMIZERS = []


if __name__=='__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES
    @print_durations()
    def demo():
        for objective in CLASSIC_OBJECTIVES:
            print(' ')
            print(objective.__name__)
            for optimizer in AX_OPTIMIZERS:
                print(optimizer(objective, n_trials=250, n_dim=6, with_count=True))

