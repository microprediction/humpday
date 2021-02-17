import nlopt

NLOPTIMIZERS = {'gn_direct' :nlopt.GN_DIRECT,
                'gn_ags':nlopt.GN_AGS,
                'gn_esch':nlopt.GN_ESCH,
                'gn_isres':nlopt.GN_ISRES,
                'gn_crs2lm':nlopt.GN_CRS2_LM,
                'gn_directr':nlopt.GN_DIRECT_L_RAND,
                'gn_directo':nlopt.GN_ORIG_DIRECT}


def nlopt_cube_factory(objective ,n_trials, n_dim, with_count, method='gn_direct'):
    optim = NLOPTIMIZERS[method]

    global feval_count
    feval_count = 0

    lb = [0. for _ in range(n_dim)]
    ub = [1. for _ in range(n_dim)]

    def _objective(u, grad) -> float:
        global feval_count
        feval_count += 1
        return objective(u)

    opt = nlopt.opt(optim, n_dim)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(_objective)
    opt.set_maxeval(n_trials - 1)  # Groan
    best_x = opt.optimize([0.5] * n_dim)
    best_val = _objective(best_x, grad=None)  # <-- Stupid

    return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


def nlopt_direct_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_direct')


def nlopt_ags_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_ags')


def nlopt_esch_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_esch')


def nlopt_isres_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_isres')


def nlopt_mlsl_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_mlsl')


def nlopt_crs2lm_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_crs2lm')


def nlopt_directr_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_directr')


def nlopt_directo_cube(objective ,n_trials, n_dim, with_count):
    return nlopt_cube_factory(objective=objective,n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='gn_directo')


# SEEMINGLY_BROKEN = [nlopt_mlsl_cube] # Will crash Python with uncaught exception


NLOPT_OPTIMIZERS = [nlopt_direct_cube, nlopt_ags_cube, nlopt_esch_cube,
                    nlopt_isres_cube, nlopt_crs2lm_cube, nlopt_directr_cube,
                    nlopt_directo_cube]


if __name__=='__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in NLOPT_OPTIMIZERS:
            print((optimizer(objective, n_trials=250, n_dim=6, with_count=True)))