import time
from humpday.optimizers.dlibcube import dlib_default_cube
from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube
from humpday.optimizers.scipycube import scipy_slsqp_cube
from humpday.optimizers.ultraoptcube import ultraopt_gbrt_cube
from humpday.optimizers.nloptcube import nlopt_direct_cube, nlopt_isres_cube, nlopt_esch_cube, nlopt_directr_cube
from humpday.objectives.portfolio import markowitz_realized_on_cube, markowitz_analytic_on_cube, make_solution,\
    markowitz_return_on_cube, make_sigma_matrix
from pprint import pprint
from typing import List
import numpy as np


def nice_div(a,b):
    if isinstance(a,List):
        return [ nice_div(ai,bi) for ai,bi in zip(a,b) ]
    else:
        return 1.0 if abs(a-b)<0.01*(abs(a)+abs(b)) else 100. if abs(b)<1e-6 else a/b


def normalize(x):
    return [ xi/sum(x) for xi in x ] if sum(x)>0 else [ 1.0/len(x) for xi in x ]


def verify_markowitz(optimizer, n_dim, n_trials):
    """ How good are derivative free solvers?

        This little exercise tasks some of the speedier optimizers with variance minimization
        for a portfolio, and also looks at how consistent the answers are between two runs.

        Some examples of speedy optimizers are suggested by the unused imports.

        Obviously, you're better off doing this with derivs or worse case, quadratic solvers

    """

    # Run optimizer twice
    st = time.time()
    v1, u1, t1 = optimizer(markowitz_realized_on_cube, n_dim=n_dim, n_trials=n_trials, with_count=True)
    tau1 = time.time()-st

    st = time.time()
    v2, u2, t2 = optimizer(markowitz_analytic_on_cube, n_dim=n_dim, n_trials=n_trials, with_count=True)
    tau2 = time.time()-st

    # Solve
    u3 = make_solution(x_dim=n_dim)
    v3 = markowitz_analytic_on_cube(u3)
    v4 = markowitz_realized_on_cube(u3)

    r = [ nice_div(u1j, u2j) for u1j, u2j in zip(u1,u2) ]
    r1 = [ nice_div(u1j, u3j) for u1j, u3j in zip(u1,u3) ]
    r2 = [ nice_div(u2j, u3j) for u2j, u3j in zip(u2,u3) ]

    nu1 = normalize(u1)
    nu2 = normalize(u2)
    nu3 = normalize(u3)

    results = {'ratio_of_solutions':r,'ratio_to_analytic_1':r1,'ratios_to_analytic_2':r2,
               'minimum_1':v1,'minimum_2':v2,'minium_3':v3, 'minimum_4':v4,'trials_1':t1,'trials_2':t2,
               'seconds_1':tau1,'seconds_2':tau2,
               'weights_1':nu1,
               'weights_2':nu2,
               'weight_ratio_1':nice_div(nu1,nu3),
               'weight_ratio_2':nice_div(nu2,nu3),
               'diagonals':np.diag(make_sigma_matrix()[:n_dim,:n_dim])}
    return results


def markowitz_return(optimizer, n_dim, n_trials):
    """
        Maximizing a different objective, just for fun
    """

    # Run optimizer twice
    st = time.time()
    v1, u1, t1 = optimizer(markowitz_return_on_cube, n_dim=n_dim, n_trials=n_trials, with_count=True)
    tau1 = time.time()-st

    st = time.time()
    v2, u2, t2 = optimizer(markowitz_return_on_cube, n_dim=n_dim, n_trials=n_trials, with_count=True)
    tau2 = time.time()-st

    # Use Markowitz approximation
    u3 = make_solution(x_dim=n_dim)
    v3 = markowitz_return_on_cube(u3)
    u4 = [1/n_dim for _ in range(n_dim)]
    v4 = markowitz_return_on_cube(u4)

    r = [ nice_div(u1j, u2j) for u1j, u2j in zip(u1,u2) ]
    r1 = [ nice_div(u1j, u3j) for u1j, u3j in zip(u1,u3) ]
    r2 = [ nice_div(u2j, u3j) for u2j, u3j in zip(u2,u3) ]

    nu1 = normalize(u1)
    nu2 = normalize(u2)
    nu3 = normalize(u3)

    results = {'ratio_of_solutions':r,'ratio_to_analytic_1':r1,'ratios_to_analytic_2':r2,
               'minimum_1':v1,'minimum_2':v2,'minimum_markowitz':v3,
               'minimum_market':v4,'trials_1':t1,'trials_2':t2,
               'seconds_1':tau1,'seconds_2':tau2,
               'weights_1':nu1,
               'weights_2':nu2,
               'weight_ratio_1':nice_div(nu1,nu3),
               'weight_ratio_2':nice_div(nu2,nu3),
               'diagonals':np.diag(make_sigma_matrix()[:n_dim,:n_dim])}
    return results


if __name__=='__main__':
    optimizer = nlopt_directr_cube
    results = markowitz_return(optimizer=optimizer, n_dim=5, n_trials=150000)
    pprint(results)