import numpy as np
from sklearn.datasets import make_spd_matrix
import time
from scipy.stats import skew
import math

# This problem is randomly generated, but only once per day
# If a solution run straddles GMT midnight, well that's life in the big city.

global sigma_matrix
sigma_matrix = None
global sigma_scenarios
sigma_scenarios = None
global adjusted_scenarios
adjusted_scenarios = None
N_ASSETS = 233
N_SCENARIOS = 15000
global solutions
solutions = None
DT = 1/2000  #
SQDT = math.sqrt(DT)
YEARLY = 3.0


def make_sigma_matrix():
    from datetime import datetime
    day_of_year = datetime.now().timetuple().tm_yday
    return make_spd_matrix(n_dim=N_ASSETS, random_state=day_of_year)


def make_sigma_scenarios():
    """
    :return:  N_SCENARIOS X FIVE_HUNDRED
    """
    global sigma_scenarios
    if sigma_scenarios is None:
        global sigma_matrix
        if sigma_matrix is None:
            sigma_matrix = make_sigma_matrix()
        mu = np.zeros(N_ASSETS)
        sigma_scenarios = SQDT*YEARLY*np.random.multivariate_normal(mu, sigma_matrix, N_SCENARIOS, check_valid='warn', tol=1e-8)
    return sigma_scenarios


def make_adjusted_scenarios():
    """
       :return:
    """
    global adjusted_scenarios
    if adjusted_scenarios is None:
        sigma_scenarios = make_sigma_scenarios()
        mean_ratio = np.mean(np.exp(sigma_scenarios),axis=0)
        mean_expon = np.log(mean_ratio)
        adjusted_scenarios = sigma_scenarios.copy()
        n_dim = sigma_scenarios.shape[1]
        for j in range(n_dim):
            adjusted_scenarios[:,j] = sigma_scenarios[:,j]-mean_expon[j]
        # Blach
        if False: # Later...,
            realized_sigma = np.cov(np.transpose(sigma_scenarios))
            diagonals = np.diagonal(np.dot(np.transpose(realized_sigma),realized_sigma)).copy()
    return adjusted_scenarios


def make_solution(x_dim):
    global solutions
    global sigma_matrix
    if sigma_matrix is None:
        sigma_matrix = make_sigma_matrix()
    if not solutions:
        solutions = dict()
    if not solutions.get(x_dim):
        sigma_sub_matrix = sigma_matrix[0:x_dim, 0:x_dim]
        soln = np.linalg.solve(sigma_sub_matrix,np.ones(x_dim))
        soln_sum = np.sum(soln)
        normalized_soln = [ s/soln_sum for s in soln ]
        solutions[x_dim]=normalized_soln
    return solutions[x_dim]


def markowitz_analytic_on_cube(u:[float])->float:
    """ Min-var portfolio (known solution)
    :param u:    Portfolio weights
    :return:
    """
    # Roughly equal on rays through origin
    assert all([ 0<=ui<=1 for ui in u])
    global sigma_matrix
    if sigma_matrix is None:
        sigma_matrix=make_sigma_matrix()

    x_dim = len(u)
    u_sum = np.sum(u)
    x = cube_to_weights(u)

    sigma_sub_matrix = sigma_matrix[0:x_dim,0:x_dim]
    portfolio_var = np.linalg.multi_dot( [np.array(x).transpose(), sigma_sub_matrix, x] )
    return portfolio_var+abs(1.0-u_sum)+push(u)


def push(u):
    # Tiny push away from origin
    return -0.1*min(0.1*len(u),np.sum(np.abs(u)))


def cube_to_weights(u):
    if not all([0 <= ui <= 1 for ui in u]):
        raise ValueError("u should be in hypercube")
    u_sum = np.sum(u)
    w = [(1e-8 + ui) / (1e-8 + u_sum) for ui in u]
    w = [ wi/sum(w) for wi in w ]
    return w


def realized_something_factory(g,u, adjust=False):
    """
    :param u:
    :param g:   [float] -> float  Some function on realized P/L distribution samples
    :return:
    """
    w = cube_to_weights(u)
    w_dim = len(w)

    global sigma_matrix
    global sigma_scenarios
    global adjusted_scenarios
    if sigma_matrix is None:
        sigma_matrix = make_sigma_matrix()
    if sigma_scenarios is None:
        sigma_scenarios = make_sigma_scenarios()
    if adjust and adjusted_scenarios is None:
        adjusted_scenarios = make_adjusted_scenarios()

    y = adjusted_scenarios[:, 0:w_dim] if adjust else sigma_scenarios[:, 0:w_dim]
    return g(u=u,w=w,y=np.array(y))


def markowitz_realized_on_cube(u:[float])->float:
    """ Brute force using samples, but should be the same
    :param u:
    :return:
    """
    def g(u,w,y):
        return np.var(np.dot(y, np.array(w))) + abs(1.0 - sum(u)) + push(u)

    return realized_something_factory(g=g,u=u)


def markowitz_skew_on_cube(u:[float])->float:
    """ Adds skew for fun
    :param u:
    :return:
    """
    def g(u,w,y):
        w = np.dot(y, np.array(w))
        return np.var(w) + abs(1.0 - sum(u)) + push(u) - skew(w)

    return realized_something_factory(g=g,u=u)


def markowitz_return_on_cube(u:[float])->float:
    """
    :param u:
    :return:
    """
    def g(u,w,y):
        growth = np.exp(y)
        change = (growth - 1.0)  # Could cache this
        portfolio_change = np.dot(change, np.array(w))
        log_wealth = np.log(1+portfolio_change)
        annualized_return_bp = 10000*np.mean(log_wealth)/DT
        return 0.001*abs(1.0 - sum(u)) + push(u) - annualized_return_bp

    return realized_something_factory(g=g,u=u, adjust=True)



PORTFOLIO_OBJECTIVES = [markowitz_analytic_on_cube, markowitz_realized_on_cube,
                        markowitz_return_on_cube, markowitz_skew_on_cube ]


def nice_div(a,b):
    return 0 if abs(b)<1e-6 else a/b


def troublesome_example():
    from humpday.optimizers.scipycube import scipy_nelder_cube
    from humpday.objectives.portfolio import markowitz_skew_on_cube
    v,x = scipy_nelder_cube(markowitz_skew_on_cube,n_dim=4,n_trials=100)


def markowitz_example():
    from humpday.optimizers.dlibcube import dlib_default_cube
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube
    from humpday.optimizers.scipycube import scipy_slsqp_cube
    from humpday.optimizers.ultraoptcube import ultraopt_gbrt_cube
    from humpday.optimizers.nloptcube import nlopt_direct_cube, nlopt_isres_cube, nlopt_esch_cube, nlopt_directr_cube
    optimizer = nlopt_directr_cube
    objective = markowitz_skew_on_cube

    n_dim = 4
    st = time.time()
    v1, u1, t1 = optimizer(objective, n_dim=n_dim, n_trials=5000, with_count=True)
    tau1 = time.time()-st

    st = time.time()
    v2, u2, t2 = optimizer(objective, n_dim=n_dim, n_trials=5000, with_count=True)
    tau2 = time.time()-st
    u3 = make_solution(x_dim=n_dim)
    v3 = objective(u3)
    v4 = markowitz_analytic_on_cube(u3)
    r = [ nice_div(u1j, u2j) for u1j, u2j in zip(u1,u2) ]
    r1 = [ nice_div(u1j, u3j) for u1j, u3j in zip(u1,u3) ]
    r2 = [ nice_div(u2j, u3j) for u2j, u3j in zip(u2,u3) ]
    results = {'r':r,'r1':r1,'r2':r2,'u1':u1,'u2':u2,'u3':u3, 'v1':v1,'v2':v2,'v3':v3, 'v4':v4,'t1':t1,'t2':t2,'tau1':tau1,'tau2':tau2}
    from pprint import pprint
    pprint(results)



if __name__=='__main__':
    troublesome_example()
