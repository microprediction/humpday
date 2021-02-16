# Create a plot of strength versus speed
from humpday.comparison.timingretrieval import get_timing
from humpday.comparison.eloretrieval import get_elo_leaderboard
from typing import List, Tuple
from humpday.optimizers.scipycube import scipy_powell_cube as DEFAULT_OPTIMIZER

FIBONACCI = [ 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
FIBONACCI_X10 = [ k*10 for k in FIBONACCI ]


def closest(lst, k):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]


def f_dim(n_dim):
    """ Nearest Fibonacci """
    return closest(FIBONACCI,n_dim)


def f_trials(n_trials):
    """ Nearest 10x multiple of Fibonacci """
    return closest(FIBONACCI_X10, n_trials)


def suggest(n_dim:int, n_trials:int, n_seconds:float, category='classic')->List[Tuple]:
    """
         n_seconds:  Maximum allowed time for entire minimization routine
    """

    n_dim = f_dim(n_dim)
    n_trials = f_trials(n_trials)

    cpu = get_timing()
    shortlist = list()
    for opt, opt_stats in cpu.items():
        for dim, dim_stats in opt_stats.items():
            if int(dim)==n_dim:
                if dim_stats.get(str(n_trials)) is not None and dim_stats[str(n_trials)]<=n_seconds:
                    shortlist.append((opt,dim_stats[str(n_trials)]))

    lb = get_elo_leaderboard(category=category, n_dim=n_dim, n_trials=n_trials)
    suggestions = list()
    if lb:
        for opt,t in shortlist:
            try:
                ndx = lb['name'].index(opt)
                elo = lb['rating'][ndx]
                suggestions.append((elo,t,opt))
            except ValueError:
                pass

    if not suggestions:
        return [(DEFAULT_OPTIMIZER,None,None)]
    else:
        return list(sorted( suggestions, reverse=True))


def recommend(objective, n_dim:int, n_trials:int, category='classic'):
    """
    :param objective:    function taking  list -> float
    :param n_dim:
    :param n_trials:
    :param category:
    :return: List of recommended optimizers
    """
    st = time.time()
    for _ in range(2):
        u = [0.5]*n_dim
        _ = objective(u)
    t = (time.time()-st)/2.0
    n_seconds = 0.1*t*n_trials
    return suggest(n_dim=n_dim, n_trials=n_trials, n_seconds=n_seconds, category=category)


if __name__=='__main__':
    import time
    from pprint import pprint
    import math

    def my_objective(u):
        time.sleep(0.1)
        return u[0]*math.sin(u[1])

    pprint(recommend(my_objective, n_dim=4, n_trials=120)[:3])