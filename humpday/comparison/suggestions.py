# Create a plot of strength versus speed
from humpday.comparison.timingretrieval import get_timing
from humpday.comparison.eloretrieval import get_elo_leaderboard
from humpday import OPTIMIZERS
from humpday.comparison.naming import optimizer_name
from pprint import pprint


def get_suggestions(n_dim, n_trials, n_seconds, category='classic'):
    """
       Returns suggestions:
         n_seconds:  Maximum allowed time for entire minimization routine
    """
    cpu = get_timing()
    shortlist = list()
    for opt, opt_stats in cpu.items():
        for dim, dim_stats in opt_stats.items():
            if int(dim)==n_dim:
                if dim_stats.get(str(n_trials)) is not None and dim_stats[str(n_trials)]<=n_seconds:
                    shortlist.append((opt,dim_stats[str(n_trials)]))

    lb = get_elo_leaderboard(category=category, n_dim=n_dim, n_trials=n_trials)
    suggestions = list()
    for opt,t in shortlist:
        ndx = lb['name'].index(opt)
        elo = lb['rating'][ndx]
        suggestions.append((elo,t,opt))

    return list(sorted( suggestions, reverse=True))


if __name__=='__main__':
    pprint(get_suggestions(n_dim=2, n_trials=130, n_seconds=100))