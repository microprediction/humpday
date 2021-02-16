# Create a plot of strength versus speed
from humpday.comparison.timingretrieval import get_strategy_time
from humpday.comparison.eloretrieval import get_elo_rating
from humpday import OPTIMIZERS
from humpday.comparison.naming import optimizer_name
from pprint import pprint


def get_name_elo_cpu(n_dim=5, n_trials=130):
    """
    :return:   [names], [elos], [cpus]
    """
    names = [ optimizer_name(o) for o in OPTIMIZERS ]
    elos  = [ get_elo_rating(o.__name__, n_dim=n_dim, n_trials=n_trials ) for o in OPTIMIZERS ]
    cpus  = [ get_strategy_time(optimizer_name(o), n_dim=n_dim, n_trials=n_trials) for o in OPTIMIZERS ]
    return names, elos, cpus


def get_suggestions(n_dim=5, n_trials=130, n_seconds=5 * 60):
    """
       Returns suggestions:
         n_seconds:  Maximum allowed time for entire minimization routine
    """
    names, elos, cpus = get_name_elo_cpu(n_dim=n_dim, n_trials=n_trials)
    return sorted( [ (elo,cpu,name) for elo, cpu, name in zip(names,elos,cpus) if cpu is not None and cpu>-0.5 and cpu<n_seconds ], reverse=True )



if __name__=='__main__':
    pprint(get_suggestions(n_dim=5, n_trials=130, n_seconds=100))