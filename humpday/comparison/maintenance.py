from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.objectives.classic import CLASSIC_OBJECTIVES
from humpday.comparison.naming import optimizer_name

# TODO: Add speed check


def whats_broken():
    for n_trials in [2,10,40]:
        for n_dim in [2,3,8]:
            for optimizer in OPTIMIZERS:
                for objective in CLASSIC_OBJECTIVES:
                    try:
                        optimizer(objective, n_trials=n_trials,n_dim=n_dim,with_count=True)
                    except Exception as e:
                        print(e)
                        print(optimizer_name(optimizer) + ' fails on ' + objective.__name__ +
                              ' in dimension '+str(n_dim)+' when n_trials='+str(n_trials))


if __name__=='__main__':
    whats_broken()