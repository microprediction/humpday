from humpday.objectives.allobjectives import A_CLASSIC_OBJECTIVE
from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.comparison.limitations import max_n_dim
import json
from getjson import getjson
import time
import warnings
from pprint import pprint
import traceback
import sys
import random


# Generate lookup of optimizer times. This can take a while.


def get_timing():
    return getjson('https://raw.githubusercontent.com/microprediction/humpday/main/humpday/comparison/timing.json')


def exclude(name,n_dim,n_trials):
    return 'shgo' in name and n_dim>13


def create_timing(max_elapsed=5*60):
    """ Add to the timing JSON file """

    try:
        with open('timing.json', 'rt') as fp:
            cpu = json.load(fp=fp)
    except Exception:
        cpu = dict()

    for n_dim in [2, 3, 5, 8]:
        print(' ')
        print('*********** Dimension '+str(n_dim)+' ************')
        print(' ')
        for opt in OPTIMIZERS:
            print(opt.__name__)
            dim_okay = n_dim <= max_n_dim(opt.__name__)

            if not cpu.get(opt.__name__):
                cpu[opt.__name__] = dict()
            n_failures = 0
            elapsed = -1

            if not cpu[opt.__name__].get(n_dim):
                cpu[opt.__name__][n_dim]=dict()
                for j,n_trials in enumerate([20,30,50,80,130,210,340,550]):
                    print(' '*j+str(n_trials))
                    if exclude(opt.__name__, n_dim, n_trials):
                        cpu[opt.__name__][n_dim][n_trials] = -1

                    if not cpu[opt.__name__][n_dim].get(n_trials):
                        tck = time.time()
                        try:
                            best_val,best_x = opt(objective=A_CLASSIC_OBJECTIVE,n_trials=n_trials,n_dim=n_dim)
                            elapsed = time.time()-tck
                            cpu[opt.__name__][n_dim][n_trials] = elapsed
                            if elapsed>60:
                                pprint({'name':opt.__name__,'n_dim':n_dim,'n_trials':n_trials,"elapsed":elapsed})
                        except Exception:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            warnings.warn(opt.__name__+' failed ')
                            pprint({"name":opt.__name__,"n_dim":n_dim,"n_trials":n_trials})
                            traceback.print_tb(tb=exc_traceback)
                            cpu[opt.__name__][n_dim][n_trials] = -1
                            n_failures+=1
                        if n_failures>=1:
                            break
                        if elapsed>max_elapsed:
                            break
                    if n_failures >= 1:
                        break
                    if elapsed > 60 * 10:
                        break

            with open('timing.json','wt') as fp:
                json.dump(cpu,fp=fp)


if __name__=='__main__':
    create_timing()
