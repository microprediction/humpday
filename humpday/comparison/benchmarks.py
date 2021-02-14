from humpday.objectives.allobjectives import CLASSIC_OBJECTIVES
from humpday.optimizers.scipycube import scipy_dogleg_cube
import json
from pprint import pprint

def get_benchmarks():
    


if __name__ == '__main__':

    benchmarks = dict()
    for objective in CLASSIC_OBJECTIVES:
        benchmarks[objective.__name__] = dict()
        print(objective.__name__)
        for n_dim in [2,3,5,8,13,21,34]:
            benchmarks[objective.__name__][n_dim]=dict()
            for n_trials in [20,30,50,80,130,210,340]:
                best_val,best_x = scipy_dogleg_cube(objective=objective,n_trials=n_trials,n_dim=n_dim)
                benchmarks[objective.__name__][n_dim][n_trials] = best_val
            with open('bencharks.json','wt') as fp:
                json.dump(benchmarks,fp=fp)


