# humpday ![tests](https://github.com/microprediction/humpday/workflows/tests/badge.svg) ![nlopt](https://github.com/microprediction/humpday/workflows/test-nlopt/badge.svg) ![ax-platform](https://github.com/microprediction/humpday/workflows/test-ax/badge.svg) ![py-bobyqa](https://github.com/microprediction/humpday/workflows/test-bobyqa/badge.svg) ![dlib](https://github.com/microprediction/humpday/workflows/test-dlib/badge.svg) ![hyperopt](https://github.com/microprediction/humpday/workflows/test-hyperopt/badge.svg) ![pySOT](https://github.com/microprediction/humpday/workflows/test-pySOT/badge.svg) ![skopt](https://github.com/microprediction/humpday/workflows/test-skopt/badge.svg)![hebo](https://github.com/microprediction/humpday/workflows/test-hebo/badge.svg) ![nevergrad](https://github.com/microprediction/humpday/workflows/test-nevergrad/badge.svg) ![optuna](https://github.com/microprediction/humpday/workflows/test-optuna/badge.svg) ![bayesopt](https://github.com/microprediction/humpday/workflows/test-bayesopt/badge.svg) ![platypus](https://github.com/microprediction/humpday/workflows/test-platypus/badge.svg) ![pymoo](https://github.com/microprediction/humpday/workflows/test-pymoo/badge.svg) ![ultraopt](https://github.com/microprediction/humpday/workflows/test-ultraopt/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Behold! A [colab notebook](https://github.com/microprediction/humpday/blob/main/black_box_optimization_package_recommender.ipynb) that recommends a black-box optimizer for your objective function. 

Behold! Fifty strategies assigned [Elo ratings](https://microprediction.github.io/optimizer-elo-ratings/html_leaderboards/overall.html) depending on dimension of the problem and number of function evaluations allowed. 

Hello and welcome to HumpDay, a package that helps you choose a Python global optimizer package, and strategy therein, from [Ax-Platform](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/axcube.py), [bayesian-optimization](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bayesoptcube.py), [DLib](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/dlibcube.py), [HyperOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/hyperoptcube.py), [NeverGrad](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nevergradcube.py), [Optuna](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/optunacube.py), [Platypus](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/platypuscube.py), [PyMoo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pymoocube.py), [PySOT](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pysotcube.py), Scipy [classic](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/scipycube.py) and [shgo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/shgocube.py), [Skopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/skoptcube.py),
[nlopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nloptcube.py), [Py-Bobyaq](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bobyqacube.py), 
[UltraOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/ultraoptcube.py) and maybe others by the time you read this. It also presents *some* of their functionality in a common calling syntax.  
 
 
### Install base library

    pip install humpday
    
Bleeding edge:

    pip install git+https://github.com/microprediction/humpday
  
File an issue if you have problems. See [this thread](https://stackoverflow.com/questions/65745683/how-to-install-scipy-on-apple-silicon-arm-m1) if you have 
issues on mac silicon M1. 

#### This might help some of you sometimes

    pip install cython pybind11
    brew install openblas
    export OPENBLAS=/opt/homebrew/opt/openblas/lib/

#### Add optimizers 

    pip install scikit-optimize
    pip install nevergrad
    pip install optuna
    pip install platypus-opt
    pip install poap
    pip install pysot
    pip install bayesian-optimization
    
    

Some of these are really good, but not 100% stable on all platforms we've used. 

    pip install cmake
    pip install ultraopt
    pip install dlib 
    pip install ax-platform
    pip install py-bobyqa
    pip install hebo
    pip install nlopt

## Recommendations

Pass the dimensions of the problem, function evaluation budget and
 time budget to receive [suggestions](https://github.com/microprediction/humpday/blob/main/humpday/comparison/suggestions.py) that are independent of your problem set,
 
        from pprint import pprint 
        from humpday import suggest
        pprint(suggest(n_dim=5, n_trials=130,n_seconds=5*60))
        
where *n_seconds* is the total computation budget for the optimizer (not the objective function) over all 130 function evaluations. Or simply pass your objective function, and it will time it and do something sensible:
     
        from humpday import recommend
    
        def my_objective(u):
            time.sleep(0.01)
            return u[0]*math.sin(u[1])

        recommendations = recommend(my_objective, n_dim=21, n_trials=130)

## Meta-minimizer

If you are feeling lucky, the [meta](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/meta.py) minimizer which will
 choose an optimizer based only on dimension and number of function evaluations, then run it:   

        from humpday import minimize
        best_val, best_x = minimize(objective, n_dim=13, n_trials=130 )
        
Here and elsewhere, *objective* is intended to be minimized on the hypercube [0,1]^n_dim.  

## Points race
        
If you have more time, call [points_race](https://github.com/microprediction/humpday/blob/main/humpday/comparison/odious.py) on a list of your own objective functions:

        from humpday import points_race
        points_race(objectives=[my_objective]*2,n_dim=5, n_trials=100)
        
See the [colab notebook](https://github.com/microprediction/humpday/blob/main/black_box_optimization_package_recommender.ipynb).

## How it works 

In the background, 50+ strategies are assigned [Elo ratings](https://github.com/microprediction/optimizer-elo-ratings/tree/main/results/leaderboards) by sister repo [optimizer-elo-ratings](https://github.com/microprediction/optimizer-elo-ratings).

## Contribute

By all means contribute more to [optimizers](https://github.com/microprediction/humpday/tree/main/humpday/optimizers). 



![](https://i.imgur.com/FCiSrMQ.png)
 

    
## Articles 

- (most recent) [HumpDay: A Package to Take the Pain Out of Choosing a Python Optimizer](https://www.microprediction.com/blog/humpday). 
- [Comparing Python Global Optimizers](https://www.microprediction.com/blog/optimize).

