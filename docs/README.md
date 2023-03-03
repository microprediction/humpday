# Humpday documentation ([install](https://github.com/microprediction/humpday/blob/main/INSTALL.md))

A package to help you in two main ways:

  1. Choose a Python global optimizer package without learning dozens of quirky usage conventions
  2. Optimize on a simplex (see [how](https://github.com/microprediction/humpday/blob/main/humpday/transforms/cubetosimplex.py)) regardless of the optimizer chosen

### Basic usage
Import an optimizer or three, and run them:

    from humpday.objectives.classic import CLASSIC_OBJECTIVES
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in DLIB_OPTIMIZERS:
            f_best, x_best, actual_num_trials = optimizer(objective, n_trials=500, n_dim=34, with_count=True)
            

Find what you want in [humpday/optimizers](https://github.com/microprediction/humpday/tree/main/humpday/optimizers), then in every definition file (e.g. [hyperoptcube.py](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/hyperoptcube.py)) you'll find an example of how to run the optimizers. There are 
more patterns in [basic usage](https://github.com/microprediction/humpday/tree/main/examples/basic_usage) examples, maybe.

### Packages leveraged
[Ax-Platform](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/axcube.py), [bayesian-optimization](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bayesoptcube.py), [DLib](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/dlibcube.py), [HyperOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/hyperoptcube.py), [NeverGrad](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nevergradcube.py), [Optuna](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/optunacube.py), [Platypus](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/platypuscube.py), [PyMoo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pymoocube.py), [PySOT](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pysotcube.py), Scipy [classic](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/scipycube.py) and [shgo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/shgocube.py), [Skopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/skoptcube.py),
[nlopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nloptcube.py), [Py-Bobyaq](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bobyqacube.py), 
[UltraOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/ultraoptcube.py), [FreeLunch](https://github.com/MDCHAMP/FreeLunch) and maybe others in a common calling syntax.  

### Elo ratings
See [article](https://microprediction.medium.com/interpreting-the-elo-ratings-for-python-global-optimizers-65304573e422).

### Articles:

- [A New Family of Diffeomorphisms from the Simplex to the Cube](https://medium.com/@microprediction/a-new-family-of-diffeomorphisms-from-the-simplex-to-the-cube-with-application-to-global-6d358714f429)
- [The Python Optimizer Form Guide](https://medium.com/geekculture/the-python-optimizer-form-guide-3b8ea3b4d78f)
- [Comparing Python Global Optimizer Packages](https://www.microprediction.com/blog/optimize)
- [Interpreting the Python Global Optimizer Elo Ratings](https://microprediction.medium.com/interpreting-the-elo-ratings-for-python-global-optimizers-65304573e422)


View as [web page](https://microprediction.github.io/humpday/) or [source](https://github.com/microprediction/humpday/blob/main/docs/README.md)


![simplex](/humpday/assets/images/simplex_map.png)
