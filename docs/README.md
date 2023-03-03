# Humpday documentation ([install](https://github.com/microprediction/humpday/blob/main/INSTALL.md))

A package to help you choose a Python global optimizer package

### Basic usage
Import an optimizer or three, and run them:

    from humpday.objectives.classic import CLASSIC_OBJECTIVES
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in DLIB_OPTIMIZERS:
            print(optimizer(objective, n_trials=500, n_dim=34, with_count=True))

Find what you want in [humpday/optimizers](https://github.com/microprediction/humpday/tree/main/humpday/optimizers), then in every definition file (e.g. [hyperoptcube.py](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/hyperoptcube.py)) you'll find an example of how to run the optimizers

More in [basic usage](https://github.com/microprediction/humpday/tree/main/examples/basic_usage) examples, maybe.

### Articles:

- [A New Family of Diffeomorphisms from the Simplex to the Cube](https://medium.com/@microprediction/a-new-family-of-diffeomorphisms-from-the-simplex-to-the-cube-with-application-to-global-6d358714f429)
- [The Python Optimizer Form Guide](https://medium.com/geekculture/the-python-optimizer-form-guide-3b8ea3b4d78f)
- [Comparing Python Global Optimizer Packages](https://www.microprediction.com/blog/optimize)


View as [web page](https://microprediction.github.io/humpday/) or [source](https://github.com/microprediction/humpday/blob/main/docs/README.md)



![simplex](/humpday/assets/images/simplex_map.png)
