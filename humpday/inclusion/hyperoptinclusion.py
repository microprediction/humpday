
try:
    from hyperopt import fmin, hp, tpe, Trials
    from hyperopt.tpe import suggest as tpe_suggest
    from hyperopt.rand import suggest as rand_suggest
    from hyperopt.atpe import suggest as atpe_suggest
    using_hyperopt = True
except ImportError:
    using_hyperopt = False

if __name__=='__main__':
    if not using_hyperopt:
        print('pip install hyperopt')