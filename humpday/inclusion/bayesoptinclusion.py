try:
    from bayes_opt import BayesianOptimization
    using_bayesopt = True
    print('Bayesian-Optimization warning:  https://github.com/fmfn/BayesianOptimization/issues/300')
except ImportError:
    using_bayesopt = False

using_bayesopt = False # <--- Just failing too often for my liking

if __name__=='__main__':
    print(using_bayesopt)