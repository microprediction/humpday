

class NormHelper():

    # Statistics standard library introduced normal distribution but only in versions above 3.8
    # This adds a tiny amount of backward compatibility, but note that scipy is not a formal dependency so some users
    # will need to install that of their own volition. Python caches imports so don't worry too much about this.

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def normcdf(x):
        g = NormHelper._normcdf_function()
        return g(x)

    @staticmethod
    def norminv(p):
        f = NormHelper._norminv_function()
        return f(p)

    @staticmethod
    def _norminv_function():
        try:
            from statistics import NormalDist
            return NormalDist(mu=0, sigma=1.0).inv_cdf
        except ImportError:
            from scipy.stats import norm
            return norm.ppf

    @staticmethod
    def _normcdf_function():
        try:
            from statistics import NormalDist
            return NormalDist(mu=0, sigma=1.0).cdf
        except ImportError:
            from scipy.stats import norm
            return norm.cdf

