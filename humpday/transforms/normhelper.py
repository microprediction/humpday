class NormHelper:
    # Statistics standard library introduced normal distribution but only in versions above 3.8
    # This adds a tiny amount of backward compatibility, but note that scipy is not a formal dependency so some users
    # will need to install that of their own volition. Python caches imports so don't worry too much about this.

    def __init__(self, **kwargs):
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
        """The standard normal CDF, computed through erfc so that the lower tail survives.

        statistics.NormalDist.cdf evaluates 0.5 * (1 + erf(x / sqrt(2))). For x below about -6
        the erf term is within a double's last bit of -1 and the sum cancels: NormalDist().cdf
        returns 1.11e-16 at -8.2, where the true value is 1.20e-16, and exactly 0.0 at -9.21,
        where it is 1.63e-20. Anything inverting that CDF -- the simplex transform below does --
        loses every concentrated point to a flat zero.

        0.5 * erfc(-x / sqrt(2)) is the same function without the cancellation, accurate into the
        1e-300s, and math.erfc is in the standard library everywhere this package runs, so this
        needs no fallback of its own.
        """
        import math

        _SQRT2 = math.sqrt(2.0)
        return lambda x: 0.5 * math.erfc(-float(x) / _SQRT2)
