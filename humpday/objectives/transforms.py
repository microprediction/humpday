import math
from typing import Union, Tuple, List
from microconventions.zcurve_conventions import ZCurveConventions
BOUNDS_TYPE = List[Union[Tuple, List]]  # scipy.optimize style bounds [ (low,high), (low, high),... ]


# Misc mappings that might be useful


def positive_log_scale(u, low, high):
    """ Map u in (0,1) to (low,high) """
    assert 0 <= u <= 1
    assert 0 < low < high
    log_low = math.log(low)
    log_high = math.log(high)
    x = log_low + u * (log_high - log_low)
    return math.exp(x)


def to_log_space_1d(u, low, high):
    """ Approximately logarithmic map, but allows for ranges spanning zero
         returns:  float between low and high
    """
    assert 0 <= u <= 1
    assert low<high

    if 1e-8 < low < high:
        return positive_log_scale(u=u, low=low, high=high)
    elif low < -1e-8 < high < 1e-8:
        return -positive_log_scale(1 - u, low=-high, high=-low)
    elif -1e-8 < low < 1e-8 < high:
        return positive_log_scale(u=u, low=1e-8, high=high)
    elif low < -1e-8 < high < 1e-8:
        return -positive_log_scale(1 - u, low=1e-8, high=-low)
    else:
        scale = abs(high - low) / 100
        if u < 0.475:
            u1 = 1 - u / 0.475
            return -positive_log_scale(u=u1, low=scale, high=-low)
        elif 0.475 < u < 0.525:
            u2 = 20 * (u - 0.475)
            return -scale + 2 * u2 * scale
        else:
            u3 = (u - 0.525) / 0.525
            return positive_log_scale(u3, low=scale, high=high)


def to_space(p: float, bounds: BOUNDS_TYPE = None, dim: int = 1):
    """ Interprets p as a point in a rectangle in R^2 or R^3

         :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube
         :param dim     Dimension. Only used if bounds are not supplied.

    """
    if bounds is None:
        bounds = [(0, 1) for _ in range(dim)]
    else:
        dim = len(bounds)

    if dim>1:
        us = reversed(ZCurveConventions().to_cube(zpercentile=p, dim=dim))  # 0 < us[i] < 1
    else:
        us = [p]
    return [u * (b[1] - b[0]) + b[0] for u, b in zip(us, bounds)]


def from_space(ps: [float], bounds: BOUNDS_TYPE=None)->float:
    """ [ , ]^n -> [0,1] """
    if bounds is None:
        bounds = [(0, 1) for _ in range(len(ps))]
    us = [(pi - b[0]) / (b[1] - b[0]) for pi, b in zip(ps, bounds)]
    for u in us:
        assert 0 <= u <= 1, "bounds are inconsistent with p=" + str(ps)
    if len(us)>1:
        return ZCurveConventions().from_cube(list(reversed(us)))
    else:
        return us[0]


def to_log_space(p:float, bounds: BOUNDS_TYPE):
    """ Interprets p as a point in a rectangle in R^2 or R^3 using Morton space-filling curve

            :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube
            :param dim     Dimension. Only used if bounds are not supplied.

       Very similar to "to_space" but assumes speed varies with logarithm
       """
    assert 0 <= p <= 1
    dim = len(bounds)
    us = list(reversed(ZCurveConventions().to_cube(zpercentile=p, dim=dim)))  # 0 < us[i] < 1
    return [to_log_space_1d(u, low=b[0], high=b[1]) for u, b in zip(us, bounds)]


def to_int_log_space(p: float, bounds: BOUNDS_TYPE):
    """ Interprets p as a point in an integer lattice in R^2 or R^3 using Morton space-filling curve, integers only

            :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube

       Very similar to "to_space" but assumes speed varies with logarithm
       """
    assert 0 <= p <= 1
    prms = to_log_space(p=p, bounds=bounds)
    return [int(prm) for prm in prms]
