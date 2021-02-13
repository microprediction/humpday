import math
from typing import Union, Tuple, List
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

