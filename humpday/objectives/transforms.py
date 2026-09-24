import math
from typing import List, Tuple, Union

BOUNDS_TYPE = List[
    Union[Tuple, List]
]  # scipy.optimize style bounds [ (low,high), (low, high),... ]


# Misc mappings that might be useful


def positive_log_scale(u, low, high):
    """Map u in [0,1] to [low,high] logarithmically, with both endpoints exact"""
    assert 0 <= u <= 1
    assert 0 < low < high
    if u == 0:
        return low
    if u == 1:
        return high
    log_low = math.log(low)
    log_high = math.log(high)
    x = log_low + u * (log_high - log_low)
    return min(high, max(low, math.exp(x)))


# Share of the unit interval given to each logarithmic side of a zero-crossing map
_LOG_SIDE_WIDTH = 0.475


def to_log_space_1d(u, low, high):
    """Approximately logarithmic map, but allows for ranges spanning zero

    An interval that excludes zero is mapped logarithmically in its magnitude. One that
    includes zero is linear on [-s, s] around it, with s one hundredth of the width
    (shrunk so it does not pass either endpoint), and logarithmic on each side beyond
    that. A side with nothing beyond s, such as a zero endpoint, gets no share of u.

    returns:  float between low and high, monotone in u, equal to low at 0 and high at 1
    """
    assert 0 <= u <= 1
    assert low < high

    if u == 0:
        return low
    if u == 1:
        return high
    if low > 0:
        return positive_log_scale(u=u, low=low, high=high)
    if high < 0:
        return -positive_log_scale(1 - u, low=-high, high=-low)

    scale = (high - low) / 100
    neg_scale = min(scale, -low)
    pos_scale = min(scale, high)
    neg_width = _LOG_SIDE_WIDTH if -low > neg_scale else 0.0
    pos_width = _LOG_SIDE_WIDTH if high > pos_scale else 0.0
    mid_width = 1 - neg_width - pos_width

    if u < neg_width:
        return -positive_log_scale(u=1 - u / neg_width, low=neg_scale, high=-low)
    if u <= neg_width + mid_width:
        v = (u - neg_width) / mid_width
        return min(high, max(low, -neg_scale + v * (neg_scale + pos_scale)))
    v = (u - neg_width - mid_width) / pos_width
    return positive_log_scale(u=min(1.0, v), low=pos_scale, high=high)
