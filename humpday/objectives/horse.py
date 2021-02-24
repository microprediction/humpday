
# The horse racing problem
import numpy as np
import math
from winning.lattice import skew_normal_density
from winning.lattice_calibration import implied_ability, ability_implied_dividends

global OFFSETS
OFFSETS = None
HORSE_DIM = 500 # Maximum dimension
global DIVIDENDS
DIVIDENDS = None
unit = 0.01
L = 500
DENSITY = skew_normal_density(L=500, unit=unit, a=1.5)


def make_offsets():
    global OFFSETS
    if OFFSETS is None:
        from datetime import datetime
        day_of_year = datetime.now().timetuple().tm_yday
        np.random.seed(day_of_year)
        OFFSETS = sorted(np.random.randn(HORSE_DIM)/unit)
        OFFSETS = [0] + [ o-OFFSETS[0] for o in OFFSETS[1:] ]
    return OFFSETS


def make_dividends(n_dim):
    global DIVIDENDS
    if DIVIDENDS is None:
        DIVIDENDS = dict()
    if not DIVIDENDS.get(n_dim):
        DIVIDENDS[n_dim] = ability_implied_dividends(ability=make_offsets()[:n_dim], density=DENSITY)
    return DIVIDENDS[n_dim]


def cube_to_offsets(u:[float])->[float]:
    return [0] + list(np.arctanh(np.array(np.minimum(u, 1 - 1e-5))) / unit)


def horse_dividends_on_cube(u:[float])->float:
    """ Find relative ability matching market prices
    :param u:  Determines ability of 2nd through last horse, via arctanh
    :return: [float] of dividends (return on betting $1)
    """
    n_dim = len(u)+1
    dividends = make_dividends(n_dim)
    offsets = cube_to_offsets(u=u)
    implied_dividends = ability_implied_dividends(ability=offsets, density=DENSITY)
    discrepancy = np.mean( [ abs(math.sqrt(d1)-math.sqrt(d2)) for d1,d2 in zip(dividends,implied_dividends)])
    return discrepancy


HORSE_OBJECTIVES = [ horse_dividends_on_cube ]


if __name__=='__main__':
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube
    v, u = nevergrad_ngopt8_cube(horse_dividends_on_cube,n_dim=20, n_trials=25000)
    o = cube_to_offsets(u)
    print(' ')
    print('Horse ability, and best solution found ')
    print(list(zip(o,OFFSETS)))