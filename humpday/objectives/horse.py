
# The horse racing problem
import numpy as np
import math
from winning.std_calibration import std_ability_implied_dividends

global ABILITIES
ABILITIES = None
HORSE_DIM = 500 # Maximum dimension
global DIVIDENDS
DIVIDENDS = None



def make_abilities():
    global ABILITIES
    if ABILITIES is None:
        from datetime import datetime
        day_of_year = datetime.now().timetuple().tm_yday
        np.random.seed(day_of_year)
        ABILITIES = sorted(np.random.randn(HORSE_DIM))
    return ABILITIES


def make_dividends(n_dim):
    global DIVIDENDS
    if DIVIDENDS is None:
        DIVIDENDS = dict()
    if not DIVIDENDS.get(n_dim):
        DIVIDENDS[n_dim] = std_ability_implied_dividends(ability=make_abilities()[:n_dim])
    return DIVIDENDS[n_dim]


def cube_to_ability(u:[float])->[float]:
    offsets =  [0] + list(np.arctanh(np.array(np.minimum(u, 1 - 1e-5))))
    ability = [ min(5,o/100) for o in offsets ]
    return ability


def horse_dividends_on_cube(u:[float])->float:
    """ Find relative ability matching market prices
    :param u:  Determines ability of 2nd through last horse, via arctanh
    :return: [float] of dividends (return on betting $1)
    """
    n_dim = len(u)+1
    dividends = make_dividends(n_dim)
    ability = cube_to_ability(u=u)
    implied_dividends = std_ability_implied_dividends(ability=ability)
    discrepancy = np.mean( [ abs(math.sqrt(d1)-math.sqrt(d2)) for d1,d2 in zip(dividends,implied_dividends)])
    return discrepancy


HORSE_OBJECTIVES = [ horse_dividends_on_cube ] # Seems unstable and needs fixin'



if __name__=='__main__':
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube
    v, u = nevergrad_ngopt8_cube(horse_dividends_on_cube,n_dim=20, n_trials=25000)
    o = cube_to_ability(u)
    print(' ')
    print('Horse ability, and best solution found ')
    print(list(zip(o, ABILITIES)))