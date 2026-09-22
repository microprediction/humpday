# The horse racing problem
import math

from humpday import _array as _A
from humpday.transforms.thurstone_transform import (
    simple_ability_implied_dividends as std_ability_implied_dividends,
)

global ABILITIES
ABILITIES = None
HORSE_DIM = 500  # Maximum dimension
global DIVIDENDS
DIVIDENDS = None


def make_abilities():
    global ABILITIES
    if ABILITIES is None:
        from datetime import datetime

        # Still date-seeded, which is #373's complaint, not this change's. What changes here is
        # only where the draws come from: the backend shim rather than numpy, so this module
        # imports on a dependency-free install (#377). The optimizers' stream is saved and
        # restored around it.
        import humpday._array as _shim

        day_of_year = datetime.now().timetuple().tm_yday
        saved = _shim._portable
        try:
            _A.use_portable_rng(day_of_year)
            ABILITIES = sorted(_A.rng_gauss() for _ in range(HORSE_DIM))
        finally:
            _shim._portable = saved
    return ABILITIES


def make_dividends(n_dim):
    global DIVIDENDS
    if DIVIDENDS is None:
        DIVIDENDS = dict()
    if not DIVIDENDS.get(n_dim):
        DIVIDENDS[n_dim] = std_ability_implied_dividends(
            ability=make_abilities()[:n_dim]
        )
    return DIVIDENDS[n_dim]


def cube_to_ability(u: [float]) -> [float]:
    offsets = [0] + [math.atanh(min(float(ui), 1 - 1e-5)) for ui in u]
    ability = [min(5, o / 100) for o in offsets]
    return ability


def horse_dividends_on_cube(u: [float]) -> float:
    """Find relative cubetosimplex.py matching market prices
    :param u:  Determines cubetosimplex.py of 2nd through last horse, via arctanh
    :return: [float] of dividends (return on betting $1)
    """
    n_dim = len(u) + 1
    dividends = make_dividends(n_dim)
    ability = cube_to_ability(u=u)
    implied_dividends = std_ability_implied_dividends(ability=ability)
    gaps = [
        abs(math.sqrt(d1) - math.sqrt(d2))
        for d1, d2 in zip(dividends, implied_dividends)
    ]
    discrepancy = _A.sum(gaps) / max(1, len(gaps))
    return discrepancy


HORSE_OBJECTIVES = [horse_dividends_on_cube]  # Seems unstable and needs fixin'


if __name__ == "__main__":
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube

    v, u = nevergrad_ngopt8_cube(horse_dividends_on_cube, n_dim=20, n_trials=25000)
    o = cube_to_ability(u)
    print(" ")
    print("Horse cubetosimplex.py, and best solution found ")
    print(list(zip(o, ABILITIES)))
