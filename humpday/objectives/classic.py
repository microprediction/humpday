# Some test objective functions to help guide optimizer choices
# -------------------------------------------------------------
#
# We'll use DEAP's set of groovy benchmarks, and landscapes, swarmpackagepy also
#
# See pretty pictures at https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks
# Some hardness assessment is at https://github.com/nathanrooy/landscapes#available-functions-from-single_objective but we'll do our own
## Basis of tricky functions
import datetime
import math
import random as _random

from humpday import _array as _A
from humpday.objectives.deapobjectives import (
    bohachevsky,
    griewank,
    rastrigin,
    rosenbrock,
    schaffer,
    schwefel,
    shekel,
)

DAY = datetime.datetime.today().day
OFFSET = DAY / 50
POWER = 1 + (DAY % 3) / 3.0
SHIFT = DAY / 100


def smoosh(ui):
    """Distort the interval to avoid obvious minima and avoid memorization"""
    ui_rotate = ui + SHIFT % 1.0
    ui_shift = ui_rotate + SHIFT
    xi = ui_shift**POWER
    low = SHIFT**POWER
    high = (1 + SHIFT) ** POWER
    yi = (xi - low) / (high - low)
    return yi**POWER


def schwefel_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schwefel
    u_squished = [1000 * (smoosh(ui) - 0.5) for ui in u]
    try:
        return 0.001 * schwefel(u_squished)[0] / 0.71063
    except Exception as e:
        raise Exception(e)


def griewank_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.griewank
    u_squished = [1200 * (ui**1.1 - 0.5) for ui in u]
    return griewank(u_squished)[0] / 0.532075


def rastrigin_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.rastrigin
    u_squished = [10.24 * (ui**1.1 - 0.5) for ui in u]
    return 0.01 * rastrigin(u_squished)[0] / 0.059697


def bohachevsky_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.bohachevsky
    u_squished = [10 * (ui**1.1 - 0.5) for ui in u]
    return 1.0 + bohachevsky(u_squished)[0]


def rosenbrock_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.rosenbrock
    u_squished = [200 * (ui**1.1 - 0.5) for ui in u]
    return 1 + 0.1 * rosenbrock(u_squished)[0] / 0.008949


def shaffer_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schaffer
    u_squished = [200 * (ui**1.1 - 0.5) for ui in u]
    return 0.01 * schaffer(u_squished)[0] / (0.1042133 * 0.71809)


_SHEKEL_PEAKS: dict = {}


def _shekel_peaks(n_dim: int):
    """Shekel's 15 peak locations in [0, 10]^n and widths, fixed per dimension.

    They used to be drawn from np.random on every call, which made the
    surface a different function each evaluation and consumed the global
    stream the optimizers draw from; and the cube was mapped to [-400, 400]^n
    while the peaks sat in [0, 10]^n, so the surface was constant to 1e-6
    everywhere. A fixed seed gives one landscape; DEAP's domain gives it
    structure.
    """
    if n_dim not in _SHEKEL_PEAKS:
        # A private stream, seeded by dimension, taken through the backend shim rather than
        # numpy so this module imports on a dependency-free install (#377). `use_portable_rng`
        # gives the cross-language PCG32; the optimizers' own stream is saved and restored
        # around it so building a landscape cannot disturb a run in progress.
        import humpday._array as _shim

        saved = _shim._portable
        try:
            _A.use_portable_rng(1729 + n_dim)
            peaks = [[10.0 * _A.rng_random() for _ in range(n_dim)] for _ in range(15)]
            widths = [_A.rng_random() for _ in range(15)]
        finally:
            _shim._portable = saved
        _SHEKEL_PEAKS[n_dim] = (peaks, widths)
    return _SHEKEL_PEAKS[n_dim]


def shekel_on_cube(u: [float]) -> float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.shekel
    A, C = _shekel_peaks(len(u))
    u_scaled = [10 * smoosh(ui) for ui in u]
    return 1.2298 - shekel(u_scaled, A, C)[0]


## Combinations
#
# A combination exists to blend behaviours, which only happens if each
# component is put on a common scale first. The `_on_cube` wrappers do not
# achieve that: on the unit cube qing_on_cube returns values around 1e13 and
# michaelewicz_on_cube around 1.4, so summing them and dividing by a constant
# gave qing plus a rounding error, rank-identical to qing alone. Each
# component is therefore scaled by its own empirical range on the cube at the
# dimension being evaluated -- measured once on a fixed sample and cached --
# and the combination is the mean of the scaled components.

_COMBO_SAMPLE = 64
_COMBO_RANGES: dict = {}


def _component_range(f, n_dim: int):
    key = (f.__name__, n_dim)
    if key not in _COMBO_RANGES:
        rnd = _random.Random(1729 + n_dim)
        values = [f([rnd.random() for _ in range(n_dim)]) for _ in range(_COMBO_SAMPLE)]
        lo, hi = min(values), max(values)
        _COMBO_RANGES[key] = (lo, (hi - lo) or 1.0)
    return _COMBO_RANGES[key]


def _blend(u, *components) -> float:
    n_dim = len(u)
    total = 0.0
    for f in components:
        lo, span = _component_range(f, n_dim)
        total += (f(u) - lo) / span
    return total / len(components)


def deap_combo1_on_cube(u: [float]) -> float:
    return _blend(u, schwefel_on_cube, griewank_on_cube, shekel_on_cube)


def deap_combo2_on_cube(u: [float]) -> float:
    return _blend(u, shaffer_on_cube, shekel_on_cube)


def deap_combo3_on_cube(u: [float]) -> float:
    return _blend(u, rosenbrock_on_cube, bohachevsky_on_cube, shekel_on_cube)


DEAP_OBJECTIVES = [
    schwefel_on_cube,
    rastrigin_on_cube,
    griewank_on_cube,
    bohachevsky_on_cube,
    rosenbrock_on_cube,
    shaffer_on_cube,
    shekel_on_cube,
    deap_combo1_on_cube,
    deap_combo2_on_cube,
    deap_combo3_on_cube,
]


# By hand...


def rosenbrock_modified_on_cube(u: [float]) -> float:
    """https://en.wikipedia.org/wiki/Rosenbrock_function"""
    u_scaled = [4 * ui - 2 for ui in u]
    if len(u) == 1:
        return (0.25 - u_scaled[0]) ** 2
    else:
        return 5 + 0.001 * _A.sum(
            [
                100 * (ui_plus - ui * ui) + (1 - ui) * (1 - ui)
                for ui, ui_plus in zip(u_scaled[1:], u_scaled)
            ]
        )


# According to http://infinity77.net/global_optimization/test_functions.html#test-functions-index
# there are some really hard ones
# See https://github.com/andyfaff/ampgo/blob/master/%20ampgo%20--username%20andrea.gavana%40gmail.com/go_benchmark.py
# See also https://arxiv.org/pdf/1308.4008v1.pdf


def damavandi_on_cube(u: [float]) -> float:
    """A trivial multi-dimensional extension of Damavandi's function"""
    return 0.01 * damavandi2(u[0], u[1]) - 0.46


def damavandi2(u1, u2) -> float:
    """Pretty evil function this one"""
    # http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Damavandi
    x1 = u1 / 14.0
    x2 = u2 / 14.0
    numerator = math.sin(math.pi * (x1 - 2.0)) * math.sin(math.pi * (x2 - 2.0))
    denumerator = (math.pi**2) * (x1 - 2.0) * (x2 - 2.0)
    factor1 = 1.0 - (abs(numerator / denumerator)) ** 5.0
    factor2 = 2 + (x1 - 7.0) ** 2.0 + 2 * (x2 - 7.0) ** 2.0
    return factor1 * factor2


def paviani_on_cube(u: [float]) -> float:
    # http://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Paviani
    x = [2.001 + 5.996 * smoosh(ui) for ui in u]

    def safe_log(values):
        return [math.log(max(float(v), 1e-6)) for v in values]

    lo = safe_log([xi - 2 for xi in x])
    hi = safe_log([10.0 - xi for xi in x])
    product = 1.0
    for xi in x:
        product *= float(xi)
    return (
        float(_A.sum([a * a + b * b for a, b in zip(lo, hi)]) - product**0.2) / 8.6456
    )


# These six were imported from `landscapes` when it happened to be installed, and silently
# replaced with `sum(xi**2)` -- the sphere -- when it was not. It is not a declared dependency and
# is not installed, so six names that read as distinct multimodal benchmarks were one separable
# convex quadratic. Measured Spearman rank correlation between them, d=5 over 200 random points:
# five of the six pairwise correlations were exactly 1.0000.
#
# That reached `humpday.objectives.SURFACES` (6 of 34 entries), the recommendation grid, which
# lists three of them as separate regimes, and `papers/planar_search`. In the grid it biased
# results toward coordinate-wise methods, which is precisely the axis-alignment advantage the
# rotated variants exist to remove.
#
# Implemented directly instead. The optional import is gone entirely rather than kept as a
# preferred path: an objective that changes definition depending on what is installed cannot be
# compared across runs, which is the same defect in a slower form. `humpday/objectives/README.md`
# records `enhanced_surfaces` being deleted for exactly this, around `opfunu`.
#
# Formulas as given by Jamil & Yang (2013) and the Simon Fraser test-function repository, with the
# domains the `_on_cube` wrappers below already assume.


def styblinski_tang(x) -> float:
    """0.5 * sum(x^4 - 16x^2 + 5x). Domain [-5, 5]; minimum -39.16599n at x_i = -2.903534."""
    return 0.5 * sum(xi**4 - 16.0 * xi**2 + 5.0 * xi for xi in x)


def salomon(x) -> float:
    """1 - cos(2*pi*||x||) + 0.1*||x||. Domain [-100, 100]; minimum 0 at the origin.

    Concentric ridges around the origin -- the reason it is in a benchmark suite at all, and
    exactly what the sphere fallback erased.
    """
    norm = math.sqrt(sum(xi * xi for xi in x))
    return 1.0 - math.cos(2.0 * math.pi * norm) + 0.1 * norm


def michalewicz(x, m: int = 10) -> float:
    """-sum(sin(x_i) * sin(i*x_i^2/pi)^(2m)). Steep ridges; `m` controls their sharpness.

    `i` is 1-based, as in the published definition.
    """
    total = 0.0
    for i, xi in enumerate(x, start=1):
        total += math.sin(xi) * math.sin(i * xi * xi / math.pi) ** (2 * m)
    return -total


def qing(x) -> float:
    """sum((x_i^2 - i)^2). Domain [-500, 500]; 2^n global minima at x_i = +/- sqrt(i)."""
    return sum((xi * xi - i) ** 2 for i, xi in enumerate(x, start=1))


def rotated_hyper_ellipsoid(x) -> float:
    """sum_{i=1..n} sum_{j=1..i} x_j^2. Domain [-65.536, 65.536]; convex but not separable."""
    total = 0.0
    running = 0.0
    for xi in x:
        running += xi * xi
        total += running
    return total


def zakharov(x) -> float:
    """sum(x^2) + (sum(0.5*i*x))^2 + (sum(0.5*i*x))^4. Domain [-5, 10]; minimum 0 at the origin."""
    s1 = sum(xi * xi for xi in x)
    s2 = sum(0.5 * i * xi for i, xi in enumerate(x, start=1))
    return s1 + s2**2 + s2**4


def styblinski_tang_on_cube(u: [float]) -> float:
    u_scaled = [10 * (smoosh(ui) - 0.5) for ui in u]
    return 3.3499 + 0.01 * styblinski_tang(u_scaled)


def zakharov_on_cube(u: [float]) -> float:
    u_scaled = [15 * smoosh(ui) - 10 for ui in u]
    return 0.01 * zakharov(u_scaled) / 0.3462


def salomon_on_cube(u: [float]) -> float:
    u_scaled = [200 * smoosh(ui) - 100 for ui in u]
    return salomon(u_scaled) / 3.09999


def rotated_hyper_ellipsoid_on_cube(u: [float]) -> float:
    u_scaled = [2 * 65.536 * smoosh(ui) - 65.536 for ui in u]
    return 0.1 * rotated_hyper_ellipsoid(u_scaled)


def qing_on_cube(u: [float]) -> float:
    u_scaled = [1000 * smoosh(ui) - 500 for ui in u]
    return qing(u_scaled) / 0.01805


def michaelewicz_on_cube(u: [float]) -> float:
    u_scaled = [4 * smoosh(ui) - 2 for ui in u]
    return 1.4439 + 0.1 * michalewicz(u_scaled, m=20)


def landscapes_combo1_on_cube(u: [float]) -> float:
    return _blend(u, qing_on_cube, michaelewicz_on_cube)


def landscapes_combo2_on_cube(u: [float]) -> float:
    return _blend(u, rotated_hyper_ellipsoid_on_cube, salomon_on_cube)


def landscapes_combo3_on_cube(u: [float]) -> float:
    return _blend(u, zakharov_on_cube, styblinski_tang_on_cube)


LANDSCAPES_OBJECTIVES = [
    styblinski_tang_on_cube,
    zakharov_on_cube,
    salomon_on_cube,
    rotated_hyper_ellipsoid_on_cube,
    qing_on_cube,
    michaelewicz_on_cube,
    landscapes_combo1_on_cube,
    landscapes_combo2_on_cube,
    landscapes_combo3_on_cube,
]


# Some copied from peabox
# https://github.com/stromatolith/peabox/blob/master/peabox/peabox_testfuncs.py
# as that isn't deployed to PyPI as far as I can determine


def ackley_on_cube(u: [float]) -> float:
    # allow parameter range -32.768<=x(i)<=32.768, global minimum at x=(0,0,...,0)
    rescaled_u = [2 * 32.768 * smoosh(ui) - 32.768 for ui in u]
    x = _A.asarray(rescaled_u)
    ndim = len(rescaled_u)
    a = 20.0
    b = 0.2
    c = 2.0 * math.pi
    return (
        -a * math.exp(-b * math.sqrt(1.0 / ndim * float(_A.sum(x * x))))
        - math.exp(1.0 / ndim * float(_A.sum(_A.cos(c * x))))
        + a
        + math.e
    ) / 20.0


# Adapted from https://github.com/SISDevelop/SwarmPackagePy/blob/master/SwarmPackagePy/testFunctions.py


def cross_on_cube(u):
    x = [5 * smoosh(ui) - 2.5 for ui in u]
    return round(
        -0.0001
        * (
            abs(
                math.sin(x[0])
                * math.sin(x[1])
                * math.exp(abs(100 - math.sqrt(sum([i**2 for i in x])) / math.pi))
            )
            + 1
        )
        ** 0.1,
        7,
    )


def powers_on_cube(u):
    x = [5 * smoosh(ui) - 2.5 for ui in u]
    return sum([abs(x[i]) ** (i + 2) for i in range(len(x))])


def booth_on_cube(u):
    x = [5 * smoosh(ui) - 2.5 for ui in u]
    return sum([abs(x[i]) ** (i + 2) for i in range(len(x))])


def matyas_on_cube(u):
    x = [3 * smoosh(ui) - 1.5 for ui in u]

    def sphere_function(x):
        return sum([i**2 for i in x])

    return 0.26 * sphere_function(x) - 0.48 * x[0] * x[1]


def drop_wave_on_cube(u):
    x = [3 * smoosh(ui) - 1.5 for ui in u]

    def sphere_function(x):
        return sum([i**2 for i in x])

    return -(1 + math.cos(12 * math.sqrt(sphere_function(x)))) / (
        0.5 * sphere_function(x) + 2
    )


SWARM_OBJECTIVES = [
    cross_on_cube,
    powers_on_cube,
    booth_on_cube,
    matyas_on_cube,
    drop_wave_on_cube,
]


# -----------------------------------------------------------------------------
# Rotated benchmarks
# -----------------------------------------------------------------------------
# Standard benchmarks like Rosenbrock and Rastrigin are axis-aligned — their
# coordinate-wise structure means coordinate-descent / Powell / NM all get a
# structural bonus that doesn't generalise. The literature standard for
# levelling the playing field is to multiply the input by a random rotation
# matrix Q before evaluating: the global optimum stays put, the function
# value is unchanged, but the level sets are no longer axis-aligned.
#
# We pick Q deterministically per (n_dim, seed) using the
# Mezzadri (2007) sign-corrected QR construction, which gives Q drawn from
# the Haar measure on O(n). Cached so the same Q is used by every worker
# process and every evaluation, without touching np.random's global state
# (instance-based RNG only).
import functools


@functools.cache
def _rotation_for(n_dim: int, seed: int = 12345):
    """Deterministic uniform-random orthogonal matrix Q(n_dim, seed).

    Built through the backend shim rather than numpy, so this module imports on a
    dependency-free install (#377). The optimizers' own RNG state is saved and restored, as in
    `_shekel_peaks`: constructing a landscape must not consume the stream a run is drawing from.
    """
    import humpday._array as _shim

    saved = _shim._portable
    try:
        _A.use_portable_rng(seed)
        rows = [[_A.rng_gauss() for _ in range(n_dim)] for _ in range(n_dim)]
    finally:
        _shim._portable = saved

    Q, R = _A.linalg.qr(rows)
    # Fix the sign convention so Q is unique: without it QR is only determined up to the sign
    # of each column, and two backends can disagree on a matrix that is meant to be fixed.
    signs = [
        1.0 if float(_A.linalg.diagonal(R)[i]) >= 0.0 else -1.0 for i in range(n_dim)
    ]
    return [[float(Q[i][j]) * signs[j] for j in range(n_dim)] for i in range(n_dim)]


def rotated_rosenbrock_on_cube(u):
    """Rosenbrock evaluated on Q·x, where x is the unit-cube point linearly
    mapped to a domain wide enough to contain Rosenbrock's banana valley.
    Q removes the axis-aligned structure that coordinate-wise methods
    exploit."""
    n = len(u)
    Q = _rotation_for(n)
    x = [4.0 * (float(ui) - 0.5) for ui in u]  # [-2, 2]^n
    y = _A.linalg.matvec(Q, x)
    return float(
        sum(100.0 * (y[i + 1] - y[i] ** 2) ** 2 + (1 - y[i]) ** 2 for i in range(n - 1))
    )


def rotated_rastrigin_on_cube(u):
    """Rastrigin evaluated on Q·x. Tests algorithms' ability to handle a
    multimodal landscape whose local-minima grid is rotated off-axis."""
    n = len(u)
    Q = _rotation_for(n)
    x = [10.24 * (float(ui) - 0.5) for ui in u]  # [-5.12, 5.12]^n
    y = _A.linalg.matvec(Q, x)
    return float(
        10.0 * n
        + _A.sum(
            [
                float(yi) * float(yi) - 10.0 * math.cos(2.0 * math.pi * float(yi))
                for yi in y
            ]
        )
    )


def rotated_ackley_on_cube(u):
    """Ackley evaluated on Q·x. Ackley is mildly multimodal with smooth
    global structure; rotating it specifically tests whether covariance-
    adapting algorithms (CMA-ES) recover their literature advantage on
    non-separable landscapes."""
    n = len(u)
    Q = _rotation_for(n)
    x = [65.536 * (float(ui) - 0.5) for ui in u]  # [-32.768, 32.768]^n
    y = [float(v) for v in _A.linalg.matvec(Q, x)]
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    return float(
        -a * math.exp(-b * math.sqrt(_A.sum([yi * yi for yi in y]) / n))
        - math.exp(_A.sum([math.cos(c * yi) for yi in y]) / n)
        + a
        + math.e
    )


A_CLASSIC_OBJECTIVE = rastrigin_on_cube  # Just pick one for testing

MISC_OBJECTIVES = [
    paviani_on_cube,
    damavandi_on_cube,
    rosenbrock_modified_on_cube,
    ackley_on_cube,
]

CLASSIC_OBJECTIVES = (
    DEAP_OBJECTIVES + LANDSCAPES_OBJECTIVES + MISC_OBJECTIVES + SWARM_OBJECTIVES
)

if __name__ == "__main__":
    for objective in CLASSIC_OBJECTIVES:
        objective(u=[0.0, 0.5, 1.0])
        objective(u=[0.0, 0.5, 0.0, 0.0, 1.0])
    print(len(CLASSIC_OBJECTIVES))
