"""
Simplified cube-to-simplex transformations using Thurstone conventions.
Replaces the winning package dependency with thurstone-based implementations.
"""

import functools
import math
from typing import List, Sequence

# Neither scipy nor numpy is a dependency. scipy appears in pyproject.toml only under the
# `reference` test extra and numpy only under `fast`, so a plain `pip install humpday` provides
# neither. A module-scope `from scipy.stats import norm` here once made this module,
# cubetosimplex, objectives.horse and objectives.allobjectives all unimportable from a real
# install while passing every test, because the dev environment has scipy; `import numpy as np`
# was still doing the same thing to the same four modules. Everything this module needs is scalar
# arithmetic, so it is done in `math`, and NormHelper wraps the two normal functions with a scipy
# fallback for environments without statistics.NormalDist.
from humpday.transforms.normhelper import NormHelper

# Scale of the latent-normal ("Thurstone") map between cube and simplex: the cube coordinate u_i
# becomes the log-odds z_i = Phi^-1(u_i) / scale against a reference component held at zero, and
# the simplex point is the softmax of those. It cancels exactly in the round trip, so any positive
# value is a bijection between the open cube and its image, but it sets how much of the simplex
# the cube can reach, and how the cube's interior is spread over it. Large values compress the
# image toward the centroid -- the old value of 500, a horse-racing convention and formerly the
# sole use of the `thurstone` package dependency, collapsed the whole cube onto a tiny ball there.
# Small values push the image out toward the vertices. At 1.0 this is exactly softmax(probit(u)),
# the logistic-normal (additive-logistic) map.
#
# This is the default for the `scale` argument of the functions below rather than the only
# available value, because it is the preconditioner the README and the papers treat as tunable.
STD_L = 1.0

# The widest cube a double can carry. Phi^-1 is finite over exactly this interval: below 5e-324
# there are no positive doubles left, and above 1 - 2^-53 there are no doubles between the
# argument and one. The previous [1e-10, 1 - 1e-10] clip was three hundred million times narrower
# at the bottom end and, with `scale` at 1, confined a two-component simplex to weights in
# [0.00172, 0.99828] -- so a portfolio wanting a component below a sixth of a percent could not be
# expressed at all, whatever the optimizer or its budget.
_U_MIN = math.nextafter(0.0, 1.0)
_U_MAX = math.nextafter(1.0, 0.0)

# Phi^-1 at those two ends: about -38.47 and +8.21. The gap between them is not a bug to be fixed
# but the shape of the double-precision cube: doubles crowd towards zero and are spaced 2.2e-16
# apart just below one, so a coordinate can say "this component is vanishingly small" far more
# precisely than it can say "this component is nearly everything".
_Z_MIN = NormHelper._norminv_function()(_U_MIN)
_Z_MAX = NormHelper._norminv_function()(_U_MAX)


def attainable_weights(n_dim: int, scale: float = None) -> dict:
    """What `cube_to_simplex_simple` can actually return, for `n_dim` cube coordinates.

    The image is an open subset of the (n_dim + 1)-simplex, and these are its bounds, so a caller
    with a concentrated allocation in mind can check whether this map at this scale can express it
    rather than discovering a silent clip in the answer.

    ``non_reference`` is the range available to any one of components 1..n, and ``reference`` the
    range available to component 0, the one held at a logit of zero. Each bound is what that
    component reaches with the other coordinates pushed the other way, so it is the range of the
    image and not the range along the diagonal.

    The reference component is the asymmetric one: it can be driven to within 2e-17 of the whole
    allocation, but not below 1 / (1 + n * exp(8.21 / scale)), which at the default scale and two
    components is 2.7e-4. That asymmetry is the double-precision cube showing through -- a
    coordinate can say "this component is vanishingly small" much more precisely than "this
    component is nearly everything".

    A smaller `scale` widens every bound, which is what it is for: at ``scale=0.24``, the value
    the dfo_recommender outline tunes to, a two-component simplex reaches 2.5e-70 and 1 - 1.4e-15.
    """
    scale = STD_L if scale is None else float(scale)
    _check_scale(scale)
    lo, hi = math.exp(_Z_MIN / scale), math.exp(_Z_MAX / scale)
    others = n_dim - 1
    return {
        "non_reference": (
            lo / (1.0 + lo + others * hi),
            hi / (1.0 + hi + others * lo),
        ),
        "reference": (1.0 / (1.0 + n_dim * hi), 1.0 / (1.0 + n_dim * lo)),
    }


def _check_scale(scale: float) -> None:
    if not (scale > 0.0 and math.isfinite(scale)):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")


# Alternative implementation without winning package dependency
def cube_to_simplex_simple(u: Sequence[float], scale: float = None) -> List[float]:
    """
    Convert point on hypercube to simplex using a simplified method.
    This replaces the winning package dependency with a direct implementation.

    Coordinates outside the representable open cube are clamped to its ends rather than rejected,
    because optimizers propose the closed cube and an exact 0.0 or 1.0 is a legitimate proposal;
    see `attainable_weights` for what that reaches.

    :param u: point on the interior of the hyper-cube (0,1)^n
    :param scale: latent scale, defaulting to STD_L
    :returns: a point p in (0,1)^{n+1} with sum(p)=1
    """
    scale = STD_L if scale is None else float(scale)
    _check_scale(scale)
    _ppf = NormHelper._norminv_function()
    logits = [0.0]
    for ui in u:
        ui = float(ui)
        if math.isnan(ui):
            raise ValueError("cube coordinates must be numbers, got nan")
        logits.append(_ppf(min(_U_MAX, max(_U_MIN, ui))) / scale)

    # Softmax about the largest logit. Without the shift, a concentrated allocation overflows on
    # the way to being expressed: at scale 0.24 the largest logit is exp(34.2), and at the scales
    # a caller may reasonably want below that, inf/inf is nan and the point is lost.
    hi = max(logits)
    weights = [math.exp(z - hi) for z in logits]
    total = math.fsum(weights)
    return [w / total for w in weights]


def simplex_to_cube_simple(p: Sequence[float], scale: float = None) -> List[float]:
    """
    Inverse transformation from simplex to cube.

    Exact zeros are permitted, being the closure of the image, and come back as the smallest
    cube coordinate rather than as an error: the open simplex is what the forward map covers, so
    a vertex round-trips to the most concentrated allocation the map can express.

    :param p: point in [0,1]^{n+1} with entries summing to unity
    :param scale: latent scale, defaulting to STD_L
    :returns: point in (0,1)^n
    """
    scale = STD_L if scale is None else float(scale)
    _check_scale(scale)
    weights = [float(v) for v in p]
    if len(weights) < 2:
        raise ValueError(
            f"a simplex point needs at least two components, got {len(weights)}"
        )
    if any(not math.isfinite(v) or v < 0.0 for v in weights):
        raise ValueError(
            f"a simplex point needs finite non-negative weights, got {list(p)!r}"
        )
    total = math.fsum(weights)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"a simplex point must sum to one, got {total!r}")

    # log of the smallest positive double, so a zero weight becomes a log-ratio far outside
    # anything Phi can distinguish rather than the silent 1e-10 floor this used to apply to every
    # weight, tiny or not -- that floor turned a legitimate 1e-12 allocation into 1e-10.
    log_reference = math.log(max(weights[0], _U_MIN))
    _cdf = NormHelper._normcdf_function()
    return [
        min(
            _U_MAX,
            max(_U_MIN, _cdf(scale * (math.log(max(w, _U_MIN)) - log_reference))),
        )
        for w in weights[1:]
    ]


def lift_to_cube_simple(objective, fail_value=100000, scale: float = None):
    """
    Modify a function's domain from the simplex to the cube.
    Uses the simplified transformation above.

    `scale` is the preconditioner: it decides where in the cube an allocation lands, and so
    whether an optimizer sampling the cube evenly ever proposes it. A concentrated optimum is
    reachable at any scale, and findable only at a scale that puts it somewhere an optimizer
    looks.
    """

    @functools.wraps(objective)
    def wrapper(us):
        try:
            s = cube_to_simplex_simple(us, scale)
        except:
            return fail_value

        try:
            return objective(s)
        except:
            warn_msg = (
                f"WARNING: The func {objective.__name__} failed on the point {str(s)}"
            )
            raise ValueError(warn_msg)

    return wrapper


def minimize_optimizer_on_simplex_simple(
    optimizer,
    objective,
    n_trials,
    n_dim,
    with_count=False,
    fail_value=100000,
    return_point_on_simplex=False,
    scale: float = None,
    **kwargs,
):
    """
    Minimize objective on the n_dim-simplex using simplified transformations.

    :param optimizer: Optimizer function
    :param objective: Objective function expecting a k+1 vector
    :param n_trials: Number of trials
    :param n_dim: The manifold dimension of the simplex (1 less than actual dimension)
    :param with_count: Whether to return evaluation count
    :param fail_value: Value returned if mapping fails
    :param return_point_on_simplex: If True, return point on simplex
    :return: Same format as other optimizers
    """
    lifted_objective_on_cube = lift_to_cube_simple(
        objective=objective, fail_value=fail_value, scale=scale
    )
    f_best, x_best, feval_count = optimizer(
        lifted_objective_on_cube,
        n_trials=n_trials,
        n_dim=n_dim,
        with_count=True,
        **kwargs,
    )

    if return_point_on_simplex:
        try:
            s_best = cube_to_simplex_simple(x_best, scale)
        except:
            print(x_best)
            raise ValueError("Could not move optimal point back to simplex")
    else:
        s_best = list(x_best)

    if with_count:
        return f_best, s_best, feval_count
    else:
        return f_best, s_best


# Simple horse racing objective that doesn't require winning package
def simple_ability_implied_dividends(ability: List[float]) -> List[float]:
    """
    Simplified version of ability to dividends conversion.
    This replaces the winning package function with a direct implementation.
    """
    scaled = [float(a) / STD_L for a in ability]
    hi = max(scaled) if scaled else 0.0
    exp_ability = [math.exp(a - hi) for a in scaled]
    total = math.fsum(exp_ability)
    probabilities = [w / total for w in exp_ability]

    # Convert probabilities to dividends (inverse probabilities)
    return [1.0 / max(pi, 1e-10) for pi in probabilities]


if __name__ == "__main__":
    # Test the transformations
    import random

    # Test cube to simplex and back
    for _ in range(5):
        u = [random.random() for _ in range(3)]
        s = cube_to_simplex_simple(u)
        u_back = simplex_to_cube_simple(s)

        print(f"Original u: {u}")
        print(f"Simplex s: {s} (sum={math.fsum(s):.6f})")
        print(f"Recovered u: {u_back}")
        error = math.fsum(abs(a - b) for a, b in zip(u, u_back)) / len(u)
        print(f"Error: {error}")
        print()
