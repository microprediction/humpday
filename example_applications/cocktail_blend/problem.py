"""
Cocktail Blend — an inverse problem on the unit simplex.

Mix six ingredients into proportions that sum to 1 so the resulting
flavour profile matches a target as closely as possible. The decision
variable is a *composition* (a point on the 5-simplex), not a box point,
so this is the canonical case for the cube->simplex bijection: we hand
HumpDay an ordinary `[0,1]^5` objective whose first act is to lift the
cube point onto the simplex via `humpday.transforms.cubetosimplex`.

What it stresses:
  - Unit-simplex geometry (proportions summing to 1) — a constraint type
    none of the box-bounded demos exercise. The bijection turns it into a
    plain box problem, which is the whole point.
  - An over-determined target (6 flavour axes, 5 free proportions) so the
    optimum is a genuine constrained least-squares blend, not a vertex and
    not an exact zero.

Flavour axes: sweet, sour, bitter, boozy, herbal, citrus (each 0–10).
Mixing is linear in the proportions (perceived intensity = weighted mean
of the ingredient profiles); soda water is the pure-dilution ingredient.
"""

from __future__ import annotations

import math

from humpday.transforms.cubetosimplex import cube_to_simplex

INGREDIENTS = (
    "gin",
    "sweet_vermouth",
    "campari",
    "lime_juice",
    "simple_syrup",
    "soda_water",
)

AXES = ("sweet", "sour", "bitter", "boozy", "herbal", "citrus")

# Flavour profile of each ingredient over AXES, on a 0–10 scale.
FLAVOUR = {
    "gin": (0.0, 0.0, 0.0, 9.0, 3.0, 1.0),
    "sweet_vermouth": (7.0, 0.0, 1.0, 4.0, 6.0, 0.0),
    "campari": (5.0, 0.0, 9.0, 3.0, 4.0, 1.0),
    "lime_juice": (0.0, 9.0, 0.0, 0.0, 0.0, 8.0),
    "simple_syrup": (10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "soda_water": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

# Desired blend: a bittersweet, lightly herbal, gently boozy spritz.
# Deliberately not exactly reachable, so the optimum is interior-ish.
TARGET = (4.0, 1.0, 4.5, 3.5, 3.5, 1.0)

# The simplex lives in one higher dimension than the cube we optimise over.
N_DIM = len(INGREDIENTS) - 1  # = 5


def _blend_flavour(proportions):
    """Perceived flavour of a blend: proportion-weighted mean of profiles."""
    return [
        sum(proportions[i] * FLAVOUR[name][a] for i, name in enumerate(INGREDIENTS))
        for a in range(len(AXES))
    ]


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^5` -> RMS flavour error.

    Lifts the cube point onto the 6-component simplex of ingredient
    proportions, then scores the blend against TARGET."""
    proportions = cube_to_simplex(u)  # 6 non-negative weights summing to 1
    flavour = _blend_flavour(proportions)
    sq = sum((flavour[a] - TARGET[a]) ** 2 for a in range(len(AXES)))
    return math.sqrt(sq / len(AXES))


def decode(u):
    """Convenience: physical recipe + achieved flavour for a `[0,1]^5` point."""
    proportions = cube_to_simplex(u)
    flavour = _blend_flavour(proportions)
    return {
        "recipe": {name: proportions[i] for i, name in enumerate(INGREDIENTS)},
        "flavour": dict(zip(AXES, flavour)),
        "rms_error": objective(u),
    }
