"""
Least-cost diet (Stigler's problem).

Choose quantities of eight foods to meet five nutrient requirements at minimum cost.
Each food has a price and a nutrient profile; falling short of any requirement is
penalised. The cheap foods rarely cover every nutrient on their own, so the optimum mixes
a few complementary ones, the classic constrained linear-diet problem.

The HumpDay objective takes an 8-D point in [0,1]^8 (food quantities, mapped to 0..5
servings) and returns food cost plus a penalty for unmet nutrient requirements.
"""

from __future__ import annotations

N_FOODS = 8
N_DIM = N_FOODS
N_NUTRIENTS = 5
MAX_QTY = 5.0
PENALTY = 10.0

COST = (2.0, 1.5, 3.0, 1.0, 2.5, 1.8, 1.2, 2.2)
# CONTENT[n][i] = amount of nutrient n in one serving of food i
CONTENT = (
    (8, 2, 5, 1, 9, 3, 2, 6),  # calories (x100)
    (4, 1, 9, 0, 2, 7, 1, 3),  # protein
    (1, 6, 2, 3, 0, 2, 8, 1),  # carbs
    (0, 3, 1, 5, 1, 1, 2, 4),  # vitamins
    (2, 0, 3, 1, 6, 4, 0, 2),  # minerals
)
REQ = (30.0, 22.0, 20.0, 14.0, 16.0)


def decode(u):
    return [MAX_QTY * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    q = decode(u)
    cost = sum(COST[i] * q[i] for i in range(N_FOODS))
    for n in range(N_NUTRIENTS):
        intake = sum(CONTENT[n][i] * q[i] for i in range(N_FOODS))
        cost += PENALTY * max(0.0, REQ[n] - intake) ** 2
    return cost
