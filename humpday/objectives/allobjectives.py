"""Legacy aggregate. Prefer ``humpday.objectives.SURFACES``.

Kept because ``papers/planar_search`` reports results computed over exactly this collection, and
changing it underneath them would silently change published numbers. It is not screened: several
members read only their leading coordinates whatever the dimension, and the portfolio objectives
run tens of milliseconds per evaluation. ``SURFACES`` is the curated, screened list.
"""

from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
from humpday.objectives.classic import CLASSIC_OBJECTIVES
from humpday.objectives.horse import HORSE_OBJECTIVES
from humpday.objectives.portfolio import PORTFOLIO_OBJECTIVES

OBJECTIVES = (
    CLASSIC_OBJECTIVES + PORTFOLIO_OBJECTIVES + HORSE_OBJECTIVES + CHATGPT_OBJECTIVES
)
