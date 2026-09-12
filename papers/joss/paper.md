---
title: 'humpday: comparing and recommending derivative-free optimizers in pure Python'
tags:
  - Python
  - derivative-free optimization
  - black-box optimization
  - benchmarking
  - Elo rating
authors:
  - name: Peter Cotton
    orcid: 0000-0003-1832-2924
    affiliation: 1
affiliations:
  - name: Microprediction, LLC
    index: 1
date: 12 September 2026
bibliography: paper.bib
---

# Summary

`humpday` implements twenty-three derivative-free optimizers behind one calling convention, races
them against each other, and recommends one for a problem of a given dimension and evaluation
budget. Every optimizer is a pure Python implementation with **no hard dependencies at all**:
`numpy` is used through a shim when it is present and the same code runs without it, which makes the
package installable and runnable in restricted environments and portable to a browser.

The comparison machinery is the substance of the package rather than an accessory to it. Optimizers
are rated by Elo over tournaments of paired outcomes, recorded separately for each problem dimension
and for two kinds of objective: randomly morphed analytic surfaces, and a suite of seventy-six
engineering and physics problems drawn from applications such as a brachistochrone descent, a truss,
a cantilever beam, and a cart-pole control policy. Each problem is posed on the unit hypercube, so
the caller's domain is transformed rather than the optimizer's.

# Statement of need

A practitioner facing a black-box objective has many implementations available — `scipy.optimize`
[@scipy], `nlopt`, `pdfo`, `Optuna` [@optuna], and the derivative-free trust-region methods of the
PRIMA family [@prima] among them — and little basis for choosing between them. Published benchmarks
such as COCO/BBOB [@hansen2021coco] are authoritative and thorough, but they answer the question at
the granularity of a research comparison rather than at the point of use: *given twelve variables
and two hundred function evaluations, what should I call?*

`humpday` answers that question from recorded evidence, and the recording turned out to matter more
than the answer. Three properties of the tournament decide whether its output means anything, and
each is easy to get wrong.

The evaluation budget must scale with dimension. The trust-region methods construct an
interpolation set of roughly $2n+1$ points before proposing anything, which is 201 points at
$n=100$. Given a flat hundred evaluations they return almost immediately having built no model, and
on a leaderboard recorded that way they placed first, second and third — rewarded precisely for
doing nothing.

A fixed objective can be learned. Objectives are regenerated with a fresh random shift,
rotation, scale, noise level and modal frequency for every run, so repeated tournaments are new
evidence rather than the same match replayed.

Least expected, the choice of objective suite substantially determines the ranking.
Analytic surfaces are smooth, and smoothness rewards local search: on morphed surfaces the three
trust-region methods swept the top three places at every dimension recorded. Raced on the
engineering problems at the same dimensions they are displaced, and `PRIMA_UOBYQA` — third on
surfaces — falls to sixteenth. A ranking taken from analytic surfaces alone substantially
measures how smooth those surfaces are.

`humpday` therefore records both suites and, by default, ranks by *worst* position across them, so
the recommendation is the optimizer that is never terrible rather than one that wins a suite and
collapses on the other. A caller who knows the character of their objective can ask for either suite
explicitly. Ratings are never interpolated between dimensions: optimizer performance is not smooth
in dimension, and a dimension nothing has raced falls back to a documented rule and reports no
ratings rather than borrowing a neighbour's.

# Quality control

One thousand and thirteen tests run in continuous integration across Python 3.9 through 3.13, and
separately against the pure-Python backend with `numpy` absent, so the no-dependency claim is tested
rather than asserted. Twelve of the optimizers are additionally ported to JavaScript and checked for
cross-language agreement against golden transition vectors, which pins the recursions far more
tightly than an end-to-end tolerance would. The objective registries are themselves screened by
tests: an objective that accepts a vector of any length and then reads only its leading coordinates
is a fixed-dimension problem in disguise, and would quietly corrupt a high-dimensional tournament.

# Acknowledgements

The optimizer ports follow the published descriptions and reference implementations of their
respective authors; `pdfo` and `mealpy` were used as references when validating them.

# References
