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
budget. Every optimizer is a pure Python implementation and the package declares no dependencies at
all: `numpy` is used through a shim when present, and the same code runs without it.

Optimizers are rated by Elo over tournaments of paired outcomes, recorded separately for each
problem dimension and for two kinds of objective. The first is a suite of classic analytic surfaces,
regenerated with fresh random parameters on every run. The second is a set of engineering and
physics problems drawn from worked applications: a brachistochrone descent, a truss, a cantilever
beam, a cart-pole control policy, each posed at the dimension its problem actually has. Every
problem is presented on the unit hypercube, so the caller's domain is transformed rather than the
optimizer's.

# Statement of need

A practitioner facing a black-box objective has many implementations available and little basis for
choosing between them. Published benchmark suites are authoritative, but they answer at the
granularity of a research comparison rather than at the point of use: *given twelve variables and
two hundred function evaluations, what should I call?*

`humpday` answers that from recorded evidence, and the recording turned out to matter more than the
answer. Three properties of the tournament decide whether its output means anything, and each is
easy to get wrong.

The evaluation budget must scale with dimension. NEWUOA and BOBYQA build an interpolation set of
roughly $2n+1$ points before proposing anything, which is 201 points at $n=100$; UOBYQA requires a
full quadratic set of $(n+1)(n+2)/2$ points, which is 5,151. Given a flat hundred evaluations these
methods return almost immediately having built no usable model. On a leaderboard recorded that way
they placed first, second and third, rewarded precisely for doing nothing.

A fixed objective can be learned. Instances in the analytic suite are therefore regenerated with a
fresh shift, rotation, scale, noise level and modal frequency on every run, so repeated tournaments
there are new evidence rather than the same match replayed. The engineering problems are fixed by
construction, since a truss is a truss; for those, repetition varies only the optimizer's own
randomization, and generalization rests on the number and variety of problems rather than on
resampling any one of them.

Least expected, the choice of objective suite substantially determines the ranking. Analytic
surfaces are smooth, and smoothness rewards local search: on the morphed surfaces the three
trust-region methods took the top three places at every dimension recorded. Raced on the
engineering problems at the same dimensions they are displaced. `PRIMA_UOBYQA`, third on surfaces
averaged over ten dimensions, falls to sixteenth. A ranking taken from analytic surfaces
alone substantially measures how smooth those surfaces are.

`humpday` therefore records both suites and, by default, ranks by *worst* position across them, so
the recommendation is the optimizer that is never terrible rather than one that wins a suite and
collapses on the other. A caller who knows the character of their objective can request either
suite. Ratings are never interpolated between dimensions: optimizer performance is not smooth in
dimension, and a dimension nothing has raced falls back to a documented rule and reports no ratings
rather than borrowing a neighbour's.

# State of the field

`scipy.optimize` [@scipy] is the default destination and supplies Nelder–Mead, Powell,
differential evolution and more, but leaves the choice among them to the user. `nlopt` and the
PRIMA reference implementations [@prima] provide high-quality derivative-free methods without
addressing selection. `Optuna` [@optuna] and similar frameworks tune hyperparameters with their own
samplers, a related problem approached from the machine-learning side.

The benchmarking side is well served by COCO/BBOB [@hansen2021coco], which is the standard for
comparing continuous optimizers and far exceeds `humpday` in rigour and breadth of analysis. What
COCO does not do, because it is not its purpose, is ship the optimizers and a recommendation
alongside the benchmark, in a package with no dependencies, so that the comparison can be re-run and
the answer consumed programmatically. `humpday` is a practitioner-facing complement, not a
replacement.

# Software design

Optimizers share a single `_run()` generator protocol driven through `suggest_next`/`receive_update`,
which is what makes a tournament possible without per-optimizer adapters. Everything is written
against a small array shim with two backends, `numpy` and pure Python, chosen at import; the
algorithm code calls the same functions either way. That is the central trade-off of the package:
reimplementing well-tested algorithms in pure Python costs accuracy risk and speed, and buys
installability anywhere, a browser port, and the ability to test the recursions against golden
transition vectors.

Recorded ratings ship inside the package as data, so a recommendation needs no network and no
benchmark run. The engineering demos are the exception and remain a repository asset rather than
package data: each is a directory with a README and a runnable script as much as an objective, and
they are not in the wheel, so `physics_objectives()` returns an empty list when the repository is
absent rather than raising from an installed package.

# Research impact statement

`humpday` is used to generate the optimizer comparisons underlying `papers/dfo_recommender`, which
develops the cost-weighted Borda recommender that `eligibility.recommend` implements, and
`papers/planar_search`. It is published on PyPI, the JavaScript port runs the optimizers in the
browser for the worked applications, and the recorded tournaments are committed so results can be
reproduced or contested.

The credible near-term significance is as instrumentation rather than as a new method. The finding
above, that a ranking taken from smooth analytic surfaces substantially measures smoothness, is a
caution that applies to any optimizer comparison, and the package makes it cheap to check on a
different suite.

# Quality control

Continuous integration runs three jobs. The full suite of 1,013 tests runs on Python 3.11. A
portable-contract job runs the transition-vector, PRNG-parity and portable-mode tests on Python 3.11
and 3.13 with the pure-Python backend forced, which is what tests the no-dependency claim rather
than asserting it. A minimal-dependency job checks that the package imports and functions with
nothing installed beyond the standard library.

Three optimizers (`RandomSearch`, `GridSearch` and `Rechenberg`) have JavaScript implementations
verified as bit-exact statement-level twins of the Python generators, replayed against committed
transition vectors and compared as IEEE-754 patterns. The remaining ports are covered by a coarser
end-to-end parity check with known divergent cases, and that distinction is recorded in the tests
rather than blurred. Objective registries are themselves screened: an objective that accepts a
vector of any length and then reads only its leading coordinates is a fixed-dimension problem in
disguise, and would quietly corrupt a high-dimensional tournament.

# AI usage disclosure

Generative AI (Claude Code) was used substantially in this work: in porting and fixing optimizers,
building the tournament and objective registries, writing tests, and drafting this paper. Every
quantitative claim here is reproducible from the committed scripts and recorded ratings. Claims were
checked against the repository rather than against the generating model's expectations, and review
of an earlier draft of this paper found several that did not hold: a misstated interpolation-point
count, an overstated morphing claim, and a cross-language coverage figure that conflated optimizers
with test vectors. Those are corrected here. Responsibility for the content rests with the author.

# Acknowledgements

The optimizer ports follow the published descriptions and reference implementations of their
respective authors; `pdfo` and `mealpy` were used as references when validating them.

# References
