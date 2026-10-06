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
problem dimension, each evaluation budget, and each of two kinds of objective: a suite of classic
analytic surfaces regenerated with fresh random parameters on every run, and a set of engineering
and physics problems drawn from worked applications -- a brachistochrone descent, a truss, a
cantilever beam, a cart-pole control policy -- each posed at the dimension its problem actually has.
Every problem is presented on the unit hypercube, so the caller's domain is transformed rather than
the optimizer's.

# Statement of need

A practitioner facing a black-box objective has many implementations and little basis for choosing
between them. Published benchmark suites are authoritative, but answer at the granularity of a
research comparison rather than the point of use: *given twelve variables and two hundred function
evaluations, what should I call?* `humpday` answers that from recorded evidence, and the recording
turned out to matter more than the answer. Three properties of the tournament decide whether its
output means anything.

The evaluation budget belongs in the index, not in the fine print. NEWUOA and BOBYQA build an
interpolation set of roughly $2n+1$ points before proposing anything, which is 201 points at
$n=100$; UOBYQA requires a full quadratic set of $(n+1)(n+2)/2$ points, which is 5,151. Given a flat hundred evaluations these methods return almost immediately having
built no usable model, and on a leaderboard recorded that way they placed first, second and third,
rewarded precisely for doing nothing. The tournament is therefore recorded at four budgets from 50
to 5,000, and a query is answered from the recorded budget nearest below the caller's -- or, for a
caller more frugal than the smallest budget recorded, from that smallest cell rather than not at
all, since there is no evidence below 50 evaluations and the alternative is silence.

A fixed objective can be learned. Analytic instances are therefore regenerated with a fresh shift,
rotation, scale, noise level and modal frequency each run, so repeated tournaments are new evidence
rather than the same match replayed. The engineering problems are fixed by
construction, since a truss is a truss; for those, repetition varies only the optimizer's own
randomization, and generalization rests on the number and variety of problems rather than on
resampling any one of them.

The worked applications are distributed by what someone wrote up rather than by dimension --
fifteen at four variables, one at sixteen, none above sixty -- so where fewer than six exist, the
suite is topped up with three scalable stand-ins: a circle packing, a battery dispatch schedule and
a discretised descent. Seven of the twelve recorded dimensions draw on them, and at fifty and a
hundred variables the engineering suite is made of them entirely. Three stand-ins are not seventy-six worked problems, so a rating from the high-dimensional
engineering cells is better evidence than analytic surfaces alone, and weaker than the demos.

The choice of objective suite substantially determines the ranking, and the mechanism is not that
synthetic benchmarks are harder or easier than real objectives but that they test one narrow
character: globally deceptive, so a naive sampler struggles to find the right basin, yet locally
smooth once found, which is exactly what rewards a method that fits and trusts a local quadratic
model. This is established rather than incidental: @cotton2026benchmark ranks a panel of optimizers
on a memorisation-proof suite of real-world objectives -- rugged, coupled, noisy or multi-scale
rather than smooth -- and finds the rank correlation with their synthetic-benchmark ranking
statistically indistinguishable from zero, with model-based trust-region methods the ones the
synthetic ranking systematically over-trusts. The recorded tournaments reproduce the pattern where
the methods can build a model at all. Below about twelve variables the three trust-region methods
dominate the morphed surfaces, sweeping the top three places in 17 of the 48 recorded surface cells;
above it they fall away, and at fifty and a hundred variables none is rated at any budget, the
interpolation set they need exceeding what they are given. `PRIMA_UOBYQA` averages rank 3.7 across
the surface cells where it is rated and 9.3 on the engineering cells at the same dimensions -- still
worse there, but the gap a smooth analytic suite alone would report is the wrong one to trust.

`humpday` therefore records both suites and, by default, ranks by *worst* position across them, so
the recommendation is never-terrible rather than a suite-winner that collapses on the other. The
rank taken from a cell is first shrunk toward the optimizer's cross-suite mean in proportion to how
many problems back that cell, so a thin tournament moves the ordering less than a deep one, though a
caller who knows their objective's character can request either suite directly. Ratings are never
interpolated between dimensions: performance is not smooth in dimension, and a dimension nothing has
raced falls back to a documented rule rather than borrowing a neighbour's.

# State of the field

`scipy.optimize` [@scipy] is the default destination, supplying Nelder-Mead, Powell and
differential evolution but leaving the choice to the user. `nlopt` and the PRIMA reference
implementations [@prima] provide derivative-free methods without addressing selection, and `Optuna`
[@optuna] and similar frameworks tune hyperparameters with their own samplers -- a related problem
approached from the machine-learning side.

The benchmarking side is well served by COCO/BBOB [@hansen2021coco], the standard for comparing
continuous optimizers and far exceeding `humpday` in rigour and breadth of analysis. It does not,
because it is not its purpose, ship the optimizers and a recommendation alongside the benchmark,
dependency-free and re-runnable, so `humpday` is a practitioner-facing complement rather than a
replacement.

# Software design

Optimizers share one `_run()` generator protocol driven through `suggest_next`/`receive_update`,
which makes a tournament possible without per-optimizer adapters. Everything is written against a
small array shim with `numpy` and pure-Python backends chosen at import, calling the same functions
either way. That is the package's central trade-off: reimplementing well-tested algorithms in pure
Python costs accuracy risk and speed, and buys installability anywhere, a browser port, and the
ability to test the recursions against golden transition vectors.

Every run is capped in wall-clock time, at the largest of twenty-five times what a pure-sampling
reference costs on that problem, twenty milliseconds per evaluation of budget, a five-second floor,
and four times the median wall-clock of every run already completed in that cell, then at a
three-minute ceiling. Without a cap the grid does not terminate; with a cap fixed in absolute
seconds it would disqualify a method for the cost of the objective rather than its own. The
convergence-tracking term exists because the reference alone was once not enough: an objective whose
evaluation cost depends on location gives an optimizer converging into an expensive basin a cost the
reference sampler never visited: it once overran eighteen of twenty-three optimizers in a single
cell against a reference that finished in under a second -- one mis-calibrated allowance, not
eighteen overhead problems. Tracking the slowest completed run fixed it: none of the 96 shipped
cells now disqualifies a third of its contenders, the threshold above which timeouts are recorded
but not acted on. An optimizer that twice fails to return within its allowance is recorded
as timed out rather than as having lost, and ranks last in that cell. Because every optimizer here is
a pure Python port, a timeout is a statement about this implementation at that size rather than about
the algorithm: at a hundred variables BOBYQA's interpolation algebra dominates in a way it does not
in the Fortran reference -- the right answer for a caller choosing what to run from this package, the
wrong one to cite about the method.

Recorded ratings ship inside the package as data, so a recommendation needs no network and no
benchmark run. The engineering demos are the exception: each is a directory with a README and a
runnable script as much as an objective, so they stay a repository asset, and
`physics_objectives()` returns an empty list off-repository rather than raising from site-packages.

# Research impact statement

`humpday` generates the optimizer comparisons underlying `papers/dfo_recommender`, which develops
the cost-weighted Borda recommender `eligibility.recommend` implements, and `papers/planar_search`.
It is published on PyPI, its JavaScript port runs the optimizers in-browser for the worked
applications, and the recorded tournaments are committed so results can be reproduced or contested.

Its credible near-term significance is instrumentation rather than a new method: what it adds to
@cotton2026benchmark is that the comparison is recorded per dimension, shipped as data, and
re-runnable on a different suite by anyone who doubts it.

# Quality control

Continuous integration runs three jobs: the full suite of 1,144 tests on Python 3.11; a
portable-contract job running the transition-vector, PRNG-parity and portable-mode tests on 3.11 and
3.13 with the pure-Python backend forced, which tests the no-dependency claim rather than asserting
it; and a minimal-dependency job checking the package imports and functions with nothing beyond the
standard library.

Thirteen of the twenty-three optimizers have JavaScript twins verified bit-exact at statement level,
replayed against committed transition vectors and compared as IEEE-754 patterns; the remaining ten
get a coarser end-to-end check with known divergent cases, and that distinction is recorded rather
than blurred. Objective registries are themselves screened: one that accepts a vector of any length
and reads only its leading coordinates is a fixed-dimension problem in disguise that would quietly
corrupt a high-dimensional tournament.

# AI usage disclosure

Generative AI (Claude Code) was used substantially in this work: porting and fixing optimizers,
building the tournament and objective registries, writing tests, and drafting this paper. Every
quantitative claim here is reproducible from the committed scripts and recorded ratings, and was
checked against the repository rather than against the generating model's expectations. Successive
reviews found several that did not hold -- including two traced to real defects in the package --
and those are corrected here. Responsibility for the content rests with the author.

# Acknowledgements

The optimizer ports follow the published descriptions and reference implementations of their
respective authors; `pdfo` and `mealpy` were used as references when validating them.

# References
