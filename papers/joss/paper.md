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
problem dimension, each evaluation budget, and each of two kinds of objective. The first is a suite
of classic analytic surfaces,
regenerated with fresh random parameters on every run. The second is a set of engineering and
physics problems drawn from worked applications: a brachistochrone descent, a truss, a cantilever
beam, a cart-pole control policy, each posed at the dimension its problem actually has. Every
problem is presented on the unit hypercube, so the caller's domain is transformed rather than the
optimizer's.

# Statement of need

A practitioner facing a black-box objective has many implementations and little basis for choosing
between them. Published benchmark suites are authoritative, but answer at the granularity of a
research comparison rather than the point of use: *given twelve variables and two hundred function
evaluations, what should I call?*

`humpday` answers that from recorded evidence, and the recording turned out to matter more than the
answer. Three properties of the tournament decide whether its output means anything.

The evaluation budget belongs in the index, not in the fine print. NEWUOA and BOBYQA build an
interpolation set of roughly $2n+1$ points before proposing anything, which is 201 points at
$n=100$; UOBYQA requires a full quadratic set of $(n+1)(n+2)/2$ points, which is 5,151. Given a flat hundred evaluations these methods return almost immediately having
built no usable model, and on a leaderboard recorded that way they placed first, second and third,
rewarded precisely for doing nothing. The tournament is therefore recorded at four budgets from 50 to 5,000 and a query is
answered from the recorded budget nearest below the caller's. The exception is a caller more
frugal than the smallest budget recorded, who is answered from that smallest cell rather than not
at all; there is no evidence below 50 evaluations to give them, and the alternative is silence.

A fixed objective can be learned. Analytic instances are therefore regenerated with a fresh shift,
rotation, scale, noise level and modal frequency each run, so repeated tournaments are new evidence
rather than the same match replayed. The engineering problems are fixed by
construction, since a truss is a truss; for those, repetition varies only the optimizer's own
randomization, and generalization rests on the number and variety of problems rather than on
resampling any one of them.

The worked applications are distributed by what someone wrote up rather than by dimension --
fifteen at four variables, one at sixteen, none above sixty -- so where fewer than six exist the
suite is topped up with three scalable stand-ins: a circle packing, a battery dispatch schedule and
a discretised descent. Seven of the twelve recorded dimensions draw on them, and at fifty and a
hundred variables the engineering suite is made of them entirely. Raced at twelve variables against
the real demos they sit between those demos and the analytic surfaces. Three structures do not
stand in for seventy-six worked problems, and a rating from the high-dimensional engineering cells
is better evidence than analytic surfaces alone and weaker than the demos.

The choice of objective suite substantially determines the ranking. This is established rather
than incidental: @cotton2026benchmark ranks a panel of optimizers on a memorisation-proof suite of
real-world objectives and finds the rank correlation with their synthetic-benchmark ranking
statistically indistinguishable from zero, with the mechanism being that benchmarks over-trust the
model-based trust-region methods. The recorded tournaments reproduce it where the methods can build
a model at all. Below about twelve variables the three trust-region methods dominate the morphed
surfaces, sweeping the top three places in 13 of the 48 recorded surface cells; above it they fall
away, and at fifty and a hundred variables none is rated at any budget, the interpolation set they
need exceeding what they are given. `PRIMA_UOBYQA` averages rank 4.9 across the surface cells where
it is rated and 14.4 on the engineering cells at the same dimensions.

`humpday` therefore records both suites and, by default, ranks by *worst* position across them, so
the recommendation is the optimizer that is never terrible rather than one that wins a suite and
collapses on the other. The rank taken from a cell is first shrunk toward the optimizer's
cross-suite mean in proportion to how many problems back that cell, so a thin tournament moves the
ordering less than a deep one. A caller who knows the character of their objective can request
either suite. Ratings are never interpolated between dimensions: optimizer performance is not
smooth in dimension, and a dimension nothing has raced falls back to a documented rule and reports
no ratings rather than borrowing a neighbour's.

# State of the field

`scipy.optimize` [@scipy] is the default destination, supplying Nelder–Mead, Powell and
differential evolution among others, but leaves the choice to the user. `nlopt` and the PRIMA reference
implementations [@prima] provide derivative-free methods without addressing selection. `Optuna` [@optuna] and similar frameworks tune hyperparameters with their own
samplers, a related problem approached from the machine-learning side.

The benchmarking side is well served by COCO/BBOB [@hansen2021coco], which is the standard for
comparing continuous optimizers and far exceeds `humpday` in rigour and breadth of analysis. What COCO does not do, because it is not its
purpose, is ship the optimizers and a recommendation alongside the benchmark, dependency-free, so
the comparison can be re-run and the answer consumed programmatically. `humpday` is a practitioner-facing complement, not a
replacement.

# Software design

Optimizers share one `_run()` generator protocol driven through `suggest_next`/`receive_update`,
which is what makes a tournament possible without per-optimizer adapters. Everything is written
against a small array shim with two backends, `numpy` and pure Python, chosen at import; the
algorithm code calls the same functions either way. That is the central trade-off of the package:
reimplementing well-tested algorithms in pure Python costs accuracy risk and speed, and buys
installability anywhere, a browser port, and the ability to test the recursions against golden
transition vectors.

Every run is capped in wall-clock time, at the largest of twenty-five times what the objective's
evaluations cost, twenty milliseconds per evaluation of budget, and a five-second floor, then at a
three-minute ceiling. Across the 96 cells the per-evaluation term bound 48 times and the floor 47;
the term that adapts to the objective never did. Without a cap
the grid does not terminate; with a cap fixed in absolute seconds it would disqualify a method for
the cost of the objective rather than its own, which is why the allowance is measured per cell
against a pure-sampling reference. An optimizer that twice fails to return is recorded as having
timed out rather than as having lost, and ranks last in that cell. The calibration has a known
limit: where an objective's evaluation cost depends on location, an optimizer that converges into
an expensive region pays a cost the sampler never saw. One cell of ninety-six shows it, with
eighteen of twenty-three optimizers overrunning against a reference that finished in under a second,
and where more than a third of a cell is disqualified the timeouts are recorded but not acted on. Because every optimizer here is
a pure Python port, a timeout is a statement about this implementation at that size rather than
about the algorithm: at a hundred variables BOBYQA's interpolation algebra dominates in a way it
does not in the Fortran reference. That is the right answer for a caller choosing what to run from
this package and the wrong one to cite about the method.

Recorded ratings ship inside the package as data, so a recommendation needs no network and no
benchmark run. The engineering demos are the exception: each is a directory with a README and a
runnable script as much as an objective, so they stay a repository asset, and
`physics_objectives()` returns an empty list off-repository rather than raising from site-packages.

# Research impact statement

`humpday` is used to generate the optimizer comparisons underlying `papers/dfo_recommender`, which
develops the cost-weighted Borda recommender that `eligibility.recommend` implements, and
`papers/planar_search`. It is published on PyPI, the JavaScript port runs the optimizers in the
browser for the worked applications, and the recorded tournaments are committed so results can be
reproduced or contested.

Its credible near-term significance is instrumentation rather than a new method. The suite
dependence above is a caution that applies to any optimizer comparison, and what the package adds to
@cotton2026benchmark is that the comparison is now recorded per dimension, shipped as data, and
re-runnable on a different suite by anyone who doubts it.

# Quality control

Continuous integration runs three jobs: the full suite of 1,135 tests on Python 3.11; a
portable-contract job running the transition-vector, PRNG-parity and portable-mode tests on 3.11 and
3.13 with the pure-Python backend forced, which tests the no-dependency claim rather than asserting
it; and a minimal-dependency job checking the package imports and functions with nothing beyond the
standard library.

Thirteen of the twenty-three optimizers have JavaScript twins verified as bit-exact at statement
level, replayed against committed transition vectors and compared as IEEE-754 patterns. The
remaining ten are covered by a coarser end-to-end check with known divergent cases, and that
distinction is recorded rather than blurred. Objective registries are themselves screened: one that
accepts a vector of any length and reads only its leading coordinates is a fixed-dimension problem
in disguise, and would quietly corrupt a high-dimensional tournament.

# AI usage disclosure

Generative AI (Claude Code) was used substantially in this work: in porting and fixing optimizers,
building the tournament and objective registries, writing tests, and drafting this paper. Every
quantitative claim here is reproducible from the committed scripts and recorded ratings. Claims were
checked against the repository rather than against the generating model's expectations. Successive
reviews found several that did not hold -- a misstated interpolation-point count, an overstated
claim about which methods lead the analytic surfaces, and a test total off by a hundred -- and an
audit of the package found defects behind two of them. Those are corrected here. Responsibility for the content rests with the author.

# Acknowledgements

The optimizer ports follow the published descriptions and reference implementations of their
respective authors; `pdfo` and `mealpy` were used as references when validating them.

# References
