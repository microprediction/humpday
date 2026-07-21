"""Equivalence proofs for the online (generator) conversion pilot.

The converted DifferentialEvolution and NelderMead must reproduce the
frozen pre-conversion implementations' trajectories exactly: same points
in the same order with the same values, across seeds, dimensions,
objectives and budgets. The native ask/tell drive must match optimize()
just as exactly, and close() must abandon a run without threads or hangs.
"""

import math
import random

import pytest

from humpday.optimizers.alloy import Alloy
from humpday.optimizers.evolutionary_algorithms import (
    AntColonyOpt,
    BayesianOpt,
    CMAEvolutionStrategy,
    DifferentialEvolution,
    EvolutionStrategy,
    FireflyAlgorithm,
    GeneticAlgorithm,
    HarmonySearch,
    HillClimbing,
    ParticleSwarm,
    RandomSearch,
    SimulatedAnnealing,
)
from humpday.optimizers.scipy_algorithms import LBFGSB, NelderMead, Powell
from humpday.optimizers.search_algorithms import (
    CoordinateDescent,
    GridSearch,
    PatternSearch,
    Rechenberg,
)

from .reference_impls_pre_online import (
    FrozenAlloy,
    FrozenAntColonyOpt,
    FrozenBayesianOpt,
    FrozenCMAEvolutionStrategy,
    FrozenCoordinateDescent,
    FrozenDifferentialEvolution,
    FrozenEvolutionStrategy,
    FrozenFireflyAlgorithm,
    FrozenGeneticAlgorithm,
    FrozenGridSearch,
    FrozenHarmonySearch,
    FrozenHillClimbing,
    FrozenLBFGSB,
    FrozenNelderMead,
    FrozenParticleSwarm,
    FrozenPatternSearch,
    FrozenPowell,
    FrozenRandomSearch,
    FrozenRechenberg,
    FrozenSimulatedAnnealing,
)

try:
    import numpy as np
except ImportError:  # pure backend
    np = None


def _seed(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def sphere(x):
    return sum((float(v) - 0.3) ** 2 for v in x)


def rastriginish(x):
    return sum(
        (float(v) - 0.7) ** 2 - 0.1 * math.cos(6 * math.pi * (float(v) - 0.7))
        for v in x
    )


OBJECTIVES = [sphere, rastriginish]
CASES = [(0, 2, 60), (1, 2, 200), (2, 5, 120), (3, 5, 200)]
PAIRS = [
    (FrozenDifferentialEvolution, DifferentialEvolution),
    (FrozenNelderMead, NelderMead),
    # Batch 1
    (FrozenRechenberg, Rechenberg),
    (FrozenCoordinateDescent, CoordinateDescent),
    (FrozenPatternSearch, PatternSearch),
    (FrozenGridSearch, GridSearch),
    (FrozenRandomSearch, RandomSearch),
    (FrozenEvolutionStrategy, EvolutionStrategy),
    (FrozenHillClimbing, HillClimbing),
    (FrozenHarmonySearch, HarmonySearch),
    # Batch 2
    (FrozenParticleSwarm, ParticleSwarm),
    (FrozenSimulatedAnnealing, SimulatedAnnealing),
    (FrozenGeneticAlgorithm, GeneticAlgorithm),
    (FrozenFireflyAlgorithm, FireflyAlgorithm),
    (FrozenAntColonyOpt, AntColonyOpt),
    # Batch 3
    (FrozenBayesianOpt, BayesianOpt),
    (FrozenCMAEvolutionStrategy, CMAEvolutionStrategy),
    (FrozenPowell, Powell),
    (FrozenLBFGSB, LBFGSB),
    (FrozenAlloy, Alloy),
]


def _traj_optimize(cls, objective, seed, n_dim, n_trials):
    _seed(seed)
    log = []

    def obj(x):
        v = float(objective(x))
        log.append((tuple(float(c) for c in x), v))
        return v

    opt = cls(obj, n_trials, n_dim)
    best_value, _ = opt.optimize()
    return log, float(best_value), opt.evaluations


def _traj_asktell_scalar(cls, objective, seed, n_dim, n_trials):
    _seed(seed)
    log = []
    opt = cls(objective, n_trials, n_dim)
    while True:
        x = opt.suggest_next()
        if x is None:
            break
        v = float(objective(x))
        log.append((tuple(float(c) for c in x), v))
        opt.receive_update(v)
    return log, float(opt.best_value), opt.evaluations


def _traj_asktell_batch(cls, objective, seed, n_dim, n_trials):
    _seed(seed)
    log = []
    opt = cls(objective, n_trials, n_dim)
    while True:
        xs = opt.suggest_batch()
        if xs is None:
            break
        vs = []
        for x in xs:
            v = float(objective(x))
            log.append((tuple(float(c) for c in x), v))
            vs.append(v)
        opt.tell_batch(vs)
    return log, float(opt.best_value), opt.evaluations


@pytest.mark.parametrize("frozen_cls,new_cls", PAIRS)
@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("seed,n_dim,n_trials", CASES)
def test_trajectory_identical_to_frozen(
    frozen_cls, new_cls, objective, seed, n_dim, n_trials
):
    ref = _traj_optimize(frozen_cls, objective, seed, n_dim, n_trials)
    new = _traj_optimize(new_cls, objective, seed, n_dim, n_trials)
    assert new == ref


@pytest.mark.parametrize("frozen_cls,new_cls", PAIRS)
@pytest.mark.parametrize("seed,n_dim,n_trials", CASES[:2])
def test_native_asktell_scalar_matches_optimize(
    frozen_cls, new_cls, seed, n_dim, n_trials
):
    ref = _traj_optimize(new_cls, sphere, seed, n_dim, n_trials)
    driven = _traj_asktell_scalar(new_cls, sphere, seed, n_dim, n_trials)
    assert driven == ref


@pytest.mark.parametrize("frozen_cls,new_cls", PAIRS)
def test_native_asktell_batch_matches_optimize(frozen_cls, new_cls):
    ref = _traj_optimize(new_cls, sphere, 0, 2, 60)
    driven = _traj_asktell_batch(new_cls, sphere, 0, 2, 60)
    assert driven == ref


@pytest.mark.parametrize("frozen_cls,new_cls", PAIRS)
def test_native_asktell_uses_no_thread(frozen_cls, new_cls):
    _seed(0)
    opt = new_cls(sphere, 50, 2)
    x = opt.suggest_next()
    assert x is not None
    assert opt._at is None  # never fell back to the worker-thread shim
    opt.receive_update(sphere(x))
    opt.close()
    assert opt.is_done()


@pytest.mark.parametrize("frozen_cls,new_cls", PAIRS)
def test_close_midway_is_clean(frozen_cls, new_cls):
    _seed(1)
    opt = new_cls(sphere, 200, 3)
    for _ in range(7):
        x = opt.suggest_next()
        opt.receive_update(sphere(x))
    opt.close()
    assert opt.is_done()
    assert opt.suggest_next() is None
