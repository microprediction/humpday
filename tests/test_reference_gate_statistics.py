"""The reference gate's own statistics, which had been measuring themselves (#409).

`tests/test_reference_alignment.py` decides whether every port matches its third-party
reference, so what it computes has to be worth believing. Three things were wrong with it:

  * the reference's starting point was drawn from `U[0.3, 0.7]` while twenty-seven of the
    thirty x0 sites in `humpday/optimizers/` draw from `U[0, 1]`, so the reference began in
    the middle of the cube, near the optimum, and the port began anywhere;
  * two references ran unconstrained on objectives that happen to be defined outside the
    cube, so they were solving an easier problem invisibly;
  * the headline `hd/ref` was a ratio of medians of four runs, and on a multimodal problem
    the outcome is bimodal -- found the funnel, or trapped on the ring -- so the median of
    four flips between the modes and the ratio with it. Shifting the seed block with the code
    untouched moved `CoordinateDescent/ackley` by a factor of 889 and `PatternSearch/ackley`
    by 460,000.

These tests hold the instrument still. They do not run the gate.
"""

import math

import pytest

from tests import test_reference_alignment as G


class TestTheStartingPoint:
    def test_x0_covers_the_whole_cube(self):
        """Not the middle 40% of it, which is where the optimum lives."""
        draws = [c for seed in range(400) for c in G._draw_x0(seed, 2)]
        assert min(draws) < 0.05
        assert max(draws) > 0.95

    def test_x0_matches_what_the_ports_actually_do(self):
        """`_A.random_uniform` is `U[0, 1]`, so `_draw_x0` has to be too.

        The old bounds were 0.3 and 0.7 under a comment claiming they were the same
        distribution the optimisers use. They were not, and the reference got a head start.
        """
        assert (G.X0_LO, G.X0_HI) == (0.0, 1.0)

    def test_x0_is_reproducible_for_a_seed(self):
        assert G._draw_x0(7, 3) == G._draw_x0(7, 3)
        assert G._draw_x0(7, 3) != G._draw_x0(8, 3)


class TestTheCubeGuard:
    def test_a_point_outside_the_cube_raises(self):
        counter = {"n": 0}
        f = G._in_cube(lambda x: sum(x), 2, counter)
        with pytest.raises(G.OutsideTheCube):
            f([1.4, 0.5])
        with pytest.raises(G.OutsideTheCube):
            f([0.5, -1e-9])

    def test_the_boundary_itself_is_allowed(self):
        counter = {"n": 0}
        f = G._in_cube(lambda x: sum(x), 2, counter)
        assert f([0.0, 1.0]) == 1.0

    def test_every_call_is_counted_including_the_one_that_raised(self):
        counter = {"n": 0}
        f = G._in_cube(lambda x: sum(x), 1, counter)
        f([0.5])
        with pytest.raises(G.OutsideTheCube):
            f([2.0])
        assert counter["n"] == 2

    def test_the_scipy_adapters_that_used_to_run_unbounded_now_pass_bounds(self):
        """Nelder-Mead and Powell reached (-0.98, 0.60) on Rosenbrock without them."""
        scipy_optimize = pytest.importorskip("scipy.optimize")
        assert scipy_optimize is not None
        for adapter in (G._ref_scipy_neldermead, G._ref_scipy_powell):
            # Would raise OutsideTheCube if the reference escaped.
            out = adapter(G.PROBLEMS["rosenbrock"]["func"], 200, 2, seed=0)
            assert math.isfinite(out["best_value"])


class TestHeadToHead:
    def test_losing_everything_is_one_and_winning_everything_is_zero(self):
        assert G.head_to_head([9.0, 9.0], [1.0, 1.0]) == 1.0
        assert G.head_to_head([1.0, 1.0], [9.0, 9.0]) == 0.0

    def test_two_ports_that_both_converged_exactly_tie(self):
        """The case `CONVERGED_GAP` exists to special-case for the ratio. Here it is free:
        ties count a half, so two runs that both reached the optimum score 0.5."""
        assert G.head_to_head([0.0] * 5, [0.0] * 5) == 0.5

    def test_it_compares_the_samples_rather_than_pairing_by_index(self):
        """Seed 3 is not the same problem for both sides -- they draw from different streams --
        so pairing by index would manufacture a difference. Reordering changes nothing."""
        hd, ref = [3.0, 1.0, 2.0], [2.5, 0.5, 1.5]
        assert G.head_to_head(hd, ref) == G.head_to_head(sorted(hd), sorted(ref))

    def test_it_is_bounded_however_far_apart_the_scales_are(self):
        """The property the ratio does not have. `Rechenberg/ackley` read 519,288 under the
        ratio and loses about half its pairings, which is to say it matches its reference."""
        assert G.head_to_head([1e300], [1e-300]) == 1.0
        assert 0.0 <= G.head_to_head([1e-300], [1e300]) <= 1.0

    def test_a_single_worse_run_among_many_moves_it_a_little(self):
        assert G.head_to_head([1.0, 1.0, 1.0, 9.0], [2.0] * 4) == 0.25


class TestTheGateIsWiredUp:
    def test_enough_runs_that_a_bimodal_median_is_not_a_coin_flip(self):
        assert G.N_RUNS >= 21

    def test_an_odd_number_of_runs_so_the_median_is_a_run(self):
        assert G.N_RUNS % 2 == 1

    def test_every_ceiling_names_a_pair_the_gate_knows_about(self):
        known = {
            (algorithm, problem) for algorithm in G.REFERENCES for problem in G.PROBLEMS
        }
        assert set(G.RATIO_CEILING) <= known
        assert set(G.WIN_CEILING) <= known

    def test_the_default_win_ceiling_is_above_a_coin_flip(self):
        """A port that matches its reference loses about half its pairings. A ceiling at or
        below 0.5 would fail every port that matches."""
        assert 0.5 < G.DEFAULT_WIN_CEILING < 1.0
        assert all(0.5 < v <= 1.0 for v in G.WIN_CEILING.values())


class TestTheConvergedFloorGovernsBothTests:
    """The win rate has no scale, so it needs the same floor the ratio has.

    Powell reaches 7.5e-11 on Ackley against scipy's bounded 4.8e-12 and loses 0.82 of
    head-to-head pairings. That is a real ordering -- humpday is consistently the second of
    the two -- and not a deficiency: both have solved the problem to ten decimal places. A
    gate that fails on it is reporting the last few digits, which is what `CONVERGED_GAP`
    exists to stop the ratio doing.
    """

    def test_the_floor_is_tight_enough_to_mean_solved(self):
        assert G.CONVERGED_GAP <= 1e-10

    def test_both_tests_are_guarded_by_it(self):
        import inspect

        src = inspect.getsource(G.test_reference_alignment)
        win_test, ratio_test = (
            "if lost > win_cap and hd_gap > CONVERGED_GAP:",
            "if lost > 0.5 and relative > ceiling and hd_gap > CONVERGED_GAP:",
        )
        assert win_test in src
        assert ratio_test in src
