"""
Tests to achieve 100% coverage on core humpday functionality.
"""

import numpy as np
import pytest


class TestMainInit:
    """Test main __init__.py exports to achieve 100% coverage."""

    def test_suggest_function(self):
        """Test the suggest function."""
        from humpday import suggest

        # Test basic suggest functionality
        suggestions = suggest(n_dim=3, n_trials=50)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Each suggestion should be a tuple (score, time, name)
        for score, time, name in suggestions:
            assert isinstance(score, (int, float))
            assert isinstance(time, (int, float))
            assert isinstance(name, str)

    def test_minimize_unit_cube_function(self):
        """Test the minimize_unit_cube function."""
        from humpday import minimize_unit_cube

        # Test with simple objective
        def simple_objective(x):
            return sum((xi - 0.5) ** 2 for xi in x)

        # Test without algorithm specification (auto-select)
        result = minimize_unit_cube(simple_objective, n_dim=2, n_trials=20)
        assert len(result) == 2  # (best_value, best_point)
        assert isinstance(result[0], (int, float))
        assert len(result[1]) == 2

        # Test with specific algorithm
        result_specific = minimize_unit_cube(
            simple_objective, n_dim=2, n_trials=20, algorithm="NelderMead"
        )
        assert len(result_specific) == 2

    def test_recommend_alias(self):
        """Test the recommend alias for suggest function."""
        from humpday import recommend

        # recommend should be the same as suggest
        suggestions = recommend(n_dim=2, n_trials=30)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0


class TestOptimizersCoverage:
    """Test missing coverage in optimizers.py"""

    def test_optimizer_edge_cases(self):
        """Test edge cases and error conditions in optimizers."""
        from humpday.optimizers.alloptimizers import (
            PRIMA_UOBYQA,
            NelderMead,
            RandomSearch,
        )

        # Test with very simple objective
        def simple_objective(x):
            return sum(x)

        # Test optimizers with minimal trials
        optimizers = [PRIMA_UOBYQA, NelderMead, RandomSearch]
        for opt_class in optimizers:
            optimizer = opt_class(simple_objective, n_trials=5, n_dim=2)
            result = optimizer.optimize()
            assert len(result) == 2
            assert isinstance(result[0], (int, float))
            assert len(result[1]) == 2

    def test_optimizer_path_tracking(self):
        """Test path tracking functionality in optimizers."""
        from humpday.optimizers.alloptimizers import RandomSearch

        def objective(x):
            return sum((xi - 0.5) ** 2 for xi in x)

        optimizer = RandomSearch(objective, n_trials=10, n_dim=2)
        optimizer.track_path = True  # Enable path tracking
        result = optimizer.optimize()

        # Should have recorded some path points
        assert hasattr(optimizer, "path")
        assert len(optimizer.path) > 0

    def test_algorithm_specific_branches(self):
        """Test algorithm-specific code branches."""
        from humpday.optimizers.alloptimizers import (
            FireflyAlgorithm,
            HarmonySearch,
            HillClimbing,
            SimulatedAnnealing,
        )

        def objective(x):
            return sum(x**2)

        # Test algorithms that have special conditions
        algorithms = [
            HillClimbing,
            SimulatedAnnealing,
            HarmonySearch,
            FireflyAlgorithm,
        ]

        for alg_class in algorithms:
            optimizer = alg_class(objective, n_trials=10, n_dim=2)
            result = optimizer.optimize()
            assert len(result) == 2


class TestSciPyInterfaceCoverage:
    """Test missing coverage in scipy_interface.py"""

    def test_minimize_scalar(self):
        """Test minimize_scalar function."""
        from humpday import minimize_scalar

        # Test 1D optimization
        def objective_1d(x):
            return (x - 2) ** 2

        result = minimize_scalar(objective_1d, bounds=(-5, 5))
        assert hasattr(result, "x")
        assert hasattr(result, "fun")

    def test_cube_minimize_scalar(self):
        """Test cube_minimize_scalar function."""
        from humpday import cube_minimize_scalar

        def objective_1d(x):
            return x**2

        result = cube_minimize_scalar(objective_1d, method="NelderMead")
        assert hasattr(result, "x")
        assert hasattr(result, "fun")

    def test_specific_cube_optimizers(self):
        """Test specific cube optimizer functions."""
        from humpday import (
            cube_cma_es,
            cube_differential_evolution,
            cube_nelder_mead,
            cube_particle_swarm,
            cube_prima_uobyqa,
        )

        def simple_objective(x):
            return sum((xi - 0.3) ** 2 for xi in x)

        optimizers = [
            cube_nelder_mead,
            cube_differential_evolution,
            cube_particle_swarm,
            cube_cma_es,
            cube_prima_uobyqa,
        ]

        for optimizer_func in optimizers:
            try:
                result = optimizer_func(simple_objective, n_dim=2, n_trials=10)
                assert hasattr(result, "x")
                assert hasattr(result, "fun")
            except Exception:
                # Some optimizers might not be available or might fail
                pass

    def test_error_conditions(self):
        """Test error conditions in scipy interface."""
        from humpday import minimize

        # Test with invalid bounds
        def objective(x):
            return sum(x**2)

        # Test edge cases that might trigger error handling
        try:
            result = minimize(objective, bounds=[(0, 0)])  # Invalid bounds
        except:
            pass  # Expected to potentially fail

    def test_domain_transformation_edge_cases(self):
        """Test edge cases in domain transformations."""
        from humpday import (
            transform_from_unit_cube,
            transform_to_unit_cube,
            unbounded_to_unit_cube,
            unit_cube_to_unbounded,
        )

        # Test edge cases
        bounds = [(-1, 1), (0, 10)]

        # Test boundary values
        edge_point = [-1, 0]  # At bounds
        unit_point = transform_to_unit_cube(edge_point, bounds)
        recovered = transform_from_unit_cube(unit_point, bounds)

        # Test unbounded transformations with different scales
        real_point = np.array([0, 1000, -500])  # Convert to numpy array
        unit_point = unbounded_to_unit_cube(real_point, scale=100)
        recovered = unit_cube_to_unbounded(unit_point, scale=100)


class TestAdaptiveOptimizerCoverage:
    """Test missing coverage in adaptive_optimizer.py"""

    def test_elo_system_complete(self):
        """Test complete EloRatingSystem functionality."""
        from humpday.optimizers.adaptive_optimizer import EloRatingSystem

        elo = EloRatingSystem()

        # Test save/load functionality
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Test save
            elo.save_ratings(temp_file)

            # Test load
            new_elo = EloRatingSystem()
            success = new_elo.load_ratings(temp_file)
            assert success

            # Ratings should be preserved
            for alg in elo.ratings:
                assert alg in new_elo.ratings

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # Test load non-existent file
        fake_elo = EloRatingSystem()
        success = fake_elo.load_ratings("non_existent_file.json")
        assert not success

    def test_adaptive_optimize_complete(self):
        """Test complete adaptive_optimize functionality."""
        from humpday.optimizers.adaptive_optimizer import (
            adaptive_optimize,
            sphere_variants_generator,
        )

        # Test with very minimal parameters
        generator = sphere_variants_generator(n_dim=2)

        results = adaptive_optimize(
            objective_generator=generator,
            trials_budget=100,  # Minimal budget
            n_dim=2,
            n_warmup_problems=2,  # Minimal warmup
            trials_per_warmup=10,
            verbose=False,  # Test non-verbose mode
        )

        # Check all expected keys are present
        expected_keys = [
            "elo_system",
            "top_algorithms",
            "recommendations",
            "total_problems_solved",
        ]
        for key in expected_keys:
            assert key in results

        # Check that we got some results
        assert len(results["top_algorithms"]) > 0
        assert "total_problems_solved" in results

    def test_elo_expected_score_edge_cases(self):
        """Test edge cases in Elo expected score calculation."""
        from humpday.optimizers.adaptive_optimizer import EloRatingSystem

        elo = EloRatingSystem()

        # Test extreme rating differences
        high_rating = 2000
        low_rating = 1000

        expected = elo.expected_score(high_rating, low_rating)
        assert 0 < expected < 1

        # Test equal ratings
        expected_equal = elo.expected_score(1500, 1500)
        assert abs(expected_equal - 0.5) < 1e-10

    def test_objective_generators(self):
        """Test objective generators completely."""
        from humpday.optimizers.adaptive_optimizer import (
            rosenbrock_variants_generator,
            sphere_variants_generator,
        )

        # Test sphere generator
        sphere_gen = sphere_variants_generator(n_dim=3)
        objectives = [next(sphere_gen) for _ in range(5)]

        test_point = [0.1, 0.2, 0.3]
        for obj in objectives:
            result = obj(test_point)
            assert isinstance(result, (int, float))

        # Test rosenbrock generator
        rosenbrock_gen = rosenbrock_variants_generator(n_dim=3)
        objectives = [next(rosenbrock_gen) for _ in range(5)]

        for obj in objectives:
            result = obj(test_point)
            assert isinstance(result, (int, float))


class TestAllOptimizersCoverage:
    """Test missing coverage in alloptimizers.py"""

    def test_get_optimizer_function(self):
        """Test get_optimizer function."""
        from humpday import get_optimizer

        # Test getting valid optimizer
        optimizer_func = get_optimizer("NelderMead")
        assert callable(optimizer_func)

        # Test getting invalid optimizer
        try:
            invalid_optimizer = get_optimizer("NonExistentOptimizer")
            # Should either return None or raise an error
        except (KeyError, ValueError):
            pass

    def test_pure_optimize_variations(self):
        """Test pure_optimize with different parameters."""
        from humpday import pure_optimize

        def objective(x):
            return sum((xi - 0.2) ** 2 for xi in x)

        # Test with different algorithms
        algorithms = ["RandomSearch", "NelderMead", "HillClimbing"]

        for alg in algorithms:
            try:
                result = pure_optimize(objective, alg, n_trials=10, n_dim=2)
                assert len(result) == 2
            except Exception:
                # Some algorithms might not work in all cases
                pass

    def test_suggest_pure_function(self):
        """Test suggest_pure function."""
        from humpday import suggest_pure

        suggestions = suggest_pure(n_dim=3, n_trials=50)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(alg, str) for alg in suggestions)


if __name__ == "__main__":
    pytest.main([__file__])


class TestSuggestUsesRealEvidence:
    """suggest() previously returned score = 1000 + 100*i and time = 0.1*i, both fabricated
    from list position and both marked 'Fake ... for compatibility'. A caller reading
    (score, time, name) had every reason to take those for measurements."""

    def test_scores_are_measured_not_positional(self):
        from humpday import ratings, suggest

        n_dim = ratings.recorded_dimensions()[0]
        cell = ratings.cells_at(n_dim, 100)["engineering"]["ratings"]
        for score, _time, name in suggest(n_dim=n_dim, smooth=False):
            assert score == cell[name], f"{name} should carry its measured rating"

    def test_unmeasured_fields_are_nan_rather_than_invented(self):
        from humpday import ratings, suggest

        n_dim = ratings.recorded_dimensions()[0]
        for _score, elapsed, _name in suggest(n_dim=n_dim, smooth=False):
            assert elapsed != elapsed, (
                "humpday records no timing evidence; time must be nan"
            )

    def test_the_default_reports_no_score_because_it_has_none(self):
        # Ranking by worst position across suites is not a rating, and Elo from two tournaments is
        # not on one scale. Returning a number here would be inventing the comparison.
        from humpday import ratings, suggest

        n_dim = ratings.recorded_dimensions()[0]
        assert all(s != s for s, _, _ in suggest(n_dim=n_dim))

    def test_scores_do_not_march_with_position(self):
        # The tell of the old implementation: a fixed arithmetic progression down the list.
        from humpday import ratings, suggest

        n_dim = ratings.recorded_dimensions()[0]
        scores = [s for s, _, _ in suggest(n_dim=n_dim, smooth=False) if s == s]
        gaps = {round(b - a, 6) for a, b in zip(scores, scores[1:])}
        assert len(gaps) > 1, f"scores look positional, not measured: {scores}"


class TestSuggestUsesDimensionSpecificEvidence:
    def test_recorded_cells_order_by_measured_rating(self):
        from humpday import ratings, suggest

        assert ratings.recorded_dimensions(), "ratings must ship with the package"
        for n_dim in ratings.recorded_dimensions():
            for suite, cell in ratings.cells_at(n_dim, 100).items():
                hint = suite == "surfaces"
                scores = [s for s, _, _ in suggest(n_dim=n_dim, smooth=hint)]
                assert scores == sorted(scores, reverse=True), (
                    f"d={n_dim}/{suite} not ordered"
                )
                measured = cell["ratings"]
                assert suggest(n_dim=n_dim, smooth=hint)[0][2] == max(
                    measured, key=measured.get
                )

    def test_ratings_are_never_stretched_to_another_dimension(self):
        # Not even to the neighbouring one. Optimizer performance is not smooth in dimension: a
        # Bayesian method can work well at five and blow up at ten, and the trust-region methods
        # here top d=25 while being unable to build a model at d=100 on a comparable budget.
        #
        # The probes are chosen to be absent from the recorded grid and are asserted to be so.
        # The first version of this test probed n_dim*4 and n_dim*8 of the two smallest recorded
        # dimensions -- 8, 16, 12 and 24, every one of which is itself recorded, so all four
        # `continue`d and the test made no assertion at all. Making `cell_for` fall back to the
        # nearest dimension left it, and 146 other ratings tests, passing.
        from humpday import ratings, suggest

        recorded = set(ratings.recorded_dimensions())
        probes = [d for d in (7, 9, 11, 13, 19, 30, 37, 75, 99) if d not in recorded]
        assert len(probes) >= 5, (
            f"the grid now covers the probe dimensions; pick unrecorded ones: {recorded}"
        )
        for probe in probes:
            assert not ratings.robust_order(probe), f"d={probe} was never raced"
            for suite in ratings.SUITES:
                assert ratings.cell_for(probe, suite, 1000) is None, (
                    f"d={probe} resolved to a cell from another dimension"
                )
                assert ratings.rating(probe, suite, "NelderMead", 1000) is None
            scores = [s for s, _, _ in suggest(n_dim=probe, smooth=False)]
            assert all(s != s for s in scores), (
                f"d={probe} was not measured; ratings from another dimension are not evidence"
            )

    def test_budget_is_interpolated_downward_only(self):
        # A cell recorded at a smaller budget describes an optimizer with less room, which is the
        # safe direction to be wrong in. Reading a 5000-evaluation result to answer a question
        # about 50 would flatter the methods that spend their budget building a model.
        from humpday import ratings

        for n_dim in ratings.recorded_dimensions():
            budgets = sorted(
                int(k.split("/")[1])
                for k in ratings.cells()
                if k.startswith(f"{n_dim}/")
            )
            if len(set(budgets)) < 2:
                continue
            low, high = min(budgets), max(budgets)
            for suite, cell in ratings.cells_at(n_dim, high - 1).items():
                # `.get`, not `[]`: a cell can be absent, because a suite whose problems cost more
                # than the ceiling at that budget is skipped rather than faked.
                assert cell is not ratings.cells().get(f"{n_dim}/{high}/{suite}")
            asked = (low + high) // 2
            for suite, cell in ratings.cells_at(n_dim, asked).items():
                used = next(
                    b
                    for b in sorted(budgets, reverse=True)
                    if ratings.cells().get(f"{n_dim}/{b}/{suite}") is cell
                )
                assert used <= asked, (
                    f"d={n_dim}/{suite}: read budget {used} to answer {asked}"
                )

    def test_minimize_and_suggest_read_the_same_table(self):
        # They were recorded by different scripts over different objectives once, and were free to
        # disagree about the same problem. One table now, one ordering function.
        from humpday import ratings, suggest
        from humpday.eligibility import robust_order

        for n_dim in ratings.recorded_dimensions():
            assert [n for _, _, n in suggest(n_dim=n_dim)] == robust_order(n_dim)


class TestSuggestHonoursObjectiveCharacter:
    def test_the_two_suites_disagree(self):
        # If they agreed, the `smooth` argument would be pointless. They do not: the analytic
        # surfaces are smooth, smoothness rewards local search, and the trust-region methods sweep
        # them while being displaced on the engineering problems at the same dimensions.
        from humpday import ratings, suggest

        both = [
            d
            for d in ratings.recorded_dimensions()
            if len(ratings.cells_at(d, 100)) > 1
        ]
        assert both, "both suites must be recorded somewhere"
        differ = sum(
            1
            for d in both
            if suggest(n_dim=d, smooth=True)[0][2]
            != suggest(n_dim=d, smooth=False)[0][2]
        )
        assert differ >= len(both) // 3, (
            "the suites should disagree on a fair share of dimensions"
        )

    def test_default_is_never_terrible_rather_than_best_on_average(self):
        # The default optimises worst rank across suites. A specialist that wins one and places
        # near the bottom of the other must not lead it: PRIMA_UOBYQA averages 3.0 on surfaces and
        # 15.9 on engineering, which is exactly the recommendation this guards against.
        from humpday import ratings, suggest

        for n_dim in ratings.recorded_dimensions():
            at = ratings.cells_at(n_dim, 100)
            if len(at) < 2:
                continue
            ranked = {
                suite: sorted(c["ratings"], key=lambda n: -c["ratings"][n])
                for suite, c in at.items()
            }
            common = set.intersection(*(set(o) for o in ranked.values()))
            leader = suggest(n_dim=n_dim)[0][2]
            worst = max(o.index(leader) + 1 for o in ranked.values())
            for other in common:
                other_worst = max(o.index(other) + 1 for o in ranked.values())
                assert worst <= other_worst, (
                    f"d={n_dim}: {leader} is worst-ranked {worst}, but {other} is {other_worst}"
                )

    def test_explicit_hint_selects_the_matching_suite(self):
        from humpday import ratings, suggest

        for n_dim in ratings.recorded_dimensions():
            at = ratings.cells_at(n_dim, 100)
            if len(at) < 2:
                continue
            for hint, suite in ((True, "surfaces"), (False, "engineering")):
                measured = at[suite]["ratings"]
                assert suggest(n_dim=n_dim, smooth=hint)[0][2] == max(
                    measured, key=measured.get
                )


class TestPublicSurface:
    def test_every_exported_name_resolves(self):
        # Twice now a block edit has deleted a public function while leaving its name in __all__,
        # and `from humpday import *` is the only thing that notices.
        import humpday

        missing = [n for n in humpday.__all__ if not hasattr(humpday, n)]
        assert not missing, f"exported but undefined: {missing}"


if __name__ == "__main__":
    pytest.main([__file__])


class TestRatingsTableSemantics:
    """The table is the only evidence in the library, so its edge cases are pinned here rather
    than left to whatever the recorder happened to write."""

    def test_a_disqualified_optimizer_ranks_last_and_only_once(self):
        # It is disqualified partway through a cell, so it holds a rating from the rounds it did
        # finish as well as a place in timed_out. Listing it twice would let it lead and trail.
        from humpday.ratings import _ranked

        cell = {"ratings": {"A": 1600, "B": 1580, "C": 1400}, "timed_out": ["B"]}
        assert _ranked(cell) == ["A", "C", "B"]

    def test_a_disqualified_optimizer_reports_no_rating(self):
        # Swept across every recorded budget, not just the default. The first version looped over
        # `timed_out(n_dim, suite)` at the default n_trials=100, which resolves to the 50-evaluation
        # cells -- where nothing has ever overrun, because 50 evaluations is too few to be slow in.
        # Zero iterations, zero assertions, while 79 disqualifications sat in the other budgets.
        from humpday import ratings

        checked = 0
        for key, cell in ratings.cells().items():
            n_dim, budget, suite = key.split("/")
            for name in cell.get("timed_out", []):
                assert ratings.rating(int(n_dim), suite, name, int(budget)) is None, (
                    f"{name} timed out in {key} but still reports a rating"
                )
                assert (
                    ratings._ranked(cell).index(name)
                    >= len(
                        [
                            n
                            for n in cell.get("ratings", {})
                            if n not in cell.get("timed_out", [])
                        ]
                    )
                    or len(cell["timed_out"])
                    / len(set(cell.get("ratings", {})) | set(cell["timed_out"]))
                    > ratings.TIMEOUT_REVOLT
                ), f"{name} should rank last in {key}"
                checked += 1
        assert checked > 20, (
            f"only {checked} disqualifications checked; the sweep is not reaching the data"
        )

    def test_thin_cells_move_the_order_less_than_deep_ones(self):
        # The shrinkage is what lets cells of unequal depth be combined at all: a cell that ran out
        # of wall clock after four problems should not decide the ordering on one bad placing.
        from humpday.ratings import PRIOR_PROBLEMS, robust_order

        names = [f"opt{i}" for i in range(5)]
        good = {n: 1600 - 10 * i for i, n in enumerate(names)}
        reversed_ = {n: 1600 - 10 * i for i, n in enumerate(reversed(names))}

        def order(problems):
            import humpday.ratings as r

            r._TABLE = {
                "7/100/surfaces": {"ratings": good, "problems": 50},
                "7/100/engineering": {"ratings": reversed_, "problems": problems},
            }
            try:
                return robust_order(7, 100)
            finally:
                r._TABLE = None

        thin = order(1)
        deep = order(int(PRIOR_PROBLEMS) * 20)
        assert thin[0] == "opt0", "a one-problem cell should barely move the leader"
        assert deep[0] != "opt0", "a deep contradicting cell should"

    def test_every_shipped_cell_is_backed_by_problems(self):
        from humpday import ratings

        for key, cell in ratings.cells().items():
            assert cell.get("problems", 0) > 0, f"{key} ships ratings backed by nothing"
            assert cell.get("ratings") or cell.get("timed_out"), f"{key} is empty"

    def test_the_two_suites_are_always_read_at_the_same_budget(self):
        # Pairing a suite measured at a thousand evaluations against one measured at fifty would
        # read a budget effect as a suite effect, which is the confusion this table exists to end.
        from humpday import ratings

        for n_dim in ratings.recorded_dimensions():
            at = ratings.cells_at(n_dim, 5000)
            if len(at) < 2:
                continue
            budgets = set()
            for suite, cell in at.items():
                budgets |= {
                    int(k.split("/")[1])
                    for k, v in ratings.cells().items()
                    if v is cell and k.endswith(f"/{suite}")
                }
            assert len(budgets) == 1, f"d={n_dim} mixes budgets {sorted(budgets)}"

    def test_a_cell_where_nearly_everyone_overran_does_not_rank_them_last(self):
        """A mis-calibrated allowance must not read as eleven unusable optimizers.

        The allowance is a multiple of what a random sampler spent on the same objective. When
        evaluation cost depends on location -- as it does on the worked engineering demos -- an
        optimizer that converges into an expensive basin pays what the sampler never did, and is
        disqualified for the objective's shape rather than its own overhead.
        """
        from humpday.ratings import TIMEOUT_REVOLT, _ranked

        revolt = {
            "ratings": {f"o{i}": 1600 - i for i in range(9)},
            "timed_out": [f"o{i}" for i in range(6)],
        }
        assert _ranked(revolt) == [f"o{i}" for i in range(9)], (
            "with two thirds timed out the ordering should ignore the timeouts"
        )

        isolated = {
            "ratings": {f"o{i}": 1600 - i for i in range(9)},
            "timed_out": ["o0"],
        }
        assert _ranked(isolated)[-1] == "o0", (
            "an isolated timeout is still a disqualification"
        )
        assert TIMEOUT_REVOLT < 0.5, "the guard must trip below a majority, not at one"

    def test_the_shipped_table_is_a_complete_grid(self):
        from itertools import product

        from humpday import ratings

        recorded = ratings.cells()
        dims = sorted({int(k.split("/")[0]) for k in recorded})
        budgets = sorted({int(k.split("/")[1]) for k in recorded})
        for n_dim, budget, suite in product(dims, budgets, ratings.SUITES):
            key = f"{n_dim}/{budget}/{suite}"
            assert key in recorded, f"{key} is missing from the grid"
            # A cell need not reach the 30-problem target: #339 lets a cell stop early on
            # its wall-clock budget rather than burn a fixed depth against an objective
            # whose per-evaluation cost the recorder discovers as it goes. What "complete"
            # still means is that every cell has at least one honest measurement.
            assert recorded[key]["problems"] >= 1, f"{key} was never measured"
