#!/usr/bin/env python3
"""
Single objective optimization contest platform.
Users describe problems in plain English, optimizers compete to find the minimum.
"""

import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

# Import our optimizers and surfaces
sys.path.append("/Users/petercotton/github/humpday/humpday/optimizers")
from primacube import prima_newuoa_cube, prima_uobyqa_cube

sys.path.append("/Users/petercotton/github/humpday/humpday/objectives")
from stochastic_surfaces import StochasticSurfaceGenerator


@dataclass
class ProblemSpec:
    """Single objective problem specification."""

    description: str
    dimensions: int
    surface_type: str
    difficulty: str
    evaluation_budget: int
    domain: str


class SingleObjectivePlatform:
    """Contest platform for single objective optimization."""

    def __init__(self):
        self.surface_generator = StochasticSurfaceGenerator(seed=12345)
        self.elo_ratings = {
            "PRIMA_UOBYQA": 1500,
            "PRIMA_NEWUOA": 1500,
            "SciPy_BFGS": 1500,
            "SciPy_Powell": 1500,
            "SciPy_NelderMead": 1500,
        }
        self.match_history = []

    def interpret_problem_description(self, description: str) -> ProblemSpec:
        """Simple NLP to interpret optimization problem description."""

        description = description.lower()

        # Extract dimensions
        dimensions = 5  # Default
        dim_patterns = [
            r"(\d+)\s*(?:dimension|parameter|variable|feature)",
            r"(\d+)d\s",
            r"with\s*(\d+)\s*(?:dim|param|var)",
        ]

        for pattern in dim_patterns:
            match = re.search(pattern, description)
            if match:
                dimensions = int(match.group(1))
                break

        # Determine surface type based on keywords
        if any(
            word in description for word in ["smooth", "quadratic", "simple", "sphere"]
        ):
            surface_type = "smooth"
        elif any(
            word in description for word in ["valley", "rosenbrock", "banana", "curved"]
        ):
            surface_type = "valley"
        elif any(
            word in description
            for word in ["multimodal", "multiple", "local minima", "peaks"]
        ):
            surface_type = "multimodal"
        elif any(word in description for word in ["noisy", "stochastic", "random"]):
            surface_type = "noisy"
        else:
            surface_type = "mixed"  # Default to varied

        # Determine difficulty
        if dimensions <= 5:
            difficulty = "easy"
        elif dimensions <= 15:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Set evaluation budget based on difficulty
        budget_map = {"easy": 60, "medium": 100, "hard": 150}
        evaluation_budget = budget_map[difficulty]

        # Classify domain (simple keyword matching)
        if any(
            word in description
            for word in ["portfolio", "trading", "finance", "investment"]
        ):
            domain = "finance"
        elif any(
            word in description for word in ["neural", "learning", "model", "ml", "ai"]
        ):
            domain = "machine_learning"
        elif any(
            word in description for word in ["design", "antenna", "engine", "material"]
        ):
            domain = "engineering"
        else:
            domain = "general"

        return ProblemSpec(
            description=description,
            dimensions=dimensions,
            surface_type=surface_type,
            difficulty=difficulty,
            evaluation_budget=evaluation_budget,
            domain=domain,
        )

    def generate_challenge_surfaces(
        self, spec: ProblemSpec, n_surfaces: int = 5
    ) -> List[Tuple[str, Callable]]:
        """Generate appropriate challenge surfaces for the problem spec."""

        surfaces = []

        if spec.surface_type == "smooth":
            # Smooth functions like sphere variants
            for i in range(n_surfaces):
                surfaces.append(
                    (
                        f"sphere_variant_{i}",
                        self.surface_generator.stochastic_sphere(f"smooth_{i}"),
                    )
                )

        elif spec.surface_type == "valley":
            # Valley functions like Rosenbrock variants
            for i in range(n_surfaces):
                surfaces.append(
                    (
                        f"valley_{i}",
                        self.surface_generator.stochastic_rosenbrock(f"valley_{i}"),
                    )
                )

        elif spec.surface_type == "multimodal":
            # Multi-modal functions
            for i in range(n_surfaces):
                if i % 2 == 0:
                    surfaces.append(
                        (
                            f"rastrigin_{i}",
                            self.surface_generator.stochastic_rastrigin(f"multi_{i}"),
                        )
                    )
                else:
                    surfaces.append(
                        (
                            f"ackley_{i}",
                            self.surface_generator.stochastic_ackley(f"multi_{i}"),
                        )
                    )

        elif spec.surface_type == "noisy":
            # Add noise to base functions
            base_functions = [
                self.surface_generator.stochastic_sphere,
                self.surface_generator.stochastic_rosenbrock,
            ]
            for i, base_func in enumerate(base_functions * (n_surfaces // 2 + 1)):
                if i >= n_surfaces:
                    break
                surfaces.append((f"noisy_{i}", base_func(f"noisy_{i}")))

        else:  # mixed
            # Mix of different surface types
            function_types = [
                self.surface_generator.stochastic_sphere,
                self.surface_generator.stochastic_rosenbrock,
                self.surface_generator.stochastic_rastrigin,
                self.surface_generator.stochastic_ackley,
                self.surface_generator.stochastic_griewank,
            ]
            for i in range(n_surfaces):
                func_type = function_types[i % len(function_types)]
                surfaces.append((f"mixed_{i}", func_type(f"mixed_{i}")))

        return surfaces

    def get_optimizers(self) -> Dict[str, Callable]:
        """Get available optimizers for competition."""

        from scipy.optimize import minimize

        optimizers = {}

        # PRIMA methods
        optimizers["PRIMA_UOBYQA"] = prima_uobyqa_cube
        optimizers["PRIMA_NEWUOA"] = prima_newuoa_cube

        # SciPy methods with improved wrappers
        def make_scipy_optimizer(method_name):
            def optimizer(objective, n_trials, n_dim, with_count=False):

                def sigmoid_transform(x_unbounded):
                    """Transform unbounded to [0,1] using sigmoid."""
                    return 1 / (1 + np.exp(-np.array(x_unbounded)))

                def inverse_sigmoid_transform(x_bounded):
                    """Transform [0,1] to unbounded using inverse sigmoid."""
                    x_bounded = np.clip(x_bounded, 0.001, 0.999)  # Avoid infinities
                    return np.log(x_bounded / (1 - x_bounded))

                # For methods that don't handle bounds well
                if method_name in ["Nelder-Mead", "Powell"]:

                    def transformed_objective(x_unbounded):
                        x_bounded = sigmoid_transform(x_unbounded)
                        return objective(x_bounded)

                    # Start from random bounded point, transform to unbounded
                    x0_bounded = np.random.rand(n_dim)
                    x0_unbounded = inverse_sigmoid_transform(x0_bounded)

                    try:
                        result = minimize(
                            transformed_objective,
                            x0_unbounded,
                            method=method_name,
                            options={"maxfev": n_trials},
                        )

                        if result.success:
                            best_x_bounded = sigmoid_transform(result.x)
                            best_val = result.fun
                            n_evals = result.nfev
                        else:
                            best_x_bounded = x0_bounded
                            best_val = objective(x0_bounded)
                            n_evals = (
                                result.nfev if hasattr(result, "nfev") else n_trials
                            )

                    except:
                        best_x_bounded = x0_bounded
                        best_val = objective(x0_bounded)
                        n_evals = n_trials

                else:
                    # Methods that handle bounds well
                    def bounded_objective(x):
                        return objective(np.clip(x, 0, 1))

                    x0 = np.random.rand(n_dim)

                    try:
                        result = minimize(
                            bounded_objective,
                            x0,
                            method=method_name,
                            bounds=[(0.001, 0.999)] * n_dim,
                            options={"maxfev": n_trials},
                        )

                        if result.success:
                            best_x_bounded = np.clip(result.x, 0, 1)
                            best_val = result.fun
                            n_evals = result.nfev
                        else:
                            best_x_bounded = x0
                            best_val = objective(x0)
                            n_evals = (
                                result.nfev if hasattr(result, "nfev") else n_trials
                            )

                    except:
                        best_x_bounded = x0
                        best_val = objective(x0)
                        n_evals = n_trials

                if with_count:
                    return best_val, best_x_bounded, n_evals
                else:
                    return best_val

            return optimizer

        optimizers["SciPy_BFGS"] = make_scipy_optimizer("L-BFGS-B")
        optimizers["SciPy_Powell"] = make_scipy_optimizer("Powell")
        optimizers["SciPy_NelderMead"] = make_scipy_optimizer("Nelder-Mead")

        return optimizers

    def run_single_contest(self, problem_spec: ProblemSpec) -> Dict:
        """Run a single optimization contest."""

        print(f"🏁 Running Contest: {problem_spec.description}")
        print(
            f"📊 {problem_spec.dimensions}D {problem_spec.surface_type} ({problem_spec.difficulty})"
        )
        print(f"🎯 Budget: {problem_spec.evaluation_budget} evaluations")
        print()

        # Generate challenge surfaces
        surfaces = self.generate_challenge_surfaces(problem_spec, n_surfaces=3)
        optimizers = self.get_optimizers()

        contest_results = []

        for surface_name, surface_func in surfaces:
            print(f"Surface: {surface_name}")

            surface_results = {}

            for opt_name, opt_func in optimizers.items():
                print(f"  {opt_name:15}...", end=" ", flush=True)

                # Run multiple trials
                trials = []
                for trial in range(3):
                    np.random.seed(trial * 100 + hash(opt_name + surface_name) % 1000)

                    start_time = time.time()
                    try:
                        result = opt_func(
                            surface_func,
                            problem_spec.evaluation_budget,
                            problem_spec.dimensions,
                            with_count=True,
                        )

                        if isinstance(result, tuple) and len(result) >= 3:
                            val, x, evals = result[:3]
                            if np.isfinite(val):
                                trials.append(val)

                    except Exception:
                        pass  # Trial failed

                if trials:
                    avg_result = np.mean(trials)
                    surface_results[opt_name] = avg_result
                    print(f"✓ {avg_result:.4f}")
                else:
                    print("❌ Failed")

            # Store results for this surface
            if surface_results:
                # Rank optimizers for this surface (lower is better)
                ranked = sorted(surface_results.items(), key=lambda x: x[1])

                for rank, (opt_name, score) in enumerate(ranked):
                    contest_results.append(
                        {
                            "surface": surface_name,
                            "optimizer": opt_name,
                            "score": score,
                            "rank": rank + 1,
                            "points": len(ranked)
                            - rank,  # Higher points for better rank
                        }
                    )

            print()

        return contest_results

    def update_elo_ratings(self, contest_results: List[Dict]):
        """Update Elo ratings based on contest results."""

        # Group by surface to get head-to-head comparisons
        surfaces = {}
        for result in contest_results:
            surface = result["surface"]
            if surface not in surfaces:
                surfaces[surface] = {}
            surfaces[surface][result["optimizer"]] = result["score"]

        # Update Elo for each surface (all pairwise comparisons)
        for surface, scores in surfaces.items():
            optimizers = list(scores.keys())

            for i, opt1 in enumerate(optimizers):
                for j, opt2 in enumerate(optimizers[i + 1 :], i + 1):
                    # Determine winner (lower score wins)
                    if scores[opt1] < scores[opt2]:
                        winner, loser = opt1, opt2
                        score1, score2 = 1.0, 0.0  # Winner gets 1, loser gets 0
                    elif scores[opt2] < scores[opt1]:
                        winner, loser = opt2, opt1
                        score1, score2 = 0.0, 1.0
                    else:
                        # Tie
                        score1, score2 = 0.5, 0.5

                    # Update Elo ratings
                    K = 32  # Elo K-factor

                    rating1 = self.elo_ratings[opt1]
                    rating2 = self.elo_ratings[opt2]

                    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
                    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))

                    self.elo_ratings[opt1] += K * (score1 - expected1)
                    self.elo_ratings[opt2] += K * (score2 - expected2)

    def get_leaderboard(self) -> List[Dict]:
        """Get current Elo leaderboard."""

        leaderboard = []
        sorted_optimizers = sorted(
            self.elo_ratings.items(), key=lambda x: x[1], reverse=True
        )

        for rank, (optimizer, elo) in enumerate(sorted_optimizers, 1):
            leaderboard.append(
                {
                    "rank": rank,
                    "optimizer": optimizer,
                    "elo_rating": round(elo, 1),
                    "badge": (
                        "🥇"
                        if rank == 1
                        else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
                    ),
                }
            )

        return leaderboard


def demo_single_objective_platform():
    """Demonstrate the single objective platform."""

    print("🚀 Single Objective Optimization Platform Demo")
    print("=" * 55)

    platform = SingleObjectivePlatform()

    # Test problem descriptions
    problems = [
        "Optimize a neural network with 8 parameters",
        "Find minimum of smooth 5-dimensional function",
        "Portfolio optimization with 12 assets",
        "Engineering design with 15 variables and multiple local minima",
    ]

    print("🤖 Problem Interpretation Examples:")
    print("-" * 40)

    for problem in problems:
        spec = platform.interpret_problem_description(problem)
        print(f"Input:  '{problem}'")
        print(
            f"Output: {spec.dimensions}D {spec.surface_type} problem ({spec.difficulty})"
        )
        print(f"        Budget: {spec.evaluation_budget} evals, Domain: {spec.domain}")
        print()

    # Run a sample contest
    print("🏁 Running Sample Contest")
    print("=" * 30)

    sample_problem = "Optimize smooth function with 5 parameters"
    spec = platform.interpret_problem_description(sample_problem)

    results = platform.run_single_contest(spec)
    platform.update_elo_ratings(results)

    # Show leaderboard
    print("🏆 Updated Leaderboard:")
    print("-" * 25)
    leaderboard = platform.get_leaderboard()

    for entry in leaderboard:
        print(
            f"{entry['badge']} {entry['rank']}. {entry['optimizer']:15} | Elo: {entry['elo_rating']}"
        )


if __name__ == "__main__":
    demo_single_objective_platform()
