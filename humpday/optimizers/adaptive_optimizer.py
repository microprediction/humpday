"""
Adaptive optimization with Elo rating system.

This module provides a function that takes an objective generator and trials budget,
runs algorithm comparisons, updates Elo ratings, and provides running suggestions
for which algorithms to use based on their performance.
"""

import json
import os
from collections.abc import Generator
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .alloptimizers import PURE_OPTIMIZERS, pure_optimize


class EloRatingSystem:
    """Elo rating system for optimization algorithms."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings = dict.fromkeys(PURE_OPTIMIZERS.keys(), initial_rating)
        self.match_history = []

    def get_rating(self, algorithm: str) -> float:
        """Get current Elo rating for an algorithm."""
        return self.ratings.get(algorithm, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for algorithm A vs algorithm B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, algorithm_a: str, algorithm_b: str, score_a: float):
        """
        Update Elo ratings after a match.

        Args:
            algorithm_a: Name of first algorithm
            algorithm_b: Name of second algorithm
            score_a: Score for algorithm A (1.0 = win, 0.5 = tie, 0.0 = loss)
        """
        rating_a = self.get_rating(algorithm_a)
        rating_b = self.get_rating(algorithm_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1.0 - score_a) - expected_b)

        self.ratings[algorithm_a] = new_rating_a
        self.ratings[algorithm_b] = new_rating_b

        # Record match
        self.match_history.append(
            {
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "score_a": score_a,
                "rating_a_before": rating_a,
                "rating_b_before": rating_b,
                "rating_a_after": new_rating_a,
                "rating_b_after": new_rating_b,
            }
        )

    def get_top_algorithms(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N algorithms by Elo rating."""
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_ratings[:n]

    def save_ratings(self, filepath: str):
        """Save ratings and history to file."""
        data = {
            "ratings": self.ratings,
            "match_history": self.match_history,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_ratings(self, filepath: str) -> bool:
        """Load ratings and history from file. Returns True if successful."""
        try:
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)

                self.ratings = data.get("ratings", {})
                self.match_history = data.get("match_history", [])
                self.initial_rating = data.get("initial_rating", 1500.0)
                self.k_factor = data.get("k_factor", 32.0)

                # Ensure all algorithms have ratings
                for alg in PURE_OPTIMIZERS.keys():
                    if alg not in self.ratings:
                        self.ratings[alg] = self.initial_rating

                return True
        except Exception:
            pass
        return False


def normalize_performance(values: List[float]) -> List[float]:
    """Normalize performance values to [0, 1] range."""
    if not values or len(values) < 2:
        return [0.5] * len(values)

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return [0.5] * len(values)

    # Invert because lower objective values are better
    return [(max_val - val) / (max_val - min_val) for val in values]


def run_algorithm_tournament(
    objective_generator: Generator[Callable[[np.ndarray], float], None, None],
    trials_per_problem: int,
    n_problems: int,
    n_dim: int,
    elo_system: Optional[EloRatingSystem] = None,
    algorithms_to_test: Optional[List[str]] = None,
) -> EloRatingSystem:
    """
    Run a tournament between algorithms on multiple problems.

    Args:
        objective_generator: Generator yielding objective functions
        trials_per_problem: Budget per algorithm per problem
        n_problems: Number of problems to test
        n_dim: Problem dimension
        elo_system: Existing Elo system to update (creates new if None)
        algorithms_to_test: List of algorithm names to test (tests all if None)

    Returns:
        Updated EloRatingSystem
    """
    if elo_system is None:
        elo_system = EloRatingSystem()

    if algorithms_to_test is None:
        algorithms_to_test = list(PURE_OPTIMIZERS.keys())

    print(
        f"Running tournament with {len(algorithms_to_test)} algorithms on {n_problems} problems..."
    )

    for problem_idx in range(n_problems):
        try:
            objective = next(objective_generator)
        except StopIteration:
            print(f"Objective generator exhausted after {problem_idx} problems")
            break

        print(f"Problem {problem_idx + 1}/{n_problems}")

        # Run all algorithms on this problem
        results = {}
        for alg_name in algorithms_to_test:
            try:
                best_value, _ = pure_optimize(
                    objective, alg_name, trials_per_problem, n_dim
                )
                results[alg_name] = best_value
            except Exception as e:
                print(f"Algorithm {alg_name} failed: {e}")
                results[alg_name] = float("inf")

        # Convert to normalized scores
        performance_values = list(results.values())
        normalized_scores = normalize_performance(performance_values)

        # Update Elo ratings with pairwise comparisons
        alg_names = list(results.keys())
        for i in range(len(alg_names)):
            for j in range(i + 1, len(alg_names)):
                alg_a = alg_names[i]
                alg_b = alg_names[j]

                score_a = normalized_scores[i]
                score_b = normalized_scores[j]

                # Convert to Elo score (0, 0.5, 1)
                if score_a > score_b:
                    elo_score_a = 1.0
                elif score_a < score_b:
                    elo_score_a = 0.0
                else:
                    elo_score_a = 0.5

                elo_system.update_ratings(alg_a, alg_b, elo_score_a)

    return elo_system


def adaptive_optimize(
    objective_generator: Generator[Callable[[np.ndarray], float], None, None],
    trials_budget: int,
    n_dim: int,
    n_warmup_problems: int = 5,
    trials_per_warmup: int = 50,
    elo_ratings_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Adaptive optimization that learns which algorithms work best.

    This function takes an objective generator and trials budget, runs comparisons,
    updates Elo ratings, and provides running suggestions for which algorithms to use.

    Args:
        objective_generator: Generator yielding objective functions to optimize
        trials_budget: Total budget of function evaluations
        n_dim: Problem dimension
        n_warmup_problems: Number of problems for initial algorithm comparison
        trials_per_warmup: Trials per algorithm during warmup
        elo_ratings_file: File to load/save Elo ratings (optional)
        verbose: Print progress information

    Returns:
        Dictionary containing:
        - 'elo_system': Final EloRatingSystem
        - 'top_algorithms': List of (algorithm, rating) tuples
        - 'recommendations': Algorithm suggestions by problem type
        - 'total_problems_solved': Number of problems processed
        - 'best_results': Best results found per problem
    """

    # Initialize or load Elo system
    elo_system = EloRatingSystem()
    if elo_ratings_file and elo_system.load_ratings(elo_ratings_file):
        if verbose:
            print(f"Loaded existing Elo ratings from {elo_ratings_file}")
    else:
        if verbose:
            print("Initializing new Elo rating system")

    # Warmup phase: test all algorithms on several problems
    if verbose:
        print(f"\nWarmup phase: testing all algorithms on {n_warmup_problems} problems")

    elo_system = run_algorithm_tournament(
        objective_generator=objective_generator,
        trials_per_problem=trials_per_warmup,
        n_problems=n_warmup_problems,
        n_dim=n_dim,
        elo_system=elo_system,
    )

    # Get current top algorithms
    top_algorithms = elo_system.get_top_algorithms(10)
    if verbose:
        print("\nTop algorithms after warmup:")
        for i, (alg, rating) in enumerate(top_algorithms[:5], 1):
            print(f"{i}. {alg}: {rating:.1f}")

    # Adaptive phase: focus on top algorithms
    remaining_budget = trials_budget - (
        n_warmup_problems * len(PURE_OPTIMIZERS) * trials_per_warmup
    )
    if remaining_budget > 0:
        # Select top algorithms for continued testing
        top_algorithm_names = [alg for alg, _ in top_algorithms[:8]]

        if verbose:
            print(
                f"\nAdaptive phase: focusing on top {len(top_algorithm_names)} algorithms"
            )
            print(f"Remaining budget: {remaining_budget} trials")

        # Continue testing with adaptive strategy
        adaptive_problems = max(
            1, remaining_budget // (len(top_algorithm_names) * trials_per_warmup)
        )

        elo_system = run_algorithm_tournament(
            objective_generator=objective_generator,
            trials_per_problem=trials_per_warmup,
            n_problems=adaptive_problems,
            n_dim=n_dim,
            elo_system=elo_system,
            algorithms_to_test=top_algorithm_names,
        )

    # Final results
    final_top_algorithms = elo_system.get_top_algorithms()

    # Create recommendations by problem characteristics
    recommendations = {
        "low_dim_smooth": [
            alg
            for alg, _ in final_top_algorithms
            if alg in ["NelderMead", "PRIMA_UOBYQA", "Powell", "LBFGSB"]
        ][:3],
        "medium_dim_multimodal": [
            alg
            for alg, _ in final_top_algorithms
            if alg in ["DifferentialEvolution", "CMAEvolutionStrategy", "ParticleSwarm"]
        ][:3],
        "high_dim_complex": [
            alg
            for alg, _ in final_top_algorithms
            if alg
            in ["CMAEvolutionStrategy", "AdaptiveRandomSearch", "EvolutionStrategy"]
        ][:3],
        "general_purpose": [alg for alg, _ in final_top_algorithms[:5]],
    }

    # Save updated ratings
    if elo_ratings_file:
        elo_system.save_ratings(elo_ratings_file)
        if verbose:
            print(f"\nSaved updated Elo ratings to {elo_ratings_file}")

    if verbose:
        print("\nFinal top algorithms:")
        for i, (alg, rating) in enumerate(final_top_algorithms[:5], 1):
            print(f"{i}. {alg}: {rating:.1f}")

    return {
        "elo_system": elo_system,
        "top_algorithms": final_top_algorithms,
        "recommendations": recommendations,
        "total_problems_solved": (
            n_warmup_problems + adaptive_problems
            if remaining_budget > 0
            else n_warmup_problems
        ),
        "total_matches": len(elo_system.match_history),
    }


def suggest_algorithm_from_elo(
    elo_system: EloRatingSystem, n_dim: int, problem_type: str = "general"
) -> str:
    """
    Suggest best algorithm based on Elo ratings and problem characteristics.

    Args:
        elo_system: EloRatingSystem with current ratings
        n_dim: Problem dimension
        problem_type: 'smooth', 'multimodal', 'noisy', or 'general'

    Returns:
        Algorithm name suggestion
    """
    top_algorithms = elo_system.get_top_algorithms(20)

    # Filter by problem characteristics
    if problem_type == "smooth" and n_dim <= 10:
        candidates = ["NelderMead", "PRIMA_UOBYQA", "Powell", "LBFGSB"]
    elif problem_type == "multimodal":
        candidates = [
            "DifferentialEvolution",
            "CMAEvolutionStrategy",
            "ParticleSwarm",
            "GeneticAlgorithm",
        ]
    elif problem_type == "noisy":
        candidates = ["CMAEvolutionStrategy", "AdaptiveRandomSearch", "ParticleSwarm"]
    else:  # general
        candidates = [alg for alg, _ in top_algorithms[:10]]

    # Find highest-rated candidate
    for alg, rating in top_algorithms:
        if alg in candidates:
            return alg

    # Fallback to highest-rated overall
    return top_algorithms[0][0] if top_algorithms else "NelderMead"


# Lightweight self-contained objective functions - no 3rd party dependencies!
# These are tested against reference implementations to ensure correctness


# Pure lightweight implementations
def sphere_variants_generator(
    n_dim: int = 2,
) -> Generator[Callable[[np.ndarray], float], None, None]:
    """Generator yielding variants of the sphere function - pure lightweight implementation."""

    def sphere_pure(x):
        """Pure sphere function: sum(x^2)"""
        x = np.asarray(x)
        return np.sum(x * x)

    def shifted_sphere(x):
        """Sphere with random shift"""
        x = np.asarray(x)
        shift = np.random.uniform(-0.3, 0.3, len(x))
        x_shifted = x + shift
        return np.sum(x_shifted * x_shifted)

    def scaled_sphere(x):
        """Sphere with different scales per dimension"""
        x = np.asarray(x)
        scales = np.random.uniform(0.5, 2.0, len(x))
        x_scaled = scales * x
        return np.sum(x_scaled * x_scaled)

    variants = [sphere_pure, shifted_sphere, scaled_sphere]

    while True:
        yield np.random.choice(variants)


def rosenbrock_variants_generator(
    n_dim: int = 2,
) -> Generator[Callable[[np.ndarray], float], None, None]:
    """Generator yielding variants of the Rosenbrock function - pure lightweight implementation."""

    def rosenbrock_pure(x):
        """Pure Rosenbrock function: sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)"""
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def scaled_rosenbrock(x):
        """Rosenbrock with random scaling"""
        x = np.asarray(x)
        scale = np.random.uniform(0.1, 5.0)
        return scale * np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def shifted_rosenbrock(x):
        """Rosenbrock with random shift"""
        x = np.asarray(x)
        shift = np.random.uniform(-0.2, 0.2, len(x))
        x_shifted = x + shift
        return np.sum(
            100.0 * (x_shifted[1:] - x_shifted[:-1] ** 2) ** 2
            + (1 - x_shifted[:-1]) ** 2
        )

    variants = [rosenbrock_pure, scaled_rosenbrock, shifted_rosenbrock]

    while True:
        yield np.random.choice(variants)
