#!/usr/bin/env python3
"""
Example of adaptive optimization with Elo rating system.

This demonstrates the new adaptive_optimize function that takes an objective
generator and trials budget, runs comparisons, updates Elos, and gives
running suggestions for which algorithm to use.
"""

from collections.abc import Generator
from typing import Callable

import numpy as np

from humpday import (
    EloRatingSystem,
    adaptive_optimize,
    pure_optimize,
    suggest_algorithm_from_elo,
)


def mixed_function_generator(
    n_dim: int = 2,
) -> Generator[Callable[[np.ndarray], float], None, None]:
    """Generator yielding a mix of different optimization problems - pure lightweight implementations."""

    def sphere(x):
        """Sphere function: sum(x^2)"""
        x = np.asarray(x)
        return np.sum(x * x)

    def rosenbrock(x):
        """Rosenbrock function: sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)"""
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def ackley(x):
        """Ackley function - multimodal with global minimum at origin"""
        x = np.asarray(x)
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
        cos_term = -np.exp(np.sum(np.cos(c * x)) / len(x))
        return a + np.exp(1) + sum_sq_term + cos_term

    def rastrigin(x):
        """Rastrigin function - highly multimodal"""
        x = np.asarray(x)
        return 10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))

    def griewank(x):
        """Griewank function - multimodal with product term"""
        x = np.asarray(x)
        sum_term = np.sum(x**2) / 4000.0
        prod_term = np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])
        return sum_term - prod_term + 1.0

    def schwefel(x):
        """Schwefel function - deceptive global structure"""
        x = np.asarray(x)
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    # Pure lightweight implementations - no 3rd party dependencies
    functions = [sphere, rosenbrock, ackley, rastrigin, griewank, schwefel]

    while True:
        base_func = np.random.choice(functions)

        # Add random transformations
        shift = np.random.uniform(-0.3, 0.3, n_dim)
        scale = np.random.uniform(0.5, 2.0)

        def transformed_func(x, base=base_func, s=shift, sc=scale):
            # Map from [0,1] to appropriate domain
            x = np.asarray(x)
            if base in [schwefel]:
                # Schwefel needs [-500, 500] domain
                x_scaled = (x - 0.5) * 1000
            elif base in [ackley, rastrigin]:
                # These work well on [-5, 5]
                x_scaled = (x - 0.5) * 10
            else:
                # Others work on [-2, 2] or similar
                x_scaled = (x - 0.5) * 4

            x_shifted = x_scaled + s
            return sc * base(x_shifted)

        yield transformed_func


def demonstrate_adaptive_optimization():
    """Demonstrate the adaptive optimization system."""

    print("=" * 80)
    print("ADAPTIVE OPTIMIZATION WITH ELO RATING SYSTEM")
    print("=" * 80)

    # Problem setup
    n_dim = 3
    trials_budget = 5000  # Total evaluation budget
    elo_file = "algorithm_elos.json"

    print(f"\nProblem dimension: {n_dim}")
    print(f"Total trials budget: {trials_budget}")
    print()

    # Create objective generator
    objective_gen = mixed_function_generator(n_dim)

    # Run adaptive optimization
    results = adaptive_optimize(
        objective_generator=objective_gen,
        trials_budget=trials_budget,
        n_dim=n_dim,
        n_warmup_problems=8,  # Test all algorithms on 8 diverse problems
        trials_per_warmup=60,  # 60 evaluations per algorithm per problem
        elo_ratings_file=elo_file,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"\nTotal problems solved: {results['total_problems_solved']}")
    print(f"Total algorithm matches: {results['total_matches']}")

    print("\nTop 10 algorithms by Elo rating:")
    for i, (alg, rating) in enumerate(results["top_algorithms"][:10], 1):
        print(f"{i:2d}. {alg:20s}: {rating:8.1f}")

    print("\nAlgorithm recommendations by problem type:")
    for problem_type, algs in results["recommendations"].items():
        print(f"{problem_type:20s}: {', '.join(algs[:3])}")

    # Demonstrate algorithm suggestion
    elo_system = results["elo_system"]

    print("\n" + "=" * 60)
    print("ALGORITHM SUGGESTIONS")
    print("=" * 60)

    problem_types = ["smooth", "multimodal", "noisy", "general"]
    for problem_type in problem_types:
        suggested = suggest_algorithm_from_elo(elo_system, n_dim, problem_type)
        rating = elo_system.get_rating(suggested)
        print(f"{problem_type:12s} problems: {suggested:20s} (Elo: {rating:.1f})")

    print("\n" + "=" * 60)
    print("LIVE OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Test the suggestion on a new problem
    def test_problem(x):
        # Shifted Rosenbrock
        shift = np.array([0.1, 0.2, 0.3])
        x_shifted = x + shift
        return sum(
            100.0 * (x_shifted[i + 1] - x_shifted[i] ** 2) ** 2
            + (1 - x_shifted[i]) ** 2
            for i in range(len(x_shifted) - 1)
        )

    suggested_alg = suggest_algorithm_from_elo(elo_system, n_dim, "smooth")
    print(f"\nOptimizing shifted Rosenbrock with suggested algorithm: {suggested_alg}")

    best_val, best_x = pure_optimize(test_problem, suggested_alg, 100, n_dim)
    print(f"Best value found: {best_val:.6f}")
    print(f"Best point: {best_x}")

    # Compare with random algorithm
    import random

    from humpday import PURE_OPTIMIZERS

    random_alg = random.choice(list(PURE_OPTIMIZERS.keys()))
    best_val_random, _ = pure_optimize(test_problem, random_alg, 100, n_dim)
    print(f"\nComparison with random algorithm ({random_alg}):")
    print(f"Suggested algorithm result: {best_val:.6f}")
    print(f"Random algorithm result:    {best_val_random:.6f}")
    print(f"Improvement ratio: {best_val_random / (best_val + 1e-10):.2f}x")


def demonstrate_elo_tracking():
    """Show how Elo ratings evolve over time."""

    print("\n" + "=" * 80)
    print("ELO RATING EVOLUTION DEMONSTRATION")
    print("=" * 80)

    elo_system = EloRatingSystem()

    # Simulate some algorithm battles
    algorithms = [
        "NelderMead",
        "DifferentialEvolution",
        "ParticleSwarm",
        "CMAEvolutionStrategy",
    ]

    print("Initial ratings:")
    for alg in algorithms:
        print(f"{alg:20s}: {elo_system.get_rating(alg):.1f}")

    # Simulate differential evolution consistently beating others
    print("\nSimulating DifferentialEvolution winning against others...")
    for _ in range(10):
        for other in algorithms:
            if other != "DifferentialEvolution":
                elo_system.update_ratings(
                    "DifferentialEvolution", other, 1.0
                )  # DE wins

    print("\nRatings after DE winning streak:")
    for alg in algorithms:
        print(f"{alg:20s}: {elo_system.get_rating(alg):.1f}")

    # Now simulate some losses for DE
    print("\nSimulating CMAEvolutionStrategy beating DifferentialEvolution...")
    for _ in range(5):
        elo_system.update_ratings("CMAEvolutionStrategy", "DifferentialEvolution", 1.0)

    print("\nFinal ratings:")
    for alg in algorithms:
        print(f"{alg:20s}: {elo_system.get_rating(alg):.1f}")


if __name__ == "__main__":
    # Run the demonstrations
    demonstrate_adaptive_optimization()
    demonstrate_elo_tracking()

    print("\n" + "=" * 80)
    print("QUICK START GUIDE")
    print("=" * 80)
    print("""
To use adaptive optimization in your code:

    from humpday import adaptive_optimize

    # Create your objective generator
    def my_objective_generator():
        while True:
            # Generate diverse test problems
            yield lambda x: np.sum((x - np.random.random(len(x)))**2)

    # Run adaptive optimization
    results = adaptive_optimize(
        objective_generator=my_objective_generator(),
        trials_budget=2000,
        n_dim=5,
        n_warmup_problems=5,
        trials_per_warmup=50
    )

    # Get best algorithms
    top_algorithms = results['top_algorithms'][:3]
    print("Best algorithms:", [alg for alg, rating in top_algorithms])

This automatically:
- Tests all 22 algorithms on diverse problems
- Maintains Elo ratings based on performance
- Adapts suggestions based on learned performance
- Provides tailored recommendations for your problem domain
""")
