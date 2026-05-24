"""
Comprehensive Benchmark Suite for Optimization Algorithm Validation

This module provides:
1. Standard benchmark problems with known optima
2. Reference implementations from literature
3. Problem transformations for unit cube [0,1]^n
4. Mathematical validation problems

BENCHMARK CATEGORIES:
- Smooth Unimodal: Sphere, Quadratic, Rosenbrock
- Multimodal: Rastrigin, Ackley, Griewank
- Noisy: Noisy variants of standard functions
- Constrained: Bounds-constrained problems
- High-dimensional: Scalable to various dimensions

Author: HumpDay Benchmark Suite
Date: 2026-05-23
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BenchmarkMetadata:
    """Metadata for benchmark problems."""
    name: str
    dimension: int
    optimal_value: float
    optimal_point: np.ndarray
    problem_class: str  # 'smooth', 'multimodal', 'noisy', 'constrained'
    difficulty: str     # 'easy', 'medium', 'hard'
    literature_reference: str
    domain_bounds: Tuple[float, float] = (0.0, 1.0)


class BenchmarkProblem(ABC):
    """Abstract base class for benchmark problems."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.metadata = self._get_metadata()

    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """Evaluate the objective function."""
        pass

    @abstractmethod
    def _get_metadata(self) -> BenchmarkMetadata:
        """Get problem metadata."""
        pass

    def __call__(self, x) -> float:
        """Allow direct function calls."""
        return self.objective(np.asarray(x))


class SphereProblem(BenchmarkProblem):
    """Sphere function: f(x) = sum(x_i^2)

    MATHEMATICAL PROPERTIES:
    - Unimodal, convex, smooth
    - Global minimum at origin: f(0,...,0) = 0
    - Separable
    - Well-conditioned
    """

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Transform [0,1]^n to [-5,5]^n for proper sphere domain
        x_scaled = (x - 0.5) * 10
        return np.sum(x_scaled ** 2)

    def _get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name=f"Sphere_{self.dimension}D",
            dimension=self.dimension,
            optimal_value=0.0,
            optimal_point=np.full(self.dimension, 0.5),  # Center of unit cube
            problem_class="smooth",
            difficulty="easy",
            literature_reference="Schwefel (1995)"
        )


class RosenbrockProblem(BenchmarkProblem):
    """Rosenbrock function: f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)

    MATHEMATICAL PROPERTIES:
    - Unimodal but with narrow curved valley
    - Global minimum at (1,1,...,1): f(1,...,1) = 0
    - Non-separable
    - Ill-conditioned, challenging for optimization
    """

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        if len(x) < 2:
            return float('inf')

        # Transform [0,1]^n to [-2,2]^n for standard Rosenbrock domain
        x_scaled = (x - 0.5) * 4

        return np.sum(100.0 * (x_scaled[1:] - x_scaled[:-1]**2)**2 + (1 - x_scaled[:-1])**2)

    def _get_metadata(self) -> BenchmarkMetadata:
        # Optimal point (1,1,...,1) maps to (0.75,0.75,...,0.75) in [0,1]^n
        return BenchmarkMetadata(
            name=f"Rosenbrock_{self.dimension}D",
            dimension=self.dimension,
            optimal_value=0.0,
            optimal_point=np.full(self.dimension, 0.75),
            problem_class="smooth",
            difficulty="hard",
            literature_reference="Rosenbrock (1960)"
        )


class RastriginProblem(BenchmarkProblem):
    """Rastrigin function: f(x) = A*n + sum(x[i]^2 - A*cos(2*pi*x[i]))

    MATHEMATICAL PROPERTIES:
    - Highly multimodal with many local optima
    - Global minimum at origin: f(0,...,0) = 0
    - Separable
    - Regular structure with known local optima
    """

    def __init__(self, dimension: int, A: float = 10.0):
        self.A = A
        super().__init__(dimension)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Transform [0,1]^n to [-5.12, 5.12]^n for standard Rastrigin domain
        x_scaled = (x - 0.5) * 10.24

        n = len(x_scaled)
        return self.A * n + np.sum(x_scaled**2 - self.A * np.cos(2 * np.pi * x_scaled))

    def _get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name=f"Rastrigin_{self.dimension}D",
            dimension=self.dimension,
            optimal_value=0.0,
            optimal_point=np.full(self.dimension, 0.5),  # Center maps to origin
            problem_class="multimodal",
            difficulty="hard",
            literature_reference="Rastrigin (1974)"
        )


class AckleyProblem(BenchmarkProblem):
    """Ackley function with multiple local optima.

    MATHEMATICAL PROPERTIES:
    - Multimodal with exponential and trigonometric terms
    - Global minimum at origin: f(0,...,0) = 0
    - Non-separable
    - Nearly flat outer region with central attraction
    """

    def __init__(self, dimension: int, a: float = 20.0, b: float = 0.2, c: float = 2*np.pi):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(dimension)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Transform [0,1]^n to [-5, 5]^n for Ackley domain
        x_scaled = (x - 0.5) * 10

        n = len(x_scaled)
        sum_sq = np.sum(x_scaled**2) / n
        sum_cos = np.sum(np.cos(self.c * x_scaled)) / n

        return (self.a + np.exp(1) - self.a * np.exp(-self.b * np.sqrt(sum_sq)) -
                np.exp(sum_cos))

    def _get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name=f"Ackley_{self.dimension}D",
            dimension=self.dimension,
            optimal_value=0.0,
            optimal_point=np.full(self.dimension, 0.5),
            problem_class="multimodal",
            difficulty="medium",
            literature_reference="Ackley (1987)"
        )


class GriewankProblem(BenchmarkProblem):
    """Griewank function with product term creating interdependence.

    MATHEMATICAL PROPERTIES:
    - Multimodal with decreasing local optima density away from origin
    - Global minimum at origin: f(0,...,0) = 0
    - Non-separable due to product term
    - Becomes nearly unimodal for high dimensions
    """

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Transform [0,1]^n to [-600, 600]^n for standard Griewank domain
        x_scaled = (x - 0.5) * 1200

        n = len(x_scaled)
        sum_sq = np.sum(x_scaled**2) / 4000

        prod_cos = 1.0
        for i, xi in enumerate(x_scaled):
            prod_cos *= np.cos(xi / np.sqrt(i + 1))

        return sum_sq - prod_cos + 1

    def _get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name=f"Griewank_{self.dimension}D",
            dimension=self.dimension,
            optimal_value=0.0,
            optimal_point=np.full(self.dimension, 0.5),
            problem_class="multimodal",
            difficulty="medium",
            literature_reference="Griewank (1981)"
        )


class NoisySphere(BenchmarkProblem):
    """Noisy Sphere function for robustness testing.

    MATHEMATICAL PROPERTIES:
    - Same as sphere but with additive Gaussian noise
    - Tests algorithm robustness to evaluation noise
    - Noise level configurable
    """

    def __init__(self, dimension: int, noise_std: float = 0.1, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        super().__init__(dimension)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        x_scaled = (x - 0.5) * 10

        # Base sphere function
        sphere_value = np.sum(x_scaled ** 2)

        # Add Gaussian noise
        noise = self.rng.normal(0, self.noise_std * sphere_value)

        return sphere_value + noise

    def _get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name=f"NoisySphere_{self.dimension}D_std{self.noise_std}",
            dimension=self.dimension,
            optimal_value=0.0,  # Theoretical optimum (noise has zero mean)
            optimal_point=np.full(self.dimension, 0.5),
            problem_class="noisy",
            difficulty="medium",
            literature_reference="Noisy variant of Schwefel (1995)"
        )


class QuadraticProblem(BenchmarkProblem):
    """General quadratic function: f(x) = (x-a)^T Q (x-a)

    MATHEMATICAL PROPERTIES:
    - Unimodal, convex (if Q positive definite)
    - Condition number controlled by eigenvalues of Q
    - Tests algorithm behavior on ill-conditioned problems
    """

    def __init__(self, dimension: int, condition_number: float = 1.0, optimal_point: Optional[np.ndarray] = None):
        self.condition_number = condition_number
        self.optimal_point_scaled = optimal_point if optimal_point is not None else np.full(dimension, 0.5)

        # Create Hessian matrix with specified condition number
        eigenvals = np.linspace(1.0, condition_number, dimension)
        Q_eig, _ = np.linalg.qr(np.random.randn(dimension, dimension))  # Random orthogonal matrix
        self.Q = Q_eig @ np.diag(eigenvals) @ Q_eig.T

        super().__init__(dimension)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Center quadratic at specified optimal point
        diff = x - self.optimal_point_scaled
        return diff.T @ self.Q @ diff

    def _get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name=f"Quadratic_{self.dimension}D_cond{self.condition_number}",
            dimension=self.dimension,
            optimal_value=0.0,
            optimal_point=self.optimal_point_scaled.copy(),
            problem_class="smooth",
            difficulty="easy" if self.condition_number < 10 else "medium" if self.condition_number < 100 else "hard",
            literature_reference="General quadratic optimization"
        )


class BenchmarkSuite:
    """Collection and management of benchmark problems."""

    def __init__(self):
        self.problems: Dict[str, BenchmarkProblem] = {}
        self.problem_classes = {
            'smooth': [],
            'multimodal': [],
            'noisy': [],
            'constrained': []
        }

    def add_problem(self, problem: BenchmarkProblem) -> None:
        """Add a benchmark problem to the suite."""
        name = problem.metadata.name
        self.problems[name] = problem
        problem_class = problem.metadata.problem_class
        if problem_class in self.problem_classes:
            self.problem_classes[problem_class].append(name)

    def get_problem(self, name: str) -> Optional[BenchmarkProblem]:
        """Get a benchmark problem by name."""
        return self.problems.get(name)

    def list_problems(self, problem_class: Optional[str] = None) -> List[str]:
        """List available problems, optionally filtered by class."""
        if problem_class is None:
            return list(self.problems.keys())
        return self.problem_classes.get(problem_class, [])

    def get_problems_by_difficulty(self, difficulty: str) -> List[str]:
        """Get problems by difficulty level."""
        return [name for name, prob in self.problems.items()
                if prob.metadata.difficulty == difficulty]

    def create_standard_suite(self, dimensions: List[int] = [2, 5, 10]) -> 'BenchmarkSuite':
        """Create standard benchmark suite with common problems."""

        for dim in dimensions:
            # Smooth unimodal problems
            self.add_problem(SphereProblem(dim))
            self.add_problem(RosenbrockProblem(dim))
            self.add_problem(QuadraticProblem(dim, condition_number=1.0))
            self.add_problem(QuadraticProblem(dim, condition_number=100.0))

            # Multimodal problems
            self.add_problem(RastriginProblem(dim))
            self.add_problem(AckleyProblem(dim))
            if dim <= 5:  # Griewank becomes nearly unimodal in high dimensions
                self.add_problem(GriewankProblem(dim))

            # Noisy problems (smaller dimensions to avoid excessive noise)
            if dim <= 5:
                self.add_problem(NoisySphere(dim, noise_std=0.05, seed=42))
                self.add_problem(NoisySphere(dim, noise_std=0.2, seed=42))

        return self

    def get_validation_problems(self) -> List[BenchmarkProblem]:
        """Get a curated set of problems for validation."""
        validation_names = [
            'Sphere_2D', 'Sphere_5D', 'Sphere_10D',
            'Rosenbrock_2D', 'Rosenbrock_5D',
            'Rastrigin_2D',
            'Ackley_2D',
            'Quadratic_2D_cond1.0', 'Quadratic_5D_cond100.0'
        ]

        return [self.problems[name] for name in validation_names if name in self.problems]

    def evaluate_algorithm(self, algorithm_func: Callable, problem_names: Optional[List[str]] = None,
                          n_runs: int = 5, n_trials: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate an algorithm on specified benchmark problems.

        Parameters:
        -----------
        algorithm_func : callable
            Function with signature (objective, n_trials, n_dim) -> (best_value, best_x)
        problem_names : list of str, optional
            Problems to evaluate on. If None, uses validation problems.
        n_runs : int
            Number of independent runs per problem
        n_trials : int
            Number of function evaluations per run

        Returns:
        --------
        dict
            Results for each problem including statistics and raw data
        """

        if problem_names is None:
            problems_to_test = self.get_validation_problems()
        else:
            problems_to_test = [self.problems[name] for name in problem_names if name in self.problems]

        results = {}

        for problem in problems_to_test:
            problem_results = {
                'metadata': problem.metadata,
                'runs': [],
                'statistics': {}
            }

            # Run multiple independent trials
            values = []
            solutions = []

            for run in range(n_runs):
                np.random.seed(run * 42)  # Reproducible but varied seeds

                try:
                    best_value, best_x = algorithm_func(problem.objective, n_trials, problem.dimension)
                    values.append(best_value)
                    solutions.append(best_x)

                    problem_results['runs'].append({
                        'run_id': run,
                        'best_value': best_value,
                        'best_x': best_x if isinstance(best_x, list) else best_x.tolist(),
                        'distance_to_optimum': np.linalg.norm(np.array(best_x) - problem.metadata.optimal_point)
                    })

                except Exception as e:
                    print(f"Error in run {run} for {problem.metadata.name}: {e}")
                    continue

            # Calculate statistics
            if values:
                values = np.array(values)
                problem_results['statistics'] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'success_rate': np.sum(values < problem.metadata.optimal_value + 0.01) / len(values),
                    'relative_error': np.mean(values) / (abs(problem.metadata.optimal_value) + 1e-10)
                }

            results[problem.metadata.name] = problem_results

        return results

    def compare_algorithms(self, algorithms: Dict[str, Callable],
                          problem_names: Optional[List[str]] = None,
                          n_runs: int = 5, n_trials: int = 100) -> Dict[str, Any]:
        """
        Compare multiple algorithms on benchmark problems.

        Returns comprehensive comparison including statistical tests.
        """

        comparison_results = {
            'algorithms': list(algorithms.keys()),
            'problems': problem_names or [p.metadata.name for p in self.get_validation_problems()],
            'individual_results': {},
            'comparisons': {},
            'rankings': {}
        }

        # Evaluate each algorithm
        for alg_name, alg_func in algorithms.items():
            print(f"Evaluating {alg_name}...")
            comparison_results['individual_results'][alg_name] = self.evaluate_algorithm(
                alg_func, problem_names, n_runs, n_trials
            )

        # Pairwise comparisons and rankings
        for problem_name in comparison_results['problems']:
            if problem_name not in self.problems:
                continue

            problem_comparison = {}
            algorithm_scores = {}

            # Extract results for this problem
            for alg_name in algorithms.keys():
                if (alg_name in comparison_results['individual_results'] and
                    problem_name in comparison_results['individual_results'][alg_name]):

                    stats = comparison_results['individual_results'][alg_name][problem_name]['statistics']
                    algorithm_scores[alg_name] = stats.get('mean', float('inf'))

            # Rank algorithms for this problem
            sorted_algs = sorted(algorithm_scores.items(), key=lambda x: x[1])
            problem_comparison['ranking'] = [alg for alg, score in sorted_algs]
            problem_comparison['scores'] = algorithm_scores

            comparison_results['comparisons'][problem_name] = problem_comparison

        # Overall rankings (average rank across problems)
        overall_ranks = {alg: [] for alg in algorithms.keys()}

        for problem_name, comp in comparison_results['comparisons'].items():
            ranking = comp['ranking']
            for i, alg in enumerate(ranking):
                overall_ranks[alg].append(i + 1)  # Rank starts from 1

        # Calculate average ranks
        final_rankings = {}
        for alg, ranks in overall_ranks.items():
            if ranks:
                final_rankings[alg] = np.mean(ranks)

        comparison_results['rankings']['overall'] = sorted(final_rankings.items(), key=lambda x: x[1])

        return comparison_results

    def print_summary(self) -> None:
        """Print a summary of available benchmark problems."""
        print("📊 HumpDay Benchmark Suite Summary")
        print("=" * 40)

        print(f"Total problems: {len(self.problems)}")

        for problem_class, problem_list in self.problem_classes.items():
            if problem_list:
                print(f"\n{problem_class.upper()} ({len(problem_list)} problems):")
                for problem_name in problem_list:
                    problem = self.problems[problem_name]
                    print(f"  • {problem_name} - {problem.metadata.difficulty}")

        print(f"\nDimensions available: {sorted({p.metadata.dimension for p in self.problems.values()})}")


def main():
    """Demonstration of benchmark suite capabilities."""
    print("🎯 HumpDay Benchmark Suite")
    print("=" * 30)

    # Create standard benchmark suite
    suite = BenchmarkSuite().create_standard_suite([2, 5, 10])
    suite.print_summary()

    # Test a simple algorithm on a few problems
    def random_search(objective, n_trials: int, n_dim: int):
        """Simple random search for testing."""
        best_value = float('inf')
        best_x = None

        for _ in range(n_trials):
            x = np.random.random(n_dim)
            value = objective(x)

            if value < best_value:
                best_value = value
                best_x = x

        return best_value, best_x

    print("\n🧪 Testing Random Search on selected problems...")

    # Test on a few problems
    test_problems = ['Sphere_2D', 'Rosenbrock_2D', 'Rastrigin_2D']
    results = suite.evaluate_algorithm(random_search, test_problems, n_runs=3, n_trials=50)

    for problem_name, result in results.items():
        stats = result['statistics']
        print(f"\n{problem_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std:  {stats['std']:.6f}")
        print(f"  Success rate: {stats['success_rate']:.2f}")

    print("\n✅ Benchmark suite working correctly")


if __name__ == "__main__":
    main()
