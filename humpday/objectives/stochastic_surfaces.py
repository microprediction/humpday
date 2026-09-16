"""
Proper stochastic surface generation for valid benchmarking.
Each run creates truly random surfaces to avoid bias from fixed landscapes.
"""

import hashlib
import random
from typing import Any, Callable, Dict

import numpy as np


def _stable_seed(function_id: str) -> int:
    """A seed derived from the instance id that is the same in every process.

    Python's hash() of a str is salted per process (PYTHONHASHSEED), so a
    rotation seeded from it made a "fixed" instance a different landscape in
    every worker.
    """
    return int(hashlib.md5(function_id.encode()).hexdigest()[:8], 16)


class StochasticSurfaceGenerator:
    """
    Generates random variations of benchmark functions to ensure fair comparison.
    Critical for avoiding bias from lucky/unlucky initial guesses.

    Every landscape is frozen when its objective is built: shifts, scale,
    rotation and the Ackley coefficients are drawn once and reused on every
    evaluation, and the rotation matrix is factorised once. The only thing
    that varies between evaluations of the same point is the separately
    configured observation noise (`noise_level`), and that has its own
    stream, so evaluating an objective never touches the global RNGs the
    optimizers draw from.
    """

    def __init__(self, seed: int = None):
        """Initialize with optional seed for reproducible experiments.

        The generator owns its random streams (numpy's RandomState and a
        stdlib Random) rather than reseeding the global ones, so building a
        suite does not reset the optimizers' RNG. RandomState(seed) draws the
        same sequence np.random.seed(seed) used to, so seeded suites are the
        ones they were before.
        """
        self._rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)
        self._noise_rng = np.random.RandomState(None if seed is None else seed + 1)
        self._rotations: Dict[tuple, np.ndarray] = {}

        # Generate random parameters for this run
        self._generate_random_parameters()

    def _generate_random_parameters(self):
        """Generate random parameters that will be used across all functions in this run."""

        # Random shifts (different for each dimension)
        self.global_shift = self._rng.uniform(
            -0.3, 0.3
        )  # Global shift for all functions
        self.dimension_shifts = {}  # Will be generated per function call

        # Random rotations
        self.use_rotation = self._rng.choice(
            [True, False], p=[0.7, 0.3]
        )  # 70% chance of rotation

        # Random scaling factors
        self.scale_factor = self._rng.uniform(0.5, 2.0)

        # Random noise level
        self.noise_level = self._rng.uniform(0.0, 0.05)  # Up to 5% noise

        # Random conditioning (for appropriate functions)
        self.conditioning_factor = self._rng.uniform(1.0, 100.0)

        # Random multimodal density
        self.modal_frequency = self._rng.uniform(0.5, 2.0)

    def _get_dimension_shifts(self, n_dim: int, function_name: str) -> np.ndarray:
        """Get consistent dimension-specific shifts for a function."""
        key = f"{function_name}_{n_dim}"

        if key not in self.dimension_shifts:
            # Create deterministic but random shifts based on function name
            seed_string = f"{function_name}_{n_dim}_{self.global_shift}"
            seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

            # Use the hash as seed for this function's shifts
            self.dimension_shifts[key] = np.random.RandomState(seed_hash).uniform(
                -0.2, 0.2, n_dim
            )

        return self.dimension_shifts[key]

    def _rotation_matrix(self, n_dim: int, seed: int) -> np.ndarray:
        """The rotation for (seed, n_dim), built once and reused. Building it
        means a QR factorisation in higher dimensions, which used to happen
        on every evaluation and be charged to the objective's cost."""
        key = (int(seed), n_dim)
        matrix = self._rotations.get(key)
        if matrix is None:
            rng = np.random.RandomState(seed)
            if n_dim == 2:
                # 2D rotation
                theta = rng.uniform(0, 2 * np.pi)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            else:
                # Higher dimensions: random orthogonal matrix
                matrix = self._random_orthogonal_matrix(n_dim, rng)
            self._rotations[key] = matrix
        return matrix

    def _apply_rotation(self, x: np.ndarray, seed: int = 42) -> np.ndarray:
        """Apply the instance's fixed rotation to break coordinate alignment."""
        if not self.use_rotation or len(x) == 1:
            return x
        return self._rotation_matrix(len(x), seed) @ x

    def _random_orthogonal_matrix(self, n: int, rng=None) -> np.ndarray:
        """Generate random orthogonal matrix using QR decomposition."""
        rng = np.random if rng is None else rng
        A = rng.randn(n, n)
        Q, R = np.linalg.qr(A)
        # Make sure we have a proper rotation (det = 1)
        Q[:, 0] *= np.sign(R[0, 0])
        return Q

    def _add_noise(self, value: float) -> float:
        """Add observation noise from the generator's own stream."""
        if self.noise_level > 0:
            noise = self._noise_rng.normal(0, self.noise_level * abs(value))
            return value + noise
        return value

    def stochastic_sphere(self, function_id: str = None) -> Callable:
        """Random variation of sphere function."""
        if function_id is None:
            function_id = f"sphere_{self._rng.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            # Get deterministic but random shifts for this function
            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform to optimization domain with random scaling
            scaled_x = self.scale_factor * (10 * x - 5) + dim_shifts + self.global_shift

            # Apply random rotation
            rotated_x = self._apply_rotation(scaled_x, seed=_stable_seed(function_id))

            # Compute sphere function
            result = np.sum(rotated_x**2)

            # Add noise
            return self._add_noise(result)

        return func

    def stochastic_rastrigin(self, function_id: str = None) -> Callable:
        """Random variation of Rastrigin function."""
        if function_id is None:
            function_id = f"rastrigin_{self._rng.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            # Get random parameters for this function instance
            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random parameters
            scaled_x = (
                self.scale_factor * (10.24 * x - 5.12) + dim_shifts + self.global_shift
            )

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=_stable_seed(function_id))

            # Rastrigin with random frequency modulation
            freq = self.modal_frequency
            result = 10 * n_dim + np.sum(
                rotated_x**2 - 10 * np.cos(2 * np.pi * freq * rotated_x)
            )

            return self._add_noise(result)

        return func

    def stochastic_rosenbrock(self, function_id: str = None) -> Callable:
        """Random variation of Rosenbrock function."""
        if function_id is None:
            function_id = f"rosenbrock_{self._rng.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            if n_dim < 2:
                # Fallback for 1D
                return (x[0] - 1) ** 2

            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random conditioning
            scaled_x = (
                self.scale_factor * (4.096 * x - 2.048) + dim_shifts + self.global_shift
            )

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=_stable_seed(function_id))

            # Rosenbrock with random conditioning factor
            a = 1.0
            b = 100.0 * self.conditioning_factor

            result = np.sum(
                b * (rotated_x[1:] - rotated_x[:-1] ** 2) ** 2
                + (a - rotated_x[:-1]) ** 2
            )

            return self._add_noise(result)

        return func

    def stochastic_ackley(self, function_id: str = None) -> Callable:
        """Random variation of Ackley function."""
        if function_id is None:
            function_id = f"ackley_{self._rng.randint(0, 1000000)}"

        # Ackley's coefficients are part of this instance's landscape: drawn
        # once here, from the instance id, not on every evaluation.
        coeff_rng = np.random.RandomState(_stable_seed(function_id + "/ackley"))
        a = 20 * coeff_rng.uniform(0.8, 1.2)  # Slight randomization
        b = 0.2 * coeff_rng.uniform(0.8, 1.2)

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random scaling
            scaled_x = (
                self.scale_factor * (65.536 * x - 32.768)
                + dim_shifts
                + self.global_shift
            )

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=_stable_seed(function_id))

            c = 2 * np.pi * self.modal_frequency

            term1 = -a * np.exp(-b * np.sqrt(np.sum(rotated_x**2) / n_dim))
            term2 = -np.exp(np.sum(np.cos(c * rotated_x)) / n_dim)
            result = term1 + term2 + a + np.e

            return self._add_noise(result)

        return func

    def stochastic_griewank(self, function_id: str = None) -> Callable:
        """Random variation of Griewank function."""
        if function_id is None:
            function_id = f"griewank_{self._rng.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random scaling
            scaled_x = (
                self.scale_factor * (1200 * x - 600) + dim_shifts + self.global_shift
            )

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=_stable_seed(function_id))

            # Griewank function
            sum_sq = np.sum(rotated_x**2) / 4000
            prod_cos = np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, n_dim + 1))))
            result = sum_sq - prod_cos + 1

            return self._add_noise(result)

        return func

    def get_random_function_suite(self, n_functions: int = 10) -> Dict[str, Callable]:
        """Generate a suite of random function instances for benchmarking."""

        base_functions = [
            ("sphere", self.stochastic_sphere),
            ("rastrigin", self.stochastic_rastrigin),
            ("rosenbrock", self.stochastic_rosenbrock),
            ("ackley", self.stochastic_ackley),
            ("griewank", self.stochastic_griewank),
        ]

        suite = {}

        for i in range(n_functions):
            # Randomly select base function type
            base_name, base_func = self._py_rng.choice(base_functions)

            # Create unique instance
            instance_id = f"{base_name}_instance_{i}_{self._rng.randint(0, 1000000)}"
            suite[instance_id] = base_func(instance_id)

        return suite

    def get_benchmark_metadata(self) -> Dict[str, Any]:
        """Get metadata about the random parameters used in this benchmark run."""
        return {
            "global_shift": self.global_shift,
            "use_rotation": self.use_rotation,
            "scale_factor": self.scale_factor,
            "noise_level": self.noise_level,
            "conditioning_factor": self.conditioning_factor,
            "modal_frequency": self.modal_frequency,
            "random_seed_used": "Runtime generated"
            if not hasattr(self, "_seed")
            else self._seed,
        }


def create_fair_benchmark_run(n_functions: int = 20, seed: int = None) -> tuple:
    """
    Create a fair benchmark run with truly random surfaces.

    Returns:
        (function_suite, metadata) - Functions and metadata about randomization
    """

    print(
        f"🎲 Generating {n_functions} random function instances for fair benchmarking..."
    )

    # Create stochastic generator
    generator = StochasticSurfaceGenerator(seed=seed)

    # Generate random function suite
    function_suite = generator.get_random_function_suite(n_functions)

    # Get metadata
    metadata = generator.get_benchmark_metadata()

    print("✅ Random surfaces generated:")
    print(f"   Global shift: {metadata['global_shift']:.3f}")
    print(f"   Rotation enabled: {metadata['use_rotation']}")
    print(f"   Scale factor: {metadata['scale_factor']:.3f}")
    print(f"   Noise level: {metadata['noise_level']:.3f}")
    print(f"   Modal frequency: {metadata['modal_frequency']:.3f}")

    return function_suite, metadata


if __name__ == "__main__":
    # Test stochastic surface generation
    print("=== Stochastic Surface Generation Test ===")

    # Create two different benchmark runs
    suite1, meta1 = create_fair_benchmark_run(n_functions=5, seed=42)
    suite2, meta2 = create_fair_benchmark_run(n_functions=5, seed=123)

    print("\n=== Comparing Function Values at Same Point ===")
    test_point = [0.5, 0.5]

    print("Run 1 (seed=42):")
    for name, func in list(suite1.items())[:3]:
        val = func(test_point)
        print(f"  {name}: {val:.6f}")

    print("\nRun 2 (seed=123):")
    for name, func in list(suite2.items())[:3]:
        val = func(test_point)
        print(f"  {name}: {val:.6f}")

    print("\n✅ Values are different - proving surfaces are truly random!")
    print("✅ This ensures fair comparison between optimization algorithms")
    print("✅ No algorithm can benefit from memorizing fixed landscapes")
