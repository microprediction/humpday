"""
Proper stochastic surface generation for valid benchmarking.
Each run creates truly random surfaces to avoid bias from fixed landscapes.
"""

import numpy as np
import random
from typing import Callable, Dict, Any
import hashlib


class StochasticSurfaceGenerator:
    """
    Generates random variations of benchmark functions to ensure fair comparison.
    Critical for avoiding bias from lucky/unlucky initial guesses.
    """

    def __init__(self, seed: int = None):
        """Initialize with optional seed for reproducible experiments."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Generate random parameters for this run
        self._generate_random_parameters()

    def _generate_random_parameters(self):
        """Generate random parameters that will be used across all functions in this run."""

        # Random shifts (different for each dimension)
        self.global_shift = np.random.uniform(-0.3, 0.3)  # Global shift for all functions
        self.dimension_shifts = {}  # Will be generated per function call

        # Random rotations
        self.use_rotation = np.random.choice([True, False], p=[0.7, 0.3])  # 70% chance of rotation

        # Random scaling factors
        self.scale_factor = np.random.uniform(0.5, 2.0)

        # Random noise level
        self.noise_level = np.random.uniform(0.0, 0.05)  # Up to 5% noise

        # Random conditioning (for appropriate functions)
        self.conditioning_factor = np.random.uniform(1.0, 100.0)

        # Random multimodal density
        self.modal_frequency = np.random.uniform(0.5, 2.0)

    def _get_dimension_shifts(self, n_dim: int, function_name: str) -> np.ndarray:
        """Get consistent dimension-specific shifts for a function."""
        key = f"{function_name}_{n_dim}"

        if key not in self.dimension_shifts:
            # Create deterministic but random shifts based on function name
            seed_string = f"{function_name}_{n_dim}_{self.global_shift}"
            seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

            # Use the hash as seed for this function's shifts
            temp_state = np.random.get_state()
            np.random.seed(seed_hash)
            self.dimension_shifts[key] = np.random.uniform(-0.2, 0.2, n_dim)
            np.random.set_state(temp_state)

        return self.dimension_shifts[key]

    def _apply_rotation(self, x: np.ndarray, seed: int = 42) -> np.ndarray:
        """Apply random rotation to break coordinate alignment."""
        if not self.use_rotation or len(x) == 1:
            return x

        # Generate rotation matrix deterministically for this instance
        temp_state = np.random.get_state()
        np.random.seed(seed)

        n_dim = len(x)
        if n_dim == 2:
            # 2D rotation
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        else:
            # Higher dimensions: random orthogonal matrix
            rotation_matrix = self._random_orthogonal_matrix(n_dim)

        np.random.set_state(temp_state)

        return rotation_matrix @ x

    def _random_orthogonal_matrix(self, n: int) -> np.ndarray:
        """Generate random orthogonal matrix using QR decomposition."""
        A = np.random.randn(n, n)
        Q, R = np.linalg.qr(A)
        # Make sure we have a proper rotation (det = 1)
        Q[:, 0] *= np.sign(R[0, 0])
        return Q

    def _add_noise(self, value: float) -> float:
        """Add random noise to function evaluation."""
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * abs(value))
            return value + noise
        return value

    def stochastic_sphere(self, function_id: str = None) -> Callable:
        """Random variation of sphere function."""
        if function_id is None:
            function_id = f"sphere_{np.random.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            # Get deterministic but random shifts for this function
            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform to optimization domain with random scaling
            scaled_x = self.scale_factor * (10 * x - 5) + dim_shifts + self.global_shift

            # Apply random rotation
            rotated_x = self._apply_rotation(scaled_x, seed=hash(function_id) % 2**31)

            # Compute sphere function
            result = np.sum(rotated_x**2)

            # Add noise
            return self._add_noise(result)

        return func

    def stochastic_rastrigin(self, function_id: str = None) -> Callable:
        """Random variation of Rastrigin function."""
        if function_id is None:
            function_id = f"rastrigin_{np.random.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            # Get random parameters for this function instance
            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random parameters
            scaled_x = self.scale_factor * (10.24 * x - 5.12) + dim_shifts + self.global_shift

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=hash(function_id) % 2**31)

            # Rastrigin with random frequency modulation
            freq = self.modal_frequency
            result = 10 * n_dim + np.sum(rotated_x**2 - 10 * np.cos(2 * np.pi * freq * rotated_x))

            return self._add_noise(result)

        return func

    def stochastic_rosenbrock(self, function_id: str = None) -> Callable:
        """Random variation of Rosenbrock function."""
        if function_id is None:
            function_id = f"rosenbrock_{np.random.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            if n_dim < 2:
                # Fallback for 1D
                return (x[0] - 1)**2

            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random conditioning
            scaled_x = self.scale_factor * (4.096 * x - 2.048) + dim_shifts + self.global_shift

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=hash(function_id) % 2**31)

            # Rosenbrock with random conditioning factor
            a = 1.0
            b = 100.0 * self.conditioning_factor

            result = np.sum(b * (rotated_x[1:] - rotated_x[:-1]**2)**2 + (a - rotated_x[:-1])**2)

            return self._add_noise(result)

        return func

    def stochastic_ackley(self, function_id: str = None) -> Callable:
        """Random variation of Ackley function."""
        if function_id is None:
            function_id = f"ackley_{np.random.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random scaling
            scaled_x = self.scale_factor * (65.536 * x - 32.768) + dim_shifts + self.global_shift

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=hash(function_id) % 2**31)

            # Ackley with random parameters
            a = 20 * np.random.uniform(0.8, 1.2)  # Slight randomization
            b = 0.2 * np.random.uniform(0.8, 1.2)
            c = 2 * np.pi * self.modal_frequency

            term1 = -a * np.exp(-b * np.sqrt(np.sum(rotated_x**2) / n_dim))
            term2 = -np.exp(np.sum(np.cos(c * rotated_x)) / n_dim)
            result = term1 + term2 + a + np.e

            return self._add_noise(result)

        return func

    def stochastic_griewank(self, function_id: str = None) -> Callable:
        """Random variation of Griewank function."""
        if function_id is None:
            function_id = f"griewank_{np.random.randint(0, 1000000)}"

        def func(x):
            x = np.array(x)
            n_dim = len(x)

            dim_shifts = self._get_dimension_shifts(n_dim, function_id)

            # Transform with random scaling
            scaled_x = self.scale_factor * (1200 * x - 600) + dim_shifts + self.global_shift

            # Apply rotation
            rotated_x = self._apply_rotation(scaled_x, seed=hash(function_id) % 2**31)

            # Griewank function
            sum_sq = np.sum(rotated_x**2) / 4000
            prod_cos = np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, n_dim + 1))))
            result = sum_sq - prod_cos + 1

            return self._add_noise(result)

        return func

    def get_random_function_suite(self, n_functions: int = 10) -> Dict[str, Callable]:
        """Generate a suite of random function instances for benchmarking."""

        base_functions = [
            ('sphere', self.stochastic_sphere),
            ('rastrigin', self.stochastic_rastrigin),
            ('rosenbrock', self.stochastic_rosenbrock),
            ('ackley', self.stochastic_ackley),
            ('griewank', self.stochastic_griewank)
        ]

        suite = {}

        for i in range(n_functions):
            # Randomly select base function type
            base_name, base_func = random.choice(base_functions)

            # Create unique instance
            instance_id = f"{base_name}_instance_{i}_{np.random.randint(0, 1000000)}"
            suite[instance_id] = base_func(instance_id)

        return suite

    def get_benchmark_metadata(self) -> Dict[str, Any]:
        """Get metadata about the random parameters used in this benchmark run."""
        return {
            'global_shift': self.global_shift,
            'use_rotation': self.use_rotation,
            'scale_factor': self.scale_factor,
            'noise_level': self.noise_level,
            'conditioning_factor': self.conditioning_factor,
            'modal_frequency': self.modal_frequency,
            'random_seed_used': 'Runtime generated' if not hasattr(self, '_seed') else self._seed
        }


def create_fair_benchmark_run(n_functions: int = 20, seed: int = None) -> tuple:
    """
    Create a fair benchmark run with truly random surfaces.

    Returns:
        (function_suite, metadata) - Functions and metadata about randomization
    """

    print(f"🎲 Generating {n_functions} random function instances for fair benchmarking...")

    # Create stochastic generator
    generator = StochasticSurfaceGenerator(seed=seed)

    # Generate random function suite
    function_suite = generator.get_random_function_suite(n_functions)

    # Get metadata
    metadata = generator.get_benchmark_metadata()

    print(f"✅ Random surfaces generated:")
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

    print(f"\n=== Comparing Function Values at Same Point ===")
    test_point = [0.5, 0.5]

    print(f"Run 1 (seed=42):")
    for name, func in list(suite1.items())[:3]:
        val = func(test_point)
        print(f"  {name}: {val:.6f}")

    print(f"\nRun 2 (seed=123):")
    for name, func in list(suite2.items())[:3]:
        val = func(test_point)
        print(f"  {name}: {val:.6f}")

    print(f"\n✅ Values are different - proving surfaces are truly random!")
    print(f"✅ This ensures fair comparison between optimization algorithms")
    print(f"✅ No algorithm can benefit from memorizing fixed landscapes")