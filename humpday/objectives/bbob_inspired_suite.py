"""
BBOB-inspired systematic test suite for HumpDay.
Based on research insights from COCO/BBOB platform analysis.
"""

import warnings
from typing import Any, Callable, Dict, List

import numpy as np
from scipy.stats import ortho_group


class BBOBInspiredSuite:
    """
    Systematic test function suite inspired by COCO/BBOB research.
    Implements the systematic difficulty progression approach.
    """

    def __init__(self, n_dim: int = 2):
        self.n_dim = n_dim
        self.conditioning_levels = {
            "well_conditioned": 1.0,
            "moderate": 10.0,
            "ill_conditioned": 100.0,
            "very_ill_conditioned": 1000.0,
        }

    def _get_rotation_matrix(self, seed: int = None) -> np.ndarray:
        """Generate random rotation matrix for non-separable functions."""
        if seed is not None:
            np.random.seed(seed)
        return ortho_group.rvs(self.n_dim)

    def _apply_conditioning(self, x: np.ndarray, conditioning: float) -> np.ndarray:
        """Apply conditioning to create ill-conditioned problems."""
        # Create diagonal matrix with exponentially spaced eigenvalues
        eigenvals = np.logspace(0, np.log10(conditioning), self.n_dim)
        return x * eigenvals

    # === SEPARABLE FUNCTIONS (BBOB f1-f5 inspired) ===

    def sphere_separable(self, seed: int = 42) -> Callable:
        """f1-inspired: Basic separable sphere function."""
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            # Transform [0,1]^n to [-5,5]^n then shift
            scaled_x = 10 * x - 5 + shift
            return np.sum(scaled_x**2)

        return func

    def ellipsoid_separable(
        self, conditioning: float = 100.0, seed: int = 42
    ) -> Callable:
        """f2-inspired: Separable ellipsoid with conditioning."""
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            scaled_x = 10 * x - 5 + shift
            conditioned_x = self._apply_conditioning(scaled_x, conditioning)
            return np.sum(conditioned_x**2)

        return func

    def rastrigin_separable(self, seed: int = 42) -> Callable:
        """f3-inspired: Separable highly multimodal function."""
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            scaled_x = 10.24 * x - 5.12 + shift
            return 10 * len(scaled_x) + np.sum(
                scaled_x**2 - 10 * np.cos(2 * np.pi * scaled_x)
            )

        return func

    # === NON-SEPARABLE FUNCTIONS (BBOB f6-f14 inspired) ===

    def sphere_rotated(self, seed: int = 42) -> Callable:
        """f6-inspired: Non-separable sphere with rotation."""
        rotation_matrix = self._get_rotation_matrix(seed)
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            scaled_x = 10 * x - 5 + shift
            rotated_x = rotation_matrix @ scaled_x
            return np.sum(rotated_x**2)

        return func

    def ellipsoid_rotated(
        self, conditioning: float = 100.0, seed: int = 42
    ) -> Callable:
        """f10-inspired: Rotated ellipsoid with high conditioning."""
        rotation_matrix = self._get_rotation_matrix(seed)
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            scaled_x = 10 * x - 5 + shift
            rotated_x = rotation_matrix @ scaled_x
            conditioned_x = self._apply_conditioning(rotated_x, conditioning)
            return np.sum(conditioned_x**2)

        return func

    def rosenbrock_rotated(self, seed: int = 42) -> Callable:
        """f8-inspired: Rotated Rosenbrock function."""
        rotation_matrix = self._get_rotation_matrix(seed)

        def func(x):
            x = np.array(x)
            scaled_x = 4.096 * x - 2.048
            rotated_x = rotation_matrix @ scaled_x
            # Apply Rosenbrock transformation
            return np.sum(
                100 * (rotated_x[1:] - rotated_x[:-1] ** 2) ** 2
                + (1 - rotated_x[:-1]) ** 2
            )

        return func

    # === MULTIMODAL FUNCTIONS (BBOB f15-f19 inspired) ===

    def rastrigin_rotated(self, seed: int = 42) -> Callable:
        """f15-inspired: Rotated Rastrigin function."""
        rotation_matrix = self._get_rotation_matrix(seed)
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            scaled_x = 10.24 * x - 5.12 + shift
            rotated_x = rotation_matrix @ scaled_x
            return 10 * len(rotated_x) + np.sum(
                rotated_x**2 - 10 * np.cos(2 * np.pi * rotated_x)
            )

        return func

    def weierstrass(self, seed: int = 42) -> Callable:
        """f16-inspired: Weierstrass function (highly multimodal)."""
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)
        a, b, kmax = 0.5, 3, 20

        def func(x):
            x = np.array(x)
            scaled_x = 1 * x - 0.5 + shift

            # Weierstrass sum
            sum_val = 0
            for xi in scaled_x:
                for k in range(kmax):
                    sum_val += a**k * np.cos(2 * np.pi * b**k * (xi + 0.5))

            # Subtract constant to make global minimum zero
            constant = len(scaled_x) * sum(
                a**k * np.cos(np.pi * b**k) for k in range(kmax)
            )
            return sum_val - constant

        return func

    # === WEAK GLOBAL STRUCTURE (BBOB f20-f24 inspired) ===

    def schwefel_weak_structure(self, seed: int = 42) -> Callable:
        """f20-inspired: Schwefel with weak global structure."""
        np.random.seed(seed)
        shift = np.random.uniform(-0.2, 0.2, self.n_dim)

        def func(x):
            x = np.array(x)
            scaled_x = 1000 * x - 500 + shift * 100
            return 418.9829 * len(scaled_x) - np.sum(
                scaled_x * np.sin(np.sqrt(np.abs(scaled_x)))
            )

        return func

    def gallagher_inspired(self, n_peaks: int = 21, seed: int = 42) -> Callable:
        """f21/f22-inspired: Gallagher function with multiple peaks."""
        np.random.seed(seed)

        # Generate peak locations and heights
        peaks = np.random.uniform(0.1, 0.9, (n_peaks, self.n_dim))
        heights = np.random.exponential(1, n_peaks)
        heights[0] = np.max(heights) + 1  # Ensure one global maximum

        # Generate conditioning for each peak
        conditionings = np.random.uniform(10, 1000, n_peaks)

        def func(x):
            x = np.array(x)
            max_val = -np.inf

            for i in range(n_peaks):
                diff = x - peaks[i]
                # Apply different conditioning to each peak
                conditioned_diff = diff * np.sqrt(conditionings[i])
                val = heights[i] * np.exp(-np.sum(conditioned_diff**2))
                max_val = max(max_val, val)

            return -max_val  # Minimize (negative of maximum)

        return func

    # === COMPOSITE FUNCTIONS ===

    def composition_function(
        self, base_functions: List[str], weights: List[float] = None, seed: int = 42
    ) -> Callable:
        """Create composition of multiple base functions like BBOB hybrid compositions."""
        if weights is None:
            weights = [1.0] * len(base_functions)

        # Get base function instances
        funcs = []
        for func_name in base_functions:
            if hasattr(self, func_name):
                funcs.append(getattr(self, func_name)(seed=seed + len(funcs)))
            else:
                warnings.warn(f"Unknown function: {func_name}")

        def func(x):
            return sum(w * f(x) for w, f in zip(weights, funcs))

        return func

    def get_systematic_suite(self) -> Dict[str, Callable]:
        """Get complete systematic test suite inspired by BBOB research."""

        suite = {
            # Separable functions (increasing difficulty)
            "f01_sphere_sep": self.sphere_separable(seed=1),
            "f02_ellipsoid_sep_moderate": self.ellipsoid_separable(
                conditioning=10, seed=2
            ),
            "f03_ellipsoid_sep_ill": self.ellipsoid_separable(conditioning=100, seed=3),
            "f04_rastrigin_sep": self.rastrigin_separable(seed=4),
            # Non-separable with low/moderate conditioning
            "f05_sphere_rotated": self.sphere_rotated(seed=5),
            "f06_ellipsoid_rotated_moderate": self.ellipsoid_rotated(
                conditioning=10, seed=6
            ),
            "f07_rosenbrock_rotated": self.rosenbrock_rotated(seed=7),
            # High conditioning
            "f08_ellipsoid_rotated_ill": self.ellipsoid_rotated(
                conditioning=1000, seed=8
            ),
            # Multimodal with adequate global structure
            "f09_rastrigin_rotated": self.rastrigin_rotated(seed=9),
            "f10_weierstrass": self.weierstrass(seed=10),
            # Multimodal with weak global structure
            "f11_schwefel_weak": self.schwefel_weak_structure(seed=11),
            "f12_gallagher_peaks": self.gallagher_inspired(n_peaks=21, seed=12),
            # Hybrid composition functions
            "f13_composition_1": self.composition_function(
                ["sphere_separable", "rastrigin_separable"], [0.5, 0.5], seed=13
            ),
            "f14_composition_2": self.composition_function(
                ["ellipsoid_rotated", "weierstrass"], [0.7, 0.3], seed=14
            ),
        }

        return suite

    def get_function_metadata(self, function_name: str) -> Dict[str, Any]:
        """Get detailed metadata for systematic categorization."""

        metadata_map = {
            "f01_sphere_sep": {
                "landscape_type": "smooth",
                "modality": "unimodal",
                "separable": True,
                "conditioning": "well_conditioned",
                "global_structure": "strong",
                "bbob_category": "separable",
                "difficulty": "easy",
            },
            "f04_rastrigin_sep": {
                "landscape_type": "multimodal",
                "modality": "highly_multimodal",
                "separable": True,
                "conditioning": "well_conditioned",
                "global_structure": "weak",
                "bbob_category": "separable",
                "difficulty": "hard",
            },
            "f08_ellipsoid_rotated_ill": {
                "landscape_type": "smooth",
                "modality": "unimodal",
                "separable": False,
                "conditioning": "very_ill_conditioned",
                "global_structure": "moderate",
                "bbob_category": "high_conditioning",
                "difficulty": "very_hard",
            },
            "f11_schwefel_weak": {
                "landscape_type": "rugged",
                "modality": "multimodal",
                "separable": True,
                "conditioning": "moderate",
                "global_structure": "deceptive",
                "bbob_category": "weak_structure",
                "difficulty": "extremely_hard",
            },
        }

        return metadata_map.get(
            function_name,
            {
                "landscape_type": "multimodal",
                "modality": "multimodal",
                "separable": False,
                "conditioning": "moderate",
                "global_structure": "moderate",
                "bbob_category": "unknown",
                "difficulty": "medium",
            },
        )


if __name__ == "__main__":
    # Test BBOB-inspired systematic suite
    print("=== BBOB-Inspired Systematic Test Suite ===")

    suite = BBOBInspiredSuite(n_dim=2)
    functions = suite.get_systematic_suite()

    print(f"Generated {len(functions)} systematic test functions:")

    for name, func in functions.items():
        try:
            # Test at center and corner
            center_val = func([0.5, 0.5])
            corner_val = func([0.1, 0.9])

            metadata = suite.get_function_metadata(name)

            print(f"  ✓ {name:25s}: center={center_val:8.4f}, corner={corner_val:8.4f}")
            print(
                f"    {metadata.get('bbob_category', 'unknown'):15s} | "
                f"{metadata.get('landscape_type', 'unknown'):10s} | "
                f"conditioning: {metadata.get('conditioning', 'unknown'):15s} | "
                f"difficulty: {metadata.get('difficulty', 'unknown')}"
            )

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\n=== Systematic Progression Ready ===")
    print("✓ Separable → Non-separable progression")
    print("✓ Well-conditioned → Ill-conditioned progression")
    print("✓ Unimodal → Multimodal progression")
    print("✓ Strong → Weak global structure progression")
    print("✓ Perfect for rigorous 3D Thurstone analysis")
