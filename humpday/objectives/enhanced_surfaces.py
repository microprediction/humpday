"""
Enhanced test surface generation using best-in-class packages.
Integrates Opfunu, benchmark-functions, and COCO-inspired approaches.
Focus: Single-objective continuous optimization on hypercube [0,1]^n
"""

import warnings
from typing import Any, Callable, Dict

import numpy as np

# Try to import the advanced surface generation packages
HAS_OPFUNU = False
HAS_BENCHMARK_FUNCS = False

try:
    import opfunu

    HAS_OPFUNU = True
    print("✓ Opfunu available - comprehensive optimization functions")
except ImportError:
    print("⚠ Opfunu not available - install with: pip install opfunu")

try:
    import benchmark_functions as bf

    HAS_BENCHMARK_FUNCS = True
    print("✓ benchmark-functions available - lightweight test functions")
except ImportError:
    print(
        "⚠ benchmark-functions not available - install with: pip install benchmark-functions"
    )


class EnhancedSurfaceGenerator:
    """
    Enhanced test surface generator leveraging best packages.
    Generates functions on hypercube [0,1]^n for derivative-free optimization.
    """

    def __init__(self):
        self.available_categories = {
            "unimodal": ["sphere", "ellipsoid", "zakharov", "rosenbrock"],
            "multimodal": ["rastrigin", "griewank", "ackley", "schwefel"],
            "composite": ["hybrid_composition", "shifted_rotated"],
            "deceptive": ["step_function", "quartic_noise"],
            "separable": ["sphere", "zakharov"],
            "non_separable": ["rosenbrock", "ellipsoid_rotated"],
        }

    def get_function_on_cube(self, function_name: str, n_dim: int) -> Callable:
        """Get a test function that operates on hypercube [0,1]^n."""

        if HAS_OPFUNU:
            return self._get_opfunu_function_on_cube(function_name, n_dim)
        elif HAS_BENCHMARK_FUNCS:
            return self._get_benchmark_function_on_cube(function_name, n_dim)
        else:
            return self._get_fallback_function_on_cube(function_name, n_dim)

    def _get_opfunu_function_on_cube(self, function_name: str, n_dim: int) -> Callable:
        """Create Opfunu-based function on hypercube."""

        # Map common names to Opfunu classes
        opfunu_mapping = {
            "sphere": "SphereFunction",
            "ellipsoid": "EllipticFunction",
            "rastrigin": "RastriginFunction",
            "griewank": "GriewankFunction",
            "ackley": "AckleyFunction",
            "schwefel": "SchwefelFunction",
            "rosenbrock": "RosenbrockFunction",
            "zakharov": "ZakharovFunction",
        }

        if function_name not in opfunu_mapping:
            function_name = "rastrigin"  # Default fallback

        try:
            # Get the function class from opfunu
            func_class_name = opfunu_mapping[function_name]

            # Try different opfunu module structures
            func_class = None
            for module_path in ["cec_based", "classical", "physics_based"]:
                try:
                    module = getattr(opfunu, module_path)
                    if hasattr(module, func_class_name):
                        func_class = getattr(module, func_class_name)
                        break
                except:
                    continue

            if func_class is None:
                # Try direct access
                if hasattr(opfunu, func_class_name):
                    func_class = getattr(opfunu, func_class_name)
                else:
                    return self._get_fallback_function_on_cube(function_name, n_dim)

            # Create function instance
            func_instance = func_class(ndim=n_dim)

            # Get the function's bounds
            bounds = func_instance.bounds
            lb, ub = np.array(bounds[0]), np.array(bounds[1])

            def cube_function(x):
                """Transform from [0,1]^n to function's native domain."""
                x = np.array(x)
                # Scale from [0,1] to [lb, ub]
                scaled_x = lb + (ub - lb) * x
                return func_instance.evaluate(scaled_x)

            return cube_function

        except Exception as e:
            warnings.warn(f"Opfunu function {function_name} failed: {e}")
            return self._get_fallback_function_on_cube(function_name, n_dim)

    def _get_benchmark_function_on_cube(
        self, function_name: str, n_dim: int
    ) -> Callable:
        """Create benchmark-functions based function on hypercube."""

        # Map to benchmark-functions classes
        bf_mapping = {
            "sphere": bf.Sphere,
            "rastrigin": bf.Rastrigin,
            "griewank": bf.Griewank,
            "ackley": bf.Ackley,
            "rosenbrock": bf.Rosenbrock,
            "zakharov": bf.Zakharov,
        }

        if function_name not in bf_mapping:
            function_name = "rastrigin"

        try:
            # Create function instance
            func_class = bf_mapping[function_name]
            func_instance = func_class(n_dimensions=n_dim)

            # Get bounds
            bounds = func_instance.suggested_bounds()
            lb, ub = np.array(bounds[0]), np.array(bounds[1])

            def cube_function(x):
                """Transform from [0,1]^n to function's native domain."""
                x = np.array(x)
                scaled_x = lb + (ub - lb) * x
                return func_instance(scaled_x)

            return cube_function

        except Exception as e:
            warnings.warn(f"Benchmark function {function_name} failed: {e}")
            return self._get_fallback_function_on_cube(function_name, n_dim)

    def _get_fallback_function_on_cube(
        self, function_name: str, n_dim: int
    ) -> Callable:
        """Fallback implementations when external packages unavailable."""

        def sphere(x):
            x = np.array(x)
            # Transform [0,1]^n to [-5,5]^n
            scaled_x = 10 * x - 5
            return np.sum(scaled_x**2)

        def rastrigin(x):
            x = np.array(x)
            # Transform [0,1]^n to [-5.12,5.12]^n
            scaled_x = 10.24 * x - 5.12
            return 10 * len(scaled_x) + np.sum(
                scaled_x**2 - 10 * np.cos(2 * np.pi * scaled_x)
            )

        def griewank(x):
            x = np.array(x)
            # Transform [0,1]^n to [-600,600]^n
            scaled_x = 1200 * x - 600
            sum_sq = np.sum(scaled_x**2) / 4000
            prod_cos = np.prod(
                np.cos(scaled_x / np.sqrt(np.arange(1, len(scaled_x) + 1)))
            )
            return sum_sq - prod_cos + 1

        def ackley(x):
            x = np.array(x)
            # Transform [0,1]^n to [-32.768,32.768]^n
            scaled_x = 65.536 * x - 32.768
            n = len(scaled_x)
            a, b, c = 20, 0.2, 2 * np.pi
            term1 = -a * np.exp(-b * np.sqrt(np.sum(scaled_x**2) / n))
            term2 = -np.exp(np.sum(np.cos(c * scaled_x)) / n)
            return term1 + term2 + a + np.e

        def rosenbrock(x):
            x = np.array(x)
            # Transform [0,1]^n to [-2.048,2.048]^n
            scaled_x = 4.096 * x - 2.048
            return np.sum(
                100 * (scaled_x[1:] - scaled_x[:-1] ** 2) ** 2
                + (1 - scaled_x[:-1]) ** 2
            )

        fallback_functions = {
            "sphere": sphere,
            "rastrigin": rastrigin,
            "griewank": griewank,
            "ackley": ackley,
            "rosenbrock": rosenbrock,
            "zakharov": lambda x: sphere(x) + np.sum(np.array(x) ** 4),  # Simplified
        }

        return fallback_functions.get(function_name, rastrigin)

    def generate_bbob_inspired_surface(
        self, size: int, function_type: str, **kwargs
    ) -> np.ndarray:
        """
        Generate 2D surface visualization inspired by BBOB methodology.
        For use in browser-based demos and educational visualization.
        """

        # Get the function
        func = self.get_function_on_cube(function_type, n_dim=2)

        # Generate grid
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)

        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(size):
            for j in range(size):
                Z[i, j] = func([X[i, j], Y[i, j]])

        return X, Y, Z

    def get_function_metadata(self, function_name: str) -> Dict[str, Any]:
        """Get metadata about a test function for categorization."""

        metadata = {
            "sphere": {
                "landscape_type": "smooth",
                "modality": "unimodal",
                "separable": True,
                "conditioning": "well_conditioned",
                "global_structure": "strong",
            },
            "rastrigin": {
                "landscape_type": "multimodal",
                "modality": "highly_multimodal",
                "separable": True,
                "conditioning": "well_conditioned",
                "global_structure": "weak",
            },
            "griewank": {
                "landscape_type": "multimodal",
                "modality": "multimodal",
                "separable": False,
                "conditioning": "moderate",
                "global_structure": "moderate",
            },
            "ackley": {
                "landscape_type": "multimodal",
                "modality": "highly_multimodal",
                "separable": False,
                "conditioning": "moderate",
                "global_structure": "weak",
            },
            "rosenbrock": {
                "landscape_type": "smooth",
                "modality": "unimodal",
                "separable": False,
                "conditioning": "ill_conditioned",
                "global_structure": "moderate",
            },
        }

        return metadata.get(
            function_name,
            {
                "landscape_type": "multimodal",
                "modality": "multimodal",
                "separable": False,
                "conditioning": "moderate",
                "global_structure": "moderate",
            },
        )


def get_enhanced_test_functions(n_dim: int = 2) -> Dict[str, Callable]:
    """Get dictionary of enhanced test functions for HumpDay benchmarking."""

    generator = EnhancedSurfaceGenerator()

    functions = {}
    function_names = [
        "sphere",
        "rastrigin",
        "griewank",
        "ackley",
        "rosenbrock",
        "zakharov",
    ]

    for name in function_names:
        try:
            func = generator.get_function_on_cube(name, n_dim)
            functions[f"{name}_enhanced"] = func
        except Exception as e:
            print(f"Failed to create {name}: {e}")

    return functions


if __name__ == "__main__":
    # Test the enhanced surface generator
    generator = EnhancedSurfaceGenerator()

    print("\n=== Enhanced Surface Generation Test ===")

    # Test different functions
    test_functions = ["sphere", "rastrigin", "griewank", "ackley", "rosenbrock"]

    for func_name in test_functions:
        print(f"\nTesting {func_name}:")

        try:
            # Get 2D function
            func = generator.get_function_on_cube(func_name, 2)

            # Test evaluation
            test_point = [0.5, 0.5]  # Center of cube
            result = func(test_point)

            print(f"  ✓ {func_name}: f([0.5, 0.5]) = {result:.6f}")

            # Get metadata
            metadata = generator.get_function_metadata(func_name)
            print(f"    Landscape: {metadata.get('landscape_type', 'unknown')}")
            print(f"    Modality: {metadata.get('modality', 'unknown')}")
            print(f"    Separable: {metadata.get('separable', 'unknown')}")

        except Exception as e:
            print(f"  ✗ {func_name}: {e}")

    print("\n=== Package Status ===")
    print(f"Opfunu available: {HAS_OPFUNU}")
    print(f"benchmark-functions available: {HAS_BENCHMARK_FUNCS}")

    if not HAS_OPFUNU and not HAS_BENCHMARK_FUNCS:
        print("\nTo get enhanced functions, install:")
        print("  pip install opfunu  # Comprehensive optimization functions")
        print("  pip install benchmark-functions  # Lightweight test functions")
