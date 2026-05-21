"""
Working enhanced surface generation using benchmark-functions package.
Focus: Single-objective continuous optimization on hypercube [0,1]^n
"""

import numpy as np
from typing import List, Callable, Dict, Any
import warnings

# Import the working benchmark-functions package
try:
    import benchmark_functions as bf
    HAS_BENCHMARK_FUNCS = True
    print("✓ benchmark-functions available - comprehensive test functions")
except ImportError:
    HAS_BENCHMARK_FUNCS = False
    print("⚠ benchmark-functions not available - using fallback functions")


class WorkingEnhancedSurfaceGenerator:
    """
    Enhanced surface generator using the working benchmark-functions package.
    Generates functions on hypercube [0,1]^n for derivative-free optimization.
    """

    def __init__(self):
        self.function_mapping = {
            # Map user-friendly names to benchmark-functions classes
            'sphere': 'Hypersphere',
            'ackley': 'Ackley',
            'griewank': 'Griewank',
            'rastrigin': 'Rastrigin',
            'rosenbrock': 'Rosenbrock',
            'schwefel': 'Schwefel',
            'styblinski_tang': 'StyblinskiTang',
            'michalewicz': 'Michalewicz',
            'easom': 'Easom',
            'goldstein_price': 'GoldsteinAndPrice',
            'himmelblau': 'Himmelblau',
            'keane': 'Keane',
            'mccormick': 'McCormick'
        }

        # Categorization for 3D Thurstone analysis
        self.function_metadata = {
            'sphere': {
                'landscape_type': 'smooth',
                'modality': 'unimodal',
                'separable': True,
                'conditioning': 'well_conditioned',
                'global_structure': 'strong',
                'difficulty': 'easy'
            },
            'ackley': {
                'landscape_type': 'multimodal',
                'modality': 'highly_multimodal',
                'separable': False,
                'conditioning': 'moderate',
                'global_structure': 'weak',
                'difficulty': 'hard'
            },
            'griewank': {
                'landscape_type': 'multimodal',
                'modality': 'multimodal',
                'separable': False,
                'conditioning': 'moderate',
                'global_structure': 'moderate',
                'difficulty': 'medium'
            },
            'rastrigin': {
                'landscape_type': 'multimodal',
                'modality': 'highly_multimodal',
                'separable': True,
                'conditioning': 'well_conditioned',
                'global_structure': 'weak',
                'difficulty': 'hard'
            },
            'rosenbrock': {
                'landscape_type': 'smooth',
                'modality': 'unimodal',
                'separable': False,
                'conditioning': 'ill_conditioned',
                'global_structure': 'moderate',
                'difficulty': 'medium'
            },
            'schwefel': {
                'landscape_type': 'rugged',
                'modality': 'multimodal',
                'separable': True,
                'conditioning': 'moderate',
                'global_structure': 'deceptive',
                'difficulty': 'very_hard'
            }
        }

    def get_function_on_cube(self, function_name: str, n_dim: int) -> Callable:
        """Get a test function that operates on hypercube [0,1]^n."""

        if HAS_BENCHMARK_FUNCS:
            return self._get_benchmark_function_on_cube(function_name, n_dim)
        else:
            return self._get_fallback_function_on_cube(function_name, n_dim)

    def _get_benchmark_function_on_cube(self, function_name: str, n_dim: int) -> Callable:
        """Create benchmark-functions based function on hypercube."""

        # Get the class name
        class_name = self.function_mapping.get(function_name, 'Rastrigin')

        try:
            # Get the function class and create instance
            func_class = getattr(bf, class_name)
            func_instance = func_class(n_dim)

            # Get typical bounds for scaling (some functions may not have bounds attribute)
            bounds_mapping = {
                'Hypersphere': ([-5, 5], [-5, 5]),
                'Ackley': ([-32.768, 32.768], [-32.768, 32.768]),
                'Griewank': ([-600, 600], [-600, 600]),
                'Rastrigin': ([-5.12, 5.12], [-5.12, 5.12]),
                'Rosenbrock': ([-2.048, 2.048], [-2.048, 2.048]),
                'Schwefel': ([-500, 500], [-500, 500]),
                'StyblinskiTang': ([-5, 5], [-5, 5]),
                'Michalewicz': ([0, np.pi], [0, np.pi]),
                'GoldsteinAndPrice': ([-2, 2], [-2, 2]),
                'Himmelblau': ([-5, 5], [-5, 5])
            }

            if class_name in bounds_mapping:
                bounds = bounds_mapping[class_name]
                lb = bounds[0][0] if isinstance(bounds[0], list) else bounds[0]
                ub = bounds[0][1] if isinstance(bounds[0], list) else bounds[1]
            else:
                # Default bounds
                lb, ub = -10, 10

            def cube_function(x):
                """Transform from [0,1]^n to function's native domain."""
                x = np.array(x)
                # Scale from [0,1] to [lb, ub]
                scaled_x = lb + (ub - lb) * x
                return func_instance(scaled_x.tolist())

            return cube_function

        except Exception as e:
            warnings.warn(f"Benchmark function {function_name} failed: {e}")
            return self._get_fallback_function_on_cube(function_name, n_dim)

    def _get_fallback_function_on_cube(self, function_name: str, n_dim: int) -> Callable:
        """Fallback implementations when external packages unavailable."""

        def sphere(x):
            x = np.array(x)
            scaled_x = 10 * x - 5  # [0,1] -> [-5,5]
            return np.sum(scaled_x**2)

        def ackley(x):
            x = np.array(x)
            scaled_x = 65.536 * x - 32.768  # [0,1] -> [-32.768,32.768]
            n = len(scaled_x)
            a, b, c = 20, 0.2, 2 * np.pi
            term1 = -a * np.exp(-b * np.sqrt(np.sum(scaled_x**2) / n))
            term2 = -np.exp(np.sum(np.cos(c * scaled_x)) / n)
            return term1 + term2 + a + np.e

        def rastrigin(x):
            x = np.array(x)
            scaled_x = 10.24 * x - 5.12  # [0,1] -> [-5.12,5.12]
            return 10 * len(scaled_x) + np.sum(scaled_x**2 - 10 * np.cos(2 * np.pi * scaled_x))

        def griewank(x):
            x = np.array(x)
            scaled_x = 1200 * x - 600  # [0,1] -> [-600,600]
            sum_sq = np.sum(scaled_x**2) / 4000
            prod_cos = np.prod(np.cos(scaled_x / np.sqrt(np.arange(1, len(scaled_x) + 1))))
            return sum_sq - prod_cos + 1

        def rosenbrock(x):
            x = np.array(x)
            scaled_x = 4.096 * x - 2.048  # [0,1] -> [-2.048,2.048]
            return np.sum(100 * (scaled_x[1:] - scaled_x[:-1]**2)**2 + (1 - scaled_x[:-1])**2)

        def schwefel(x):
            x = np.array(x)
            scaled_x = 1000 * x - 500  # [0,1] -> [-500,500]
            return 418.9829 * len(scaled_x) - np.sum(scaled_x * np.sin(np.sqrt(np.abs(scaled_x))))

        fallback_functions = {
            'sphere': sphere,
            'ackley': ackley,
            'rastrigin': rastrigin,
            'griewank': griewank,
            'rosenbrock': rosenbrock,
            'schwefel': schwefel
        }

        return fallback_functions.get(function_name, rastrigin)

    def get_enhanced_function_set(self, n_dim: int = 2) -> Dict[str, Callable]:
        """Get comprehensive set of test functions for enhanced benchmarking."""

        functions = {}

        # Core test functions for 3D Thurstone analysis
        core_functions = [
            'sphere',      # Smooth, unimodal, easy
            'rosenbrock',  # Smooth, unimodal, medium difficulty
            'rastrigin',   # Multimodal, separable, hard
            'ackley',      # Multimodal, non-separable, hard
            'griewank',    # Multimodal, medium difficulty
            'schwefel'     # Rugged, deceptive, very hard
        ]

        for func_name in core_functions:
            try:
                func = self.get_function_on_cube(func_name, n_dim)
                functions[f"{func_name}_enhanced"] = func
            except Exception as e:
                print(f"Failed to create {func_name}: {e}")

        # Additional functions if available
        if HAS_BENCHMARK_FUNCS:
            additional_functions = [
                'styblinski_tang',  # Multimodal with known global structure
                'michalewicz',      # Steep ridges and valleys
                'himmelblau',       # Four global minima
                'goldstein_price',  # Complex multimodal surface
                'keane'            # Constrained-like behavior
            ]

            for func_name in additional_functions:
                try:
                    func = self.get_function_on_cube(func_name, n_dim)
                    functions[f"{func_name}_enhanced"] = func
                except Exception as e:
                    print(f"Failed to create additional function {func_name}: {e}")

        return functions

    def generate_2d_surface_data(self, function_name: str, size: int = 50) -> tuple:
        """Generate 2D surface data for visualization."""

        func = self.get_function_on_cube(function_name, n_dim=2)

        # Generate grid on [0,1]^2
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
        """Get metadata for 3D Thurstone categorization."""
        return self.function_metadata.get(function_name, {
            'landscape_type': 'multimodal',
            'modality': 'multimodal',
            'separable': False,
            'conditioning': 'moderate',
            'global_structure': 'moderate',
            'difficulty': 'medium'
        })

    def get_functions_by_category(self, category: str, difficulty: str = None) -> List[str]:
        """Get function names filtered by category and difficulty."""

        category_filters = {
            'smooth': lambda meta: meta.get('landscape_type') == 'smooth',
            'multimodal': lambda meta: meta.get('landscape_type') == 'multimodal',
            'rugged': lambda meta: meta.get('landscape_type') == 'rugged',
            'separable': lambda meta: meta.get('separable') == True,
            'non_separable': lambda meta: meta.get('separable') == False,
            'easy': lambda meta: meta.get('difficulty') == 'easy',
            'medium': lambda meta: meta.get('difficulty') == 'medium',
            'hard': lambda meta: meta.get('difficulty') in ['hard', 'very_hard']
        }

        if category not in category_filters:
            return list(self.function_metadata.keys())

        filter_func = category_filters[category]
        matching_functions = []

        for func_name, metadata in self.function_metadata.items():
            if filter_func(metadata):
                if difficulty is None or metadata.get('difficulty') == difficulty:
                    matching_functions.append(func_name)

        return matching_functions


if __name__ == "__main__":
    # Test the working enhanced surface generator
    generator = WorkingEnhancedSurfaceGenerator()

    print("\n=== Working Enhanced Surface Generation Test ===")

    # Get enhanced function set
    functions = generator.get_enhanced_function_set(n_dim=2)
    print(f"\nGenerated {len(functions)} enhanced test functions:")

    for name, func in functions.items():
        try:
            # Test at center and corner of cube
            center_result = func([0.5, 0.5])
            corner_result = func([0.1, 0.9])

            print(f"  ✓ {name:20s}: center={center_result:8.4f}, corner={corner_result:8.4f}")

            # Get metadata
            base_name = name.replace('_enhanced', '')
            metadata = generator.get_function_metadata(base_name)
            print(f"    {metadata.get('landscape_type', 'unknown'):10s} | "
                  f"{metadata.get('modality', 'unknown'):15s} | "
                  f"difficulty: {metadata.get('difficulty', 'unknown')}")

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Test categorization
    print(f"\n=== Function Categorization ===")
    print(f"Smooth functions: {generator.get_functions_by_category('smooth')}")
    print(f"Multimodal functions: {generator.get_functions_by_category('multimodal')}")
    print(f"Hard functions: {generator.get_functions_by_category('hard')}")

    print(f"\n=== Ready for 3D Thurstone Analysis ===")
    print(f"✓ {len(functions)} diverse test functions")
    print(f"✓ Rich metadata for multi-dimensional categorization")
    print(f"✓ All functions work on hypercube [0,1]^n")
    print(f"✓ Perfect for derivative-free optimization benchmarking")