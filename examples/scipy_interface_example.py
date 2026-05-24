#!/usr/bin/env python3
"""
Example usage of Humpday's cube-based interface with rectangular bounds.

This demonstrates how to use Humpday optimizers with arbitrary rectangular
domains using cube transformation to the unit hypercube [0,1]^n.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

from humpday import (
    cube_cma_es,
    cube_differential_evolution,
    cube_minimize,
    cube_minimize_scalar,
    cube_nelder_mead,
    cube_particle_swarm,
    cube_prima_uobyqa,
)


def example_1_basic_usage():
    """Basic usage with rectangular bounds."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Rectangular Bounds")
    print("=" * 60)

    # Define objective function on arbitrary domain
    def objective(x):
        """Rosenbrock function with minimum at (1, 1)."""
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    # Optimize on rectangular domain [-2, 2] x [-1, 3]
    bounds = [(-2, 2), (-1, 3)]

    result = cube_minimize(objective, bounds=bounds, method="NelderMead")

    print(f"Optimization successful: {result.success}")
    print(f"Function value: {result.fun:.6e}")
    print(f"Solution: [{result.x[0]:.4f}, {result.x[1]:.4f}]")
    print("Expected: [1.0000, 1.0000]")
    print(f"Function evaluations: {result.nfev}")

    # Verify solution is close to expected
    error = np.linalg.norm(result.x - np.array([1, 1]))
    print(f"Solution error: {error:.4f}")


def example_2_algorithm_comparison():
    """Compare different algorithms on the same problem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Algorithm Comparison")
    print("=" * 60)

    # Rastrigin function - multimodal, challenging
    def rastrigin(x):
        """Rastrigin function with global minimum at origin."""
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

    # Search on [-5, 5]^2
    bounds = [(-5, 5), (-5, 5)]

    algorithms = [
        ("Nelder-Mead", cube_nelder_mead),
        ("Differential Evolution", cube_differential_evolution),
        ("Particle Swarm", cube_particle_swarm),
        ("CMA-ES", cube_cma_es),
        ("PRIMA UOBYQA", cube_prima_uobyqa),
    ]

    print(
        f"{'Algorithm':<20} {'Best Value':<12} {'Solution':<20} {'Distance to Optimum'}"
    )
    print("-" * 70)

    for name, algorithm in algorithms:
        try:
            result = algorithm(rastrigin, bounds=bounds, options={"maxiter": 500})

            # Distance from global optimum at (0, 0)
            distance = np.linalg.norm(result.x)

            print(
                f"{name:<20} {result.fun:<12.2e} "
                f"[{result.x[0]:6.3f}, {result.x[1]:6.3f}] {distance:>12.3f}"
            )

        except Exception as e:
            print(f"{name:<20} {'FAILED':<12} {str(e)[:30]}")


def example_3_scalar_optimization():
    """Scalar (1D) optimization example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Scalar (1D) Optimization")
    print("=" * 60)

    # 1D function with multiple local minima
    def complex_1d(x):
        """Complex 1D function with multiple local minima."""
        return np.sin(5 * x) * np.exp(-(x**2)) + 0.1 * x**2

    # Optimize on [-3, 3]
    result = cube_minimize_scalar(
        complex_1d, bounds=(-3, 3), method="DifferentialEvolution"
    )

    print(f"Minimum found at x = {result.x:.4f}")
    print(f"Function value: {result.fun:.6f}")

    # Create plot to visualize
    if HAVE_MATPLOTLIB:
        try:
            x_plot = np.linspace(-3, 3, 1000)
            y_plot = [complex_1d(x) for x in x_plot]

            plt.figure(figsize=(10, 6))
            plt.plot(x_plot, y_plot, "b-", linewidth=2, label="Objective function")
            plt.plot(
                result.x,
                result.fun,
                "ro",
                markersize=10,
                label=f"Optimum ({result.x:.3f}, {result.fun:.3f})",
            )
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("1D Optimization Example")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("scipy_interface_1d_example.png", dpi=150, bbox_inches="tight")
            plt.show()
            print("Plot saved as 'scipy_interface_1d_example.png'")
        except Exception as e:
            print(f"Plotting failed: {e}")
    else:
        print("Matplotlib not available - skipping plot")


def example_4_constrained_domain():
    """Optimization with constrained domain."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Constrained Domain")
    print("=" * 60)

    # Portfolio optimization example
    def portfolio_risk(weights):
        """Simple portfolio risk function."""
        # Correlation matrix (example)
        corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])

        # Risk (variance)
        risk = np.dot(weights, np.dot(corr, weights))

        # Add penalty for constraint violation (weights should sum to 1)
        constraint_penalty = 1000 * (np.sum(weights) - 1) ** 2

        return risk + constraint_penalty

    # Each weight between 0 and 1
    bounds = [(0, 1), (0, 1), (0, 1)]

    result = cube_minimize(
        portfolio_risk,
        bounds=bounds,
        method="DifferentialEvolution",
        options={"maxiter": 200},
    )

    print(f"Optimal portfolio weights: {result.x}")
    print(f"Sum of weights: {np.sum(result.x):.4f} (should be ~1.0)")
    print(f"Portfolio risk: {result.fun:.6f}")


def example_5_high_dimensional():
    """High-dimensional optimization example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: High-Dimensional Optimization")
    print("=" * 60)

    # 50-dimensional sphere function with random offset
    n_dim = 50
    np.random.seed(42)
    target = np.random.uniform(-1, 1, n_dim)

    def high_dim_sphere(x):
        """High-dimensional shifted sphere function."""
        return np.sum((x - target) ** 2)

    # Optimize on [-2, 2]^50
    bounds = [(-2, 2)] * n_dim

    print(f"Optimizing {n_dim}-dimensional problem...")

    result = cube_minimize(
        high_dim_sphere,
        bounds=bounds,
        method="ParticleSwarm",
        options={"maxiter": 1000},
    )

    print(f"Best function value: {result.fun:.2e}")
    print(f"Distance to true optimum: {np.linalg.norm(result.x - target):.4f}")
    print(f"First 5 solution components: {result.x[:5]}")
    print(f"First 5 target components:   {target[:5]}")

    # Check if solution is reasonable
    if result.fun < 1e-2:
        print("✅ Successfully solved high-dimensional problem!")
    else:
        print("⚠️  High-dimensional optimization challenging - try different algorithm")


def example_6_domain_transformations():
    """Demonstrate domain transformation utilities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Domain Transformation Utilities")
    print("=" * 60)

    from humpday import transform_from_unit_cube, transform_to_unit_cube

    # Define custom bounds
    bounds = [(-10, 10), (0, 100), (-5, 5)]

    # Some points in the rectangular domain
    real_points = [
        np.array([0, 50, 0]),  # Center
        np.array([-10, 0, -5]),  # Lower corner
        np.array([10, 100, 5]),  # Upper corner
        np.array([5, 75, -2.5]),  # Random point
    ]

    print(f"{'Real Domain':<20} {'Unit Cube':<20} {'Recovered':<20}")
    print("-" * 60)

    for point in real_points:
        # Transform to unit cube
        unit_point = transform_to_unit_cube(point, bounds)

        # Transform back
        recovered = transform_from_unit_cube(unit_point, bounds)

        print(f"{str(point):<20} {str(unit_point):<20} {str(recovered):<20}")

        # Check transformation is reversible
        error = np.linalg.norm(point - recovered)
        assert error < 1e-12, f"Transformation error: {error}"

    print("\n✅ All transformations are reversible!")


def example_7_comparison_with_unit_cube():
    """Compare SciPy interface with direct unit cube optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: SciPy Interface vs Unit Cube Direct")
    print("=" * 60)

    from humpday import pure_optimize

    # Function defined on [-5, 5]^2
    def objective_real(x):
        return (x[0] + 2) ** 2 + (x[1] - 1) ** 2

    # Same function mapped to unit cube
    def objective_unit(x):
        # Map [0,1]^2 to [-5,5]^2
        x_real = -5 + 10 * np.array(x)
        return (x_real[0] + 2) ** 2 + (x_real[1] - 1) ** 2

    bounds = [(-5, 5), (-5, 5)]

    # Method 1: SciPy interface
    result_scipy = cube_minimize(objective_real, bounds=bounds, method="NelderMead")

    # Method 2: Direct unit cube optimization
    best_val_unit, best_x_unit = pure_optimize(objective_unit, "NelderMead", 1000, 2)

    # Transform unit cube result to real domain
    best_x_real = -5 + 10 * np.array(best_x_unit)

    print("SciPy Interface Result:")
    print(f"  Solution: [{result_scipy.x[0]:.4f}, {result_scipy.x[1]:.4f}]")
    print(f"  Function value: {result_scipy.fun:.6f}")

    print("Unit Cube Direct Result:")
    print(f"  Solution: [{best_x_real[0]:.4f}, {best_x_real[1]:.4f}]")
    print(f"  Function value: {best_val_unit:.6f}")

    print("Expected Solution: [-2.0000, 1.0000]")

    # Both should be similar
    diff = np.linalg.norm(result_scipy.x - best_x_real)
    print(f"Solution difference: {diff:.6f}")

    if diff < 0.1:
        print("✅ Both methods give consistent results!")
    else:
        print(
            "⚠️  Results differ - this can happen due to randomness in initialization"
        )


def main():
    """Run all examples."""
    print("🔧 HUMPDAY CUBE-BASED INTERFACE EXAMPLES")
    print("🔧 Demonstrating rectangular bounds with cube transformation")

    try:
        example_1_basic_usage()
        example_2_algorithm_comparison()
        example_3_scalar_optimization()
        example_4_constrained_domain()
        example_5_high_dimensional()
        example_6_domain_transformations()
        example_7_comparison_with_unit_cube()

        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("📏 SciPy-style interface with rectangular bounds working perfectly!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
