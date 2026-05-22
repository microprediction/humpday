#!/usr/bin/env python3
"""
Test whether the Underpromoted wrapper actually helps find plateau regions.
"""

import numpy as np
from embarrassingly.underpromoted import Underpromoted2d
from scipy.optimize import minimize


def create_plateau_landscape():
    """Create a function with a clear plateau that should be found."""

    def landscape(x):
        x = np.array(x)

        # Create a "helicopter landing pad" - broad flat region around [0.3, 0.7]
        target = np.array([0.3, 0.7])
        distance = np.linalg.norm(x - target)

        if distance < 0.15:  # Large plateau region
            # Very flat in the plateau with tiny slope toward center
            return 0.1 + 0.01 * distance
        elif distance < 0.25:  # Gentle slopes around plateau
            return 0.1 + 0.5 * (distance - 0.15)
        else:  # Steep outside plateau
            return 0.15 + 2.0 * (distance - 0.25) ** 2

    return landscape


def create_multiple_plateaus():
    """Create a function with multiple plateau regions of different quality."""

    def landscape(x):
        x = np.array(x)

        # Plateau 1: Best plateau at [0.2, 0.8]
        p1_dist = np.linalg.norm(x - np.array([0.2, 0.8]))
        if p1_dist < 0.1:
            return 0.05 + 0.01 * p1_dist  # Best plateau

        # Plateau 2: Decent plateau at [0.7, 0.3]
        p2_dist = np.linalg.norm(x - np.array([0.7, 0.3]))
        if p2_dist < 0.08:
            return 0.15 + 0.01 * p2_dist  # Second-best plateau

        # Sharp minimum at [0.5, 0.5] - should be avoided for robustness
        sharp_dist = np.linalg.norm(x - np.array([0.5, 0.5]))
        if sharp_dist < 0.05:
            return 0.001 + 100 * sharp_dist**4  # Very sharp, fragile minimum

        # Background landscape
        return 1.0 + np.sum(x**2)

    return landscape


def run_optimization_test(objective_func, method="Powell", max_evals=50):
    """Run optimization and return results."""
    eval_count = [0]

    def wrapped_objective(x):
        eval_count[0] += 1
        return objective_func(x)

    # Multiple random starts to test robustness
    results = []

    for seed in [42, 123, 456]:
        np.random.seed(seed)
        x0 = np.random.rand(2)

        try:
            result = minimize(
                wrapped_objective,
                x0,
                method=method,
                bounds=[(0, 1), (0, 1)],
                options={"maxfev": max_evals},
            )

            results.append(
                {
                    "final_x": result.x.copy(),
                    "final_value": float(result.fun),
                    "success": result.success,
                    "evaluations": eval_count[0],
                }
            )
        except Exception as e:
            results.append(
                {
                    "final_x": x0,
                    "final_value": float("inf"),
                    "success": False,
                    "error": str(e),
                }
            )

    return results


def analyze_plateau_preference(results, landscape_func):
    """Analyze whether results prefer plateau regions vs sharp minima."""
    analysis = []

    for result in results:
        if not result["success"] or result["final_value"] == float("inf"):
            continue

        x = result["final_x"]

        # Check which region the solution landed in
        target_plateau = np.array([0.2, 0.8])  # Best plateau location
        decent_plateau = np.array([0.7, 0.3])  # Second plateau
        sharp_minimum = np.array([0.5, 0.5])  # Sharp minimum (fragile)

        dist_to_best = np.linalg.norm(x - target_plateau)
        dist_to_decent = np.linalg.norm(x - decent_plateau)
        dist_to_sharp = np.linalg.norm(x - sharp_minimum)

        # Categorize the solution
        if dist_to_best < 0.15:
            region = "best_plateau"
        elif dist_to_decent < 0.12:
            region = "decent_plateau"
        elif dist_to_sharp < 0.1:
            region = "sharp_minimum"
        else:
            region = "other"

        analysis.append(
            {
                "final_x": x,
                "final_value": result["final_value"],
                "region": region,
                "dist_to_best_plateau": dist_to_best,
                "dist_to_sharp": dist_to_sharp,
            }
        )

    return analysis


def test_plateau_finding():
    """Test whether Underpromoted helps find robust plateau solutions."""

    print("🏔️ Testing Embarrassingly Underpromoted (Plateau Finding)")
    print("=" * 60)

    # Test on simple plateau landscape
    print("\n📊 Test 1: Simple Plateau Landscape")
    print("-" * 40)

    simple_landscape = create_plateau_landscape()
    bounds = [[0, 1], [0, 1]]

    methods = ["Powell", "Nelder-Mead"]

    for method in methods:
        print(f"\n{method} optimizer:")

        # Standard optimization
        standard_results = run_optimization_test(simple_landscape, method)
        standard_avg = np.mean(
            [r["final_value"] for r in standard_results if r["success"]]
        )

        # Plateau-enhanced optimization
        plateau_landscape = Underpromoted2d(
            simple_landscape, bounds=bounds, radius=0.08
        )
        plateau_results = run_optimization_test(plateau_landscape, method)
        plateau_avg = np.mean(
            [r["final_value"] for r in plateau_results if r["success"]]
        )

        improvement = (
            (standard_avg - plateau_avg) / standard_avg * 100
            if standard_avg != 0
            else 0
        )

        print(f"  Standard: {standard_avg:.4f}")
        print(f"  Plateau:  {plateau_avg:.4f}")

        if improvement > 5:
            print(f"  🎯 Underpromoted finds {improvement:.1f}% better solutions!")
        elif improvement < -5:
            print(f"  📉 Underpromoted finds {abs(improvement):.1f}% worse solutions")
        else:
            print(f"  ≈ Similar performance ({improvement:+.1f}%)")

    # Test on multiple plateaus with sharp minimum trap
    print("\n📊 Test 2: Multiple Plateaus vs Sharp Minimum")
    print("-" * 40)

    complex_landscape = create_multiple_plateaus()

    for method in methods:
        print(f"\n{method} optimizer:")

        # Standard optimization
        standard_results = run_optimization_test(complex_landscape, method)
        standard_analysis = analyze_plateau_preference(
            standard_results, complex_landscape
        )

        # Plateau-enhanced optimization
        plateau_complex = Underpromoted2d(complex_landscape, bounds=bounds, radius=0.06)
        plateau_results = run_optimization_test(plateau_complex, method)
        plateau_analysis = analyze_plateau_preference(
            plateau_results, complex_landscape
        )

        # Count preferences
        standard_plateau_count = sum(
            1 for a in standard_analysis if "plateau" in a["region"]
        )
        standard_sharp_count = sum(
            1 for a in standard_analysis if a["region"] == "sharp_minimum"
        )

        plateau_plateau_count = sum(
            1 for a in plateau_analysis if "plateau" in a["region"]
        )
        plateau_sharp_count = sum(
            1 for a in plateau_analysis if a["region"] == "sharp_minimum"
        )

        print("  Standard optimization:")
        print(f"    Found plateaus: {standard_plateau_count}/{len(standard_analysis)}")
        print(
            f"    Found sharp minimum: {standard_sharp_count}/{len(standard_analysis)}"
        )

        print("  Plateau-enhanced optimization:")
        print(f"    Found plateaus: {plateau_plateau_count}/{len(plateau_analysis)}")
        print(f"    Found sharp minimum: {plateau_sharp_count}/{len(plateau_analysis)}")

        if plateau_plateau_count > standard_plateau_count:
            print("  🏆 Underpromoted prefers robust plateau regions!")
        elif plateau_sharp_count < standard_sharp_count:
            print("  🛡️ Underpromoted avoids fragile sharp minima!")
        else:
            print("  ≈ No clear preference change")

    return True


if __name__ == "__main__":
    print("🧪 Testing Plateau Finding with Embarrassingly Underpromoted")
    print("=" * 70)

    try:
        results = test_plateau_finding()
        print("\n✅ Plateau finding tests completed!")
        print(
            "Check results above to see if Underpromoted helps find robust solutions."
        )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
