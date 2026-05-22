"""
Advanced multi-dimensional problem categorization for HumpDay.
Based on research insights from COCO/BBOB, IOH Profiler, and academic literature.
"""

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class ProblemCharacteristics:
    """
    Comprehensive problem characteristics for advanced Thurstone analysis.
    Based on research insights from major benchmarking suites.
    """

    # Core dimensions (original 3D approach)
    landscape_type: str  # smooth, multimodal, rugged
    dimensionality: str  # low, medium, high, very_high
    budget_class: str  # low, medium, high

    # Advanced dimensions (from research)
    conditioning: (
        str  # well_conditioned, moderate, ill_conditioned, very_ill_conditioned
    )
    separability: str  # separable, semi_separable, non_separable
    global_structure: str  # strong, moderate, weak, deceptive, none
    modality_density: str  # unimodal, few_modes, many_modes, highly_multimodal
    noise_level: str  # none, low, moderate, high

    # Meta characteristics
    difficulty: str  # easy, medium, hard, very_hard, extremely_hard
    bbob_category: str  # separable, low_conditioning, high_conditioning, multimodal, weak_structure


class AdvancedProblemCategorizer:
    """
    Advanced categorization system inspired by COCO/BBOB research insights.
    Provides 5D+ categorization for sophisticated Thurstone analysis.
    """

    def __init__(self):
        self.dimension_mappings = {
            "landscape_type": {
                "smooth": ["sphere", "ellipsoid", "rosenbrock", "zakharov"],
                "multimodal": ["rastrigin", "griewank", "ackley", "weierstrass"],
                "rugged": ["schwefel", "gallagher", "schaffer", "composition"],
            },
            "conditioning": {
                "well_conditioned": [1, 10],
                "moderate": [10, 100],
                "ill_conditioned": [100, 1000],
                "very_ill_conditioned": [1000, float("inf")],
            },
            "separability": {
                "separable": ["sphere_sep", "rastrigin_sep", "ellipsoid_sep"],
                "semi_separable": ["rosenbrock", "mixed_functions"],
                "non_separable": ["rotated", "composition", "hybrid"],
            },
            "global_structure": {
                "strong": ["sphere", "ellipsoid", "quadratic"],
                "moderate": ["rosenbrock", "griewank"],
                "weak": ["rastrigin", "ackley", "weierstrass"],
                "deceptive": ["schwefel", "gallagher"],
                "none": ["random", "noise"],
            },
        }

    def categorize_function(
        self, function_name: str, n_dim: int, n_trials: int, metadata: Dict = None
    ) -> ProblemCharacteristics:
        """Categorize a function using advanced multi-dimensional scheme."""

        if metadata is None:
            metadata = {}

        # Core dimensions
        landscape_type = self._infer_landscape_type(function_name, metadata)
        dimensionality = self._categorize_dimensionality(n_dim)
        budget_class = self._categorize_budget(n_trials, n_dim)

        # Advanced dimensions
        conditioning = self._infer_conditioning(function_name, metadata)
        separability = self._infer_separability(function_name, metadata)
        global_structure = self._infer_global_structure(function_name, metadata)
        modality_density = self._infer_modality_density(function_name, metadata)
        noise_level = metadata.get("noise_level", "none")

        # Meta characteristics
        difficulty = self._infer_difficulty(
            landscape_type, conditioning, modality_density, global_structure, n_dim
        )
        bbob_category = self._infer_bbob_category(function_name, metadata)

        return ProblemCharacteristics(
            landscape_type=landscape_type,
            dimensionality=dimensionality,
            budget_class=budget_class,
            conditioning=conditioning,
            separability=separability,
            global_structure=global_structure,
            modality_density=modality_density,
            noise_level=noise_level,
            difficulty=difficulty,
            bbob_category=bbob_category,
        )

    def _infer_landscape_type(self, function_name: str, metadata: Dict) -> str:
        """Infer landscape type from function name and metadata."""
        if metadata.get("landscape_type"):
            return metadata["landscape_type"]

        name_lower = function_name.lower()
        if any(
            smooth in name_lower
            for smooth in ["sphere", "ellipsoid", "rosenbrock", "zakharov"]
        ):
            return "smooth"
        elif any(
            rugged in name_lower for rugged in ["schwefel", "gallagher", "composition"]
        ):
            return "rugged"
        else:
            return "multimodal"

    def _categorize_dimensionality(self, n_dim: int) -> str:
        """Categorize dimensionality based on research insights."""
        if n_dim <= 2:
            return "low"
        elif n_dim <= 5:
            return "medium"
        elif n_dim <= 20:
            return "high"
        else:
            return "very_high"

    def _categorize_budget(self, n_trials: int, n_dim: int) -> str:
        """Categorize computational budget per dimension."""
        budget_per_dim = n_trials / n_dim
        if budget_per_dim < 10:
            return "low"
        elif budget_per_dim < 50:
            return "medium"
        else:
            return "high"

    def _infer_conditioning(self, function_name: str, metadata: Dict) -> str:
        """Infer conditioning based on function characteristics."""
        if metadata.get("conditioning"):
            return metadata["conditioning"]

        name_lower = function_name.lower()
        if "ill" in name_lower or "condition" in name_lower:
            return "ill_conditioned"
        elif "ellipsoid" in name_lower:
            return "moderate"
        elif any(well in name_lower for well in ["sphere", "rastrigin"]):
            return "well_conditioned"
        else:
            return "moderate"

    def _infer_separability(self, function_name: str, metadata: Dict) -> str:
        """Infer separability based on function characteristics."""
        if metadata.get("separable") is not None:
            return "separable" if metadata["separable"] else "non_separable"

        name_lower = function_name.lower()
        if "sep" in name_lower or any(
            sep in name_lower for sep in ["sphere", "rastrigin"]
        ):
            return "separable"
        elif "rotated" in name_lower or "composition" in name_lower:
            return "non_separable"
        else:
            return "semi_separable"

    def _infer_global_structure(self, function_name: str, metadata: Dict) -> str:
        """Infer global structure strength."""
        if metadata.get("global_structure"):
            return metadata["global_structure"]

        name_lower = function_name.lower()
        if any(strong in name_lower for strong in ["sphere", "ellipsoid"]):
            return "strong"
        elif "schwefel" in name_lower:
            return "deceptive"
        elif any(weak in name_lower for weak in ["rastrigin", "ackley"]):
            return "weak"
        else:
            return "moderate"

    def _infer_modality_density(self, function_name: str, metadata: Dict) -> str:
        """Infer modality density."""
        if metadata.get("modality"):
            return metadata["modality"]

        name_lower = function_name.lower()
        if any(uni in name_lower for uni in ["sphere", "ellipsoid", "rosenbrock"]):
            return "unimodal"
        elif any(high in name_lower for high in ["rastrigin", "ackley", "weierstrass"]):
            return "highly_multimodal"
        else:
            return "multimodal"

    def _infer_difficulty(
        self,
        landscape_type: str,
        conditioning: str,
        modality_density: str,
        global_structure: str,
        n_dim: int,
    ) -> str:
        """Infer overall difficulty based on multiple factors."""

        difficulty_score = 0

        # Landscape contribution
        landscape_scores = {"smooth": 0, "multimodal": 2, "rugged": 3}
        difficulty_score += landscape_scores.get(landscape_type, 2)

        # Conditioning contribution
        conditioning_scores = {
            "well_conditioned": 0,
            "moderate": 1,
            "ill_conditioned": 2,
            "very_ill_conditioned": 3,
        }
        difficulty_score += conditioning_scores.get(conditioning, 1)

        # Modality contribution
        modality_scores = {"unimodal": 0, "multimodal": 1, "highly_multimodal": 2}
        difficulty_score += modality_scores.get(modality_density, 1)

        # Global structure contribution
        structure_scores = {
            "strong": 0,
            "moderate": 1,
            "weak": 2,
            "deceptive": 3,
            "none": 3,
        }
        difficulty_score += structure_scores.get(global_structure, 1)

        # Dimensionality contribution
        if n_dim > 10:
            difficulty_score += 2
        elif n_dim > 5:
            difficulty_score += 1

        # Convert score to difficulty level
        if difficulty_score <= 2:
            return "easy"
        elif difficulty_score <= 4:
            return "medium"
        elif difficulty_score <= 6:
            return "hard"
        elif difficulty_score <= 8:
            return "very_hard"
        else:
            return "extremely_hard"

    def _infer_bbob_category(self, function_name: str, metadata: Dict) -> str:
        """Infer BBOB-style category."""
        if metadata.get("bbob_category"):
            return metadata["bbob_category"]

        name_lower = function_name.lower()
        if "sep" in name_lower:
            return "separable"
        elif "ill" in name_lower or "condition" in name_lower:
            return "high_conditioning"
        elif any(multi in name_lower for multi in ["rastrigin", "ackley", "griewank"]):
            return "multimodal"
        elif "schwefel" in name_lower or "weak" in name_lower:
            return "weak_structure"
        else:
            return "low_conditioning"


class Advanced5DThurstonAnalyzer:
    """
    5D+ Thurstone analysis using advanced categorization.
    Creates multi-dimensional performance tensor for sophisticated recommendations.
    """

    def __init__(self):
        self.categorizer = AdvancedProblemCategorizer()
        self.performance_tensor = None

    def analyze_benchmark_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze benchmark results using 5D+ categorization.
        Creates advanced performance tensor for context-specific recommendations.
        """

        # Add advanced categorization to results
        categorized_results = []

        for _, row in results_df.iterrows():
            if not row.get("success", False):
                continue

            # Get function metadata (would need to be provided or inferred)
            metadata = {}  # Could be enhanced with actual function metadata

            characteristics = self.categorizer.categorize_function(
                row["objective"], row["n_dim"], row["n_trials"], metadata
            )

            result_with_categories = row.to_dict()
            result_with_categories.update(
                {
                    "landscape_type": characteristics.landscape_type,
                    "dimensionality": characteristics.dimensionality,
                    "budget_class": characteristics.budget_class,
                    "conditioning": characteristics.conditioning,
                    "separability": characteristics.separability,
                    "global_structure": characteristics.global_structure,
                    "modality_density": characteristics.modality_density,
                    "difficulty": characteristics.difficulty,
                    "bbob_category": characteristics.bbob_category,
                }
            )

            categorized_results.append(result_with_categories)

        enhanced_df = pd.DataFrame(categorized_results)

        # Calculate relative performance within each problem context
        enhanced_df["relative_performance"] = 0.0

        context_columns = [
            "landscape_type",
            "dimensionality",
            "budget_class",
            "conditioning",
            "separability",
            "global_structure",
        ]

        # Group by specific problem instances
        problem_groups = enhanced_df.groupby(
            ["objective", "n_dim", "n_trials", "repeat"]
        )

        for name, group in problem_groups:
            if len(group) < 2:
                continue

            # Rank by performance (lower objective value is better)
            sorted_group = group.sort_values("best_value")
            n_optimizers = len(sorted_group)

            for i, idx in enumerate(sorted_group.index):
                relative_perf = (n_optimizers - i) / n_optimizers
                enhanced_df.loc[idx, "relative_performance"] = relative_perf

        # Build 5D+ performance tensor
        performance_analysis = self._build_5d_performance_tensor(enhanced_df)

        return {
            "enhanced_results": enhanced_df,
            "performance_tensor": performance_analysis,
            "advanced_recommendations": self._generate_advanced_recommendations(
                performance_analysis
            ),
            "categorization_insights": self._analyze_categorization_patterns(
                enhanced_df
            ),
        }

    def _build_5d_performance_tensor(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build 5D+ performance tensor for advanced analysis."""

        optimizers = sorted(df["optimizer"].unique())

        # Core 3D dimensions
        landscapes = sorted(df["landscape_type"].unique())
        dimensions = sorted(df["dimensionality"].unique())
        budgets = sorted(df["budget_class"].unique())

        # Advanced dimensions
        conditionings = sorted(df["conditioning"].unique())
        separabilities = sorted(df["separability"].unique())
        structures = sorted(df["global_structure"].unique())

        # Build multi-dimensional tensor
        tensor_data = {}

        for optimizer in optimizers:
            opt_data = df[df["optimizer"] == optimizer]

            profile = {}

            # 5D tensor: landscape × dimensionality × budget × conditioning × separability
            for landscape in landscapes:
                for dim in dimensions:
                    for budget in budgets:
                        for conditioning in conditionings:
                            for separability in separabilities:
                                context_data = opt_data[
                                    (opt_data["landscape_type"] == landscape)
                                    & (opt_data["dimensionality"] == dim)
                                    & (opt_data["budget_class"] == budget)
                                    & (opt_data["conditioning"] == conditioning)
                                    & (opt_data["separability"] == separability)
                                ]

                                context_key = f"{landscape}_{dim}_{budget}_{conditioning}_{separability}"

                                if len(context_data) > 0:
                                    profile[context_key] = {
                                        "performance": context_data[
                                            "relative_performance"
                                        ].mean(),
                                        "std": context_data[
                                            "relative_performance"
                                        ].std(),
                                        "n_samples": len(context_data),
                                    }
                                else:
                                    profile[context_key] = {
                                        "performance": 0.5,  # Neutral
                                        "std": 0.0,
                                        "n_samples": 0,
                                    }

            tensor_data[optimizer] = profile

        return {
            "tensor_data": tensor_data,
            "dimensions": {
                "landscapes": landscapes,
                "dimensionalities": dimensions,
                "budgets": budgets,
                "conditionings": conditionings,
                "separabilities": separabilities,
            },
        }

    def _generate_advanced_recommendations(
        self, tensor_analysis: Dict
    ) -> Dict[str, str]:
        """Generate sophisticated context-specific recommendations."""

        recommendations = {}
        tensor_data = tensor_analysis["tensor_data"]

        # Find best optimizer for each 5D context
        for context_key in list(tensor_data.values())[0].keys():
            context_scores = {
                opt: profile[context_key]["performance"]
                for opt, profile in tensor_data.items()
                if profile[context_key]["n_samples"] > 0
            }

            if context_scores:
                best_optimizer = max(
                    context_scores.keys(), key=lambda x: context_scores[x]
                )
                best_score = context_scores[best_optimizer]

                if best_score > 0.6:  # Only recommend if clearly better
                    recommendations[context_key] = (
                        f"{best_optimizer} ({best_score:.3f})"
                    )

        return recommendations

    def _analyze_categorization_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in the advanced categorization."""

        return {
            "difficulty_distribution": df["difficulty"].value_counts().to_dict(),
            "bbob_category_distribution": df["bbob_category"].value_counts().to_dict(),
            "separability_performance": df.groupby("separability")[
                "relative_performance"
            ]
            .mean()
            .to_dict(),
            "conditioning_performance": df.groupby("conditioning")[
                "relative_performance"
            ]
            .mean()
            .to_dict(),
            "structure_performance": df.groupby("global_structure")[
                "relative_performance"
            ]
            .mean()
            .to_dict(),
        }


if __name__ == "__main__":
    # Test advanced categorization system
    print("=== Advanced 5D+ Categorization System Test ===")

    categorizer = AdvancedProblemCategorizer()

    # Test function categorization
    test_functions = [
        ("f01_sphere_sep", 2, 50),
        ("f04_rastrigin_sep", 5, 100),
        ("f08_ellipsoid_rotated_ill", 10, 25),
        ("f11_schwefel_weak", 3, 75),
    ]

    for func_name, n_dim, n_trials in test_functions:
        characteristics = categorizer.categorize_function(func_name, n_dim, n_trials)

        print(f"\n{func_name}:")
        print(f"  Landscape: {characteristics.landscape_type}")
        print(f"  Dimensionality: {characteristics.dimensionality}")
        print(f"  Budget: {characteristics.budget_class}")
        print(f"  Conditioning: {characteristics.conditioning}")
        print(f"  Separability: {characteristics.separability}")
        print(f"  Global structure: {characteristics.global_structure}")
        print(f"  Modality: {characteristics.modality_density}")
        print(f"  Difficulty: {characteristics.difficulty}")
        print(f"  BBOB category: {characteristics.bbob_category}")

    print("\n=== 5D+ Thurstone Analysis Ready ===")
    print("✓ Advanced multi-dimensional categorization")
    print("✓ BBOB-inspired systematic approach")
    print("✓ Sophisticated context-specific recommendations")
    print("✓ Research-backed problem characteristics")
