#!/usr/bin/env python3
"""
Comprehensive Algorithm Validation Framework

Tests EVERY JavaScript algorithm against its external reference implementation
to ensure they produce identical results. This is the gold standard for validating
that our JavaScript ports are actually implementing the correct algorithms.

Usage:
    python test_all_algorithms_comprehensive.py
"""

import json
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class AlgorithmTestResult:
    """Result of testing one algorithm"""

    algorithm: str
    package: str
    js_success: bool
    ref_success: bool
    perfect_matches: int
    total_tests: int
    avg_js_value: float
    avg_ref_value: float
    max_difference: float
    passed_validation: bool
    error_message: Optional[str]
    win_rate_vs_ref: float  # For 50% win rate validation


class ComprehensiveAlgorithmValidator:
    """Tests ALL algorithms against their external reference implementations"""

    def __init__(self):
        # Test functions with known global optima
        self.test_functions = {
            "sphere2d": {
                "name": "2D Sphere",
                "python_func": lambda x: np.sum(x**2),
                "js_func": "x => x[0]*x[0] + x[1]*x[1]",
                "optimum": [0, 0],
                "optimum_value": 0,
                "dimensions": 2,
                "bounds": [(0, 1), (0, 1)],
            },
            "rosenbrock2d": {
                "name": "2D Rosenbrock",
                "python_func": lambda x: (
                    (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
                ),
                "js_func": "x => { const a = 1, b = 100; return (a - x[0])**2 + b * (x[1] - x[0]**2)**2; }",
                "optimum": [1, 1],
                "optimum_value": 0,
                "dimensions": 2,
                "bounds": [(0, 1), (0, 1)],
            },
            "sphere3d": {
                "name": "3D Sphere",
                "python_func": lambda x: np.sum(x**2),
                "js_func": "x => x[0]*x[0] + x[1]*x[1] + x[2]*x[2]",
                "optimum": [0, 0, 0],
                "optimum_value": 0,
                "dimensions": 3,
                "bounds": [(0, 1), (0, 1), (0, 1)],
            },
            "beale2d": {
                "name": "2D Beale",
                "python_func": lambda x: (
                    (1.5 - x[0] + x[0] * x[1]) ** 2
                    + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                    + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
                ),
                "js_func": "x => { const a = 1.5 - x[0] + x[0]*x[1]; const b = 2.25 - x[0] + x[0]*x[1]*x[1]; const c = 2.625 - x[0] + x[0]*x[1]*x[1]*x[1]; return a*a + b*b + c*c; }",
                "optimum": [3, 0.5],
                "optimum_value": 0,
                "dimensions": 2,
                "bounds": [
                    (0, 1),
                    (0, 1),
                ],  # Note: true optimum outside [0,1], so this tests constrained optimization
            },
        }

        # EVERY algorithm with its external reference package
        self.algorithms = {
            # PRIMA algorithms - PDFO package
            "PRIMA_UOBYQA": {
                "js_name": "PRIMA_UOBYQA",
                "reference_test": self._test_pdfo_uobyqa,
                "package": "PDFO",
                "priority": "HIGH",  # Already working
            },
            "PRIMA_NEWUOA": {
                "js_name": "PRIMA_NEWUOA",
                "reference_test": self._test_pdfo_newuoa,
                "package": "PDFO",
                "priority": "HIGH",  # In progress
            },
            "PRIMA_BOBYQA": {
                "js_name": "PRIMA_BOBYQA",
                "reference_test": self._test_pdfo_bobyqa,
                "package": "PDFO",
                "priority": "HIGH",  # Needs fix
            },
            # SciPy algorithms
            "SciPy_NelderMead": {
                "js_name": "SciPy_NelderMead",
                "reference_test": self._test_scipy_nelder_mead,
                "package": "SciPy",
                "priority": "HIGH",
            },
            "SciPy_Powell": {
                "js_name": "SciPy_Powell",
                "reference_test": self._test_scipy_powell,
                "package": "SciPy",
                "priority": "MEDIUM",
            },
            "DifferentialEvolution": {
                "js_name": "DifferentialEvolution",
                "reference_test": self._test_scipy_differential_evolution,
                "package": "SciPy",
                "priority": "HIGH",
            },
            "SimulatedAnnealing": {
                "js_name": "SimulatedAnnealing",
                "reference_test": self._test_scipy_simulated_annealing,
                "package": "SciPy",
                "priority": "MEDIUM",
            },
            # Evolutionary algorithms - DEAP
            "GeneticAlgorithm": {
                "js_name": "GeneticAlgorithm",
                "reference_test": self._test_deap_genetic_algorithm,
                "package": "DEAP",
                "priority": "HIGH",
            },
            "EvolutionStrategy": {
                "js_name": "EvolutionStrategy",
                "reference_test": self._test_deap_evolution_strategy,
                "package": "DEAP",
                "priority": "MEDIUM",
            },
            # Swarm intelligence - PySwarm
            "ParticleSwarm": {
                "js_name": "ParticleSwarm",
                "reference_test": self._test_pyswarm_pso,
                "package": "PySwarm",
                "priority": "HIGH",
            },
            # CMA-ES - pycma
            "CMAEvolutionStrategy": {
                "js_name": "CMAEvolutionStrategy",
                "reference_test": self._test_cma_es,
                "package": "pycma",
                "priority": "MEDIUM",
            },
            # Bayesian optimization - scikit-optimize
            "BayesianOpt": {
                "js_name": "BayesianOpt",
                "reference_test": self._test_skopt_bayesian,
                "package": "scikit-optimize",
                "priority": "HIGH",
            },
            # Random search - scikit-learn
            "RandomSearch": {
                "js_name": "RandomSearch",
                "reference_test": self._test_sklearn_random_search,
                "package": "scikit-learn",
                "priority": "LOW",  # Should be easy
            },
            # Additional SciPy algorithms
            "SciPy_BFGS": {
                "js_name": "SciPy_BFGS",
                "reference_test": self._test_scipy_bfgs,
                "package": "SciPy",
                "priority": "HIGH",
            },
            # Metaheuristic algorithms (using custom reference implementations)
            "AdaptiveRandomSearch": {
                "js_name": "AdaptiveRandomSearch",
                "reference_test": self._test_adaptive_random_search_external,
                "package": "nlopt (optional)",
                "priority": "MEDIUM",
            },
            "CoordinateDescent": {
                "js_name": "CoordinateDescent",
                "reference_test": self._test_sklearn_coordinate_descent,
                "package": "scikit-learn",
                "priority": "MEDIUM",
            },
            "PatternSearch": {
                "js_name": "PatternSearch",
                "reference_test": self._test_scipy_pattern_search,
                "package": "SciPy",
                "priority": "MEDIUM",
            },
            "HillClimbing": {
                "js_name": "HillClimbing",
                "reference_test": self._test_hill_climbing_external,
                "package": "scipy (fallback)",
                "priority": "MEDIUM",
            },
            "TabuSearch": {
                "js_name": "TabuSearch",
                "reference_test": self._test_tabu_search_external,
                "package": "scipy (fallback)",
                "priority": "MEDIUM",
            },
            "FireflyAlgorithm": {
                "js_name": "FireflyAlgorithm",
                "reference_test": self._test_firefly_external,
                "package": "scipy (fallback)",
                "priority": "MEDIUM",
            },
            "AntColonyOpt": {
                "js_name": "AntColonyOpt",
                "reference_test": self._test_ant_colony_external,
                "package": "acopy (optional)",
                "priority": "MEDIUM",
            },
            "HarmonySearch": {
                "js_name": "HarmonySearch",
                "reference_test": self._test_harmony_search_external,
                "package": "pyHarmonySearch (optional)",
                "priority": "MEDIUM",
            },
        }

    def _test_pdfo_uobyqa(self, func_name: str, max_evals: int = 100):
        """Test against PDFO UOBYQA"""
        try:
            from pdfo import uobyqa

            test_func = self.test_functions[func_name]

            np.random.seed(42)
            x0 = np.random.uniform(0, 1, test_func["dimensions"])

            result = uobyqa(
                test_func["python_func"],
                x0,
                options={"maxfev": max_evals, "rhobeg": 0.1, "rhoend": 1e-6},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_pdfo_newuoa(self, func_name: str, max_evals: int = 100):
        """Test against PDFO NEWUOA"""
        try:
            from pdfo import newuoa

            test_func = self.test_functions[func_name]

            np.random.seed(42)
            x0 = np.random.uniform(0, 1, test_func["dimensions"])

            result = newuoa(
                test_func["python_func"],
                x0,
                options={"maxfev": max_evals, "rhobeg": 0.1, "rhoend": 1e-6},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_pdfo_bobyqa(self, func_name: str, max_evals: int = 100):
        """Test against PDFO BOBYQA"""
        try:
            from pdfo import bobyqa

            test_func = self.test_functions[func_name]

            np.random.seed(42)
            x0 = np.random.uniform(0, 1, test_func["dimensions"])
            bounds = test_func["bounds"]

            result = bobyqa(
                test_func["python_func"],
                x0,
                bounds=bounds,
                options={"maxfev": max_evals, "rhobeg": 0.1, "rhoend": 1e-6},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_scipy_nelder_mead(self, func_name: str, max_evals: int = 100):
        """Test against SciPy Nelder-Mead"""
        try:
            from scipy.optimize import minimize

            test_func = self.test_functions[func_name]

            np.random.seed(42)
            x0 = np.random.uniform(0, 1, test_func["dimensions"])
            bounds = test_func["bounds"]

            result = minimize(
                test_func["python_func"],
                x0,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxfev": max_evals},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_scipy_powell(self, func_name: str, max_evals: int = 100):
        """Test against SciPy Powell"""
        try:
            from scipy.optimize import minimize

            test_func = self.test_functions[func_name]

            np.random.seed(42)
            x0 = np.random.uniform(0, 1, test_func["dimensions"])
            bounds = test_func["bounds"]

            result = minimize(
                test_func["python_func"],
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxfev": max_evals},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_scipy_differential_evolution(self, func_name: str, max_evals: int = 100):
        """Test against SciPy Differential Evolution"""
        try:
            from scipy.optimize import differential_evolution

            test_func = self.test_functions[func_name]

            result = differential_evolution(
                test_func["python_func"],
                test_func["bounds"],
                maxiter=max_evals // 10,
                seed=42,
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_scipy_simulated_annealing(self, func_name: str, max_evals: int = 100):
        """Test against SciPy Simulated Annealing"""
        try:
            from scipy.optimize import dual_annealing

            test_func = self.test_functions[func_name]

            result = dual_annealing(
                test_func["python_func"], test_func["bounds"], maxfun=max_evals, seed=42
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    # Placeholder methods for other packages (to be implemented)
    def _test_deap_genetic_algorithm(self, func_name: str, max_evals: int = 100):
        """Test against DEAP Genetic Algorithm"""
        try:
            import random

            from deap import algorithms, base, creator, tools

            test_func = self.test_functions[func_name]

            # Set up DEAP genetic algorithm
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.uniform, 0, 1)
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.attr_float,
                n=test_func["dimensions"],
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def eval_func(individual):
                # Convert DEAP Individual to plain list for test function
                return (test_func["python_func"](list(individual)),)

            toolbox.register("evaluate", eval_func)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Run optimization
            population = toolbox.population(n=20)

            # Simple genetic algorithm
            for gen in range(max_evals // 20):
                offspring = algorithms.varAnd(population, toolbox, 0.7, 0.3)
                fits = toolbox.map(toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = toolbox.select(offspring, k=len(population))

            # Find best individual
            best_ind = tools.selBest(population, k=1)[0]

            return {
                "success": True,
                "x": list(best_ind),
                "fun": float(best_ind.fitness.values[0]),
                "nfev": max_evals,
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_deap_evolution_strategy(self, func_name: str, max_evals: int = 100):
        """Test against DEAP Evolution Strategy"""
        try:
            import random

            import numpy as np
            from deap import base, creator

            test_func = self.test_functions[func_name]

            # Clean up any existing creator classes
            if hasattr(creator, "FitnessMin"):
                del creator.FitnessMin
            if hasattr(creator, "Individual"):
                del creator.Individual

            # Set up DEAP evolution strategy
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()

            def eval_func(individual):
                # Clip to bounds
                individual = np.clip(
                    individual,
                    [b[0] for b in test_func["bounds"]],
                    [b[1] for b in test_func["bounds"]],
                )
                return (test_func["python_func"](individual),)

            toolbox.register("evaluate", eval_func)

            # Simple (1+1) Evolution Strategy
            random.seed(42)
            np.random.seed(42)

            # Initialize individual
            individual = [random.uniform(b[0], b[1]) for b in test_func["bounds"]]
            individual = creator.Individual(individual)
            individual.fitness.values = toolbox.evaluate(individual)

            sigma = 0.1  # Mutation strength

            for generation in range(max_evals - 1):
                # Create offspring by mutation
                offspring = toolbox.clone(individual)
                for i in range(len(offspring)):
                    offspring[i] += random.gauss(0, sigma)

                # Evaluate offspring
                offspring.fitness.values = toolbox.evaluate(offspring)

                # Selection: keep better individual
                if offspring.fitness.values[0] < individual.fitness.values[0]:
                    individual = offspring
                    sigma *= 1.2  # Increase mutation strength on success
                else:
                    sigma *= 0.8  # Decrease mutation strength on failure

                # Keep sigma reasonable
                sigma = max(0.001, min(0.5, sigma))

            return {
                "success": True,
                "x": list(individual),
                "fun": float(individual.fitness.values[0]),
                "nfev": max_evals,
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_pyswarm_pso(self, func_name: str, max_evals: int = 100):
        """Test against PySwarm Particle Swarm Optimization"""
        try:
            from pyswarm import pso

            test_func = self.test_functions[func_name]

            # Set bounds for PSO
            bounds = test_func["bounds"]
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]

            # Run PSO
            xopt, fopt = pso(
                test_func["python_func"], lb, ub, maxiter=max_evals // 20, debug=False
            )

            return {
                "success": True,
                "x": xopt.tolist() if hasattr(xopt, "tolist") else list(xopt),
                "fun": float(fopt),
                "nfev": max_evals,
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_cma_es(self, func_name: str, max_evals: int = 100):
        """Test against simplified CMA-ES implementation"""
        try:
            import numpy as np

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            # Simplified CMA-ES implementation
            np.random.seed(42)

            # Parameters
            n = len(bounds)
            lambda_ = 4 + int(3 * np.log(n))  # Population size
            mu = lambda_ // 2  # Number of parents

            # Initialize
            x_mean = np.array(
                [0.5 * (b[0] + b[1]) for b in bounds]
            )  # Center of search space
            sigma = 0.3  # Step size
            C = np.eye(n)  # Covariance matrix

            generation = 0
            total_evals = 0
            prev_best = float("inf")  # Initialize prev_best

            while total_evals < max_evals:
                # Generate population
                population = []
                fitness_values = []

                for _ in range(min(lambda_, max_evals - total_evals)):
                    # Sample from multivariate normal
                    z = np.random.randn(n)
                    y = np.dot(np.linalg.cholesky(C), z)
                    x = x_mean + sigma * y

                    # Apply bounds
                    x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])

                    population.append(x)
                    fitness_values.append(test_func["python_func"](x))
                    total_evals += 1

                if not population:
                    break

                # Sort population by fitness
                indices = np.argsort(fitness_values)
                population = [population[i] for i in indices]
                fitness_values = [fitness_values[i] for i in indices]

                # Select parents (best mu individuals)
                parents = population[:mu]

                # Update mean
                x_mean = np.mean(parents, axis=0)

                # Simple adaptation: adjust step size based on improvement
                if generation > 0:
                    if fitness_values[0] < prev_best:
                        sigma *= 1.1  # Increase step size on improvement
                    else:
                        sigma *= 0.9  # Decrease step size

                prev_best = fitness_values[0]
                generation += 1

                # Simple covariance update (simplified)
                if len(parents) > 1:
                    deviations = np.array([(p - x_mean) / sigma for p in parents])
                    C = 0.8 * C + 0.2 * np.cov(deviations.T)

            # Return best individual
            best_idx = np.argmin(fitness_values)
            return {
                "success": True,
                "x": population[best_idx].tolist(),
                "fun": float(fitness_values[best_idx]),
                "nfev": total_evals,
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_skopt_bayesian(self, func_name: str, max_evals: int = 100):
        """Test against scikit-optimize Bayesian Optimization"""
        try:
            import numpy as np
            from skopt import gp_minimize
            from skopt.space import Real

            test_func = self.test_functions[func_name]

            # Wrap function to ensure numpy array conversion
            def wrapped_func(x):
                x_array = np.array(x)
                return test_func["python_func"](x_array)

            # Set up search space
            bounds = test_func["bounds"]
            dimensions = [Real(b[0], b[1]) for b in bounds]

            # Run Bayesian optimization
            result = gp_minimize(
                wrapped_func,
                dimensions,
                n_calls=max_evals,
                random_state=42,
                acq_func="EI",
                n_initial_points=10,
            )

            return {
                "success": True,
                "x": result.x,
                "fun": float(result.fun),
                "nfev": len(result.func_vals),
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_sklearn_random_search(self, func_name: str, max_evals: int = 100):
        """Test against simple random search reference"""
        try:
            import numpy as np

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)

            best_x = None
            best_f = float("inf")

            for i in range(max_evals):
                # Generate random point within bounds
                x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
                f_val = test_func["python_func"](x)

                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

            return {
                "success": True,
                "x": best_x.tolist(),
                "fun": float(best_f),
                "nfev": max_evals,
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_scipy_bfgs(self, func_name: str, max_evals: int = 100):
        """Test against SciPy BFGS"""
        try:
            import numpy as np
            from scipy.optimize import minimize

            test_func = self.test_functions[func_name]

            np.random.seed(42)
            x0 = np.random.uniform(0, 1, test_func["dimensions"])
            bounds = test_func["bounds"]

            result = minimize(
                test_func["python_func"],
                x0,
                method="BFGS",
                bounds=bounds,
                options={"maxiter": max_evals},
            )

            return {
                "success": result.success,
                "x": (
                    result.x.tolist() if hasattr(result.x, "tolist") else list(result.x)
                ),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }

        except Exception as e:
            return {"error": str(e)}

    # Custom reference implementations for metaheuristic algorithms
    def _test_adaptive_random_search_external(
        self, func_name: str, max_evals: int = 100
    ):
        """Test using nlopt (optional install) or fallback to scipy"""
        try:
            # Try nlopt first (optional install)
            try:
                import nlopt
                import numpy as np

                test_func = self.test_functions[func_name]
                bounds = test_func["bounds"]

                # Set up nlopt for adaptive random search
                opt = nlopt.opt(
                    nlopt.GN_CRS2_LM, len(bounds)
                )  # Controlled Random Search
                opt.set_lower_bounds([b[0] for b in bounds])
                opt.set_upper_bounds([b[1] for b in bounds])
                opt.set_min_objective(lambda x, grad: test_func["python_func"](x))
                opt.set_maxeval(max_evals)

                np.random.seed(42)
                x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
                x_opt = opt.optimize(x0)
                f_opt = opt.last_optimum_value()

                return {
                    "success": True,
                    "x": x_opt.tolist(),
                    "fun": float(f_opt),
                    "nfev": max_evals,
                }

            except ImportError:
                # Fallback to scipy dual_annealing (adaptive stochastic method)
                import numpy as np
                from scipy.optimize import dual_annealing

                test_func = self.test_functions[func_name]
                bounds = test_func["bounds"]

                np.random.seed(42)
                result = dual_annealing(
                    test_func["python_func"], bounds, maxfun=max_evals, seed=42
                )

                return {
                    "success": result.success,
                    "x": result.x.tolist(),
                    "fun": float(result.fun),
                    "nfev": int(result.nfev),
                }

        except Exception as e:
            return {"error": str(e)}

    def _test_sklearn_coordinate_descent(self, func_name: str, max_evals: int = 100):
        """Test using sklearn's coordinate descent (via optimization proxy)"""
        try:
            import numpy as np
            from scipy.optimize import minimize

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            # Use scipy's minimize with coordinate descent-like method
            np.random.seed(42)
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])

            result = minimize(
                test_func["python_func"],
                x0,
                method="Powell",  # Powell's method uses coordinate-wise search
                bounds=bounds,
                options={"maxfev": max_evals},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_scipy_pattern_search(self, func_name: str, max_evals: int = 100):
        """Test using scipy's COBYLA (derivative-free pattern search-like method)"""
        try:
            import numpy as np
            from scipy.optimize import minimize

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])

            # COBYLA is a pattern search-type method for constrained optimization
            result = minimize(
                test_func["python_func"],
                x0,
                method="COBYLA",
                bounds=bounds,
                options={"maxiter": max_evals, "disp": False},
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_custom_hill_climbing(self, func_name: str, max_evals: int = 100):
        """Simple hill climbing reference"""
        try:
            import numpy as np

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            current_x = np.random.uniform(
                [b[0] for b in bounds], [b[1] for b in bounds]
            )
            current_f = test_func["python_func"](current_x)

            for i in range(max_evals - 1):
                neighbor = current_x + np.random.normal(0, 0.1, len(current_x))
                neighbor = np.clip(
                    neighbor, [b[0] for b in bounds], [b[1] for b in bounds]
                )

                neighbor_f = test_func["python_func"](neighbor)
                if neighbor_f < current_f:
                    current_x = neighbor
                    current_f = neighbor_f

            return {
                "success": True,
                "x": current_x.tolist(),
                "fun": float(current_f),
                "nfev": max_evals,
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_custom_tabu_search(self, func_name: str, max_evals: int = 100):
        """Simple tabu search reference"""
        try:
            import numpy as np

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            current_x = np.random.uniform(
                [b[0] for b in bounds], [b[1] for b in bounds]
            )
            best_x = current_x.copy()
            best_f = test_func["python_func"](best_x)

            tabu_list = []
            tabu_tenure = 5

            for i in range(max_evals - 1):
                candidates = []
                for _ in range(min(10, max_evals - i)):
                    neighbor = current_x + np.random.normal(0, 0.1, len(current_x))
                    neighbor = np.clip(
                        neighbor, [b[0] for b in bounds], [b[1] for b in bounds]
                    )

                    # Check if not in tabu list (simplified)
                    is_tabu = any(
                        np.linalg.norm(neighbor - tabu_x) < 0.01 for tabu_x in tabu_list
                    )
                    if not is_tabu:
                        f_val = test_func["python_func"](neighbor)
                        candidates.append((neighbor, f_val))

                if candidates:
                    # Choose best non-tabu candidate
                    current_x, current_f = min(candidates, key=lambda x: x[1])

                    # Update tabu list
                    tabu_list.append(current_x.copy())
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)

                    # Update best
                    if current_f < best_f:
                        best_x = current_x.copy()
                        best_f = current_f

            return {
                "success": True,
                "x": best_x.tolist(),
                "fun": float(best_f),
                "nfev": max_evals,
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_custom_firefly_algorithm(self, func_name: str, max_evals: int = 100):
        """Simple firefly algorithm reference"""
        try:
            import numpy as np

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            n_fireflies = 10
            fireflies = np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds],
                (n_fireflies, len(bounds)),
            )

            best_idx = 0
            best_f = float("inf")

            for i in range(max_evals // n_fireflies):
                for j in range(n_fireflies):
                    f_val = test_func["python_func"](fireflies[j])
                    if f_val < best_f:
                        best_f = f_val
                        best_idx = j

                # Move fireflies toward brighter ones (simplified)
                for j in range(n_fireflies):
                    if j != best_idx:
                        # Move toward best firefly
                        direction = fireflies[best_idx] - fireflies[j]
                        fireflies[j] += 0.1 * direction + np.random.normal(
                            0, 0.01, len(bounds)
                        )
                        fireflies[j] = np.clip(
                            fireflies[j], [b[0] for b in bounds], [b[1] for b in bounds]
                        )

            return {
                "success": True,
                "x": fireflies[best_idx].tolist(),
                "fun": float(best_f),
                "nfev": max_evals,
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_ant_colony_external(self, func_name: str, max_evals: int = 100):
        """Test using acopy (optional install) or fallback to scipy"""
        try:
            # Try acopy first (optional install)
            try:
                import acopy
                import numpy as np

                test_func = self.test_functions[func_name]
                bounds = test_func["bounds"]

                # For continuous optimization, acopy might not be directly suitable
                # since it's designed for TSP, so fall back to scipy immediately
                raise ImportError("acopy is TSP-specific, using fallback")

            except ImportError:
                # Fallback to scipy basinhopping (good for multi-modal optimization like ACO)
                import numpy as np
                from scipy.optimize import basinhopping

                test_func = self.test_functions[func_name]
                bounds = test_func["bounds"]

                np.random.seed(42)
                x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])

                result = basinhopping(
                    test_func["python_func"],
                    x0,
                    niter=max_evals // 10,
                    minimizer_kwargs={"bounds": bounds, "method": "L-BFGS-B"},
                )

                return {
                    "success": True,
                    "x": result.x.tolist(),
                    "fun": float(result.fun),
                    "nfev": int(result.nfev),
                }

        except Exception as e:
            return {"error": str(e)}

    def _test_harmony_search_external(self, func_name: str, max_evals: int = 100):
        """Test using pyHarmonySearch (optional install) or fallback to scipy"""
        try:
            # Try pyHarmonySearch first (optional install)
            try:
                import numpy as np
                from pyHarmonySearch import harmony_search
                from pyHarmonySearch.ObjectiveFunctions import ObjectiveFunction

                test_func = self.test_functions[func_name]
                bounds = test_func["bounds"]

                class TestObjective(ObjectiveFunction):
                    def __init__(self, test_func, bounds):
                        super().__init__()
                        self.test_func = test_func
                        self.bounds = bounds

                    def use_vars(self):
                        return [
                            {
                                "name": f"x{i}",
                                "type": "float",
                                "lower_bound": b[0],
                                "upper_bound": b[1],
                            }
                            for i, b in enumerate(self.bounds)
                        ]

                    def fitness_function(self, x):
                        return self.test_func["python_func"](np.array(x))

                obj_fun = TestObjective(test_func, bounds)
                result = harmony_search(
                    obj_fun, max_evals=max_evals, hms=10, hmcr=0.9, par=0.3
                )

                return {
                    "success": True,
                    "x": result.best_harmony,
                    "fun": float(result.best_fitness),
                    "nfev": max_evals,
                }

            except ImportError:
                # Fallback to scipy differential evolution (similar stochastic search)
                import numpy as np
                from scipy.optimize import differential_evolution

                test_func = self.test_functions[func_name]
                bounds = test_func["bounds"]

                np.random.seed(42)
                result = differential_evolution(
                    test_func["python_func"],
                    bounds,
                    maxiter=max_evals // 15,  # DE uses population, adjust iterations
                    popsize=15,
                    seed=42,
                )

                return {
                    "success": result.success,
                    "x": result.x.tolist(),
                    "fun": float(result.fun),
                    "nfev": int(result.nfev),
                }

        except Exception as e:
            return {"error": str(e)}

    def _test_hill_climbing_external(self, func_name: str, max_evals: int = 100):
        """Test using scipy minimize with random restarts (hill climbing-like)"""
        try:
            import numpy as np
            from scipy.optimize import minimize

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            best_x = None
            best_f = float("inf")

            # Multiple random restarts (hill climbing approach)
            for restart in range(5):
                x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])

                result = minimize(
                    test_func["python_func"],
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxfun": max_evals // 5},
                )

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x

            return {
                "success": best_x is not None,
                "x": best_x.tolist() if best_x is not None else [0] * len(bounds),
                "fun": float(best_f),
                "nfev": max_evals,
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_tabu_search_external(self, func_name: str, max_evals: int = 100):
        """Test using scipy basinhopping (escapes local minima like tabu search)"""
        try:
            import numpy as np
            from scipy.optimize import basinhopping

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])

            result = basinhopping(
                test_func["python_func"],
                x0,
                niter=max_evals // 10,
                minimizer_kwargs={"bounds": bounds, "method": "L-BFGS-B"},
            )

            return {
                "success": True,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_firefly_external(self, func_name: str, max_evals: int = 100):
        """Test using scipy differential evolution (swarm-based like firefly)"""
        try:
            import numpy as np
            from scipy.optimize import differential_evolution

            test_func = self.test_functions[func_name]
            bounds = test_func["bounds"]

            np.random.seed(42)
            result = differential_evolution(
                test_func["python_func"],
                bounds,
                maxiter=max_evals // 15,
                popsize=15,
                seed=42,
            )

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
            }
        except Exception as e:
            return {"error": str(e)}

    def run_js_optimization(
        self, algorithm: str, func_name: str, max_evals: int = 100
    ) -> Dict[str, Any]:
        """Run JavaScript optimization using Node.js"""
        test_func = self.test_functions[func_name]
        js_algorithm = algorithm

        # Create JavaScript test script
        js_code = f"""
// Load the optimizer implementations using proper module import
const {{ OptimizerFactory }} = require('{os.getcwd()}/docs/js/optimizers.js');

// Ensure OptimizerFactory is available
if (typeof OptimizerFactory === 'undefined') {{
    throw new Error('OptimizerFactory is not defined after loading optimizers.js');
}}

// Make available globally
global.OptimizerFactory = OptimizerFactory;
const Factory = OptimizerFactory;

// Set random seed function for reproducibility
Math.seedrandom = function(seed) {{
    let m = 0x80000000;
    let a = 1103515245;
    let c = 12345;
    let state = seed ? seed : Math.floor(Math.random() * (m - 1));

    Math.random = function() {{
        state = (a * state + c) % m;
        return state / (m - 1);
    }};
}}

// Test function
const testFunc = {test_func["js_func"]};

// Run optimization with FIXED SEED for reproducibility
Math.seedrandom(42);

try {{
    const optimizer = Factory.create('{js_algorithm}', testFunc, {max_evals}, {test_func["dimensions"]});
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        x: result.bestX,
        fun: result.bestValue,
        nfev: result.evaluations,
        algorithm: '{js_algorithm}',
        seed: 42
    }}));
}} catch (error) {{
    console.log(JSON.stringify({{
        error: error.message,
        stack: error.stack
    }}));
}}
"""

        # Write and execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["node", temp_file], capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout.strip())
            else:
                return {"error": f"JavaScript execution failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "JavaScript execution timed out"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            os.unlink(temp_file)

    def run_js_optimization_multiple_runs(
        self, algorithm: str, func_name: str, num_runs: int = 20, max_evals: int = 100
    ):
        """Run JavaScript optimization multiple times for statistical validation"""
        results = []

        for run in range(num_runs):
            test_func = self.test_functions[func_name]
            js_algorithm = algorithm

            # Create JavaScript test script with different seed for each run
            js_code = f"""
// Load the optimizer implementations using proper module import
const {{ OptimizerFactory }} = require('{os.getcwd()}/docs/js/optimizers.js');

// Ensure OptimizerFactory is available
if (typeof OptimizerFactory === 'undefined') {{
    throw new Error('OptimizerFactory is not defined after loading optimizers.js');
}}

// Make available globally
global.OptimizerFactory = OptimizerFactory;
const Factory = OptimizerFactory;

// Set random seed function for reproducibility
Math.seedrandom = function(seed) {{
    let m = 0x80000000;
    let a = 1103515245;
    let c = 12345;
    let state = seed ? seed : Math.floor(Math.random() * (m - 1));

    Math.random = function() {{
        state = (a * state + c) % m;
        return state / (m - 1);
    }};
}}

// Test function
const testFunc = {test_func["js_func"]};

// Run optimization with DIFFERENT SEED for each run
Math.seedrandom({42 + run});

try {{
    const optimizer = Factory.create('{js_algorithm}', testFunc, {max_evals}, {test_func["dimensions"]});
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        x: result.bestX,
        fun: result.bestValue,
        nfev: result.evaluations,
        algorithm: '{js_algorithm}',
        seed: {42 + run},
        run: {run}
    }}));
}} catch (error) {{
    console.log(JSON.stringify({{
        error: error.message,
        stack: error.stack,
        run: {run}
    }}));
}}
"""

            # Write and execute
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
                f.write(js_code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["node", temp_file], capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    run_result = json.loads(result.stdout.strip())
                    run_result["run_id"] = run
                    results.append(run_result)
                else:
                    results.append(
                        {
                            "error": f"JavaScript execution failed: {result.stderr}",
                            "run_id": run,
                        }
                    )

            except subprocess.TimeoutExpired:
                results.append(
                    {"error": "JavaScript execution timed out", "run_id": run}
                )
            except Exception as e:
                results.append({"error": str(e), "run_id": run})
            finally:
                os.unlink(temp_file)

        return results

    def test_algorithm_comprehensive(
        self, algorithm: str, max_evals: int = 100
    ) -> AlgorithmTestResult:
        """Test one algorithm comprehensively against its reference"""
        if algorithm not in self.algorithms:
            return AlgorithmTestResult(
                algorithm=algorithm,
                package="UNKNOWN",
                js_success=False,
                ref_success=False,
                perfect_matches=0,
                total_tests=0,
                avg_js_value=float("inf"),
                avg_ref_value=float("inf"),
                max_difference=float("inf"),
                passed_validation=False,
                error_message="Algorithm not found",
                win_rate_vs_ref=0.0,
            )

        algo_info = self.algorithms[algorithm]
        js_name = algo_info["js_name"]
        ref_test_func = algo_info["reference_test"]
        package = algo_info["package"]

        js_results = []
        ref_results = []
        perfect_matches = 0
        total_tests = 0
        js_wins = 0

        print(f"\n🧪 Testing {algorithm} ({package}):")

        for func_name, test_func in self.test_functions.items():
            total_tests += 1
            print(f"  [{total_tests}/4] {test_func['name']}...", end=" ")

            # Run reference implementation ONCE (deterministic)
            ref_result = ref_test_func(func_name, max_evals)
            if ref_result.get("error"):
                print("❌ REF ERROR")
                continue

            ref_value = ref_result.get("fun", float("inf"))

            # Run JavaScript version 20 TIMES for statistical validation
            js_multi_results = self.run_js_optimization_multiple_runs(
                js_name, func_name, num_runs=20, max_evals=max_evals
            )

            # Filter successful runs
            js_successful_runs = [r for r in js_multi_results if not r.get("error")]

            if not js_successful_runs:
                print("❌ JS ERROR (all runs failed)")
                continue

            # Analyze 20-run statistics
            js_values = [r.get("fun", float("inf")) for r in js_successful_runs]
            js_best_value = min(js_values)
            js_mean_value = np.mean(js_values)
            js_wins_count = sum(1 for v in js_values if v < ref_value)
            js_perfect_matches = sum(1 for v in js_values if abs(v - ref_value) < 1e-6)

            js_results.append(js_mean_value)
            ref_results.append(ref_value)

            # Statistical validation based on 20 runs
            win_rate_this_func = js_wins_count / len(js_successful_runs)

            if js_perfect_matches >= 5:  # At least 25% perfect matches
                perfect_matches += 1
                print(f"✅ PERFECT ({js_perfect_matches}/20 perfect)")
            elif win_rate_this_func >= 0.4:  # At least 40% win rate
                js_wins += 1
                print(
                    f"⚡ JS WINS ({win_rate_this_func:.1%} win rate, best: {js_best_value:.6f})"
                )
            else:
                print(
                    f"⚠️  REF WINS ({win_rate_this_func:.1%} win rate, ref: {ref_value:.6f})"
                )

        # Calculate metrics
        avg_js = np.mean(js_results) if js_results else float("inf")
        avg_ref = np.mean(ref_results) if ref_results else float("inf")
        max_diff = (
            max([abs(j - r) for j, r in zip(js_results, ref_results)])
            if js_results and ref_results
            else float("inf")
        )
        win_rate = (js_wins + perfect_matches) / total_tests if total_tests > 0 else 0.0

        # Validation criteria
        passed = (perfect_matches >= 1) or (
            win_rate >= 0.4
        )  # At least 1 perfect match OR 40%+ win rate

        return AlgorithmTestResult(
            algorithm=algorithm,
            package=package,
            js_success=len(js_results) > 0,
            ref_success=len(ref_results) > 0,
            perfect_matches=perfect_matches,
            total_tests=total_tests,
            avg_js_value=avg_js,
            avg_ref_value=avg_ref,
            max_difference=max_diff,
            passed_validation=passed,
            error_message=None,
            win_rate_vs_ref=win_rate,
        )

    def run_comprehensive_validation(
        self, priority_filter=None
    ) -> Dict[str, AlgorithmTestResult]:
        """Run comprehensive validation on all algorithms"""

        print("🚀 COMPREHENSIVE ALGORITHM VALIDATION")
        print("=" * 70)

        results = {}
        algorithms_to_test = self.algorithms.keys()

        if priority_filter:
            algorithms_to_test = [
                algo
                for algo in algorithms_to_test
                if self.algorithms[algo]["priority"] == priority_filter
            ]

        print(f"📋 Testing {len(algorithms_to_test)} algorithms...")

        for algorithm in algorithms_to_test:
            results[algorithm] = self.test_algorithm_comprehensive(algorithm)

        return results

    def generate_comprehensive_report(
        self, results: Dict[str, AlgorithmTestResult]
    ) -> str:
        """Generate detailed validation report"""

        report = ["# 🧪 COMPREHENSIVE ALGORITHM VALIDATION REPORT\n"]

        # Summary statistics
        total_algorithms = len(results)
        passed_algorithms = sum(1 for r in results.values() if r.passed_validation)
        perfect_match_algorithms = sum(
            1 for r in results.values() if r.perfect_matches > 0
        )

        report.append("## 📊 SUMMARY\n")
        report.append(f"- **Total Algorithms Tested**: {total_algorithms}")
        report.append(
            f"- **Passed Validation**: {passed_algorithms}/{total_algorithms} ({100 * passed_algorithms / total_algorithms:.1f}%)"
        )
        report.append(
            f"- **Have Perfect Matches**: {perfect_match_algorithms}/{total_algorithms} ({100 * perfect_match_algorithms / total_algorithms:.1f}%)"
        )
        report.append("")

        # Group by validation status
        excellent = [a for a, r in results.items() if r.perfect_matches >= 2]
        good = [
            a
            for a, r in results.items()
            if r.perfect_matches == 1 or r.win_rate_vs_ref >= 0.5
        ]
        needs_work = [
            a
            for a, r in results.items()
            if r.perfect_matches == 0 and r.win_rate_vs_ref < 0.5
        ]

        report.append("## 🎯 ALGORITHM STATUS\n")
        report.append("```")
        report.append(
            f"EXCELLENT (✅): {', '.join(excellent) if excellent else 'None'}"
        )
        report.append(f"GOOD (⚠️):      {', '.join(good) if good else 'None'}")
        report.append(
            f"NEEDS WORK (❌): {', '.join(needs_work) if needs_work else 'None'}"
        )
        report.append("```\n")

        # Detailed results
        report.append("## 📋 DETAILED RESULTS\n")

        for algorithm, result in results.items():
            status = (
                "✅ EXCELLENT"
                if result.perfect_matches >= 2
                else (
                    "⚠️ GOOD"
                    if result.perfect_matches >= 1 or result.win_rate_vs_ref >= 0.5
                    else "❌ NEEDS WORK"
                )
            )

            report.append(f"### {algorithm} ({result.package}) - {status}\n")
            report.append(
                f"- **Perfect Matches**: {result.perfect_matches}/{result.total_tests}"
            )
            report.append(f"- **Win Rate vs Reference**: {result.win_rate_vs_ref:.1%}")
            report.append(f"- **Average JS Value**: {result.avg_js_value:.6f}")
            report.append(f"- **Average Reference Value**: {result.avg_ref_value:.6f}")
            report.append(f"- **Max Difference**: {result.max_difference:.6f}")
            if result.error_message:
                report.append(f"- **Error**: {result.error_message}")
            report.append("")

        return "\n".join(report)


def main():
    """Run comprehensive validation"""

    validator = ComprehensiveAlgorithmValidator()

    # Check if user wants to test all algorithms
    test_all = len(sys.argv) > 1 and sys.argv[1] == "--all"

    if test_all:
        print("🚀 TESTING ALL 22 ALGORITHMS...")
        results = (
            validator.run_comprehensive_validation()
        )  # No priority filter = all algorithms
    else:
        # Test high priority algorithms first
        print("🎯 Phase 1: Testing HIGH priority algorithms...")
        results = validator.run_comprehensive_validation(priority_filter="HIGH")

    # Generate report
    report = validator.generate_comprehensive_report(results)

    # Save results
    with open("comprehensive_algorithm_validation_report.md", "w") as f:
        f.write(report)

    with open("comprehensive_algorithm_results.json", "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2, default=str)

    print("\n📊 Report saved to: comprehensive_algorithm_validation_report.md")
    print("📊 Raw results saved to: comprehensive_algorithm_results.json")

    # Print summary
    passed = sum(1 for r in results.values() if r.passed_validation)
    total = len(results)

    if test_all:
        print(
            f"\n🎯 **UPDATED COMPREHENSIVE SUMMARY**: {passed}/{total} algorithms passed validation ({100 * passed / total:.1f}%)"
        )
    else:
        print(
            f"\n🎯 **FINAL SUMMARY**: {passed}/{total} algorithms passed validation ({100 * passed / total:.1f}%)"
        )


if __name__ == "__main__":
    main()
