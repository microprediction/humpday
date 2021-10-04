
# DEAP package uses 2to3 which isn't supported
# This is causing problems, so the deap benchmarks are copied here for the time being
# Many thanks to the DEAP team for compiling them.

import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce


def rand(individual):
    return random.random(),


def plane(individual):
    return individual[0],


def sphere(individual):
    return sum(gene * gene for gene in individual),


def cigar(individual):
    return individual[0] ** 2 + 1e6 * sum(gene * gene for gene in individual),


def rosenbrock(individual):
    return sum(100 * (x * x - y) ** 2 + (1. - x) ** 2
               for x, y in zip(individual[:-1], individual[1:])),


def h1(individual):
    num = (sin(individual[0] - individual[1] / 8)) ** 2 + (sin(individual[1] + individual[0] / 8)) ** 2
    denum = ((individual[0] - 8.6998) ** 2 + (individual[1] - 6.7665) ** 2) ** 0.5 + 1
    return num / denum,


def ackley(individual):
    N = len(individual)
    return 20 - 20 * exp(-0.2 * sqrt(1.0 / N * sum(x ** 2 for x in individual))) \
           + e - exp(1.0 / N * sum(cos(2 * pi * x) for x in individual)),


def bohachevsky(individual):
    return sum(x ** 2 + 2 * x1 ** 2 - 0.3 * cos(3 * pi * x) - 0.4 * cos(4 * pi * x1) + 0.7
               for x, x1 in zip(individual[:-1], individual[1:])),


def griewank(individual):
    return 1.0 / 4000.0 * sum(x ** 2 for x in individual) - \
           reduce(mul, (cos(x / sqrt(i + 1.0)) for i, x in enumerate(individual)), 1) + 1,


def rastrigin(individual):
    return 10 * len(individual) + sum(gene * gene - 10 *
                                      cos(2 * pi * gene) for gene in individual),


def rastrigin_scaled(individual):
    N = len(individual)
    return 10 * N + sum((10 ** (i / (N - 1)) * x) ** 2 -
                        10 * cos(2 * pi * 10 ** (i / (N - 1)) * x) for i, x in enumerate(individual)),


def rastrigin_skew(individual):
    N = len(individual)
    return 10 * N + sum((10 * x if x > 0 else x) ** 2
                        - 10 * cos(2 * pi * (10 * x if x > 0 else x)) for x in individual),


def schaffer(individual):
    return sum((x ** 2 + x1 ** 2) ** 0.25 * ((sin(50 * (x ** 2 + x1 ** 2) ** 0.1)) ** 2 + 1.0)
               for x, x1 in zip(individual[:-1], individual[1:])),


def schwefel(individual):
    N = len(individual)
    return 418.9828872724339 * N - sum(x * sin(sqrt(abs(x))) for x in individual),


def himmelblau(individual):
    return (individual[0] * individual[0] + individual[1] - 11) ** 2 + \
           (individual[0] + individual[1] * individual[1] - 7) ** 2,


def shekel(individual, a, c):
    return sum((1. / (c[i] + sum((individual[j] - aij) ** 2 for j, aij in enumerate(a[i])))) for i in range(len(c))),

