"""
Signal deconvolution: recover a sharp signal from a blurred, noisy measurement.

A length-12 signal is observed only after convolution with a Gaussian blur and a small
additive perturbation. We reconstruct the signal by choosing its twelve samples to fit
the observation, regularised by a smoothness term that penalises large jumps between
adjacent samples. This is the classic regularised least-squares inverse problem: smooth
and near-convex, with a single well-defined optimum a good optimizer should reach.

The HumpDay objective takes a 12-D point in [0,1]^12 (samples, scaled to [0, 1.5]) and
returns the squared reconstruction residual plus the smoothness penalty.
"""
from __future__ import annotations

import math

N = 12
N_DIM = N
SIGMA = 1.5
LAMBDA = 0.05
X_TRUE = (0.0, 0.0, 0.2, 0.8, 1.0, 0.6, 0.1, 0.0, 0.4, 0.9, 0.3, 0.0)

# Gaussian blur matrix, rows normalised to sum to one.
A = []
for i in range(N):
    row = [math.exp(-((i - j) ** 2) / (2 * SIGMA ** 2)) for j in range(N)]
    s = sum(row)
    A.append([v / s for v in row])


def _matvec(x):
    return [sum(A[i][j] * x[j] for j in range(N)) for i in range(N)]


# Observation: blurred truth plus a small deterministic perturbation.
Y = [v + 0.02 * math.sin(2.0 * i) for i, v in enumerate(_matvec(list(X_TRUE)))]


def decode(u):
    return [1.5 * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    x = decode(u)
    rec = _matvec(x)
    residual = sum((rec[i] - Y[i]) ** 2 for i in range(N))
    smooth = sum((x[i] - x[i - 1]) ** 2 for i in range(1, N))
    return residual + LAMBDA * smooth
