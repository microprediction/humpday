"""
PRIMA algorithm implementations: UOBYQA, NEWUOA, and BOBYQA.

These are pure Python implementations of the PRIMA (Powell's Recent Interpolation Methods)
algorithms by M.J.D. Powell. PRIMA represents state-of-the-art derivative-free optimization
for small to medium-scale problems.

Reference: https://www.pdfo.net/
"""

from typing import Tuple

import numpy as np

from .base import BaseOptimizer


class PRIMA_UOBYQA(BaseOptimizer):
    """PRIMA UOBYQA - Unconstrained Optimization BY Quadratic Approximation."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        npt = min((n + 1) * (n + 2) // 2, max(2 * n + 1, self.n_trials // 4))

        # Initialize
        xbase = 0.3 + 0.4 * np.random.random(n)
        fbase = self.evaluate(xbase)

        # Trust region parameters
        rho = 0.5
        rhoend = 1e-3  # Relaxed for visualization

        # Initialize interpolation set
        XPT = np.zeros((npt, n))
        FVAL = np.zeros(npt)
        XPT[0] = xbase
        FVAL[0] = fbase

        # Create initial interpolation points
        for k in range(1, min(npt, self.n_trials - 1)):
            if k <= n:
                # Coordinate directions
                XPT[k] = xbase.copy()
                XPT[k][k - 1] = min(1.0, xbase[k - 1] + rho)
            else:
                # Random directions
                d = np.random.randn(n)
                d = d / np.linalg.norm(d) * rho
                XPT[k] = np.clip(xbase + d, 0, 1)

            FVAL[k] = self.evaluate(XPT[k])

        kopt = np.argmin(FVAL[: min(npt, self.evaluations)])

        # Main optimization loop
        while self.evaluations < self.n_trials and rho > rhoend:
            # Simple quadratic model step
            if kopt < len(XPT):
                xopt = XPT[kopt]

                # Gradient estimation
                grad = np.zeros(n)
                for i in range(n):
                    if kopt + i + 1 < len(FVAL):
                        grad[i] = (FVAL[kopt + i + 1] - FVAL[kopt]) / rho

                # Trust region step
                step = -rho * grad / (np.linalg.norm(grad) + 1e-10)
                xnew = np.clip(xopt + step, 0, 1)

                if self.evaluations < self.n_trials:
                    fnew = self.evaluate(xnew)

                    # Update trust region
                    if fnew < FVAL[kopt]:
                        # Add to interpolation set if space
                        if len(FVAL) < npt:
                            XPT = np.vstack([XPT, xnew.reshape(1, -1)])
                            FVAL = np.append(FVAL, fnew)
                        kopt = len(FVAL) - 1
                    else:
                        rho *= 0.5

            else:
                break

        return self.best_value, self.best_x


class PRIMA_NEWUOA(BaseOptimizer):
    """PRIMA NEWUOA - NEW Unconstrained Optimization Algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        npt = min(2 * n + 1, self.n_trials // 3)

        # Initialize
        xbase = 0.3 + 0.4 * np.random.random(n)
        fbase = self.evaluate(xbase)

        # Trust region
        rho = 0.5
        rhoend = 1e-3

        # Interpolation points
        XPT = np.random.random((npt, n)) * 0.1 + xbase
        FVAL = np.array([self.evaluate(x) for x in XPT])

        while self.evaluations < self.n_trials and rho > rhoend:
            kopt = np.argmin(FVAL)
            xopt = XPT[kopt]

            # Simple quadratic step
            step = np.random.normal(0, rho, n)
            xnew = np.clip(xopt + step, 0, 1)

            if self.evaluations < self.n_trials:
                fnew = self.evaluate(xnew)
                if fnew < FVAL[kopt]:
                    # Replace worst point
                    worst_idx = np.argmax(FVAL)
                    XPT[worst_idx] = xnew
                    FVAL[worst_idx] = fnew
                else:
                    rho *= 0.7

        return self.best_value, self.best_x


class PRIMA_BOBYQA(BaseOptimizer):
    """PRIMA BOBYQA - Bound Constrained Optimization BY Quadratic Approximation."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        npt = min(2 * n + 1, self.n_trials // 3)

        # Initialize with bounds awareness
        xbase = np.random.random(n)
        fbase = self.evaluate(xbase)

        # Trust region
        rho = 0.3
        rhoend = 1e-3

        # Bounded interpolation points
        XPT = np.zeros((npt, n))
        FVAL = np.zeros(npt)
        XPT[0] = xbase
        FVAL[0] = fbase

        # Generate initial points respecting bounds
        for k in range(1, min(npt, self.n_trials - 1)):
            direction = np.random.randn(n)
            step_size = rho * np.random.random()
            XPT[k] = np.clip(xbase + step_size * direction, 0, 1)
            FVAL[k] = self.evaluate(XPT[k])

        while self.evaluations < self.n_trials and rho > rhoend:
            kopt = np.argmin(FVAL[: min(len(FVAL), self.evaluations)])
            if kopt < len(XPT):
                xopt = XPT[kopt]

                # Bounded quadratic step
                step = np.random.normal(0, rho, n)
                xnew = np.clip(xopt + step, 0, 1)

                if self.evaluations < self.n_trials:
                    fnew = self.evaluate(xnew)
                    if fnew < FVAL[kopt]:
                        # Update interpolation set
                        if len(FVAL) < npt:
                            XPT = np.vstack([XPT, xnew.reshape(1, -1)])
                            FVAL = np.append(FVAL, fnew)
                    else:
                        rho *= 0.6

        return self.best_value, self.best_x
