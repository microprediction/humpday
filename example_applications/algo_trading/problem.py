"""
Z-score channel strategy with two-regime synthetic prices.

Pure-Python — no pandas, no vectorbt, no numpy required for the
core path (a tiny numpy fast-path is used only for Sharpe computation
when numpy is available). Designed to run in the browser too.
"""

from __future__ import annotations

import math
import random

N_DIM = 4
N_BARS = 2400  # synthetic series length
TRAIN_FRAC = 0.5  # in-sample portion
COST_BPS = 5.0  # round-trip transaction cost in basis points
ANNUALISATION = math.sqrt(252.0)

# Parameter bounds (after decoding from [0,1]^4).
LOOKBACK_LO, LOOKBACK_HI = 10, 200
ENTRY_LO, ENTRY_HI = 0.5, 3.0
EXIT_LO, EXIT_HI = 0.0, 2.0


def _generate_prices(seed=42):
    """Two-regime synthetic price series. First half: momentum (positive
    autocorrelation). Second half: mean-reversion (negative autocorrelation
    around a slowly drifting mean)."""
    rng = random.Random(seed)
    half = N_BARS // 2
    prices = [100.0]

    # Momentum regime — log-returns with positive autocorrelation.
    prev_ret = 0.0
    for _ in range(half):
        innov = rng.gauss(0.0002, 0.012)
        ret = 0.3 * prev_ret + innov
        prices.append(prices[-1] * math.exp(ret))
        prev_ret = ret

    # Mean-reversion regime — log-returns pull toward a slowly moving mean.
    target = prices[-1]
    for k in range(half):
        # Target drifts slowly upward over the regime.
        target *= math.exp(0.0001)
        innov = rng.gauss(0.0, 0.011)
        pull = 0.05 * (math.log(target) - math.log(prices[-1]))
        ret = pull + innov
        prices.append(prices[-1] * math.exp(ret))

    return prices


_PRICES = _generate_prices()
_SPLIT = int(len(_PRICES) * TRAIN_FRAC)


def _decode(u):
    """[0,1]^4 → (lookback, entry_z, exit_z, _unused)."""
    lookback = int(LOOKBACK_LO + (LOOKBACK_HI - LOOKBACK_LO) * u[0])
    entry_z = ENTRY_LO + (ENTRY_HI - ENTRY_LO) * u[1]
    exit_z = EXIT_LO + (EXIT_HI - EXIT_LO) * u[2]
    return lookback, entry_z, exit_z


def _backtest(prices, lookback, entry_z, exit_z):
    """Return the bar-by-bar return series produced by the z-channel
    strategy on `prices`. Long-only; cost charged on each open/close."""
    n = len(prices)
    if lookback >= n - 2 or lookback < 2:
        return []
    pos = 0  # 0 or 1
    returns = []
    cost = COST_BPS * 1e-4  # per side

    # Rolling sums for O(n) z-score.
    window = []
    sum_ = 0.0
    sum_sq = 0.0
    for i in range(n - 1):
        x = math.log(prices[i])
        window.append(x)
        sum_ += x
        sum_sq += x * x
        if len(window) > lookback:
            old = window.pop(0)
            sum_ -= old
            sum_sq -= old * old

        bar_return = 0.0
        if len(window) == lookback:
            mean = sum_ / lookback
            var = max(1e-12, sum_sq / lookback - mean * mean)
            sd = math.sqrt(var)
            z = (x - mean) / sd

            # Position decision based on the just-computed z.
            new_pos = pos
            if pos == 0 and z < -entry_z:
                new_pos = 1
            elif pos == 1 and z > exit_z:
                new_pos = 0

            # Realise the next bar's log-return if we were long this bar.
            if pos == 1:
                bar_return = math.log(prices[i + 1]) - x

            # Apply transaction cost on position change.
            if new_pos != pos:
                bar_return -= cost
            pos = new_pos

        returns.append(bar_return)

    return returns


def _sharpe(returns):
    """Annualised Sharpe ratio of a return stream. Returns 0.0 if degenerate."""
    n = len(returns)
    if n < 30:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    sd = math.sqrt(max(var, 1e-12))
    if sd < 1e-10:
        return 0.0
    return ANNUALISATION * mean / sd


def objective(u):
    """HumpDay objective: negative in-sample Sharpe of the strategy."""
    lookback, entry_z, exit_z = _decode(u)
    in_sample = _PRICES[: _SPLIT + 1]
    rets = _backtest(in_sample, lookback, entry_z, exit_z)
    return -_sharpe(rets)


def out_of_sample_sharpe(u):
    """Sharpe of the same strategy on the held-out second half."""
    lookback, entry_z, exit_z = _decode(u)
    oos = _PRICES[_SPLIT - LOOKBACK_HI :]  # include enough warm-up
    rets = _backtest(oos, lookback, entry_z, exit_z)
    return _sharpe(rets)


def decode(u):
    """Return the human-readable parameters."""
    lookback, entry_z, exit_z = _decode(u)
    return {"lookback": lookback, "entry_z": entry_z, "exit_z": exit_z}
