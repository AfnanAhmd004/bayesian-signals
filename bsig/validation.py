"""Statistical validation of a backtested Sharpe ratio: probabilistic and deflated Sharpe ratios.

Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012) and
"The Deflated Sharpe Ratio" (2014).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EULER = 0.5772156649


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r)
    return float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0


def probabilistic_sharpe(r: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true per-period SR > benchmark) given sample length, skew and kurtosis of returns."""
    r = np.asarray(r)
    sr, n = sharpe(r), len(r)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period SR among n_trials strategies with true SR 0 (the selection bias hurdle)."""
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(r: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """PSR against the Sharpe you would expect from the best of `len(trial_sharpes)` worthless strategies."""
    trial_sharpes = np.asarray(trial_sharpes)
    hurdle = expected_max_sharpe(len(trial_sharpes), np.var(trial_sharpes, ddof=1))
    return probabilistic_sharpe(r, hurdle)
