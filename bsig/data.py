"""Synthetic prediction problem: 3 real signals (one nonlinear, one regime-dependent) hidden among 37 noise features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_data(n: int = 6000, n_noise: int = 37, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3 + n_noise))
    drift = np.where(np.arange(n) < n // 2, 1.0, 0.5)  # the linear edge decays in the second half
    signal = 0.10 * X[:, 0] * drift + 0.12 * np.tanh(2 * X[:, 1]) - 0.10 * (X[:, 2] ** 2 - 1)
    y = signal + rng.standard_normal(n)  # signal-to-noise far below 1, as in real return forecasting
    cols = ["mom", "flow", "vol_sq"] + [f"noise_{i}" for i in range(n_noise)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="fwd_ret"), signal
