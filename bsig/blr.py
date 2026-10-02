"""Bayesian linear regression with a conjugate Normal–Inverse-Gamma prior and exact online updates."""
from __future__ import annotations

import numpy as np
from scipy import stats


class BayesianLinearRegression:
    """w | σ² ~ N(m, σ² V),  σ² ~ InvGamma(a, b).

    `update` folds in new observations exactly (no refitting), so the model can learn online as data
    arrives. `predict` returns the posterior predictive Student-t mean and standard deviation, which
    can scale positions by confidence.
    """

    def __init__(self, dim: int, prior_scale: float = 1.0, a0: float = 1.0, b0: float = 1.0, forgetting: float = 1.0):
        self.m = np.zeros(dim)
        self.Vinv = np.eye(dim) / prior_scale**2
        self.a, self.b = a0, b0
        self.lam = forgetting  # < 1 discounts old data (adapts to drifting relationships)

    def update(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        X, y = np.atleast_2d(X), np.atleast_1d(y)
        Vinv0, m0 = self.lam * self.Vinv, self.m
        Vinv = Vinv0 + X.T @ X
        m = np.linalg.solve(Vinv, Vinv0 @ m0 + X.T @ y)
        self.a = self.lam * self.a + len(y) / 2
        self.b = self.lam * self.b + 0.5 * (y @ y + m0 @ Vinv0 @ m0 - m @ Vinv @ m)
        self.m, self.Vinv = m, Vinv
        return self

    def predict(self, X: np.ndarray):
        X = np.atleast_2d(X)
        V = np.linalg.inv(self.Vinv)
        mean = X @ self.m
        scale2 = self.b / self.a * (1 + np.einsum("ij,jk,ik->i", X, V, X))
        dof = 2 * self.a
        sd = np.sqrt(scale2 * dof / (dof - 2)) if dof > 2 else np.full(len(X), np.inf)
        return mean, sd

    def coef_interval(self, level: float = 0.95):
        V = np.linalg.inv(self.Vinv)
        t = stats.t.ppf(0.5 + level / 2, 2 * self.a)
        half = t * np.sqrt(np.diag(V) * self.b / self.a)
        return self.m - half, self.m + half
