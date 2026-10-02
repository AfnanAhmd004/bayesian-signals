"""Gaussian-process regression with an ARD RBF kernel, hyperparameters fit by marginal likelihood."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class GaussianProcess:
    """k(x, x') = s² exp(-½ Σ_d (x_d − x'_d)² / ℓ_d²) + σ_n² δ.

    One length-scale per feature (automatic relevance determination): irrelevant features get long
    length-scales, which doubles as a nonlinear feature-relevance measure.
    """

    def __init__(self, n_restarts: int = 2, seed: int = 0):
        self.n_restarts, self.seed = n_restarts, seed

    def _K(self, A, B, ls, s2):
        d = (A[:, None, :] - B[None, :, :]) / ls
        return s2 * np.exp(-0.5 * np.sum(d * d, -1))

    def _nll(self, theta, X, y):
        D = X.shape[1]
        ls, s2, n2 = np.exp(theta[:D]), np.exp(theta[D]), np.exp(theta[D + 1])
        K = self._K(X, X, ls, s2) + (n2 + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve(c, y)
        return 0.5 * y @ alpha + np.sum(np.log(np.diag(c[0]))) + 0.5 * len(y) * np.log(2 * np.pi)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.X, self.y_mean = np.asarray(X, float), float(np.mean(y))
        y = np.asarray(y, float) - self.y_mean
        D = X.shape[1]
        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(self.n_restarts):
            theta0 = np.r_[np.log(np.full(D, 1.0) * (1 + r)), np.log(np.var(y) * 0.1), np.log(np.var(y) * 0.9)]
            theta0 += 0.1 * rng.standard_normal(len(theta0))
            bounds = [(-3, 5)] * D + [(-12, 3), (-6, 3)]  # cap length-scales: e^5 ≈ 150 = 'irrelevant'
            res = minimize(self._nll, theta0, args=(self.X, y), method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        D_ = D
        self.ls, self.s2, self.n2 = np.exp(best.x[:D_]), np.exp(best.x[D_]), np.exp(best.x[D_ + 1])
        K = self._K(self.X, self.X, self.ls, self.s2) + (self.n2 + 1e-8) * np.eye(len(self.X))
        self.chol = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.chol, y)
        return self

    def predict(self, Xs: np.ndarray):
        Ks = self._K(np.asarray(Xs, float), self.X, self.ls, self.s2)
        mean = Ks @ self.alpha + self.y_mean
        v = cho_solve(self.chol, Ks.T)
        var = self.s2 - np.sum(Ks * v.T, 1) + self.n2
        return mean, np.sqrt(np.maximum(var, 1e-12))
