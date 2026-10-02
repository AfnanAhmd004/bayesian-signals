"""bayesian-signals: Bayesian and Gaussian-process signal models with statistically honest validation."""
from .blr import BayesianLinearRegression
from .data import make_data
from .gp import GaussianProcess
from .validation import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, sharpe

__all__ = ["BayesianLinearRegression", "GaussianProcess", "deflated_sharpe", "expected_max_sharpe", "make_data",
           "probabilistic_sharpe", "sharpe"]
