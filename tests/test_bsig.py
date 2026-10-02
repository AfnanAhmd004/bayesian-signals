import numpy as np

from bsig import BayesianLinearRegression, GaussianProcess, deflated_sharpe, expected_max_sharpe, probabilistic_sharpe


def test_blr_online_equals_batch_and_recovers_weights():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3)); w = np.array([0.5, -1.0, 2.0])
    y = X @ w + 0.1 * rng.standard_normal(500)
    batch = BayesianLinearRegression(3).update(X, y)
    online = BayesianLinearRegression(3)
    for i in range(500):
        online.update(X[i], y[i])
    assert np.allclose(batch.m, online.m, atol=1e-8) and np.allclose(batch.m, w, atol=0.05)
    lo, hi = batch.coef_interval()
    assert np.all((lo < w) & (w < hi))


def test_blr_predictive_uncertainty_grows_off_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 1)); y = X[:, 0] + 0.1 * rng.standard_normal(200)
    blr = BayesianLinearRegression(1).update(X, y)
    _, sd = blr.predict(np.array([[0.0], [50.0]]))
    assert sd[1] > sd[0]


def test_gp_fits_nonlinear_function_and_ard_ignores_noise_dim():
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, (200, 2)); y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(200)
    gp = GaussianProcess().fit(X, y)
    mu, sd = gp.predict(np.array([[1.0, 0.0], [-1.5, 2.0]]))
    assert np.allclose(mu, np.sin([1.0, -1.5]), atol=0.15)
    assert gp.ls[1] > 5 * gp.ls[0]


def test_psr_and_deflation():
    rng = np.random.default_rng(3)
    good = rng.normal(0.1, 1.0, 2000)
    assert probabilistic_sharpe(good) > 0.99
    noise_srs = np.array([np.mean(r) / np.std(r) for r in rng.normal(0, 1, (300, 500))])
    best = rng.normal(0, 1, 500) + noise_srs.max()  # a "best" strategy that only reaches the selection hurdle
    assert expected_max_sharpe(300, noise_srs.var()) > 0
    assert deflated_sharpe(best, noise_srs) < probabilistic_sharpe(best)
