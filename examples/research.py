"""Find weak signals among noise, model them with uncertainty, and test whether the result is real."""
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ARDRegression

from bsig import BayesianLinearRegression, GaussianProcess, deflated_sharpe, make_data, probabilistic_sharpe, sharpe

X, y, _ = make_data(n=6000, seed=0)
half = len(X) // 2
Xtr, ytr = X.values[:half], y.values[:half]

# 1. feature selection: linear ARD vs mutual information (catches nonlinear effects)
ard = ARDRegression().fit(Xtr, ytr)
lin_sel = [X.columns[i] for i in np.argsort(-np.abs(ard.coef_))[:3]]
mi = mutual_info_regression(Xtr, ytr, random_state=0)
cand = list(np.argsort(-mi)[:6])
print("linear ARD top-3:      ", lin_sel)
print("mutual-information top-6:", [X.columns[i] for i in cand], "<- unreliable at this signal-to-noise")

# 2. GP with ARD length-scales: can it rank relevance (including nonlinear effects) on a controlled set?
cols = ["mom", "flow", "vol_sq", "noise_0", "noise_1", "noise_2"]
idx = [X.columns.get_loc(c) for c in cols]
gp = GaussianProcess(n_restarts=2).fit(Xtr[:1500][:, idx], ytr[:1500])
print("GP length-scales (short = relevant, 148 = switched off):",
      {c: round(float(l), 1) for c, l in sorted(zip(cols, gp.ls), key=lambda t: t[1])})
mu_gp, _ = gp.predict(X.values[half : half + 1500][:, idx])
lin = ARDRegression().fit(Xtr[:, idx], ytr)
yo = y.values[half : half + 1500]
print(f"out-of-sample corr with returns: GP {np.corrcoef(mu_gp, yo)[0, 1]:.3f} vs linear ARD "
      f"{np.corrcoef(lin.predict(X.values[half : half + 1500][:, idx]), yo)[0, 1]:.3f}\n")

# 3. online Bayesian regression, walk-forward, positions scaled by predictive confidence
feats = [X.columns.get_loc(c) for c in ["mom", "flow"]]
Z = np.c_[X.values[:, feats], X.values[:, X.columns.get_loc("vol_sq")] ** 2 - 1, np.ones(len(X))]
for lam, name in [(1.0, "no forgetting"), (0.999, "forgetting 0.999")]:
    blr = BayesianLinearRegression(Z.shape[1], prior_scale=0.1, forgetting=lam).update(Z[:300], y.values[:300])
    pnl = []
    for t in range(300, len(Z)):
        mu, sd = blr.predict(Z[t])
        pnl.append(float(mu[0] / sd[0] ** 2) * y.values[t])  # Kelly-style size: mean / variance
        blr.update(Z[t], y.values[t])
    pnl = np.array(pnl)
    first, second = pnl[: half - 300], pnl[half - 300 :]
    print(f"BLR {name:<17} Sharpe 1st half {sharpe(first) * np.sqrt(252):5.2f} | 2nd half {sharpe(second) * np.sqrt(252):5.2f}"
          f" | PSR {probabilistic_sharpe(pnl):.3f}")

# 4. multiple testing: 200 strategies on noise features alone
rng = np.random.default_rng(1)
noise_cols = [i for i, c in enumerate(X.columns) if c.startswith("noise")]
trials = []
for _ in range(200):
    w = rng.standard_normal(3)
    cols = rng.choice(noise_cols, 3, replace=False)
    trials.append(np.sign(X.values[half:, cols] @ w) * y.values[half:])
trial_sr = np.array([sharpe(t) for t in trials])
best = trials[int(np.argmax(trial_sr))]
print(f"\nbest of 200 noise strategies: annualised Sharpe {trial_sr.max() * np.sqrt(252):.2f}, "
      f"PSR {probabilistic_sharpe(best):.3f}, deflated SR {deflated_sharpe(best, trial_sr):.3f}")
