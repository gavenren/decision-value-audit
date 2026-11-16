import numpy as np
import pytest

from oed import core, panel


def test_profile_regret_matches_numeric_integral():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=300)
    y = rng.integers(0, 2, size=300).astype(float)
    # The integrand jumps at theta = p, so a midpoint rule converges only at
    # rate 1/K; the tolerance below is that discretisation error, not slack in
    # the closed form (the error shrinks tenfold for every tenfold finer grid).
    th = (np.arange(200000) + 0.5) / 200000
    for a, b in core.PROFILES.values():
        w = core.weight_profile(th, a, b)
        numeric = core.regret(p, y, th) @ w
        np.testing.assert_allclose(panel.profile_regret(p, y, a, b), numeric, atol=3e-5)


def test_uniform_profile_is_half_brier():
    p = np.linspace(0, 1, 11)
    for yy in (0.0, 1.0):
        y = np.full_like(p, yy)
        np.testing.assert_allclose(panel.profile_regret(p, y, 1, 1), 0.5 * (p - y) ** 2, atol=1e-12)


def test_regret_at_matches_core():
    rng = np.random.default_rng(1)
    p = np.round(rng.uniform(size=500), 1)
    y = rng.integers(0, 2, size=500).astype(float)
    for th in (0.1, 0.5, 0.93):
        np.testing.assert_allclose(panel.regret_at(p, y, th), core.regret(p, y, th))


def test_two_way_fe_recovers_planted_effects_on_unbalanced_panel():
    rng = np.random.default_rng(2)
    M, I = 30, 400
    alpha = rng.normal(size=M)
    gamma = rng.normal(size=I)
    # each model sees a random 30% of items; model 0 sees everything (keeps it connected)
    mask = rng.uniform(size=(M, I)) < 0.3
    mask[0] = True
    m, i = np.nonzero(mask)
    loss = alpha[m] + gamma[i]                      # no noise: recovery must be exact
    fe = panel.TwoWayFE(m, i, M, I)
    assert fe.n_components == 1
    g = fe.fit(loss)
    np.testing.assert_allclose(g - g.mean(), gamma - gamma.mean(), atol=1e-8)
    adj = panel.adjusted_mean(loss, g, m, i, M, g.mean())
    # adjusted mean = alpha + mean(gamma): what each model would score on all items
    np.testing.assert_allclose(adj, alpha + gamma.mean(), atol=1e-8)
    # and the raw means are NOT that (models saw different items), so the test has teeth
    raw = np.bincount(m, weights=loss, minlength=M) / np.bincount(m, minlength=M)
    assert np.abs(raw - (alpha + gamma.mean())).max() > 0.05


def test_two_way_fe_matches_dense_least_squares_with_noise():
    rng = np.random.default_rng(3)
    M, I = 8, 40
    mask = rng.uniform(size=(M, I)) < 0.5
    mask[0] = True
    m, i = np.nonzero(mask)
    loss = rng.normal(size=len(m))
    X = np.zeros((len(m), M + I))
    X[np.arange(len(m)), m] = 1
    X[np.arange(len(m)), M + i] = 1
    beta = np.linalg.lstsq(X, loss, rcond=None)[0]
    fitted = X @ beta
    fe = panel.TwoWayFE(m, i, M, I)
    g = fe.fit(loss)
    adj = panel.adjusted_mean(loss, g, m, i, M, 0.0)   # = alpha under this gamma
    np.testing.assert_allclose(adj[m] + g[i], fitted, atol=1e-8)
