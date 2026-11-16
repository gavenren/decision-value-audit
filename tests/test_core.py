import numpy as np
import pytest

from oed import core


def brute_curve(p, y, thetas):
    return core.regret(p, y, thetas).mean(axis=0)


def test_regret_by_hand():
    # theta = 0.2: acting costs 0.2.  p = 0.5 -> act.
    assert core.regret([0.5], [0], 0.2)[0] == pytest.approx(0.2)   # false alarm: paid 0.2 for nothing
    assert core.regret([0.5], [1], 0.2)[0] == pytest.approx(0.0)   # acted, event happened: same as oracle
    # p = 0.1 -> do not act.
    assert core.regret([0.1], [0], 0.2)[0] == pytest.approx(0.0)
    assert core.regret([0.1], [1], 0.2)[0] == pytest.approx(0.8)   # miss: lost 1 where oracle paid 0.2
    # tie: half of each
    assert core.regret([0.2], [1], 0.2)[0] == pytest.approx(0.4)
    assert core.regret([0.2], [0], 0.2)[0] == pytest.approx(0.1)


def test_fast_curve_matches_brute_force():
    rng = np.random.default_rng(1)
    p = np.round(rng.uniform(size=2000), 2)      # many ties with the grid
    y = (rng.uniform(size=2000) < p).astype(float)
    th = core.default_thetas()
    np.testing.assert_allclose(core.mean_regret_curve(p, y, th), brute_curve(p, y, th), atol=1e-12)
    w = rng.uniform(0.5, 2, size=2000)
    brute_w = (core.regret(p, y, th) * w[:, None]).sum(0) / w.sum()
    np.testing.assert_allclose(core.mean_regret_curve(p, y, th, w), brute_w, atol=1e-12)


def test_brier_identity_exact_on_grid_forecasts():
    # Forecasts on multiples of 0.01, thetas on the 0.005 offsets: the midpoint
    # rule integrates the piecewise-linear regret exactly.
    rng = np.random.default_rng(2)
    p = rng.integers(0, 101, size=5000) / 100
    y = rng.integers(0, 2, size=5000).astype(float)
    th = (np.arange(100) + 0.5) / 100
    curve = core.mean_regret_curve(p, y, th)
    assert core.brier_from_curve(curve, th) == pytest.approx(core.brier(p, y), abs=1e-12)


def test_brier_identity_close_for_continuous_forecasts():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(float)
    th = (np.arange(1000) + 0.5) / 1000
    curve = core.mean_regret_curve(p, y, th)
    assert core.brier_from_curve(curve, th) == pytest.approx(core.brier(p, y), abs=1e-5)


def test_value_captured_anchors():
    rng = np.random.default_rng(4)
    y = (rng.uniform(size=4000) < 0.3).astype(float)
    th = core.default_thetas()
    # oracle captures everything
    np.testing.assert_allclose(core.value_captured(y, y, th), 1.0)
    # the base rate itself captures nothing
    base = np.full_like(y, y.mean())
    v = core.value_captured(base, y, th)
    assert np.nanmax(np.abs(v)) < 1e-9
    # an anti-oracle is worse than ignoring the forecaster everywhere
    assert np.all(core.value_captured(1 - y, y, th) < 0)


def test_hedged_forecaster_has_zero_value_in_the_tails():
    # A forecaster with real skill who never goes below 10% or above 90%
    # tells a theta = 0.05 user to act every time: exactly the always-act policy.
    rng = np.random.default_rng(5)
    q = rng.beta(0.3, 0.9, size=20000)          # true probabilities, mostly small
    y = (rng.uniform(size=20000) < q).astype(float)
    hedged = np.clip(q, 0.10, 0.90)
    th = np.array([0.05, 0.5])
    v_true = core.value_captured(q, y, th)
    v_hedged = core.value_captured(hedged, y, th)
    assert v_true[0] > 0.2
    assert v_hedged[0] <= 1e-9                  # no value where it never crosses the threshold
    assert v_hedged[1] == pytest.approx(v_true[1], abs=1e-12)   # untouched at theta = 0.5
    # ...while the Brier scores are nearly indistinguishable
    assert abs(core.brier(q, y) - core.brier(hedged, y)) < 0.005


def test_pav_is_monotone_and_preserves_mean():
    rng = np.random.default_rng(6)
    x = np.round(rng.uniform(size=3000), 2)
    y = (rng.uniform(size=3000) < x ** 2).astype(float)
    knots, fit = core.pav(x, y)
    assert np.all(np.diff(fit) >= -1e-12)
    pred = core.pav_predict(knots, fit, x)
    assert pred.mean() == pytest.approx(y.mean(), abs=1e-12)
    # recalibration cannot increase in-sample regret at any theta (CORP)
    th = core.default_thetas()
    assert np.all(core.mean_regret_curve(pred, y, th) <= core.mean_regret_curve(x, y, th) + 1e-12)


def test_pav_matches_reference_on_small_case():
    # violators at (0.2 -> 1, 0.3 -> 0) pool to 0.5
    knots, fit = core.pav([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    np.testing.assert_allclose(fit, [0, 0.5, 0.5, 1])


def test_crossfit_never_uses_own_cluster():
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(50), 4)
    p = rng.uniform(size=200)
    y = rng.integers(0, 2, size=200).astype(float)
    a = core.crossfit_recalibrate(p, y, clusters, seed=1)
    y2 = y.copy()
    target = clusters == 7
    y2[target] = 1 - y2[target]                 # change outcomes of one cluster only
    b = core.crossfit_recalibrate(p, y2, clusters, seed=1)
    np.testing.assert_allclose(a[target], b[target])   # its own predictions do not move


def test_profiles_normalise_and_uniform_is_brier():
    th = (np.arange(200) + 0.5) / 200
    for a, b in core.PROFILES.values():
        assert core.weight_profile(th, a, b).sum() == pytest.approx(1.0)
    rng = np.random.default_rng(8)
    p = rng.integers(0, 101, size=3000) / 100
    y = rng.integers(0, 2, size=3000).astype(float)
    w = core.weight_profile(th, 1, 1)
    assert 2 * (core.mean_regret_curve(p, y, th) * w).sum() == pytest.approx(core.brier(p, y), abs=1e-12)
