"""Decision-value scoring of probability forecasts for binary events.

Everything here follows from one decision model, the classical cost-loss
problem (Murphy 1977; Richardson 2000):

    A decision-maker can pay a cost C now to avoid a loss L that is incurred
    if the event happens (y = 1).  Write theta = C / L, with 0 < theta < 1.
    Given a probability p for the event, expected expense is minimised by
    acting exactly when p > theta.

Expenses below are in units of L per decision.

    expense(p, y; theta) = theta          if p > theta   (acted)
                         = y              otherwise      (did not act)
    oracle expense       = theta * y      (acts exactly when the event happens)
    regret(p, y; theta)  = expense - oracle expense
                         = theta * (1 - y) * [p > theta]      (false alarm)
                         + (1 - theta) * y * [p <= theta]     (miss)

Two facts make this useful for auditing forecast leaderboards:

1.  2 * integral_0^1 regret(p, y; theta) d theta = (p - y)^2.
    The Brier score is twice the average regret of a population of
    decision-makers whose cost-loss ratios are uniform on (0, 1)
    (Schervish 1989; Ehm, Gneiting, Jordan & Krueger 2016).  Ranking by Brier
    score is ranking for that particular population of users.

2.  At a single theta the regret depends on p only through which side of
    theta it falls.  Two forecasters can be ordered one way by Brier score and
    the other way at the thresholds a particular user cares about.

`value_captured` rescales regret so that 0 is "no better than acting on the
base rate" and 1 is "as good as perfect foresight".  This is the relative
economic value of Richardson (2000).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "default_thetas",
    "regret",
    "mean_regret_curve",
    "reference_regret_curve",
    "value_captured",
    "brier",
    "brier_from_curve",
    "pav",
    "pav_predict",
    "crossfit_recalibrate",
    "weight_profile",
    "PROFILES",
]


def default_thetas(n: int = 99) -> np.ndarray:
    """Evenly spaced cost-loss ratios strictly inside (0, 1): 0.01 .. 0.99."""
    return np.arange(1, n + 1) / (n + 1)


def _act(p: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Probability of acting.  1 if p > theta, 0 if p < theta, 1/2 at ties.

    At p == theta the decision-maker is indifferent in expectation.  Forecasts
    cluster on round numbers, so a one-sided tie rule would systematically
    favour hedged forecasts on one side of every round threshold.  Splitting
    ties is the convention that keeps the Brier identity exact on any grid.
    """
    return (p > theta).astype(float) + 0.5 * (p == theta)


def regret(p, y, theta) -> np.ndarray:
    """Regret versus perfect foresight, per forecast, in units of the loss L.

    p, y: arrays of shape (n,).  theta: scalar or array of shape (k,).
    Returns shape (n,) for scalar theta, else (n, k).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 0:
        a = _act(p, theta)
        return theta * (1 - y) * a + (1 - theta) * y * (1 - a)
    a = _act(p[:, None], theta[None, :])
    yy = y[:, None]
    return theta[None, :] * (1 - yy) * a + (1 - theta[None, :]) * yy * (1 - a)


def mean_regret_curve(p, y, thetas, weights=None) -> np.ndarray:
    """Murphy curve: mean regret at each theta.  Shape (k,).

    Computed by sorting once instead of materialising an n-by-k matrix.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ps, ys, ws = p[order], y[order], w[order]
    # cumulative weighted event mass and non-event mass, sorted by forecast
    c1 = np.concatenate([[0.0], np.cumsum(ws * ys)])
    c0 = np.concatenate([[0.0], np.cumsum(ws * (1 - ys))])
    lo = np.searchsorted(ps, thetas, side="left")   # forecasts strictly below theta
    hi = np.searchsorted(ps, thetas, side="right")  # forecasts at or below theta
    miss = c1[lo] + 0.5 * (c1[hi] - c1[lo])                       # y=1 and not acted
    false_alarm = (c0[-1] - c0[hi]) + 0.5 * (c0[hi] - c0[lo])     # y=0 and acted
    return (thetas * false_alarm + (1 - thetas) * miss) / ws.sum()


def reference_regret_curve(y, thetas, groups=None, weights=None) -> np.ndarray:
    """Regret of the best policy that ignores the forecaster.

    Within each group (for example, each question source) the reference
    decision-maker knows the base rate and takes the same action on every
    question: act if base rate > theta.  With groups=None there is a single
    pooled base rate.  The base rate is computed in-sample, which makes the
    reference slightly too strong and value_captured slightly conservative.
    """
    y = np.asarray(y, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if groups is None:
        groups = np.zeros(len(y), dtype=int)
    groups = np.asarray(groups)
    total = np.zeros_like(thetas)
    for g in np.unique(groups):
        m = groups == g
        wg = w[m].sum()
        base = (w[m] * y[m]).sum() / wg
        total += wg * np.minimum(thetas * (1 - base), (1 - thetas) * base)
    return total / w.sum()


def value_captured(p, y, thetas, groups=None, weights=None) -> np.ndarray:
    """Share of the value of perfect foresight that the forecaster delivers.

    V(theta) = 1 - regret_forecaster(theta) / regret_reference(theta)

    1 means decisions as good as an oracle's; 0 means no better than acting on
    the base rate; negative means the user would have done better ignoring
    the forecaster.  Undefined (nan) where the reference regret is zero.
    """
    r = mean_regret_curve(p, y, thetas, weights)
    ref = reference_regret_curve(y, thetas, groups, weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ref > 0, 1 - r / ref, np.nan)


def brier(p, y, weights=None) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.average((p - y) ** 2, weights=weights))


def brier_from_curve(curve, thetas) -> float:
    """2 * integral of the Murphy curve, by the midpoint rule on an even grid.

    Exact when every forecast lies on a grid boundary; otherwise off by at most
    one grid cell's worth of regret.  Used as a check, not as a result.
    """
    thetas = np.asarray(thetas, dtype=float)
    step = thetas[1] - thetas[0]
    return float(2 * np.sum(curve) * step)


# ---------------------------------------------------------------------------
# Recalibration (isotonic regression by pool-adjacent-violators)
# ---------------------------------------------------------------------------

def pav(x, y, weights=None):
    """Isotonic regression of y on x.  Returns (knots_x, fitted_values).

    Forecasts with identical x are pooled first, then adjacent violators are
    merged.  The fitted value at a knot is the weighted event frequency of its
    pool, the "calibrated" probability in the CORP sense (Dimitriadis,
    Gneiting & Jordan 2021).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sy = np.bincount(inv, weights=w * y)
    # stack of blocks: (sum_wy, sum_w, first_index, last_index)
    blocks: list[list[float]] = []
    for i in range(len(ux)):
        blocks.append([sy[i], sw[i], i, i])
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] >= blocks[-1][0] * blocks[-2][1]:
            b = blocks.pop()
            a = blocks[-1]
            a[0] += b[0]
            a[1] += b[1]
            a[3] = b[3]
    fit = np.empty(len(ux))
    for s_y, s_w, i0, i1 in blocks:
        fit[int(i0): int(i1) + 1] = s_y / s_w
    return ux, fit


def pav_predict(knots, fit, x):
    """Evaluate an isotonic fit at new x by linear interpolation, flat outside."""
    return np.interp(np.asarray(x, dtype=float), knots, fit)


def crossfit_recalibrate(p, y, clusters, n_folds: int = 5, seed: int = 0):
    """Out-of-fold isotonic recalibration.

    The recalibration map for each forecast is learned on other clusters
    (questions) only, so the recalibrated forecasts are ones a user could
    actually have produced from the forecaster's past record.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    fold_of_cluster = rng.permutation(len(uniq)) % n_folds
    fold = fold_of_cluster[inv]
    out = np.empty_like(p)
    for f in range(n_folds):
        test = fold == f
        if not test.any():
            continue
        knots, fit = pav(p[~test], y[~test])
        out[test] = pav_predict(knots, fit, p[test])
    return out


# ---------------------------------------------------------------------------
# User profiles: distributions over the cost-loss ratio
# ---------------------------------------------------------------------------

def _beta_pdf(x, a, b):
    from math import lgamma
    logc = lgamma(a + b) - lgamma(a) - lgamma(b)
    return np.exp(logc + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x))


#: name -> (a, b) of a Beta density over theta.  Beta(1, 1) reproduces the
#: Brier score.  The others describe users the Brier score under-weights.
PROFILES = {
    "uniform (Brier)": (1.0, 1.0),
    "precaution (rare, costly events)": (1.0, 9.0),
    "long shot (act only if near-certain)": (9.0, 1.0),
    "toss-up (even stakes)": (20.0, 20.0),
}


def weight_profile(thetas, a: float, b: float) -> np.ndarray:
    """Normalised weights over a theta grid for a Beta(a, b) user population."""
    w = _beta_pdf(np.asarray(thetas, dtype=float), a, b)
    return w / w.sum()
