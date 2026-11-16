"""The pooled ForecastBench panel: filtering, row scores, difficulty adjustment.

Mirrors the public leaderboard code (forecastingresearch/forecastbench,
src/leaderboard/main.py) wherever a choice had to be made:

  * combination questions are dropped;
  * dataset questions are scored at every resolved horizon, market questions
    only once resolved;
  * a forecast file is dropped if more than 5% of its forecasts were imputed
    (overall, or within dataset or market questions); otherwise imputed values
    are scored as submitted, as the leaderboard does;
  * difficulty is removed with a two-way fixed-effects model, fitted separately
    for dataset and market questions:  loss[m, i] = alpha[m] + gamma[i] + e;
  * "overall" is the unweighted mean of the dataset and market figures.

Where this file departs from the leaderboard, on purpose:

  * the loss is not only the Brier score but the regret of a cost-loss
    decision-maker (see core.py), at single thresholds and averaged over Beta
    distributed user populations;
  * question effects are fitted on every non-dummy forecaster, not only on
    ForecastBench's own recent models (the leaderboard's restriction needs
    model release dates, which are not in the public forecast files);
  * the bootstrap resamples underlying questions (source, id) across all
    horizons and rounds, rather than (question, horizon) pairs within a round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import betainc

IMPUTED_CUTOFF = 0.05
DUMMIES = {"Always 0", "Always 1", "Always 0.5", "Random Uniform", "Imputed Forecaster"}
FB = "ForecastBench"


def load(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["forecaster"] = (d["team"].astype(str) + " / " + d["model"].astype(str)).astype("category")
    # leaderboard's imputation rule, on all single questions the file will eventually be scored on
    g = d.groupby("file", observed=True)["imputed"].mean()
    gq = d.groupby(["file", "qtype"], observed=True)["imputed"].mean().unstack()
    own_dummy = d.drop_duplicates("file").set_index("file")
    is_dummy_file = (own_dummy["team"].astype(str) == FB) & (own_dummy["model_org"].astype(str) == FB)
    bad = (g > IMPUTED_CUTOFF) | (gq.max(axis=1) > IMPUTED_CUTOFF)
    bad = bad & ~is_dummy_file.reindex(bad.index).fillna(False)
    d = d[~d["file"].isin(bad[bad].index)]
    d = d[d["resolved"] & d["resolved_to"].isin([0.0, 1.0]) & d["p"].between(0, 1)].copy()
    d["y"] = d["resolved_to"].astype(float)
    d["item"] = (d["due_date"].astype(str) + "|" + d["source"].astype(str) + "|" + d["id"].astype(str)
                 + "|" + np.where(d["qtype"] == "market", "", d["resolution_date"].astype(str)))
    d["cluster"] = d["source"].astype(str) + "|" + d["id"].astype(str)
    d["horizon"] = (pd.to_datetime(d["resolution_date"].astype(str)) - pd.to_datetime(d["due_date"].astype(str))).dt.days
    # a handful of files list a market question twice; keep the first
    d = d.drop_duplicates(["file", "item"])
    # one forecaster can only have one forecast per item
    d = d.drop_duplicates(["forecaster", "item"])
    d["is_dummy"] = (d["team"].astype(str) == FB) & d["model"].astype(str).isin(DUMMIES)
    d["is_ensemble"] = (d["team"].astype(str) == FB) & d["model"].astype(str).str.startswith("LLM Crowd")
    d.attrs["files_dropped_for_imputation"] = int(bad.sum())
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Row-level losses
# ---------------------------------------------------------------------------

def regret_at(p, y, theta: float) -> np.ndarray:
    a = (p > theta) + 0.5 * (p == theta)
    return theta * (1 - y) * a + (1 - theta) * y * (1 - a)


def profile_regret(p, y, a: float, b: float) -> np.ndarray:
    """Mean regret of users with theta ~ Beta(a, b), in closed form.

    integral of theta * f(theta) over (0, p)      = a/(a+b) * I_p(a+1, b)       [y = 0]
    integral of (1-theta) * f(theta) over (p, 1)  = b/(a+b) * (1 - I_p(a, b+1)) [y = 1]

    With a = b = 1 this is half the Brier score.  For general (a, b) it is the
    Beta family of proper scoring rules of Buja, Stuetzle & Shen (2005).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    false_alarm = a / (a + b) * betainc(a + 1, b, p)
    miss = b / (a + b) * (1 - betainc(a, b + 1, p))
    return (1 - y) * false_alarm + y * miss


# ---------------------------------------------------------------------------
# Two-way fixed effects
# ---------------------------------------------------------------------------

class TwoWayFE:
    """loss[m, i] = alpha[m] + gamma[i] + e, solved exactly for many losses.

    The model-by-item incidence matrix is factorised once (Schur complement on
    the model block), after which each new loss vector costs two bincounts and
    one small solve.
    """

    def __init__(self, model_codes: np.ndarray, item_codes: np.ndarray, n_models: int, n_items: int):
        self.m, self.i = model_codes, item_codes
        self.M, self.I = n_models, n_items
        self.dm = np.bincount(model_codes, minlength=n_models).astype(float)
        self.di = np.bincount(item_codes, minlength=n_items).astype(float)
        N = sp.csr_matrix((np.ones(len(model_codes)), (model_codes, item_codes)), shape=(n_models, n_items))
        self.N = N
        S = np.diag(self.dm) - (N @ sp.diags(1 / self.di) @ N.T).toarray()
        # S is singular along the all-ones direction (and along one direction per
        # extra disconnected component).  The pseudo-inverse gives the
        # minimum-norm solution; everything reported downstream is invariant to
        # the choice within a connected component.
        self.S_pinv = np.linalg.pinv(S, hermitian=True)
        self.n_components = int(sp.csgraph.connected_components(sp.bmat([[None, N], [N.T, None]]), directed=False)[0])

    def fit(self, loss: np.ndarray) -> np.ndarray:
        """Return gamma (item effects) for one loss vector aligned with the codes."""
        bm = np.bincount(self.m, weights=loss, minlength=self.M)
        bi = np.bincount(self.i, weights=loss, minlength=self.I)
        alpha = self.S_pinv @ (bm - self.N @ (bi / self.di))
        return (bi - self.N.T @ alpha) / self.di


def adjusted_mean(loss, gamma, model_codes, item_codes, n_models, all_items_mean_gamma):
    """Difficulty-adjusted mean loss per model, on the scale of the full item set.

    mean over the model's items of (loss - gamma[item]), plus the mean gamma of
    all items.  Under the additive model this is the loss the model would have
    had if it had forecast every item.  It does not depend on how gamma is
    normalised.
    """
    s = np.bincount(model_codes, weights=loss - gamma[item_codes], minlength=n_models)
    n = np.bincount(model_codes, minlength=n_models)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n + all_items_mean_gamma
