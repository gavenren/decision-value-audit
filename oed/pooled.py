"""Build the difficulty-adjusted panel used by every pooled result.

`build` returns a `Pooled` object holding, for each question type, the rows,
integer codes, and a function that turns any row-level loss into
difficulty-adjusted per-forecaster means, mirroring the tournament leaderboard:

  dataset questions: item effect from a two-way fixed-effects fit;
  market questions:  item effect = the loss of the market price itself
                     (ForecastBench's "Imputed Forecaster"), i.e. every
                     forecaster is scored relative to the price it was shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import core, panel

MIN_DAYS = 50               # leaderboard: rounds must be at least this old
MIN_DATASET_ITEMS = 225     # leaderboard: per round
MIN_MARKET_ITEMS = 50       # leaderboard: per round
MARKET_PRICE = "ForecastBench / Imputed Forecaster"
ALWAYS_HALF = "ForecastBench / Always 0.5"


@dataclass
class Block:
    """One question type (dataset or market)."""
    qtype: str
    rows: pd.DataFrame
    f: np.ndarray                 # forecaster codes (global)
    i: np.ndarray                 # item codes (within block)
    c: np.ndarray                 # cluster codes of each row (global)
    n_items: int
    item_cluster: np.ndarray      # cluster code per item
    item_y: np.ndarray
    item_base: np.ndarray         # base rate of the item's source
    est: np.ndarray               # boolean mask: rows used to fit item effects
    fe: panel.TwoWayFE | None
    price_row_of_item: np.ndarray | None   # market only: row index of the price forecaster per item

    def p(self):
        return self.rows["p"].to_numpy(dtype=float)

    def y(self):
        return self.rows["y"].to_numpy(dtype=float)


@dataclass
class Pooled:
    forecasters: np.ndarray
    clusters: np.ndarray
    blocks: dict[str, Block]
    info: dict = field(default_factory=dict)

    @property
    def F(self) -> int:
        return len(self.forecasters)

    def code(self, name: str) -> int:
        return int(np.flatnonzero(self.forecasters == name)[0])

    # -- item effects and adjusted means ------------------------------------
    def gamma(self, qtype: str, loss: np.ndarray) -> np.ndarray:
        b = self.blocks[qtype]
        if qtype == "market":
            return loss[b.price_row_of_item]
        return b.fe.fit(loss[b.est])

    def adjusted(self, qtype: str, loss: np.ndarray, gamma: np.ndarray | None = None):
        """(adjusted mean loss per forecaster, n rows per forecaster)."""
        b = self.blocks[qtype]
        g = self.gamma(qtype, loss) if gamma is None else gamma
        n = np.bincount(b.f, minlength=self.F)
        return panel.adjusted_mean(loss, g, b.f, b.i, self.F, g.mean()), n

    def reference(self, qtype: str, loss_fn) -> float:
        """Mean loss, over all items of the block, of always forecasting the source base rate."""
        b = self.blocks[qtype]
        return float(loss_fn(b.item_base, b.item_y).mean())


def build(parquet_path: str, as_of: str) -> Pooled:
    d = panel.load(parquet_path)
    info = {"rows_after_filters": len(d), "files_dropped_for_imputation": d.attrs["files_dropped_for_imputation"]}
    # leaderboard round filters
    age = (pd.Timestamp(as_of) - pd.to_datetime(d["due_date"].astype(str))).dt.days
    d = d[age >= MIN_DAYS]
    per_round = d.drop_duplicates(["item"]).groupby(["due_date", "qtype"], observed=True).size().unstack(fill_value=0)
    ok = per_round[(per_round.get("dataset", 0) >= MIN_DATASET_ITEMS) & (per_round.get("market", 0) >= MIN_MARKET_ITEMS)].index
    d = d[d["due_date"].isin(ok)].reset_index(drop=True)
    info["rounds"] = sorted(str(x) for x in ok)

    fcat = pd.Categorical(d["forecaster"].astype(str))
    ccat = pd.Categorical(d["cluster"])
    d["f"] = fcat.codes
    d["c"] = ccat.codes
    # source base rates, pooled over every scored item of every round
    items_all = d.drop_duplicates("item")
    base = items_all.groupby("source", observed=True)["y"].mean()
    info["base_rates"] = base.round(4).to_dict()

    blocks = {}
    for qt in ("dataset", "market"):
        rows = d[d["qtype"] == qt].reset_index(drop=True)
        icat = pd.Categorical(rows["item"])
        i = icat.codes.astype(np.int64)
        first = rows.drop_duplicates("item").set_index("item").loc[icat.categories]
        est = (~rows["is_dummy"] & ~rows["is_ensemble"]).to_numpy()
        fe = None
        price_row = None
        if qt == "dataset":
            em = pd.Categorical(rows.loc[est, "f"])
            fe = panel.TwoWayFE(em.codes.astype(np.int64), i[est], len(em.categories), len(icat.categories))
            info["fe_components_dataset"] = fe.n_components
        else:
            pr = np.flatnonzero((rows["forecaster"].astype(str) == MARKET_PRICE).to_numpy())
            price_row = np.full(len(icat.categories), -1)
            price_row[i[pr]] = pr
            assert (price_row >= 0).all(), "market price missing for some market items"
        blocks[qt] = Block(
            qtype=qt, rows=rows, f=rows["f"].to_numpy(np.int64), i=i, c=rows["c"].to_numpy(np.int64),
            n_items=len(icat.categories), item_cluster=first["c"].to_numpy(np.int64),
            item_y=first["y"].to_numpy(float), item_base=first["source"].astype(str).map(base).to_numpy(float),
            est=est, fe=fe, price_row_of_item=price_row,
        )
        info[f"n_rows_{qt}"] = len(rows)
        info[f"n_items_{qt}"] = len(icat.categories)
        info[f"n_questions_{qt}"] = int(first["c"].nunique())
    info["n_forecasters"] = len(fcat.categories)
    info["n_questions"] = len(ccat.categories)
    return Pooled(np.asarray(fcat.categories), np.asarray(ccat.categories), blocks, info)


# loss functions with a common signature (p, y) -> per-row loss ----------------

def brier_loss(p, y):
    return (p - y) ** 2


def theta_loss(theta: float):
    return lambda p, y: panel.regret_at(p, y, theta)


def profile_loss(a: float, b: float):
    return lambda p, y: panel.profile_regret(p, y, a, b)


def value_table(P: Pooled, loss_fn) -> pd.DataFrame:
    """Difficulty-adjusted loss and value captured, per forecaster and question type.

    value = 1 - adjusted_loss / reference_loss, with the reference the
    source-base-rate policy on the same (full) item set.  'overall' averages
    the dataset and market figures with equal weight, as the leaderboard does.
    """
    out = {}
    for qt, b in P.blocks.items():
        loss = loss_fn(b.p(), b.y())
        adj, n = P.adjusted(qt, loss)
        ref = P.reference(qt, loss_fn)
        out[f"loss_{qt}"] = adj
        out[f"ref_{qt}"] = np.full(P.F, ref)
        out[f"n_{qt}"] = n
    t = pd.DataFrame(out, index=P.forecasters)
    t["loss_overall"] = (t["loss_dataset"] + t["loss_market"]) / 2
    t["ref_overall"] = (t["ref_dataset"] + t["ref_market"]) / 2
    for k in ("dataset", "market", "overall"):
        t[f"V_{k}"] = 1 - t[f"loss_{k}"] / t[f"ref_{k}"]
    return t
