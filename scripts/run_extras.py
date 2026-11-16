"""Three supporting analyses.

1. Humans vs AI on identical items (round 2024-07-21), no difficulty model.
2. Does anyone beat the market price on market questions, in decision terms?
3. How much wider are confidence intervals when whole questions are resampled
   rather than (question, horizon) items?

Usage: python scripts/run_extras.py out/forecasts.parquet out/pooled_results.pkl out/extras.pkl
"""

import pickle
import sys

import numpy as np
import pandas as pd

from oed import core, panel, pooled, resample

THETAS = core.default_thetas(99)
SUPER = "ForecastBench / Superforecaster median forecast"
PUBLIC = "ForecastBench / Public median forecast"
ROUND1 = "2024-07-21"


def round1(parquet, base_rates, n_boot=2000, seed=3):
    d = panel.load(parquet)
    d = d[d["due_date"].astype(str) == ROUND1]
    sup_items = set(d.loc[d["forecaster"].astype(str) == SUPER, "item"])
    d = d[d["item"].isin(sup_items)]
    wide = d.pivot(index="item", columns="forecaster", values="p")
    wide = wide.loc[:, wide.notna().all()]                      # forecasters with every item
    meta = d.drop_duplicates("item").set_index("item").loc[wide.index]
    y = meta["y"].to_numpy(float)
    base = meta["source"].astype(str).map(base_rates).to_numpy(float)
    qtype = meta["qtype"].astype(str).to_numpy()
    names = np.array([str(c) for c in wide.columns])
    is_ai = ~pd.Series(names).str.contains(r"^ForecastBench / (?:Always|Random|Imputed|Naive|LLM Crowd|Superforecaster|Public)").to_numpy()
    Pm = wide.to_numpy(float)
    clusters, inv = np.unique(meta["cluster"].to_numpy(), return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(clusters))]

    def stats(idx):
        yy, bb = y[idx], base[idx]
        ref = core.mean_regret_curve(bb, yy, THETAS)
        out = {}
        briers = ((Pm[idx] - yy[:, None]) ** 2).mean(0)
        ai_idx = np.flatnonzero(is_ai)
        best_ai = ai_idx[np.argmin(briers[ai_idx])]              # re-selected inside each resample
        for label, col in (("super", int(np.flatnonzero(names == SUPER)[0])),
                           ("public", int(np.flatnonzero(names == PUBLIC)[0])),
                           ("best_ai", int(best_ai))):
            r = core.mean_regret_curve(Pm[idx, col], yy, THETAS)
            out[label] = {"curve": 1 - r / ref, "brier": briers[col]}
            for pname, (a, b) in core.PROFILES.items():
                w = core.weight_profile(THETAS, a, b)
                out[label][pname] = 1 - (w * r).sum() / (w * ref).sum()
        # the median AI forecaster's value at each theta
        R = np.vstack([core.mean_regret_curve(Pm[idx, c], yy, THETAS) for c in ai_idx])
        out["median_ai"] = {"curve": np.median(1 - R / ref, axis=0), "brier": float(np.median(briers[ai_idx]))}
        for pname, (a, b) in core.PROFILES.items():
            w = core.weight_profile(THETAS, a, b)
            out["median_ai"][pname] = float(np.median(1 - (R * w).sum(1) / (w * ref).sum()))
        out["best_ai_name"] = names[best_ai]
        return out

    point = stats(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        draws.append(stats(np.concatenate([rows_of[k] for k in pick])))
    return dict(point=point, draws=draws, n_items=len(y), n_questions=len(clusters),
                n_market=int((qtype == "market").sum()), n_ai=int(is_ai.sum()), thetas=THETAS)


def design_effect(P, listed, n_boot=500):
    """SE of difficulty-adjusted dataset Brier: resample items vs resample questions."""
    b = P.blocks["dataset"]
    loss = pooled.brier_loss(b.p(), b.y())
    g = P.gamma("dataset", loss)
    ref_item = pooled.brier_loss(b.item_base, b.item_y)
    by_question = resample.Sums(P, "dataset", loss, g, ref_item)
    W = resample.bootstrap_weights(len(P.clusters), n_boot, seed=11)
    se_q = np.nanstd(by_question.evaluate(W)[0], axis=1)
    # same machinery with every item as its own "question"
    import copy
    P2 = copy.copy(P)
    b2 = copy.copy(b)
    b2.c = b.i.copy()
    b2.item_cluster = np.arange(b.n_items)
    P2.blocks = {"dataset": b2}
    P2.clusters = np.arange(b.n_items)
    by_item = resample.Sums(P2, "dataset", loss, g, ref_item)
    W2 = resample.bootstrap_weights(b.n_items, n_boot, seed=12)
    se_i = np.nanstd(by_item.evaluate(W2)[0], axis=1)
    ratio = se_q[listed] / se_i[listed]
    return dict(se_question=se_q, se_item=se_i, ratio_listed=ratio,
                items_per_question=b.n_items / len(np.unique(b.item_cluster)))


def main(parquet, pooled_pkl, out):
    R = pickle.load(open(pooled_pkl, "rb"))
    P = pooled.build(parquet, as_of=R["info"]["as_of"])
    res = dict(round1=round1(parquet, R["info"]["base_rates"]),
               design=design_effect(P, R["listed"]))
    pickle.dump(res, open(out, "wb"), protocol=4)
    r1 = res["round1"]
    print("round 1:", r1["n_items"], "items,", r1["n_questions"], "questions,", r1["n_ai"], "AI forecasters; best AI:", r1["point"]["best_ai_name"])
    d = res["design"]
    print("design effect: median SE ratio (question / item) = %.2f, IQR [%.2f, %.2f]; items per question %.1f"
          % (np.nanmedian(d["ratio_listed"]), *np.nanpercentile(d["ratio_listed"], [25, 75]), d["items_per_question"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
