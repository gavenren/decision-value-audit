"""Pooled analysis: decision-value leaderboard, curves, selection experiment.

Usage: python scripts/run_pooled.py out/forecasts.parquet out/pooled_results.pkl
"""

import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from oed import core, pooled, resample

AS_OF = "2026-11-11"          # date of the dataset snapshot (last nightly commit)
N_BOOT = 1000
N_SPLITS = 400
THETAS = core.default_thetas(99)                       # 0.01 .. 0.99
SEL_THETAS = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98])
SUPER = "ForecastBench / Superforecaster median forecast"
PUBLIC = "ForecastBench / Public median forecast"
NAIVE = "ForecastBench / Naive Forecaster"
QT = ("dataset", "market")


def spearman_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlation of two (n, k) arrays."""
    ra = np.apply_along_axis(rankdata, 0, a)
    rb = np.apply_along_axis(rankdata, 0, b)
    ra -= ra.mean(0)
    rb -= rb.mean(0)
    return (ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))


def main(parquet: str, out: str):
    t0 = time.time()
    P = pooled.build(parquet, as_of=AS_OF)
    C = len(P.clusters)
    names = pd.Series(P.forecasters)
    is_real = ~names.str.contains(r"^ForecastBench / (?:Always|Random|Imputed|LLM Crowd)").to_numpy()
    n_d = np.bincount(P.blocks["dataset"].f, minlength=P.F)
    n_m = np.bincount(P.blocks["market"].f, minlength=P.F)
    listed = is_real & (n_d >= pooled.MIN_DATASET_ITEMS) & (n_m >= pooled.MIN_MARKET_ITEMS)
    humans = names.isin([SUPER, PUBLIC]).to_numpy()
    # candidates for the selection experiment: AI forecasters with enough questions to split in half
    cand = listed & ~humans & (names != NAIVE).to_numpy() & (n_d >= 2000) & (n_m >= 200)
    cand_wide = listed & ~humans & (names != NAIVE).to_numpy() & (n_d >= 1000) & (n_m >= 100)
    print(f"built panel in {time.time() - t0:.0f}s; listed {listed.sum()}, candidates {cand.sum()} (wide {cand_wide.sum()})")

    losses = {"brier": pooled.brier_loss}
    for name, (a, b) in core.PROFILES.items():
        losses[f"profile:{name}"] = pooled.profile_loss(a, b)
    for th in THETAS:
        losses[f"theta:{th:.2f}"] = pooled.theta_loss(float(th))

    Wb = resample.bootstrap_weights(C, N_BOOT, seed=1)
    Ws = resample.split_weights(C, N_SPLITS, seed=2)

    point = {}      # loss name -> DataFrame (forecaster x [loss_dataset, loss_market, ref_dataset, ref_market])
    boot_V = {}     # loss name -> dict(kind -> (F, N_BOOT) float32) for brier and profiles; quantiles only for thetas
    q_V = {}        # loss name -> dict(kind -> (F, 3)) 2.5/50/97.5 percentiles
    split = {}      # loss name -> dict(train/test -> kind -> (F, N_SPLITS) float32) for brier + SEL_THETAS + profiles
    keep_split = {"brier"} | {f"theta:{t:.2f}" for t in SEL_THETAS} | {k for k in losses if k.startswith("profile:")}

    for li, (lname, fn) in enumerate(losses.items()):
        sums = {}
        pt = {}
        for qt in QT:
            b = P.blocks[qt]
            loss = fn(b.p(), b.y())
            g = P.gamma(qt, loss)
            ref_item = fn(b.item_base, b.item_y)
            sums[qt] = resample.Sums(P, qt, loss, g, ref_item)
            adj, n = P.adjusted(qt, loss, g)
            pt[f"loss_{qt}"] = adj
            pt[f"ref_{qt}"] = np.full(P.F, ref_item.mean())
        point[lname] = pd.DataFrame(pt, index=P.forecasters)

        def V_from(W):
            a_d, r_d, _ = sums["dataset"].evaluate(W)
            a_m, r_m, _ = sums["market"].evaluate(W)
            with np.errstate(invalid="ignore", divide="ignore"):
                return {
                    "dataset": 1 - a_d / r_d, "market": 1 - a_m / r_m,
                    "overall": 1 - (a_d + a_m) / (r_d + r_m),
                    "loss_dataset": a_d, "loss_market": a_m, "loss_overall": (a_d + a_m) / 2,
                }

        vb = V_from(Wb)
        q_V[lname] = {k: np.nanpercentile(vb[k], [2.5, 50, 97.5], axis=1).T.astype(np.float32)
                      for k in ("dataset", "market", "overall")}
        if not lname.startswith("theta:"):
            boot_V[lname] = {k: vb[k].astype(np.float32) for k in ("dataset", "market", "overall", "loss_overall")}
        else:
            # keep draws for the human anchors so that differences can be formed later
            boot_V[lname] = {k: vb[k].astype(np.float32) for k in ("dataset", "market", "overall")}
        if lname in keep_split:
            tr = V_from(Ws)
            te = V_from(1 - Ws)
            split[lname] = {"train": {k: tr[k].astype(np.float32) for k in ("dataset", "market", "overall")},
                            "test": {k: te[k].astype(np.float32) for k in ("dataset", "market", "overall")}}
        if li % 20 == 0:
            print(f"  {li + 1}/{len(losses)} losses, {time.time() - t0:.0f}s", flush=True)

    # theta draws for all forecasters are large; keep them only where needed
    keep_rows = np.flatnonzero(listed)
    for lname in list(boot_V):
        if lname.startswith("theta:"):
            boot_V[lname] = {k: v[keep_rows] for k, v in boot_V[lname].items()}

    res = dict(
        info=P.info | {"as_of": AS_OF, "n_boot": N_BOOT, "n_splits": N_SPLITS},
        forecasters=P.forecasters, listed=listed, cand=cand, cand_wide=cand_wide, humans=humans,
        n_dataset=n_d, n_market=n_m, thetas=THETAS, sel_thetas=SEL_THETAS,
        point=point, q_V=q_V, boot_V=boot_V, boot_rows=keep_rows, split=split,
    )
    with open(out, "wb") as fh:
        pickle.dump(res, fh, protocol=4)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
