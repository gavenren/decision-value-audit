"""How much decision value is lost to miscalibration, and how hedged are forecasts?

For every listed forecaster and each question type, forecasts are recalibrated
with an isotonic map learned on *other questions* (5-fold by underlying
question), and value captured is recomputed on the forecaster's own items.
No difficulty adjustment is needed because each forecaster is compared with
itself.

Usage: python scripts/run_recal.py out/forecasts.parquet out/pooled_results.pkl out/recal_results.pkl
"""

import pickle
import sys
import time

import numpy as np
import pandas as pd

from oed import core, pooled

THETAS = core.default_thetas(99)


def main(parquet, pooled_pkl, out):
    t0 = time.time()
    R = pickle.load(open(pooled_pkl, "rb"))
    P = pooled.build(parquet, as_of=R["info"]["as_of"])
    assert list(P.forecasters) == list(R["forecasters"])
    listed = R["listed"]
    rows = []
    curves = {}
    for qt, b in P.blocks.items():
        p_all, y_all = b.p(), b.y()
        base_of_row = b.item_base[b.i]
        order = np.argsort(b.f, kind="stable")
        bounds = np.searchsorted(b.f[order], np.arange(P.F + 1))
        for f in np.flatnonzero(listed):
            idx = order[bounds[f]: bounds[f + 1]]
            if len(idx) < 200:
                continue
            p, y, cl, base = p_all[idx], y_all[idx], b.c[idx], base_of_row[idx]
            pc = core.crossfit_recalibrate(p, y, cl, n_folds=5, seed=0)
            raw = core.mean_regret_curve(p, y, THETAS)
            cal = core.mean_regret_curve(pc, y, THETAS)
            # reference: act on the source base rate (pooled over all rounds), on this forecaster's items
            a = (base[:, None] > THETAS[None, :]).astype(float)
            ref = (THETAS[None, :] * (1 - y[:, None]) * a + (1 - THETAS[None, :]) * y[:, None] * (1 - a)).mean(0)
            curves[(qt, f)] = np.vstack([raw, cal, ref]).astype(np.float32)
            rec = dict(qtype=qt, f=f, forecaster=P.forecasters[f], n=len(idx),
                       brier_raw=core.brier(p, y), brier_cal=core.brier(pc, y),
                       share_le_05=float((p <= 0.05).mean()), share_ge_95=float((p >= 0.95).mean()),
                       share_le_10=float((p <= 0.10).mean()), share_ge_90=float((p >= 0.90).mean()),
                       share_mid=float(((p > 0.2) & (p < 0.8)).mean()),
                       p_min=float(p.min()), p_max=float(p.max()), base_rate=float(y.mean()))
            for name, (aa, bb) in core.PROFILES.items():
                w = core.weight_profile(THETAS, aa, bb)
                rec[f"V_raw[{name}]"] = 1 - float((w * raw).sum() / (w * ref).sum())
                rec[f"V_cal[{name}]"] = 1 - float((w * cal).sum() / (w * ref).sum())
            rows.append(rec)
        print(f"{qt}: done in {time.time() - t0:.0f}s", flush=True)
    res = dict(table=pd.DataFrame(rows), curves=curves, thetas=THETAS)
    pickle.dump(res, open(out, "wb"), protocol=4)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
