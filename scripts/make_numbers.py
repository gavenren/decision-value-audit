"""Collect every number quoted in the report into out/numbers.json.

Usage: python scripts/make_numbers.py out/ /path/to/leaderboard_tournament.csv
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from run_pooled import spearman_cols  # noqa: E402

SUPER = "ForecastBench / Superforecaster median forecast"
PUBLIC = "ForecastBench / Public median forecast"
NAIVE = "ForecastBench / Naive Forecaster"
PRICE = "ForecastBench / Imputed Forecaster"
HALF = "ForecastBench / Always 0.5"
PROFILES = {
    "uniform": "profile:uniform (Brier)",
    "precaution": "profile:precaution (rare, costly events)",
    "high_bar": "profile:long shot (act only if near-certain)",
    "even": "profile:toss-up (even stakes)",
}
RAW_PROFILE = {k: v.split(":", 1)[1] for k, v in PROFILES.items()}


def r(x, k=3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, k) for v in x]
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def main(outdir, leaderboard_csv):
    R = pickle.load(open(f"{outdir}/pooled_results.pkl", "rb"))
    Q = pickle.load(open(f"{outdir}/recal_results.pkl", "rb"))
    E = pickle.load(open(f"{outdir}/extras.pkl", "rb"))
    names = np.asarray(R["forecasters"])
    idx = {n: i for i, n in enumerate(names)}
    listed, cand, cand_wide = R["listed"], R["cand"], R["cand_wide"]
    th = R["thetas"]
    N = {}

    def V(lname, kind="overall"):
        p = R["point"][lname]
        if kind == "overall":
            return (1 - (p.loss_dataset + p.loss_market) / (p.ref_dataset + p.ref_market)).to_numpy()
        return (1 - p[f"loss_{kind}"] / p[f"ref_{kind}"]).to_numpy()

    # ---- data ------------------------------------------------------------
    info = R["info"]
    N["data"] = dict(
        as_of=info["as_of"], rounds=len(info["rounds"]), first_round=info["rounds"][0], last_round=info["rounds"][-1],
        forecasts=int(info["n_rows_dataset"] + info["n_rows_market"]),
        forecasts_dataset=int(info["n_rows_dataset"]), forecasts_market=int(info["n_rows_market"]),
        items_dataset=int(info["n_items_dataset"]), items_market=int(info["n_items_market"]),
        questions=int(info["n_questions"]), questions_dataset=int(info["n_questions_dataset"]),
        questions_market=int(info["n_questions_market"]), forecasters=int(info["n_forecasters"]),
        files_dropped_for_imputation=int(info["files_dropped_for_imputation"]),
        listed=int(listed.sum()), candidates=int(cand.sum()), candidates_wide=int(cand_wide.sum()),
        n_boot=info["n_boot"], n_splits=info["n_splits"], base_rates=info["base_rates"],
    )

    # ---- reproduction of the published leaderboard -----------------------
    B = R["point"]["brier"].copy()
    B["n_dataset"], B["n_market"] = R["n_dataset"], R["n_market"]
    sh_d = 0.25 - B.loc[HALF, "loss_dataset"]
    sh_m = 0.25 - B.loc[HALF, "loss_market"]
    B["bd"], B["bm"] = B.loss_dataset + sh_d, B.loss_market + sh_m
    B["bo"] = (B.bd + B.bm) / 2
    pub = pd.read_csv(leaderboard_csv)
    pub["key"] = pub["Team"].astype(str) + " / " + pub["Model"].astype(str)
    m = pub.drop_duplicates("key").merge(B, left_on="key", right_index=True)
    m = m[(m["N dataset"] == m.n_dataset) & (m["N market"] == m.n_market)]
    N["reproduction"] = dict(
        published_rows=len(pub), matched=len(m),
        dataset=dict(spearman=r(spearmanr(m["Brier Dataset"], m.bd)[0], 4), mean_abs_diff=r(np.abs(m["Brier Dataset"] - m.bd).mean(), 4), max_abs_diff=r(np.abs(m["Brier Dataset"] - m.bd).max(), 4)),
        market=dict(spearman=r(spearmanr(m["Brier Market"], m.bm)[0], 4), mean_abs_diff=r(np.abs(m["Brier Market"] - m.bm).mean(), 4), max_abs_diff=r(np.abs(m["Brier Market"] - m.bm).max(), 4)),
        overall=dict(spearman=r(spearmanr(m["Brier Overall"], m.bo)[0], 4), mean_abs_diff=r(np.abs(m["Brier Overall"] - m.bo).mean(), 4), max_abs_diff=r(np.abs(m["Brier Overall"] - m.bo).max(), 4)),
        super_published=dict(dataset=float(pub.loc[pub.key == SUPER, "Brier Dataset"].iloc[0]), market=float(pub.loc[pub.key == SUPER, "Brier Market"].iloc[0]), overall=float(pub.loc[pub.key == SUPER, "Brier Overall"].iloc[0])),
        super_ours=dict(dataset=r(B.loc[SUPER, "bd"]), market=r(B.loc[SUPER, "bm"]), overall=r(B.loc[SUPER, "bo"])),
    )

    # ---- value captured under each user profile --------------------------
    Vp = pd.DataFrame({k: V(l) for k, l in PROFILES.items()}, index=names)
    Vp["brier_adj"] = B["bo"].to_numpy()
    Vp["n_dataset"], Vp["n_market"] = R["n_dataset"], R["n_market"]

    def row(name):
        i = idx[name]
        out = dict(name=name, brier_adj=r(Vp.brier_adj.iloc[i]), n_dataset=int(Vp.n_dataset.iloc[i]), n_market=int(Vp.n_market.iloc[i]))
        for k, l in PROFILES.items():
            lo, _, hi = R["q_V"][l]["overall"][i]
            out[k] = dict(v=r(Vp[k].iloc[i]), lo=r(lo), hi=r(hi))
        return out

    N["anchors"] = {k: row(n) for k, n in dict(super=SUPER, public=PUBLIC, naive=NAIVE, price=PRICE, half=HALF).items()}
    Vc = Vp[cand].sort_values("uniform", ascending=False)
    top = []
    for name in Vc.index[:10]:
        d = row(name)
        for k in PROFILES:
            d[k]["rank"] = int(Vc[k].rank(ascending=False)[name])
        top.append(d)
    N["top_candidates"] = top
    N["candidate_summary"] = {k: dict(median=r(Vc[k].median()), q25=r(Vc[k].quantile(.25)), q75=r(Vc[k].quantile(.75)),
                                      share_nonpositive=r((Vc[k] <= 0).mean())) for k in PROFILES}
    N["profile_rank_corr"] = dict(
        candidates={k: r(spearmanr(Vc["uniform"], Vc[k])[0]) for k in PROFILES if k != "uniform"},
        listed={k: r(spearmanr(Vp[listed]["uniform"], Vp[listed][k])[0]) for k in PROFILES if k != "uniform"},
    )
    # by how many places does a candidate's rank move between Brier and each profile?
    N["rank_moves"] = {}
    for k in PROFILES:
        if k == "uniform":
            continue
        mv = (Vc[k].rank(ascending=False) - Vc["uniform"].rank(ascending=False)).abs()
        top20 = Vc.index[:20]
        N["rank_moves"][k] = dict(median=r(mv.median(), 1), p90=r(mv.quantile(.9), 1),
                                  top20_by_brier_still_top20=int((Vc[k].rank(ascending=False)[top20] <= 20).sum()))

    # ---- superforecasters vs today's best AI, difficulty-linked ----------
    best = Vc.index[0]
    N["best_ai"] = best
    N["super_vs_best_ai_linked"] = {}
    for k, l in PROFILES.items():
        bs = R["boot_V"][l]["overall"]
        d = bs[idx[SUPER]] - bs[idx[best]]
        N["super_vs_best_ai_linked"][k] = dict(diff=r(Vp.loc[SUPER, k] - Vp.loc[best, k]), lo=r(np.nanpercentile(d, 2.5)), hi=r(np.nanpercentile(d, 97.5)), p_super_ahead=r(np.nanmean(d > 0)))

    # ---- curves across theta ---------------------------------------------
    curves = {}
    for kind in ("overall", "dataset", "market"):
        M = np.array([V(f"theta:{t:.2f}", kind) for t in th]).T      # (F, T)
        c = M[cand]
        curves[kind] = dict(
            cand_median=r(np.median(c, 0)), cand_q25=r(np.percentile(c, 25, 0)), cand_q75=r(np.percentile(c, 75, 0)),
            cand_q10=r(np.percentile(c, 10, 0)), cand_q90=r(np.percentile(c, 90, 0)), cand_max=r(c.max(0)),
            share_nonpositive=r((c <= 0).mean(0)),
            best_ai=r(M[idx[best]]), super=r(M[idx[SUPER]]), public=r(M[idx[PUBLIC]]), price=r(M[idx[PRICE]]), naive=r(M[idx[NAIVE]]),
            share_beating_price=r((c > M[idx[PRICE]][None, :]).mean(0)),
        )
        rows = {n: j for j, n in enumerate(R["boot_rows"])}
        for lab, nm in (("best_ai", best), ("super", SUPER)):
            q = np.array([R["q_V"][f"theta:{t:.2f}"][kind][idx[nm]] for t in th])
            curves[kind][f"{lab}_lo"], curves[kind][f"{lab}_hi"] = r(q[:, 0]), r(q[:, 2])
    N["thetas"] = r(th, 2)
    N["curves"] = curves
    # where is value concentrated?
    cm = np.array(curves["overall"]["cand_median"], float)
    N["median_candidate_value_at"] = {f"{t:.2f}": r(cm[np.argmin(np.abs(th - t))]) for t in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98)}
    N["median_candidate_peak"] = dict(theta=r(th[int(np.argmax(cm))], 2), value=r(cm.max()))
    N["share_beating_price_profiles"] = {k: r((V(l, "market")[cand] > V(l, "market")[idx[PRICE]]).mean()) for k, l in PROFILES.items()}
    N["price_value_market"] = {k: r(V(l, "market")[idx[PRICE]]) for k, l in PROFILES.items()}
    N["median_candidate_value_market"] = {k: r(np.median(V(l, "market")[cand])) for k, l in PROFILES.items()}
    N["median_candidate_value_dataset"] = {k: r(np.median(V(l, "dataset")[cand])) for k, l in PROFILES.items()}

    # ---- selection experiment --------------------------------------------
    S = R["split"]
    sel = {}
    for cname, cset in (("candidates", cand), ("wide", cand_wide)):
        sel[cname] = {}
        for kind in ("overall", "dataset", "market"):
            trB = S["brier"]["train"][kind][cset]
            ok_cols = np.isfinite(trB).all(0)
            rows_ = []
            for t in R["sel_thetas"]:
                tr = S[f"theta:{t:.2f}"]["train"][kind][cset]
                te = S[f"theta:{t:.2f}"]["test"][kind][cset]
                good = ok_cols & np.isfinite(tr).all(0) & np.isfinite(te).all(0)
                trg, teg, trBg = tr[:, good], te[:, good], trB[:, good]
                cols = np.arange(good.sum())
                rec = dict(theta=r(t, 2), n_splits=int(good.sum()),
                           rho_brier=r(spearman_cols(trBg, teg).mean()), rho_own=r(spearman_cols(trg, teg).mean()),
                           median_candidate=r(np.median(teg, 0).mean()))
                for k in (1, 5):
                    byB = teg[np.argsort(-trBg, 0)[:k], cols].mean(0)
                    byT = teg[np.argsort(-trg, 0)[:k], cols].mean(0)
                    d = byT - byB
                    rec[f"top{k}"] = dict(by_brier=r(byB.mean()), by_own=r(byT.mean()), diff=r(d.mean()),
                                          lo=r(np.percentile(d, 2.5)), hi=r(np.percentile(d, 97.5)),
                                          p_own_better=r((d > 0).mean()), p_brier_better=r((d < 0).mean()))
                rows_.append(rec)
            sel[cname][kind] = rows_
    N["selection"] = sel
    # profile-level selection
    selp = {}
    for kind in ("overall", "dataset", "market"):
        trB = S["brier"]["train"][kind][cand]
        selp[kind] = {}
        for k, l in PROFILES.items():
            tr, te = S[l]["train"][kind][cand], S[l]["test"][kind][cand]
            cols = np.arange(tr.shape[1])
            byB, byT = te[np.argmax(trB, 0), cols], te[np.argmax(tr, 0), cols]
            selp[kind][k] = dict(by_brier=r(byB.mean()), by_own=r(byT.mean()), rho_brier=r(spearman_cols(trB, te).mean()), rho_own=r(spearman_cols(tr, te).mean()))
    N["selection_profiles"] = selp

    # ---- recalibration and hedging (each forecaster on its own items) -----
    T = Q["table"]
    Tc = T[T.f.isin(np.flatnonzero(cand))]
    rec = {}
    for qt in ("dataset", "market"):
        t = Tc[Tc.qtype == qt]
        C = np.array([Q["curves"][(qt, f)] for f in t.f])
        with np.errstate(invalid="ignore", divide="ignore"):
            Vraw, Vcal = 1 - C[:, 0] / C[:, 2], 1 - C[:, 1] / C[:, 2]
        d = dict(n=len(t), brier_raw_median=r(t.brier_raw.median(), 4), brier_cal_median=r(t.brier_cal.median(), 4),
                 share_le_05=r(t.share_le_05.median()), share_ge_95=r(t.share_ge_95.median()),
                 share_le_10=r(t.share_le_10.median()), share_ge_90=r(t.share_ge_90.median()), share_mid=r(t.share_mid.median()),
                 V_raw_median=r(np.nanmedian(Vraw, 0)), V_cal_median=r(np.nanmedian(Vcal, 0)),
                 share_raw_nonpositive=r(np.nanmean(Vraw <= 0, 0)), share_cal_nonpositive=r(np.nanmean(Vcal <= 0, 0)),
                 profiles={})
        for k, raw in RAW_PROFILE.items():
            g = t[f"V_cal[{raw}]"] - t[f"V_raw[{raw}]"]
            d["profiles"][k] = dict(raw=r(t[f"V_raw[{raw}]"].median()), cal=r(t[f"V_cal[{raw}]"].median()), gain_median=r(g.median()),
                                    gain_q25=r(g.quantile(.25)), gain_q75=r(g.quantile(.75)), share_improved=r((g > 0).mean()))
        rec[qt] = d
    hs = T[(T.forecaster == SUPER)].set_index("qtype")
    rec["super_dataset_hedging"] = dict(share_le_05=r(hs.loc["dataset", "share_le_05"]), share_ge_95=r(hs.loc["dataset", "share_ge_95"]), share_mid=r(hs.loc["dataset", "share_mid"]))
    N["recalibration"] = rec

    # ---- round 1: humans and AI on identical items ------------------------
    r1 = E["round1"]
    pt, dr = r1["point"], r1["draws"]
    one = dict(n_items=r1["n_items"], n_questions=r1["n_questions"], n_market=r1["n_market"], n_ai=r1["n_ai"], best_ai_name=pt["best_ai_name"], who={})
    for lab in ("super", "public", "best_ai", "median_ai"):
        one["who"][lab] = dict(brier=r(pt[lab]["brier"], 4))
        for k, raw in RAW_PROFILE.items():
            x = np.array([d[lab][raw] for d in dr])
            one["who"][lab][k] = dict(v=r(pt[lab][raw]), lo=r(np.percentile(x, 2.5)), hi=r(np.percentile(x, 97.5)))
        one["who"][lab]["curve"] = r(pt[lab]["curve"])
        cz = np.array([d[lab]["curve"] for d in dr])
        one["who"][lab]["curve_lo"], one["who"][lab]["curve_hi"] = r(np.nanpercentile(cz, 2.5, 0)), r(np.nanpercentile(cz, 97.5, 0))
    one["super_minus_best_ai"] = {}
    for k, raw in RAW_PROFILE.items():
        x = np.array([d["super"][raw] - d["best_ai"][raw] for d in dr])
        one["super_minus_best_ai"][k] = dict(diff=r(pt["super"][raw] - pt["best_ai"][raw]), lo=r(np.percentile(x, 2.5)), hi=r(np.percentile(x, 97.5)), p=r((x > 0).mean()))
    N["round1"] = one

    # ---- interval width --------------------------------------------------
    d = E["design"]
    N["design_effect"] = dict(median_ratio=r(np.nanmedian(d["ratio_listed"]), 2), q25=r(np.nanpercentile(d["ratio_listed"], 25), 2),
                              q75=r(np.nanpercentile(d["ratio_listed"], 75), 2), items_per_question=r(d["items_per_question"], 1),
                              share_ratio_above_1=r(np.nanmean(d["ratio_listed"] > 1)))

    json.dump(N, open(f"{outdir}/numbers.json", "w"), indent=1)
    print("wrote", f"{outdir}/numbers.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
