"""Independent re-derivation of headline numbers without the oed package.

Plain pandas/numpy/scipy on the flattened forecast table.  Any disagreement
with out/numbers.json beyond the stated tolerance is printed as FAIL.
"""
import json, sys
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

N = json.load(open("out/numbers.json"))
d = pd.read_parquet("out/forecasts.parquet")
d["fc"] = d["team"].astype(str) + " / " + d["model"].astype(str)
ok = True
def check(label, got, want, tol):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} {label}: independent {got:.4f} vs report {want:.4f}")

# --- filters, written again from the leaderboard's description ---------------
dummy_file = (d.team.astype(str) == "ForecastBench") & (d.model_org.astype(str) == "ForecastBench")
imp_all = d.groupby("file", observed=True).imputed.mean()
imp_qt = d.groupby(["file", "qtype"], observed=True).imputed.mean().groupby(level=0).max()
bad = set(imp_all[(imp_all > .05) | (imp_qt.reindex(imp_all.index) > .05)].index) - set(d.loc[dummy_file, "file"].unique())
d = d[~d.file.isin(bad)]
d = d[d.resolved & d.resolved_to.isin([0, 1])]
age = (pd.Timestamp(N["data"]["as_of"]) - pd.to_datetime(d.due_date.astype(str))).dt.days
d = d[age >= 50].copy()
d["item"] = d.due_date.astype(str) + "|" + d.source.astype(str) + "|" + d.id.astype(str) + "|" + np.where(d.qtype == "market", "", d.resolution_date.astype(str))
n_before = len(d)
d = d.drop_duplicates(["file", "item"])
n_file_dups = n_before - len(d)
n_before = len(d)
d = d.drop_duplicates(["fc", "item"])
print(f"info duplicate rows within a file: {n_file_dups}; same forecaster name twice on an item: {n_before - len(d)} of {n_before}")
cnt = d.drop_duplicates("item").groupby(["due_date", "qtype"], observed=True).size().unstack(fill_value=0)
keep = cnt[(cnt.dataset >= 225) & (cnt.market >= 50)].index
d = d[d.due_date.isin(keep)]
d["y"] = d.resolved_to.astype(float)
check("rounds", len(keep), N["data"]["rounds"], 0)
check("forecasts", len(d), N["data"]["forecasts"], 0)
check("underlying questions", d.groupby(["source", "id"], observed=True).ngroups, N["data"]["questions"], 0)

# --- is the 'Imputed Forecaster' the due-date market price? -------------------
m = d[d.qtype == "market"]
price = m[m.fc == "ForecastBench / Imputed Forecaster"].set_index("item")
print("info |price forecast - market_value_on_due_date| max:", float((price.p - price.market_value).abs().max()),
      "; share differing from the day-before value:", float(((price.p - price.market_value_minus_one).abs() > 1e-9).mean().round(3)))

base = d.drop_duplicates("item").groupby("source", observed=True).y.mean()
TH = (np.arange(2000) + 0.5) / 2000
def regret_matrix(p, y, th):           # brute force, (n, k)
    p, y = p[:, None], y[:, None]
    a = (p > th) + 0.5 * (p == th)
    return th * (1 - y) * a + (1 - th) * y * (1 - a)
def profile_loss(p, y, w):             # numeric integral over theta, chunked
    out = np.empty(len(p))
    for s in range(0, len(p), 20000):
        out[s:s + 20000] = regret_matrix(p[s:s + 20000], y[s:s + 20000], TH) @ w
    return out
from scipy.stats import beta as beta_dist
W = {"uniform": np.full(len(TH), 1 / len(TH))}
for k, (a, b) in {"precaution": (1, 9), "high_bar": (9, 1)}.items():
    w = beta_dist.pdf(TH, a, b); W[k] = w / w.sum()

best = N["best_ai"]; SUPER = "ForecastBench / Superforecaster median forecast"
items_m = m.drop_duplicates("item").set_index("item")
ds = d[d.qtype == "dataset"]
items_d = ds.drop_duplicates("item").set_index("item")

# dataset fixed effects by sparse least squares (a different solver from the report's)
est = ds[~((ds.team.astype(str) == "ForecastBench") & (ds.model.astype(str).isin(["Always 0", "Always 1", "Always 0.5", "Random Uniform", "Imputed Forecaster"]) | ds.model.astype(str).str.startswith("LLM Crowd")))]
mc = pd.Categorical(est.fc); ic = pd.Categorical(est.item, categories=items_d.index)
X = sp.hstack([sp.csr_matrix((np.ones(len(est)), (np.arange(len(est)), mc.codes)), shape=(len(est), len(mc.categories))),
               sp.csr_matrix((np.ones(len(est)), (np.arange(len(est)), ic.codes)), shape=(len(est), len(items_d)))]).tocsr()

def adjusted_value(profile):
    w = W[profile]
    out = {}
    # market: relative to the price on the same items
    lm = pd.Series(profile_loss(m.p.to_numpy(float), m.y.to_numpy(float), w), index=m.index)
    pl = lm[m.fc == "ForecastBench / Imputed Forecaster"].set_axis(m.item[m.fc == "ForecastBench / Imputed Forecaster"])
    ref_m = profile_loss(items_m.source.astype(str).map(base).to_numpy(float), items_m.y.to_numpy(float), w).mean()
    # dataset: two-way fixed effects
    ld = profile_loss(ds.p.to_numpy(float), ds.y.to_numpy(float), w)
    sol = lsqr(X, pd.Series(ld, index=ds.index).loc[est.index].to_numpy(), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    gamma = pd.Series(sol[len(mc.categories):], index=items_d.index)
    ref_d = profile_loss(items_d.source.astype(str).map(base).to_numpy(float), items_d.y.to_numpy(float), w).mean()
    ld = pd.Series(ld, index=ds.index)
    for who in (best, SUPER):
        mm = m[m.fc == who]
        adj_m = (lm[mm.index].to_numpy() - pl.loc[mm.item].to_numpy()).mean() + pl.mean()
        dd = ds[ds.fc == who]
        adj_d = (ld[dd.index].to_numpy() - gamma.loc[dd.item].to_numpy()).mean() + gamma.mean()
        out[who] = (1 - (adj_d + adj_m) / (ref_d + ref_m), 1 - adj_m / ref_m)
    # how many candidates beat the price on market questions (uniform only)
    if profile == "uniform":
        g = pd.DataFrame({"fc": m.fc, "diff": lm.to_numpy() - pl.loc[m.item].to_numpy()}).groupby("fc")["diff"].mean()
        nd = ds.groupby("fc").size(); nm = m.groupby("fc").size()
        real = ~g.index.str.contains(r"^ForecastBench / (?:Always|Random|Imputed|LLM Crowd|Naive|Superforecaster|Public)")
        cand = g.index[real & (nd.reindex(g.index).fillna(0) >= 2000).to_numpy() & (nm.reindex(g.index).fillna(0) >= 200).to_numpy()]
        out["n_cand"] = len(cand); out["beat"] = int((g.loc[cand] < 0).sum())
    return out

u = adjusted_value("uniform")
check("candidates", u["n_cand"], N["data"]["candidates"], 0)
check("candidates beating the market price (uniform)", u["beat"], round(N["share_beating_price_profiles"]["uniform"] * N["data"]["candidates"]), 0)
check("best AI value, uniform", u[best][0], N["top_candidates"][0]["uniform"]["v"], 0.002)
check("superforecaster value, uniform", u[SUPER][0], N["anchors"]["super"]["uniform"]["v"], 0.002)
pv = adjusted_value("precaution")
check("best AI value, precaution", pv[best][0], N["top_candidates"][0]["precaution"]["v"], 0.002)
check("superforecaster value, precaution", pv[SUPER][0], N["anchors"]["super"]["precaution"]["v"], 0.002)
check("super minus best AI, precaution", pv[SUPER][0] - pv[best][0], N["super_vs_best_ai_linked"]["precaution"]["diff"], 0.003)
hv = adjusted_value("high_bar")
check("best AI value, high bar", hv[best][0], N["top_candidates"][0]["high_bar"]["v"], 0.002)
check("superforecaster value, high bar", hv[SUPER][0], N["anchors"]["super"]["high_bar"]["v"], 0.002)
print("ALL OK" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
