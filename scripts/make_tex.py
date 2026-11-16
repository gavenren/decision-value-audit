"""Turn out/numbers.json into LaTeX macros and the main table.

Every number in the report is one of these macros, so the text cannot drift
from the analysis output.

Usage: python scripts/make_tex.py out/numbers.json paper/
"""

import json
import sys

N = json.load(open(sys.argv[1]))
out = sys.argv[2]
M = {}


def f(x, k=2):
    s = f"{x:.{k}f}"
    return s.replace("-", "$-$")


def pc(x, k=0):
    return f"{100 * x:.{k}f}".replace("-", "$-$")


def ci(d, k=2, key="diff"):
    return f"{f(d[key], k)} (95\\% CI {f(d['lo'], k)} to {f(d['hi'], k)})"


def oed(d):
    return f"{pc(d['v'])} ({pc(d['lo'])} to {pc(d['hi'])})"


th = N["thetas"]
at = lambda arr, t: arr[th.index(t)]
D = N["data"]

M["nForecasts"] = f"{D['forecasts']:,}"
M["nForecastsM"] = f"{D['forecasts'] / 1e6:.2f}"
M["nForecastsDataset"] = f"{D['forecasts_dataset']:,}"
M["nForecastsMarket"] = f"{D['forecasts_market']:,}"
M["nItems"] = f"{D['items_dataset'] + D['items_market']:,}"
M["nItemsDataset"] = f"{D['items_dataset']:,}"
M["nItemsMarket"] = f"{D['items_market']:,}"
M["nQuestions"] = f"{D['questions']:,}"
M["nQuestionsDataset"] = f"{D['questions_dataset']:,}"
M["nRounds"] = str(D["rounds"])
M["nListed"] = str(D["listed"])
M["nCand"] = str(D["candidates"])
M["nCandWide"] = str(D["candidates_wide"])
M["nFilesDropped"] = str(D["files_dropped_for_imputation"])
M["nBoot"] = f"{D['n_boot']:,}"
M["nSplits"] = str(D["n_splits"])

R = N["reproduction"]
M["reproMatched"], M["reproPublished"] = str(R["matched"]), str(R["published_rows"])
M["reproRhoOverall"], M["reproRhoDataset"], M["reproRhoMarket"] = f(R["overall"]["spearman"], 3), f(R["dataset"]["spearman"], 3), f(R["market"]["spearman"], 3)
M["reproMadDataset"], M["reproMadMarket"], M["reproMadOverall"] = f(R["dataset"]["mean_abs_diff"], 3), f(R["market"]["mean_abs_diff"], 4), f(R["overall"]["mean_abs_diff"], 3)
M["superPubOverall"], M["superOursOverall"] = f(R["super_published"]["overall"], 3), f(R["super_ours"]["overall"], 3)

A = N["anchors"]
for who, key in (("super", "Super"), ("public", "Public")):
    for p, pk in (("uniform", "Uni"), ("precaution", "Pre"), ("high_bar", "High"), ("even", "Even")):
        M[f"oed{key}{pk}"] = oed(A[who][p])
        M[f"v{key}{pk}"] = pc(A[who][p]["v"])
best = N["top_candidates"][0]
for p, pk in (("uniform", "Uni"), ("precaution", "Pre"), ("high_bar", "High"), ("even", "Even")):
    M[f"oedBest{pk}"] = oed(best[p])
    M[f"vBest{pk}"] = pc(best[p]["v"])
    M[f"vMed{pk}"] = pc(N["candidate_summary"][p]["median"])
    M[f"linkDiff{pk}"] = ci(N["super_vs_best_ai_linked"][p])
    M[f"linkP{pk}"] = pc(N["super_vs_best_ai_linked"][p]["p_super_ahead"], 1)
M["shareNonposHigh"] = pc(N["candidate_summary"]["high_bar"]["share_nonpositive"])
M["shareNonposUni"] = pc(N["candidate_summary"]["uniform"]["share_nonpositive"])
C = N["profile_rank_corr"]["candidates"]
M["rhoPre"], M["rhoHigh"], M["rhoEven"] = f(C["precaution"]), f(C["high_bar"]), f(C["even"])
M["moveHigh"] = f"{N['rank_moves']['high_bar']['median']:.0f}"
M["movePre"] = f"{N['rank_moves']['precaution']['median']:.0f}"
M["keepHigh"] = str(N["rank_moves"]["high_bar"]["top20_by_brier_still_top20"])
M["keepPre"] = str(N["rank_moves"]["precaution"]["top20_by_brier_still_top20"])

cu = N["curves"]
M["peakTheta"], M["peakValue"] = f(N["median_candidate_peak"]["theta"]), pc(N["median_candidate_peak"]["value"])
mv = N["median_candidate_value_at"]
M["medAtLow"], M["medAtHigh"], M["medAtHalf"] = pc(mv["0.02"]), pc(mv["0.95"], 1), pc(mv["0.50"])
M["medAtFive"], M["medAtNinety"] = pc(mv["0.05"]), pc(mv["0.90"])
M["nonposLow"] = pc(at(cu["overall"]["share_nonpositive"], 0.02))
M["nonposMid"] = pc(at(cu["overall"]["share_nonpositive"], 0.2))
M["nonposHigh"] = pc(at(cu["overall"]["share_nonpositive"], 0.95))
M["bestAtLow"], M["bestAtHigh"] = pc(at(cu["overall"]["best_ai"], 0.02)), pc(at(cu["overall"]["best_ai"], 0.95))

M["beatPriceShare"] = pc(N["share_beating_price_profiles"]["uniform"], 1)
M["beatPriceCount"] = str(round(N["share_beating_price_profiles"]["uniform"] * D["candidates"]))
M["beatPriceMax"] = pc(max(N["share_beating_price_profiles"].values()), 1)
M["priceValue"] = pc(N["price_value_market"]["uniform"])
M["medMarketValue"] = pc(N["median_candidate_value_market"]["uniform"])
M["medDatasetValue"] = pc(N["median_candidate_value_dataset"]["uniform"])

S = N["selection"]["candidates"]
W = N["selection"]["wide"]
row = lambda rows, t: next(x for x in rows if abs(x["theta"] - t) < 1e-9)
for t, tk in ((0.30, "Mid"), (0.90, "Ninety"), (0.95, "NinetyFive")):
    x = row(S["dataset"], t)
    M[f"rhoB{tk}"], M[f"rhoO{tk}"] = f(x["rho_brier"]), f(x["rho_own"])
x = row(W["dataset"], 0.90)
M["rhoBNinetyWide"], M["rhoONinetyWide"] = f(x["rho_brier"]), f(x["rho_own"])
x = row(S["dataset"], 0.90)["top5"]
M["pickB"], M["pickO"] = pc(x["by_brier"]), pc(x["by_own"])
M["pickDiff"] = f"{pc(x['diff'])} points (middle 95\\% of splits: {pc(x['lo'])} to {pc(x['hi'])})"
M["pickPOwn"] = pc(x["p_own_better"])
x = row(W["dataset"], 0.90)["top5"]
M["pickDiffWide"] = f"{pc(x['diff'])} points ({pc(x['lo'])} to {pc(x['hi'])})"
x = row(S["market"], 0.50)
M["rhoBMarket"], M["rhoOMarket"] = f(x["rho_brier"]), f(x["rho_own"])
M["brierWinsOverall"] = str(sum(1 for x in S["overall"] if x["top1"]["diff"] < 0))
M["brierWinsMarket"] = str(sum(1 for x in S["market"] if x["top5"]["diff"] < 0))
M["nSelThetas"] = str(len(S["overall"]))
sp = N["selection_profiles"]["overall"]
M["selPreB"], M["selPreO"] = pc(sp["precaution"]["by_brier"]), pc(sp["precaution"]["by_own"])
M["selHighB"], M["selHighO"] = pc(sp["high_bar"]["by_brier"]), pc(sp["high_bar"]["by_own"])
M["selEvenB"], M["selEvenO"] = pc(sp["even"]["by_brier"]), pc(sp["even"]["by_own"])

rc = N["recalibration"]["dataset"]
M["recalGainUni"], M["recalGainPre"], M["recalGainHigh"] = pc(rc["profiles"]["uniform"]["gain_median"], 1), pc(rc["profiles"]["precaution"]["gain_median"], 1), pc(rc["profiles"]["high_bar"]["gain_median"], 1)
M["recalBrierRaw"], M["recalBrierCal"] = f(rc["brier_raw_median"], 3), f(rc["brier_cal_median"], 3)
M["shareGeNinety"], M["shareLeTen"] = pc(rc["share_ge_90"], 1), pc(rc["share_le_10"], 1)
M["dsNonposNinetyRaw"], M["dsNonposNinetyCal"] = pc(at(rc["share_raw_nonpositive"], 0.9)), pc(at(rc["share_cal_nonpositive"], 0.9))
M["dsNonposHalfRaw"] = pc(at(rc["share_raw_nonpositive"], 0.5))

r1 = N["round1"]
M["rOneItems"], M["rOneQuestions"], M["rOneAI"], M["rOneMarket"] = str(r1["n_items"]), str(r1["n_questions"]), str(r1["n_ai"]), str(r1["n_market"])
w = r1["who"]
M["rOneSuperUni"], M["rOneBestUni"], M["rOneMedUni"], M["rOnePublicUni"] = oed(w["super"]["uniform"]), oed(w["best_ai"]["uniform"]), oed(w["median_ai"]["uniform"]), oed(w["public"]["uniform"])
M["rOneSuperPre"], M["rOneBestPre"] = pc(w["super"]["precaution"]["v"]), pc(w["best_ai"]["precaution"]["v"])
M["rOneDiffUni"] = ci(r1["super_minus_best_ai"]["uniform"])
M["rOneDiffPre"] = ci(r1["super_minus_best_ai"]["precaution"])
M["rOneDiffHigh"] = ci(r1["super_minus_best_ai"]["high_bar"])
M["rOneSuperLow"], M["rOneBestLow"] = pc(at(w["super"]["curve"], 0.02)), pc(at(w["best_ai"]["curve"], 0.02))

de = N["design_effect"]
M["deRatio"], M["deLo"], M["deHi"], M["dePer"] = f(de["median_ratio"]), f(de["q25"]), f(de["q75"]), f(de["items_per_question"], 0)
M["deShare"] = pc(de["share_ratio_above_1"])

with open(f"{out}/numbers.tex", "w") as fh:
    fh.write("% generated by scripts/make_tex.py from out/numbers.json; do not edit\n")
    for k, v in M.items():
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- Table 1 ---------------------------------------------------------------
SHORT = {
    "Google DeepMind / Gemini": "Google DeepMind / Gemini",
    "ForecastBench / gpt-5.6-sol-standard-medium-web-search-code-execution-zero-shot-with-freeze-values": "ForecastBench / gpt-5.6-sol (search, code; shown price)",
    "ForecastBench / gpt-5.6-sol-standard-medium-web-search-code-execution": "ForecastBench / gpt-5.6-sol (search, code)",
    "Artificial Judgement / aj-v1.5": "Artificial Judgement / aj-v1.5",
    "Artificial Judgement / aj-v1": "Artificial Judgement / aj-v1",
    "Anthral Labs / blf-ensemble-opus45-gpt55": "Anthral Labs / blf-ensemble-opus45-gpt55",
}
P = ("uniform", "precaution", "high_bar", "even")
lines = []


def cell(d, p, rank=True):
    s = pc(d[p]["v"])
    if rank and "rank" in d[p]:
        s += f" \\textcolor{{black!55}}{{({d[p]['rank']})}}"
    return s


sup = A["super"]
lines.append("Superforecaster median$^\\dagger$ & " + f(sup["brier_adj"], 3) + " & " + " & ".join(cell(sup, p, False) for p in P) + " \\\\")
lines.append("\\addlinespace")
for d in N["top_candidates"][:6]:
    lines.append(SHORT[d["name"]].replace("_", "\\_") + " & " + f(d["brier_adj"], 3) + " & " + " & ".join(cell(d, p) for p in P) + " \\\\")
lines.append("\\addlinespace")
cs = N["candidate_summary"]
lines.append(f"Median of {D['candidates']} AI forecasters & -- & " + " & ".join(pc(cs[p]["median"]) for p in P) + " \\\\")
pub = A["public"]
lines.append("Public median$^\\dagger$ & " + f(pub["brier_adj"], 3) + " & " + " & ".join(cell(pub, p, False) for p in P) + " \\\\")
lines.append("\\bottomrule")
open(f"{out}/table1.tex", "w").write("\n".join(lines) + "\n")
print(f"wrote {len(M)} macros and table1.tex to {out}")
