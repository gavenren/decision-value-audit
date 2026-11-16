"""Figures for the report, drawn only from out/numbers.json.

Usage: python scripts/make_figures.py out/numbers.json figs/
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150,
})
XT = [0.02, 0.25, 0.5, 0.75, 0.98]
XLAB = r"cost-loss ratio $\theta$"


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def frame(ax, ylim, zero=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_xticks(XT)
    ax.set_xticklabels([f"{t:g}" for t in XT])
    if zero:
        ax.axhline(0, color=MUTED, lw=0.8)


def fig1(N, out):
    th = arr(N["thetas"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15), sharey=True)
    # (a) identical items, round 1
    ax = axes[0]
    who = N["round1"]["who"]
    for lab, col, ls, name in (("super", ORANGE, "-", "Superforecaster median"), ("best_ai", BLUE, "-", "Best AI of July 2024")):
        ax.fill_between(th, arr(who[lab]["curve_lo"]), arr(who[lab]["curve_hi"]), color=col, alpha=0.13, lw=0)
        ax.plot(th, arr(who[lab]["curve"]), color=col, lw=1.5, ls=ls, label=name)
    ax.plot(th, arr(who["public"]["curve"]), color=AQUA, lw=1.3, ls=(0, (4, 1.5)), label="Public median")
    ax.plot(th, arr(who["median_ai"]["curve"]), color=MUTED, lw=1.2, ls=":", label="Median AI of July 2024")
    frame(ax, (-0.35, 0.85))
    ax.set_title(f"a  Same {N['round1']['n_items']} forecasts, July 2024 round")
    ax.set_ylabel("value captured\n(1 = perfect foresight, 0 = base rates)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.2, columnspacing=1.2)
    ax.set_xlabel(XLAB)
    # (b) pooled
    ax = axes[1]
    c = N["curves"]["overall"]
    ax.fill_between(th, arr(c["cand_q10"]), arr(c["cand_q90"]), color=MUTED, alpha=0.10, lw=0)
    ax.fill_between(th, arr(c["cand_q25"]), arr(c["cand_q75"]), color=MUTED, alpha=0.18, lw=0)
    ax.plot(th, arr(c["cand_median"]), color=MUTED, lw=1.2, ls=":", label=f"Median of {N['data']['candidates']} AI forecasters")
    ax.fill_between(th, arr(c["super_lo"]), arr(c["super_hi"]), color=ORANGE, alpha=0.13, lw=0)
    ax.plot(th, arr(c["super"]), color=ORANGE, lw=1.5, label="Superforecaster median (linked)")
    ax.fill_between(th, arr(c["best_ai_lo"]), arr(c["best_ai_hi"]), color=BLUE, alpha=0.15, lw=0)
    ax.plot(th, arr(c["best_ai"]), color=BLUE, lw=1.5, label="Best AI by Brier, 2024-26")
    frame(ax, (-0.35, 0.85))
    ax.set_title(f"b  {N['data']['rounds']} rounds, difficulty-adjusted")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.2, columnspacing=1.2)
    ax.set_xlabel(XLAB)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{out}/fig1_value_curves.pdf")
    plt.close(fig)


def fig2(N, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.0))
    sel = N["selection"]["candidates"]
    for ax, kind, title in ((axes[0], "dataset", "a  Dataset: rank agreement"), (axes[1], "market", "b  Market: rank agreement")):
        rows = sel[kind]
        th = arr([x["theta"] for x in rows])
        ax.plot(th, arr([x["rho_own"] for x in rows]), color=VIOLET, lw=1.4, marker="o", ms=3, label="ranked by past value at this $\\theta$")
        ax.plot(th, arr([x["rho_brier"] for x in rows]), color=BLUE, lw=1.4, marker="s", ms=3, ls=(0, (4, 1.5)), label="ranked by past Brier score")
        frame(ax, (0.2, 1.0), zero=False)
        ax.set_xticks([0.02, 0.25, 0.5, 0.75, 0.98])
        ax.set_xticklabels(["0.02", "0.25", "0.5", "0.75", "0.98"])
        ax.set_title(title)
        ax.set_ylabel("Spearman correlation with\nvalue on held-out questions" if kind == "dataset" else "")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.4)
    ax = axes[2]
    rows = sel["dataset"]
    th = arr([x["theta"] for x in rows])
    ax.plot(th, arr([x["top5"]["by_own"] for x in rows]), color=VIOLET, lw=1.4, marker="o", ms=3, label="five best by past value at this $\\theta$")
    ax.plot(th, arr([x["top5"]["by_brier"] for x in rows]), color=BLUE, lw=1.4, marker="s", ms=3, ls=(0, (4, 1.5)), label="five best by past Brier score")
    ax.plot(th, arr([x["median_candidate"] for x in rows]), color=MUTED, lw=1.1, ls=":", label="median forecaster")
    frame(ax, (-0.05, 0.6))
    ax.set_xticks([0.02, 0.25, 0.5, 0.75, 0.98])
    ax.set_xticklabels(["0.02", "0.25", "0.5", "0.75", "0.98"])
    ax.set_title("c  Dataset: value of the pick")
    ax.set_ylabel("held-out value captured")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.4)
    for a in axes:
        a.set_xlabel(XLAB)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(f"{out}/fig2_selection.pdf")
    plt.close(fig)


def fig3(N, out):
    th = arr(N["thetas"])
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.0))
    rec = N["recalibration"]["dataset"]
    ax = axes[0]
    ax.plot(th, arr(rec["V_raw_median"]), color=BLUE, lw=1.4, label="as submitted")
    ax.plot(th, arr(rec["V_cal_median"]), color=VIOLET, lw=1.4, ls=(0, (4, 1.5)), label="after recalibration")
    frame(ax, (-0.05, 0.6))
    ax.set_title("a  Dataset: median AI value")
    ax.set_ylabel("value captured")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.4)
    ax = axes[1]
    ax.plot(th, 100 * arr(rec["share_raw_nonpositive"]), color=BLUE, lw=1.4, label="as submitted")
    ax.plot(th, 100 * arr(rec["share_cal_nonpositive"]), color=VIOLET, lw=1.4, ls=(0, (4, 1.5)), label="after recalibration")
    frame(ax, (0, 100), zero=False)
    ax.set_title("b  Dataset: no value added")
    ax.set_ylabel("% of AI forecasters\nno better than base rates")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.4)
    ax = axes[2]
    c = N["curves"]["market"]
    ax.fill_between(th, arr(c["cand_q25"]), arr(c["cand_q75"]), color=MUTED, alpha=0.18, lw=0)
    ax.plot(th, arr(c["cand_median"]), color=MUTED, lw=1.2, ls=":", label="median AI forecaster (band: middle 50%)")
    ax.plot(th, arr(c["price"]), color=ORANGE, lw=1.5, label="market price itself")
    ax.plot(th, arr(c["best_ai"]), color=BLUE, lw=1.4, ls=(0, (4, 1.5)), label="best AI by Brier")
    frame(ax, (-0.35, 0.85))
    ax.set_title("c  Market: against the price")
    ax.set_ylabel("value captured")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.4)
    for a in axes:
        a.set_xticks([0.02, 0.25, 0.5, 0.75, 0.98])
        a.set_xticklabels(["0.02", "0.25", "0.5", "0.75", "0.98"])
    for a in axes:
        a.set_xlabel(XLAB)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(f"{out}/fig3_recalibration_price.pdf")
    plt.close(fig)


if __name__ == "__main__":
    N = json.load(open(sys.argv[1]))
    for f in (fig1, fig2, fig3):
        f(N, sys.argv[2])
    print("figures written to", sys.argv[2])
