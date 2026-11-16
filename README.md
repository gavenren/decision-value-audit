# oed: decision value of probability forecasts

Code for "Does a Better Brier Score Buy Better Decisions? A Decision-Value Audit
of AI Forecasters on ForecastBench" (AI x Epistemics Research Sprint, November 2026).
All code in this repository was written during the sprint.

## What it does

For binary forecasts it computes, at any cost-loss ratio `theta` (the user acts
when the forecast exceeds `theta`):

* regret against perfect foresight;
* **value captured** = 1 - regret / regret of acting on base rates
  (1 = perfect foresight, 0 = base rates, below 0 = worse than not asking);
* the same averaged over a Beta-distributed population of users. Beta(1,1)
  reproduces the Brier ranking.

It applies this to every ForecastBench processed forecast set, with the
tournament leaderboard's difficulty adjustment.

## Reproduce

Needs Python 3.11+, numpy, pandas, scipy, pyarrow, matplotlib, pytest, and
pdflatex for the report.

1. Download and extract
   `https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz`
2. Get `leaderboards/csv/leaderboard_tournament.csv` from
   `https://github.com/forecastingresearch/forecastbench-datasets`
   (used only for the reproduction check).
3. `./run_all.sh <extracted folder> <leaderboard_tournament.csv>`

This runs the tests, rebuilds `out/numbers.json`, re-derives the headline
numbers with independent code (`scripts/verify_independent.py`), and rebuilds
the figures and `paper/report.pdf`. About five minutes on two cores. Seeds are
fixed; two runs give identical `numbers.json`.

Results in the report use the processed forecast sets downloaded on
13 November 2026 and the leaderboard file from the 11 November 2026 commit.
The data update nightly, so later downloads will give slightly different numbers.

## Layout

| Path | Contents |
|---|---|
| `oed/core.py` | regret, Murphy curves, value captured, isotonic recalibration |
| `oed/panel.py` | leaderboard filters, closed-form profile scores, two-way fixed effects |
| `oed/pooled.py` | the difficulty-adjusted panel |
| `oed/resample.py` | question-level bootstrap and half-splits |
| `scripts/ingest.py` | flatten the forecast files into one table |
| `scripts/run_pooled.py` | value tables, curves, selection experiment |
| `scripts/run_recal.py` | recalibration and hedging |
| `scripts/run_extras.py` | humans vs AI on identical items; interval-width check |
| `scripts/make_numbers.py` | every number in the report -> `out/numbers.json` |
| `scripts/make_figures.py`, `scripts/make_tex.py` | figures, LaTeX macros and table |
| `scripts/verify_independent.py` | independent re-derivation of headline numbers |
| `tests/` | 18 unit tests |
| `out/numbers.json` | the numbers behind the report |

## Data licence

ForecastBench data are CC BY-SA 4.0 (Karger et al., ICLR 2025). No data are
redistributed here.
