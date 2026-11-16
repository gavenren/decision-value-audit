#!/usr/bin/env bash
# Rebuild every number, table and figure in the report from the public data.
#
#   ./run_all.sh <dir with extracted processed_forecast_sets.tar.gz> <leaderboard_tournament.csv> [out dir]
#
# The first argument is the folder that contains one sub-folder per round
# (2024-07-21/, 2025-03-02/, ...).  The second is leaderboards/csv/leaderboard_tournament.csv
# from github.com/forecastingresearch/forecastbench-datasets, used only for the reproduction check.
set -euo pipefail
SETS="$1"; LB="$2"; OUT="${3:-out}"
cd "$(dirname "$0")"
export PYTHONPATH=.
mkdir -p "$OUT" figs paper
python3 -m pytest -q tests
python3 scripts/ingest.py "$SETS" "$OUT/forecasts.parquet"
python3 scripts/run_pooled.py "$OUT/forecasts.parquet" "$OUT/pooled_results.pkl"
python3 scripts/run_recal.py "$OUT/forecasts.parquet" "$OUT/pooled_results.pkl" "$OUT/recal_results.pkl"
python3 scripts/run_extras.py "$OUT/forecasts.parquet" "$OUT/pooled_results.pkl" "$OUT/extras.pkl"
python3 scripts/make_numbers.py "$OUT" "$LB"
if [ "$OUT" = "out" ]; then
  python3 scripts/verify_independent.py
  python3 scripts/make_figures.py "$OUT/numbers.json" figs
  python3 scripts/make_tex.py "$OUT/numbers.json" paper
  (cd paper && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null)
  echo "report: paper/report.pdf"
fi
