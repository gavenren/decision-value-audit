"""Flatten every ForecastBench processed forecast set into one table.

Usage: python scripts/ingest.py <extracted-archive-dir> <out.parquet>

Keeps single (non-combination) questions only.  Every row is kept with its
flags (resolved, imputed, binary outcome) so that the analysis can state
exactly what it dropped and why.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "infer", "kalshi"}


def read_one(path: str):
    raw = json.loads(Path(path).read_text())
    rows = []
    for r in raw["forecasts"]:
        if isinstance(r["id"], list):
            continue
        rows.append((
            str(r["id"]), r["source"], (r.get("resolution_date") or "")[:10],
            r.get("forecast"), r.get("resolved_to"), bool(r.get("resolved")), bool(r.get("imputed")),
            r.get("market_value_on_due_date"), r.get("market_value_on_due_date_minus_one"),
        ))
    df = pd.DataFrame(rows, columns=["id", "source", "resolution_date", "p", "resolved_to", "resolved",
                                     "imputed", "market_value", "market_value_minus_one"])
    for c in ("p", "resolved_to", "market_value", "market_value_minus_one"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["file"] = Path(path).name
    df["due_date"] = str(raw.get("forecast_due_date"))[:10]
    df["team"] = raw.get("organization")
    df["model"] = raw.get("model")
    df["model_org"] = raw.get("model_organization")
    df["eligible"] = bool(raw.get("leaderboard_eligible"))
    return df


def main(src: str, out: str):
    files = sorted(str(p) for p in Path(src).rglob("*.json"))
    with ProcessPoolExecutor(2) as ex:
        parts = list(ex.map(read_one, files, chunksize=16))
    df = pd.concat(parts, ignore_index=True)
    df["qtype"] = df["source"].map(lambda s: "market" if s in MARKET_SOURCES else "dataset")
    for c in ("id", "source", "resolution_date", "file", "due_date", "team", "model", "model_org", "qtype"):
        df[c] = df[c].astype("category")
    df.to_parquet(out, index=False)
    print(f"{len(files)} files -> {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
