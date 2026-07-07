#!/usr/bin/env python
"""Aggregate experiments/*/results.json into paper-ready LaTeX/Markdown tables.

    python scripts/aggregate_results.py --glob "experiments/*/results.json" --fmt md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def load(pattern: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(pattern):
        name = Path(path).parent.name
        data = json.loads(Path(path).read_text())
        for dataset, m in data.items():
            rows.append({"model": name, "dataset": dataset,
                         "eer": m.get("eer"), "min_tdcf": m.get("min_tdcf"),
                         "auroc": m.get("auroc"), "ece": m.get("ece")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/*/results.json")
    ap.add_argument("--fmt", choices=["md", "latex", "csv"], default="md")
    ap.add_argument("--metric", default="eer")
    args = ap.parse_args()

    df = load(args.glob)
    if df.empty:
        print("no results found — run scripts/evaluate.py first")
        return
    pivot = df.pivot_table(index="model", columns="dataset", values=args.metric)
    if args.fmt == "md":
        print(pivot.round(4).to_markdown())
    elif args.fmt == "latex":
        print(pivot.round(4).to_latex(float_format="%.4f"))
    else:
        print(pivot.round(4).to_csv())


if __name__ == "__main__":
    main()
