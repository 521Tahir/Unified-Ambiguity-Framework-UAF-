#!/usr/bin/env python3
"""
Collect test results from all method/dataset run directories
and print Table 1 (main comparison table).

Usage:
  python scripts/eval/collect_results_table1.py \
      --results_root experiments/results \
      --out experiments/tables/table1.csv
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


METHODS_ORDER = ["baseline", "mult", "mmdfn", "ctnet", "chado"]
DATASETS_ORDER = ["iemocap", "meld", "mosei"]


def load_result(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="experiments/results")
    p.add_argument("--out", default="experiments/tables/table1.csv")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = []

    for dataset in DATASETS_ORDER:
        for method in METHODS_ORDER:
            run_dir = root / dataset / method
            result_file = run_dir / "test_results.json"
            r = load_result(result_file)
            if not r:
                continue

            row = {
                "Dataset": dataset.upper(),
                "Method": method.upper(),
                "Accuracy": round(r.get("accuracy", r.get("subset_accuracy", float("nan"))) * 100, 2),
                "Macro-F1": round(r.get("macro_f1", float("nan")) * 100, 2),
                "Weighted-F1": round(r.get("weighted_f1", float("nan")) * 100, 2),
                "Macro-Prec": round(r.get("macro_precision", float("nan")) * 100, 2),
                "Macro-Rec": round(r.get("macro_recall", float("nan")) * 100, 2),
            }
            rows.append(row)

    if not rows:
        print("[WARN] No results found. Run training first.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
