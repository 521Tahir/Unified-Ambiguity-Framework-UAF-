#!/usr/bin/env python3
"""
Compute mean ± std across 3 seeds for all models on a given dataset.

Usage:
  python3 scripts/eval/compute_multiseed_table.py --dataset iemocap
  python3 scripts/eval/compute_multiseed_table.py --dataset mosei
  python3 scripts/eval/compute_multiseed_table.py --dataset meld
  python3 scripts/eval/compute_multiseed_table.py --dataset all
"""
import argparse
import json
import os
import glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "experiments", "results")
OUT  = os.path.join(ROOT, "experiments", "results", "multiseed_tables")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 3407, 456]

MODELS = [
    ("AR-GOT",  "argot"),
    ("BP-MulT", "bpmult"),
    ("CTNet",   "ctnet"),
    ("MulT",    "mult"),
    ("MM-DFN",  "mmdfn"),
    ("LF-LSTM", "lflstm"),
    ("AER-LLM", "aerllm"),
    ("EmoCLIP", "emoclip"),
    ("OV-MER",  "ovmer"),
]

# Metrics per dataset
METRICS = {
    "iemocap": [("Acc",  "accuracy"),
                ("MF1",  "macro_f1"),
                ("WF1",  "weighted_f1")],
    "meld":    [("Acc",  "accuracy"),
                ("MF1",  "macro_f1"),
                ("WF1",  "weighted_f1")],
    "mosei":   [("wAcc", "wacc"),
                ("MF1",  "macro_f1"),
                ("WF1",  "weighted_f1")],
}


def load_seed_results(dataset, model_dir, seed):
    path = os.path.join(RES, dataset, f"{model_dir}_ms",
                        f"seed_{seed}", "test_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_metric(d, key):
    if d is None:
        return None
    # Accept both 'accuracy' and 'test_accuracy' etc.
    for k in [key, f"test_{key}"]:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def compute_table(dataset):
    metrics = METRICS[dataset]
    rows = []

    for display, model_dir in MODELS:
        seed_vals = {mk: [] for _, mk in metrics}
        missing = 0
        for seed in SEEDS:
            d = load_seed_results(dataset, model_dir, seed)
            if d is None:
                missing += 1
                continue
            for _, mk in metrics:
                v = get_metric(d, mk)
                if v is not None:
                    seed_vals[mk].append(v * 100.0)  # convert to %

        row = {"model": display, "missing_seeds": missing}
        for label, mk in metrics:
            vals = seed_vals[mk]
            if len(vals) >= 2:
                row[f"{label}_mean"] = np.mean(vals)
                row[f"{label}_std"]  = np.std(vals, ddof=1)
                row[f"{label}_vals"] = vals
            elif len(vals) == 1:
                row[f"{label}_mean"] = vals[0]
                row[f"{label}_std"]  = 0.0
                row[f"{label}_vals"] = vals
            else:
                row[f"{label}_mean"] = None
                row[f"{label}_std"]  = None
                row[f"{label}_vals"] = []
        rows.append(row)

    return rows, metrics


def print_table(dataset, rows, metrics):
    w_model = 10
    w_col   = 18

    border = "=" * (w_model + len(metrics) * w_col + 4)
    print(f"\n{border}")
    print(f"  {dataset.upper()}  —  Mean ± Std (%)  across seeds {SEEDS}")
    print(border)

    # Header
    hdr = f"  {'Model':<{w_model}}"
    for label, _ in metrics:
        hdr += f"  {label:^{w_col}}"
    print(hdr)
    print("-" * len(border))

    # Rows
    best_per_metric = {}
    for label, mk in metrics:
        vals = [r[f"{label}_mean"] for r in rows if r[f"{label}_mean"] is not None]
        best_per_metric[label] = max(vals) if vals else None

    for r in rows:
        line = f"  {r['model']:<{w_model}}"
        for label, mk in metrics:
            mean = r[f"{label}_mean"]
            std  = r[f"{label}_std"]
            if mean is None:
                cell = "    —    "
            else:
                cell = f"{mean:5.2f}±{std:4.2f}"
                if best_per_metric[label] is not None and abs(mean - best_per_metric[label]) < 0.005:
                    cell = f"**{cell}**"
            line += f"  {cell:^{w_col}}"

        miss = r["missing_seeds"]
        if miss:
            line += f"  ← {miss} seed(s) missing"
        print(line)

    print(border)
    print(f"  ** = best in column   |  seeds = {SEEDS}")
    print(border)


def save_json(dataset, rows, metrics):
    out = {}
    for r in rows:
        model = r["model"]
        out[model] = {}
        for label, mk in metrics:
            out[model][label] = {
                "mean": round(r[f"{label}_mean"], 4) if r[f"{label}_mean"] is not None else None,
                "std":  round(r[f"{label}_std"],  4) if r[f"{label}_std"]  is not None else None,
                "per_seed": [round(v, 4) for v in r[f"{label}_vals"]],
            }
    path = os.path.join(OUT, f"{dataset}_multiseed_table.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["iemocap", "mosei", "meld", "all"],
                   default="all")
    args = p.parse_args()

    datasets = ["iemocap", "mosei", "meld"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        rows, metrics = compute_table(ds)
        print_table(ds, rows, metrics)
        save_json(ds, rows, metrics)


if __name__ == "__main__":
    main()
