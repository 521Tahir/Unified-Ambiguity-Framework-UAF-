#!/usr/bin/env python3
"""
Collect sensitivity sweep results into a unified JSON and print the paper table.

Run:
  python3 scripts/sensitivity/collect_sensitivity_results.py
"""
import os, json, glob

ROOT    = ""
RES_DIR = f"{ROOT}/experiments/results/mosei/sensitivity"
OUT     = f"{RES_DIR}/sensitivity_summary.json"

# Expected experiments and their role in the paper table
EXPERIMENTS = [
    # (run_name, component, lambda_value, lambda_name)
    # MAD sweep (λ1), λ2=0.0, λ3=0.0
    ("mad_00",  "MAD",  0.0,  "λ1"),
    ("mad_01",  "MAD",  0.1,  "λ1"),
    ("mad_03",  "MAD",  0.3,  "λ1"),
    ("mad_05",  "MAD",  0.5,  "λ1"),
    ("mad_07",  "MAD",  0.7,  "λ1"),
    ("mad_10",  "MAD",  1.0,  "λ1"),
    # OT sweep (λ2), λ1=0.5, λ3=0.0
    ("mad_05",  "OT",   0.0,  "λ2"),   # shared baseline
    ("ot_001",  "OT",   0.01, "λ2"),
    ("ot_003",  "OT",   0.03, "λ2"),
    ("ot_005",  "OT",   0.05, "λ2"),
    ("ot_01",   "OT",   0.10, "λ2"),
    # Hyperbolic sweep (λ3), λ1=0.5, λ2=0.0
    ("mad_05",  "Hyp",  0.0,  "λ3"),   # shared baseline
    ("hyp_001", "Hyp",  0.01, "λ3"),
    ("hyp_005", "Hyp",  0.05, "λ3"),
    ("hyp_01",  "Hyp",  0.10, "λ3"),
    ("hyp_02",  "Hyp",  0.20, "λ3"),
]

COMP_LABEL = {
    "MAD": "MAD (λ₁)",
    "OT":  "OT (λ₂)",
    "Hyp": "Radial reg. (λ₃)",
}

results = {}
missing = []

for name, comp, val, lname in EXPERIMENTS:
    if name in results:
        continue   # already loaded (shared baseline)
    path = os.path.join(RES_DIR, name, "test_results.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        results[name] = {
            "wacc":        d.get("wacc", d.get("subset_accuracy", 0.0)),
            "weighted_f1": d.get("weighted_f1", 0.0),
            "macro_f1":    d.get("macro_f1", 0.0),
        }
        print(f"  [OK]     {name:12s}  WAcc={results[name]['wacc']*100:.2f}%  WF1={results[name]['weighted_f1']*100:.2f}%")
    else:
        missing.append(name)
        print(f"  [MISS]   {name:12s}  → {path}")

# ── Print paper table ─────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  CHADO — CMU-MOSEI Hyperparameter Sensitivity")
print(f"  {'Component':<28} {'λ':>5}  {'Acc(%)':>8}  {'WF1(%)':>8}")
print("  " + "-" * 55)

prev_comp = None
for name, comp, val, lname in EXPERIMENTS:
    if comp != prev_comp:
        if prev_comp is not None:
            print()
        prev_comp = comp

    r = results.get(name)
    if r:
        acc_str = f"{r['wacc']*100:8.2f}"
        wf1_str = f"{r['weighted_f1']*100:8.2f}"
    else:
        acc_str = "    N/A "
        wf1_str = "    N/A "

    label = COMP_LABEL[comp]
    print(f"  {label:<28} {val:>5.2f}  {acc_str}  {wf1_str}")

print("=" * 62)
if missing:
    print(f"\n  Missing results: {missing}")

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "experiments": [
        {
            "name": name, "component": comp, "lambda_val": val, "lambda_name": lname,
            "wacc":        results[name]["wacc"]        if name in results else None,
            "weighted_f1": results[name]["weighted_f1"] if name in results else None,
            "macro_f1":    results[name]["macro_f1"]    if name in results else None,
        }
        for name, comp, val, lname in EXPERIMENTS
    ]
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved → {OUT}")
