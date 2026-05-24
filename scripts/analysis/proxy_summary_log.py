"""
Appends a timestamped accuracy/F1 snapshot for all proxy runs to
experiments/results/proxy_comparison_summary.log

Run once to snapshot, or with --loop N to repeat every N seconds.
Usage:
    python3 scripts/analysis/proxy_summary_log.py              # one snapshot
    python3 scripts/analysis/proxy_summary_log.py --loop 900   # every 15 min
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

RUNS = {
    "IEMOCAP": {
        "mad":     "experiments/results/iemocap/proxy_comparison/mad/seed_42",
        "entropy": "experiments/results/iemocap/proxy_comparison/entropy/seed_42",
        "margin":  "experiments/results/iemocap/proxy_comparison/margin/seed_42",
    },
    "MELD": {
        "mad":     "experiments/results/meld/proxy_comparison/mad/seed_42",
        "entropy": "experiments/results/meld/proxy_comparison/entropy/seed_42",
        "margin":  "experiments/results/meld/proxy_comparison/margin/seed_42",
    },
    "MOSEI": {
        "mad":     "experiments/results/mosei/proxy_comparison/mad/seed_42",
        "entropy": "experiments/results/mosei/proxy_comparison/entropy/seed_42",
        "margin":  "experiments/results/mosei/proxy_comparison/margin/seed_42",
    },
}

SUMMARY_LOG = "experiments/results/proxy_comparison_summary.log"


def snapshot():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"\n{'='*72}", f"  Proxy Comparison Snapshot — {now}", f"{'='*72}"]

    for ds, proxies in RUNS.items():
        lines.append(f"\n  [{ds}]")
        lines.append(f"  {'Proxy':<10} {'Status':<12} {'Epoch':>7} {'Acc':>7} {'Macro-F1':>9} {'W-F1':>8} {'Best-F1':>9}")
        lines.append(f"  {'-'*70}")
        for proxy, path in proxies.items():
            test_f = os.path.join(path, "test_results.json")
            hist_f = os.path.join(path, "history.json")
            if os.path.exists(test_f):
                d   = json.load(open(test_f))
                acc = float(d.get("accuracy", d.get("wacc", d.get("subset_accuracy", 0))) or 0)
                mf1 = float(d.get("macro_f1", 0) or 0)
                wf1 = float(d.get("weighted_f1", 0) or 0)
                lines.append(f"  {proxy:<10} {'COMPLETE':<12} {'—':>7} {acc:>7.4f} {mf1:>9.4f} {wf1:>8.4f} {'—':>9}")
            elif os.path.exists(hist_f):
                h   = json.load(open(hist_f))
                ep  = h.get("epoch", [])
                acc = h.get("val_acc", [])
                mf1 = h.get("val_macro_f1", [])
                wf1 = h.get("val_weighted_f1", [])
                if ep:
                    best = max(mf1) if mf1 else 0
                    lines.append(
                        f"  {proxy:<10} {'running':<12} {ep[-1]:>4}/60 "
                        f"{acc[-1]:>7.4f} {mf1[-1]:>9.4f} {wf1[-1]:>8.4f} {best:>9.4f}"
                    )
                else:
                    lines.append(f"  {proxy:<10} {'starting':<12}")
            else:
                lines.append(f"  {proxy:<10} {'no data':<12}")

    block = "\n".join(lines) + "\n"
    print(block)
    os.makedirs(os.path.dirname(SUMMARY_LOG), exist_ok=True)
    with open(SUMMARY_LOG, "a") as f:
        f.write(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0, help="repeat every N seconds (0 = once)")
    args = parser.parse_args()

    snapshot()
    if args.loop > 0:
        while True:
            time.sleep(args.loop)
            snapshot()


if __name__ == "__main__":
    main()
