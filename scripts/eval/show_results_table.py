#!/usr/bin/env python3
"""
Display completed multi-seed results for IEMOCAP and MOSEI in table format.
Run at any time to see current results (even partial).
"""
import json, sys
from pathlib import Path
from typing import Optional

PROJ = Path("/home/tahirahmad/Project_Code/CHADO_EMNLP")
RES  = PROJ / "experiments/results"

SEEDS = [42, 3407, 456]

# Model display names and order
MODELS = [
    ("argot_ms",   "AR-GOT (Ours)"),
    ("bpmult_ms",  "BP-MulT"),
    ("ctnet_ms",   "CTNet"),
    ("mult_ms",    "MulT"),
    ("mmdfn_ms",   "MM-DFN"),
    ("lflstm_ms",  "LF-LSTM"),
    ("aerllm_ms",  "AER-LLM"),
    ("emoclip_ms", "EmoCLIP"),
    ("ovmer_ms",   "OV-MER"),
]


def load_result(dataset: str, model_dir: str, seed: int) -> Optional[dict]:
    p = RES / dataset / model_dir / f"seed_{seed}" / "test_results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def get_metric(r: dict, dataset: str):
    """Return (accuracy, weighted_f1, macro_f1) or None if missing."""
    if r is None:
        return None
    # MOSEI uses 'wacc' (weighted accuracy) as main metric
    if dataset == "mosei":
        acc = r.get("wacc") or r.get("accuracy") or r.get("subset_accuracy")
    else:
        acc = r.get("accuracy") or r.get("test_accuracy") or r.get("acc")
    wf1 = r.get("weighted_f1") or r.get("test_weighted_f1") or r.get("w_f1")
    mf1 = r.get("macro_f1") or r.get("test_macro_f1") or r.get("m_f1")
    return acc, wf1, mf1


def mean_std(vals):
    if not vals:
        return None, None
    import statistics
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt(v, pct=True):
    if v is None:
        return "  ---  "
    if pct:
        return f"{v*100:6.2f}"
    return f"{v:6.4f}"


def fmt_mean_std(mean, std, pct=True):
    if mean is None:
        return "   ---   "
    if pct:
        return f"{mean*100:5.1f}±{std*100:4.1f}"
    return f"{mean:5.3f}±{std:5.3f}"


def print_dataset_table(dataset: str, title: str):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    # Count completed
    completed = sum(
        1 for mdir, _ in MODELS
        for s in SEEDS
        if (RES / dataset / mdir / f"seed_{s}" / "test_results.json").exists()
    )
    total = len(MODELS) * len(SEEDS)
    print(f"  Completed: {completed}/{total} runs")
    print(f"{'='*90}")

    # Header
    header = f"  {'Model':<18}  {'S42':>8}  {'S3407':>8}  {'S456':>8}  {'Mean±Std':>12}  {'WF1 Mean±Std':>14}"
    print(header)
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*14}")

    best_mean_wf1 = 0
    best_model = ""

    for mdir, mname in MODELS:
        results = [load_result(dataset, mdir, s) for s in SEEDS]
        metrics = [get_metric(r, dataset) for r in results]

        # Accuracy per seed
        accs = [m[0] for m in metrics if m and m[0] is not None]
        wf1s = [m[1] for m in metrics if m and m[1] is not None]

        per_seed_acc = []
        for m in metrics:
            if m and m[0] is not None:
                per_seed_acc.append(f"{m[0]*100:6.2f}")
            else:
                per_seed_acc.append("  ---  ")

        acc_mean, acc_std = mean_std(accs)
        wf1_mean, wf1_std = mean_std(wf1s)

        mark = " *" if mname.startswith("AR-GOT") else "  "
        row = (
            f"{mark}{mname:<18}"
            f"  {per_seed_acc[0]:>8}"
            f"  {per_seed_acc[1]:>8}"
            f"  {per_seed_acc[2]:>8}"
            f"  {fmt_mean_std(acc_mean, acc_std):>12}"
            f"  {fmt_mean_std(wf1_mean, wf1_std):>14}"
        )
        print(f"  {row}")

        if wf1_mean and wf1_mean > best_mean_wf1:
            best_mean_wf1 = wf1_mean
            best_model = mname

    print(f"{'='*90}")
    if best_model:
        print(f"  Best Weighted-F1: {best_model}  ({best_mean_wf1*100:.1f}%)")


def print_gpu_status():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        print("\n--- GPU Status ---")
        for line in out.strip().splitlines():
            idx, util, mem = [x.strip() for x in line.split(",")]
            bar = "█" * (int(util) // 10) + "░" * (10 - int(util) // 10)
            print(f"  GPU {idx}: [{bar}] {util:>3}%  {int(mem):>5} MiB used")
    except Exception:
        pass


def print_running_jobs():
    import subprocess
    try:
        r = subprocess.run(["ps", "auxww"], capture_output=True, text=True)
        running = []
        for line in r.stdout.splitlines():
            if "torchrun" not in line and "train_" not in line:
                continue
            if "grep" in line or "gpu_scheduler" in line:
                continue
            if "CHADO_EMNLP" not in line:
                continue
            out_dir = ""
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "--out-dir" and i + 1 < len(parts):
                    out_dir = parts[i + 1]
                    break
            if out_dir:
                running.append(out_dir.replace(str(RES) + "/", ""))
        running = sorted(set(running))
        if running:
            print(f"\n--- Currently Running ({len(running)}) ---")
            for r in running:
                print(f"  {r}")
    except Exception:
        pass


def main():
    from datetime import datetime
    print(f"\n{'#'*90}")
    print(f"#  RESULTS TABLE  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*90}")

    print_gpu_status()
    print_running_jobs()

    print_dataset_table("iemocap", "IEMOCAP — Accuracy (%) per seed   |   Mean±Std")
    print_dataset_table("mosei",   "CMU-MOSEI — Accuracy (%) per seed  |   Mean±Std")

    # Check scheduler log for progress
    sched_log = PROJ / "logs/priority_scheduler.log"
    if sched_log.exists():
        lines = sched_log.read_text().splitlines()
        recent = [l for l in lines if "[DONE" in l or "[FAIL" in l or "[START" in l]
        if recent:
            print(f"\n--- Scheduler Activity (last 10) ---")
            for l in recent[-10:]:
                print(f"  {l.strip()}")

    print()


if __name__ == "__main__":
    main()
