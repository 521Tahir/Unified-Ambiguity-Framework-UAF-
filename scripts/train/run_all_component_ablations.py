#!/usr/bin/env python3
"""
Component Ablation Launcher — Full 4-variant × 3-dataset run.
============================================================
Ablations:
  wo_mad    — CHADO without MAD ambiguity proxy (use_mad=False)
  wo_ot     — CHADO without Optimal Transport   (use_ot=False)
  wo_radial — CHADO without Hyperbolic space    (use_hyperbolic=False)
  wo_causal — CHADO without 5 latent factors    (use_causal=False)

12 jobs launched in parallel across allowed GPUs.
Results → experiments/results/{dataset}/component_ablations/{ablation}/seed_42/
Summary → experiments/results/component_ablation_table.json
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]
MIN_FREE_MiB = 30_000
SEED         = 42

TRAIN_SCRIPTS = {
    "iemocap": ROOT / "scripts/train/train_iemocap.py",
    "mosei":   ROOT / "scripts/train/train_mosei.py",
    "meld":    ROOT / "scripts/train/train_meld.py",
}

ABLATIONS = ["wo_mad", "wo_ot", "wo_radial", "wo_causal"]
DATASETS  = ["iemocap", "mosei", "meld"]

OUT_BASE  = ROOT / "experiments/results"
LOG_DIR   = ROOT / "logs/component_ablations"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def ts():
    return datetime.now().strftime("%H:%M:%S")


def free_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        result = []
        for line in out.strip().splitlines():
            idx, free = line.split(",")
            if int(idx) in ALLOWED_GPUS and int(free) >= MIN_FREE_MiB:
                result.append(int(idx))
        return result
    except Exception:
        return []


def out_dir(ds: str, abl: str) -> Path:
    return OUT_BASE / ds / "component_ablations" / abl / f"seed_{SEED}"


def result_path(ds: str, abl: str) -> Path:
    return out_dir(ds, abl) / "test_results.json"


def parse_result(ds: str, abl: str):
    p = result_path(ds, abl)
    if not p.exists():
        return None
    d = json.load(open(p))
    acc = d.get("accuracy", d.get("acc", d.get("wacc")))
    f1  = d.get("macro_f1", d.get("f1"))
    wf1 = d.get("weighted_f1", d.get("w_f1"))
    return {"acc": acc, "macro_f1": f1, "weighted_f1": wf1}


def launch_job(ds: str, abl: str, gpu: int):
    cfg   = ROOT / "configs" / ds / f"comp_abl_{abl}.yaml"
    odir  = out_dir(ds, abl)
    odir.mkdir(parents=True, exist_ok=True)
    log   = LOG_DIR / f"{ds}_{abl}.log"
    cmd   = [
        "python3", "-u", str(TRAIN_SCRIPTS[ds]),
        "--config", str(cfg),
        "--seed", str(SEED),
        "--out-dir", str(odir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]        = str(gpu)
    env["TOKENIZERS_PARALLELISM"]      = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    print(f"  [{ts()}] LAUNCH  GPU={gpu}  {ds}/{abl}  → {log.name}", flush=True)
    fh   = open(log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc, log


def get_epoch(ds: str, abl: str) -> str:
    log = LOG_DIR / f"{ds}_{abl}.log"
    if not log.exists():
        return ""
    lines = open(log).readlines()
    for line in reversed(lines):
        if "Epoch" in line and "/" in line:
            try:
                ep_part = line.split("Epoch")[1].strip().split()[0]
                return f"ep={ep_part}"
            except Exception:
                pass
    return ""


def build_summary():
    table = {}
    print("\n" + "=" * 80)
    print("COMPONENT ABLATION RESULTS  (seed=42, CHADO full training)")
    print("=" * 80)
    header = f"  {'Ablation':<14} {'Dataset':<10} {'Acc':>8} {'Macro-F1':>10} {'Wt-F1':>10}"
    print(header)
    print("  " + "-" * 56)

    for abl in ABLATIONS:
        for ds in DATASETS:
            r = parse_result(ds, abl)
            acc = f"{r['acc']*100:.2f}%" if r and r.get("acc") else "—"
            mf1 = f"{r['macro_f1']*100:.2f}%" if r and r.get("macro_f1") else "—"
            wf1 = f"{r['weighted_f1']*100:.2f}%" if r and r.get("weighted_f1") else "—"
            print(f"  {abl:<14} {ds.upper():<10} {acc:>8} {mf1:>10} {wf1:>10}")
            if abl not in table:
                table[abl] = {}
            table[abl][ds] = r

    out = OUT_BASE / "component_ablation_table.json"
    json.dump(table, open(out, "w"), indent=2)
    print(f"\n  Saved → {out}")


def main():
    # Clear existing results and rerun everything fresh
    for abl in ABLATIONS:
        for ds in DATASETS:
            p = result_path(ds, abl)
            if p.exists():
                p.unlink()
                print(f"  [CLEAR] {ds}/{abl}")

    jobs_todo = [(ds, abl) for abl in ABLATIONS for ds in DATASETS]
    print(f"\n  [{ts()}] Launching {len(jobs_todo)} ablation jobs")
    print(f"  Variants: {ABLATIONS}")
    print(f"  Datasets: {DATASETS}\n")

    running = {}   # (ds, abl) → (proc, log, gpu, t_start)
    done, failed = [], []

    def collect_finished():
        for key in list(running):
            proc, logp, gpu, ts_ = running[key]
            ret = proc.poll()
            if ret is None:
                continue
            elapsed = (time.time() - ts_) / 60
            ds, abl = key
            if ret == 0:
                r = parse_result(ds, abl)
                acc_str = f"{r['acc']*100:.2f}%" if r and r.get("acc") else "?"
                print(f"  [{ts()}] DONE   GPU={gpu}  {ds}/{abl}  "
                      f"{elapsed:.1f}min  acc={acc_str}", flush=True)
                done.append(key)
            else:
                print(f"  [{ts()}] FAIL   GPU={gpu}  {ds}/{abl}  exit={ret}", flush=True)
                failed.append(key)
            del running[key]

    while jobs_todo or running:
        collect_finished()
        gpus = free_gpus()
        # fill up to 7 parallel jobs
        while jobs_todo and len(running) < len(ALLOWED_GPUS) and gpus:
            ds, abl = jobs_todo.pop(0)
            gpu     = gpus.pop(0)
            proc, logp = launch_job(ds, abl, gpu)
            running[(ds, abl)] = (proc, logp, gpu, time.time())

        if running:
            # status line
            status_parts = [f"{ds}/{abl}({get_epoch(ds,abl)})" for (ds, abl) in running]
            print(f"  [{ts()}] running={len(running)} queue={len(jobs_todo)} "
                  f"done={len(done)} fail={len(failed)}", flush=True)
            for (ds, abl), (_, _, gpu, ts_) in sorted(running.items()):
                ela = (time.time() - ts_) / 60
                print(f"    GPU={gpu}  {ds}/{abl}  {get_epoch(ds,abl)}  {ela:.1f}min")
            time.sleep(60)

    print(f"\n  [{ts()}] All done.  Done={len(done)}  Failed={len(failed)}")
    if failed:
        print("  FAILED:", failed)
    build_summary()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.chdir(ROOT)
    main()
