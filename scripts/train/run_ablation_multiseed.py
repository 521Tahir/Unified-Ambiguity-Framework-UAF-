#!/usr/bin/env python3
"""
Table 3 Fix — Ablation multi-seed runner.

4 ablations × 3 datasets × 3 seeds = 36 jobs.
Ablations: wo_mad, wo_ot, wo_radial (wo_hyperbolic), wo_causal
Uses same rolling GPU scheduler as proxy comparison.
Results → experiments/results/{ds}/ablation_multiseed/{ablation}/seed_{seed}/
"""
import os, sys, subprocess, time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]
MIN_FREE_MB  = 38_000
POLL_SEC     = 60

LOG_DIR = ROOT / "logs/ablation_multiseed"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]

ABLATIONS = {
    "wo_mad":      {"iemocap": "comp_abl_wo_mad.yaml",      "mosei": "comp_abl_wo_mad.yaml",      "meld": "comp_abl_wo_mad.yaml"},
    "wo_ot":       {"iemocap": "comp_abl_wo_ot.yaml",       "mosei": "comp_abl_wo_ot.yaml",       "meld": "comp_abl_wo_ot.yaml"},
    "wo_radial":   {"iemocap": "comp_abl_wo_radial.yaml",   "mosei": "comp_abl_wo_radial.yaml",   "meld": "comp_abl_wo_radial.yaml"},
    "wo_causal":   {"iemocap": "comp_abl_wo_causal.yaml",   "mosei": "comp_abl_wo_causal.yaml",   "meld": "comp_abl_wo_causal.yaml"},
}

DATASETS = {
    "iemocap": {"script": "scripts/train/train_iemocap.py", "config_dir": "configs/iemocap", "has_seed": True},
    "mosei":   {"script": "scripts/train/train_mosei.py",   "config_dir": "configs/mosei",   "has_seed": False},
    "meld":    {"script": "scripts/train/train_meld.py",    "config_dir": "configs/meld",    "has_seed": True},
}

def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def out_dir(ds, abl, seed):
    return ROOT / f"experiments/results/{ds}/ablation_multiseed/{abl}/seed_{seed}"

def result_exists(ds, abl, seed):
    return (out_dir(ds, abl, seed) / "test_results.json").exists()

def gpu_free_mb(g):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={g}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip())
    except Exception:
        return 0

def pick_free_gpu(used: set):
    for g in ALLOWED_GPUS:
        if g in used:
            continue
        if gpu_free_mb(g) >= MIN_FREE_MB:
            return g
    return None

def make_env(gpu):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    e["TOKENIZERS_PARALLELISM"]            = "false"
    e["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    e["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    return e

def launch_job(ds, abl, seed, gpu, ds_info, cfg_name):
    odir = out_dir(ds, abl, seed)
    odir.mkdir(parents=True, exist_ok=True)
    label    = f"{ds}/{abl}/s{seed}"
    log_path = LOG_DIR / f"{ds}_{abl}_s{seed}.log"
    cfg_path = ROOT / ds_info["config_dir"] / cfg_name

    cmd = [
        "python3", "-u", str(ROOT / ds_info["script"]),
        "--config", str(cfg_path),
        "--out-dir", str(odir),
    ]
    if ds_info["has_seed"]:
        cmd += ["--seed", str(seed)]

    log(f"  LAUNCH  GPU={gpu}  {label}  → {log_path.name}")
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=make_env(gpu), stdout=fh, stderr=fh,
                            preexec_fn=os.setpgrp)
    return proc, label

def compute_summary():
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for abl in ABLATIONS:
            vals = []
            for seed in SEEDS:
                p = out_dir(ds, abl, seed) / "test_results.json"
                if p.exists():
                    with open(p) as f:
                        m = json.load(f)
                    vals.append(m.get("macro_f1", 0))
            if vals:
                mean = sum(vals) / len(vals)
                std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                summary[ds][abl] = {"mean": mean, "std": std, "n": len(vals), "vals": vals}

    out = ROOT / "experiments/results/ablation_multiseed_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 72)
    log("TABLE 3 ABLATION SUMMARY (Macro-F1)")
    log("=" * 72)
    hdr = f"  {'Ablation':<16}" + "".join(f"  {ds.upper():<20}" for ds in DATASETS)
    log(hdr)
    for abl in ABLATIONS:
        row = f"  {abl:<16}"
        for ds in DATASETS:
            d = summary.get(ds, {}).get(abl, {})
            if d:
                row += f"  {d['mean']:.2f}±{d['std']:.2f} (n={d['n']})"
            else:
                row += "  ---"
        log(row)
    log(f"\n  Full results → {out}")

def main():
    log("=" * 72)
    log("Table 3 Fix — Ablation Multi-Seed")
    log(f"  {len(ABLATIONS)} ablations × {len(DATASETS)} datasets × {len(SEEDS)} seeds")
    log("=" * 72)

    queue = []
    for ds, ds_info in DATASETS.items():
        for abl, cfgs in ABLATIONS.items():
            cfg_name = cfgs[ds]
            cfg_path = ROOT / ds_info["config_dir"] / cfg_name
            if not cfg_path.exists():
                log(f"  SKIP  {ds}/{abl}  (config not found: {cfg_name})")
                continue
            for seed in SEEDS:
                if result_exists(ds, abl, seed):
                    log(f"  SKIP  {ds}/{abl}/s{seed}  (exists)")
                    continue
                queue.append((ds, abl, seed, ds_info, cfg_name))

    log(f"\n  {len(queue)} jobs to run\n")

    if not queue:
        compute_summary()
        return

    running: dict[int, tuple] = {}

    while queue or running:
        for gpu in list(running):
            proc, label, t0 = running[gpu]
            ret = proc.poll()
            if ret is not None:
                ela = (time.time() - t0) / 60
                status = "DONE" if ret == 0 else f"FAIL(exit={ret})"
                log(f"  {status}  GPU={gpu}  {label}  {ela:.1f}min")
                del running[gpu]

        while queue:
            gpu = pick_free_gpu(set(running.keys()))
            if gpu is None:
                break
            ds, abl, seed, ds_info, cfg_name = queue.pop(0)
            proc, label = launch_job(ds, abl, seed, gpu, ds_info, cfg_name)
            running[gpu] = (proc, label, time.time())
            time.sleep(5)

        if running or queue:
            status = " | ".join(f"GPU{g}:{lbl}" for g, (_, lbl, _) in running.items())
            log(f"  running={len(running)} queue={len(queue)}  [{status}]")
            time.sleep(POLL_SEC)

    log("\n=== All ablation jobs complete ===")
    compute_summary()

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
