#!/usr/bin/env python3
"""
MAD vs Entropy vs Margin — head-to-head comparison.

3 proxy types × 3 datasets × 3 seeds = 27 jobs (multi-seed, full training).
Skip any job where test_results.json already exists.
Uses rolling GPU polling: launches next job as soon as a GPU becomes free.

Reviewer response: "It's just entropy" objection killed by evidence.
Results → experiments/results/{ds}/proxy_comparison/{proxy}/seed_{seed}/
"""
import os, sys, subprocess, time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]   # never touch 0,2,3 (mmansoor)
MIN_FREE_MB  = 38_000                    # 38 GB free required (roberta-large needs ~25 GB)
POLL_SEC     = 60

LOG_DIR = ROOT / "logs/proxy_comparison"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Job matrix ─────────────────────────────────────────────────────────────────
PROXIES = ["mad", "entropy", "margin"]
SEEDS   = [42, 123, 456]

DATASETS = {
    "iemocap": {
        "script": "scripts/train/train_iemocap.py",
        "config": "configs/iemocap/chado_iemocap_v2.yaml",
        "has_seed": True,
        "nproc":  1,   # single GPU per job (rolling allocation)
    },
    "mosei": {
        "script": "scripts/train/train_mosei.py",
        "config": "configs/mosei/chado_mosei_best.yaml",
        "has_seed": False,
        "nproc":  1,
    },
    "meld": {
        "script": "scripts/train/train_meld.py",
        "config": "configs/meld/chado_meld_final.yaml",
        "has_seed": True,
        "nproc":  1,
    },
}

def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def out_dir(ds, proxy, seed):
    return ROOT / f"experiments/results/{ds}/proxy_comparison/{proxy}/seed_{seed}"

def result_exists(ds, proxy, seed):
    return (out_dir(ds, proxy, seed) / "test_results.json").exists()

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

def make_env(gpu, **extra):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    e["TOKENIZERS_PARALLELISM"]            = "false"
    e["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    e["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    e.update(extra)
    return e

def launch_job(ds, proxy, seed, gpu, ds_info):
    odir = out_dir(ds, proxy, seed)
    odir.mkdir(parents=True, exist_ok=True)
    label = f"{ds}/{proxy}/s{seed}"
    log_path = LOG_DIR / f"{ds}_{proxy}_s{seed}.log"

    cmd = [
        "python3", "-u", str(ROOT / ds_info["script"]),
        "--config", str(ROOT / ds_info["config"]),
        "--out-dir", str(odir),
        "--proxy", proxy,
    ]
    if ds_info["has_seed"]:
        cmd += ["--seed", str(seed)]

    env = make_env(gpu)
    log(f"  LAUNCH  GPU={gpu}  {label}  → {log_path.name}")
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh,
                            preexec_fn=os.setpgrp)
    return proc, label

def build_jobs():
    jobs = []
    for ds, info in DATASETS.items():
        for proxy in PROXIES:
            for seed in SEEDS:
                if result_exists(ds, proxy, seed):
                    log(f"  SKIP  {ds}/{proxy}/s{seed}  (exists)")
                    continue
                jobs.append((ds, proxy, seed, info))
    return jobs

def compute_and_save_summary():
    """After all training: aggregate results and compute summary table."""
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for proxy in PROXIES:
            metrics_list = []
            for seed in SEEDS:
                p = out_dir(ds, proxy, seed) / "test_results.json"
                if p.exists():
                    with open(p) as f:
                        m = json.load(f)
                    metrics_list.append(m)
            if not metrics_list:
                continue
            # Average across seeds
            keys = ["macro_f1", "weighted_f1", "accuracy"]
            avg = {}
            for k in keys:
                vals = [m.get(k, m.get(k.replace("_", " "), 0)) for m in metrics_list]
                vals = [v for v in vals if v]
                if vals:
                    avg[k] = {"mean": sum(vals)/len(vals),
                              "std": (sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5,
                              "n": len(vals), "vals": vals}
            summary[ds][proxy] = avg

    out = ROOT / "experiments/results/proxy_comparison_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 72)
    log("PROXY COMPARISON SUMMARY")
    log("=" * 72)
    for ds in DATASETS:
        log(f"\n  {ds.upper()}")
        for proxy in PROXIES:
            if proxy not in summary.get(ds, {}):
                continue
            d = summary[ds][proxy]
            f1 = d.get("macro_f1", {})
            if f1:
                vals_str = ", ".join(f"{v:.2f}" for v in f1.get("vals", []))
                log(f"    {proxy:<8}  macro_f1={f1['mean']:.2f}±{f1['std']:.2f}  [{vals_str}]")
    log(f"\n  Full results → {out}")

def main():
    log("=" * 72)
    log("Proxy Comparison: MAD vs Entropy vs Margin")
    log(f"  {len(PROXIES)} proxies × {len(list(DATASETS))} datasets × {len(SEEDS)} seeds")
    log("=" * 72)

    queue = build_jobs()
    log(f"\n  {len(queue)} jobs to run\n")

    if not queue:
        log("Nothing to run — computing summary from existing results.")
        compute_and_save_summary()
        return

    # ── Rolling GPU scheduler ────────────────────────────────────────────────
    running: dict[int, tuple] = {}   # gpu → (proc, label, t0)

    while queue or running:
        # Collect finished
        for gpu in list(running):
            proc, label, t0 = running[gpu]
            ret = proc.poll()
            if ret is not None:
                ela = (time.time() - t0) / 60
                if ret == 0:
                    log(f"  DONE  GPU={gpu}  {label}  {ela:.1f}min")
                else:
                    log(f"  FAIL  GPU={gpu}  {label}  exit={ret}  {ela:.1f}min")
                del running[gpu]

        # Fill free GPUs
        while queue:
            gpu = pick_free_gpu(set(running.keys()))
            if gpu is None:
                break
            ds, proxy, seed, info = queue.pop(0)
            proc, label = launch_job(ds, proxy, seed, gpu, info)
            running[gpu] = (proc, label, time.time())
            time.sleep(5)   # stagger launches

        if running or queue:
            status = " | ".join(f"GPU{g}:{lbl}" for g, (_, lbl, _) in running.items())
            q_cnt = len(queue)
            log(f"  running={len(running)} queue={q_cnt}  [{status}]")
            time.sleep(POLL_SEC)

    log("\n=== All training jobs complete ===")
    compute_and_save_summary()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
