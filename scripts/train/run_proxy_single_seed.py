#!/usr/bin/env python3
"""
Single-seed proxy comparison: MAD vs entropy vs margin × 3 datasets = 9 jobs.
Seed: 42 for all. Results go to proxy_comparison/{proxy}/seed_42/.
"""
import os, sys, subprocess, time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]   # never 0, 2, 3 (mmansoor)
MIN_FREE_MB  = 25_000
POLL_SEC     = 30
SEED         = 42

for d in ("logs/proxy_comparison",):
    (ROOT / d).mkdir(parents=True, exist_ok=True)

DATASETS = {
    "iemocap": {"script": "scripts/train/train_iemocap.py",
                "config": "configs/iemocap/chado_iemocap_v2.yaml", "has_seed": True},
    "mosei":   {"script": "scripts/train/train_mosei.py",
                "config": "configs/mosei/chado_mosei_best.yaml",   "has_seed": False},
    "meld":    {"script": "scripts/train/train_meld.py",
                "config": "configs/meld/chado_meld_final.yaml",    "has_seed": True},
}
PROXIES = ["mad", "entropy", "margin"]


def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def gpu_free_mb(g):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={g}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip())
    except Exception:
        return 0

def pick_free_gpu(used):
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


def build_jobs():
    jobs = []
    for proxy in PROXIES:
        for ds, dinfo in DATASETS.items():
            out = ROOT / f"experiments/results/{ds}/proxy_comparison/{proxy}/seed_{SEED}"
            if (out / "test_results.json").exists():
                log(f"  SKIP  {ds}/{proxy}/s{SEED}  (exists)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            log_path = ROOT / f"logs/proxy_comparison/{ds}_{proxy}_s{SEED}.log"
            cmd = ["python3", "-u", str(ROOT / dinfo["script"]),
                   "--config", str(ROOT / dinfo["config"]),
                   "--out-dir", str(out),
                   "--proxy", proxy]
            if dinfo["has_seed"]:
                cmd += ["--seed", str(SEED)]
            jobs.append((f"{ds}/{proxy}/s{SEED}", log_path, cmd))
    return jobs


def summarise():
    log("\n" + "=" * 64)
    log("PROXY COMPARISON — seed 42 — Macro-F1")
    log("=" * 64)
    header = f"  {'proxy':<10}" + "".join(f"  {ds.upper():<14}" for ds in DATASETS)
    log(header)
    for proxy in PROXIES:
        row = f"  {proxy:<10}"
        for ds in DATASETS:
            p = ROOT / f"experiments/results/{ds}/proxy_comparison/{proxy}/seed_{SEED}/test_results.json"
            if p.exists():
                m = json.load(open(p))
                row += f"  {m.get('macro_f1', 0):.4f}      "
            else:
                row += "  ---           "
        log(row)
    out = ROOT / "experiments/results/proxy_single_seed_summary.json"
    results = {}
    for proxy in PROXIES:
        results[proxy] = {}
        for ds in DATASETS:
            p = ROOT / f"experiments/results/{ds}/proxy_comparison/{proxy}/seed_{SEED}/test_results.json"
            if p.exists():
                results[proxy][ds] = json.load(open(p))
    json.dump(results, open(out, "w"), indent=2)
    log(f"\n  JSON → {out}")


def main():
    log("=" * 64)
    log("Proxy Single-Seed Comparison  (seed=42)")
    log("  MAD vs Entropy vs Margin × IEMOCAP / MOSEI / MELD")
    log(f"  GPUs: {ALLOWED_GPUS}  MIN_FREE={MIN_FREE_MB//1000}GB")
    log("=" * 64)

    queue = build_jobs()
    log(f"\n  {len(queue)} jobs to run\n")

    if not queue:
        summarise()
        return

    running = {}   # gpu → (proc, label, t0)

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
            label, log_path, cmd = queue.pop(0)
            log(f"  LAUNCH  GPU={gpu}  {label}  → {log_path.name}")
            fh = open(log_path, "w")
            proc = subprocess.Popen(cmd, env=make_env(gpu), stdout=fh, stderr=fh,
                                    preexec_fn=os.setpgrp)
            running[gpu] = (proc, label, time.time())
            time.sleep(5)

        if running or queue:
            active = " | ".join(f"GPU{g}:{lbl}" for g, (_, lbl, _) in running.items())
            log(f"  running={len(running)} queue={len(queue)}  [{active}]")
            time.sleep(POLL_SEC)

    log("\n=== All 9 jobs complete ===")
    summarise()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
