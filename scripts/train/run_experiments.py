#!/usr/bin/env python3
"""
Unified launcher for all reviewer-response experiments.

Proxy comparison  : MAD vs entropy vs margin  (3 × 3 ds × 3 seeds = 27 jobs)
Ablation multiseed: wo_mad/ot/radial/causal   (4 × 3 ds × 3 seeds = 36 jobs)
Total: 63 jobs

Single GPU scheduler — no race condition between launchers.
MIN_FREE_MB=43000 avoids stealing from active DDP jobs.

Results:
  Proxy   → experiments/results/{ds}/proxy_comparison/{proxy}/seed_{seed}/
  Ablation → experiments/results/{ds}/ablation_multiseed/{abl}/seed_{seed}/
Logs:
  Proxy   → logs/proxy_comparison/{ds}_{proxy}_s{seed}.log
  Ablation → logs/ablation_multiseed/{ds}_{abl}_s{seed}.log
"""
import os, sys, subprocess, time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]   # never 0, 2, 3 (mmansoor)
MIN_FREE_MB  = 25_000                    # ~25 GB free → fits any single-GPU job on idle GPUs
POLL_SEC     = 60

for d in ("logs/proxy_comparison", "logs/ablation_multiseed"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# ── Dataset definitions ───────────────────────────────────────────────────────
DATASETS = {
    "iemocap": {"script": "scripts/train/train_iemocap.py",
                "config": "configs/iemocap/chado_iemocap_v2.yaml",  "has_seed": True},
    "mosei":   {"script": "scripts/train/train_mosei.py",
                "config": "configs/mosei/chado_mosei_best.yaml",    "has_seed": False},
    "meld":    {"script": "scripts/train/train_meld.py",
                "config": "configs/meld/chado_meld_final.yaml",     "has_seed": True},
}

PROXIES = ["mad", "entropy", "margin"]
SEEDS   = [42, 123, 456]

ABLATIONS = {
    "wo_mad":    {ds: "comp_abl_wo_mad.yaml"    for ds in DATASETS},
    "wo_ot":     {ds: "comp_abl_wo_ot.yaml"     for ds in DATASETS},
    "wo_radial": {ds: "comp_abl_wo_radial.yaml" for ds in DATASETS},
    "wo_causal": {ds: "comp_abl_wo_causal.yaml" for ds in DATASETS},
}

CONFIG_DIRS = {"iemocap": "configs/iemocap", "mosei": "configs/mosei", "meld": "configs/meld"}


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def gpu_free_mb(g: int) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={g}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip())
    except Exception:
        return 0


def pick_free_gpu(used: set) -> int | None:
    for g in ALLOWED_GPUS:
        if g in used:
            continue
        if gpu_free_mb(g) >= MIN_FREE_MB:
            return g
    return None


def make_env(gpu: int) -> dict:
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    e["TOKENIZERS_PARALLELISM"]            = "false"
    e["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    e["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    return e


def launch(label: str, log_path: Path, cmd: list, gpu: int):
    log(f"  LAUNCH  GPU={gpu}  {label}  → {log_path.name}")
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=make_env(gpu), stdout=fh, stderr=fh,
                            preexec_fn=os.setpgrp)
    return proc


# ── Build job queues ───────────────────────────────────────────────────────────

def build_proxy_jobs():
    jobs = []
    for ds, dinfo in DATASETS.items():
        for proxy in PROXIES:
            for seed in SEEDS:
                out = ROOT / f"experiments/results/{ds}/proxy_comparison/{proxy}/seed_{seed}"
                if (out / "test_results.json").exists():
                    log(f"  SKIP  proxy  {ds}/{proxy}/s{seed}")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                log_path = ROOT / f"logs/proxy_comparison/{ds}_{proxy}_s{seed}.log"
                cmd = ["python3", "-u", str(ROOT / dinfo["script"]),
                       "--config", str(ROOT / dinfo["config"]),
                       "--out-dir", str(out),
                       "--proxy", proxy]
                if dinfo["has_seed"]:
                    cmd += ["--seed", str(seed)]
                jobs.append((f"proxy/{ds}/{proxy}/s{seed}", log_path, cmd))
    return jobs


def build_ablation_jobs():
    jobs = []
    for ds, dinfo in DATASETS.items():
        for abl, cfgs in ABLATIONS.items():
            cfg_path = ROOT / CONFIG_DIRS[ds] / cfgs[ds]
            if not cfg_path.exists():
                log(f"  SKIP  ablation  {ds}/{abl}  (config missing)")
                continue
            for seed in SEEDS:
                out = ROOT / f"experiments/results/{ds}/ablation_multiseed/{abl}/seed_{seed}"
                if (out / "test_results.json").exists():
                    log(f"  SKIP  ablation  {ds}/{abl}/s{seed}")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                log_path = ROOT / f"logs/ablation_multiseed/{ds}_{abl}_s{seed}.log"
                cmd = ["python3", "-u", str(ROOT / dinfo["script"]),
                       "--config", str(cfg_path),
                       "--out-dir", str(out)]
                if dinfo["has_seed"]:
                    cmd += ["--seed", str(seed)]
                jobs.append((f"ablation/{ds}/{abl}/s{seed}", log_path, cmd))
    return jobs


# ── Summary computation ────────────────────────────────────────────────────────

def summarise():
    log("\n" + "=" * 72)
    log("PROXY COMPARISON — Macro-F1 (mean ± std, 3 seeds)")
    log("=" * 72)
    for ds in DATASETS:
        log(f"\n  {ds.upper()}")
        for proxy in PROXIES:
            vals = []
            for seed in SEEDS:
                p = ROOT / f"experiments/results/{ds}/proxy_comparison/{proxy}/seed_{seed}/test_results.json"
                if p.exists():
                    with open(p) as f:
                        m = json.load(f)
                    vals.append(m.get("macro_f1", 0))
            if vals:
                mean = sum(vals) / len(vals)
                std  = (sum((v-mean)**2 for v in vals) / len(vals))**0.5
                log(f"    {proxy:<10} {mean:.2f} ± {std:.2f}  n={len(vals)}  {vals}")

    log("\n" + "=" * 72)
    log("TABLE 3 ABLATION — Macro-F1 (mean ± std, 3 seeds)")
    log("=" * 72)
    for ds in DATASETS:
        log(f"\n  {ds.upper()}")
        for abl in ABLATIONS:
            vals = []
            for seed in SEEDS:
                p = ROOT / f"experiments/results/{ds}/ablation_multiseed/{abl}/seed_{seed}/test_results.json"
                if p.exists():
                    with open(p) as f:
                        m = json.load(f)
                    vals.append(m.get("macro_f1", 0))
            if vals:
                mean = sum(vals) / len(vals)
                std  = (sum((v-mean)**2 for v in vals) / len(vals))**0.5
                log(f"    {abl:<14} {mean:.2f} ± {std:.2f}  n={len(vals)}  {vals}")

    # Save JSON
    results = {}
    for exp_type, iter_outer, path_tpl in [
        ("proxy", [(ds, p) for ds in DATASETS for p in PROXIES],
         "experiments/results/{}/proxy_comparison/{}/seed_{}/test_results.json"),
        ("ablation", [(ds, a) for ds in DATASETS for a in ABLATIONS],
         "experiments/results/{}/ablation_multiseed/{}/seed_{}/test_results.json"),
    ]:
        results[exp_type] = {}
        for (ds, key) in iter_outer:
            vals = []
            for seed in SEEDS:
                p = ROOT / path_tpl.format(ds, key, seed)
                if p.exists():
                    with open(p) as f:
                        m = json.load(f)
                    vals.append(m.get("macro_f1", 0))
            if vals:
                mean = sum(vals) / len(vals)
                std  = (sum((v-mean)**2 for v in vals) / len(vals))**0.5
                results[exp_type].setdefault(ds, {})[key] = {
                    "mean": mean, "std": std, "n": len(vals), "vals": vals}

    out_json = ROOT / "experiments/results/reviewer_experiments_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  JSON summary → {out_json}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log("=" * 72)
    log("Unified Reviewer Experiment Launcher")
    log("  Proxy comparison (MAD vs Entropy vs Margin): 27 jobs")
    log("  Ablation multi-seed (Table 3 fix): 36 jobs")
    log(f"  GPUs: {ALLOWED_GPUS}  MIN_FREE={MIN_FREE_MB//1000}GB")
    log("=" * 72)

    proxy_jobs   = build_proxy_jobs()
    ablation_jobs = build_ablation_jobs()

    # Interleave proxy and ablation to mix workloads fairly
    # Proxy jobs come first (higher reviewer priority)
    all_jobs = proxy_jobs + ablation_jobs
    log(f"\n  {len(all_jobs)} jobs to run "
        f"({len(proxy_jobs)} proxy + {len(ablation_jobs)} ablation)\n")

    if not all_jobs:
        summarise()
        return

    running: dict[int, tuple] = {}   # gpu → (proc, label, t0)
    queue = list(all_jobs)

    while queue or running:
        # Reap finished
        for gpu in list(running):
            proc, label, t0 = running[gpu]
            ret = proc.poll()
            if ret is not None:
                ela = (time.time() - t0) / 60
                status = "DONE" if ret == 0 else f"FAIL(exit={ret})"
                log(f"  {status}  GPU={gpu}  {label}  {ela:.1f}min")
                del running[gpu]

        # Fill free GPUs
        while queue:
            gpu = pick_free_gpu(set(running.keys()))
            if gpu is None:
                break
            label, log_path, cmd = queue.pop(0)
            proc = launch(label, log_path, cmd, gpu)
            running[gpu] = (proc, label, time.time())
            time.sleep(6)   # stagger to avoid simultaneous GPU checks

        if running or queue:
            active = " | ".join(f"GPU{g}:{lbl.split('/')[-3]}/{lbl.split('/')[-2]}"
                                 for g, (_, lbl, _) in running.items())
            log(f"  running={len(running)} queue={len(queue)}  [{active}]")
            time.sleep(POLL_SEC)

    log("\n=== All jobs complete ===")
    summarise()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
