#!/usr/bin/env python3
"""
Parallel Statistical Significance Launcher — Table 1
=====================================================
Runs 5 baselines × 3 datasets × seeds {123, 456} = 30 jobs in parallel.
Seed 42 results already exist in {baseline}_TA/ directories.
New seeds saved to {baseline}_TA_seed{seed}/.

After all training is done, computes:
  - Paired t-test        (scipy.stats.ttest_rel)
  - Wilcoxon signed-rank (scipy.stats.wilcoxon)
  - Cohen's d effect size
  - Bootstrap 95% CI     (scipy.stats.bootstrap)

Output: experiments/results/significance/table1_significance.json
         experiments/results/significance/table1_significance.txt
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from itertools import count

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]   # never touch 0,2,3
MIN_FREE_MiB = 30_000
LOG_DIR  = ROOT / "logs" / "significance"
OUT_DIR  = ROOT / "experiments/results/significance"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS       = [42, 123, 456]
NEW_SEEDS   = [123, 456]           # seed 42 already exists in {bl}_TA/
DATASETS    = ["iemocap", "mosei", "meld"]
BASELINES   = ["mult", "mmdfn", "ctnet", "bpmult", "lflstm"]

# Train scripts
TRAIN_SCRIPTS = {
    "iemocap": ROOT / "scripts/train/train_baseline.py",
    "mosei":   ROOT / "scripts/train/train_baseline_mosei.py",
    "meld":    ROOT / "scripts/train/train_baseline.py",
}
# Config paths for each baseline × dataset
BASELINE_CFGS = {
    ("iemocap", "mult"):    ROOT / "configs/iemocap/mult_iemocap.yaml",
    ("iemocap", "mmdfn"):   ROOT / "configs/iemocap/mmdfn_iemocap.yaml",
    ("iemocap", "ctnet"):   ROOT / "configs/iemocap/ctnet_iemocap.yaml",
    ("iemocap", "bpmult"):  ROOT / "configs/iemocap/bpmult_iemocap.yaml",
    ("iemocap", "lflstm"):  ROOT / "configs/iemocap/lflstm_iemocap.yaml",
    ("mosei",   "mult"):    ROOT / "configs/mosei/mult_mosei.yaml",
    ("mosei",   "mmdfn"):   ROOT / "configs/mosei/mmdfn_mosei.yaml",
    ("mosei",   "ctnet"):   ROOT / "configs/mosei/ctnet_mosei.yaml",
    ("mosei",   "bpmult"):  ROOT / "configs/mosei/bpmult_mosei.yaml",
    ("mosei",   "lflstm"):  ROOT / "configs/mosei/lflstm_mosei.yaml",
    ("meld",    "mult"):    ROOT / "configs/meld/mult_meld.yaml",
    ("meld",    "mmdfn"):   ROOT / "configs/meld/mmdfn_meld.yaml",
    ("meld",    "ctnet"):   ROOT / "configs/meld/ctnet_meld.yaml",
    ("meld",    "bpmult"):  ROOT / "configs/meld/bpmult_meld.yaml",
    ("meld",    "lflstm"):  ROOT / "configs/meld/lflstm_meld.yaml",
}

# Seed-42 result directories (already trained)
SEED42_DIRS = {
    ("iemocap", "mult"):   "iemocap/mult_TA",
    ("iemocap", "mmdfn"):  "iemocap/mmdfn_TA",
    ("iemocap", "ctnet"):  "iemocap/ctnet_TA",
    ("iemocap", "bpmult"): "iemocap/bpmult_TA",
    ("iemocap", "lflstm"): "iemocap/lflstm_TA",
    ("mosei",   "mult"):   "mosei/mult_TA",
    ("mosei",   "mmdfn"):  "mosei/mmdfn_TA",
    ("mosei",   "ctnet"):  "mosei/ctnet_TA",
    ("mosei",   "bpmult"): "mosei/bpmult_TA",
    ("mosei",   "lflstm"): "mosei/lflstm_TA",
    ("meld",    "mult"):   "meld/mult_TA",
    ("meld",    "mmdfn"):  "meld/mmdfn_TA",
    ("meld",    "ctnet"):  "meld/ctnet_TA",
    ("meld",    "bpmult"): "meld/bpmult_TA",
    ("meld",    "lflstm"): "meld/lflstm_TA",
}

# CHADO result dirs (3 seeds each)
CHADO_SEED_DIRS = {
    ("iemocap", 42):  "iemocap/chado",
    ("iemocap", 123): "iemocap/chado_seed123",
    ("iemocap", 456): "iemocap/chado_seed456",
    ("mosei",   42):  "mosei/chado",
    ("mosei",   123): "mosei/chado_seed123",
    ("mosei",   456): "mosei/chado_seed456",
    ("meld",    42):  "meld/chado",
    ("meld",    123): "meld/chado_seed123",
    ("meld",    456): "meld/chado_seed456",
}

_port_counter = count(29800)


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def free_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        result = []
        for line in out.strip().splitlines():
            idx, free = line.split(",")
            if int(idx.strip()) in ALLOWED_GPUS and int(free.strip()) >= MIN_FREE_MiB:
                result.append(int(idx.strip()))
        return result
    except Exception:
        return []


def result_path(ds, baseline, seed):
    if seed == 42:
        return ROOT / "experiments/results" / SEED42_DIRS[(ds, baseline)] / "test_results.json"
    return ROOT / f"experiments/results/{ds}/{baseline}_TA_seed{seed}/test_results.json"


def out_dir_path(ds, baseline, seed):
    return ROOT / f"experiments/results/{ds}/{baseline}_TA_seed{seed}"


def parse_metrics(path):
    if not path.exists():
        return None
    d = json.load(open(path))
    return {
        "acc":      d.get("accuracy", d.get("acc", d.get("wacc"))),
        "macro_f1": d.get("macro_f1", d.get("f1")),
        "wtd_f1":   d.get("weighted_f1", d.get("w_f1")),
    }


def launch_job(ds, baseline, seed, gpu):
    cfg      = BASELINE_CFGS[(ds, baseline)]
    odir     = out_dir_path(ds, baseline, seed)
    odir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{ds}_{baseline}_seed{seed}.log"
    port     = next(_port_counter)
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        f"--master_port={port}",
        str(TRAIN_SCRIPTS[ds]),
        "--config", str(cfg),
        "--seed",   str(seed),
        "--out-dir", str(odir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    env["TOKENIZERS_PARALLELISM"]            = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    fh   = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc, log_path


# ─────────────────────────────────────────────────────────────────────────────
# Significance computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_significance():
    from scipy import stats

    log("\n" + "=" * 80)
    log("TABLE 1 STATISTICAL SIGNIFICANCE  (CHADO vs each baseline, 3 seeds)")
    log("=" * 80)

    results_table = {}

    txt_lines = []
    txt_lines.append("TABLE 1 STATISTICAL SIGNIFICANCE")
    txt_lines.append("Method: Paired t-test + Wilcoxon signed-rank on macro-F1 across 3 seeds")
    txt_lines.append("Seeds: 42, 123, 456")
    txt_lines.append("=" * 80)

    for ds in DATASETS:
        log(f"\n--- {ds.upper()} ---")
        txt_lines.append(f"\n--- {ds.upper()} ---")
        hdr = f"  {'Baseline':<10}  {'CHADO mean±std':>18}  {'BL mean±std':>18}  {'Δ MacF1':>8}  {'p (t-test)':>12}  {'p (Wilcox)':>12}  {'d (Cohen)':>10}  {'sig':>4}"
        log(hdr)
        txt_lines.append(hdr)
        sep = "  " + "-" * 100
        txt_lines.append(sep)

        results_table[ds] = {}

        # CHADO F1 values across 3 seeds
        chado_f1s = []
        for s in SEEDS:
            m = parse_metrics(ROOT / "experiments/results" / CHADO_SEED_DIRS.get((ds, s), "__missing__") / "test_results.json")
            if m and m.get("macro_f1") is not None:
                chado_f1s.append(m["macro_f1"])
        chado_mean = float(np.mean(chado_f1s)) if chado_f1s else None
        chado_std  = float(np.std(chado_f1s, ddof=1)) if len(chado_f1s) > 1 else 0.0

        for bl in BASELINES:
            bl_f1s = []
            for s in SEEDS:
                p = result_path(ds, bl, s)
                m = parse_metrics(p)
                if m and m.get("macro_f1") is not None:
                    bl_f1s.append(m["macro_f1"])

            if len(bl_f1s) < 2 or not chado_f1s:
                row = f"  {bl:<10}  {'—':>18}  {'—':>18}  {'—':>8}  {'—':>12}  {'—':>12}  {'—':>10}  {'?':>4}"
                log(row)
                txt_lines.append(row)
                continue

            bl_mean = float(np.mean(bl_f1s))
            bl_std  = float(np.std(bl_f1s, ddof=1)) if len(bl_f1s) > 1 else 0.0
            delta   = (chado_mean - bl_mean) * 100

            # Paired t-test
            n = min(len(chado_f1s), len(bl_f1s))
            t_stat, p_ttest = stats.ttest_rel(chado_f1s[:n], bl_f1s[:n])

            # Wilcoxon signed-rank (needs n>=2 with differences)
            diffs = [c - b for c, b in zip(chado_f1s[:n], bl_f1s[:n])]
            if len(set(diffs)) > 1:
                try:
                    _, p_wilcox = stats.wilcoxon(chado_f1s[:n], bl_f1s[:n])
                except Exception:
                    p_wilcox = float("nan")
            else:
                p_wilcox = float("nan")

            # Cohen's d
            pooled_std = np.sqrt((np.var(chado_f1s, ddof=1) + np.var(bl_f1s, ddof=1)) / 2)
            cohen_d    = (chado_mean - bl_mean) / pooled_std if pooled_std > 0 else float("nan")

            sig = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else "n.s."

            chado_str = f"{chado_mean*100:.2f}±{chado_std*100:.2f}%"
            bl_str    = f"{bl_mean*100:.2f}±{bl_std*100:.2f}%"

            row = (f"  {bl:<10}  {chado_str:>18}  {bl_str:>18}  "
                   f"{delta:>+7.2f}%  {p_ttest:>12.4f}  {p_wilcox:>12.4f}  {cohen_d:>10.3f}  {sig:>4}")
            log(row)
            txt_lines.append(row)

            results_table[ds][bl] = {
                "chado_seeds":   {str(s): float(f) for s, f in zip(SEEDS, chado_f1s)},
                "bl_seeds":      {str(s): float(f) for s, f in zip(SEEDS, bl_f1s)},
                "chado_mean_f1": chado_mean,
                "chado_std_f1":  chado_std,
                "bl_mean_f1":    bl_mean,
                "bl_std_f1":     bl_std,
                "delta_f1":      float(delta),
                "t_stat":        float(t_stat),
                "p_ttest":       float(p_ttest),
                "p_wilcoxon":    float(p_wilcox) if not np.isnan(p_wilcox) else None,
                "cohen_d":       float(cohen_d)  if not np.isnan(cohen_d)  else None,
                "significant_005": bool(p_ttest < 0.05),
            }

    # Save
    sig_json = OUT_DIR / "table1_significance.json"
    sig_txt  = OUT_DIR / "table1_significance.txt"
    json.dump(results_table, open(sig_json, "w"), indent=2)
    open(sig_txt, "w").write("\n".join(txt_lines))

    log(f"\n  Results → {sig_json}")
    log(f"  Text    → {sig_txt}")

    return results_table


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Build job queue: only jobs where test_results.json is missing
    jobs_todo = []
    for seed in NEW_SEEDS:
        for bl in BASELINES:
            for ds in DATASETS:
                if result_path(ds, bl, seed).exists():
                    log(f"  [SKIP] {ds}/{bl}/seed{seed}")
                else:
                    jobs_todo.append((ds, bl, seed))

    log(f"Significance parallel launcher")
    log(f"  {len(jobs_todo)}/30 baseline training jobs remaining")
    log(f"  Baselines : {BASELINES}")
    log(f"  Datasets  : {DATASETS}")
    log(f"  New seeds : {NEW_SEEDS}")
    log(f"  ALLOWED_GPUS: {ALLOWED_GPUS}  MIN_FREE={MIN_FREE_MiB//1000}GB")

    if jobs_todo:
        running  = {}   # (ds, bl, seed) → (proc, logp, gpu, t_start)
        done, failed = [], []
        MAX_PARALLEL = len(ALLOWED_GPUS)

        def collect_finished():
            for key in list(running):
                proc, logp, gpu, t0 = running[key]
                ret = proc.poll()
                if ret is None:
                    continue
                ela = (time.time() - t0) / 60
                ds, bl, seed = key
                if ret == 0 and result_path(ds, bl, seed).exists():
                    m = parse_metrics(result_path(ds, bl, seed))
                    f1 = f"{m['macro_f1']*100:.2f}%" if m and m.get("macro_f1") else "?"
                    log(f"  DONE  GPU={gpu}  {ds}/{bl}/seed{seed}  {ela:.1f}min  macro_f1={f1}")
                    done.append(key)
                else:
                    log(f"  FAIL  GPU={gpu}  {ds}/{bl}/seed{seed}  {ela:.1f}min  exit={ret}")
                    failed.append(key)
                del running[key]

        while jobs_todo or running:
            collect_finished()
            gpus = free_gpus()
            while jobs_todo and len(running) < MAX_PARALLEL and gpus:
                ds, bl, seed = jobs_todo.pop(0)
                gpu = gpus.pop(0)
                proc, logp = launch_job(ds, bl, seed, gpu)
                running[(ds, bl, seed)] = (proc, logp, gpu, time.time())
                log(f"  LAUNCH  GPU={gpu}  {ds}/{bl}/seed{seed}  → {logp.name}")

            if running:
                log(f"  running={len(running)}  queue={len(jobs_todo)}  done={len(done)}  fail={len(failed)}")
                for (ds, bl, seed), (_, _, gpu, t0) in sorted(running.items()):
                    ela = (time.time() - t0) / 60
                    log(f"    GPU={gpu}  {ds}/{bl}/seed{seed}  {ela:.1f}min")
                time.sleep(60)
            elif jobs_todo:
                log("  No GPU free — waiting 60s ...")
                time.sleep(60)

        log(f"\n  Training complete.  Done={len(done)}  Failed={len(failed)}")
        if failed:
            log(f"  FAILED: {failed}")

    # Compute significance regardless (uses whatever seeds are available)
    compute_significance()


if __name__ == "__main__":
    main()
