#!/usr/bin/env python3
"""
Sequential experiment runner — one job at a time, one GPU.
Covers all three experiment suites in order:
  1. Component ablations  (12 jobs: 4 ablations × 3 datasets)
  2. Proxy multiseed      (45 jobs: 5 proxies × 3 datasets × 3 seeds)
  3. LOEO                 (45 jobs: 3 modalities × 5 folds × 3 datasets)

Skips any job whose test_results.json already exists.
On failure: logs the error and moves to the next job (no retry).
GPU: picks the first allowed GPU with enough free memory.
"""

import json
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]   # never touch 0, 2, 3 (other user)
MIN_FREE_MiB = 30_000

MAIN_LOG = LOG_DIR / "sequential_launcher.log"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, fh=None):
    print(f"[{ts()}] {msg}", flush=True)


def free_gpu():
    """Return first allowed GPU with >= MIN_FREE_MiB free, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            idx, free = line.split(",")
            if int(idx.strip()) in ALLOWED_GPUS and int(free.strip()) >= MIN_FREE_MiB:
                return int(idx.strip())
    except Exception:
        pass
    return None


def wait_for_gpu(fh=None):
    while True:
        g = free_gpu()
        if g is not None:
            return g
        log("  No GPU free — waiting 60s ...", fh)
        time.sleep(60)


def run_job(cmd, gpu, log_path, env_extra=None, fh=None):
    """Run cmd on gpu, stream to log_path. Returns (returncode, elapsed_min)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    env["TOKENIZERS_PARALLELISM"]            = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    if env_extra:
        env.update(env_extra)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=lf)
    elapsed = (time.time() - t0) / 60
    return proc.returncode, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPONENT ABLATIONS
# ─────────────────────────────────────────────────────────────────────────────

COMP_ABLATIONS = ["wo_mad", "wo_ot", "wo_radial", "wo_causal"]
COMP_DATASETS  = ["iemocap", "mosei", "meld"]
COMP_SCRIPTS   = {
    "iemocap": ROOT / "scripts/train/train_iemocap.py",
    "mosei":   ROOT / "scripts/train/train_mosei.py",
    "meld":    ROOT / "scripts/train/train_meld.py",
}
COMP_CFGS = {
    ("iemocap", "wo_mad"):    ROOT / "configs/iemocap/comp_abl_wo_mad.yaml",
    ("iemocap", "wo_ot"):     ROOT / "configs/iemocap/comp_abl_wo_ot.yaml",
    ("iemocap", "wo_radial"): ROOT / "configs/iemocap/comp_abl_wo_radial.yaml",
    ("iemocap", "wo_causal"): ROOT / "configs/iemocap/comp_abl_wo_causal.yaml",
    ("mosei",   "wo_mad"):    ROOT / "configs/mosei/comp_abl_wo_mad.yaml",
    ("mosei",   "wo_ot"):     ROOT / "configs/mosei/comp_abl_wo_ot.yaml",
    ("mosei",   "wo_radial"): ROOT / "configs/mosei/comp_abl_wo_radial.yaml",
    ("mosei",   "wo_causal"): ROOT / "configs/mosei/comp_abl_wo_causal.yaml",
    ("meld",    "wo_mad"):    ROOT / "configs/meld/comp_abl_wo_mad.yaml",
    ("meld",    "wo_ot"):     ROOT / "configs/meld/comp_abl_wo_ot.yaml",
    ("meld",    "wo_radial"): ROOT / "configs/meld/comp_abl_wo_radial.yaml",
    ("meld",    "wo_causal"): ROOT / "configs/meld/comp_abl_wo_causal.yaml",
}
COMP_HAS_SEED = {"iemocap": True, "mosei": False, "meld": True}


def comp_result_path(ds, abl):
    return ROOT / f"experiments/results/{ds}/component_ablations/{abl}/seed_42/test_results.json"


def comp_out_dir(ds, abl):
    return ROOT / f"experiments/results/{ds}/component_ablations/{abl}/seed_42"


def build_comp_jobs():
    jobs = []
    for abl in COMP_ABLATIONS:
        for ds in COMP_DATASETS:
            if comp_result_path(ds, abl).exists():
                continue
            jobs.append(("comp", ds, abl))
    return jobs


def run_comp_job(ds, abl, gpu, fh):
    cfg  = COMP_CFGS[(ds, abl)]
    odir = comp_out_dir(ds, abl)
    odir.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", "-u", str(COMP_SCRIPTS[ds]),
           "--config", str(cfg), "--out-dir", str(odir)]
    if COMP_HAS_SEED[ds]:
        cmd += ["--seed", "42"]
    log_path = LOG_DIR / f"comp_{ds}_{abl}.log"
    log(f"  COMP  GPU={gpu}  {ds}/{abl}  → {log_path.name}", fh)
    rc, ela = run_job(cmd, gpu, log_path, fh=fh)
    if rc == 0 and comp_result_path(ds, abl).exists():
        d = json.load(open(comp_result_path(ds, abl)))
        acc = d.get("accuracy", d.get("acc", d.get("wacc", "?")))
        log(f"  DONE  {ds}/{abl}  {ela:.1f}min  acc={acc}", fh)
        return True
    else:
        log(f"  FAIL  {ds}/{abl}  {ela:.1f}min  exit={rc}", fh)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROXY ABLATION MULTISEED
# ─────────────────────────────────────────────────────────────────────────────

PROXY_PROXIES  = ["mad", "entropy", "margin", "mc_var", "bald"]
PROXY_DATASETS = ["iemocap", "mosei", "meld"]
PROXY_SEEDS    = [42, 123, 456]
PROXY_SCRIPTS  = {
    "iemocap": ROOT / "scripts/train/train_iemocap.py",
    "mosei":   ROOT / "scripts/train/train_mosei_dom.py",
    "meld":    ROOT / "scripts/train/train_meld.py",
}
PROXY_BASE_CFGS = {
    "iemocap": ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
    "mosei":   ROOT / "configs/mosei/chado_mosei_best.yaml",
    "meld":    ROOT / "configs/meld/chado_meld_final.yaml",
}
PROXY_HAS_SEED = {"iemocap": True, "mosei": False, "meld": True}
PROXY_LOG_DIR  = LOG_DIR / "proxy_ablation_multiseed"
PROXY_LOG_DIR.mkdir(exist_ok=True)


def proxy_result_path(ds, proxy, seed):
    return ROOT / f"experiments/results/proxy_ablation_multiseed/{ds}/proxy_{proxy}/seed_{seed}/test_results.json"


def proxy_out_dir(ds, proxy, seed):
    return ROOT / f"experiments/results/proxy_ablation_multiseed/{ds}/proxy_{proxy}/seed_{seed}"


def proxy_make_config(ds, proxy, seed):
    base = yaml.safe_load(open(PROXY_BASE_CFGS[ds]))
    if "chado" not in base:
        base["chado"] = {}
    base["chado"]["proxy"]      = proxy
    base["train"]["seed"]       = seed
    odir = str(proxy_out_dir(ds, proxy, seed))
    base["logging"]["run_name"] = f"chado_{ds}_proxy_{proxy}_seed{seed}"
    base["logging"]["out_dir"]  = odir
    cfg_path = PROXY_LOG_DIR / f"cfg_{ds}_{proxy}_seed{seed}.yaml"
    yaml.dump(base, open(cfg_path, "w"), default_flow_style=False)
    return cfg_path


def build_proxy_jobs():
    jobs = []
    for seed in PROXY_SEEDS:
        for proxy in PROXY_PROXIES:
            for ds in PROXY_DATASETS:
                if proxy_result_path(ds, proxy, seed).exists():
                    continue
                jobs.append(("proxy", ds, proxy, seed))
    return jobs


def run_proxy_job(ds, proxy, seed, gpu, fh):
    cfg  = proxy_make_config(ds, proxy, seed)
    odir = proxy_out_dir(ds, proxy, seed)
    odir.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", "-u", str(PROXY_SCRIPTS[ds]),
           "--config", str(cfg), "--out-dir", str(odir)]
    if PROXY_HAS_SEED[ds]:
        cmd += ["--seed", str(seed)]
    log_path = PROXY_LOG_DIR / f"{ds}_{proxy}_seed{seed}.log"
    log(f"  PROXY GPU={gpu}  {ds}/proxy_{proxy}/seed{seed}  → {log_path.name}", fh)
    rc, ela = run_job(cmd, gpu, log_path, fh=fh)
    if rc == 0 and proxy_result_path(ds, proxy, seed).exists():
        d = json.load(open(proxy_result_path(ds, proxy, seed)))
        acc = d.get("accuracy", d.get("acc", d.get("wacc", "?")))
        log(f"  DONE  {ds}/proxy_{proxy}/seed{seed}  {ela:.1f}min  acc={acc}", fh)
        return True
    else:
        log(f"  FAIL  {ds}/proxy_{proxy}/seed{seed}  {ela:.1f}min  exit={rc}", fh)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOEO
# ─────────────────────────────────────────────────────────────────────────────

LOEO_DATASETS   = ["iemocap", "mosei", "meld"]
LOEO_MODALITIES = [
    ("text_only", False, False),
    ("tv",        False, True),
    ("tav",       True,  True),
]
LOEO_N_FOLDS  = 5
LOEO_SEED     = 42
LOEO_BASE_CFGS = {
    "iemocap": ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
    "mosei":   ROOT / "configs/mosei/chado_mosei_best.yaml",
    "meld":    ROOT / "configs/meld/chado_meld_final.yaml",
}
LOEO_SCRIPTS = {
    "iemocap": ROOT / "scripts/train/train_iemocap.py",
    "mosei":   ROOT / "scripts/train/train_mosei.py",
    "meld":    ROOT / "scripts/train/train_meld.py",
}
LOEO_HAS_SEED = {"iemocap": True, "mosei": False, "meld": True}
LOEO_LOG_DIR  = LOG_DIR / "loeo"
LOEO_LOG_DIR.mkdir(exist_ok=True)
LOEO_DATA_DIR = ROOT / "data/loeo"


def loeo_result_path(ds, mod, fold):
    return ROOT / f"experiments/results/loeo/{ds}/{mod}/fold_{fold}/test_results.json"


def loeo_out_dir(ds, mod, fold):
    return ROOT / f"experiments/results/loeo/{ds}/{mod}/fold_{fold}"


def loeo_make_config(ds, mod, fold, use_audio, use_video):
    base = yaml.safe_load(open(LOEO_BASE_CFGS[ds]))
    fold_dir = LOEO_DATA_DIR / ds / f"fold_{fold}"
    if ds == "mosei":
        base["data"]["train_manifest"] = str(fold_dir / "train.jsonl")
        base["data"]["val_manifest"]   = str(fold_dir / "val.jsonl")
        base["data"]["test_manifest"]  = str(fold_dir / "test.jsonl")
    else:
        base["data"]["train_csv"] = str(fold_dir / "train.csv")
        base["data"]["val_csv"]   = str(fold_dir / "val.csv")
        base["data"]["test_csv"]  = str(fold_dir / "test.csv")
    base["model"]["use_audio"] = use_audio
    base["model"]["use_video"] = use_video
    base["model"]["use_text"]  = True
    base["train"]["seed"] = LOEO_SEED
    odir = str(loeo_out_dir(ds, mod, fold))
    base["logging"]["run_name"] = f"loeo_{ds}_{mod}_fold{fold}"
    base["logging"]["out_dir"]  = odir
    cfg_path = LOEO_LOG_DIR / f"cfg_{ds}_{mod}_fold{fold}.yaml"
    yaml.dump(base, open(cfg_path, "w"), default_flow_style=False)
    return cfg_path


def build_loeo_jobs():
    jobs = []
    for mod_name, use_audio, use_video in LOEO_MODALITIES:
        for fold in range(LOEO_N_FOLDS):
            for ds in LOEO_DATASETS:
                if loeo_result_path(ds, mod_name, fold).exists():
                    continue
                jobs.append(("loeo", ds, mod_name, fold, use_audio, use_video))
    return jobs


def run_loeo_job(ds, mod, fold, use_audio, use_video, gpu, fh):
    cfg  = loeo_make_config(ds, mod, fold, use_audio, use_video)
    odir = loeo_out_dir(ds, mod, fold)
    odir.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", "-u", str(LOEO_SCRIPTS[ds]),
           "--config", str(cfg), "--out-dir", str(odir)]
    if LOEO_HAS_SEED[ds]:
        cmd += ["--seed", str(LOEO_SEED)]
    log_path = LOEO_LOG_DIR / f"{ds}_{mod}_fold{fold}.log"
    log(f"  LOEO  GPU={gpu}  {ds}/{mod}/fold{fold}  → {log_path.name}", fh)
    rc, ela = run_job(cmd, gpu, log_path, fh=fh)
    if rc == 0 and loeo_result_path(ds, mod, fold).exists():
        d = json.load(open(loeo_result_path(ds, mod, fold)))
        acc = d.get("accuracy", d.get("acc", d.get("wacc", "?")))
        log(f"  DONE  {ds}/{mod}/fold{fold}  {ela:.1f}min  acc={acc}", fh)
        return True
    else:
        log(f"  FAIL  {ds}/{mod}/fold{fold}  {ela:.1f}min  exit={rc}", fh)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fh = None

    comp_jobs  = build_comp_jobs()
    proxy_jobs = build_proxy_jobs()
    loeo_jobs  = build_loeo_jobs()

    total = len(comp_jobs) + len(proxy_jobs) + len(loeo_jobs)
    log(f"Sequential launcher started", fh)
    log(f"  Component ablations : {len(comp_jobs)}/12 remaining", fh)
    log(f"  Proxy multiseed     : {len(proxy_jobs)}/45 remaining", fh)
    log(f"  LOEO                : {len(loeo_jobs)}/45 remaining", fh)
    log(f"  Total               : {total} jobs to run (one at a time)", fh)

    done, failed = 0, 0

    # ── Phase 1: Component ablations ──────────────────────────────────────────
    log(f"\n=== PHASE 1: Component Ablations ({len(comp_jobs)} jobs) ===", fh)
    for job in comp_jobs:
        _, ds, abl = job
        gpu = wait_for_gpu(fh)
        ok = run_comp_job(ds, abl, gpu, fh)
        if ok:
            done += 1
        else:
            failed += 1
        log(f"  Progress: {done+failed}/{total}  done={done}  fail={failed}", fh)

    # ── Phase 2: Proxy multiseed ───────────────────────────────────────────────
    log(f"\n=== PHASE 2: Proxy Ablation Multiseed ({len(proxy_jobs)} jobs) ===", fh)
    for job in proxy_jobs:
        _, ds, proxy, seed = job
        gpu = wait_for_gpu(fh)
        ok = run_proxy_job(ds, proxy, seed, gpu, fh)
        if ok:
            done += 1
        else:
            failed += 1
        log(f"  Progress: {done+failed}/{total}  done={done}  fail={failed}", fh)

    # ── Phase 3: LOEO ─────────────────────────────────────────────────────────
    log(f"\n=== PHASE 3: LOEO ({len(loeo_jobs)} jobs) ===", fh)
    for job in loeo_jobs:
        _, ds, mod, fold, use_audio, use_video = job
        gpu = wait_for_gpu(fh)
        ok = run_loeo_job(ds, mod, fold, use_audio, use_video, gpu, fh)
        if ok:
            done += 1
        else:
            failed += 1
        log(f"  Progress: {done+failed}/{total}  done={done}  fail={failed}", fh)

    log(f"\n=== ALL DONE  done={done}  failed={failed} ===")


if __name__ == "__main__":
    main()
