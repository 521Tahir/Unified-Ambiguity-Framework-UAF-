#!/usr/bin/env python3
"""
LOEO (Leave-One-Environment-Out) Cross-Environment Launcher.
=============================================================
Evaluates CHADO under speaker-disjoint cross-environment generalization.

  Datasets  : IEMOCAP (5 sessions), CMU-MOSEI (5 video-id clusters), MELD (5 episode blocks)
  Modalities: text_only | tv (text+video) | tav (text+audio+video)
  Folds     : 5 per dataset
  Total jobs: 3 modalities × 5 folds × 3 datasets = 45

Results: mean ± std across folds → experiments/results/loeo/loeo_summary.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ALLOWED_GPUS = [1, 5, 7, 9]   # exclusive pool — no overlap with proxy launcher [4,6,8]
MIN_FREE_MiB = 46_000

LOEO_DIR  = ROOT / "data" / "loeo"
OUT_BASE  = ROOT / "experiments/results/loeo"
LOG_DIR   = ROOT / "logs/loeo"
N_FOLDS   = 5
SEED      = 42

OUT_BASE.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["iemocap", "mosei", "meld"]

# Modality configurations: (name, use_audio, use_video)
MODALITIES = [
    ("text_only", False, False),
    ("tv",        False, True),
    ("tav",       True,  True),
]

BASE_CFGS = {
    "iemocap": ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
    "mosei":   ROOT / "configs/mosei/chado_mosei_best.yaml",
    "meld":    ROOT / "configs/meld/chado_meld_final.yaml",
}

TRAIN_SCRIPTS = {
    "iemocap": ROOT / "scripts/train/train_iemocap.py",
    "mosei":   ROOT / "scripts/train/train_mosei.py",
    "meld":    ROOT / "scripts/train/train_meld.py",
}

# train_mosei.py has no --seed CLI flag
HAS_SEED_FLAG = {"iemocap": True, "mosei": False, "meld": True}


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


def out_dir(ds: str, mod: str, fold: int) -> Path:
    return OUT_BASE / ds / mod / f"fold_{fold}"


def result_path(ds: str, mod: str, fold: int) -> Path:
    return out_dir(ds, mod, fold) / "test_results.json"


def parse_result(ds: str, mod: str, fold: int):
    p = result_path(ds, mod, fold)
    if not p.exists():
        return None
    d = json.load(open(p))
    acc = d.get("accuracy", d.get("acc", d.get("wacc")))
    f1  = d.get("macro_f1", d.get("f1"))
    wf1 = d.get("weighted_f1", d.get("w_f1"))
    return {"acc": acc, "macro_f1": f1, "weighted_f1": wf1}


def make_config(ds: str, mod: str, fold: int, use_audio: bool, use_video: bool) -> Path:
    base = yaml.safe_load(open(BASE_CFGS[ds]))

    # Override data paths with fold-specific splits
    fold_dir = LOEO_DIR / ds / f"fold_{fold}"
    if ds == "mosei":
        base["data"]["train_manifest"] = str(fold_dir / "train.jsonl")
        base["data"]["val_manifest"]   = str(fold_dir / "val.jsonl")
        base["data"]["test_manifest"]  = str(fold_dir / "test.jsonl")
    else:
        base["data"]["train_csv"] = str(fold_dir / "train.csv")
        base["data"]["val_csv"]   = str(fold_dir / "val.csv")
        base["data"]["test_csv"]  = str(fold_dir / "test.csv")

    # Override modality flags
    base["model"]["use_audio"] = use_audio
    base["model"]["use_video"] = use_video
    base["model"]["use_text"]  = True

    base["train"]["seed"] = SEED
    odir = str(out_dir(ds, mod, fold))
    base["logging"]["run_name"] = f"loeo_{ds}_{mod}_fold{fold}"
    base["logging"]["out_dir"]  = odir

    cfg_path = LOG_DIR / f"cfg_{ds}_{mod}_fold{fold}.yaml"
    yaml.dump(base, open(cfg_path, "w"), default_flow_style=False)
    return cfg_path


def launch_job(ds: str, mod: str, fold: int, use_audio: bool, use_video: bool, gpu: int):
    cfg_path = make_config(ds, mod, fold, use_audio, use_video)
    odir     = out_dir(ds, mod, fold)
    odir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{ds}_{mod}_fold{fold}.log"
    cmd = ["python3", "-u", str(TRAIN_SCRIPTS[ds]),
           "--config", str(cfg_path),
           "--out-dir", str(odir)]
    if HAS_SEED_FLAG[ds]:
        cmd += ["--seed", str(SEED)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    env["TOKENIZERS_PARALLELISM"]            = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    print(f"  [{ts()}] LAUNCH  GPU={gpu}  {ds}/{mod}/fold{fold}  → {log_path.name}", flush=True)
    fh   = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc, log_path


def build_summary():
    table = {}
    print("\n" + "=" * 96)
    print("LOEO CROSS-ENVIRONMENT RESULTS  (mean ± std across 5 folds, seed=42)")
    print("=" * 96)

    header = (f"\n  {'Modality':<12} {'Dataset':<10} "
              f"{'Acc (mean±std)':>20} {'MacF1 (mean±std)':>22} {'WtF1 (mean±std)':>22}")
    print(header)
    print("  " + "-" * 88)

    for mod_name, _, _ in MODALITIES:
        for ds in DATASETS:
            results = [parse_result(ds, mod_name, f) for f in range(N_FOLDS)]
            accs  = [r["acc"]         for r in results if r and r.get("acc")         is not None]
            mf1s  = [r["macro_f1"]    for r in results if r and r.get("macro_f1")    is not None]
            wf1s  = [r["weighted_f1"] for r in results if r and r.get("weighted_f1") is not None]

            def fmt(vals):
                if len(vals) > 1:
                    return f"{mean(vals)*100:.2f}±{stdev(vals)*100:.2f}%"
                elif len(vals) == 1:
                    return f"{vals[0]*100:.2f}% (1 fold)"
                return "—"

            print(f"  {mod_name:<12} {ds.upper():<10} {fmt(accs):>20} {fmt(mf1s):>22} {fmt(wf1s):>22}")

            if mod_name not in table:
                table[mod_name] = {}
            table[mod_name][ds] = {
                "folds":    {str(f): parse_result(ds, mod_name, f) for f in range(N_FOLDS)},
                "acc_mean": mean(accs)  if accs else None,
                "acc_std":  stdev(accs) if len(accs) > 1 else 0,
                "mf1_mean": mean(mf1s)  if mf1s else None,
                "mf1_std":  stdev(mf1s) if len(mf1s) > 1 else 0,
                "wf1_mean": mean(wf1s)  if wf1s else None,
                "wf1_std":  stdev(wf1s) if len(wf1s) > 1 else 0,
            }
        print()

    out = OUT_BASE / "loeo_summary.json"
    json.dump(table, open(out, "w"), indent=2)
    print(f"  Results → {out}")


def main():
    jobs_todo = []
    for mod_name, use_audio, use_video in MODALITIES:
        for fold in range(N_FOLDS):
            for ds in DATASETS:
                if parse_result(ds, mod_name, fold) is not None:
                    print(f"  [SKIP] {ds}/{mod_name}/fold{fold}")
                else:
                    jobs_todo.append((ds, mod_name, fold, use_audio, use_video))

    total = len(MODALITIES) * N_FOLDS * len(DATASETS)
    print(f"\n  [{ts()}] LOEO Launcher")
    print(f"  {len(jobs_todo)}/{total} jobs to run  "
          f"({len(MODALITIES)} modalities × {N_FOLDS} folds × {len(DATASETS)} datasets)")
    print(f"  Modalities: {[m[0] for m in MODALITIES]}")
    print(f"  Allowed GPUs: {ALLOWED_GPUS}  MIN_FREE={MIN_FREE_MiB//1000}GB\n")

    if not jobs_todo:
        print("  All results exist.")
        build_summary()
        return

    running  = {}   # (ds, mod, fold) → (proc, logp, gpu, t_start)
    done, failed = [], []
    MAX_PARALLEL = len(ALLOWED_GPUS)  # 7

    def collect_finished():
        for key in list(running):
            proc, logp, gpu, t0 = running[key]
            ret = proc.poll()
            if ret is None:
                continue
            elapsed = (time.time() - t0) / 60
            ds, mod, fold = key
            if ret == 0:
                r = parse_result(ds, mod, fold)
                acc_str = f"{r['acc']*100:.2f}%" if r and r.get("acc") else "?"
                print(f"  [{ts()}] DONE  GPU={gpu}  {ds}/{mod}/fold{fold}  "
                      f"{elapsed:.1f}min  acc={acc_str}", flush=True)
                done.append(key)
            else:
                print(f"  [{ts()}] FAIL  GPU={gpu}  {ds}/{mod}/fold{fold}  exit={ret}", flush=True)
                failed.append(key)
            del running[key]

    while jobs_todo or running:
        collect_finished()
        gpus = free_gpus()
        while jobs_todo and len(running) < MAX_PARALLEL and gpus:
            ds, mod, fold, use_audio, use_video = jobs_todo.pop(0)
            gpu = gpus.pop(0)
            proc, logp = launch_job(ds, mod, fold, use_audio, use_video, gpu)
            running[(ds, mod, fold)] = (proc, logp, gpu, time.time())

        if running:
            print(f"  [{ts()}] running={len(running)}  queue={len(jobs_todo)}  "
                  f"done={len(done)}  fail={len(failed)}", flush=True)
            for (ds, mod, fold), (_, _, gpu, t0) in sorted(running.items()):
                ela = (time.time() - t0) / 60
                print(f"    GPU={gpu}  {ds}/{mod}/fold{fold}  {ela:.1f}min")
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
