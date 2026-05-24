#!/usr/bin/env python3
"""
Proxy Ablation Multi-Seed Launcher — Table 16 (robust version)
================================================================
Replaces MAD curriculum weight with 4 alternatives and trains full
CHADO on all three datasets with 3 seeds each.

  Proxies : mad | entropy | margin | mc_var | bald
  Datasets: iemocap (60 ep / patience 15)
            mosei   (40 ep / patience 10)
            meld    (60 ep / patience 15)
  Seeds   : 42, 123, 456

45 total jobs launched across allowed GPUs as they free up.
Results → experiments/results/proxy_ablation_multiseed/
Summary → experiments/results/proxy_ablation_multiseed/summary.json
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

ALLOWED_GPUS = [4, 6, 8]   # exclusive pool — no overlap with LOEO launcher [1,5,7,9]
MIN_FREE_MiB = 46_000

PROXIES  = ["mad", "entropy", "margin", "mc_var", "bald"]
DATASETS = ["iemocap", "mosei", "meld"]
SEEDS    = [42, 123, 456]

OUT_BASE = ROOT / "experiments/results/proxy_ablation_multiseed"
LOG_DIR  = ROOT / "logs/proxy_ablation_multiseed"
OUT_BASE.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_CFGS = {
    "iemocap": ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
    "mosei":   ROOT / "configs/mosei/chado_mosei_best.yaml",
    "meld":    ROOT / "configs/meld/chado_meld_final.yaml",
}

TRAIN_SCRIPTS = {
    "iemocap": ROOT / "scripts/train/train_iemocap.py",
    "mosei":   ROOT / "scripts/train/train_mosei_dom.py",
    "meld":    ROOT / "scripts/train/train_meld.py",
}

# train_mosei_dom.py has no --seed CLI flag; seed injected via config
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


def out_dir(ds: str, proxy: str, seed: int) -> Path:
    return OUT_BASE / ds / f"proxy_{proxy}" / f"seed_{seed}"


def result_path(ds: str, proxy: str, seed: int) -> Path:
    return out_dir(ds, proxy, seed) / "test_results.json"


def parse_result(ds: str, proxy: str, seed: int):
    p = result_path(ds, proxy, seed)
    if not p.exists():
        return None
    d = json.load(open(p))
    acc = d.get("accuracy", d.get("acc", d.get("wacc")))
    f1  = d.get("macro_f1", d.get("f1"))
    wf1 = d.get("weighted_f1", d.get("w_f1"))
    return {"acc": acc, "macro_f1": f1, "weighted_f1": wf1}


def make_config(ds: str, proxy: str, seed: int) -> Path:
    base = yaml.safe_load(open(BASE_CFGS[ds]))
    if "chado" not in base:
        base["chado"] = {}
    base["chado"]["proxy"]       = proxy
    base["train"]["seed"]        = seed
    # Use actual epochs from config (no override) — patience stays as-is
    odir = str(out_dir(ds, proxy, seed))
    base["logging"]["run_name"] = f"chado_{ds}_proxy_{proxy}_seed{seed}"
    base["logging"]["out_dir"]  = odir
    cfg_path = LOG_DIR / f"cfg_{ds}_{proxy}_seed{seed}.yaml"
    yaml.dump(base, open(cfg_path, "w"), default_flow_style=False)
    return cfg_path


def launch_job(ds: str, proxy: str, seed: int, gpu: int):
    cfg_path = make_config(ds, proxy, seed)
    odir     = out_dir(ds, proxy, seed)
    odir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{ds}_{proxy}_seed{seed}.log"
    cmd = ["python3", "-u", str(TRAIN_SCRIPTS[ds]),
           "--config", str(cfg_path),
           "--out-dir", str(odir)]
    if HAS_SEED_FLAG[ds]:
        cmd += ["--seed", str(seed)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    env["TOKENIZERS_PARALLELISM"]            = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    print(f"  [{ts()}] LAUNCH  GPU={gpu}  {ds}/proxy_{proxy}/seed{seed}  → {log_path.name}",
          flush=True)
    fh   = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc, log_path


def build_summary():
    table = {}
    print("\n" + "=" * 88)
    print("PROXY ABLATION — Multi-Seed Results  (seeds: 42, 123, 456 | full training)")
    print("=" * 88)
    print(f"\n  {'Proxy':<10} {'Dataset':<10} "
          f"{'Acc (mean±std)':>20} {'MacF1 (mean±std)':>22} {'WtF1 (mean±std)':>22}")
    print("  " + "-" * 84)

    for proxy in PROXIES:
        for ds in DATASETS:
            results = [parse_result(ds, proxy, s) for s in SEEDS]
            accs  = [r["acc"]          for r in results if r and r.get("acc")      is not None]
            mf1s  = [r["macro_f1"]     for r in results if r and r.get("macro_f1") is not None]
            wf1s  = [r["weighted_f1"]  for r in results if r and r.get("weighted_f1") is not None]

            def fmt(vals):
                if len(vals) > 1:
                    return f"{mean(vals)*100:.2f}±{stdev(vals)*100:.2f}%"
                elif len(vals) == 1:
                    return f"{vals[0]*100:.2f}%"
                return "—"

            print(f"  {proxy:<10} {ds.upper():<10} {fmt(accs):>20} {fmt(mf1s):>22} {fmt(wf1s):>22}")
            if proxy not in table:
                table[proxy] = {}
            table[proxy][ds] = {
                "seeds":    {str(s): parse_result(ds, proxy, s) for s in SEEDS},
                "acc_mean": mean(accs)  if accs  else None,
                "acc_std":  stdev(accs) if len(accs) > 1 else 0,
                "mf1_mean": mean(mf1s)  if mf1s  else None,
                "mf1_std":  stdev(mf1s) if len(mf1s) > 1 else 0,
                "wf1_mean": mean(wf1s)  if wf1s  else None,
                "wf1_std":  stdev(wf1s) if len(wf1s) > 1 else 0,
            }
        print()

    out = OUT_BASE / "summary.json"
    json.dump(table, open(out, "w"), indent=2)
    print(f"  Results → {out}")


def main():
    # Build job list: seed-major, proxy-minor, dataset-minor
    # → all datasets start across all proxies ASAP
    jobs_todo = []
    for seed in SEEDS:
        for proxy in PROXIES:
            for ds in DATASETS:
                if parse_result(ds, proxy, seed) is not None:
                    print(f"  [SKIP] {ds}/proxy_{proxy}/seed{seed} — result exists")
                else:
                    jobs_todo.append((ds, proxy, seed))

    print(f"\n  [{ts()}] Proxy Ablation Multi-Seed Launcher")
    print(f"  {len(jobs_todo)} jobs  ({len(PROXIES)} proxies × {len(DATASETS)} datasets × {len(SEEDS)} seeds)")
    print(f"  Epochs: IEMOCAP=60/p15  MOSEI=40/p10  MELD=60/p15  (no override)\n")

    if not jobs_todo:
        print("  All results already exist.")
        build_summary()
        return

    running  = {}   # (ds, proxy, seed) → (proc, logp, gpu, t_start)
    done, failed = [], []
    MAX_PARALLEL = len(ALLOWED_GPUS)   # 7

    def collect_finished():
        for key in list(running):
            proc, logp, gpu, t0 = running[key]
            ret = proc.poll()
            if ret is None:
                continue
            elapsed = (time.time() - t0) / 60
            ds, proxy, seed = key
            if ret == 0:
                r = parse_result(ds, proxy, seed)
                acc_str = f"{r['acc']*100:.2f}%" if r and r.get("acc") else "?"
                print(f"  [{ts()}] DONE   GPU={gpu}  {ds}/proxy_{proxy}/seed{seed}  "
                      f"{elapsed:.1f}min  acc={acc_str}", flush=True)
                done.append(key)
            else:
                print(f"  [{ts()}] FAIL   GPU={gpu}  {ds}/proxy_{proxy}/seed{seed}  "
                      f"exit={ret}", flush=True)
                failed.append(key)
            del running[key]

    while jobs_todo or running:
        collect_finished()
        gpus = free_gpus()
        while jobs_todo and len(running) < MAX_PARALLEL and gpus:
            ds, proxy, seed = jobs_todo.pop(0)
            gpu = gpus.pop(0)
            proc, logp = launch_job(ds, proxy, seed, gpu)
            running[(ds, proxy, seed)] = (proc, logp, gpu, time.time())

        if running:
            print(f"  [{ts()}] running={len(running)}  queue={len(jobs_todo)}  "
                  f"done={len(done)}  fail={len(failed)}", flush=True)
            for (ds, proxy, seed), (_, _, gpu, t0) in sorted(running.items()):
                ela = (time.time() - t0) / 60
                print(f"    GPU={gpu}  {ds}/proxy_{proxy}/seed{seed}  {ela:.1f}min")
            time.sleep(60)

    print(f"\n  [{ts()}] All jobs complete.  Done={len(done)}  Failed={len(failed)}")
    if failed:
        print("  FAILED:", failed)
    build_summary()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.chdir(ROOT)
    main()
