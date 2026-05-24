#!/usr/bin/env python3
"""
Proxy Ablation Launcher — Table 4 Training Impact
===================================================
Replaces the MAD curriculum weight with 4 alternatives and trains
full CHADO on all three datasets (1 seed each for speed).

Runs up to 5 jobs in parallel, one per free GPU.

Proxies: mad | entropy | margin | mc_var | bald
Datasets: iemocap | mosei | meld

Results → experiments/results/proxy_ablation/summary.json
"""

import os, sys, json, time, signal, subprocess, shutil, yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / "experiments/results/proxy_ablation"
LOG_DIR  = ROOT / "logs/proxy_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PROXIES  = ["mad", "entropy", "margin", "mc_var", "bald"]
DATASETS = ["iemocap", "mosei", "meld"]
SEED     = 42

# ── GPU pool ───────────────────────────────────────────────────────────────────
ALLOWED_GPUS   = [1, 4, 5, 6, 7, 8, 9]
MIN_FREE_MiB   = 38_000          # need ~38 GB for trimodal; MOSEI needs ~20 GB

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

# ── Config generation (inject proxy: <name> into chado section) ───────────────
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

def make_config(dataset: str, proxy: str) -> Path:
    """Write a temporary config with proxy injected."""
    base = yaml.safe_load(open(BASE_CFGS[dataset]))
    if "chado" not in base:
        base["chado"] = {}
    base["chado"]["proxy"]       = proxy
    base["train"]["seed"]        = SEED
    base["train"]["epochs"]      = 5
    base["train"]["patience"]    = 999   # disable early stopping for 5-epoch run
    run_name = f"chado_{dataset}_proxy_{proxy}"
    base["logging"]["run_name"] = run_name
    base["logging"]["out_dir"]  = str(OUT_DIR / dataset / f"proxy_{proxy}")
    cfg_path = LOG_DIR / f"cfg_{dataset}_{proxy}.yaml"
    yaml.dump(base, open(cfg_path, "w"), default_flow_style=False)
    return cfg_path

# ── Result parsing ─────────────────────────────────────────────────────────────
def parse_result(dataset: str, proxy: str):
    """Read test_results.json from the run output dir (written directly to out_dir)."""
    res_dir = OUT_DIR / dataset / f"proxy_{proxy}"
    for cand in [res_dir / "test_results.json",
                 res_dir / "results.json",
                 res_dir / f"seed_{SEED}" / "test_results.json"]:
        if cand.exists():
            d = json.load(open(cand))
            acc = d.get("accuracy", d.get("acc", d.get("test_acc", None)))
            f1  = d.get("macro_f1", d.get("f1", d.get("test_f1", None)))
            wf1 = d.get("weighted_f1", d.get("w_f1", None))
            return {"acc": acc, "macro_f1": f1, "weighted_f1": wf1}
    return None

# ── Job runner ─────────────────────────────────────────────────────────────────
def launch_job(dataset, proxy, gpu):
    cfg_path = make_config(dataset, proxy)
    log_path = LOG_DIR / f"{dataset}_{proxy}.log"
    # train_mosei_dom.py has no --seed flag (seed set via config)
    has_seed_flag = dataset in ("iemocap", "meld")
    cmd = [
        "python3", "-u",
        str(TRAIN_SCRIPTS[dataset]),
        "--config", str(cfg_path),
    ]
    if has_seed_flag:
        cmd += ["--seed", str(SEED)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    print(f"  [LAUNCH] GPU={gpu}  {dataset}/{proxy}  → {log_path.name}")
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc, log_path

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Build job list — interleave datasets so all 3 run in parallel from the start
    # Order: proxy-major, dataset-minor → mad/iemocap, mad/mosei, mad/meld, entropy/...
    jobs_todo = []
    for proxy in PROXIES:
        for ds in DATASETS:
            r = parse_result(ds, proxy)
            if r and r.get("acc") is not None:
                print(f"  [SKIP] {ds}/{proxy} — result exists")
            else:
                jobs_todo.append((ds, proxy))

    print(f"\n  {len(jobs_todo)} jobs to run across {DATASETS} × {PROXIES}\n")
    if not jobs_todo:
        print("  All done."); build_summary(); return

    running = {}    # {(ds,proxy): (proc, log_path, gpu, t_start)}
    done, failed = [], []
    MAX_PARALLEL = 7

    def collect_finished():
        for key in list(running):
            proc, logp, gpu, ts = running[key]
            ret = proc.poll()
            if ret is None: continue
            elapsed = (time.time() - ts) / 60
            ds, proxy = key
            if ret == 0:
                r = parse_result(ds, proxy)
                print(f"  [DONE]  {ds}/{proxy}  GPU={gpu}  {elapsed:.1f}min  "
                      f"acc={r['acc'] if r else '?'}")
                done.append(key)
            else:
                print(f"  [FAIL]  {ds}/{proxy}  GPU={gpu}  exit={ret}")
                failed.append(key)
            del running[key]

    while jobs_todo or running:
        collect_finished()
        gpus = free_gpus()
        while jobs_todo and len(running) < MAX_PARALLEL and gpus:
            ds, proxy = jobs_todo.pop(0)
            gpu = gpus.pop(0)
            proc, logp = launch_job(ds, proxy, gpu)
            running[(ds, proxy)] = (proc, logp, gpu, time.time())
        if running:
            time.sleep(30)

    print(f"\n  Done={len(done)}  Failed={len(failed)}")
    if failed:
        print("  FAILED:", failed)
    build_summary()

def build_summary():
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for proxy in PROXIES:
            r = parse_result(ds, proxy)
            results[ds][proxy] = r

    # Print table
    print("\n" + "="*80)
    print("PROXY ABLATION — Accuracy & F1 (seed 42)")
    print("="*80)
    print(f"\n  {'Proxy':<16} | {'IEMOCAP':^26} | {'CMU-MOSEI':^26} | {'MELD':^26}")
    print(f"  {'':16} | {'Acc':>8} {'MacF1':>8} {'WtF1':>8} | "
          f"{'Acc':>8} {'MacF1':>8} {'WtF1':>8} | "
          f"{'Acc':>8} {'MacF1':>8} {'WtF1':>8}")
    print("  " + "-"*90)
    for proxy in PROXIES:
        row = f"  {proxy:<16} |"
        for ds in DATASETS:
            r = results[ds].get(proxy) or {}
            acc = f"{r['acc']*100:.2f}" if r.get('acc') else "   —  "
            f1  = f"{r['macro_f1']*100:.2f}" if r.get('macro_f1') else "   —  "
            wf1 = f"{r['weighted_f1']*100:.2f}" if r.get('weighted_f1') else "   —  "
            row += f" {acc:>8} {f1:>8} {wf1:>8} |"
        print(row)

    # Save JSON
    out = OUT_DIR / "summary.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n  Results → {out}")

if __name__ == "__main__":
    main()
