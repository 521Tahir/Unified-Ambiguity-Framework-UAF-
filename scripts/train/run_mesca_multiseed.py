#!/usr/bin/env python3
"""
MESCA Multi-Seed Launcher
==========================
Runs MESCA baseline on IEMOCAP, CMU-MOSEI, MELD × seeds [42, 123, 456].
9 total jobs launched in parallel across available GPUs.

Results → experiments/results/multiseed_tables/mesca_multiseed_table.json
Reports: mean ± std for Accuracy and Macro-F1 across 3 seeds per dataset.
"""

import os, sys, json, time, subprocess, yaml
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_BASE = ROOT / "experiments/results"
LOG_DIR  = ROOT / "logs/mesca_multiseed"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["iemocap", "mosei", "meld"]
SEEDS    = [42, 123, 456]

ALLOWED_GPUS = [4, 6, 7, 8, 9]
MIN_FREE_MiB = 25_000          # avoid conflict with wo_causal jobs on GPUs 1,5

BASE_CFGS = {
    "iemocap": ROOT / "configs/iemocap/mesca_iemocap.yaml",
    "mosei":   ROOT / "configs/mosei/mesca_mosei.yaml",
    "meld":    ROOT / "configs/meld/mesca_meld.yaml",
}

TRAIN_SCRIPTS = {
    "iemocap": ROOT / "scripts/train/train_baseline.py",
    "mosei":   ROOT / "scripts/train/train_baseline_mosei.py",
    "meld":    ROOT / "scripts/train/train_baseline.py",
}


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


def make_config(dataset: str, seed: int) -> Path:
    base = yaml.safe_load(open(BASE_CFGS[dataset]))
    base["train"]["seed"] = seed
    out_dir = str(OUT_BASE / dataset / "mesca" / f"seed_{seed}")
    base["logging"]["out_dir"]  = out_dir
    base["logging"]["run_name"] = f"mesca_{dataset}_seed{seed}"
    cfg_path = LOG_DIR / f"cfg_{dataset}_seed{seed}.yaml"
    yaml.dump(base, open(cfg_path, "w"), default_flow_style=False)
    return cfg_path


def parse_result(dataset: str, seed: int):
    res_dir = OUT_BASE / dataset / "mesca" / f"seed_{seed}"
    for cand in [res_dir / "test_results.json", res_dir / "results.json"]:
        if cand.exists():
            d = json.load(open(cand))
            acc = d.get("accuracy", d.get("acc", d.get("wacc")))
            f1  = d.get("macro_f1", d.get("f1"))
            wf1 = d.get("weighted_f1", d.get("w_f1"))
            return {"acc": acc, "macro_f1": f1, "weighted_f1": wf1}
    return None


def launch_job(dataset, seed, gpu):
    cfg_path = make_config(dataset, seed)
    log_path = LOG_DIR / f"{dataset}_seed{seed}.log"
    cmd = ["python3", "-u", str(TRAIN_SCRIPTS[dataset]),
           "--config", str(cfg_path), "--seed", str(seed)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]   = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    print(f"  [LAUNCH] GPU={gpu}  {dataset}/seed{seed}  → {log_path.name}")
    fh   = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc, log_path


def build_summary():
    table = {}
    print("\n" + "="*72)
    print("MESCA — Multi-Seed Results  (seeds: 42, 123, 456)")
    print("="*72)
    print(f"\n  {'Dataset':<12} {'Acc (mean±std)':>20} {'MacF1 (mean±std)':>22}")
    print("  " + "-"*56)

    for ds in DATASETS:
        results = [parse_result(ds, s) for s in SEEDS]
        accs  = [r["acc"]      for r in results if r and r.get("acc")]
        mf1s  = [r["macro_f1"] for r in results if r and r.get("macro_f1")]
        wf1s  = [r["weighted_f1"] for r in results if r and r.get("weighted_f1")]

        acc_str = f"{mean(accs)*100:.2f}±{stdev(accs)*100:.2f}%" if len(accs) > 1 else (f"{accs[0]*100:.2f}%" if accs else "—")
        mf1_str = f"{mean(mf1s)*100:.2f}±{stdev(mf1s)*100:.2f}%" if len(mf1s) > 1 else (f"{mf1s[0]*100:.2f}%" if mf1s else "—")
        print(f"  {ds.upper():<12} {acc_str:>20} {mf1_str:>22}")

        table[ds] = {
            "seeds": {str(s): parse_result(ds, s) for s in SEEDS},
            "acc_mean":  mean(accs)  if accs  else None,
            "acc_std":   stdev(accs) if len(accs) > 1 else 0,
            "mf1_mean":  mean(mf1s)  if mf1s  else None,
            "mf1_std":   stdev(mf1s) if len(mf1s) > 1 else 0,
            "wf1_mean":  mean(wf1s)  if wf1s  else None,
            "wf1_std":   stdev(wf1s) if len(wf1s) > 1 else 0,
        }

    out = OUT_BASE / "multiseed_tables" / "mesca_multiseed_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(out, "w"), indent=2)
    print(f"\n  Results → {out}")


def main():
    # Build job list — interleave datasets so all 3 start immediately
    jobs_todo = []
    for seed in SEEDS:
        for ds in DATASETS:
            r = parse_result(ds, seed)
            if r and r.get("acc") is not None:
                print(f"  [SKIP] {ds}/seed{seed} — result exists")
            else:
                jobs_todo.append((ds, seed))

    print(f"\n  {len(jobs_todo)} jobs  ({len(DATASETS)} datasets × {len(SEEDS)} seeds)\n")
    if not jobs_todo:
        print("  All done.")
        build_summary()
        return

    running = {}   # {(ds, seed): (proc, logp, gpu, t_start)}
    done, failed = [], []
    MAX_PARALLEL = 9

    def collect_finished():
        for key in list(running):
            proc, logp, gpu, ts = running[key]
            ret = proc.poll()
            if ret is None:
                continue
            elapsed = (time.time() - ts) / 60
            ds, seed = key
            if ret == 0:
                r = parse_result(ds, seed)
                print(f"  [DONE]  {ds}/seed{seed}  GPU={gpu}  {elapsed:.1f}min  "
                      f"acc={r['acc'] if r else '?'}")
                done.append(key)
            else:
                print(f"  [FAIL]  {ds}/seed{seed}  GPU={gpu}  exit={ret}")
                failed.append(key)
            del running[key]

    while jobs_todo or running:
        collect_finished()
        gpus = free_gpus()
        while jobs_todo and len(running) < MAX_PARALLEL and gpus:
            ds, seed = jobs_todo.pop(0)
            gpu = gpus.pop(0)
            proc, logp = launch_job(ds, seed, gpu)
            running[(ds, seed)] = (proc, logp, gpu, time.time())
        if running:
            time.sleep(30)

    print(f"\n  Done={len(done)}  Failed={len(failed)}")
    if failed:
        print("  FAILED:", failed)
    build_summary()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    main()
