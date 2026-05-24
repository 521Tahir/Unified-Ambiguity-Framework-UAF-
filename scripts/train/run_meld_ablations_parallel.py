#!/usr/bin/env python3
"""
Standalone MELD factor-ablation launcher.

Runs 14 missing MELD ablation jobs (5 variants × 3 seeds, minus wo_spk/seed_42
which already exists) in parallel.  Assigns each job to the first GPU that has
>= MIN_FREE MiB free AND is not occupied by any compute process.

Completely independent of the iemocap/mosei scheduler — reads nvidia-smi to
detect GPU occupancy so it never conflicts with already-running jobs.
"""
import os, subprocess, time, socket, sys
from pathlib import Path
from datetime import datetime

PROJ     = Path("/home/tahirahmad/Project_Code/CHADO_EMNLP")
RES      = PROJ / "experiments/results"
CFG_DIR  = PROJ / "configs/meld"
TRAIN    = PROJ / "scripts/train/train_meld.py"
LOG_DIR  = PROJ / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

NUM_GPUS  = 10
MIN_FREE  = 30_000   # MiB — MELD batch=4 uses ~12-15 GB; leave headroom
PORT_BASE = 32000    # use different range from main scheduler (31000-31899)

SEEDS    = [42, 3407, 456]
VARIANTS = ["wo_ctx", "wo_spk", "wo_turn", "wo_modal", "wo_residual"]

# Build list of missing jobs
def missing_jobs():
    jobs = []
    for variant in VARIANTS:
        for seed in SEEDS:
            out_dir = RES / "meld/ablations" / variant / f"seed_{seed}"
            result  = out_dir / "test_results.json"
            if not result.exists():
                jobs.append({
                    "label":   f"MELD/abl_{variant}/s{seed}",
                    "config":  CFG_DIR / f"ablation_{variant}.yaml",
                    "seed":    seed,
                    "out_dir": out_dir,
                })
    return jobs

def free_mib(gpu_idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_idx}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        return int(out.split()[0])
    except Exception:
        return 0

def gpu_pids(gpu_idx):
    """Return set of PIDs using this GPU (any user)."""
    try:
        idx_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, timeout=10
        )
        uuid = None
        for line in idx_out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and int(parts[0]) == gpu_idx:
                uuid = parts[1]
                break
        if uuid is None:
            return set()
        app_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            text=True, timeout=10
        )
        pids = set()
        for line in app_out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[1] == uuid:
                try:
                    pids.add(int(parts[0]))
                except ValueError:
                    pass
        return pids
    except Exception:
        return set()

_port_idx = 0
def next_port():
    global _port_idx
    for _ in range(200):
        p = PORT_BASE + _port_idx
        _port_idx = (_port_idx + 1) % 200
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port in 32000-32199")

def find_free_gpu(busy_gpus):
    for g in range(NUM_GPUS):
        if g in busy_gpus:
            continue
        pids = gpu_pids(g)
        if pids:
            continue          # occupied by any process
        if free_mib(g) >= MIN_FREE:
            return g
    return None

def ts():
    return datetime.now().strftime("%H:%M:%S")

def main():
    jobs   = missing_jobs()
    active = {}   # gpu_idx → {proc, label, out_dir, started}
    done   = []
    failed = []

    print(f"[{ts()}] MELD ablation launcher started")
    print(f"  {len(jobs)} jobs to run:")
    for j in jobs:
        print(f"    {j['label']}")
    print()

    while jobs or active:
        # — collect finished —
        for g in list(active.keys()):
            info = active[g]
            rc = info["proc"].poll()
            if rc is None:
                continue
            elapsed = int(time.time() - info["started"])
            h, rem  = divmod(elapsed, 3600)
            m, s    = divmod(rem, 60)
            tag = "DONE" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{ts()}] [{tag}] GPU={g}  {info['label']}  "
                  f"{h:02d}h{m:02d}m{s:02d}s", flush=True)
            if rc == 0:
                done.append(info["label"])
            else:
                failed.append(info["label"])
            del active[g]

        # — launch new jobs on free GPUs —
        while jobs:
            busy = set(active.keys())
            g = find_free_gpu(busy)
            if g is None:
                break
            job  = jobs.pop(0)
            port = next_port()
            job["out_dir"].mkdir(parents=True, exist_ok=True)
            env  = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log  = open(job["out_dir"] / "train.log", "w")
            cmd  = [
                "torchrun",
                "--nproc_per_node=1",
                f"--master_port={port}",
                str(TRAIN),
                "--config",  str(job["config"]),
                "--seed",    str(job["seed"]),
                "--out-dir", str(job["out_dir"]),
            ]
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            active[g] = {"proc": proc, "label": job["label"],
                         "out_dir": job["out_dir"], "started": time.time()}
            print(f"  [{ts()}] [START] GPU={g}  {job['label']}  port={port}", flush=True)
            time.sleep(2)   # small stagger to avoid simultaneous model downloads

        # — status line every 5 min —
        print(f"  [{ts()}] running={len(active)} queue={len(jobs)} "
              f"done={len(done)} failed={len(failed)}", flush=True)
        if active:
            for g, info in sorted(active.items()):
                ela = int(time.time() - info["started"])
                print(f"    GPU={g}  {info['label']}  "
                      f"{ela//3600:02d}h{(ela%3600)//60:02d}m{ela%60:02d}s")
        time.sleep(300)

    print(f"\n[{ts()}] MELD ablation launcher finished.")
    print(f"  Done  : {len(done)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print("  Failed jobs:")
        for f in failed:
            print(f"    {f}")

if __name__ == "__main__":
    os.chdir(PROJ)
    sys.path.insert(0, str(PROJ))
    main()
