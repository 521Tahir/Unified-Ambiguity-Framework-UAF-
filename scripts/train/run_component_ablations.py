#!/usr/bin/env python3
"""
Component ablation launcher — Table 3 (w/o Causal Factors, 5-epoch fast re-run).

3 jobs total:  3 datasets × 1 variant (use_causal=false, keep Radial+OT+MAD), seed=42, 5 epochs.
Uses all 10 GPUs.  Checks every 30 s so jobs start as soon as a GPU frees.
Port range: 34000-34099 (no conflict with other launchers).
"""
import os, subprocess, time, socket, sys, json
from pathlib import Path
from datetime import datetime

PROJ = Path("/home/tahirahmad/Project_Code/CHADO_EMNLP")
RES  = PROJ / "experiments/results"
CFG  = PROJ / "configs"

TRAIN_IEM  = PROJ / "scripts/train/train_iemocap.py"
TRAIN_MSI  = PROJ / "scripts/train/train_mosei.py"
TRAIN_MELD = PROJ / "scripts/train/train_meld.py"

NUM_GPUS  = 10
MIN_FREE  = 25_000   # MiB — all three models fit well within 48 GB
PORT_BASE = 34000
PORT_RANGE= 100

JOBS = [
    # use_causal=false, 5-epoch fast re-run (new output dirs preserve old results)
    ("IEMOCAP / w/o Causal 5ep", TRAIN_IEM,  CFG/"iemocap/comp_abl_wo_causal_5ep.yaml",
     RES/"iemocap/component_ablations/wo_causal_5ep/seed_42"),
    ("MOSEI   / w/o Causal 5ep", TRAIN_MSI,  CFG/"mosei/comp_abl_wo_causal_5ep.yaml",
     RES/"mosei/component_ablations/wo_causal_5ep/seed_42"),
    ("MELD    / w/o Causal 5ep", TRAIN_MELD, CFG/"meld/comp_abl_wo_causal_5ep.yaml",
     RES/"meld/component_ablations/wo_causal_5ep/seed_42"),
]

def ts():
    return datetime.now().strftime("%H:%M:%S")

def free_mib(g):
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", f"--id={g}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().split()[0])
    except: return 0

def pids_on_gpu(g):
    try:
        idx_out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],
            text=True, timeout=10)
        uuid = next((ln.split(",")[1].strip()
                     for ln in idx_out.strip().splitlines()
                     if ln.split(",")[0].strip() == str(g)), None)
        if not uuid: return set()
        app_out = subprocess.check_output(
            ["nvidia-smi","--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader"], text=True, timeout=10)
        return {int(ln.split(",")[0].strip())
                for ln in app_out.strip().splitlines()
                if ln.split(",")[1].strip() == uuid}
    except: return set()

_pidx = 0
def next_port():
    global _pidx
    for _ in range(PORT_RANGE):
        p = PORT_BASE + _pidx
        _pidx = (_pidx + 1) % PORT_RANGE
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try: s.bind(("", p)); return p
            except OSError: continue
    raise RuntimeError("No free port 34000-34099")

def find_free_gpu(busy):
    for g in range(NUM_GPUS):
        if g in busy: continue
        if free_mib(g) >= MIN_FREE: return g
    return None

def launch(label, script, config, out_dir, gpu):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    port = next_port()
    env  = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log  = open(out_dir / "train.log", "w")
    cmd  = ["torchrun", "--nproc_per_node=1", f"--master_port={port}",
            str(script),
            "--config", str(config), "--seed", "42",
            "--out-dir", str(out_dir)]
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    print(f"  [{ts()}] START  GPU={gpu}  {label}  port={port}", flush=True)
    return proc

def main():
    # Filter already-done jobs
    queue  = [(l, s, c, o) for l, s, c, o in JOBS
              if not Path(o, "test_results.json").exists()]
    active = {}   # gpu → (proc, label, out_dir, t0)
    done, failed = [], []

    print(f"\n[{ts()}] Component Ablation Launcher started")
    print(f"  Total jobs: {len(JOBS)}   Already done: {len(JOBS)-len(queue)}   To run: {len(queue)}")
    for l, _, _, _ in queue:
        print(f"    {l}")
    print()

    while queue or active:
        # collect finished
        for g in list(active.keys()):
            proc, label, out_dir, t0 = active[g]
            rc = proc.poll()
            if rc is None: continue
            ela = int(time.time()-t0)
            tag = "DONE" if rc==0 else f"FAIL(rc={rc})"
            print(f"  [{ts()}] {tag}  GPU={g}  {label}  "
                  f"{ela//3600:02d}h{(ela%3600)//60:02d}m{ela%60:02d}s", flush=True)
            (done if rc==0 else failed).append(label)
            del active[g]

        # re-check if any queued job was already completed by another process
        still_q = []
        for item in queue:
            if Path(item[3], "test_results.json").exists():
                done.append(item[0])
                print(f"  [{ts()}] SKIP(already done)  {item[0]}", flush=True)
            else:
                still_q.append(item)
        queue = still_q

        # launch on free GPUs
        launched = True
        while queue and launched:
            launched = False
            busy = set(active.keys())
            g = find_free_gpu(busy)
            if g is not None:
                label, script, cfg, out_dir = queue.pop(0)
                proc = launch(label, script, cfg, out_dir, g)
                active[g] = (proc, label, out_dir, time.time())
                launched = True
                time.sleep(3)

        # status
        print(f"  [{ts()}]  running={len(active)}  queue={len(queue)}  "
              f"done={len(done)}  failed={len(failed)}", flush=True)
        for g, (_, label, _, t0) in sorted(active.items()):
            ela = int(time.time()-t0)
            # show epoch from log
            log_p = Path(active[g][2]) / "train.log"
            ep = ""
            if log_p.exists():
                for line in open(log_p).readlines()[::-1]:
                    if "Epoch" in line and ("/" in line):
                        ep = "  ep=" + line.strip().split("Epoch")[1].strip().split()[0]
                        break
            print(f"    GPU={g}  {label}{ep}  "
                  f"{ela//3600:02d}h{(ela%3600)//60:02d}m{ela%60:02d}s")
        time.sleep(30)   # check every 30 s for fast GPU pickup

    print(f"\n[{ts()}] All component ablation jobs complete.")
    print(f"  Done:   {len(done)}")
    for x in done:   print(f"    ✓ {x}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for x in failed: print(f"    ✗ {x}")

if __name__ == "__main__":
    os.chdir(PROJ)
    sys.path.insert(0, str(PROJ))
    main()
