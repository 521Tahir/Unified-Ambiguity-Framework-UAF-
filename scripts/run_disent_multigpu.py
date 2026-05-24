#!/usr/bin/env python3
"""
Multi-GPU launcher for CHADO disentanglement training.
IEMOCAP → GPUs 1,6,7,8  (4-GPU DDP via torchrun)
MELD    → GPUs 5,9       (2-GPU DDP via torchrun)
MOSEI   → already running on GPU4, just poll for checkpoint.
After all 3 best.pt exist, runs extraction+heatmap (Phase 2-4).
"""
import os, sys, subprocess, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LOG_DIR = ROOT / "logs/analysis"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CKPT = {
    "iemocap": ROOT / "experiments/results/iemocap/chado_disent/best.pt",
    "mosei":   ROOT / "experiments/results/mosei/chado_disent/best.pt",
    "meld":    ROOT / "experiments/results/meld/chado_disent/best.pt",
}

IEMOCAP_GPUS = "1,6,7,8"
MELD_GPUS    = "5,9"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def make_env(**extra):
    e = os.environ.copy()
    e["TOKENIZERS_PARALLELISM"]            = "false"
    e["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    e["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    e.update(extra)
    return e


def launch(name, gpus, nproc, script, config, out_dir, port, seed=None):
    ckpt = CKPT[name]
    if ckpt.exists():
        log(f"  [{name}] checkpoint exists → skip training")
        return None
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"disent_train_{name}.log"
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        str(script),
        "--config", str(config),
        "--out-dir", str(out_dir),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    env = make_env(CUDA_VISIBLE_DEVICES=gpus)
    log(f"  [{name}] LAUNCH  GPUs={gpus} ({nproc}-GPU DDP)  port={port}  log={log_path.name}")
    fh = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)


def main():
    log("=" * 68)
    log("CHADO Disentanglement — Multi-GPU Relaunch")
    log("  IEMOCAP: 4 GPUs (1,6,7,8)  |  MELD: 2 GPUs (5,9)")
    log("  MOSEI:   GPU4 already running — polling for checkpoint")
    log("=" * 68)

    procs = {}

    # IEMOCAP — 4 GPUs
    p_i = launch(
        "iemocap", IEMOCAP_GPUS, 4,
        ROOT / "scripts/train/train_iemocap.py",
        ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
        ROOT / "experiments/results/iemocap/chado_disent",
        port=29701, seed=42,
    )
    if p_i:
        procs["iemocap"] = (p_i, time.time())

    time.sleep(6)   # stagger launch

    # MELD — 2 GPUs
    p_m = launch(
        "meld", MELD_GPUS, 2,
        ROOT / "scripts/train/train_meld.py",
        ROOT / "configs/meld/chado_meld_final.yaml",
        ROOT / "experiments/results/meld/chado_disent",
        port=29702, seed=42,
    )
    if p_m:
        procs["meld"] = (p_m, time.time())

    log(f"\n  [mosei] Waiting for MOSEI checkpoint (GPU4 running independently)...")

    # ── Poll until all 3 checkpoints exist ───────────────────────────────────
    log("\n=== Monitoring ===")
    poll_interval = 120   # seconds
    while True:
        done = {ds: CKPT[ds].exists() for ds in CKPT}

        # Check managed processes for unexpected failure
        for name, (proc, t0) in list(procs.items()):
            ret = proc.poll()
            if ret is not None:
                ela = (time.time() - t0) / 60
                if ret == 0 or CKPT[name].exists():
                    log(f"  [{name}] DONE  {ela:.1f}min")
                else:
                    log(f"  [{name}] FAILED (exit={ret})  {ela:.1f}min  — check logs/analysis/disent_train_{name}.log")
                del procs[name]

        status = " | ".join(
            f"{ds}: {'✓' if ok else 'running'}" for ds, ok in done.items()
        )
        log(f"  {status}")

        if all(done.values()):
            log("\n  All 3 checkpoints ready!")
            break

        time.sleep(poll_interval)

    # ── Phase 2-4: extraction + heatmap ──────────────────────────────────────
    log("\n=== Phase 2-4: Extract factors + generate heatmaps ===")
    log("  (Phase 1 auto-skipped — all best.pt exist)")

    heatmap_log = ROOT / "logs/disentanglement_heatmap.log"
    cmd = ["python3", "-u", str(ROOT / "scripts/analysis/disentanglement_heatmap.py")]
    # All GPUs free now; script picks one automatically
    env = make_env()
    with open(heatmap_log, "a") as fh:
        ret = subprocess.call(cmd, env=env, stdout=fh, stderr=fh)

    if ret == 0:
        log("=" * 68)
        log("DONE — heatmaps in experiments/figures/disentanglement/")
        log("=" * 68)
    else:
        log(f"FAILED (exit={ret}) — check {heatmap_log}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
