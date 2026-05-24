#!/usr/bin/env python3
"""
GPU Scheduler — assigns training jobs to GPUs without conflicts.

Design:
- 1 job per GPU at any time; no two jobs share a GPU
- DDP jobs use 2 GPUs; baseline jobs use 1 GPU
- Detects already-running external processes and waits for them
- Skips runs with existing test_results.json
- Live status display every 2 minutes
- Port range: 31000-31899
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from datetime import datetime

PROJ = Path("/home/tahirahmad/Project_Code/CHADO_EMNLP")
RES  = PROJ / "experiments/results"

# ─── Training scripts ─────────────────────────────────────────────────────────
TRAIN_IEM        = PROJ / "scripts/train/train_iemocap.py"
TRAIN_MELD       = PROJ / "scripts/train/train_meld.py"
TRAIN_MOSEI      = PROJ / "scripts/train/train_mosei.py"
TRAIN_BASE       = PROJ / "scripts/train/train_baseline.py"
TRAIN_BASE_MOSEI = PROJ / "scripts/train/train_baseline_mosei.py"

# ─── Config dirs ──────────────────────────────────────────────────────────────
IC = PROJ / "configs/iemocap"
ML = PROJ / "configs/meld"
MS = PROJ / "configs/mosei"


# ─── Job ──────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    label:    str
    script:   Path
    config:   Path
    seed:     int
    out_dir:  Path
    num_gpus: int           # 1 or 2
    gpus:     List[int] = field(default_factory=list)
    port:     int        = 0
    proc:     Optional[object] = None
    started:  Optional[float]  = None
    rc:       Optional[int]    = None

    def is_done(self):
        return (self.out_dir / "test_results.json").exists()

    def elapsed(self):
        if self.started is None:
            return ""
        s = int(time.time() - self.started)
        return f"{s//3600:02d}h{(s%3600)//60:02d}m{s%60:02d}s"


# ─── Build job queue ──────────────────────────────────────────────────────────
def build_queue(skip_out_dirs: Set[str], only: Optional[List[str]] = None) -> List[Job]:
    """Build queue. only = list of datasets to include, e.g. ['iemocap','mosei']."""
    jobs: List[Job] = []

    def add(dataset, label, script, config, seed, out_dir, num_gpus):
        if only and dataset not in only:
            return
        j = Job(label=label, script=script, config=config,
                seed=seed, out_dir=Path(out_dir), num_gpus=num_gpus)
        if j.is_done():
            return
        if str(j.out_dir) in skip_out_dirs:
            return
        jobs.append(j)

    # ── IEMOCAP AR-GOT (single-GPU: 10 concurrent jobs = all GPUs at 100%) ────
    for s in [3407, 456]:
        add("iemocap", f"IC/argot/s{s}", TRAIN_IEM, IC/"chado_iemocap_final.yaml",
            s, RES/"iemocap/argot_ms"/f"seed_{s}", 1)

    # ── IEMOCAP baselines (1-GPU) ─────────────────────────────────────────────
    for name, cfg in [
        ("bpmult",  "bpmult_iemocap.yaml"),
        ("ctnet",   "ctnet_iemocap.yaml"),
        ("mult",    "mult_iemocap.yaml"),
        ("mmdfn",   "mmdfn_iemocap.yaml"),
        ("lflstm",  "lflstm_iemocap.yaml"),
        ("aerllm",  "aerllm_iemocap.yaml"),
        ("emoclip", "emoclip_iemocap.yaml"),
        ("ovmer",   "ovmer_iemocap.yaml"),
    ]:
        for s in [42, 3407, 456]:
            add("iemocap", f"IC/{name}/s{s}", TRAIN_BASE, IC/cfg,
                s, RES/"iemocap"/f"{name}_ms/seed_{s}", 1)

    # ── IEMOCAP ablations (single-GPU) ───────────────────────────────────────
    for abl in ["wo_ctx", "wo_spk", "wo_turn", "wo_modal", "wo_residual"]:
        for s in [42, 3407, 456]:
            add("iemocap", f"IC/abl_{abl}/s{s}", TRAIN_IEM, IC/f"ablation_{abl}.yaml",
                s, RES/"iemocap/ablations"/abl/f"seed_{s}", 1)

    # ── MELD AR-GOT (single-GPU) ─────────────────────────────────────────────
    for s in [3407, 456]:
        add("meld", f"ML/argot/s{s}", TRAIN_MELD, ML/"chado_meld_final.yaml",
            s, RES/"meld/argot_ms"/f"seed_{s}", 1)

    # ── MELD baselines (1-GPU) ────────────────────────────────────────────────
    for name, cfg in [
        ("bpmult",  "bpmult_meld.yaml"),
        ("ctnet",   "ctnet_meld.yaml"),
        ("mult",    "mult_meld.yaml"),
        ("mmdfn",   "mmdfn_meld.yaml"),
        ("lflstm",  "lflstm_meld.yaml"),
        ("aerllm",  "aerllm_meld.yaml"),
        ("emoclip", "emoclip_meld.yaml"),
        ("ovmer",   "ovmer_meld.yaml"),
    ]:
        for s in [42, 3407, 456]:
            add("meld", f"ML/{name}/s{s}", TRAIN_BASE, ML/cfg,
                s, RES/"meld"/f"{name}_ms/seed_{s}", 1)

    # ── MELD ablations (single-GPU) ───────────────────────────────────────────
    for abl in ["wo_ctx", "wo_spk", "wo_turn", "wo_modal", "wo_residual"]:
        for s in [42, 3407, 456]:
            add("meld", f"ML/abl_{abl}/s{s}", TRAIN_MELD, ML/f"ablation_{abl}.yaml",
                s, RES/"meld/ablations"/abl/f"seed_{s}", 1)

    # ── MOSEI AR-GOT (single-GPU, seed_456 missing) ──────────────────────────
    add("mosei", "MS/argot/s456", TRAIN_MOSEI, MS/"chado_mosei_best.yaml",
        456, RES/"mosei/argot_ms/seed_456", 1)

    # ── MOSEI baselines (1-GPU) ───────────────────────────────────────────────
    for s in [42, 456]:
        add("mosei", f"MS/ovmer/s{s}", TRAIN_BASE_MOSEI, MS/"ovmer_mosei.yaml",
            s, RES/"mosei/ovmer_ms"/f"seed_{s}", 1)

    # ── MOSEI ablations (single-GPU, seed_42 missing) ────────────────────────
    for abl in ["wo_ctx", "wo_modal", "wo_residual", "wo_turn"]:
        add("mosei", f"MS/abl_{abl}/s42", TRAIN_MOSEI, MS/f"ablation_{abl}.yaml",
            42, RES/"mosei/ablations"/abl/"seed_42", 1)

    return jobs


# ─── Detect all processes using GPUs (any user) ───────────────────────────────
def detect_external_processes():
    """
    Returns:
        skip_out_dirs: out_dirs of OUR running torchrun jobs (to skip in queue)
        gpu_to_pids:   Dict[gpu_id, Set[pid]] — ALL compute pids on each GPU
    """
    skip_out_dirs: Set[str] = set()
    gpu_to_pids: Dict[int, Set[int]] = {}

    # 1. Our torchrun jobs → extract CUDA_VISIBLE_DEVICES + out-dir
    try:
        r = subprocess.run(["ps", "auxww"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "torchrun" not in line or "--out-dir" not in line:
                continue
            parts = line.split()
            try:
                pid = int(parts[1])
            except (IndexError, ValueError):
                continue
            gpus: List[int] = []
            try:
                env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
                for token in env_bytes.split(b"\x00"):
                    if token.startswith(b"CUDA_VISIBLE_DEVICES="):
                        cvd = token[len(b"CUDA_VISIBLE_DEVICES="):].decode(errors="ignore")
                        gpus = [int(g.strip()) for g in cvd.split(",") if g.strip().isdigit()]
                        break
            except Exception:
                pass
            out_dir = ""
            for i, p in enumerate(parts):
                if p == "--out-dir" and i + 1 < len(parts):
                    out_dir = parts[i + 1]
                    break
            if out_dir:
                skip_out_dirs.add(out_dir)
            for g in gpus:
                gpu_to_pids.setdefault(g, set()).add(pid)
    except Exception:
        pass

    # 2. ALL compute processes via nvidia-smi (catches other users' jobs)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader"],
            text=True, timeout=15,
        )
        # Also get GPU index → uuid mapping
        idx_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, timeout=15,
        )
        uuid_to_idx: Dict[str, int] = {}
        for ln in idx_out.strip().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) == 2:
                try:
                    uuid_to_idx[parts[1]] = int(parts[0])
                except ValueError:
                    pass
        for ln in out.strip().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            gpu_uuid = parts[1]
            gpu_idx = uuid_to_idx.get(gpu_uuid)
            if gpu_idx is not None:
                gpu_to_pids.setdefault(gpu_idx, set()).add(pid)
    except Exception:
        pass

    return skip_out_dirs, gpu_to_pids


# ─── Scheduler ────────────────────────────────────────────────────────────────
class GPUScheduler:
    NUM_GPUS    = 10
    MIN_FREE    = 40_000   # MiB — ensures ycyoon's ~9GB processes don't cause OOM
    PORT_START  = 31000
    PORT_RANGE  = 900

    def __init__(self):
        self.queue:      List[Job]        = []
        self.active:     List[Job]        = []
        self.done:       List[Job]        = []
        self.failed:     List[Job]        = []
        self.busy_gpus:  Set[int]         = set()
        self.gpu_owners: Dict[int, Set[int]] = {}   # external PIDs per GPU
        self._port_idx = 0
        self._lock     = threading.Lock()

    # ── Port allocation — skips ports already in use ──────────────────────────
    def _next_port(self) -> int:
        import socket
        with self._lock:
            for _ in range(self.PORT_RANGE):
                p = self.PORT_START + self._port_idx
                self._port_idx = (self._port_idx + 1) % self.PORT_RANGE
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind(("", p))
                        return p
                    except OSError:
                        continue
            raise RuntimeError("No free port found in range 31000-31899")

    # ── Init external-process tracking ────────────────────────────────────────
    def init_external(self, gpu_to_pids: Dict[int, Set[int]]):
        for gpu, pids in gpu_to_pids.items():
            self.gpu_owners[gpu] = set(pids)
            self.busy_gpus.add(gpu)
        if self.busy_gpus:
            print(f"  External jobs holding GPUs: {sorted(self.busy_gpus)}", flush=True)

    # ── GPU free memory ───────────────────────────────────────────────────────
    @staticmethod
    def _free_mib(gpu: int) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits", f"-i={gpu}"],
                text=True, timeout=10,
            ).strip()
            return int(out.split()[0])
        except Exception:
            return 0

    # ── Find free GPUs for a job ───────────────────────────────────────────────
    def _find_free_gpus(self, need: int) -> Optional[List[int]]:
        candidates = [
            g for g in range(self.NUM_GPUS)
            if g not in self.busy_gpus and self._free_mib(g) >= self.MIN_FREE
        ]
        if len(candidates) >= need:
            return candidates[:need]
        return None

    # ── Launch one job ─────────────────────────────────────────────────────────
    def _launch(self, job: Job, gpus: List[int]):
        job.gpus    = gpus
        job.port    = self._next_port()
        job.started = time.time()
        job.out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        cmd = [
            "torchrun",
            f"--nproc_per_node={job.num_gpus}",
            f"--master_port={job.port}",
            str(job.script),
            "--config",  str(job.config),
            "--seed",    str(job.seed),
            "--out-dir", str(job.out_dir),
        ]
        log_file = open(job.out_dir / "train.log", "w")
        job.proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)

        with self._lock:
            self.busy_gpus.update(gpus)
            self.active.append(job)

        gpu_str = ",".join(str(g) for g in gpus)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [START {ts}] GPU={gpu_str}  {job.label}  port={job.port}", flush=True)

    # ── Collect finished jobs (scheduler + external) ──────────────────────────
    def _collect_finished(self):
        # Scheduler-managed jobs
        still_active = []
        for job in self.active:
            rc = job.proc.poll()
            if rc is None:
                still_active.append(job)
                continue
            job.rc = rc
            ts = datetime.now().strftime("%H:%M:%S")
            gpu_str = ",".join(str(g) for g in job.gpus)
            if rc == 0:
                self.done.append(job)
                print(f"  [DONE  {ts}] GPU={gpu_str}  {job.label}  {job.elapsed()}", flush=True)
            else:
                self.failed.append(job)
                print(f"  [FAIL  {ts}] GPU={gpu_str}  {job.label}  rc={rc}  {job.elapsed()}", flush=True)
            with self._lock:
                for g in job.gpus:
                    self.busy_gpus.discard(g)
        self.active = still_active

        # External PIDs — check which died and free their GPUs
        for gpu in list(self.gpu_owners.keys()):
            alive = set()
            for pid in self.gpu_owners[gpu]:
                try:
                    os.kill(pid, 0)
                    alive.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass
            died = self.gpu_owners[gpu] - alive
            if died:
                self.gpu_owners[gpu] = alive
                if not alive:   # All processes on this GPU finished
                    self.busy_gpus.discard(gpu)
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [GPU{gpu} FREE {ts}] external job done", flush=True)

    # ── Status display ─────────────────────────────────────────────────────────
    def _print_status(self):
        ts = datetime.now().strftime("%H:%M:%S")
        busy_ext = {g for g, pids in self.gpu_owners.items() if pids}
        print(
            f"\n[{ts}]  RUNNING={len(self.active)}  QUEUE={len(self.queue)}"
            f"  DONE={len(self.done)}  FAIL={len(self.failed)}"
            f"  ext_busy={sorted(busy_ext)}",
            flush=True,
        )
        for j in self.active:
            gpu_str = ",".join(str(g) for g in j.gpus)
            print(f"    GPU={gpu_str}  {j.label}  {j.elapsed()}", flush=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print(f"\n{'='*70}")
        print(f"  GPU SCHEDULER  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  GPUs 0-{self.NUM_GPUS-1}  |  min_free={self.MIN_FREE} MiB  |  {len(self.queue)} jobs queued")
        print(f"{'='*70}", flush=True)

        last_status = time.time()

        while self.queue or self.active or any(self.gpu_owners[g] for g in self.gpu_owners):
            self._collect_finished()

            # Try to fill every free GPU slot
            progress = True
            while progress and self.queue:
                progress = False
                for i, job in enumerate(self.queue):
                    gpus = self._find_free_gpus(job.num_gpus)
                    if gpus is not None:
                        self.queue.pop(i)
                        self._launch(job, gpus)
                        progress = True
                        break   # restart scan after updating busy_gpus

            if time.time() - last_status >= 120:
                self._print_status()
                last_status = time.time()

            time.sleep(20)

        # Final report
        self._print_status()
        print(f"\n{'='*70}")
        print(f"  SCHEDULER COMPLETE  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Done={len(self.done)}  Fail={len(self.failed)}")
        if self.failed:
            print("  Failed jobs:")
            for j in self.failed:
                print(f"    rc={j.rc}  {j.label}")
        print(f"{'='*70}\n", flush=True)

        # Compute result tables
        print("Computing result tables...", flush=True)
        for ds in ["iemocap", "meld", "mosei"]:
            subprocess.run(
                ["python3", str(PROJ / "scripts/eval/compute_multiseed_table.py"),
                 "--dataset", ds],
            )


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+",
                        choices=["iemocap", "meld", "mosei"],
                        help="Restrict to specific datasets (default: all)")
    args = parser.parse_args()
    only = args.only  # None = all datasets

    print("Detecting running training processes...", flush=True)
    skip_out_dirs, gpu_to_pids = detect_external_processes()

    print(f"  Active out_dirs (will skip in queue): {len(skip_out_dirs)}")
    for d in sorted(skip_out_dirs):
        print(f"    {d.replace(str(RES)+'/', '')}", flush=True)

    print(f"\nBuilding job queue (datasets={only or 'all'})...", flush=True)
    queue = build_queue(skip_out_dirs, only=only)
    print(f"  {len(queue)} jobs to run\n", flush=True)

    sched = GPUScheduler()
    sched.queue = queue
    sched.init_external(gpu_to_pids)

    try:
        sched.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Jobs still running in background.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
