#!/usr/bin/env python3
"""
IEMOCAP CHADO 3-seed launcher with GPU free-memory check, automatic free-port selection,
and final mean ± std reporting.

This script does not modify train_iemocap.py. It creates one seed-specific YAML config per run,
launches the existing DDP training script with torchrun, and summarizes all completed seeds.

Default seeds: 42, 3407, 2024
Default GPUs must be provided, e.g. --gpus 5,6,7,8,9
"""

import argparse
import copy
import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_BASE_CONFIG = "/home/tahirahmad/Project_Code/CHADO_EMNLP/configs/iemocap/chado_iemocap.yaml"
DEFAULT_TRAIN_SCRIPT = "/home/tahirahmad/Project_Code/CHADO_EMNLP/scripts/train/train_iemocap.py"
DEFAULT_OUT_ROOT = "/home/tahirahmad/Project_Code/CHADO_EMNLP/experiments/results/iemocap/chado_3seed_target"
DEFAULT_SEEDS = "42,3407,2024"
DEFAULT_MASTER_PORT = 31000

PREFERRED_METRICS = [
    "best_val_acc",
    "best_val_macro_f1",
    "best_val_weighted_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_macro_precision",
    "test_macro_recall",
]


def parse_seed_list(seed_text: str) -> List[int]:
    seeds = [int(x.strip()) for x in seed_text.replace(",", " ").split() if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, int(port))) != 0


def next_free_port(start_port: int, used_ports: set[int]) -> int:
    port = int(start_port)
    while port in used_ports or not is_port_free(port):
        port += 1
    used_ports.add(port)
    return port


def get_gpu_free_memory_gb(gpu_ids: List[str]) -> Dict[str, float]:
    """Return free memory in GB for physical GPU ids using nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        print(f"[gpu-check] Could not run nvidia-smi; skipping GPU memory check. Error: {exc}")
        return {}

    free_mb: Dict[str, float] = {}
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        idx, mem_mb = parts
        try:
            free_mb[idx] = float(mem_mb) / 1024.0
        except ValueError:
            continue

    return {gid: free_mb.get(gid, -1.0) for gid in gpu_ids}


def assert_gpus_have_memory(gpu_ids: List[str], min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    free = get_gpu_free_memory_gb(gpu_ids)
    if not free:
        return

    bad = {gid: gb for gid, gb in free.items() if gb < min_free_gb}
    print("[gpu-check] free memory:")
    for gid, gb in free.items():
        print(f"  GPU {gid}: {gb:.2f} GB free")

    if bad:
        msg = ", ".join(f"GPU {gid}: {gb:.2f} GB" for gid, gb in bad.items())
        raise RuntimeError(
            f"Selected GPU(s) do not have the required free memory >= {min_free_gb:.1f} GB. "
            f"Busy GPUs: {msg}. Clean old jobs or choose free GPUs."
        )


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warning] Could not read {path}: {exc}")
        return None


def max_numeric(values: Any) -> Optional[float]:
    if not isinstance(values, list):
        return None
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return max(nums) if nums else None


def collect_seed_result(seed: int, seed_dir: Path) -> Optional[Dict[str, Any]]:
    test_path = seed_dir / "test_results.json"
    history_path = seed_dir / "history.json"
    test = load_json(test_path)
    if not test:
        return None

    row: Dict[str, Any] = {"seed": seed, "out_dir": str(seed_dir)}
    for key, value in test.items():
        if isinstance(value, (int, float)):
            row[f"test_{key}"] = float(value)

    history = load_json(history_path)
    if history:
        mapping = {
            "val_acc": "best_val_acc",
            "val_accuracy": "best_val_acc",
            "val_macro_f1": "best_val_macro_f1",
            "val_weighted_f1": "best_val_weighted_f1",
        }
        for src, dst in mapping.items():
            value = max_numeric(history.get(src))
            if value is not None:
                row[dst] = value

    return row


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({k for row in rows for k in row.keys() if k not in {"seed", "out_dir"}})
    summary: Dict[str, Any] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not vals:
            continue
        summary[key] = {
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return summary


def format_mean_std(summary: Dict[str, Any], digits: int = 4) -> str:
    keys = [k for k in PREFERRED_METRICS if k in summary]
    keys += [k for k in sorted(summary.keys()) if k not in keys]

    lines = []
    lines.append("Metric                         Mean ± Std        Values")
    lines.append("-" * 80)
    for key in keys:
        item = summary[key]
        vals = ", ".join(f"{v:.{digits}f}" for v in item["values"])
        lines.append(f"{key:<30} {item['mean']:.{digits}f} ± {item['std']:.{digits}f}   [{vals}]")
    return "\n".join(lines)


def write_summary_files(out_root: Path, seeds: List[int], rows: List[Dict[str, Any]], target_accuracy: float, target_f1: float) -> Dict[str, Any]:
    summary_stats = summarize_rows(rows)

    ranked_by_best_val = sorted(
        rows,
        key=lambda r: r.get("best_val_macro_f1", r.get("test_macro_f1", -1.0)),
        reverse=True,
    )
    ranked_by_test_macro = sorted(rows, key=lambda r: r.get("test_macro_f1", -1.0), reverse=True)
    ranked_by_test_acc = sorted(rows, key=lambda r: r.get("test_accuracy", -1.0), reverse=True)

    target_status = {
        "target_accuracy": target_accuracy,
        "target_macro_f1": target_f1,
        "mean_accuracy_reached": summary_stats.get("test_accuracy", {}).get("mean", -1.0) >= target_accuracy,
        "mean_macro_f1_reached": summary_stats.get("test_macro_f1", {}).get("mean", -1.0) >= target_f1,
        "best_seed_accuracy_reached": bool(rows) and max(r.get("test_accuracy", -1.0) for r in rows) >= target_accuracy,
        "best_seed_macro_f1_reached": bool(rows) and max(r.get("test_macro_f1", -1.0) for r in rows) >= target_f1,
    }

    payload = {
        "seeds_requested": seeds,
        "num_completed": len(rows),
        "results": rows,
        "summary": summary_stats,
        "ranked_by_best_val_macro_f1": ranked_by_best_val,
        "ranked_by_test_macro_f1": ranked_by_test_macro,
        "ranked_by_test_accuracy": ranked_by_test_acc,
        "target_status": target_status,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "multiseed_summary.json"
    txt_path = out_root / "multiseed_mean_std.txt"
    csv_path = out_root / "multiseed_mean_std.csv"

    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    table = format_mean_std(summary_stats)
    target_lines = [
        "",
        "Target check",
        "-" * 80,
        f"Target test accuracy >= {target_accuracy:.4f}: mean reached = {target_status['mean_accuracy_reached']}, best-seed reached = {target_status['best_seed_accuracy_reached']}",
        f"Target test macro-F1 >= {target_f1:.4f}: mean reached = {target_status['mean_macro_f1_reached']}, best-seed reached = {target_status['best_seed_macro_f1_reached']}",
    ]
    txt_path.write_text(table + "\n" + "\n".join(target_lines) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "mean_pm_std", "values"])
        for key, item in summary_stats.items():
            writer.writerow([
                key,
                item["mean"],
                item["std"],
                f"{item['mean']:.4f} ± {item['std']:.4f}",
                json.dumps(item["values"]),
            ])

    print("\n" + "=" * 100)
    print(f"[multi-seed] completed {len(rows)}/{len(seeds)} seeds")
    print(f"[multi-seed] summary json: {json_path}")
    print(f"[multi-seed] mean/std txt: {txt_path}")
    print(f"[multi-seed] mean/std csv: {csv_path}")
    print("=" * 100)
    print(table)
    print("\n" + "\n".join(target_lines))

    if ranked_by_best_val:
        best = ranked_by_best_val[0]
        print(
            f"\n[multi-seed] best validation seed: {best['seed']} | "
            f"best_val_macro_f1={best.get('best_val_macro_f1')} | "
            f"test_accuracy={best.get('test_accuracy')} | test_macro_f1={best.get('test_macro_f1')}"
        )
    print("=" * 100 + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    p.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids, e.g. 5,6,7,8,9")
    p.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    p.add_argument("--no-auto-port", action="store_true", help="Disable automatic free-port selection")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--nproc-per-node", type=int, default=None)
    p.add_argument("--resume-existing", action="store_true", help="Skip a seed if test_results.json already exists")
    p.add_argument("--continue-on-error", action="store_true", help="Continue to next seed if one seed fails")
    p.add_argument("--min-free-gb", type=float, default=35.0, help="Minimum free GPU memory required before each seed. Use 0 to disable.")
    p.add_argument("--target-accuracy", type=float, default=0.80)
    p.add_argument("--target-f1", type=float, default=0.80)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra args passed after train script --config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seeds = parse_seed_list(args.seeds)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    nproc = args.nproc_per_node or len(gpu_ids)

    base_config = Path(args.base_config)
    train_script = Path(args.train_script)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with base_config.open("r") as f:
        base_cfg = yaml.safe_load(f)

    if "train" not in base_cfg:
        raise KeyError("Base config must contain a train section")
    if "logging" not in base_cfg:
        base_cfg["logging"] = {}

    rows: List[Dict[str, Any]] = []
    used_ports: set[int] = set()

    for idx, seed in enumerate(seeds):
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        test_path = seed_dir / "test_results.json"

        if args.resume_existing and test_path.exists():
            print(f"[multi-seed] seed={seed} already has test_results.json; skipping training.")
            row = collect_seed_result(seed, seed_dir)
            if row:
                rows.append(row)
            continue

        assert_gpus_have_memory(gpu_ids, args.min_free_gb)

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("train", {})["seed"] = int(seed)
        cfg.setdefault("logging", {})["run_name"] = f"chado_iemocap_seed_{seed}"
        cfg.setdefault("logging", {})["out_dir"] = str(seed_dir)

        seed_cfg_path = seed_dir / "chado_iemocap_seed.yaml"
        with seed_cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        if args.no_auto_port:
            port = args.master_port + idx
        else:
            port = next_free_port(args.master_port + idx, used_ports)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={port}",
            str(train_script),
            "--config",
            str(seed_cfg_path),
        ]
        if args.extra:
            cmd.extend(args.extra)

        print("\n" + "=" * 100)
        print(f"[multi-seed] seed: {seed}")
        print(f"[multi-seed] config: {seed_cfg_path}")
        print(f"[multi-seed] output: {seed_dir}")
        print(f"[multi-seed] CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
        print(f"[multi-seed] nproc_per_node: {nproc}")
        print(f"[multi-seed] master_port: {port}")
        print(f"[multi-seed] command: {' '.join(cmd)}")
        print("=" * 100 + "\n")

        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            print(f"[multi-seed] seed={seed} failed with exit code {exc.returncode}")
            if not args.continue_on_error:
                raise
            continue

        row = collect_seed_result(seed, seed_dir)
        if row:
            rows.append(row)
        else:
            print(f"[warning] seed={seed} finished but {test_path} was not found/readable")

    # Also include completed seeds that were not appended due to earlier failed/partial resume behavior.
    seen = {int(r["seed"]) for r in rows if "seed" in r}
    for seed in seeds:
        if seed in seen:
            continue
        row = collect_seed_result(seed, out_root / f"seed_{seed}")
        if row:
            rows.append(row)

    if not rows:
        print("[multi-seed] No completed seed results found. No mean/std can be computed.", file=sys.stderr)
        return

    write_summary_files(out_root, seeds, rows, args.target_accuracy, args.target_f1)


if __name__ == "__main__":
    main()
