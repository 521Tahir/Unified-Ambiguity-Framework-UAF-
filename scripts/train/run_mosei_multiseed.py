#!/usr/bin/env python3
"""
Run CHADO CMU-MOSEI training for multiple random seeds using the existing DDP training script.

This launcher does not modify train_mosei.py. For every seed it:
  1. reads the base YAML config,
  2. writes a seed-specific YAML config,
  3. changes only train.seed, logging.run_name, and logging.out_dir,
  4. launches the existing train_mosei.py with torchrun,
  5. collects test_results.json and history.json into multiseed_summary.json.

Default seeds: 42, 3407, 2024

Example:
  python scripts/train/run_mosei_multiseed.py \
    --gpus 5,6,7,8,9

Override seeds:
  python scripts/train/run_mosei_multiseed.py \
    --seeds 42,3407,2024 \
    --gpus 5,6,7,8,9
"""

import argparse
import copy
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_BASE_CONFIG = "/home/tahirahmad/Project_Code/CHADO_EMNLP/configs/mosei/chado_mosei.yaml"
DEFAULT_TRAIN_SCRIPT = "/home/tahirahmad/Project_Code/CHADO_EMNLP/scripts/train/train_mosei.py"
DEFAULT_SEEDS = "42,3407,2024"
DEFAULT_OUT_ROOT = "/home/tahirahmad/Project_Code/CHADO_EMNLP/experiments/results/mosei/chado_multiseed"
DEFAULT_MASTER_PORT = 29702


def parse_seed_list(seed_text: str) -> List[int]:
    seeds = [int(x.strip()) for x in seed_text.replace(",", " ").split() if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if torchrun can bind this local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def find_free_port(start_port: int, used_ports: set[int], host: str = "127.0.0.1") -> int:
    """Find a free TCP port at or above start_port, avoiding ports used by earlier seeds."""
    port = int(start_port)
    while port in used_ports or not is_port_free(port, host=host):
        port += 1
    used_ports.add(port)
    return port


def best_value_from_history(history: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not history:
        return None
    values = history.get(key, [])
    if not values:
        return None
    return float(max(values))


def best_epoch_from_history(history: Optional[Dict[str, Any]], key: str) -> Optional[int]:
    if not history:
        return None
    values = history.get(key, [])
    epochs = history.get("epoch", [])
    if not values or not epochs:
        return None
    best_idx = max(range(len(values)), key=lambda i: values[i])
    return int(epochs[best_idx])


def summarize_numeric_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    keys = set()
    for row in rows:
        keys.update(row.keys())

    for key in sorted(keys):
        if key in {"seed", "out_dir", "best_val_epoch"}:
            continue
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        summary[key] = {"mean": mean, "std": std, "values": vals}
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    p.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma/space separated seeds. Default: 42,3407,2024")
    p.add_argument("--gpus", required=True, help="Comma separated visible GPU ids, e.g. 5,6,7,8,9")
    p.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    p.add_argument(
        "--no-auto-port",
        action="store_true",
        help="Disable automatic free-port selection and use --master-port + seed_index exactly.",
    )
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--nproc-per-node", type=int, default=None)
    p.add_argument("--resume-existing", action="store_true", help="Skip seeds whose test_results.json already exists.")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next seed if one torchrun job fails. By default, stop on first failure.",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to train_mosei.py after --config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        raise ValueError("--gpus must contain at least one GPU id.")

    nproc = args.nproc_per_node or len(gpu_list)
    if nproc != len(gpu_list):
        print(f"[warning] nproc_per_node={nproc}, but --gpus has {len(gpu_list)} entries: {gpu_list}")

    base_config_path = Path(args.base_config)
    train_script_path = Path(args.train_script)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with base_config_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    if "train" not in base_cfg:
        raise KeyError("Base config is missing train section.")
    if "logging" not in base_cfg:
        raise KeyError("Base config is missing logging section.")

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    used_master_ports: set[int] = set()

    for idx, seed in enumerate(seeds):
        seed_out = out_root / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        seed_config_path = seed_out / "chado_mosei_seed.yaml"
        test_result_path = seed_out / "test_results.json"
        history_path = seed_out / "history.json"

        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["seed"] = int(seed)
        cfg["logging"]["run_name"] = f"chado_mosei_seed_{seed}"
        cfg["logging"]["out_dir"] = str(seed_out)

        with seed_config_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        if args.resume_existing and test_result_path.exists():
            print(f"[multi-seed] seed={seed} already has test_results.json; skipping.")
        else:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

            requested_port = args.master_port + idx
            master_port = (
                requested_port
                if args.no_auto_port
                else find_free_port(requested_port, used_master_ports)
            )
            if master_port != requested_port:
                print(
                    f"[multi-seed] requested port {requested_port} is busy; "
                    f"using free port {master_port} instead."
                )

            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                f"--master_port={master_port}",
                str(train_script_path),
                "--config",
                str(seed_config_path),
            ]
            if args.extra:
                cmd.extend(args.extra)

            print("\n" + "=" * 100)
            print(f"[multi-seed] seed: {seed}")
            print(f"[multi-seed] config: {seed_config_path}")
            print(f"[multi-seed] output: {seed_out}")
            print(f"[multi-seed] CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
            print(f"[multi-seed] command: {' '.join(cmd)}")
            print("=" * 100 + "\n")

            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                failed.append({"seed": seed, "returncode": exc.returncode, "cmd": cmd, "out_dir": str(seed_out)})
                if not args.continue_on_error:
                    raise
                print(f"[multi-seed] seed={seed} failed with return code {exc.returncode}; continuing.")

        test_metrics = load_json(test_result_path)
        history = load_json(history_path)
        if test_metrics is None:
            print(f"[multi-seed] warning: missing {test_result_path}")
            continue

        row: Dict[str, Any] = {
            "seed": seed,
            "out_dir": str(seed_out),
            "best_val_macro_f1": best_value_from_history(history, "val_macro_f1"),
            "best_val_weighted_f1": best_value_from_history(history, "val_weighted_f1"),
            "best_val_epoch": best_epoch_from_history(history, "val_macro_f1"),
        }
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value
        rows.append(row)

    ranked_by_val = sorted(
        rows,
        key=lambda r: (-1 if r.get("best_val_macro_f1") is None else float(r["best_val_macro_f1"])),
        reverse=True,
    )
    ranked_by_test_macro = sorted(rows, key=lambda r: float(r.get("test_macro_f1", -1)), reverse=True)
    ranked_by_test_weighted = sorted(rows, key=lambda r: float(r.get("test_weighted_f1", -1)), reverse=True)

    summary = {
        "base_config": str(base_config_path),
        "train_script": str(train_script_path),
        "gpus": gpu_list,
        "seeds": seeds,
        "num_completed": len(rows),
        "num_failed": len(failed),
        "failed": failed,
        "ranked_by_best_val_macro_f1": ranked_by_val,
        "ranked_by_test_macro_f1": ranked_by_test_macro,
        "ranked_by_test_weighted_f1": ranked_by_test_weighted,
        "summary": summarize_numeric_results(rows),
    }

    summary_path = out_root / "multiseed_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print(f"[multi-seed] completed {len(rows)}/{len(seeds)} seeds")
    print(f"[multi-seed] summary: {summary_path}")
    if ranked_by_val:
        best = ranked_by_val[0]
        print(
            "[multi-seed] best validation seed: "
            f"{best['seed']} | best_val_macro_f1={best.get('best_val_macro_f1')} | "
            f"test_macro_f1={best.get('test_macro_f1')} | "
            f"test_weighted_f1={best.get('test_weighted_f1')}"
        )
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
