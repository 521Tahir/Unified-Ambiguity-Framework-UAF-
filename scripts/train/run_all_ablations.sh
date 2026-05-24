#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unified AR-GOT ablation launcher
# Runs ablations for IEMOCAP, CMU-MOSEI, and MELD.
#
# Default mode:
#   Runs multiple single-GPU jobs in parallel across available GPUs.
#
# Example:
#   PROJECT_ROOT=/home/tahirahmad/Project_Code/CHADO_EMNLP \
#   GPUS=5,6,7,8,9 \
#   GPUS_PER_JOB=1 \
#   bash scripts/train/run_all_ablations.sh
#
# DDP sequential mode:
#   PROJECT_ROOT=/home/tahirahmad/Project_Code/CHADO_EMNLP \
#   GPUS=5,6,7,8,9 \
#   GPUS_PER_JOB=5 \
#   bash scripts/train/run_all_ablations.sh
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/home/tahirahmad/Project_Code/CHADO_EMNLP}"
GPUS="${GPUS:-0,1,2,3,4}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
BASE_PORT="${BASE_PORT:-29700}"
SEED="${SEED:-42}"

# Optional filters:
# DATASETS can be: iemocap,mosei,meld
# ABLATIONS can be: wo_mad,wo_ot,wo_radial,wo_causal
DATASETS="${DATASETS:-iemocap,mosei,meld}"
ABLATIONS="${ABLATIONS:-wo_mad,wo_ot,wo_radial,wo_causal}"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_ROOT}"

contains_csv() {
  local csv="$1"
  local item="$2"
  [[ ",${csv}," == *",${item},"* ]]
}

# ------------------------------------------------------------
# Build GPU groups
# ------------------------------------------------------------
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"

GPU_GROUPS=()
i=0
while [[ "${i}" -lt "${#GPU_ARRAY[@]}" ]]; do
  group=""
  count=0

  for ((j=i; j<i+GPUS_PER_JOB && j<${#GPU_ARRAY[@]}; j++)); do
    group+="${GPU_ARRAY[$j]},"
    count=$((count + 1))
  done

  group="${group%,}"

  if [[ "${count}" -eq "${GPUS_PER_JOB}" ]]; then
    GPU_GROUPS+=("${group}")
  fi

  i=$((i + GPUS_PER_JOB))
done

if [[ "${#GPU_GROUPS[@]}" -lt 1 ]]; then
  echo "[ERROR] No valid GPU groups created. Check GPUS and GPUS_PER_JOB."
  exit 1
fi

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] GPUS=${GPUS}"
echo "[INFO] GPUS_PER_JOB=${GPUS_PER_JOB}"
echo "[INFO] GPU_GROUPS=${GPU_GROUPS[*]}"
echo "[INFO] DATASETS=${DATASETS}"
echo "[INFO] ABLATIONS=${ABLATIONS}"
echo "[INFO] SEED=${SEED}"

# ------------------------------------------------------------
# Job list format:
# dataset | ablation | train_script | config | out_dir
# ------------------------------------------------------------
JOBS=()

add_job() {
  local dataset="$1"
  local ablation="$2"
  local script="$3"
  local config="$4"
  local outdir="$5"

  if contains_csv "${DATASETS}" "${dataset}" && contains_csv "${ABLATIONS}" "${ablation}"; then
    JOBS+=("${dataset}|${ablation}|${script}|${config}|${outdir}")
  fi
}

# ----------------------
# w/o MAD
# ----------------------
add_job "iemocap" "wo_mad" \
  "scripts/train/train_iemocap.py" \
  "configs/iemocap/comp_abl_wo_mad.yaml" \
  "experiments/results/iemocap/chado_wo_mad"

add_job "mosei" "wo_mad" \
  "scripts/train/train_mosei.py" \
  "configs/mosei/comp_abl_wo_mad.yaml" \
  "experiments/results/mosei/chado_wo_mad"

add_job "meld" "wo_mad" \
  "scripts/train/train_meld.py" \
  "configs/meld/comp_abl_wo_mad.yaml" \
  "experiments/results/meld/chado_wo_mad"

# ----------------------
# w/o OT
# ----------------------
add_job "iemocap" "wo_ot" \
  "scripts/train/train_iemocap.py" \
  "configs/iemocap/comp_abl_wo_ot.yaml" \
  "experiments/results/iemocap/chado_wo_ot"

add_job "mosei" "wo_ot" \
  "scripts/train/train_mosei.py" \
  "configs/mosei/comp_abl_wo_ot.yaml" \
  "experiments/results/mosei/chado_wo_ot"

add_job "meld" "wo_ot" \
  "scripts/train/train_meld.py" \
  "configs/meld/comp_abl_wo_ot.yaml" \
  "experiments/results/meld/chado_wo_ot"

# ----------------------
# w/o Radial
# Internally your code/config may still call this "hyperbolic".
# The paper-facing name should remain "w/o Radial".
# ----------------------
add_job "iemocap" "wo_radial" \
  "scripts/train/train_iemocap.py" \
  "configs/iemocap/comp_abl_wo_radial.yaml" \
  "experiments/results/iemocap/chado_wo_radial"

add_job "mosei" "wo_radial" \
  "scripts/train/train_mosei.py" \
  "configs/mosei/comp_abl_wo_radial.yaml" \
  "experiments/results/mosei/chado_wo_radial"

add_job "meld" "wo_radial" \
  "scripts/train/train_meld.py" \
  "configs/meld/comp_abl_wo_radial.yaml" \
  "experiments/results/meld/chado_wo_radial"

# ----------------------
# w/o all latent factors / w/o causal
# ----------------------
add_job "iemocap" "wo_causal" \
  "scripts/train/train_iemocap.py" \
  "configs/iemocap/comp_abl_wo_causal.yaml" \
  "experiments/results/iemocap/chado_wo_causal"

add_job "mosei" "wo_causal" \
  "scripts/train/train_mosei.py" \
  "configs/mosei/comp_abl_wo_causal.yaml" \
  "experiments/results/mosei/chado_wo_causal"

add_job "meld" "wo_causal" \
  "scripts/train/train_meld.py" \
  "configs/meld/comp_abl_wo_causal.yaml" \
  "experiments/results/meld/chado_wo_causal"

if [[ "${#JOBS[@]}" -eq 0 ]]; then
  echo "[ERROR] No jobs selected. Check DATASETS and ABLATIONS."
  exit 1
fi

echo "[INFO] Number of selected jobs: ${#JOBS[@]}"

# ------------------------------------------------------------
# Run one job
# ------------------------------------------------------------
run_job() {
  local job_id="$1"
  local job_spec="$2"
  local gpu_group="$3"
  local port="$4"

  IFS='|' read -r dataset ablation script config outdir <<< "${job_spec}"

  mkdir -p "${outdir}"

  echo "============================================================" | tee "${outdir}/launch.log"
  echo "[JOB ${job_id}] dataset=${dataset}" | tee -a "${outdir}/launch.log"
  echo "[JOB ${job_id}] ablation=${ablation}" | tee -a "${outdir}/launch.log"
  echo "[JOB ${job_id}] script=${script}" | tee -a "${outdir}/launch.log"
  echo "[JOB ${job_id}] config=${config}" | tee -a "${outdir}/launch.log"
  echo "[JOB ${job_id}] outdir=${outdir}" | tee -a "${outdir}/launch.log"
  echo "[JOB ${job_id}] CUDA_VISIBLE_DEVICES=${gpu_group}" | tee -a "${outdir}/launch.log"
  echo "[JOB ${job_id}] GPUS_PER_JOB=${GPUS_PER_JOB}" | tee -a "${outdir}/launch.log"
  echo "============================================================" | tee -a "${outdir}/launch.log"

  if [[ ! -f "${config}" ]]; then
    echo "[ERROR] Missing config: ${config}" | tee -a "${outdir}/launch.log"
    exit 1
  fi

  if [[ ! -f "${script}" ]]; then
    echo "[ERROR] Missing train script: ${script}" | tee -a "${outdir}/launch.log"
    exit 1
  fi

  if [[ "${GPUS_PER_JOB}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_group}" \
    python3 -u "${script}" \
      --config "${config}" \
      --out-dir "${outdir}" \
      --seed "${SEED}" \
      2>&1 | tee "${outdir}/train.log"
  else
    CUDA_VISIBLE_DEVICES="${gpu_group}" \
    torchrun \
      --nproc_per_node="${GPUS_PER_JOB}" \
      --master_port="${port}" \
      "${script}" \
      --config "${config}" \
      --out-dir "${outdir}" \
      --seed "${SEED}" \
      2>&1 | tee "${outdir}/train.log"
  fi
}

# ------------------------------------------------------------
# Scheduler:
# Runs one job per GPU group in parallel.
# With GPUS_PER_JOB=1 and five GPUs, this launches five jobs at once.
# ------------------------------------------------------------
pids=()
job_idx=0
group_idx=0
num_groups="${#GPU_GROUPS[@]}"

for job_spec in "${JOBS[@]}"; do
  gpu_group="${GPU_GROUPS[$group_idx]}"
  port=$((BASE_PORT + job_idx))

  run_job "${job_idx}" "${job_spec}" "${gpu_group}" "${port}" &
  pids+=("$!")

  job_idx=$((job_idx + 1))
  group_idx=$((group_idx + 1))

  if [[ "${group_idx}" -ge "${num_groups}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
    group_idx=0
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[INFO] All selected ablation jobs finished."

# ------------------------------------------------------------
# Create a compact summary CSV from test_results.json files
# ------------------------------------------------------------
python3 - <<'PY'
import json
import csv
from pathlib import Path

jobs = [
    ("iemocap", "wo_mad",    "experiments/results/iemocap/chado_wo_mad"),
    ("mosei",   "wo_mad",    "experiments/results/mosei/chado_wo_mad"),
    ("meld",    "wo_mad",    "experiments/results/meld/chado_wo_mad"),

    ("iemocap", "wo_ot",     "experiments/results/iemocap/chado_wo_ot"),
    ("mosei",   "wo_ot",     "experiments/results/mosei/chado_wo_ot"),
    ("meld",    "wo_ot",     "experiments/results/meld/chado_wo_ot"),

    ("iemocap", "wo_radial", "experiments/results/iemocap/chado_wo_radial"),
    ("mosei",   "wo_radial", "experiments/results/mosei/chado_wo_radial"),
    ("meld",    "wo_radial", "experiments/results/meld/chado_wo_radial"),

    ("iemocap", "wo_causal", "experiments/results/iemocap/chado_wo_causal"),
    ("mosei",   "wo_causal", "experiments/results/mosei/chado_wo_causal"),
    ("meld",    "wo_causal", "experiments/results/meld/chado_wo_causal"),
]

rows = []
all_keys = set()

for dataset, ablation, outdir in jobs:
    result_path = Path(outdir) / "test_results.json"
    row = {
        "dataset": dataset,
        "ablation": ablation,
        "out_dir": outdir,
        "result_file": str(result_path),
        "status": "missing",
    }

    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            row["status"] = "ok"
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    row[k] = v
                    all_keys.add(k)
        except Exception as e:
            row["status"] = f"error: {e}"

    rows.append(row)

fieldnames = ["dataset", "ablation", "status", "out_dir", "result_file"] + sorted(all_keys)

summary_path = Path("experiments/results/ablation_summary.csv")
summary_path.parent.mkdir(parents=True, exist_ok=True)

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"[INFO] Wrote summary: {summary_path}")
PY