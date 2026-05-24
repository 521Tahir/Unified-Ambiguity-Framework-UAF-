#!/usr/bin/env bash
# Re-run MOSEI baselines that didn't have pos_weight (BaselineFusion, MulT)
# then run MOSEI CHADO. Run this AFTER CTNet MOSEI finishes.
# Usage: bash scripts/train/run_mosei_rerun.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=5,6,7,8,9
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

PORT=29710

run_baseline() {
    local name="$1"; local cfg="$2"
    local log_dir="experiments/results/mosei/${name}"
    mkdir -p "${log_dir}"
    echo "====== Baseline: ${name} (MOSEI re-run w/ pos_weight) ======"
    torchrun --nproc_per_node=5 --master_port=${PORT} \
        scripts/train/train_baseline_mosei.py --config "${cfg}" \
        2>&1 | tee "${log_dir}/train.log"
    PORT=$((PORT + 1))
}

echo "====== Re-running MOSEI Baselines with pos_weight ======"
run_baseline "BaselineFusion" "configs/mosei/baseline_mosei.yaml"
run_baseline "MulT"          "configs/mosei/mult_mosei.yaml"

echo "====== MOSEI CHADO ======"
mkdir -p experiments/results/mosei/chado
torchrun --nproc_per_node=5 --master_port=${PORT} \
    scripts/train/train_mosei.py --config configs/mosei/chado_mosei.yaml \
    2>&1 | tee experiments/results/mosei/chado/train.log

echo "====== All MOSEI runs complete ======"
