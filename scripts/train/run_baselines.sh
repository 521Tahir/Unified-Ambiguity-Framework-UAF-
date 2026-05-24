#!/usr/bin/env bash
# Run all baselines on a given dataset.
# Usage:  bash scripts/train/run_baselines.sh iemocap
#         bash scripts/train/run_baselines.sh meld
#         bash scripts/train/run_baselines.sh mosei
#         bash scripts/train/run_baselines.sh all

set -euo pipefail

export CUDA_VISIBLE_DEVICES=5,6,7,8,9
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET="${1:-all}"

PORT=29703

run_baseline () {
    local name="$1"
    local cfg="$2"
    local script="${3:-scripts/train/train_baseline.py}"
    local dataset_dir
    dataset_dir="$(dirname ${cfg#configs/})"
    local log_dir="experiments/results/${dataset_dir}/${name}"
    mkdir -p "${log_dir}"
    echo "====== Baseline: ${name} ======"
    torchrun --nproc_per_node=5 --master_port=${PORT} \
        "${script}" --config "${cfg}" \
        2>&1 | tee "${log_dir}/train.log"
    PORT=$((PORT + 1))
}

run_iemocap () {
    run_baseline "TriModal-Baseline" "configs/iemocap/baseline_iemocap.yaml"
    run_baseline "MulT"              "configs/iemocap/mult_iemocap.yaml"
    run_baseline "MM-DFN"            "configs/iemocap/mmdfn_iemocap.yaml"
    run_baseline "CTNet"             "configs/iemocap/ctnet_iemocap.yaml"
}

run_meld () {
    run_baseline "TriModal-Baseline" "configs/meld/baseline_meld.yaml"
    run_baseline "MulT"              "configs/meld/mult_meld.yaml"
    run_baseline "MM-DFN"            "configs/meld/mmdfn_meld.yaml"
    run_baseline "CTNet"             "configs/meld/ctnet_meld.yaml"
}

run_mosei () {
    run_baseline "BaselineFusion" "configs/mosei/baseline_mosei.yaml" "scripts/train/train_baseline_mosei.py"
    run_baseline "MulT"          "configs/mosei/mult_mosei.yaml"      "scripts/train/train_baseline_mosei.py"
    run_baseline "MM-DFN"        "configs/mosei/mmdfn_mosei.yaml"     "scripts/train/train_baseline_mosei.py"
    run_baseline "CTNet"         "configs/mosei/ctnet_mosei.yaml"     "scripts/train/train_baseline_mosei.py"
}

case "${DATASET}" in
    iemocap) run_iemocap ;;
    meld)    run_meld ;;
    mosei)   run_mosei ;;
    all)     run_iemocap; run_meld; run_mosei ;;
    *)       echo "Usage: $0 [iemocap|meld|mosei|all]"; exit 1 ;;
esac

echo "====== All baselines done ======"
