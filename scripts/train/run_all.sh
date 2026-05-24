#!/usr/bin/env bash
# Launch CHADO training on all three datasets sequentially.
# Usage:  bash scripts/train/run_all.sh [iemocap|meld|mosei|all]

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

run_iemocap () {
    echo "====== IEMOCAP ======"
    torchrun --nproc_per_node=5 --master_port=29700 \
        scripts/train/train_iemocap.py --config configs/iemocap/chado_iemocap.yaml
}

run_meld () {
    echo "====== MELD ======"
    torchrun --nproc_per_node=5 --master_port=29701 \
        scripts/train/train_meld.py --config configs/meld/chado_meld.yaml
}

run_mosei () {
    echo "====== CMU-MOSEI ======"
    torchrun --nproc_per_node=5 --master_port=29702 \
        scripts/train/train_mosei.py --config configs/mosei/chado_mosei.yaml
}

case "${DATASET}" in
    iemocap) run_iemocap ;;
    meld)    run_meld ;;
    mosei)   run_mosei ;;
    all)     run_iemocap; run_meld; run_mosei ;;
    *)       echo "Usage: $0 [iemocap|meld|mosei|all]"; exit 1 ;;
esac
