#!/usr/bin/env bash
# Component ablations for IEMOCAP.
# Each run removes one CHADO component at a time.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5,6,7,8,9
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

BASE_CFG="configs/iemocap/chado_iemocap.yaml"
OUT_ROOT="experiments/results/iemocap/ablations"
mkdir -p "${OUT_ROOT}"

run_one () {
    local name="$1"; shift
    local out="${OUT_ROOT}/${name}"
    mkdir -p "${out}"
    echo "====== ablation: ${name} ======"

    # Write a temp config with overridden chado flags
    python - << PY
import yaml, copy, sys
with open("${BASE_CFG}") as f:
    cfg = yaml.safe_load(f)
cfg["logging"]["run_name"] = "${name}"
cfg["logging"]["out_dir"] = "${out}"
overrides = dict($@)
cfg["chado"].update(overrides)
with open("${out}/config.yaml", "w") as f:
    yaml.dump(cfg, f)
PY

    torchrun --nproc_per_node=5 --master_port=29710 \
        scripts/train/train_iemocap.py --config "${out}/config.yaml" \
        2>&1 | tee "${out}/train.log"
}

run_one "full"            "use_causal: true,  use_hyperbolic: true,  use_ot: true"
run_one "wo_causal"       "use_causal: false, use_hyperbolic: true,  use_ot: true"
run_one "wo_hyperbolic"   "use_causal: true,  use_hyperbolic: false, use_ot: true"
run_one "wo_ot"           "use_causal: true,  use_hyperbolic: true,  use_ot: false"
run_one "wo_causal_hyp"   "use_causal: false, use_hyperbolic: false, use_ot: true"
run_one "baseline"        "use_causal: false, use_hyperbolic: false, use_ot: false"

echo "====== Ablations done. Results in ${OUT_ROOT} ======"
