#!/bin/bash
# ============================================================
# Modality Ablation Runner — IEMOCAP and MELD
# Runs all models × all modality combos (T, T+A, T+V, T+A+V)
# T+A+V results already exist; this runs the 3 missing combos.
# Usage: bash scripts/train/run_modality_ablation.sh [dataset]
#   dataset: iemocap | meld | both (default: both)
# ============================================================
set -e
DATASET=${1:-both}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# GPUs to cycle through
GPUS=(5 6 7 8 9)
N_GPU=${#GPUS[@]}
GPU_IDX=0
declare -A PIDS   # pid -> label
LOG_DIR=/tmp/modality_ablation
mkdir -p "$LOG_DIR"

# Map model to train script
get_script() {
    local model=$1 dataset=$2
    if [[ "$model" == "chado" ]]; then
        if [[ "$dataset" == "iemocap" ]]; then echo "scripts/train/train_iemocap.py"
        else echo "scripts/train/train_meld.py"; fi
    else
        echo "scripts/train/train_baseline.py"
    fi
}

# Modality combos to run (T+A+V already done, skip)
declare -A MODALITY_FLAGS
MODALITY_FLAGS["T"]="use_audio: false\n  use_video: false"
MODALITY_FLAGS["TA"]="use_audio: true\n  use_video: false"
MODALITY_FLAGS["TV"]="use_audio: false\n  use_video: true"

# Models and their base config files
declare -A MODELS_IEMOCAP
MODELS_IEMOCAP["chado"]="configs/iemocap/chado_iemocap.yaml"
MODELS_IEMOCAP["baseline"]="configs/iemocap/baseline_iemocap.yaml"
MODELS_IEMOCAP["mult"]="configs/iemocap/mult_iemocap.yaml"
MODELS_IEMOCAP["mmdfn"]="configs/iemocap/mmdfn_iemocap.yaml"
MODELS_IEMOCAP["ctnet"]="configs/iemocap/ctnet_iemocap.yaml"
MODELS_IEMOCAP["lflstm"]="configs/iemocap/lflstm_iemocap.yaml"
MODELS_IEMOCAP["bpmult"]="configs/iemocap/bpmult_iemocap.yaml"

declare -A MODELS_MELD
MODELS_MELD["chado"]="configs/meld/chado_meld.yaml"
MODELS_MELD["baseline"]="configs/meld/baseline_meld.yaml"
MODELS_MELD["mult"]="configs/meld/mult_meld.yaml"
MODELS_MELD["mmdfn"]="configs/meld/mmdfn_meld.yaml"
MODELS_MELD["ctnet"]="configs/meld/ctnet_meld.yaml"
MODELS_MELD["lflstm"]="configs/meld/lflstm_meld.yaml"
MODELS_MELD["bpmult"]="configs/meld/bpmult_meld.yaml"

launch_job() {
    local model=$1 dataset=$2 modality=$3 base_cfg=$4
    local gpu=${GPUS[$GPU_IDX]}
    GPU_IDX=$(( (GPU_IDX + 1) % N_GPU ))

    local tag="${model}_${dataset}_${modality}"
    local tmp_cfg="/tmp/modality_ablation/cfg_${tag}.yaml"
    local out_dir="experiments/results/${dataset}/${model}_${modality}"
    local log="$LOG_DIR/${tag}.log"
    local script
    script=$(get_script "$model" "$dataset")

    # Create temp config: patch use_audio, use_video, run_name, out_dir
    python3 - <<PYEOF
import yaml, re, os
with open("$base_cfg") as f:
    cfg = yaml.safe_load(f)

modality = "$modality"
cfg["model"]["use_audio"] = "A" in modality
cfg["model"]["use_video"] = "V" in modality
cfg["model"]["use_text"]  = True  # always use text

cfg.setdefault("logging", {})
cfg["logging"]["run_name"] = "${tag}"
cfg["logging"]["out_dir"]  = "${out_dir}"

# Reduce epochs for ablation (use 60% of original)
if "train" in cfg:
    orig = cfg["train"].get("epochs", 30)
    cfg["train"]["epochs"] = max(20, int(orig * 0.7))
    cfg["train"]["patience"] = max(6, cfg["train"].get("patience", 8))

os.makedirs(os.path.dirname("$tmp_cfg"), exist_ok=True)
with open("$tmp_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written: $tmp_cfg  use_audio={'A' in modality}  use_video={'V' in modality}")
PYEOF

    echo "[LAUNCH] GPU=$gpu  $tag"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=$gpu python "$script" --config "$tmp_cfg" \
        > "$log" 2>&1 &
    PIDS[$!]="$tag"
}

run_dataset() {
    local dataset=$1
    local -n models_ref=$2

    echo "========================================"
    echo "Dataset: $dataset"
    echo "========================================"

    for model in "${!models_ref[@]}"; do
        local base_cfg="${models_ref[$model]}"
        if [[ ! -f "$base_cfg" ]]; then
            echo "[SKIP] $base_cfg not found"
            continue
        fi
        for modality in T TA TV; do
            local out_dir="experiments/results/${dataset}/${model}_${modality}"
            if [[ -f "${out_dir}/test_results.json" ]]; then
                echo "[SKIP] $model $dataset $modality — already done"
                continue
            fi
            launch_job "$model" "$dataset" "$modality" "$base_cfg"
            # Small stagger to avoid port conflicts
            sleep 5
        done
    done
}

if [[ "$DATASET" == "iemocap" || "$DATASET" == "both" ]]; then
    run_dataset iemocap MODELS_IEMOCAP
fi
if [[ "$DATASET" == "meld" || "$DATASET" == "both" ]]; then
    run_dataset meld MODELS_MELD
fi

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Monitor: watch -n30 'ls /tmp/modality_ablation/*.log | xargs -I{} sh -c \"echo {} && tail -2 {}\"'"
echo ""

# Wait for all background jobs
FAILED=()
for pid in "${!PIDS[@]}"; do
    tag="${PIDS[$pid]}"
    if wait "$pid"; then
        echo "[DONE] $tag"
    else
        echo "[FAIL] $tag (exit $?)"
        FAILED+=("$tag")
    fi
done

echo ""
echo "========================================"
echo "ALL DONE"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED: ${FAILED[*]}"
fi
echo "========================================"
