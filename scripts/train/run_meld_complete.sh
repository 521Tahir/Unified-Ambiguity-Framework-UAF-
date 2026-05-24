#!/usr/bin/env bash
# =============================================================================
#  MELD — complete ALL missing seeds  (skips already done)
#
#  GPU layout:
#    AR-GOT (DDP)  : GPU 4,5
#    Baselines     : batch of 4 in parallel → GPU 4,5,6,7
#
#  Seeds: 42 (missing baselines) → 3407 (missing baselines) → 456 (all)
#
#  Run:  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#        bash scripts/train/run_meld_complete.sh 2>&1 | tee logs/meld_complete.log
# =============================================================================
set -uo pipefail
PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
RES=$PROJ/experiments/results/meld
mkdir -p "$PROJ/logs"

TRAIN=$PROJ/scripts/train/train_meld.py
BASE=$PROJ/scripts/train/train_baseline.py

CFG_ARGOT=$PROJ/configs/meld/chado_meld_final.yaml
CFG_BPMULT=$PROJ/configs/meld/bpmult_meld.yaml
CFG_CTNET=$PROJ/configs/meld/ctnet_meld.yaml
CFG_MULT=$PROJ/configs/meld/mult_meld.yaml
CFG_MMDFN=$PROJ/configs/meld/mmdfn_meld.yaml
CFG_LFLSTM=$PROJ/configs/meld/lflstm_meld.yaml
CFG_AERLLM=$PROJ/configs/meld/aerllm_meld.yaml
CFG_EMOCLIP=$PROJ/configs/meld/emoclip_meld.yaml
CFG_OVMER=$PROJ/configs/meld/ovmer_meld.yaml

skip_done() { [ -f "$1/test_results.json" ] && { echo "  [SKIP] $(basename $(dirname $1))/$(basename $1)"; return 0; }; return 1; }

run_argot() {
    local seed=$1 out=$2 port=$3
    skip_done "$out" && return 0
    mkdir -p "$out"
    echo "  [RUN ] argot/seed_$seed  GPU=4,5  port=$port"
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=$port \
        "$TRAIN" --config "$CFG_ARGOT" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?; [ $rc -eq 0 ] && echo "  [OK ] argot/seed_$seed" || echo "  [ERR] argot/seed_$seed  exit=$rc"
}

run_base() {
    local gpu=$1 cfg=$2 seed=$3 out=$4 port=$5
    skip_done "$out" && return 0
    mkdir -p "$out"
    local name=$(basename $(dirname $out))
    echo "  [RUN ] $name/seed_$seed  GPU=$gpu  port=$port"
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$port \
        "$BASE" --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?; [ $rc -eq 0 ] && echo "  [OK ] $name/seed_$seed" || echo "  [ERR] $name/seed_$seed  exit=$rc"
}

run_baseline_batch() {
    local SEED=$1; shift
    declare -a PAIRS=("$@")
    local GPUS=(4 5 6 7)
    local PIDS=()
    for i in "${!PAIRS[@]}"; do
        local name="${PAIRS[$i]%%:*}" cfg="${PAIRS[$i]##*:}"
        local port=$((29870 + i + (SEED % 100) * 10))
        run_base "${GPUS[$i]}" "$cfg" "$SEED" "$RES/${name}_ms/seed_$SEED" "$port" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || echo "  [WARN] batch item failed"; done
}

echo "================================================================"
echo "  MELD COMPLETE"
echo "================================================================"

# ── Seed 42: argot done; ctnet/aerllm/ovmer in-progress; bpmult/mult/mmdfn/lflstm/emoclip missing ──
echo ""
echo "=== seed=42 ==="
run_argot 42 "$RES/argot_ms/seed_42" 29858
run_baseline_batch 42 \
    "bpmult:$CFG_BPMULT" \
    "mult:$CFG_MULT" \
    "mmdfn:$CFG_MMDFN" \
    "lflstm:$CFG_LFLSTM"
run_base 4 "$CFG_EMOCLIP" 42 "$RES/emoclip_ms/seed_42" 29859

# ── Seed 3407: argot/aerllm/ovmer in-progress; rest missing ──
echo ""
echo "=== seed=3407 ==="
run_argot 3407 "$RES/argot_ms/seed_3407" 29860
run_baseline_batch 3407 \
    "bpmult:$CFG_BPMULT" \
    "ctnet:$CFG_CTNET" \
    "mult:$CFG_MULT" \
    "mmdfn:$CFG_MMDFN"
run_baseline_batch 3407 \
    "lflstm:$CFG_LFLSTM" \
    "emoclip:$CFG_EMOCLIP" \
    "aerllm:$CFG_AERLLM" \
    "ovmer:$CFG_OVMER"

# ── Seed 456: all missing ──
echo ""
echo "=== seed=456 ==="
run_argot 456 "$RES/argot_ms/seed_456" 29861
run_baseline_batch 456 \
    "bpmult:$CFG_BPMULT" \
    "ctnet:$CFG_CTNET" \
    "mult:$CFG_MULT" \
    "mmdfn:$CFG_MMDFN"
run_baseline_batch 456 \
    "lflstm:$CFG_LFLSTM" \
    "aerllm:$CFG_AERLLM" \
    "emoclip:$CFG_EMOCLIP" \
    "ovmer:$CFG_OVMER"

echo ""
echo "================================================================"
echo "  MELD COMPLETE — computing table"
echo "================================================================"
python3 "$PROJ/scripts/eval/compute_multiseed_table.py" --dataset meld || true
