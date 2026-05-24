#!/usr/bin/env bash
# =============================================================================
#  IEMOCAP — complete ALL missing seeds  (skips already done)
#
#  GPU layout:
#    AR-GOT (DDP)  : GPU 0,1
#    Baselines     : batch of 4 in parallel → GPU 2,3,4,5
#                    second batch          → GPU 2,3,4,5 (next 4)
#                    remainder             → GPU 2,3
#
#  Seeds processed: 42 (baselines only) → 3407 (all) → 456 (all)
#
#  Run:  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#        bash scripts/train/run_iemocap_complete.sh 2>&1 | tee logs/iemocap_complete.log
# =============================================================================
set -uo pipefail
PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
RES=$PROJ/experiments/results/iemocap
mkdir -p "$PROJ/logs"

TRAIN=$PROJ/scripts/train/train_iemocap.py
BASE=$PROJ/scripts/train/train_baseline.py

CFG_ARGOT=$PROJ/configs/iemocap/chado_iemocap_final.yaml
CFG_BPMULT=$PROJ/configs/iemocap/bpmult_iemocap.yaml
CFG_CTNET=$PROJ/configs/iemocap/ctnet_iemocap.yaml
CFG_MULT=$PROJ/configs/iemocap/mult_iemocap.yaml
CFG_MMDFN=$PROJ/configs/iemocap/mmdfn_iemocap.yaml
CFG_LFLSTM=$PROJ/configs/iemocap/lflstm_iemocap.yaml
CFG_AERLLM=$PROJ/configs/iemocap/aerllm_iemocap.yaml
CFG_EMOCLIP=$PROJ/configs/iemocap/emoclip_iemocap.yaml
CFG_OVMER=$PROJ/configs/iemocap/ovmer_iemocap.yaml

skip_done() { [ -f "$1/test_results.json" ] && { echo "  [SKIP] $(basename $(dirname $1))/$(basename $1)"; return 0; }; return 1; }

run_argot() {
    local seed=$1 out=$2 port=$3
    skip_done "$out" && return 0
    mkdir -p "$out"
    echo "  [RUN ] argot/seed_$seed  GPU=0,1  port=$port"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$port \
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
    # runs 4 baselines in parallel on GPU 2,3,4,5
    local SEED=$1
    shift
    # $@ = list of "name:cfg" pairs
    declare -a PAIRS=("$@")
    local GPUS=(2 3 4 5)
    local PIDS=()

    for i in "${!PAIRS[@]}"; do
        local pair="${PAIRS[$i]}"
        local name="${pair%%:*}"
        local cfg="${pair##*:}"
        local gpu="${GPUS[$i]}"
        local port=$((29860 + i + SEED * 10))
        run_base "$gpu" "$cfg" "$SEED" "$RES/${name}_ms/seed_$SEED" "$port" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || echo "  [WARN] batch item failed"; done
}

echo "================================================================"
echo "  IEMOCAP COMPLETE"
echo "================================================================"

# ── Seed 42: only missing baselines (argot/ctnet/mult/mmdfn/aerllm/emoclip done) ──
echo ""
echo "=== seed=42  (bpmult lflstm ovmer still missing) ==="
run_argot 42 "$RES/argot_ms/seed_42" 29855   # skip if done
run_baseline_batch 42 \
    "bpmult:$CFG_BPMULT" \
    "lflstm:$CFG_LFLSTM" \
    "emoclip:$CFG_EMOCLIP" \
    "ovmer:$CFG_OVMER"

# ── Seed 3407: all 9 models ──
echo ""
echo "=== seed=3407 ==="
run_argot 3407 "$RES/argot_ms/seed_3407" 29856
# Batch 1: 4 baselines
run_baseline_batch 3407 \
    "bpmult:$CFG_BPMULT" \
    "ctnet:$CFG_CTNET" \
    "mult:$CFG_MULT" \
    "mmdfn:$CFG_MMDFN"
# Batch 2: remaining 4 baselines
run_baseline_batch 3407 \
    "lflstm:$CFG_LFLSTM" \
    "aerllm:$CFG_AERLLM" \
    "emoclip:$CFG_EMOCLIP" \
    "ovmer:$CFG_OVMER"

# ── Seed 456: all 9 models ──
echo ""
echo "=== seed=456 ==="
run_argot 456 "$RES/argot_ms/seed_456" 29857
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
echo "  IEMOCAP COMPLETE — computing table"
echo "================================================================"
python3 "$PROJ/scripts/eval/compute_multiseed_table.py" --dataset iemocap || true
