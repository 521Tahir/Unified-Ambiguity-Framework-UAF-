#!/usr/bin/env bash
# =============================================================================
#  MOSEI — complete ALL missing seeds (skip done, use GPU 4)
#  Missing: argot/s456, ovmer/s42 (crashed), ovmer/s456
#  GPU: 4  (26 GB free right now)
#
#  Run:  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#        bash scripts/train/run_mosei_complete.sh 2>&1 | tee logs/mosei_complete.log
# =============================================================================
set -uo pipefail
PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
RES=$PROJ/experiments/results/mosei
mkdir -p "$PROJ/logs"

skip_done() {
    local out=$1
    [ -f "$out/test_results.json" ] && { echo "  [SKIP] $(basename $out) — already done"; return 0; }
    return 1
}

run1() {
    local gpu=$1 script=$2 cfg=$3 seed=$4 out=$5 port=$6
    skip_done "$out" && return 0
    mkdir -p "$out"
    echo "  [RUN ] GPU=$gpu  $(basename $(dirname $out))/seed_$seed  port=$port"
    CUDA_VISIBLE_DEVICES=$gpu torchrun \
        --nproc_per_node=1 --master_port=$port \
        "$script" --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "  [OK ] $(basename $(dirname $out))/seed_$seed" \
                  || echo "  [ERR] $(basename $(dirname $out))/seed_$seed  exit=$rc"
}

TRAIN=$PROJ/scripts/train/train_mosei.py
BASE=$PROJ/scripts/train/train_baseline_mosei.py
CFG_ARGOT=$PROJ/configs/mosei/chado_mosei_best.yaml
CFG_OVMER=$PROJ/configs/mosei/ovmer_mosei.yaml

echo "================================================================"
echo "  MOSEI COMPLETE — GPU 4"
echo "================================================================"

# argot: only s456 missing (s42 done, s3407 done)
echo ""
echo "--- AR-GOT seed 456 ---"
run1 4 "$TRAIN" "$CFG_ARGOT" 456 "$RES/argot_ms/seed_456" 29850

# ovmer: s42 crashed + s456 missing
echo ""
echo "--- OV-MER seed 42 (rerun crashed) ---"
rm -rf "$RES/ovmer_ms/seed_42"
run1 4 "$BASE" "$CFG_OVMER" 42 "$RES/ovmer_ms/seed_42" 29851

echo ""
echo "--- OV-MER seed 456 ---"
run1 4 "$BASE" "$CFG_OVMER" 456 "$RES/ovmer_ms/seed_456" 29852

echo ""
echo "================================================================"
echo "  MOSEI COMPLETE — done"
echo "================================================================"
python3 "$PROJ/scripts/eval/compute_multiseed_table.py" --dataset mosei || true
