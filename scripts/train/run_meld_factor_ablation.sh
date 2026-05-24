#!/usr/bin/env bash
# =============================================================================
#  MELD — Factor-level ablation (A.3), 5 variants × 3 seeds
#
#  GPU layout: GPU 0,1 / 2,3 / 4,5 / 6,7 / 8,9 → 5 variants (100% utilisation)
#
#  Usage (VS Code terminal):
#    cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#    bash scripts/train/run_meld_factor_ablation.sh 2>&1 | tee logs/meld_factor_ablation.log
# =============================================================================
set -uo pipefail

PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
RES=$PROJ/experiments/results/meld/ablations
mkdir -p "$PROJ/logs" "$RES"

TRAIN=$PROJ/scripts/train/train_meld.py
CFG_DIR=$PROJ/configs/meld

SEEDS=(42 3407 456)

echo "================================================================"
echo "  MELD  — factor ablation: cleaning old ablation results"
echo "================================================================"
rm -rf "$RES"
mkdir -p "$RES"

run_variant() {
    local gpus=$1 cfg=$2 seed=$3 out=$4 port=$5 label=$6
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpus torchrun \
        --nproc_per_node=2 --master_port=$port \
        "$TRAIN" --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "  [OK ] $label/seed_$seed" \
                  || echo "  [ERR] $label/seed_$seed  exit=$rc"
    return $rc
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  MELD factor ablation  seed=$SEED  — all 5 variants parallel"
    echo "================================================================"

    run_variant "0,1" "$CFG_DIR/ablation_wo_ctx.yaml"      "$SEED" "$RES/wo_ctx/seed_$SEED"      29620 wo_ctx      & P1=$!
    run_variant "2,3" "$CFG_DIR/ablation_wo_spk.yaml"      "$SEED" "$RES/wo_spk/seed_$SEED"      29622 wo_spk      & P2=$!
    run_variant "4,5" "$CFG_DIR/ablation_wo_turn.yaml"     "$SEED" "$RES/wo_turn/seed_$SEED"     29624 wo_turn     & P3=$!
    run_variant "6,7" "$CFG_DIR/ablation_wo_modal.yaml"    "$SEED" "$RES/wo_modal/seed_$SEED"    29626 wo_modal    & P4=$!
    run_variant "8,9" "$CFG_DIR/ablation_wo_residual.yaml" "$SEED" "$RES/wo_residual/seed_$SEED" 29628 wo_residual & P5=$!

    wait $P1 || echo "  [WARN] wo_ctx seed=$SEED failed"
    wait $P2 || echo "  [WARN] wo_spk seed=$SEED failed"
    wait $P3 || echo "  [WARN] wo_turn seed=$SEED failed"
    wait $P4 || echo "  [WARN] wo_modal seed=$SEED failed"
    wait $P5 || echo "  [WARN] wo_residual seed=$SEED failed"
    echo "  seed=$SEED done"
done

echo ""
echo "================================================================"
echo "  MELD factor ablation — evaluating"
echo "================================================================"
python3 "$PROJ/scripts/eval/run_A3_factor_ablation_eval.py" --dataset meld

echo ""
echo "MELD factor ablation complete."
