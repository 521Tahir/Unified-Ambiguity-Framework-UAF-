#!/usr/bin/env bash
# =============================================================================
#  RECOVERY — runs missing seed_42 baselines + MOSEI seed_3407 gaps
#  Uses ONLY GPUs 6 and 8 (safe — other GPUs occupied)
#
#  Run from VS Code terminal:
#    cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#    bash scripts/train/run_recovery.sh 2>&1 | tee logs/recovery.log
# =============================================================================
set -uo pipefail
PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
mkdir -p "$PROJ/logs"

# ── helpers ───────────────────────────────────────────────────────────────────
run1() {
    local gpu=$1 script=$2 cfg=$3 seed=$4 out=$5 port=$6
    mkdir -p "$out"
    if [ -f "$out/test_results.json" ]; then
        echo "  [SKIP] $(basename $(dirname $out))/$(basename $out) — already done"
        return 0
    fi
    echo "  [RUN ] GPU=$gpu $(basename $(dirname $out))/seed_$seed  port=$port"
    CUDA_VISIBLE_DEVICES=$gpu torchrun \
        --nproc_per_node=1 --master_port=$port \
        "$script" --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "  [OK ] $(basename $(dirname $out))/seed_$seed" \
                  || echo "  [ERR] $(basename $(dirname $out))/seed_$seed  exit=$rc"
    return $rc
}

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  PHASE 1 — MOSEI ovmer seed_3407  (GPU 6)"
echo "================================================================"
run1 6 \
    $PROJ/scripts/train/train_baseline_mosei.py \
    $PROJ/configs/mosei/ovmer_mosei.yaml \
    3407 \
    $PROJ/experiments/results/mosei/ovmer_ms/seed_3407 \
    29700

echo ""
echo "================================================================"
echo "  PHASE 2 — MOSEI argot seed_3407  (GPU 6, single-GPU mode)"
echo "================================================================"
run1 6 \
    $PROJ/scripts/train/train_mosei.py \
    $PROJ/configs/mosei/chado_mosei_best.yaml \
    3407 \
    $PROJ/experiments/results/mosei/argot_ms/seed_3407 \
    29701

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  PHASE 3 — IEMOCAP missing baselines seed_42  (GPU 6 & 8)"
echo "================================================================"
ITRAIN_BASE=$PROJ/scripts/train/train_baseline.py
IRES=$PROJ/experiments/results/iemocap

# Batch A: bpmult (GPU 6) + ctnet (GPU 8) in parallel
echo "  Batch A: bpmult + ctnet"
run1 6 "$ITRAIN_BASE" $PROJ/configs/iemocap/bpmult_iemocap.yaml 42 $IRES/bpmult_ms/seed_42 29710 &  PA1=$!
run1 8 "$ITRAIN_BASE" $PROJ/configs/iemocap/ctnet_iemocap.yaml  42 $IRES/ctnet_ms/seed_42  29711 &  PA2=$!
wait $PA1 || echo "  [WARN] bpmult s42"
wait $PA2 || echo "  [WARN] ctnet s42"

# Batch B: lflstm (GPU 6) + emoclip (GPU 8) in parallel
echo "  Batch B: lflstm + emoclip"
run1 6 "$ITRAIN_BASE" $PROJ/configs/iemocap/lflstm_iemocap.yaml  42 $IRES/lflstm_ms/seed_42  29712 &  PB1=$!
run1 8 "$ITRAIN_BASE" $PROJ/configs/iemocap/emoclip_iemocap.yaml 42 $IRES/emoclip_ms/seed_42 29713 &  PB2=$!
wait $PB1 || echo "  [WARN] lflstm s42"
wait $PB2 || echo "  [WARN] emoclip s42"

# Batch C: ovmer (GPU 6)
echo "  Batch C: ovmer"
run1 6 "$ITRAIN_BASE" $PROJ/configs/iemocap/ovmer_iemocap.yaml 42 $IRES/ovmer_ms/seed_42 29714

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  PHASE 4 — MELD missing baselines seed_42  (GPU 6 & 8)"
echo "================================================================"
MTRAIN_BASE=$PROJ/scripts/train/train_baseline.py
MRES=$PROJ/experiments/results/meld

# Batch A: bpmult (GPU 6) + ctnet (GPU 8) in parallel
echo "  Batch A: bpmult + ctnet"
run1 6 "$MTRAIN_BASE" $PROJ/configs/meld/bpmult_meld.yaml 42 $MRES/bpmult_ms/seed_42 29720 &  PA1=$!
run1 8 "$MTRAIN_BASE" $PROJ/configs/meld/ctnet_meld.yaml  42 $MRES/ctnet_ms/seed_42  29721 &  PA2=$!
wait $PA1 || echo "  [WARN] meld bpmult s42"
wait $PA2 || echo "  [WARN] meld ctnet s42"

# Batch B: mult (GPU 6) + mmdfn (GPU 8) in parallel
echo "  Batch B: mult + mmdfn"
run1 6 "$MTRAIN_BASE" $PROJ/configs/meld/mult_meld.yaml  42 $MRES/mult_ms/seed_42  29722 &  PB1=$!
run1 8 "$MTRAIN_BASE" $PROJ/configs/meld/mmdfn_meld.yaml 42 $MRES/mmdfn_ms/seed_42 29723 &  PB2=$!
wait $PB1 || echo "  [WARN] meld mult s42"
wait $PB2 || echo "  [WARN] meld mmdfn s42"

# Batch C: lflstm (GPU 6) + emoclip (GPU 8) in parallel
echo "  Batch C: lflstm + emoclip"
run1 6 "$MTRAIN_BASE" $PROJ/configs/meld/lflstm_meld.yaml  42 $MRES/lflstm_ms/seed_42  29724 &  PC1=$!
run1 8 "$MTRAIN_BASE" $PROJ/configs/meld/emoclip_meld.yaml 42 $MRES/emoclip_ms/seed_42 29725 &  PC2=$!
wait $PC1 || echo "  [WARN] meld lflstm s42"
wait $PC2 || echo "  [WARN] meld emoclip s42"

echo ""
echo "================================================================"
echo "  RECOVERY COMPLETE"
echo "================================================================"
python3 "$PROJ/scripts/eval/compute_multiseed_table.py" --dataset mosei || true
