#!/usr/bin/env bash
# =============================================================================
#  MELD — AR-GOT + 8 baselines, 3 seeds, ALL 10 GPUs (100% utilisation)
#
#  GPU layout (all 9 models run in PARALLEL per seed):
#    GPU 0,1  →  AR-GOT   (nproc=2, DDP)
#    GPU 2    →  BP-MulT
#    GPU 3    →  CTNet
#    GPU 4    →  MulT
#    GPU 5    →  MM-DFN
#    GPU 6    →  LF-LSTM
#    GPU 7    →  AER-LLM
#    GPU 8    →  EmoCLIP
#    GPU 9    →  OV-MER
#
#  Seeds: 42  3407  456    (seeds sequential; models parallel per seed)
#
#  Usage (VS Code terminal):
#    cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#    bash scripts/train/run_meld_3seed_final.sh 2>&1 | tee logs/meld_3seed.log
# =============================================================================
set -uo pipefail

PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
RES=$PROJ/experiments/results/meld
mkdir -p "$PROJ/logs"

TRAIN_CHADO=$PROJ/scripts/train/train_meld.py
TRAIN_BASE=$PROJ/scripts/train/train_baseline.py

CFG_ARGOT=$PROJ/configs/meld/chado_meld_final.yaml
CFG_BPMULT=$PROJ/configs/meld/bpmult_meld.yaml
CFG_CTNET=$PROJ/configs/meld/ctnet_meld.yaml
CFG_MULT=$PROJ/configs/meld/mult_meld.yaml
CFG_MMDFN=$PROJ/configs/meld/mmdfn_meld.yaml
CFG_LFLSTM=$PROJ/configs/meld/lflstm_meld.yaml
CFG_AERLLM=$PROJ/configs/meld/aerllm_meld.yaml
CFG_EMOCLIP=$PROJ/configs/meld/emoclip_meld.yaml
CFG_OVMER=$PROJ/configs/meld/ovmer_meld.yaml

SEEDS=(42 3407 456)

echo "================================================================"
echo "  MELD  — cleaning old multi-seed results"
echo "================================================================"
for model in argot bpmult ctnet mult mmdfn lflstm aerllm emoclip ovmer; do
    rm -rf "$RES/${model}_ms"
    echo "  removed $RES/${model}_ms"
done

# ── helpers ──────────────────────────────────────────────────────────────────
run_base() {
    local gpu=$1 cfg=$2 seed=$3 out=$4 port=$5
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu torchrun \
        --nproc_per_node=1 --master_port=$port \
        "$TRAIN_BASE" --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "  [OK ] $(basename $(dirname $out))/$(basename $out)" \
                  || echo "  [ERR] $(basename $(dirname $out))/$(basename $out)  exit=$rc"
    return $rc
}

run_argot() {
    local seed=$1 out=$2 port=$3
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=2 --master_port=$port \
        "$TRAIN_CHADO" --config "$CFG_ARGOT" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "  [OK ] argot_ms/seed_$seed" \
                  || echo "  [ERR] argot_ms/seed_$seed  exit=$rc"
    return $rc
}

# ── 3 seeds (sequential) — all 9 models in parallel per seed ─────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  MELD  seed=$SEED  — launching all 9 models in parallel"
    echo "================================================================"

    OUT_ARGOT=$RES/argot_ms/seed_$SEED
    OUT_BPMULT=$RES/bpmult_ms/seed_$SEED
    OUT_CTNET=$RES/ctnet_ms/seed_$SEED
    OUT_MULT=$RES/mult_ms/seed_$SEED
    OUT_MMDFN=$RES/mmdfn_ms/seed_$SEED
    OUT_LFLSTM=$RES/lflstm_ms/seed_$SEED
    OUT_AERLLM=$RES/aerllm_ms/seed_$SEED
    OUT_EMOCLIP=$RES/emoclip_ms/seed_$SEED
    OUT_OVMER=$RES/ovmer_ms/seed_$SEED

    run_argot "$SEED" "$OUT_ARGOT"  29520   & PID_ARGOT=$!
    run_base 2 "$CFG_BPMULT"  "$SEED" "$OUT_BPMULT"  29521 & PID_BPMULT=$!
    run_base 3 "$CFG_CTNET"   "$SEED" "$OUT_CTNET"   29522 & PID_CTNET=$!
    run_base 4 "$CFG_MULT"    "$SEED" "$OUT_MULT"    29523 & PID_MULT=$!
    run_base 5 "$CFG_MMDFN"   "$SEED" "$OUT_MMDFN"   29524 & PID_MMDFN=$!
    run_base 6 "$CFG_LFLSTM"  "$SEED" "$OUT_LFLSTM"  29525 & PID_LFLSTM=$!
    run_base 7 "$CFG_AERLLM"  "$SEED" "$OUT_AERLLM"  29526 & PID_AERLLM=$!
    run_base 8 "$CFG_EMOCLIP" "$SEED" "$OUT_EMOCLIP" 29527 & PID_EMOCLIP=$!
    run_base 9 "$CFG_OVMER"   "$SEED" "$OUT_OVMER"   29528 & PID_OVMER=$!

    wait $PID_ARGOT   || echo "  [WARN] AR-GOT   seed=$SEED failed"
    wait $PID_BPMULT  || echo "  [WARN] BP-MulT  seed=$SEED failed"
    wait $PID_CTNET   || echo "  [WARN] CTNet    seed=$SEED failed"
    wait $PID_MULT    || echo "  [WARN] MulT     seed=$SEED failed"
    wait $PID_MMDFN   || echo "  [WARN] MM-DFN   seed=$SEED failed"
    wait $PID_LFLSTM  || echo "  [WARN] LF-LSTM  seed=$SEED failed"
    wait $PID_AERLLM  || echo "  [WARN] AER-LLM  seed=$SEED failed"
    wait $PID_EMOCLIP || echo "  [WARN] EmoCLIP  seed=$SEED failed"
    wait $PID_OVMER   || echo "  [WARN] OV-MER   seed=$SEED failed"

    echo "  seed=$SEED  all models done"
done

echo ""
echo "================================================================"
echo "  MELD  — computing mean ± std table"
echo "================================================================"
python3 "$PROJ/scripts/eval/compute_multiseed_table.py" --dataset meld

echo ""
echo "MELD 3-seed run complete."
