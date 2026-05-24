#!/usr/bin/env bash
# =============================================================================
#  Final ablation completion — all remaining seeds for IEMOCAP and MELD
#
#  Runs 4 parallel threads after waiting for GPUs to free up:
#    Thread A (GPU 0,1): IEMOCAP  5 runs sequential
#    Thread B (GPU 2,3): IEMOCAP  4 runs sequential
#    Thread C (GPU 4,5): MELD     6 runs sequential
#    Thread D (GPU 6,7): MELD     5 runs sequential
#
#  Safe: skip_done + wait_free + no rm -rf of active dirs
#
#  Run:  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
#        nohup bash scripts/train/run_ablation_complete_final.sh \
#              > logs/ablation_complete_final.log 2>&1 &
# =============================================================================
set -uo pipefail
PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
RES=$PROJ/experiments/results
mkdir -p "$PROJ/logs"

TRAIN_IEM=$PROJ/scripts/train/train_iemocap.py
TRAIN_MELD=$PROJ/scripts/train/train_meld.py

skip_done() {
    [ -f "$1/test_results.json" ] && { echo "  [SKIP] $1"; return 0; }
    return 1
}

# Wait until both GPUs have at least MIN_FREE_MiB free
wait_free() {
    local g1=$1 g2=$2 min_free=${3:-28000}
    while true; do
        local f1 f2
        f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g1" 2>/dev/null || echo 0)
        f2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g2" 2>/dev/null || echo 0)
        if [ "$f1" -ge "$min_free" ] && [ "$f2" -ge "$min_free" ]; then
            return 0
        fi
        echo "  [WAIT] GPU $g1 free=${f1}MiB, GPU $g2 free=${f2}MiB — need ${min_free}MiB. Sleeping 60s…"
        sleep 60
    done
}

run_ddp() {
    local gpu1=$1 gpu2=$2 script=$3 cfg=$4 seed=$5 out=$6 port=$7
    skip_done "$out" && return 0
    wait_free "$gpu1" "$gpu2"
    mkdir -p "$out"
    local tag
    tag="$(basename "$(dirname "$out")")/$(basename "$out")"
    echo "  [RUN ] $tag  GPU=$gpu1,$gpu2  port=$port"
    CUDA_VISIBLE_DEVICES="$gpu1,$gpu2" torchrun \
        --nproc_per_node=2 --master_port="$port" \
        "$script" --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] \
        && echo "  [OK ] $tag" \
        || echo "  [ERR] $tag  exit=$rc"
}

# ── Configs ──────────────────────────────────────────────────────────────────
CI=$PROJ/configs/iemocap
CM=$PROJ/configs/meld

# ── Thread A: GPU 0,1 — IEMOCAP (5 runs) ─────────────────────────────────────
thread_A() {
    echo "[A] IEMOCAP GPU 0,1"
    run_ddp 0 1 "$TRAIN_IEM" "$CI/ablation_wo_ctx.yaml"      42   "$RES/iemocap/ablations/wo_ctx/seed_42"      29980
    run_ddp 0 1 "$TRAIN_IEM" "$CI/ablation_wo_ctx.yaml"      3407 "$RES/iemocap/ablations/wo_ctx/seed_3407"    29981
    run_ddp 0 1 "$TRAIN_IEM" "$CI/ablation_wo_turn.yaml"     42   "$RES/iemocap/ablations/wo_turn/seed_42"     29982
    run_ddp 0 1 "$TRAIN_IEM" "$CI/ablation_wo_turn.yaml"     3407 "$RES/iemocap/ablations/wo_turn/seed_3407"   29983
    run_ddp 0 1 "$TRAIN_IEM" "$CI/ablation_wo_residual.yaml" 42   "$RES/iemocap/ablations/wo_residual/seed_42" 29984
    echo "[A] Done"
}

# ── Thread B: GPU 2,3 — IEMOCAP (4 runs) ─────────────────────────────────────
thread_B() {
    echo "[B] IEMOCAP GPU 2,3"
    run_ddp 2 3 "$TRAIN_IEM" "$CI/ablation_wo_spk.yaml"      3407 "$RES/iemocap/ablations/wo_spk/seed_3407"      29985
    run_ddp 2 3 "$TRAIN_IEM" "$CI/ablation_wo_modal.yaml"    42   "$RES/iemocap/ablations/wo_modal/seed_42"      29986
    run_ddp 2 3 "$TRAIN_IEM" "$CI/ablation_wo_modal.yaml"    3407 "$RES/iemocap/ablations/wo_modal/seed_3407"    29987
    run_ddp 2 3 "$TRAIN_IEM" "$CI/ablation_wo_residual.yaml" 3407 "$RES/iemocap/ablations/wo_residual/seed_3407" 29988
    echo "[B] Done"
}

# ── Thread C: GPU 4,5 — MELD (6 runs) ────────────────────────────────────────
thread_C() {
    echo "[C] MELD GPU 4,5"
    run_ddp 4 5 "$TRAIN_MELD" "$CM/ablation_wo_ctx.yaml"      42   "$RES/meld/ablations/wo_ctx/seed_42"      29990
    run_ddp 4 5 "$TRAIN_MELD" "$CM/ablation_wo_ctx.yaml"      3407 "$RES/meld/ablations/wo_ctx/seed_3407"    29991
    run_ddp 4 5 "$TRAIN_MELD" "$CM/ablation_wo_ctx.yaml"      456  "$RES/meld/ablations/wo_ctx/seed_456"     29992
    run_ddp 4 5 "$TRAIN_MELD" "$CM/ablation_wo_turn.yaml"     42   "$RES/meld/ablations/wo_turn/seed_42"     29993
    run_ddp 4 5 "$TRAIN_MELD" "$CM/ablation_wo_turn.yaml"     456  "$RES/meld/ablations/wo_turn/seed_456"    29994
    run_ddp 4 5 "$TRAIN_MELD" "$CM/ablation_wo_residual.yaml" 42   "$RES/meld/ablations/wo_residual/seed_42" 29995
    echo "[C] Done"
}

# ── Thread D: GPU 6,7 — MELD (5 runs) ────────────────────────────────────────
thread_D() {
    echo "[D] MELD GPU 6,7"
    run_ddp 6 7 "$TRAIN_MELD" "$CM/ablation_wo_spk.yaml"      456  "$RES/meld/ablations/wo_spk/seed_456"       29996
    run_ddp 6 7 "$TRAIN_MELD" "$CM/ablation_wo_modal.yaml"    42   "$RES/meld/ablations/wo_modal/seed_42"      29997
    run_ddp 6 7 "$TRAIN_MELD" "$CM/ablation_wo_modal.yaml"    456  "$RES/meld/ablations/wo_modal/seed_456"     29998
    run_ddp 6 7 "$TRAIN_MELD" "$CM/ablation_wo_residual.yaml" 3407 "$RES/meld/ablations/wo_residual/seed_3407" 29999
    run_ddp 6 7 "$TRAIN_MELD" "$CM/ablation_wo_residual.yaml" 456  "$RES/meld/ablations/wo_residual/seed_456"  30000
    echo "[D] Done"
}

echo "============================================================"
echo "  ABLATION COMPLETE — 4 parallel GPU threads"
echo "  IEMOCAP: 9 remaining  |  MELD: 11 remaining"
echo "  (skip_done + wait_free ensures no conflicts)"
echo "============================================================"

thread_A &
PID_A=$!
thread_B &
PID_B=$!
thread_C &
PID_C=$!
thread_D &
PID_D=$!

wait $PID_A && echo "[A] finished OK" || echo "[A] finished with errors"
wait $PID_B && echo "[B] finished OK" || echo "[B] finished with errors"
wait $PID_C && echo "[C] finished OK" || echo "[C] finished with errors"
wait $PID_D && echo "[D] finished OK" || echo "[D] finished with errors"

echo ""
echo "============================================================"
echo "  ALL THREADS DONE — computing ablation tables"
echo "============================================================"
python3 "$PROJ/scripts/eval/run_A3_factor_ablation_eval.py" --dataset iemocap || true
python3 "$PROJ/scripts/eval/run_A3_factor_ablation_eval.py" --dataset meld    || true
