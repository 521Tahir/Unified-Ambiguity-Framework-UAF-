#!/usr/bin/env bash
# ============================================================
#  IEMOCAP — AR-GOT + 8 baselines, 3 seeds, ALL 10 GPUs
#  GPU layout (100% parallel, all 10 GPUs busy simultaneously):
#    GPUs 0,1  → AR-GOT   (nproc=2, roberta-large)
#    GPU  2    → BP-MulT  (nproc=1)
#    GPU  3    → CTNet    (nproc=1)
#    GPU  4    → MulT     (nproc=1)
#    GPU  5    → MM-DFN   (nproc=1)
#    GPU  6    → LF-LSTM  (nproc=1)
#    GPU  7    → AER-LLM  (nproc=1)
#    GPU  8    → EmoCLIP  (nproc=1)
#    GPU  9    → OV-MER   (nproc=1)
#  All 9 models run in parallel per seed; seeds 42→3407→2024
#  sequential.
# ============================================================
set -e

PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
TRAIN_CHADO=$PROJ/scripts/train/train_iemocap.py
TRAIN_BASE=$PROJ/scripts/train/train_baseline.py

CFG_CHADO=$PROJ/configs/iemocap/chado_iemocap_v2.yaml
CFG_BPMULT=$PROJ/configs/iemocap/bpmult_iemocap.yaml
CFG_CTNET=$PROJ/configs/iemocap/ctnet_iemocap.yaml
CFG_MULT=$PROJ/configs/iemocap/mult_iemocap.yaml
CFG_MMDFN=$PROJ/configs/iemocap/mmdfn_iemocap.yaml
CFG_LFLSTM=$PROJ/configs/iemocap/lflstm_iemocap.yaml
CFG_AERLLM=$PROJ/configs/iemocap/aerllm_iemocap.yaml
CFG_EMOCLIP=$PROJ/configs/iemocap/emoclip_iemocap.yaml
CFG_OVMER=$PROJ/configs/iemocap/ovmer_iemocap.yaml

SEEDS=(42 3407 2024)
RES=$PROJ/experiments/results/iemocap

# ── helper: single-GPU baseline ────────────────────────────
run_base() {
    local gpu=$1 cfg=$2 seed=$3 out=$4 port=$5
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu torchrun \
        --nproc_per_node=1 --master_port=$port \
        "$TRAIN_BASE" \
        --config "$cfg" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
}

# ── helper: AR-GOT (2 GPUs) ───────────────────────────────
run_chado() {
    local seed=$1 out=$2
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=2 --master_port=29810 \
        "$TRAIN_CHADO" \
        --config "$CFG_CHADO" --seed "$seed" --out-dir "$out" \
        > "$out/train.log" 2>&1
}

# ── 3 seeds ────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  IEMOCAP  seed=$SEED  — launching all 9 models in parallel"
    echo "============================================================"

    OUT_CHADO=$RES/argot_ms/seed_$SEED
    OUT_BPMULT=$RES/bpmult_ms/seed_$SEED
    OUT_CTNET=$RES/ctnet_ms/seed_$SEED
    OUT_MULT=$RES/mult_ms/seed_$SEED
    OUT_MMDFN=$RES/mmdfn_ms/seed_$SEED
    OUT_LFLSTM=$RES/lflstm_ms/seed_$SEED
    OUT_AERLLM=$RES/aerllm_ms/seed_$SEED
    OUT_EMOCLIP=$RES/emoclip_ms/seed_$SEED
    OUT_OVMER=$RES/ovmer_ms/seed_$SEED

    # Launch ALL in background — all 10 GPUs occupied simultaneously
    run_chado   "$SEED" "$OUT_CHADO"                                  & PID_CHADO=$!
    run_base 2  "$CFG_BPMULT"  "$SEED" "$OUT_BPMULT"  29811          & PID_BPMULT=$!
    run_base 3  "$CFG_CTNET"   "$SEED" "$OUT_CTNET"   29812          & PID_CTNET=$!
    run_base 4  "$CFG_MULT"    "$SEED" "$OUT_MULT"    29813          & PID_MULT=$!
    run_base 5  "$CFG_MMDFN"   "$SEED" "$OUT_MMDFN"   29814          & PID_MMDFN=$!
    run_base 6  "$CFG_LFLSTM"  "$SEED" "$OUT_LFLSTM"  29815          & PID_LFLSTM=$!
    run_base 7  "$CFG_AERLLM"  "$SEED" "$OUT_AERLLM"  29816          & PID_AERLLM=$!
    run_base 8  "$CFG_EMOCLIP" "$SEED" "$OUT_EMOCLIP" 29817          & PID_EMOCLIP=$!
    run_base 9  "$CFG_OVMER"   "$SEED" "$OUT_OVMER"   29818          & PID_OVMER=$!

    echo "  PIDs: AR-GOT=$PID_CHADO  BP-MulT=$PID_BPMULT  CTNet=$PID_CTNET"
    echo "        MulT=$PID_MULT  MM-DFN=$PID_MMDFN  LF-LSTM=$PID_LFLSTM"
    echo "        AER-LLM=$PID_AERLLM  EmoCLIP=$PID_EMOCLIP  OV-MER=$PID_OVMER"

    # Wait for ALL jobs this seed
    wait $PID_CHADO   && echo "  [seed=$SEED] AR-GOT   done" || echo "  [WARN] AR-GOT   failed seed=$SEED"
    wait $PID_BPMULT  && echo "  [seed=$SEED] BP-MulT  done" || echo "  [WARN] BP-MulT  failed seed=$SEED"
    wait $PID_CTNET   && echo "  [seed=$SEED] CTNet    done" || echo "  [WARN] CTNet    failed seed=$SEED"
    wait $PID_MULT    && echo "  [seed=$SEED] MulT     done" || echo "  [WARN] MulT     failed seed=$SEED"
    wait $PID_MMDFN   && echo "  [seed=$SEED] MM-DFN   done" || echo "  [WARN] MM-DFN   failed seed=$SEED"
    wait $PID_LFLSTM  && echo "  [seed=$SEED] LF-LSTM  done" || echo "  [WARN] LF-LSTM  failed seed=$SEED"
    wait $PID_AERLLM  && echo "  [seed=$SEED] AER-LLM  done" || echo "  [WARN] AER-LLM  failed seed=$SEED"
    wait $PID_EMOCLIP && echo "  [seed=$SEED] EmoCLIP  done" || echo "  [WARN] EmoCLIP  failed seed=$SEED"
    wait $PID_OVMER   && echo "  [seed=$SEED] OV-MER   done" || echo "  [WARN] OV-MER   failed seed=$SEED"
done

# ── Summary: mean ± std across 3 seeds ────────────────────
echo ""
echo "============================================================"
echo "  IEMOCAP — 3-Seed Summary (AR-GOT vs All Baselines)"
echo "============================================================"
python3 - <<'PYEOF'
import json, numpy as np, os, sys

BASE  = '/home/tahirahmad/Project_Code/CHADO_EMNLP/experiments/results/iemocap'
SEEDS = [42, 3407, 2024]
MODELS = [
    ('AR-GOT',   'argot_ms'),
    ('BP-MulT',  'bpmult_ms'),
    ('CTNet',    'ctnet_ms'),
    ('MulT',     'mult_ms'),
    ('MM-DFN',   'mmdfn_ms'),
    ('LF-LSTM',  'lflstm_ms'),
    ('AER-LLM',  'aerllm_ms'),
    ('EmoCLIP',  'emoclip_ms'),
    ('OV-MER',   'ovmer_ms'),
]

print(f"\n  {'Model':<12} {'Acc (%)':>12}  {'Macro-F1 (%)':>14}  {'Weighted-F1 (%)':>16}")
print('  ' + '─'*58)

best_acc = best_mf1 = best_wf1 = 0.0
best_acc_m = best_mf1_m = best_wf1_m = ''

rows = []
for name, folder in MODELS:
    accs, mf1s, wf1s = [], [], []
    for s in SEEDS:
        p = f'{BASE}/{folder}/seed_{s}/test_results.json'
        if os.path.exists(p):
            r = json.load(open(p))
            accs.append(r['accuracy'])
            mf1s.append(r['macro_f1'])
            wf1s.append(r['weighted_f1'])
    if not accs:
        rows.append((name, None, None, None, None, None, None))
        continue
    ma, sa = np.mean(accs)*100, np.std(accs)*100
    mm, sm = np.mean(mf1s)*100, np.std(mf1s)*100
    mw, sw = np.mean(wf1s)*100, np.std(wf1s)*100
    rows.append((name, ma, sa, mm, sm, mw, sw))
    if ma > best_acc: best_acc, best_acc_m = ma, name
    if mm > best_mf1: best_mf1, best_mf1_m = mm, name
    if mw > best_wf1: best_wf1, best_wf1_m = mw, name

for name, ma, sa, mm, sm, mw, sw in rows:
    if ma is None:
        print(f"  {name:<12} {'—':>12}  {'—':>14}  {'—':>16}   (no results)")
        continue
    flag_a = ' ◀ BEST' if name == best_acc_m else ''
    flag_m = ' ◀ BEST' if name == best_mf1_m else ''
    flag_w = ' ◀ BEST' if name == best_wf1_m else ''
    print(f"  {name:<12} {ma:6.2f}±{sa:4.2f}{flag_a:8s}  "
          f"{mm:6.2f}±{sm:4.2f}{flag_m:8s}  "
          f"{mw:6.2f}±{sw:4.2f}{flag_w}")

print()
argot_row = next((r for r in rows if r[0]=='AR-GOT'), None)
if argot_row and argot_row[1] is not None:
    ma,sa,mm,sm,mw,sw = argot_row[1:]
    wins = sum([best_acc_m=='AR-GOT', best_mf1_m=='AR-GOT', best_wf1_m=='AR-GOT'])
    print(f"  AR-GOT wins {wins}/3 metrics.")
    print(f"  Acc={ma:.2f}±{sa:.2f}  mF1={mm:.2f}±{sm:.2f}  wF1={mw:.2f}±{sw:.2f}")
PYEOF
