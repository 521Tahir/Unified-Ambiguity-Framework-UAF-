# #!/usr/bin/env bash
# # Run BP-MulT baseline on IEMOCAP with 3 seeds (42, 3407, 2024) sequentially
# # on 5 GPUs, then print mean ± std of Accuracy and Macro-F1.
# set -e

# PROJ=/home/tahirahmad/Project_Code/CHADO_EMNLP
# CFG=$PROJ/configs/iemocap/bpmult_iemocap.yaml
# SCRIPT=$PROJ/scripts/train/train_baseline.py
# GPUS=5,6,7,8,9
# NPROC=5

# SEEDS=(42 3407 2024)
# OUT_DIRS=(
#   "$PROJ/experiments/results/iemocap/bpmult_seed42"
#   "$PROJ/experiments/results/iemocap/bpmult_seed3407"
#   "$PROJ/experiments/results/iemocap/bpmult_seed2024"
# )

# # ── Create output directories ─────────────────────────────────────────────────
# for d in "${OUT_DIRS[@]}"; do
#   mkdir -p "$d"
# done

# # ── Run seeds sequentially ────────────────────────────────────────────────────
# for i in 0 1 2; do
#   SEED=${SEEDS[$i]}
#   OUT=${OUT_DIRS[$i]}
#   PORT=$((29720 + i))

#   echo ""
#   echo "============================================================"
#   echo "  Running seed=$SEED  →  $OUT"
#   echo "============================================================"

#   CUDA_VISIBLE_DEVICES=$GPUS torchrun \
#     --nproc_per_node=$NPROC \
#     --master_port=$PORT \
#     "$SCRIPT" \
#     --config "$CFG" \
#     --seed "$SEED" \
#     --out-dir "$OUT" \
#     2>&1 | tee "$OUT/train.log"

#   echo ""
#   echo "  [seed=$SEED] Done. Test results:"
#   python3 -c "
# import json
# r = json.load(open('$OUT/test_results.json'))
# print(f'    acc={r[\"accuracy\"]:.4f}  macro_f1={r[\"macro_f1\"]:.4f}  weighted_f1={r[\"weighted_f1\"]:.4f}')
# "
# done

# # ── Summary: mean ± std ───────────────────────────────────────────────────────
# echo ""
# echo "============================================================"
# echo "  BP-MulT IEMOCAP — 3-Seed Summary"
# echo "============================================================"
# python3 - <<'PYEOF'
# import json, numpy as np

# seeds   = [42, 3407, 2024]
# dirs    = [
#     "experiments/results/iemocap/bpmult_seed42",
#     "experiments/results/iemocap/bpmult_seed3407",
#     "experiments/results/iemocap/bpmult_seed2024",
# ]

# accs, mf1s, wf1s = [], [], []
# print(f"  {'Seed':<8} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
# print("  " + "-"*44)
# for s, d in zip(seeds, dirs):
#     r = json.load(open(f"{d}/test_results.json"))
#     accs.append(r["accuracy"])
#     mf1s.append(r["macro_f1"])
#     wf1s.append(r["weighted_f1"])
#     print(f"  {s:<8} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")

# print("  " + "-"*44)
# print(f"  {'Mean':<8} {np.mean(accs):>10.4f} {np.mean(mf1s):>10.4f} {np.mean(wf1s):>12.4f}")
# print(f"  {'Std':<8} {np.std(accs):>10.4f} {np.std(mf1s):>10.4f} {np.std(wf1s):>12.4f}")
# print()
# print(f"  Accuracy  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
# print(f"  Macro-F1  : {np.mean(mf1s)*100:.2f} ± {np.std(mf1s)*100:.2f}")
# print(f"  Weighted-F1: {np.mean(wf1s)*100:.2f} ± {np.std(wf1s)*100:.2f}")
# PYEOF
#!/bin/bash

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

export TORCHINDUCTOR_FREEZING=1

CONFIG=/home/tahirahmad/Project_Code/CHADO_EMNLP/configs/iemocap/bpmult_iemocap.yaml

SEEDS=(42 2024 777)

for SEED in "${SEEDS[@]}"
do

echo "===================================================="
echo "Running BPMulT seed=$SEED"
echo "===================================================="

torchrun \
--standalone \
--nproc_per_node=5 \
--master_port=$((29500 + RANDOM % 1000)) \
train_bpmult_ddp.py \
--config $CONFIG \
--seed $SEED

done