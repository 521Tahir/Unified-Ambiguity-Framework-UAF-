#!/usr/bin/env bash
 bash scripts/sensitivity/run_sensitivity_sweep.sh 2>&1 | tee experiments/logs/sensitivity_sweep.log

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN="$ROOT/scripts/train/train_mosei.py"
CFG="$ROOT/configs/mosei/sensitivity"
LOG="$ROOT/experiments/logs/sensitivity"
mkdir -p "$LOG"

GPUS=(5 6 7 8 9)
# Unique ports for each slot (free range verified)
PORTS=(30110 30120 30130 30140 30150)

CONFIGS=(
  "$CFG/mad_00.yaml"
  "$CFG/mad_01.yaml"
  "$CFG/mad_03.yaml"
  "$CFG/mad_05.yaml"
  "$CFG/mad_07.yaml"
  "$CFG/mad_10.yaml"
  "$CFG/ot_001.yaml"
  "$CFG/ot_003.yaml"
  "$CFG/ot_005.yaml"
  "$CFG/ot_01.yaml"
  "$CFG/hyp_001.yaml"
  "$CFG/hyp_005.yaml"
  "$CFG/hyp_01.yaml"
  "$CFG/hyp_02.yaml"
)

echo "================================================================"
echo "  CMU-MOSEI Sensitivity Sweep — $(date)"
echo "  ${#CONFIGS[@]} configs, ${#GPUS[@]} parallel GPU slots"
echo "================================================================"

total=${#CONFIGS[@]}
i=0

while [ $i -lt $total ]; do
  # ── Launch batch ──────────────────────────────────────────────────────────
  pids=()
  names=()
  slot=0
  while [ $slot -lt ${#GPUS[@]} ] && [ $i -lt $total ]; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    gpu="${GPUS[$slot]}"
    port="${PORTS[$slot]}"

    echo "  [START] $name   GPU=$gpu  port=$port"
    CUDA_VISIBLE_DEVICES=$gpu \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    torchrun --nproc_per_node=1 --master_port=$port \
        "$TRAIN" --config "$cfg" \
        > "$LOG/${name}.log" 2>&1 &
    pids+=($!)
    names+=($name)
    ((slot++)) || true
    ((i++))    || true
  done

  # ── Wait for batch ────────────────────────────────────────────────────────
  for j in "${!pids[@]}"; do
    pid="${pids[$j]}"
    name="${names[$j]}"
    if wait "$pid"; then
      echo "  [DONE ] $name (PID=$pid)"
    else
      echo "  [FAIL ] $name (PID=$pid) — check $LOG/${name}.log"
    fi
  done
  echo "  Batch complete — $(date)"
  echo ""
done

echo "================================================================"
echo "  All configs done — collecting results..."
echo "================================================================"
python3 "$ROOT/scripts/sensitivity/collect_sensitivity_results.py"
python3 "$ROOT/scripts/plots/plot_sensitivity.py" 2>/dev/null && echo "  Sensitivity plot saved."
echo "  Sweep finished — $(date)"
