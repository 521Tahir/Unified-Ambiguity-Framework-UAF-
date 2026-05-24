#!/usr/bin/env bash
# ============================================================
# Component Ablation Study — IEMOCAP, MELD, CMU-MOSEI
# Removes one CHADO component at a time:
#   wo_causal | wo_hyperbolic | wo_ot | wo_mad (MOSEI)
# Full CHADO and Baseline results already exist — skipped.
# Uses GPUs 0,1,2,3 (currently free); falls back to 5-9 if busy.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR=/tmp/component_ablation
mkdir -p "$LOG_DIR"

RESULTS_ROOT="$ROOT/experiments/results"
CFG_TMP="$LOG_DIR"

# Two GPU groups — run 2 ablations in parallel (2 GPUs each)
GPU_GROUP_A="0,1"
GPU_GROUP_B="2,3"
NPROC=2

GPU_FREE_MIB=8000   # consider GPU free if used < this MiB
POLL_SECS=60

# ── GPU availability ──────────────────────────────────────────────────────────
gpus_free() {
    local gpus="$1"
    local ok=1
    IFS=',' read -ra arr <<< "$gpus"
    for g in "${arr[@]}"; do
        used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        [[ "$used" =~ ^[0-9]+$ ]] && (( used >= GPU_FREE_MIB )) && ok=0
    done
    echo $ok
}

wait_gpus() {
    local gpus="$1" label="$2"
    while [[ "$(gpus_free "$gpus")" -eq 0 ]]; do
        echo "[WAIT] GPUs $gpus busy for '$label' — retry in ${POLL_SECS}s ..."
        sleep $POLL_SECS
    done
    echo "[GPU OK] $gpus free for '$label'"
}

# ── Config builder ────────────────────────────────────────────────────────────
build_cfg() {
    local base="$1" out_dir="$2" run_name="$3"
    shift 3
    # remaining args: key=value pairs to override in chado section
    local tmp="$CFG_TMP/cfg_${run_name}.yaml"
    python3 - "$base" "$tmp" "$out_dir" "$run_name" "$@" << 'PY'
import sys, yaml, os
base, tmp, out_dir, run_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
overrides = {}
for kv in sys.argv[5:]:
    k, v = kv.split("=", 1)
    overrides[k] = (v.lower() == "true") if v.lower() in ("true","false") else v

with open(base) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("chado", {})
cfg["chado"].update(overrides)
cfg.setdefault("logging", {})
cfg["logging"]["run_name"] = run_name
cfg["logging"]["out_dir"]  = out_dir

# Use same epochs as Full CHADO — do NOT reduce (ensures fair comparison)
if "train" in cfg:
    cfg["train"]["patience"] = max(10, cfg["train"].get("patience", 10))

os.makedirs(os.path.dirname(tmp), exist_ok=True)
with open(tmp, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
PY
    echo "$tmp"
}

# ── Single-run launcher ───────────────────────────────────────────────────────
run_ablation() {
    local dataset="$1" name="$2" gpus="$3" script="$4" cfg="$5" out_dir="$6"
    local log="$LOG_DIR/${dataset}_${name}.log"

    if [[ -f "${out_dir}/test_results.json" ]]; then
        echo "[SKIP] ${dataset}/${name} already done"
        return 0
    fi

    wait_gpus "$gpus" "${dataset}_${name}"

    local port=$((29800 + RANDOM % 200))
    echo "[LAUNCH] ${dataset}/${name}  GPUs=$gpus  port=$port"
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES="$gpus" \
    torchrun --nproc_per_node=$NPROC --master_port=$port \
        "$script" --config "$cfg" > "$log" 2>&1
    local exit_code=$?

    if [[ $exit_code -eq 0 && -f "${out_dir}/test_results.json" ]]; then
        metrics=$(python3 -c "
import json
d = json.load(open('${out_dir}/test_results.json'))
acc  = d.get('accuracy', d.get('wacc','N/A'))
mf1  = d.get('macro_f1','N/A')
wf1  = d.get('weighted_f1','N/A')
if isinstance(acc,float): acc=f'{acc:.4f}'
if isinstance(mf1,float): mf1=f'{mf1:.4f}'
if isinstance(wf1,float): wf1=f'{wf1:.4f}'
print(f'acc={acc}  macro_f1={mf1}  wf1={wf1}')
" 2>/dev/null || echo "metrics parse error")
        echo "[DONE] ${dataset}/${name}  $metrics"
    else
        echo "[FAIL] ${dataset}/${name}  exit=$exit_code"
    fi
}

# ── Parallel pair runner (A and B groups) ─────────────────────────────────────
run_pair() {
    # Run two ablations in parallel on GPU group A and B
    local ds_a="$1" name_a="$2" script_a="$3" cfg_a="$4" out_a="$5"
    local ds_b="$6" name_b="$7" script_b="$8" cfg_b="$9" out_b="${10}"

    run_ablation "$ds_a" "$name_a" "$GPU_GROUP_A" "$script_a" "$cfg_a" "$out_a" &
    local pid_a=$!
    run_ablation "$ds_b" "$name_b" "$GPU_GROUP_B" "$script_b" "$cfg_b" "$out_b" &
    local pid_b=$!
    wait $pid_a $pid_b
}

# ═══════════════════════════════════════════════════════════════
echo "========================================================"
echo " COMPONENT ABLATION STUDY"
echo " Datasets: IEMOCAP | MELD | CMU-MOSEI"
echo " Variants: wo_causal | wo_hyperbolic | wo_ot | wo_mad"
echo "========================================================"
echo ""

# ── Scripts ───────────────────────────────────────────────────────────────────
SCR_IEM="scripts/train/train_iemocap.py"
SCR_MEL="scripts/train/train_meld.py"
SCR_MOS="scripts/train/train_mosei.py"
CFG_IEM="configs/iemocap/chado_iemocap.yaml"
CFG_MEL="configs/meld/chado_meld.yaml"
CFG_MOS="configs/mosei/chado_mosei.yaml"

# ── ROUND 1: wo_causal — IEMOCAP(A) + MELD(B) in parallel ───────────────────
echo "── Round 1: wo_causal ──────────────────────────────────"
cfg_iem_woc=$(build_cfg "$CFG_IEM" \
    "$RESULTS_ROOT/iemocap/chado_wo_causal" "chado_iemocap_wo_causal" \
    "use_causal=false" "use_hyperbolic=true" "use_ot=true")
cfg_mel_woc=$(build_cfg "$CFG_MEL" \
    "$RESULTS_ROOT/meld/chado_wo_causal" "chado_meld_wo_causal" \
    "use_causal=false" "use_hyperbolic=true" "use_ot=true")
run_pair \
    iemocap chado_wo_causal "$SCR_IEM" "$cfg_iem_woc" "$RESULTS_ROOT/iemocap/chado_wo_causal" \
    meld    chado_wo_causal "$SCR_MEL" "$cfg_mel_woc" "$RESULTS_ROOT/meld/chado_wo_causal"

# ── ROUND 2: wo_hyperbolic — IEMOCAP(A) + MELD(B) ───────────────────────────
echo "── Round 2: wo_hyperbolic ──────────────────────────────"
cfg_iem_woh=$(build_cfg "$CFG_IEM" \
    "$RESULTS_ROOT/iemocap/chado_wo_hyperbolic" "chado_iemocap_wo_hyperbolic" \
    "use_causal=true" "use_hyperbolic=false" "use_ot=true")
cfg_mel_woh=$(build_cfg "$CFG_MEL" \
    "$RESULTS_ROOT/meld/chado_wo_hyperbolic" "chado_meld_wo_hyperbolic" \
    "use_causal=true" "use_hyperbolic=false" "use_ot=true")
run_pair \
    iemocap chado_wo_hyperbolic "$SCR_IEM" "$cfg_iem_woh" "$RESULTS_ROOT/iemocap/chado_wo_hyperbolic" \
    meld    chado_wo_hyperbolic "$SCR_MEL" "$cfg_mel_woh" "$RESULTS_ROOT/meld/chado_wo_hyperbolic"

# ── ROUND 3: wo_ot — IEMOCAP(A) + MELD(B) ───────────────────────────────────
echo "── Round 3: wo_ot ──────────────────────────────────────"
cfg_iem_woot=$(build_cfg "$CFG_IEM" \
    "$RESULTS_ROOT/iemocap/chado_wo_ot" "chado_iemocap_wo_ot" \
    "use_causal=true" "use_hyperbolic=true" "use_ot=false")
cfg_mel_woot=$(build_cfg "$CFG_MEL" \
    "$RESULTS_ROOT/meld/chado_wo_ot" "chado_meld_wo_ot" \
    "use_causal=true" "use_hyperbolic=true" "use_ot=false")
run_pair \
    iemocap chado_wo_ot "$SCR_IEM" "$cfg_iem_woot" "$RESULTS_ROOT/iemocap/chado_wo_ot" \
    meld    chado_wo_ot "$SCR_MEL" "$cfg_mel_woot" "$RESULTS_ROOT/meld/chado_wo_ot"

# ── ROUND 4: MOSEI — all 4 variants sequentially (heavier model) ─────────────
echo "── Round 4: MOSEI ablations ────────────────────────────"
for variant in wo_causal wo_hyperbolic wo_ot wo_mad; do
    case $variant in
        wo_causal)     flags="use_causal=false use_hyperbolic=true  use_ot=true  use_mad=true"  ;;
        wo_hyperbolic) flags="use_causal=true  use_hyperbolic=false use_ot=true  use_mad=true"  ;;
        wo_ot)         flags="use_causal=true  use_hyperbolic=true  use_ot=false use_mad=true"  ;;
        wo_mad)        flags="use_causal=true  use_hyperbolic=true  use_ot=true  use_mad=false" ;;
    esac
    out="$RESULTS_ROOT/mosei/chado_${variant}"
    cfg=$(build_cfg "$CFG_MOS" "$out" "chado_mosei_${variant}" $flags)
    # MOSEI uses all 4 free GPUs (heavier — 60 epoch base)
    run_ablation mosei "chado_${variant}" "0,1,2,3" "$SCR_MOS" "$cfg" "$out"
done

# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================================"
echo " All component ablations complete."
echo " Generating plots ..."
echo "========================================================"

python3 scripts/plots/plot_component_ablation.py \
    --results_root "$RESULTS_ROOT" \
    --log_dir      "$LOG_DIR" \
    --out_dir      "$ROOT/experiments/figures/component_ablation" \
    && echo "[OK] Plots → experiments/figures/component_ablation/" \
    || echo "[WARN] Plot script failed — run manually"

echo "========================================================"
echo " DONE"
echo "========================================================"
