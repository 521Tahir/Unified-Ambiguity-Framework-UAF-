#!/usr/bin/env bash
# ============================================================
# Smart Modality Ablation Launcher — IEMOCAP + MELD + MOSEI
# - Waits for free GPUs before launching each job
# - Skips already-completed runs
# - Logs everything; notifies when all done
# Usage: bash scripts/train/run_all_modality_ablations.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR=/tmp/modality_ablation
mkdir -p "$LOG_DIR"

RESULTS_ROOT="$ROOT/experiments/results"
CFG_TMP="$LOG_DIR"

# GPUs available for CHADO project
ALL_GPUS=(5 6 7 8 9)

# Memory threshold: GPU is "free" if used < this MiB
GPU_FREE_THRESHOLD_MIB=6000

# How long to wait (seconds) before re-checking GPU availability
GPU_POLL_INTERVAL=60

# ============================================================
# GPU FREE CHECK
# ============================================================
get_free_gpu() {
    # Returns first GPU index with used memory < threshold, or -1 if none free
    while IFS=',' read -r idx used total util; do
        idx="${idx// /}"
        used_mib="${used// MiB/}"
        used_mib="${used_mib// /}"
        if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib < GPU_FREE_THRESHOLD_MIB )); then
            # Only use our designated GPUs
            for g in "${ALL_GPUS[@]}"; do
                if [[ "$g" == "$idx" ]]; then
                    echo "$idx"
                    return 0
                fi
            done
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                        --format=csv,noheader 2>/dev/null)
    echo "-1"
}

wait_for_free_gpu() {
    local label="$1"
    while true; do
        local gpu
        gpu=$(get_free_gpu)
        if [[ "$gpu" != "-1" ]]; then
            echo "$gpu"
            return 0
        fi
        echo "[WAIT] No free GPU for '$label' — retrying in ${GPU_POLL_INTERVAL}s ..."
        sleep "$GPU_POLL_INTERVAL"
    done
}

# ============================================================
# JOB LAUNCHER (single GPU)
# ============================================================
declare -A RUNNING_PIDS   # pid -> tag

launch_single() {
    local tag="$1"
    local script="$2"
    local cfg="$3"
    local out_dir="$4"
    local log="$LOG_DIR/${tag}.log"

    if [[ -f "${out_dir}/test_results.json" ]]; then
        echo "[SKIP] $tag — already done"
        return 0
    fi

    local gpu
    gpu=$(wait_for_free_gpu "$tag")
    echo "[LAUNCH] GPU=$gpu  $tag"
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES="$gpu" python "$script" --config "$cfg" \
        > "$log" 2>&1 &
    local pid=$!
    RUNNING_PIDS[$pid]="$tag"
    echo "[PID=$pid] $tag started on GPU $gpu"
    sleep 10   # stagger launches to avoid port/init conflicts
}

# ============================================================
# CONFIG BUILDER FOR MOSEI MODALITY ABLATIONS
# ============================================================
build_mosei_cfg() {
    local base_cfg="$1"
    local modality="$2"    # T | TA | TV
    local tag="$3"
    local out_dir="$4"
    local tmp_cfg="$CFG_TMP/cfg_${tag}.yaml"

    python3 - <<PYEOF
import yaml, os

with open("$base_cfg") as f:
    cfg = yaml.safe_load(f)

modality = "$modality"
cfg.setdefault("model", {})
cfg["model"]["use_audio"] = "A" in modality
cfg["model"]["use_video"] = "V" in modality

cfg.setdefault("logging", {})
cfg["logging"]["run_name"] = "$tag"
cfg["logging"]["out_dir"]  = "$out_dir"

# Reduce epochs for ablation speed (70% of original, min 25)
if "train" in cfg:
    orig = cfg["train"].get("epochs", 40)
    cfg["train"]["epochs"]  = max(25, int(orig * 0.7))
    cfg["train"]["patience"] = max(8,  cfg["train"].get("patience", 10))

os.makedirs(os.path.dirname("$tmp_cfg"), exist_ok=True)
with open("$tmp_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print("  cfg written: $tmp_cfg  use_audio=${'A' in modality}  use_video=${'V' in modality}")
PYEOF
    echo "$tmp_cfg"
}

# ============================================================
# STEP 1: WAIT FOR MELD bpmult_TA and bpmult_TV TO FINISH
# ============================================================
echo "========================================================"
echo "STEP 1 — Waiting for running MELD bpmult ablations ..."
echo "========================================================"

wait_for_meld_bpmult() {
    # Check test_results.json directly — avoids grep matching itself
    while true; do
        ta_done=0; tv_done=0
        [[ -f "$RESULTS_ROOT/meld/bpmult_TA/test_results.json" ]] && ta_done=1
        [[ -f "$RESULTS_ROOT/meld/bpmult_TV/test_results.json" ]] && tv_done=1
        if [[ "$ta_done" -eq 1 && "$tv_done" -eq 1 ]]; then
            echo "[OK] bpmult_meld_TA and bpmult_meld_TV both finished."
            break
        fi
        echo "[WAIT] bpmult_meld not yet done (TA=$ta_done TV=$tv_done) — checking in 60s ..."
        sleep 60
    done
}
wait_for_meld_bpmult

# Save results immediately
if [[ -f "$LOG_DIR/bpmult_meld_TA.log" ]]; then
    echo "bpmult_meld_TA final:"
    grep "^\[test\]" "$LOG_DIR/bpmult_meld_TA.log" | head -5
fi
if [[ -f "$LOG_DIR/bpmult_meld_TV.log" ]]; then
    echo "bpmult_meld_TV final:"
    grep "^\[test\]" "$LOG_DIR/bpmult_meld_TV.log" | head -5
fi

echo ""
echo "========================================================"
echo "STEP 2 — IEMOCAP status check"
echo "========================================================"
echo "IEMOCAP modality ablations: checking ..."
missing_iemocap=0
for model in baseline bpmult chado ctnet lflstm mmdfn mult; do
    for mod in T TA TV; do
        out="$RESULTS_ROOT/iemocap/${model}_${mod}"
        if [[ -f "${out}/test_results.json" ]]; then
            echo "  [DONE] iemocap/${model}_${mod}"
        else
            echo "  [MISSING] iemocap/${model}_${mod}"
            missing_iemocap=$((missing_iemocap+1))
        fi
    done
done
echo "IEMOCAP missing: $missing_iemocap"

echo ""
echo "========================================================"
echo "STEP 3 — Launching MOSEI modality ablations (T, TA, TV)"
echo "========================================================"

# Map: config file -> train script
declare -A MOSEI_MODELS
MOSEI_MODELS["baseline"]="configs/mosei/baseline_mosei.yaml:scripts/train/train_baseline_mosei.py"
MOSEI_MODELS["mult"]="configs/mosei/mult_mosei.yaml:scripts/train/train_baseline_mosei.py"
MOSEI_MODELS["mmdfn"]="configs/mosei/mmdfn_mosei.yaml:scripts/train/train_baseline_mosei.py"
MOSEI_MODELS["ctnet"]="configs/mosei/ctnet_mosei.yaml:scripts/train/train_baseline_mosei.py"
MOSEI_MODELS["lflstm"]="configs/mosei/lflstm_mosei.yaml:scripts/train/train_baseline_mosei.py"
MOSEI_MODELS["bpmult"]="configs/mosei/bpmult_mosei.yaml:scripts/train/train_baseline_mosei.py"
MOSEI_MODELS["chado"]="configs/mosei/chado_mosei.yaml:scripts/train/train_mosei.py"

MOSEI_PIDS=()

for model in baseline mult mmdfn ctnet lflstm bpmult chado; do
    entry="${MOSEI_MODELS[$model]}"
    base_cfg="${entry%%:*}"
    script="${entry##*:}"

    if [[ ! -f "$base_cfg" ]]; then
        echo "[SKIP] Config not found: $base_cfg"
        continue
    fi

    for mod in T TA TV; do
        tag="${model}_mosei_${mod}"
        out_dir="$RESULTS_ROOT/mosei/${model}_${mod}"

        if [[ -f "${out_dir}/test_results.json" ]]; then
            echo "[SKIP] $tag — already done"
            continue
        fi

        echo ""
        echo "--- Building config: $tag ---"
        tmp_cfg=$(build_mosei_cfg "$base_cfg" "$mod" "$tag" "$out_dir")

        gpu=$(wait_for_free_gpu "$tag")
        echo "[LAUNCH] GPU=$gpu  $tag"
        mkdir -p "$out_dir"
        log="$LOG_DIR/${tag}.log"

        CUDA_VISIBLE_DEVICES="$gpu" python "$script" --config "$tmp_cfg" \
            > "$log" 2>&1 &
        pid=$!
        MOSEI_PIDS+=($pid)
        echo "[PID=$pid] $tag started on GPU $gpu"
        sleep 15  # stagger to avoid init races
    done
done

# ============================================================
# STEP 4: WAIT FOR ALL MOSEI JOBS
# ============================================================
echo ""
echo "========================================================"
echo "STEP 4 — Waiting for all MOSEI ablation jobs ..."
echo "========================================================"

FAILED=()
for pid in "${MOSEI_PIDS[@]}"; do
    if wait "$pid" 2>/dev/null; then
        echo "[DONE] PID $pid"
    else
        echo "[FAIL] PID $pid (exit $?)"
        FAILED+=("$pid")
    fi
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[WARN] Some MOSEI jobs failed: ${FAILED[*]}"
else
    echo "[OK] All MOSEI modality ablations complete."
fi

# ============================================================
# STEP 5: GENERATE SUMMARY TABLE + PLOTS
# ============================================================
echo ""
echo "========================================================"
echo "STEP 5 — Generating comparison tables and plots ..."
echo "========================================================"

python3 scripts/plots/plot_modality_ablation_summary.py \
    --results_root "$RESULTS_ROOT" \
    --out_dir "$ROOT/experiments/figures/modality_ablation" \
    && echo "[OK] Plots saved to experiments/figures/modality_ablation/" \
    || echo "[WARN] Plot script failed — run manually: python3 scripts/plots/plot_modality_ablation_summary.py"

echo ""
echo "========================================================"
echo "ALL DONE — Modality ablation pipeline complete."
echo "Results: $RESULTS_ROOT"
echo "Logs:    $LOG_DIR"
echo "Plots:   $ROOT/experiments/figures/modality_ablation/"
echo "========================================================"
