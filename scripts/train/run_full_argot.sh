#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tahirahmad/Project_Code/CHADO_EMNLP"
LOG_DIR="${ROOT}/experiments/results/logs"
CONFIG_DIR="${ROOT}/configs/full"
SCRIPT_DIR="${ROOT}/scripts/train"

mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/full_argot_all_${TS}.log"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "AR-GOT FULL MODEL RUNS" | tee -a "${MASTER_LOG}"
echo "Start time: $(date)" | tee -a "${MASTER_LOG}"
echo "Host: $(hostname)" | tee -a "${MASTER_LOG}"
echo "Root: ${ROOT}" | tee -a "${MASTER_LOG}"
echo "Log dir: ${LOG_DIR}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

check_file () {
  if [ ! -f "$1" ]; then
    echo "[ERROR] Missing file: $1" | tee -a "${MASTER_LOG}"
    exit 1
  fi
}

run_one () {
  local DATASET="$1"
  local GPUS="$2"
  local PORT="$3"
  local TRAIN_SCRIPT="$4"
  local CONFIG="$5"
  local OUT_DIR="$6"

  local LOG_FILE="${LOG_DIR}/full_${DATASET}_${TS}.log"

  check_file "${TRAIN_SCRIPT}"
  check_file "${CONFIG}"

  mkdir -p "${OUT_DIR}"

  {
    echo ""
    echo "============================================================"
    echo "DATASET: ${DATASET}"
    echo "Start time: $(date)"
    echo "GPUs: ${GPUS}"
    echo "Port: ${PORT}"
    echo "Train script: ${TRAIN_SCRIPT}"
    echo "Config: ${CONFIG}"
    echo "Output dir: ${OUT_DIR}"
    echo "Log file: ${LOG_FILE}"
    echo "============================================================"
    echo ""
    echo "[CONFIG CONTENT]"
    cat "${CONFIG}"
    echo ""
    echo "============================================================"
    echo "[TRAINING OUTPUT]"
    echo "============================================================"
  } | tee -a "${LOG_FILE}" | tee -a "${MASTER_LOG}" >/dev/null

  CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 torchrun \
    --nproc_per_node=5 \
    --master_port="${PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}" | tee -a "${MASTER_LOG}"

  local STATUS=${PIPESTATUS[0]}

  {
    echo ""
    echo "============================================================"
    echo "DATASET: ${DATASET}"
    echo "End time: $(date)"
    echo "Exit status: ${STATUS}"
    echo "============================================================"
    echo ""
  } | tee -a "${LOG_FILE}" | tee -a "${MASTER_LOG}" >/dev/null

  if [ "${STATUS}" -ne 0 ]; then
    echo "[ERROR] ${DATASET} run failed. Check: ${LOG_FILE}" | tee -a "${MASTER_LOG}"
    exit "${STATUS}"
  fi

  echo "[OK] ${DATASET} run completed. Log: ${LOG_FILE}" | tee -a "${MASTER_LOG}"
}

# -------------------------
# Full IEMOCAP
# -------------------------
run_one \
  "iemocap" \
  "0,1,2,3,4" \
  "29700" \
  "${SCRIPT_DIR}/train_iemocap_tqdm_ddpfix_evalcache.py" \
  "${CONFIG_DIR}/chado_iemocap_full.yaml" \
  "${ROOT}/experiments/results/iemocap/full_argot/seed_42"

# -------------------------
# Full MELD
# -------------------------
run_one \
  "meld" \
  "0,1,2,3,4" \
  "29701" \
  "${SCRIPT_DIR}/train_meld_tqdm_ddpfix_evalcache.py" \
  "${CONFIG_DIR}/chado_meld_full.yaml" \
  "${ROOT}/experiments/results/meld/full_argot/seed_42"

# -------------------------
# Full CMU-MOSEI
# -------------------------
run_one \
  "mosei" \
  "0,1,2,3,4" \
  "29702" \
  "${SCRIPT_DIR}/train_mosei_tqdm_ddpfix_evalcache.py" \
  "${CONFIG_DIR}/chado_mosei_full.yaml" \
  "${ROOT}/experiments/results/mosei/full_argot/seed_42"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "All full AR-GOT runs completed." | tee -a "${MASTER_LOG}"
echo "End time: $(date)" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"