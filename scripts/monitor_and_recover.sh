#!/usr/bin/env bash
# Auto-monitor and recover all CHADO experiment launchers.
cd /home/tahirahmad/Project_Code/CHADO_EMNLP

LOG_DIR="logs"
PID_FILE="/tmp/chado_pids.txt"
source "$PID_FILE" 2>/dev/null || true
COMPONENT=${COMPONENT:-0}; PROXY=${PROXY:-0}; LOEO=${LOEO:-0}

ts() { date '+%H:%M:%S'; }

# ── GPU snapshot ──────────────────────────────────────────────────────────────
echo "=== GPU [$(ts)] ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '{printf "  GPU%s: used=%6sMiB free=%6sMiB\n",$1,$2,$3}'

# ── Check/restart each launcher ───────────────────────────────────────────────
check_launcher() {
  local name="$1" pid="$2" script="$3" log="$4"
  echo ""
  echo "=== $name ==="
  if [ "$pid" -gt 0 ] && kill -0 "$pid" 2>/dev/null; then
    echo "  ALIVE PID=$pid"
    grep -a "running=\|DONE\|FAIL\|LAUNCH" "$log" 2>/dev/null | tail -3 | sed 's/^/    /'
    echo "$pid" > /tmp/chado_newpid_${name}.txt
  else
    echo "  DEAD (was PID=$pid) — restarting"
    nohup python3 -u "$script" >> "$log" 2>&1 &
    local npid=$!
    echo "  new PID=$npid"
    echo "$npid" > /tmp/chado_newpid_${name}.txt
  fi
}

check_launcher "component" "$COMPONENT" \
  "scripts/train/run_all_component_ablations.py" \
  "${LOG_DIR}/component_ablations_launcher.log"

check_launcher "proxy" "$PROXY" \
  "scripts/train/run_proxy_ablation_multiseed.py" \
  "${LOG_DIR}/proxy_ablation_multiseed_launcher.log"

check_launcher "loeo" "$LOEO" \
  "scripts/loeo/run_loeo.py" \
  "${LOG_DIR}/loeo_launcher.log"

# ── Save updated PIDs ─────────────────────────────────────────────────────────
COMPONENT=$(cat /tmp/chado_newpid_component.txt 2>/dev/null || echo $COMPONENT)
PROXY=$(cat /tmp/chado_newpid_proxy.txt 2>/dev/null || echo $PROXY)
LOEO=$(cat /tmp/chado_newpid_loeo.txt 2>/dev/null || echo $LOEO)
printf "COMPONENT=%s\nPROXY=%s\nLOEO=%s\n" "$COMPONENT" "$PROXY" "$LOEO" > "$PID_FILE"
echo ""
echo "=== PIDs: COMP=$COMPONENT  PROXY=$PROXY  LOEO=$LOEO ==="

# ── Results tally ─────────────────────────────────────────────────────────────
echo ""
echo "=== RESULTS TALLY ==="
COMP_DONE=0
for abl in wo_mad wo_ot wo_radial wo_causal; do
  for ds in iemocap mosei meld; do
    [ -f "experiments/results/${ds}/component_ablations/${abl}/seed_42/test_results.json" ] \
      && { echo "  DONE  component $ds/$abl"; COMP_DONE=$((COMP_DONE+1)); }
  done
done
echo "  component total: ${COMP_DONE}/12"
PROXY_DONE=$(find experiments/results/proxy_ablation_multiseed -name "test_results.json" 2>/dev/null | wc -l)
echo "  proxy multiseed: ${PROXY_DONE}/45"
LOEO_DONE=$(find experiments/results/loeo -name "test_results.json" 2>/dev/null | wc -l)
echo "  loeo: ${LOEO_DONE}/45"
