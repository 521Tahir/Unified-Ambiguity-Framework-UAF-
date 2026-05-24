#!/bin/bash
# Runs proxy_single_seed launcher; when it exits (after 6/9 complete),
# reruns to catch the 3 MOSEI jobs that failed first time.
# Then relaunches disentanglement.
cd /home/tahirahmad/Project_Code/CHADO_EMNLP

echo "[$(date '+%H:%M:%S')] Round 1 — 6 IEMOCAP+MELD jobs running, waiting..."
wait $(pgrep -f "run_proxy_single_seed.py") 2>/dev/null

echo "[$(date '+%H:%M:%S')] Round 2 — picking up MOSEI jobs (fixed numpy bug)"
python3 -u scripts/train/run_proxy_single_seed.py \
  2>&1 | tee -a logs/proxy_single_seed_launcher.log

echo ""
echo "[$(date '+%H:%M:%S')] All 9 proxy jobs complete. Restarting disentanglement..."
bash scripts/train/restart_disent.sh
