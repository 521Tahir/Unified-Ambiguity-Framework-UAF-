#!/bin/bash
# Watch for seed training completion and regenerate Table 1 when all seeds finish.
# Run in background: bash scripts/eval/watch_seeds_and_update_table.sh &

RESULTS=/home/tahirahmad/Project_Code/CHADO_EMNLP/experiments/results
LOG=/tmp/watch_seeds.log

echo "[$(date)] Watching for seed training results..." | tee $LOG

REQUIRED=(
  "$RESULTS/iemocap/chado_seed123/test_results.json"
  "$RESULTS/iemocap/chado_seed456/test_results.json"
  "$RESULTS/meld/chado_seed123/test_results.json"
  "$RESULTS/meld/chado_seed456/test_results.json"
  "$RESULTS/mosei/chado_seed123/test_results.json"
  "$RESULTS/mosei/chado_seed456/test_results.json"
)

while true; do
  all_done=true
  n_done=0
  for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
      n_done=$((n_done+1))
    else
      all_done=false
    fi
  done

  echo "[$(date)] Seed results: $n_done / ${#REQUIRED[@]} done" | tee -a $LOG

  if $all_done; then
    echo "[$(date)] All seeds done! Regenerating Table 1..." | tee -a $LOG
    cd /home/tahirahmad/Project_Code/CHADO_EMNLP
    python3 scripts/eval/aggregate_multiseed.py >> $LOG 2>&1
    echo "[$(date)] Table 1 updated at paper/tables/main_results.tex" | tee -a $LOG
    break
  fi

  sleep 300  # check every 5 minutes
done
