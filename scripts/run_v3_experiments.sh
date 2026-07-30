#!/bin/bash
# Batch run all v3 sandbox experiments on GPU
set -e
cd ~/Folunar_

TASKS="read_greeting count_entries find_secret_note read_user_guide"
MODES="learned fallback"

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo " Starting v3 sandbox batch experiments"
echo " Tasks: $TASKS"
echo " Modes: $MODES"
echo " Date: $(date)"
echo "=========================================="

for mode in $MODES; do
  for task in $TASKS; do
    echo ""
    echo "--- Experiment: $task / $mode ---"
    echo "Start: $(date)"

    python3 scripts/phase5_no_wm_experiment.py \
      --task "$task" \
      --num-episodes 12 \
      --candidates "$mode" \
      --condition all \
      --output "results/v3_${mode}_${task}.jsonl" \
      --seed-offset 42 2>&1

    echo "End: $(date)"
    echo ""
  done
done

echo "=========================================="
echo " All experiments complete!"
echo "=========================================="

# Print summary
echo ""
echo "=== SUMMARY ==="
echo "Task                Mode       Success  AvgSteps  SCR"
echo "-------------------- ---------- ------- --------- ---"

for mode in $MODES; do
  for task in $TASKS; do
    result_file="results/v3_${mode}_${task}.jsonl"
    if [ -f "$result_file" ]; then
      total=$(wc -l < "$result_file")
      successes=$(grep -c '"success": true' "$result_file" 2>/dev/null || echo 0)
      avg_steps=$(python3 -c "
import json
with open('$result_file') as f:
    steps = [json.loads(l)['steps_count'] for l in f]
print(f'{sum(steps)/len(steps):.1f}' if steps else 'N/A')
" 2>/dev/null || echo "N/A")
      scr=$(python3 -c "
import json
with open('$result_file') as f:
    scrs = [json.loads(l).get('scr', 0) for l in f]
print(f'{sum(scrs)/len(scrs):.3f}' if scrs else 'N/A')
" 2>/dev/null || echo "N/A")
      printf "%-20s %-10s %-7d %-9s %s\n" "$task" "$mode" "$successes" "$avg_steps" "$scr"
    fi
  done
done
