#!/bin/bash
# Phase 9 FF-HG-5 rerun after unmatched-verb fallback fix (commit cd1478a).
# PE arms only (count baseline referenced from original F5, not rerun).
# Protocol identical to original: 20 eps, max_steps 10, seeds 0-19,
# image peda-sandbox:v4, same 3 tasks.
set -u
cd /home/data/Projects/Folunar_
OUT=results/phase9_hg_f5_rerun
LOG=logs/phase9_hg_f5_rerun
mkdir -p "$OUT" "$LOG"
pids=()
launch() {
  local agent=$1 alpha=$2 task=$3
  python scripts/phase9_hg_f5.py --agent "$agent" --alpha "$alpha" --task "$task" \
    --episodes 20 --max-steps 10 --image peda-sandbox:v4 --outdir "$OUT" \
    > "$LOG/${agent}_a${alpha}_${task}.log" 2>&1 &
  pids+=($!)
  echo "launched $agent a$alpha $task pid $!"
}
launch pe 0.5 read_changelog_v4
launch pe 0.5 count_measurements
launch pe 0.5 find_errors_v4
launch pe 1.0 read_changelog_v4
launch pe 1.0 count_measurements
launch pe 1.0 find_errors_v4
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
echo "ALL DONE fail=$fail"
