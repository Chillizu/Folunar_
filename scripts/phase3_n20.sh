#!/bin/bash
# Phase 3 N=20 Sequential Runner - runs all 4 conditions one at a time
# to avoid GPU OOM. Writes to phase3_sandbox_n20_*.jsonl files.
set -e

BASE_DIR="/home/ec2-user/Folunar_"
PYTHON="/opt/pytorch/bin/python"
EXPERIMENT_SCRIPT="$BASE_DIR/scripts/phase3_sandbox_experiment.py"
N=20
MAX_STEPS=10

cd "$BASE_DIR"

echo "=" >/dev/null
echo "Phase 3 N=20 Sequential Experiment"
echo "===================================="
echo ""

# Run conditions sequentially (fast first, then slow)
# 1. pragmatic_known  (fast)
echo "[1/4] pragmatic_known  (N=$N, max_steps=$MAX_STEPS)"
$PYTHON "$EXPERIMENT_SCRIPT" \
    --baseline pragmatic --condition known \
    --num-episodes $N --max-steps $MAX_STEPS \
    --output "results/phase3_sandbox_n20_pragmatic_known.jsonl" 2>&1
echo "pragmatic_known done at $(date)"
echo ""

# 2. pragmatic_unknown  (fast)
echo "[2/4] pragmatic_unknown  (N=$N, max_steps=$MAX_STEPS)"
$PYTHON "$EXPERIMENT_SCRIPT" \
    --baseline pragmatic --condition unknown \
    --num-episodes $N --max-steps $MAX_STEPS \
    --output "results/phase3_sandbox_n20_pragmatic_unknown.jsonl" 2>&1
echo "pragmatic_unknown done at $(date)"
echo ""

# 3. peda_known  (slower)
echo "[3/4] peda_known  (N=$N, max_steps=$MAX_STEPS)"
$PYTHON "$EXPERIMENT_SCRIPT" \
    --baseline peda --condition known \
    --num-episodes $N --max-steps $MAX_STEPS \
    --output "results/phase3_sandbox_n20_peda_known.jsonl" 2>&1
echo "peda_known done at $(date)"
echo ""

# 4. peda_unknown  (slowest)
echo "[4/4] peda_unknown  (N=$N, max_steps=$MAX_STEPS)"
$PYTHON "$EXPERIMENT_SCRIPT" \
    --baseline peda --condition unknown \
    --num-episodes $N --max-steps $MAX_STEPS \
    --output "results/phase3_sandbox_n20_peda_unknown.jsonl" 2>&1
echo "peda_unknown done at $(date)"
echo ""

echo "===================================="
echo "Phase 3 N=20 Experiment Complete!"
echo "Results in: $BASE_DIR/results/"
ls -la results/phase3_sandbox_n20_*.jsonl 2>/dev/null
