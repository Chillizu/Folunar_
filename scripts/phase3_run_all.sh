#!/bin/bash
# Phase 3 Full Sequential Experiment Runner
# Launched in tmux for persistence
# Runs all 4 conditions sequentially on GPU

set -e
cd /home/ec2-user/Folunar_

PYTHON=/opt/pytorch/bin/python
SCRIPT=scripts/phase3_sandbox_experiment.py
EPISODES=5
MAX_STEPS=10
RESULTS=results
mkdir -p $RESULTS

echo "========================================"
echo "Phase 3 Sandbox Experiment (full)"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "========================================"

# Condition 1: pragmatic known
echo ""
echo "=== [1/4] pragmatic known ==="
$PYTHON $SCRIPT --baseline pragmatic --condition known \
    --num-episodes $EPISODES --max-steps $MAX_STEPS \
    --output $RESULTS/phase3_sandbox_pragmatic_known.jsonl 2>&1
echo "=== [1/4] complete ==="

# Condition 2: pragmatic unknown
echo ""
echo "=== [2/4] pragmatic unknown ==="
$PYTHON $SCRIPT --baseline pragmatic --condition unknown \
    --num-episodes $EPISODES --max-steps $MAX_STEPS \
    --output $RESULTS/phase3_sandbox_pragmatic_unknown.jsonl 2>&1
echo "=== [2/4] complete ==="

# Condition 3: PEDA known (fast mode - no ensemble)
echo ""
echo "=== [3/4] PEDA known ==="
$PYTHON $SCRIPT --baseline peda --condition known \
    --num-episodes $EPISODES --max-steps $MAX_STEPS --fast \
    --output $RESULTS/phase3_sandbox_peda_known.jsonl 2>&1
echo "=== [3/4] complete ==="

# Condition 4: PEDA unknown (fast mode - no ensemble)
echo ""
echo "=== [4/4] PEDA unknown ==="
$PYTHON $SCRIPT --baseline peda --condition unknown \
    --num-episodes $EPISODES --max-steps $MAX_STEPS --fast \
    --output $RESULTS/phase3_sandbox_peda_unknown.jsonl 2>&1
echo "=== [4/4] complete ==="

echo ""
echo "========================================"
echo "Experiment complete!"
echo "End: $(date)"
echo "Results in: $RESULTS/"
ls -la $RESULTS/phase3_sandbox_*.jsonl 2>/dev/null
echo "========================================"
