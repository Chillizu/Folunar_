#!/usr/bin/env bash
set -uo pipefail

VENV_PYTHON="/home/chillizu/Projects/Folunar_/venv/bin/python"
if ! "$VENV_PYTHON" -c "import transformers; print(transformers.__file__)" >/dev/null 2>&1; then
    echo "FATAL: venv python cannot import transformers. Check $VENV_PYTHON" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

MODEL="/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER="checkpoints/phase1/partial_adapter_real_25_e3"
WEIGHTS='{"curiosity":0.5,"competence":0.5,"boredom":0.5,"novelty":0.5}'
MAX_STEPS=20
MAX_CANDIDATES=4
EPISODES=5
TIMEOUT=240

OBSTACLES_1='[[2,1],[2,2],[2,3]]'
OBSTACLES_2='[[1,2],[2,2],[3,2]]'
OBSTACLES_3='[[1,1],[3,1],[1,3],[3,3]]'

for OBSTACLES in "$OBSTACLES_1" "$OBSTACLES_2" "$OBSTACLES_3"; do
    layout=$(echo "$OBSTACLES" | python3 -c 'import sys, json; print("-".join(f"{x},{y}" for x, y in json.loads(sys.stdin.read())))')
    for VARIANT in peda_adapter pragmatic_only_adapter; do
        if [ "$VARIANT" = "peda_adapter" ]; then
            PRAG_FLAG=""
        else
            PRAG_FLAG="--pragmatic-only"
        fi
        RESULTS_FILE="results/phase1_heldout_${layout}_${VARIANT}.jsonl"
        mkdir -p results
        rm -f "$RESULTS_FILE"

        echo "Layout=$layout variant=$VARIANT"
        for seed in $(seq 0 $((EPISODES - 1))); do
            echo "  [Episode $((seed+1))/$EPISODES] seed=$seed"
            set +e
            output=$(timeout -s KILL "$TIMEOUT" "$VENV_PYTHON" scripts/phase1_heldout_episode.py \
                --model "$MODEL" \
                --adapter "$ADAPTER" \
                --seed "$seed" \
                --max-steps "$MAX_STEPS" \
                --max-candidates "$MAX_CANDIDATES" \
                --obstacles "$OBSTACLES" \
                --weights "$WEIGHTS" \
                --variant "$VARIANT" \
                $PRAG_FLAG 2>&1)
            rc=$?
            set -e
            if [ $rc -eq 0 ]; then
                echo "$output" >> "$RESULTS_FILE"
                echo "    OK: $output"
            else
                echo "    FAILED: rc=$rc"
            fi
        done
        echo "  Results in $RESULTS_FILE"
    done
    echo ""
done
