#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

MODEL="/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER="checkpoints/phase1/partial_adapter_real_25_e3"
MAX_STEPS=10
MAX_CANDIDATES=4
TIMEOUT=180
EPISODES_PER_CONFIG=2

# Top 5 weights from stub grid search
WEIGHTS_LIST=(
    '{"curiosity": 1.0, "competence": 0.1, "boredom": 0.1, "novelty": 2.0}'
    '{"curiosity": 2.0, "competence": 1.0, "boredom": 0.5, "novelty": 1.0}'
    '{"curiosity": 2.0, "competence": 0.1, "boredom": 0.5, "novelty": 2.0}'
    '{"curiosity": 0.5, "competence": 0.5, "boredom": 0.5, "novelty": 0.5}'
    '{"curiosity": 1.0, "competence": 2.0, "boredom": 0.5, "novelty": 0.5}'
)

RESULTS_FILE="results/phase1_shell_grid_search.jsonl"
mkdir -p results
rm -f "$RESULTS_FILE"

for idx in "${!WEIGHTS_LIST[@]}"; do
    WEIGHTS="${WEIGHTS_LIST[$idx]}"
    echo "[Config $((idx+1))/${#WEIGHTS_LIST[@]}] weights=$WEIGHTS"
    for ep in $(seq 0 $((EPISODES_PER_CONFIG - 1))); do
        seed=$((idx * 100 + ep))
        echo "  Episode $((ep+1))/$EPISODES_PER_CONFIG seed=$seed"
        set +e
        output=$(timeout -s KILL "$TIMEOUT" python - "$MODEL" "$ADAPTER" "$seed" "$MAX_STEPS" "$MAX_CANDIDATES" "$WEIGHTS" <<'PY'
import sys, json
sys.path.insert(0, "src")
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

model, adapter, seed, max_steps, max_candidates, weights = sys.argv[1:7]
seed = int(seed)
max_steps = int(max_steps)
max_candidates = int(max_candidates)
weights = json.loads(weights)

wm = WorldModel(model_name=model, use_stub=False, adapter_path=adapter)
env = GridWorld(width=5, height=5, max_steps=max_steps)
ec = EnsembleErrorComputer(wm, num_checkpoints=5)
ds = HomeostaticDriveSystem(DriveWeights(**weights))
lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=max_candidates, latency_budget_ms=3000.0)

traj, preds, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=seed)
result = {
    "config_idx": seed // 100,
    "seed": seed,
    "weights": weights,
    "success": metrics["success"],
    "steps": metrics["steps"],
    "reward": metrics["reward"],
    "trajectory": [s.agent for s in traj],
    "goal": traj[0].goal,
    "actions": [a.name for a in actions],
}
print(json.dumps(result))
PY
)
        rc=$?
        set -e
        if [ $rc -eq 0 ]; then
            json_line=$(echo "$output" | tail -1)
            echo "$json_line" >> "$RESULTS_FILE"
            echo "    OK: $json_line"
        else
            echo "    FAILED: rc=$rc"
        fi
    done
done

echo "Done. Results in $RESULTS_FILE"
