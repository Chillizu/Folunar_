#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

MODEL="/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER="checkpoints/phase1/partial_adapter_real_25_e3"
WEIGHTS='{"curiosity": 0.5, "competence": 0.5, "boredom": 0.5, "novelty": 0.5}'
EPISODES=10
MAX_STEPS=10
MAX_CANDIDATES=4
TIMEOUT=180

RESULTS_FILE="results/phase1_shell_eval.jsonl"
mkdir -p results
rm -f "$RESULTS_FILE"

echo "Running $EPISODES episodes, each with ${TIMEOUT}s hard timeout..."
for seed in $(seq 0 $((EPISODES - 1))); do
    echo "[Episode $((seed+1))/$EPISODES] seed=$seed"
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
    "seed": seed,
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
        echo "  OK: $json_line"
    else
        echo "  FAILED: rc=$rc"
    fi
done

echo "Done. Results in $RESULTS_FILE"
