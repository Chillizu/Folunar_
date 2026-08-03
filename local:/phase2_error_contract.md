# Phase 2 Error-Analysis Data Collection Contract

## Goal
Identify prediction errors in e2 adapter and generate targeted training data to fix them.

## Steps

### 1. Collect prediction errors
Use e2 adapter to predict next_state for ~50 random trajectories.
For each transition, compare predicted exit_code with actual.

```python
# Pseudocode (adapt to actual world_model API)
from src.phase1.world_model import WorldModel
from src.phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
import random, json

wm = WorldModel(model_path="~/models/Qwen2.5-0.5B-Instruct",
                adapter_path="checkpoints/phase2/sandbox_adapter_e2")
sb = BusyboxSandbox()

errors = []
for ep in range(10):
    state = sb.reset()
    for step in range(10):
        candidates = generate_sandbox_candidates(state)
        action = random.choice(candidates)
        pred = wm.predict(state, action)  # returns PredictedState
        next_state, _, _ = sb.step(state, action)
        if pred.exit_code != next_state.last_exit_code:
            errors.append({
                "cwd": state.cwd,
                "files": sorted(state.files),
                "action": action,
                "predicted_exit": pred.exit_code,
                "actual_exit": next_state.last_exit_code,
                "predicted_output": pred.output,
                "actual_output": next_state.last_output[:100],
            })
        state = next_state
    sb.reset()  # fresh container per episode

with open("results/phase2_error_samples.json", "w") as f:
    json.dump(errors, f, indent=2)
print(f"Found {len(errors)} prediction errors in {10*10} transitions")
```

### 2. Generate corrective training data
For each error, create a **correct** (s, a, s') record using the ACTUAL sandbox:
```python
# For each error, re-run the exact transition in a fresh sandbox
# and record the correct (cwd, files, action, next_cwd, next_files, exit_code, output)
```

### 3. Output
- `results/phase2_error_analysis.json` — error list with predictions
- `results/phase2_error_corrections.jsonl` — corrective training records

## Notes
- Use venv: `source venv/bin/activate`
- Only predict exit_code for efficiency (skip full JSON output if too slow)
- Target: 10 episodes × 10 steps = 100 transitions; expect 5-20 errors
- The wm.predict() method may be `wm._llm_predict()` or `wm.generate_text()` — check `src/phase1/world_model.py`
- If prediction API is unclear, fall back to using `scripts/phase2_collect_data.py --baseline random` and comparing against a heuristic correctness check
