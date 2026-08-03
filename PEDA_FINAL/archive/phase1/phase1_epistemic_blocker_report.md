# Phase 1 Epistemic Blocker Report

## Status

**P0: 3-epoch adapter trained.** 3 checkpoints saved (`checkpoint_epoch_1/2/3`) with decreasing loss (0.0308 → 0.0047 → 0.0035).

**P1: Blocked.** Ensemble variance ~zero after diagnostic check:
- 28 state-action probes × 4 actions across grid: only 2/28 showed any checkpoint disagreement
- `g1_test_set = 1.0` on held-out pairs (previous 1-epoch run had 0.8684)
- `mean_epistemic_error = 0.0` in smoke test (1 episode × 2 conditions, max_steps=10)
- `epistemic_error = ensemble_variance` — formula is correct in `world_model.py:509`

## Root Cause

`train_fraction=0.25` (6/25 known cells) provides enough coverage for 0.5B model with 3 epochs to perfectly generalize the 5×5 grid dynamics. The grid is too small, the transition rules too simple, and 0.5B too large for this environment.

This was already predicted:
- Feedback: "0.5B 对 5×5 太强" and "短期：降到 15-20% train_fraction"
- Previous report: "0.5B 模型在只训练 25% 的情况下，OOD 泛化准确率仍达 87%"

## What This Means

- PEDA ≈ pragmatic_only under current setup. Any "PEDA advantage" would come from drive system modulation (curiosity/boredom/novelty) not prediction-error-driven exploration.
- Running 10-episode full eval now would be wasted compute (~5h) producing a drive-ablation result that cannot validate the core PEDA hypothesis.

## What Changes in Next Session

The quickest path to epistemic_error > 0:

1. **Retrain with `--train-fraction 0.10`** (2-3 known cells out of 25). This starves the model enough that held-out cells produce OOD prediction errors. The 0.5B model must then exhibit ensemble variance for cells it has never seen starting from.

2. **Smoke test** (1 episode × 2 conditions) to confirm `mean_epistemic_error > 0` before committing to full 10-episode eval.

3. **If still zero at 0.10**, either go to `--train-fraction 0.05` or skip Grid World Phase 1 hypothesis test entirely and move to Phase 1.5 (TextWorld / busybox sandbox).

## Evidence (Run in Next Session)

```bash
# After retraining with --train-fraction 0.10:
# Diagnostic probe (5 min)
from pathlib import Path
from phase1.grid_env import GridWorld
from phase1.world_model import WorldModel, EnsembleErrorComputer

wm = WorldModel("Qwen/Qwen2.5-0.5B-Instruct", adapter_path="checkpoints/phase1/partial_adapter_real_10")
ec = EnsembleErrorComputer(wm)
ec.checkpoints = sorted(Path("checkpoints/phase1/partial_adapter_real_10").glob("checkpoint_epoch_*"))
total, diverse = 0, 0

for s in [(0,0), (2,2), (4,4), (1,3), (3,1), (4,0), (0,4)]:
    state = GridWorld(goal=(4,4)).reset(seed=42)
    state.agent = s
    for action in GridWorld.all_actions():
        preds = ec._predictions_for(state, action)
        if len(set(p.level2_next_agent for p in preds)) > 1:
            diverse += 1
        total += 1
print(f"Epistemic alive: {diverse}/{total} state-action pairs")
```
