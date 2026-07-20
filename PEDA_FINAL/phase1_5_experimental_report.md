# Phase 1.5 Experimental Report

## Background: Why Grid World Was Abandoned

| train_fraction | known_cells | probe disagreement | verdict |
|---|---|---|---|
| 25% | 6/25 | 2/28 (7%) | perfect generalization |
| 10% | 2/25 | 5/28 (18%) | below 33% threshold |

0.5B model generalizes across 25-state grid too quickly. 30-minute kill rule triggered: probe < 33% → Phase 1.5.

## 1. Environment: Custom 2-Room Text Adventure

Study (desk + rusty key, north door) ↔ Hallway (locked chest, south door). 6 legal actions. Optimal path: 3 steps.

## 2. WorldModel Text Support (Backward Compatible)

- `types.py`: `PredictedState.level2_text: str = ""`
- `grid_env.py`: `Perception.render_text(state)` for TextState
- `world_model.py`: text-mode `_llm_predict()`, `rollout()` TextState branch, `decompose_error()` text ensemble variance, `lora_finetune(text_mode=True)`
- `drive_system.py`: `compute_efe()` TextState guard (pragmatic = 0 if exit=2 else 0.5)
- All Grid World code paths untouched. 152/152 Phase 1 tests pass in stub mode.

## 3. Training

**Data:** 113 unique transitions (exhaustive + random walks, dedup by full state_text).

**LoRA:** Qwen2.5-0.5B-Instruct, 3 epochs, 3 checkpoints. Time: 623s. Loss: 0.2928 → 0.0545 → 0.0240.

## 4. Epistemic Probe

| Field | Disagreement |
|---|---|
| Room | 10% (3/30) |
| Exit code | 7% (2/30) |
| Has-key | 40% (12/30) |
| Full semantic tuple | **50% (15/30)** |

Key systematic error: all 3 checkpoints predict `take key` → exit=1 (wrong, env allows it).

## 5. Full Eval (1 episode, 1654s)

**Result:** Both agents failed (0% success, 20 steps).

**PEDA:** inventory → look → **take key** (step 3!) → inventory × 17 (stuck)
**Pragmatic:** look × 20 (never tries take key)

## 6. Key Findings

### 6.1 PEDA ≠ Pragmatic
PEDA tried `take key`. Pragmatic never did. Driver was **LLM confidence signal + drive boredom**, not ensemble epistemic (variance ≈ 0).

### 6.2 Why both failed
1. Systematic error: all 3 checkpoints learned `take key` as exit=1 (wrong).
2. After getting key: `inventory` has confidence 0.999 → lowest EFE → action loop.
3. 113 transitions insufficient for 0.5B model to learn fine-grained state-conditioned rules.

### 6.3 decompose_error undercounts
Semantic probe: 40% has-key disagreement. `decompose_error` TextState path only checks (room, exit_code) → mean_epistemic_error=0.0 despite real disagreements.

## 7. Next Steps (Upstream Decision)

**A. Increase training data** (500-1000 transitions, retrain 20-30 min, re-probe) — recommended.
**B. Fix decompose_error** to include has-key/inventory dimensions.
**C. Larger model or environment** (hardware constraints).
**D. Embrace current result** — PEDA ≠ pragmatic confirmed; driven by drive not prediction error; move to Phase 2.
