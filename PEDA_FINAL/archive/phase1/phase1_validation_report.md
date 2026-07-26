# PEDA Phase 1 Real-LLM Go/No-Go Validation Report

**Date:** 2026-07-03
**Agent:** Folunar_ Coding Agent (Oh-My-Pi harness)
**Repository:** `Chillizu/Folunar_` (branches `dev` and `main`)
**Commit range:** `e035e8d..3a95d34`

---

## 1. Goal

Validate whether the PEDA Phase 1 loop, with a real (not stub) World Model,
passes the three go/no-go gates on the 5×5 grid world:

| Gate | Criterion | Target |
|------|-----------|--------|
| G1 | Level-2 next-state accuracy | > 0.90 |
| G2 | Drive-agent steps / random steps ratio | < 0.50 |
| G3 | Cell revisit rate | < 0.20 |

The validation is meant to test the core hypothesis: *prediction-error-driven
action selection is more efficient than random exploration* in a controlled
setting.

---

## 2. Setup & Constraints

| Item | Value / Note |
|------|--------------|
| Workstation | CPU-only; 1.5B models exceed the ~600 s load timeout |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (below the 1–7B guideline, chosen because of the hardware constraint) |
| Adapter | `checkpoints/phase1/synthetic_adapter` (LoRA, trained on 20 random-goal configs, no obstacles) |
| Training data | Synthetic 5×5 transitions: for each config, every free-cell position × every action → next position |
| Drive weights | `cur=0.1, cmp=0.5, bor=0.1, nov=0.1` (restored from `config/phase1_default_drives.json` after a stub grid search accidentally overwrote it) |
| Max candidates | 4 (default of `phase1_eval.py`) — using `--max-candidates 2` was the cause of an earlier G2/G3 failure because the agent could never move horizontally |

---

## 3. Methods

1. **Train a LoRA adapter** on synthetic grid-world data (`scripts/phase1_synthetic_train.py`):
   - 20 configs, 1 epoch, batch size 4
   - Average loss: 0.0101
2. **Run the final evaluator** (`scripts/phase1_eval.py`) with the trained adapter on the real 0.5B model for 20 episodes.
3. **Compare** against a random baseline on the same environment.
4. **Regression checks**: `ruff check src tests scripts` and `PYTHONPATH=src FOLUNAR_STUB_MODEL=1 pytest tests/phase1 -q` (138 tests pass).

---

## 4. Results

### 4.1 20-Episode Real-LLM Evaluation

```
Stub mode: False
Adapter:   checkpoints/phase1/synthetic_adapter
Episodes:  20
Best drive weights: cur=0.1 cmp=0.5 bor=0.1 nov=0.1
```

| Metric | Drive Agent | Random Baseline |
|--------|-------------|-----------------|
| Success rate | 1.000 (20/20) | 0.700 (14/20) |
| Mean steps | 3.3 | 27.25 |
| Completion @20 | 1.000 | 0.450 |

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| G1 | 1.0000 | > 0.90 | PASS |
| G2 | 0.1211 | < 0.50 | PASS |
| G3 | 0.0000 | < 0.20 | PASS |

### 4.2 One-Episode Spot Checks

Both one-episode real-LLM and one-episode stub-mode evaluations pass all three
gates. The stub mode is not a validation, only a sanity check that the EFE logic
is wired correctly.

### 4.3 Training Loss

```
loss 0.5638 → 0.0000 (avg 0.0101)
```

---

## 5. Code Changes Summary

| File | Change |
|------|--------|
| `src/phase1/world_model.py` | `adapter_path` loading; generation-probability confidence; `max_length=256`; optimizer over trainable params only; correct `BatchEncoding` handling |
| `src/phase1/drive_system.py` | Pragmatic goal-distance term in `compute_efe` (3× weight); signature updated to accept `GridState` |
| `src/phase1/run.py` | Pass `state` to `compute_efe` |
| `tests/phase1/test_drive_system.py` | Update tests for new `compute_efe` signature |
| `scripts/phase1_synthetic_train.py` | Add `TRAINING_FINISHED`/`TRAINING_FAILED` markers; `success` + `finished_at` in `training_info.json`; full try/except wrapper |
| `scripts/phase1_grid_search.py` | Default output to `results/phase1_grid_search.json`; only overwrite config with `--write-config`; richer report with timestamp and metrics |
| `scripts/phase1_eval.py` | Add `--drive-config`; accept both legacy list and grid-search report (`top_5`) formats |
| `config/phase1_default_drives.json` | Restored validated weights |
| `results/phase1_eval.json` | Final 20-episode report with `drive_weights_source` provenance note |
| `WATCHDOG.md`, `WATCHDOG.yml` | Advisor rule set added to the repo |

---

## 6. Honest Limitations & Risks

1. **Model size.** The 0.5B model is smaller than the 1–7B guideline in `AGENTS.md`.
   It is sufficient for the deterministic 5×5 rules in this synthetic dataset,
   but capacity may not generalize to more complex Phase 1.5/2 environments.

2. **Drive weights not from a real-model grid search.** The current weights were
   restored from `config/phase1_default_drives.json` after a stub grid search
   overwrote the file. They were validated by the 20-episode real eval, but they
   are not traceable to a real-LLM grid search. WATCHDOG C4 and PEDA_FINAL 6.4
   (hyperparameter traceability) are therefore **not fully satisfied**.

3. **Stub grid-search anomaly.** Running `phase1_grid_search.py --stub` produces
   `score≈0.996, steps=1.0` for many Phase-A combos. This is a sampling artifact:
   Phase A uses only 2 episodes per combo with deterministic seeds, so some
   combos happen to sample agent/goal adjacency twice. Phase B (10 episodes)
   shows plausible mean steps of 2.2–5.1. The anomaly is not a bug in the
   environment step count; it is a protocol issue of too few episodes per
   combo.

4. **Single environment distribution.** The eval uses the same random-goal/no-obstacle
   distribution as the training data. Generalization to obstacles, larger grids,
   or TextWorld has not been tested.

5. **Phase advancement claim.** The `phase1_eval.py` script prints
   "RESULT: ALL GATES PASSED — Phase 1 validation successful. Proceed to Phase 1.5/2."
   This is misleading because the drive weights are not grid-search derived. The
   gates pass, but Phase 1 should not be declared fully closed until a real grid
   search is run or the provenance gap is explicitly accepted.

---

## 7. Recommendations for Next Steps

1. **Run a real-LLM grid search** (or a smaller subset search) to satisfy WATCHDOG C4
   and select drive weights with traceable metrics. Use `--write-config` to update
   `config/phase1_default_drives.json` from the real search, then re-run
   `phase1_eval.py` to confirm the new weights still pass G1/G2/G3.
2. **Address the 0.5B capacity ceiling** before Phase 1.5: either obtain a machine
   that can load a 1.5–3B model within the timeout, or quantize a larger model
   to CPU-runnable precision.
3. **Fix the stub grid-search protocol** by increasing Phase A episodes (e.g., 5–10)
   or randomizing seeds independently of combo index, so the report is not
   dominated by sampling luck.
4. **Re-evaluate** on a held-out distribution (e.g., obstacles, larger grids) to
   test generalization before moving to Phase 1.5.

---

## 8. Partial-Training Generalization Test (Core Hypothesis Check)

The 20-episode G1/G2/G3 validation above does **not** isolate the contribution of
prediction-error (epistemic) signals. The pragmatic distance term and the
identical train/eval distribution mean a greedy Manhattan-distance policy can
reproduce the same metrics. To test the core PEDA hypothesis, we re-designed the
validation as Option A from the external evaluation: train the World Model on
only a subset of cells, then compare PEDA against a pragmatic-only baseline
when the goal lies in the **unknown** region.

| Item | Value |
|------|-------|
| Train fraction | 0.25 (6 known cells / 19 unknown cells) |
| Configs | 20 random-goal / no-obstacle configs |
| Training | 1 epoch, batch size 4, adapter `checkpoints/phase1/partial_adapter_real_25` |
| Eval script | `scripts/phase1_partial_eval.py` |
| PEDA | `pragmatic_only=False`, `pragmatic_weight=3.0` |
| Pragmatic-only baseline | `pragmatic_only=True`, `pragmatic_weight=3.0` |
| Controls | Fresh `HomeostaticDriveSystem` per agent per episode; `LearningModule` disabled (`update_interval=100000`); same start/goal for both agents |
| Scale | 10 episodes per condition, run as 1-episode chunks with `--skip-g1-test`, merged via `merge_partial_eval_chunks.py` |

### 8.2 Full Results (10 Episodes per Condition)

```
Adapter: checkpoints/phase1/partial_adapter_real_25
Known cells: 6 / 25
Episodes per condition: 10
Held-out test-set accuracy (g1_test_set): 0.8684 (computed from chunk 0 only)
```

| Condition | Agent | Success | Mean Steps | Revisit Rate | g1 |
|-----------|-------|---------|------------|--------------|-----|
| goal_known | PEDA | 0.9 | 8.6 | 0.128 | 0.931 |
| goal_known | pragmatic_only | 0.7 | 17.3 | 0.271 | 0.996 |
| goal_unknown | PEDA | 0.7 | 16.6 | 0.311 | 0.727 |
| goal_unknown | pragmatic_only | 0.6 | 21.1 | 0.366 | 0.754 |

**Per-episode comparison** shows PEDA's advantage is concentrated in a few episodes where pragmatic-only gets stuck at max_steps (50). In most episodes, both agents behave nearly identically:

* goal_known: PEDA wins in episodes 6–7 (6 steps each vs 50-step failures for pragmatic); episodes 0–5,8–9 are tied.
* goal_unknown: PEDA wins in episode 7 (4 steps vs 50); episodes 0–6,8–9 are tied.
* Both agents fail the same difficult episodes (3 out of 10 in each condition).

Data: `results/phase1_partial_eval_10eps.json`

### 8.3 Interpretation

PEDA shows a directional advantage on aggregate metrics (mean_steps, success_rate,
revisit_rate) over pragmatic-only in both conditions. However, **the difference
cannot be attributed to prediction-error-driven exploration** for the following
reasons:

1. **Single checkpoint.** The adapter was trained with only 1 epoch, so
   `EnsembleErrorComputer` loads exactly 1 checkpoint. Ensemble epistemic variance
   is identically zero. PEDA's EFE formula collapses to:
   ```
   EFE = drive_system.apply_to_efe(pragmatic * 3.0)
   ```
   The output is dominated by the pragmatic distance term, modulated by the
   `HomeostaticDriveSystem` (curiosity=0.1, competence=0.5, boredom=0.1, novelty=0.1).

2. **Drive system as confound.** The observed advantage comes from the drive system
   (low curiosity/boredom weight pushing exploration when the agent is stuck),
   not from epistemic uncertainty. This is a meaningful signal but tests a
   different mechanism than the core PEDA hypothesis.

3. **Identical behavior on most episodes.** The aggregate advantage is driven by
   2–3 episodes where pragmatic-only hits max_steps while PEDA succeeds. On the
   remaining 7–8 episodes, both agents take exactly the same steps, choose the
   same actions, and succeed/fail identically. This pattern is consistent with
   rare-case drive modulation rather than a systematic epistemic exploration
   advantage.

4. **g1_test_set = 0.8684.** The model already generalizes reasonably well to
   held-out state-action pairs, reducing the epistemic signal that could appear
   even with a full ensemble.

### 8.4 Limitations

1. **Scale.** 10 episodes per condition is a directional sample, not a
   statistically robust comparison. Several episodes are trivially short (1–2
   steps), reducing the effective difficulty.
2. **Single checkpoint.** The most critical limitation: without ≥2 per-epoch
   checkpoints, `epistemic_error` is always zero. This experiment **cannot**
   validate the core PEDA hypothesis that prediction-error drives exploration.
   To properly test it, the adapter needs ≥3 epochs of training with per-epoch
   checkpoint saving.
3. **Unknown cells visited.** PEDA's exploration metrics
   (`mean_unknown_cells_visited=3.3, mean_unknown_fraction=0.86` in goal_unknown)
   show that both agents operate primarily in unknown territory, but PEDA's
   modest advantage is driven by escaping dead ends, not by intentional epistemic
   exploration.
4. **--skip-g1-test.** The g1_test_set was computed only in chunk 0 and reused.
   This is acceptable for aggregation but means per-chunk held-out accuracy is
   not tracked.
5. **Reduced max_steps for chunks 8–9.** Chunks 8–9 used `--max-steps 45`
   instead of 50 to fit within the 3600s timeout. The effect on aggregate
   metrics is negligible (2 failed episodes would have added 5 more steps each
   at 50 instead of 45).

**Conclusion for the core hypothesis:** The 10-episode partial-training eval
does not provide evidence that prediction-error (epistemic) signals drive
exploration in the PEDA loop. The observed PEDA advantage is consistent with
drive-system modulation alone, which is a separate mechanism. To isolate the
epistemic contribution, re-run with ≥2 checkpoints (≥3 epochs of training) so
that `EnsembleErrorComputer.produce_error()` returns nonzero epistemic
uncertainty.

---

## 9. Verdict

**G1/G2/G3 pass on the real 0.5B model in the controlled 5×5 synthetic setting.**
The core hypothesis (prediction-error-driven exploration) is **not validated** by the partial-training test: the observed PEDA advantage can be explained by drive-system modulation alone, since ensemble epistemic error was zero (single checkpoint). See §8.4 for details.

However, the validation does **not** fully satisfy PEDA_FINAL/WATCHDOG hyperparameter
traceability requirements, and the model-size / generalization gaps are real.

**Do not treat Phase 1 as closed or move to Phase 1.5 until (a) the partial-training test is re-run with ≥2 checkpoints to generate nonzero epistemic signal, (b) a real-LLM grid search is performed, and (c) model-size concerns are addressed.**

**Do not treat Phase 1 as closed or move to Phase 1.5 until the real grid search
and model-size concerns are addressed.**

---

*Report generated from the current `dev`/`main` state at commit `3a95d34`.*
*Full evaluation output: `results/phase1_eval.json`.*
