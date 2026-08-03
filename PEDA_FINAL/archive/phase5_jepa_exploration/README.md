# Phase 5 JEPA Exploration — Archive

**Date:** 2026-07-30
**Decision:** ARCHIVED — negative result
**Core hypothesis:** Learned forward dynamics (JEPA) → better exploration decisions

## Summary

Across 11 experiments spanning 4 sandboxes (v2/v3/v4 grid maze deterministic/stochastic), JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**.

| Environment | States | Count-based | JEPA best | Pure JEPA |
|---|---|---|---|---|
| Sandbox v2 | 65 | 50% | 50% (hybrid) | — |
| Sandbox v4 | 270 | 42% | 8% (hybrid) | — |
| Maze 10x10 | 1,100 | 100% | 0% | — |
| Maze 20x20 | 8,400 | 0% | 0% | — |
| Stochastic 10x10 | 1,100 | 100% | 67% | 0% |
| P4 EFE (4 rounds) | 65-270 | 50% | 25% | — |

## What DID Work

| Component | Status | Note |
|---|---|---|
| STRIPS action learning | KEEP | 45.8% learned vs 31.3% fallback |
| Count-based novelty (pair) | KEEP | Optimal at <1000 states, handles stochastic items |
| Data-driven candidate generation | KEEP | v2→v3 migration 0 crashes |
| JEPA MLP predictor | UNCERTAIN | Loss always converges (45→15), MLP learns dynamics |

## What Failed

- **EFE with binary pragmatic**: dominated by 0/1 term, α irrelevant
- **Hybrid (novelty + epistemic)**: never beat pure novelty
- **Pure epistemic (jepa_only)**: SCR ~0, no room exploration
- **Scaling**: 8400 states still too small for epistemic advantage

## Root Cause

JEPA predicts "(state, action) → next embedding". Its uncertainty is "how uncertain am I about this transition?" All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower.

For JEPA to beat counting, it needs to say "THIS unexplored direction is more promising than THAT one." This requires goal-conditioning or learned value in the embedding space — beyond what was tested.

## Archived Files

All source files remain in place (imports preserved) but exploration is concluded:
- `src/phase4/jepa_peda.py` — JEPA-based PEDA with EFE
- `src/phase5/jepa_wm.py` — JEPAEnsemble (MLP predictors)
- `src/phase5/explorer.py` — NoveltyExplorer (count-based)
- `src/phase5/action_model.py` — STRIPS action learner
- `src/phase6/*` — Grid maze environment + experiments
- `scripts/phase4_jepa_experiment.py` — P4 EFE experiment
- `scripts/phase5_*.py` — Phase 5 experiments
- `scripts/phase6_*.py` — Phase 6 experiments
- `Dockerfile.busybox_v{3,4}` — Expanded sandboxes

## Bugs Found (5 fixed)

1. Path predicate cwd-unaware → `_action_hits_target`
2. Novelty tie-breaking alphabetical → action priority ordering
3. `final_state.victory` always False → return next_state
4. DLR 0.53-0.80 → repeat action penalty
5. JEPA predictors cross-task contamination → `reset_predictors()`

## Next (Post-Archive)

Scale-up attempt: 5 independent research tracks testing JEPA at larger scale / different formulations.
