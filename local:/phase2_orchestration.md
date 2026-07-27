# Orchestration Plan: Phase 2 Core Hypothesis Validation

## Goal
Answer: does prediction error (epistemic signal) drive exploration better than pure pragmatic, in a partially-trained sandbox WM?

## Experimental Design (adapted from Phase 1.5 partial training)
1. Train adapter on KNOWN sandbox areas only (`/sandbox/docs`, `/sandbox/data`)
2. Test on UNKNOWN areas (`/sandbox/logs`, `/sandbox/projects`) with v2 tasks
3. Compare PEDA (epistemic+pragmatic) vs Pragmatic-only on same tasks/starts
4. Measure: FHT, SCR, behavioral entropy, coverage

## Slices

| Slice | Agent | Contract |
|-------|-------|----------|
| 1. Partial data prep | task | `local://phase2_slice1_data_split.md` |
| 2. Partial adapter train | task | `local://phase2_slice2_partial_train.md` |
| 3. Controlled experiment | task | `local://phase2_slice3_epistemic_test.md` |
| 4. Report + working log | task | `local://phase2_slice4_report.md` |

## Dependencies
- Slice 2 depends on 1
- Slice 3 depends on 2
- Slice 4 depends on 3

## Budget
- Credits needed: 4 subagent spawns
- Estimated wall time: ~30 min (training CPU) + ~30 min (evaluation CPU)
