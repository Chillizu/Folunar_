# Slice 3: Epistemic vs Pragmatic Controlled Experiment

## Goal
Compare PEDA (epistemic+pragmatic) vs Pragmatic-only in UNKNOWN sandbox areas.

## Setup
- Adapter: `checkpoints/phase2/sandbox_adapter_v2_partial`
- Agent A: `pragmatic_only=False` (PEDA)
- Agent B: `pragmatic_only=True` (Pragmatic)
- Both: same `pragmatic_weight=3.0`, same DriveWeights, same `--fast` mode
- max_candidates=12, max_steps=15
- Tasks: count_users, find_errors, read_changelog, find_admin, count_logs
- Episodes: 10 per task per agent

## Implementation
Create or adapt a script (`scripts/phase2_epistemic_test.py`) that:
1. Loads WM with partial adapter, creates both ActionGenerators
2. For each task × episode, runs BOTH agents from identical starting state
3. Records: FHT, SCR, steps, action sequence, dead_loop_rate
4. Saves per-episode results to `results/phase2_epistemic_test.json`

Reference: `scripts/phase2_collect_data.py` `_build_ag()` and `run_peda()`.

## Output
- `results/phase2_epistemic_test.json` — all episode metrics
- Summary table: PEDA vs Pragmatic per task

## Success
- ≥3/5 tasks where PEDA beats Pragmatic → positive signal
- All identical → negative result (also valid per RESEARCH_CHARTER)
