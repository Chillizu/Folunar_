# Phase 2 Remaining Work — Sub-Plan Contract

## Role
You are a **Phase 2 Mini-Orchestrator**. Your job:
1. Break this contract into independent execution slices
2. Spawn your own sub-subagents to execute each slice
3. Monitor, collect results, and report back
4. You do NOT execute code yourself — you delegate to sub-subagents

## CRITICAL: Working Directory
All code lives at `/home/chillizu/Projects/Folunar_/`. Do NOT use `Kimi_Agent_Folunar_评估与优化/`.
Always `cd /home/chillizu/Projects/Folunar_` before running anything.

## Pre-Read (MUST read before writing sub-plans)
1. `PEDA_WORKING_LOG.md` — search for "C18", "oscillation", "held-out", "C17" 
2. `PEDA_FINAL/PEDA_ENGINEERING_PLAN_v2.md` sections 5 and 14 (C17/C18 rules)
3. `WATCHDOG.md` — rules C17 and C18 full text
4. `checkpoints/phase2/` directory — list all adapters
5. `scripts/phase2_collect_data.py` — understand _run_agent()
6. `scripts/phase2_measure_l1l2l3.py` — understand how metrics are measured
7. `scripts/phase2_create_ood_test.py` — understand held-out test creation

## Tasks (order matters within each slice, slices can parallelize)

### Slice A: Held-Out Test Set (C17)
**Goal**: Create a held-out test set for sandbox v2 and measure WM accuracy on it.
- Use sandbox v2 (Docker image `peda-sandbox:v2`)
- Create test set of states/actions NOT seen in training data
- Measure L1 (exit code), L2 (filesystem delta), L3 (output summary) on best adapter (e2)
- Report numbers and comparison with in-distribution performance
- **Acceptance**: A results file with held-out L1/L2/L3 numbers

### Slice B: Post-Completion Oscillation Fix (C18)
**Goal**: Fix the behavior where PEDA oscillates between `cat docs/note.txt` and `ls` after completing a task.
- Read the WATCHDOG C18 rule for context
- The issue: after task completion (FHT=0), EFE doesn't penalize already-visited states
- Investigate root cause in `scripts/phase2_collect_data.py` _run_agent()
- Propose and implement a fix (e.g., add a "task_complete" terminal state or increase revisit penalty post-completion)
- Verify with a smoke test: run PEDA on `read_note`, confirm it stops oscillating after task completion
- **Acceptance**: Code change + smoke test showing no post-completion oscillation

### Slice C: Multi-Baseline Evaluation
**Goal**: Run PEDA evaluation against multiple baselines on sandbox v2.
- Baselines: random, pragmatic-only (epistemic_weight=0), full PEDA
- Use best adapter (e2: 200 curated, L1=1.000)
- Tasks: read_note, list_files (at minimum)
- 5 episodes per condition per task
- Report: success rate, mean steps, revisit rate, epistemic error
- **Acceptance**: A results JSON + summary table

## Output Format
Return a structured report:
```json
{
  "slice_a": { "held_out_l1": float, "held_out_l2": float, "held_out_l3": float, "file": "path" },
  "slice_b": { "fix_description": "str", "smoke_test_result": "str", "files_changed": ["path"] },
  "slice_c": { "results_file": "path", "summary": "str" }
}
```

## Rules
- Sub-subagents: spawn them aggressively for parallel work. Each slice = at least one sub-subagent.
- Do NOT run long CPU tasks yourself — delegate.
- When a sub-subagent finishes, verify its output before reporting.
- If a sub-subagent gets stuck or produces wrong output, fix the contract and re-spawn.
- Write all results to `results/phase2_remaining/` directory.
