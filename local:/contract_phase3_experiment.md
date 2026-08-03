# Phase 3 Epistemic Validation Experiment — Sub-Plan Contract

## Role
You are the **Phase 3 Mini-Orchestrator**. Your job:
1. Design the detailed experimental protocol
2. Break into independent slices, spawn sub-subagents
3. Monitor, collect results, run statistical analysis
4. Report whether the core hypothesis passes or fails

## CRITICAL: Working Directory
All code lives at `/home/chillizu/Projects/Folunar_/`. Do NOT use `Kimi_Agent_Folunar_评估与优化/`.
Always `cd /home/chillizu/Projects/Folunar_` before running anything.

## The Core Hypothesis
> Prediction-error-driven (epistemic) exploration produces measurably better task completion than pragmatic-only planning when the World Model is uncertain.

## Pre-Read (MUST read before designing experiment)
1. `PEDA_FINAL/RESEARCH_CHARTER.md` — research questions and methodology
2. `PEDA_FINAL/PEDA_ENGINEERING_PLAN_v2.md` section 6 (Phase 3 design)
3. `PEDA_WORKING_LOG.md` — search for "Slice 3", "partial", "PEDA vs pragmatic", "goal_unknown"
4. `checkpoints/phase2/sandbox_adapter_v2_partial/trained_manifest.json` — partial adapter config
5. `scripts/phase2_collect_data.py` — understand how _run_agent() works, how baselines are set
6. `scripts/phase2_synthetic_train.py` — understand training pipeline
7. `src/phase2/sandbox_env.py` — understand sandbox state/action space

## Experimental Design

### Independent Variable
- **Agent type**: Full PEDA (epistemic + pragmatic EFE) vs Pragmatic-only (epistemic_weight=0)
- Both use the SAME partial adapter (`sandbox_adapter_v2_partial`, trained on 40 known transitions)

### Dependent Variables
- Task success rate (primary)
- Mean steps to completion
- Revisit rate
- Mean epistemic error
- First action choice (exploratory vs task-directed)

### Conditions
- **goal_known**: Task goal visible in state representation (start in known cwd)
- **goal_unknown**: Task goal NOT visible (start in unknown cwd — not in training data)

### Sample Size
- N >= 10 episodes per condition per agent type
- Total: 2 agents × 2 conditions × 10 = 40 episodes minimum

### Statistical Test
- Fisher's exact test for success rate (goal_unknown: PEDA vs Pragmatic)
- Mann-Whitney U for steps-to-completion
- Alpha = 0.05, report exact p-values

## Execution Slices

### Slice 1: Code Preparation
**Goal**: Ensure the evaluation script can run PEDA vs pragmatic-only with partial adapter on sandbox v2.
- Read `scripts/phase2_collect_data.py` — verify it supports:
  - Loading a specific adapter (not default)
  - Setting `pragmatic_only=True` flag
  - Recording per-episode metrics (success, steps, revisit, epistemic error)
  - Starting from arbitrary cwd (for goal_unknown condition)
- If any capability missing, write/modify the script (delegate to sub-subagent)
- **Acceptance**: A script or confirmed existing capability that can run the experiment

### Slice 2: Run goal_known Condition
**Goal**: Run PEDA and pragmatic-only on known regions, establish baseline.
- 10 episodes PEDA, 10 episodes pragmatic-only
- Tasks: read_note (goal: find secret key in docs/note.txt)
- Start cwd: one of the 3 known directories from partial adapter training
- Record all metrics per episode
- **Acceptance**: Results file + quick check that both succeed (expected since known region)

### Slice 3: Run goal_unknown Condition
**Goal**: THE CRITICAL SLICE — run on unknown regions.
- 10 episodes PEDA, 10 episodes pragmatic-only
- Tasks: read_note (same task)
- Start cwd: one of the 4 UNKNOWN directories (NOT in training data)
- Record all metrics per episode
- **Acceptance**: Results file

### Slice 4: Statistical Analysis
**Goal**: Run statistical tests and produce conclusion.
- Load results from Slices 2 and 3
- Fisher exact test for goal_unknown success rate
- Mann-Whitney U for steps
- Effect size (Cohen's h for proportions, Cliff's delta for continuous)
- Produce a decision: PASS (PEDA > pragmatic, p < 0.05) or FAIL (no significant difference)
- **Acceptance**: Analysis report with p-values, effect sizes, and final verdict

## Output Format
```json
{
  "code_status": { "script_ready": bool, "changes_needed": ["str"] },
  "goal_known": { "peda_success_rate": float, "pragmatic_success_rate": float, "peda_mean_steps": float, "pragmatic_mean_steps": float },
  "goal_unknown": { "peda_success_rate": float, "pragmatic_success_rate": float, "peda_mean_steps": float, "pragmatic_mean_steps": float },
  "statistical_analysis": { "test": "str", "p_value": float, "effect_size": float, "verdict": "PASS|FAIL|INCONCLUSIVE" },
  "conclusion": "str (one paragraph answering: does epistemic signal drive exploration?)"
}
```

## Rules
- Sub-subagents: use them aggressively. Each slice = at least one sub-subagent.
- Do NOT run long evaluations (10+ episodes) yourself — delegate.
- When a sub-subagent finishes, verify its output before reporting.
- If the code doesn't support the experiment, fix it (via sub-subagent) before running.
- Write all results to `results/phase3_experiment/` directory.
- If inference is too slow (>5 min/episode), consider reducing max_steps or using a simpler task.
- The pilot (N=1) showed PEDA 2 steps vs Pragmatic 20 steps on goal_unknown — use this as a sanity check but do NOT treat it as confirmed.
