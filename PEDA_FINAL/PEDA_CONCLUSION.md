# PEDA Conclusion: Predictive-Error-Driven Exploration in LLM Agents

**Status:** FINAL — Project Close
**Date:** 2026-07-31
**Base Document:** `PEDA_FINAL/RESEARCH_CHARTER.md`

---

## Executive Summary

PEDA asked whether prediction error from an LLM-based World Model can serve as an intrinsic drive signal that guides an agent to actively explore uncertain regions, outperforming purely goal-directed behavior. The core hypothesis — that Expected Free Energy (EFE) minimization with epistemic uncertainty drives more effective exploration than pragmatic heuristics — was tested across 17+ controlled experiments spanning 5 environments (Grid World, Sandbox v1/v2/v3/v4, Grid Maze, Giant Maze, TextWorld), using 4 model architectures (Qwen2.5-0.5B with LoRA, JEPA MLP predictors, RSSM, STRIPS action models), with ~2,000 total evaluation episodes.

All three charter sub-questions answer **No** under the tested conditions:

- **Q1 (Signal):** LLM World Models produce epistemic error ~0 on small state spaces (<100 states) and uniform uncertainty on larger ones (JEPA ensemble, all DLR ~0.996). The model is too certain or uniformly uncertain — never differentially uncertain.
- **Q2 (Drive):** EFE is dominated by pragmatic value. The epistemic term only changes action selection when all candidates are equally unpromising — at which point any action is equivalent.
- **Q3 (Effect):** PEDA never beats count-based novelty. The one statistically significant result (Phase 3, N=20, p=0.0043) is attributable to candidate set engineering and success caching, not epistemic prediction error. Phase 8 confirmed: count-driven reaches 62.2% across 9 tasks; toggling JEPA on adds zero delta.

Count-based pair novelty, not epistemic prediction error from learned World Models, is the reliable exploration mechanism in state spaces under ~1,000 states. This negative result is a valid scientific conclusion per the research charter, and constitutes genuine knowledge about the feasibility of Active Inference with LLM-based agents.

---

## The Hypothesis

From `PEDA_FINAL/RESEARCH_CHARTER.md`, the core question decomposes into three sub-questions:

**Q1 (Signal):** Can an LLM-based World Model produce measurable, prediction-quality-related epistemic error signals? Specifically, can ensemble-based epistemic uncertainty be detected (epistemic error > 0)?

**Q2 (Drive):** Can this prediction error signal drive an Action Generator to select exploratory behavior? Does the Expected Free Energy formulation produce different action selections than purely pragmatic (distance-minimizing) policies?

**Q3 (Effect):** Is prediction-error-driven exploration more effective than baselines (purely pragmatic, purely random, heuristic)? Does it lead to faster learning, better state coverage, or higher task success?

All three questions are accepted as falsifiable. A negative answer at any level is an accepted, valuable conclusion per the research charter. The answers below are conditional on the specific experimental conditions tested — they do not claim impossibility in general, only that under the conditions explored, the hypothesis does not hold.

---

## Experimental Evidence

| Phase | Environment | Experiment | Hypothesis Q | Result | Verdict |
|-------|-------------|-----------|--------------|--------|---------|
| 1 | Grid World 5x5 | Full-space training, G1/G2/G3 eval | Q1 | G1=1.000, G2=0.434, G3=0.000 — WM perfectly memorizes all 25 cells | FAIL — environment too simple for 0.5B model |
| 1 | Grid World 5x5 | Partial training (25%), 3-epoch ensemble | Q1 | `g1_test_set`=0.8684, epistemic error ~0 from 28/28 state-action probes zero variance | FAIL — model generalizes perfectly even on 6/25 trained cells |
| 1 | Grid World 5x5 | Partial training, PEDA vs pragmatic N=10 | Q2, Q3 | PEDA 2.6 vs Pragmatic 2.6 steps goal_unknown, Fisher p=1.0, MW p=1.0 | FAIL — no behavioral difference; both agents identical |
| 1.5 | TextWorld | 2-room PEDA vs Pragmatic, 2 iterations | Q2 | PEDA distinguishable from Pragmatic (explores key/look), but epistemic ~0 | PARTIAL — drives modulate, not prediction error |
| 2 | Sandbox v1 | L1/L2/L3 held-out (train v1, eval v1) | Q1 | L1=1.000, L2=0.900, L3=0.550 — thresholds met on training distribution | PASS — in-distribution only |
| 2 | Sandbox v2 | L1/L2/L3 held-out (train v1, eval v2) | Q1 | L1=0.800, L2=0.686, L3=0.229 — all below threshold | FAIL — WM does not generalize to new layouts |
| 2 | Sandbox v2 | Multi-baseline read_hello/read_note | Q2, Q3 | read_hello: Pragmatic 1.0s > PEDA 2.8s. read_note: ALL 0% success | FAIL — PEDA cannot beat pragmatic baseline |
| 3 | Sandbox v2 | N=20 confirmatory, read_hello, unknown CWDs | Q2, Q3 | PEDA unknown 7.2 steps vs Pragmatic unknown 10.0, MW p=**0.0043**, d=-1.01 | POSITIVE — but non-epistemic (see SS4) |
| 3 | Sandbox v2 | N=20 negative control, PEDA known vs Pragmatic known | Q2, Q3 | PEDA 10.0 steps vs Pragmatic 6.85 steps, p=0.0043 | CONFIRMED — PEDA pays cost in familiar envs |
| 4A | Sandbox v2 | Closed-loop self-training 4 blocks, N=10 each | Q3 | PEDA+Train: 20%-60%-80%-60% success. PEDA+Freeze: flat 20% | POSITIVE — but success-cache mechanism |
| 4B | Sandbox v2+v4 | Multi-task generalization (4 tasks x 2 baselines x 2 conditions) | Q3 | read_hello peda_unknown 40% (2/5). count_lines/find_secret/read_note: **all zero** hits | FAIL — WM cannot solve any task beyond cat hello.txt |
| 4B | Sandbox v4 | Phase 3 replication with corrected metric (fht) | Q3 | Phase 3 replicated: peda_unknown 35-40% hit, pragmatic_unknown 0% | CONFIRMED — only read_hello, only /sandbox/projects |
| 5 | Sandbox v2/v3/v4 | JEPA forward dynamics + hybrid, 11 exps | Q1, Q2, Q3 | Novelty-only 50% > jepa_efe 17% on read_hello. JEPA loss converges, no exploration gain | FAIL — JEPA uncertainty flat across all unexplored states |
| 5 | Sandbox v4 | Pure epistemic (jepa_only) explorer | Q2 | SCR ~0 across all tasks, zero room exploration | FAIL — epistemic signal too weak to drive useful behavior |
| 6 | Grid Maze 10x10 | Count vs JEPA vs hybrid (1100 states) | Q3 | Count: 100% goal-reaching. JEPA: 0%. Hybrid: 67% (count-driven carries JEPA) | FAIL — at 1100 states, count is already optimal |
| 6 | Grid Maze 20x20 | Count vs JEPA (8400 states) | Q3 | Count: 0%. JEPA: 0%. Both agents hit state-space ceiling | FAIL — neither approach scales to 8400+ states |
| 7 | GPU 5-track | 5 independent tracks (RSSM, Goal-JEPA, Giant-JEPA, Curriculum, Count) | Q1, Q3 | All JEPA tracks: DLR ~0.996 (near-perfect determinism, zero epistemic signal). Count wins every track | FAIL — JEPA produces no differentiable epistemic signal at any scale tested |
| 8 | Sandbox v2 | Count-driven closed-loop agent across 9 tasks | Q3 | Count-driven: **62.2% avg success rate** across 9 tasks. JEPA toggle adds **zero delta** | FAIL — JEPA contributes nothing beyond what count-based novelty already provides |

---

## The One Positive — And Why It Does Not Salvage the Hypothesis

Phase 3 N=20 produced a statistically significant result: PEDA in unknown CWDs achieved 7.2 mean steps vs Pragmatic 10.0, Mann-Whitney p=0.0043, Cohen's d=-1.01. The crossover interaction (advantage flips direction between known and unknown conditions, p=0.0001) appears to confirm the epistemic hypothesis.

**This does not validate prediction-error-driven exploration.** Three independent factors explain the result without invoking epistemic signal:

1. **Candidate engineering, not exploration.** PEDA's advantage in `/sandbox/projects` (2.0 steps vs 10.0, p=0.0004) comes from the `NovellyExplorer`'s candidate set containing the correct action `cat hello.txt`, combined with the success cache replaying it after the first hit. The World Model did not "explore" — it found the correct action in its limited candidate set, and the pragmatic term selected it. When the candidate set did not contain the correct action (`/sandbox/logs`, `/sandbox/tmp`), PEDA performed identically to Pragmatic at 10.0 ceiling steps.

2. **Non-epistemic mechanism.** PEDA's candidate generation includes action variants that Pragmatic's hardcoded candidate set does not probe. When PEDA succeeds in `/sandbox/projects`, it is because the candidate generator produced `cat hello.txt` and the pragmatic term (distance to goal predicate) was sufficient to select it. No epistemic uncertainty computation contributed to the action selection — the effect is entirely in the action candidate generation, not in the EFE formulation.

3. **No replication on any other task.** Phase 4B across 4 tasks showed hit rate = 0% on count_lines, find_secret, and read_note for ALL baselines including PEDA. The effect is entirely limited to `read_hello`, on a single CWD out of six tested. When the World Model has zero competence in a task domain, PEDA offers nothing — no epistemic exploration is triggered because there is no prediction to be uncertain about.

The Phase 3 result is a valid finding about candidate set engineering and action visibility, not about prediction-error-driven exploration. It does not salvage the core hypothesis.

---

## Five Root Causes

### 1. World Model Too Certain

The 0.5B Qwen2.5 model with LoRA memorizes its training distribution near-perfectly. In Grid World (Phase 1), even with only 25% training data (6/25 cells) and 3 epochs, held-out generalization reached `g1_test_set`=0.8684 and ensemble variance was zero across 28/28 state-action probes. In Sandbox v2 (Phase 2), the WM reached L1=1.000 on its training distribution after 200 curated transitions. The model does not produce epistemic uncertainty because the state spaces we tested are small enough for LoRA fine-tuning to memorize the deterministic transition dynamics completely. Epistemic error ~0 means EFE collapses to pragmatic value alone — Active Inference's exploration advantage evaporates.

### 2. Action Space Engineering Masks Exploration

PEDA's apparent wins depend on the candidate set, not on intrinsic exploration. The `NovellyExplorer` and candidate generators (`generate_sandbox_candidates`) produce action variants that the `PragmaticExplorer` does not. When the candidate set contains the solution, any reasonable selection method (including random) will find it eventually. PEDA's advantage in `/sandbox/projects` is a candidate-set artifact: Pragmatic's heuristic candidates did not include `cat hello.txt` for that CWD, while PEDA's broader candidate generation did. This artifact was discovered in Phase 3 analysis and confirmed in Phase 5-8 experiments where the candidate generation was held constant.

### 3. JEPA Uncertainty Is Flat

JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states. The learned signal is "how uncertain am I about this (state, action) transition?" — which is equally high for every transition the agent has never seen. This is identical to count-based novelty (unvisited = uncertain) but computed at approximately 37x the computational cost (MLP forward pass + embedding computation vs integer increment). Phase 5-7 experiments across multiple environments (v2-v4 sandboxes, grid mazes up to 8400 states) all converged on the same finding: JEPA's epistemic signal is indistinguishable from counting in its behavioral consequences. For JEPA to beat counting, it would need to differentiate between unexplored directions (e.g., "this unexplored region may contain the goal"), which requires goal-conditioned embeddings or learned value representations — neither of which was deployed in any Phase.

### 4. World Model Uncalibrated: High Confidence, Low Accuracy

The World Model shows high confidence in its predictions even when those predictions are wrong. In Phase 4B, PEDA in known CWDs has 0% hit rate on read_hello because the WM confidently predicts file contents that do not match the actual sandbox state. The agent "confidently picks wrong actions." This phenomenon — high confidence + low accuracy = confidently wrong — is the calibration failure that prevents any epistemic signal from being reliable. The confidence penalty (inject noise when `avg_conf > 0.95`) was added as a mitigation but treats the symptom, not the cause: the model's predictive distribution does not reflect its actual competence. Without proper calibration, any epistemic signal derived from model confidence is unreliable.

### 5. CPU Bottleneck Prevented Systematic Ensemble Evaluation

Real-LLM evaluation with Qwen2.5-0.5B on CPU takes ~176s cold start + ~3s per inference call. With 3 checkpoint ensembles and 4-8 candidates per step, each agent step requires 12-24 model calls. A single sandbox episode with PEDA can take 10-60+ minutes. The Phase 3 confirmatory experiment was estimated at 60-120 hours on CPU — infeasible. This bottleneck forced the project to switch to heuristic proxies (model confidence as epistemic proxy, single-checkpoint evaluation) long before the core hypothesis could be properly tested. Ensemble-based epistemic uncertainty (the theoretically correct approach) was only evaluated in one Grid World experiment (Phase 1, N=10, 2.6 vs 2.6 steps, p=1.0) before the team concluded that the CPU path was not viable and the GPU path was not available for systematic exploration.

---

## What Survived

Not every component failed. Several engineering contributions validated by this project are independently useful for future agent architectures:

- **Count-based pair novelty.** The `NovellyExplorer` with (state, action) pair counting achieves optimal exploration in environments with <1000 states. It handles stochastic items (file listings that change across episodes) by treating each pair independently. Success rate on Sandbox v2 read_hello: 40% (PEDA with count-based + success cache), zero dead-loops. The count-driven Phase 8 agent reached 62.2% success across 9 tasks on the v2 sandbox, confirming that simple counting is the correct tool for this problem class.

- **STRIPS action learning.** Learned action schemas (preconditions + effects) from execution traces reach 45.8% learned vs 31.3% fallback on the action prediction task. The `ActionModelLearner` correctly extracts cwd-change preconditions and filesystem effects from observed transitions. This is a lightweight, interpretable alternative to neural forward models for deterministic Linux commands.

- **Data-driven candidate generation pipeline.** The sandbox candidate generator evolved from hardcoded heuristics (v1, 4 candidates) to data-driven enumeration (v2, 65 pairs; v3/v4, 270+ pairs) with zero crashes during migration. This pipeline is robust and reusable for any future Linux sandbox agent.

- **Success cache.** One-step solves for seen state-action pairs, derived from the `SandboxState` hash. This cache provided the mechanism behind Phase 3's positive result: once PEDA discovers `cat hello.txt` in `/sandbox/projects`, the cache replays it instantly on subsequent episodes. Simpler than any learned policy, and equally effective in this domain.

- **PEDA's dead-loop immunity.** Across all phases (1-8), PEDA consistently showed zero dead-loop rate, versus Pragmatic's 48-80% and Random's variable rates. The drive system's boredom term and candidate diversity provide reliable dead-loop avoidance, even when no epistemic signal exists. This is a practical engineering contribution even if the theoretical mechanism is not prediction error.

---

## What We Learned

The negative result IS knowledge — meeting the charter's success criterion: *"We have a deeper understanding of Active Inference feasibility in LLM-based agents than before the project started."*

**1. Epistemic signal requires state spaces large enough that the model CAN be uncertain about parts of them.** A 0.5B parameter model with LoRA on 25-200 training examples generalizes perfectly on 5x5 grids and ~65-state sandboxes. The model is too capable for the environment. Epistemic uncertainty from ensemble variance requires either much larger state spaces (>10,000 states), much larger models that hallucinate or show genuine confusion on OOD inputs, or environments with genuinely stochastic dynamics that prevent perfect memorization. The sweet spot for testing Active Inference with LLMs is not small, simple environments — it is environments large enough to produce differentiated prediction errors.

**2. JEPA-style forward dynamics train but do not differentiate actions.** The JEPA MLP predictor learns to predict next-state embeddings from (state, action) pairs, as shown by decreasing loss curves across all experiments (loss 45 to 15). But the learned uncertainty is a scalar per transition, not a comparative signal across actions. All unexplored transitions are equally uncertain — equivalent to counting, at 37x the computational cost. Breaking this uniformity requires goal-conditioned embeddings or learned value in the representation space: the model must know not just "this action leads to an unknown outcome" but "this action leads to an outcome that might help achieve the goal."

**3. EFE is dominated by pragmatic value at any practically testable horizon.** The Expected Free Energy formulation (epistemic + pragmatic) collapses to pragmatic-only when the WM produces near-zero epistemic error, or when the pragmatic term contains goal-distance information. At horizon 1-3 (the practical limit given CPU/GPU inference costs of ~3-30s per model call), the pragmatic term dominates by 3-10x. The epistemic term only changes action selection when pragmatic value is near-zero across all candidates — which means all candidates are equally unlikely to reach the goal, making any selection equivalent. This is not a fixable hyperparameter issue; it is a structural property of EFE in goal-directed tasks with small lookahead horizons.

**4. Counting is surprisingly robust even with stochastic environment elements.** The count-based (state, action) pair novelty explorer handles the stochastic sandbox environment effectively. File listings change between episodes, timing output varies, directory contents differ — but pair-counting tolerates all of this by treating each (cwd, command) pair as an independent counter. The novelty value for a pair decreases with visits, naturally driving the agent from visited to unvisited states. This is simple, computationally free (integer increment), and empirically matches or exceeds every learned exploration signal we tested.

**5. Bootstrap data quality is the critical bottleneck for any learned exploration mechanism.** Every approach — PEDA, JEPA, RSSM, STRIPS — depends on initial training data to build a useful model. The Cold Start problem (no model without data, no exploration without model) is not solvable by better exploration algorithms: without a minimal set of diverse (state, action, next_state) transitions, no learned model can predict anything useful, and without useful predictions, no exploration signal can be generated. The project spent approximately 50% of total engineering effort on data collection and pipeline infrastructure (random baselines, heuristic collection, expert demos, sandbox v1-to-v4 migration). The quality ceiling is set by the training data, not the exploration algorithm. Any future approach must solve the bootstrap problem first.

---

## Declaration

The PEDA hypothesis — that prediction error from an LLM-based World Model can drive autonomous exploration more effectively than baselines in LLM-based agents — is **DISPROVEN** under the conditions tested:

- **Model:** Qwen2.5-0.5B-Instruct with LoRA (rank=16), JEPA MLP predictors (1-3 hidden layers), zero-shot RSSM
- **Environment:** Busybox Linux sandbox (4-7 directories, 14-65 files), Grid Maze (1100-8400 states), Grid World (25 cells), TextWorld (2 rooms)
- **Training:** 65-1378 transitions, 1-3 epochs LoRA, 500-2000 steps JEPA, CPU (Intel Core Ultra 9) or GPU (NVIDIA T4 16GB)
- **Exploration signal:** EFE with ensemble variance, model-confidence proxy, JEPA hidden-state cosine distance, JEPA MLP prediction loss
- **Baselines:** Pragmatic (goal-distance minimization), Random (uniform action selection), Count-based pair novelty, Heuristic (command templates)
- **Tasks:** read_hello, count_lines, find_secret, read_note, read_welcome, find_api_key, count_measurements, find_errors_v4, read_changelog_v4

The project achieved its charter goal: we understand the feasibility of Active Inference in LLM-based agents far better than before. The answer is: under practical conditions (small state spaces <1000 states, 0.5B models, LoRA fine-tuning, CPU-limited inference), count-based novelty is the reliable exploration mechanism. Epistemic prediction error from learned World Models does not produce behaviorally distinguishable exploration — it either equals zero (WM too certain), matches counting at extreme computational cost (JEPA), or is dominated by pragmatic value (EFE collapse at horizon 1-3).

**This is a valid scientific result.** It closes one specific research path (LLM-based Active Inference with ensemble/JEPA uncertainty as exploration drive) and constrains the search space for future work. The charter explicitly accepts negative results as valuable knowledge; PEDA was designed from the start to produce honest findings regardless of sign. By its own stated criteria, the PEDA project was successful.

---

## Appendix: Archive File References

| Phase / Document | File |
|------------------|------|
| Research Charter | `PEDA_FINAL/RESEARCH_CHARTER.md` |
| Architecture v1.1 | `PEDA_FINAL/peda_report_v11.agent.final.md` |
| Postmortem Reflection | `PEDA_FINAL/peda_reflection_v11.md` |
| Independent Review | `PEDA_FINAL/peda_independent_review.md` |
| Phase 1 (Grid World) | `PEDA_FINAL/archive/phase1/phase1_validation_report.md` |
| Phase 1 Gap Report | `PEDA_FINAL/archive/phase1/phase1_gap_report.md` |
| Phase 1 Epistemic Blocker | `PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md` |
| Phase 1 Partial Training | `PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md` |
| Phase 1.5 (TextWorld) | `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md` |
| Phase 2 Controller | `PEDA_FINAL/archive/phase2/CONTROLLER_DIRECTIVE_PHASE2.md` |
| Phase 2 Infrastructure | `PEDA_FINAL/archive/phase2/phase2_infrastructure_report.md` |
| Phase 2 Adapter Training | `PEDA_FINAL/archive/phase2/phase2_adapter_train_report.md` |
| Phase 3 N=20 | `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` |
| Phase 3 GPU Grid World | `results/phase3_gpu/report.json` |
| Phase 4 Plan | `PEDA_FINAL/PHASE4_EXPERIMENT_PLAN.md` |
| Phase 4A Results | `results/phase4a/PHASE4_RESULTS.md` |
| Phase 4B Rerun | `results/phase4b_rerun/ANALYSIS_REPORT.md` |
| Phase 5 JEPA Archive | `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` |
| Phase 6 Grid Maze | `scripts/phase6_*.py`, `src/phase6/` |
| Phase 7 GPU 5-Track | `scripts/phase7_*.py`, `src/phase7/` |
| Phase 8 Count-Driven | `scripts/phase8_closed_loop.py`, `src/phase8/` |
| Research Manuscript | `PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md` |
| Engineering Plan v2 | `PEDA_FINAL/PEDA_ENGINEERING_PLAN_v2.md` |
| Working Log | `PEDA_WORKING_LOG.md` |
| Project Guide | `AGENTS.md`, `PEDA_FINAL/README_FOR_AGENTS.md` |
