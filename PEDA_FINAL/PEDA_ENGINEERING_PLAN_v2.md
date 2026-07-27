# PEDA Engineering Plan v2.0

**Version**: 2.0  
**Date**: 2026-07-26  
**Status**: Active — Phase 3 (Epistemic Validation) in progress  
**Target Audience**: AI agents (orchestrator + subagents)  
**Base Document**: PEDA Architecture v1.1 → v2.0 rewrite  

---

## 1. Executive Summary

### 1.1 One-Sentence Proposition

PEDA (Predictive-Error-Driven Autonomous Agent) replaces prompt-driven LLM agent control with **Expected Free Energy (EFE) minimization**, testing whether Active Inference can produce autonomous exploration in a Linux sandbox environment.

### 1.2 v1.1 → v2.0 Migration

| Dimension | v1.1 (2026-07-02) | v2.0 (2026-07-26) | Delta |
|-----------|-------------------|-------------------|-------|
| Phase count | 6 phases (incl. Phase 1.5) | 7 phases | +1 (Phase 1.5 merged into Phase 2; Phase 3-7 restructured) |
| Hypothesis validation | Planned, unexecuted | Phase 1 validated (infrastructure), Phase 2 validated (sandbox + data quality) | Major: core hypothesis still open, but experimental method is proven |
| Data quality methodology | Not discussed | 5 strategies with experimental validation | New |
| WM architecture | Token-space LLM prediction | Token-space + Hidden-state epistemic (Light JEPA) | Extension |
| WATCHDOG rules | B1-B9, C1-C14, N1-N3 | +C20-C22 (orchestration) | New |
| Orchestration | Implicit single-agent | Explicit orchestrator/subagent model | New |

### 1.3 Current Status

- **Phase 1** [DONE]: Grid World infrastructure validated; core hypothesis NOT verified due to environment-model mismatch (0.5B model too powerful for 5×5 grid)
- **Phase 2** [DONE]: Busybox sandbox operational; (s,a,s') data collection pipeline validated; PEDA ≠ Pragmatic behavior confirmed (2/2 iterations); decompose_error bug found and fixed
- **Phase 3** [NOW]: Partial-training controlled experiment — epistemic validation with controlled uncertainty
- **Phase 4-7** [NEXT → PLANNED → FUTURE]: Self-training loop, sandbox expansion, knowledge→application, self-modification

### 1.4 Core Hypothesis Status

| Sub-problem | Status | Evidence |
|-------------|--------|----------|
| **Signal**: LLM WM produces measurable epistemic error | **Partially validated** | decompose_error fix: 0.0 → 0.20 (20% epistemic ratio); Light JEPA hidden-state approach pending |
| **Drive**: Epistemic error drives exploration | **Partially validated** | PEDA ≠ Pragmatic confirmed (2/2); Drive system drives exploration, but via boredom+confidence, not pure epistemic |
| **Effect**: PEDA > baseline | **Directional signal only** | Partial train pilot: PEDA 2 steps vs Pragmatic 20 steps failure (goal_unknown); N=1, needs confirmatory |

---

## 2. Project Context & Charter Alignment

### 2.1 Research Questions

From `RESEARCH_CHARTER.md`:

1. **Signal problem**: Can LLM-based WM produce measurable prediction error signals? (epistemic error > 0)
2. **Drive problem**: Can prediction error drive action selection? (EFE efficacy)
3. **Effect problem**: Is prediction-error-driven exploration more effective than baselines? (PEDA > baseline)

All three are accepted as open questions. Negative results are valid research conclusions.

### 2.2 Charter-Compliant Practices

| Practice | Implementation |
|----------|---------------|
| Negative results accepted | All Phase 1/1.5 results include explicit negative findings |
| No commercial timeline | 7-phase structure with no hard deadlines |
| Open-source deliverables | Code + data + analysis; no product |
| Quantified evaluation | All success criteria are measurable (not "behavior seems interesting") |
| Pilot vs Confirmatory distinction | N=1-3 = pilot (directional); N≥10 = confirmatory (statistical) |

### 2.3 Key Lessons from Folunar_ / Predecessor

| Lesson (source: `peda_reflection_v11.md`) | PEDA v2.0 mitigation |
|-------------------------------------------|----------------------|
| 40+ modules, <150 lines each | 5 core modules; B3 gate prevents new files without justification |
| Online SGD training → catastrophic forgetting | Intermittent batch update only; no per-step training |
| Process metrics as progress (92.2% "success" via command exec, not task completion) | Go/no-go criteria based on hypothesis validation, not code volume |
| Plan document inflation (~30K words, ~3.7K lines code) | Plan = living document; WATCHDOG B4 demoted to concern for exploration |

---

## 3. Phase Structure Overview (v2.0 7-Phase Layout)

```
Phase 1: Infrastructure [DONE] — Grid World, PEDA core loop validity
      │
      ▼
Phase 2: Sandbox Foundation [DONE] — L1/L2/L3 met, multi-task data quality
      │
      ▼
Phase 3: Epistemic Validation [NOW] — Partial training controlled experiment
      │
      ▼
Phase 4: Self-Training Loop [NEXT] — Reintegrate LearningModule into sandbox
      │
      ▼
Phase 5: Sandbox Expansion [PLANNED] — v3 write-enabled → v4 Python → v5 network
      │
      ▼
Phase 6: Knowledge→Application [PLANNED] — Preference distributions, self-generated goals
      │
      ▼
Phase 7: Self-Modification [FUTURE] — Agent chooses when/what to train
```

### 3.1 Phase Transition Criteria

| Transition | Gate | Criteria | Current Status |
|------------|------|----------|----------------|
| Phase 1 → Phase 2 | G1: WM next-state accuracy > 0.90 | 1.000 (PASS) — but memorization, not learning | PASS (infrastructure) |
| Phase 2 → Phase 3 | G2-3: PEDA ≠ Pragmatic, decompose_error > 0 | PEDA 2-step vs Pragmatic 20-step (partial train); epistemic 0.0→0.20 | PASS (see 5.0) |
| Phase 3 → Phase 4 | Epistemic validation: PEDA > Pragmatic at p<0.05 (10+ eps/cond) | IN PROGRESS | IN PROGRESS |
| Phase 4 → Phase 5 | Self-training loop functional (LearningModule integrated) | NOT STARTED | PENDING |
| Phase 5 → Phase 6 | Sandbox expansion complete (v3-v5) | NOT STARTED | PLANNED |
| Phase 6 → Phase 7 | Knowledge→application pipeline validated | NOT STARTED | FUTURE |

---

## 4. Phase 1: Infrastructure Validation [DONE]

### 4.1 Setup

- **Environment**: 5×5 Grid World, 4 discrete actions (up/down/left/right), 25 discrete states
- **Model**: Qwen2.5-0.5B-Instruct + LoRA adapter
- **Training**: 20 configs × ~24 free cells × 4 actions = ~1920 transitions (synthetic)
- **Evaluation**: 20 episodes per condition

### 4.2 Results (Gate Pass Rates)

| Gate | Metric | Threshold | Achieved | Verdict |
|------|--------|-----------|----------|---------|
| G1 | WM next-state accuracy | > 0.90 | 1.0000 | PASS |
| G2 | Steps / random_steps | < 0.50 | 0.1211 (3.3 vs 27.25) | PASS |
| G3 | Revisit rate | < 0.20 | 0.0000 | PASS |

### 4.3 Critical Flaw Identified

From `PHASE1_EVALUATION.md`:

> "This is not prediction-error-driven exploration — it is perfect-memory-driven navigation."

Root cause: WM trained on full state-action space → near-zero prediction error → EFE ≈ pragmatic * 3.0 → pure greedy distance minimization. The core hypothesis was **not tested** because the environment was too simple for the 0.5B model.

### 4.4 Phase 1 Net Contribution

| Contribution | Confidence |
|-------------|------------|
| Engineering infrastructure validated (LLM loading, LoRA, EFE compute, eval loop) | High |
| Code has no critical bugs (stub and real-LLM paths both pass) | High |
| Core hypothesis NOT validated | Confirmed |
| Experiment design flaw documented (see B7/B8) | High |

---

## 5. Phase 2: Sandbox Foundation [DONE]

### 5.1 Phase 1.5: Text World (merged under Phase 2 in v2.0 schema)

**Environment**: 2-room custom text environment (room north/south; take key, go north, look, inventory)

**Key findings** (source: `PHASE1_5_COMPLETE_EVALUATION.md`, `PHASE1_5_ITERATION2_EVALUATION.md`):

| Finding | Evidence | Strength |
|---------|----------|----------|
| PEDA ≠ Pragmatic behavior distinguishable | PEDA: step 3 (e3) / step 1 (e4) tries `take key`; Pragmatic: `look` × 20 / × 10 | High (2/2 iterations reproduced) |
| Drive System has independent exploration value | Despite epistemic ≈ 0, boredom + confidence drove exploration | High |
| decompose_error bug: has-key dimension missing | Semantic probe: 40% disagreement on has-key; decompose_error reported 0% | Confirmed & Fixed |
| Data augmentation ineffective in simple environment | 200 walks × 30 steps → 114 unique samples (+1 from base) | Confirmed |

### 5.2 Phase 2: Busybox Sandbox

**Environment**: Docker busybox container (FROM busybox:latest), whitelisted commands: ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep

**Architecture decisions** (from `PROMPT_PHASE2_START.md`):

- JSON-structured state representation (replacing free text — GLM-5.2 recommendation)
- Confidence Penalty: if confidence > 0.95, inject noise to EFE (Prevents inventory dead-loop from Phase 1.5)
- Hidden-state epistemic (Light JEPA): compute cosine distance between checkpoint hidden states

**Key L1/L2/L3 validation**:

| Level | Target | Validation |
|-------|--------|------------|
| L1 (Exit code) | ≥90% accuracy | WM predicts exit code correctly for all whitelist commands |
| L2 (Filesystem delta) | ≥70% F1 | Filesystem change prediction operational |
| L3 (Output summary) | ≥50% semantic match | Semantic summary via SBERT cosine > 0.7 |

### 5.3 Data Quality > Quantity

From `PROMPT_PHASE2_START.md`:
> Phase 2 uses microtasks (5-10 steps) rather than 20-step long episodes
> Goal: 1000+ unique samples per collection session, not 10000

### 5.4 Held-Out Evaluation (2026-07-26) — CRITICAL RE-EVALUATION

**Test**: e2 adapter (best: L1=1.000 on v1 sandbox) evaluated on sandbox v2 OOD directories (logs/, projects/, README.txt).

| Level | Target | Held-Out | Threshold Met? |
|-------|--------|----------|----------------|
| L1 (Exit code) | >= 0.90 | **0.800** | FAIL |
| L2 (Filesystem delta) | >= 0.70 | **0.686** | FAIL |
| L3 (Output summary) | >= 0.50 | **0.229** | FAIL |

**Conclusion**: WM does NOT generalize to new directory layouts. The claimed Phase 2 success metrics (L1=1.000, L2=0.900, L3=0.550) are **specific to v1 sandbox (4 directories)** and do not hold on v2 sandbox (7 directories). read_note task: 0% success rate across all baselines on v2 sandbox, contradicting the earlier "1-step completion" claim which was only valid on v1.

### 5.5 Multi-Baseline Evaluation (2026-07-26)

| Baseline | read_hello | read_note |
|----------|-----------|-----------|
| PEDA | 80% (2.8 steps) | 0% (10.0 steps) |
| Pragmatic-only | 100% (1.0 steps) | 0% (10.0 steps, 80% revisit) |
| Random | 100% (3.0 steps) | 0% (10.0 steps) |

**Finding**: Pragmatic outperforms PEDA on read_hello. All baselines fail read_note on v2 sandbox. The e2 adapter's task-completion ability is sandbox-layout-dependent.

### 5.6 Phase 2 Net Contribution (Revised)

| Contribution | Confidence |
|-------------|------------|
| Docker sandbox v1 + v2 operational | High |
| JSON-structured state reduces aleatoric noise | High |
| Data quality > quantity proven (e2 200 curated > e3 10k random) | High |
| C18 post-completion oscillation fix (game_over guard) | High |
| PEDA ≠ Pragmatic confirmed (Phase 1.5 Text World) | High |
| **WM does NOT generalize v1→v2 sandbox** | **Confirmed (new)** |
| **Core hypothesis still unvalidated** | **Confirmed** |

---

## 6. Phase 3: Epistemic Validation [NOW]

**Status**: Code ready, hardware blocked (needs GPU). Scripts at `scripts/phase3_*.py`.

### 6.1 Experimental Design

Following the recommendation from `PHASE1_EVALUATION.md` Section 6 (Option A):

**Method**:
1. Train WM on **half** the state-action space of Grid World
2. The other half constitutes "unknown region" with predictable WM uncertainty
3. Evaluate from unknown-region start states
4. Compare PEDA (full EFE) vs Pragmatic-only (EFE with epistemic term zeroed)

**Why Grid World, not Sandbox**: CPU inference with Qwen2.5-0.5B-Instruct in sandbox takes ~176s cold start + ~3s per call, making 40-episode experiment impractical (60-120h). Grid World inference is 1-2s per call, experiment feasible in ~30-60min. The core hypothesis question is identical in both environments.

**Controlled variables**:
- Same pragmatic_weight = 3.0 for both agents
- Online learning disabled (update_interval = 100000)
- Fresh DriveSystem per episode
- Same (goal, start_seed) pairs

### 6.2 Pilot Results (N=1)

From `PHASE1_PARTIAL_EVALUATION.md`:

| Condition | Agent | Success | Mean Steps | Revisit Rate | g1 (test) |
|-----------|-------|---------|------------|--------------|------------|
| goal_known | PEDA | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_known | pragmatic_only | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_unknown | PEDA | 1.000 | **2.0** | 0.000 | 0.500 |
| goal_unknown | pragmatic_only | 0.000 | **20.0** | **0.905** | 1.000 |

### 6.3 Hardware Status

| Item | Status |
|------|--------|
| Scripts | 4 scripts ready (`scripts/phase3_*.py`) |
| Adapter | `checkpoints/phase1/partial_adapter_real_25_e3` |
| start_cwd support | Added to `sandbox_env.py` and `phase2_collect_data.py` |
| CPU feasibility | FAIL — too slow for sandbox, feasible for Grid World |
| GPU needed | Yes — estimated 10-30 min for full Grid World experiment |

**Interpretation**:
- goal_known: Both agents identical (fairness check PASS — PEDA not magically better)
- goal_unknown: PEDA 2-step success vs Pragmatic 20-step failure
- **Directional signal**: Strong, but N=1 cannot rule out luck
- g1_test_set = 0.8684: WM in unknown region has 86.84% accuracy (not "fully blind")

### 6.3 Statistical Requirements

From `PHASE1_PARTIAL_EVALUATION.md` Section 6:

| Parameter | Value |
|-----------|-------|
| Required N (per condition) | ≥10 episodes |
| Effect size (estimated) | 80% vs 20% success rate |
| Test | Fisher exact (success) / Mann-Whitney U (steps) |
| α | 0.05 |
| Power (1-β) | 0.80 |
| Hardware time estimate | ~5h CPU (10eps × 15min × 2 conditions) |

### 6.4 Success Criteria

| Condition | PEDA target | Pragmatic target | Test |
|-----------|-------------|------------------|------|
| goal_unknown success rate | > 60% | < 40% | Fisher exact p < 0.05 |
| goal_unknown mean steps | < 10 | > 15 | Mann-Whitney U p < 0.05 |
| goal_known fairness | ~pragmatic | ~PEDA | p > 0.05 (no sig. diff) |

### 6.5 Current Status (Phase 3)

**Status**: IN PROGRESS  
**Completed**: Pilot (N=1) shows strong directional signal  
**Pending**: Confirmatory run (N≥10/condition) — requires ~5h CPU  
**Blocking**: None  
**Next action**: Run `scripts/phase1_partial_eval.py --episodes 10`

---

## 7. Phase 4: Self-Training Loop [NEXT]

### 7.1 Objective

Reintegrate the **LearningModule** into the sandbox loop so WM updates automatically from collected (s,a,s') data, closing the training loop.

### 7.2 Design

```
Agent acts in sandbox → collects (s,a,s') tuples
       ↓
Buffer accumulates N samples (N=500-1000)
       ↓
Online LoRA fine-tune on accumulated data
       ↓
WM weights updated → next batch uses improved predictions
       ↓
EFE changes reflect improved WM confidence
```

### 7.3 Key Risks

| Risk | Mitigation |
|------|------------|
| Catastrophic forgetting (Phase 1 lesson) | Intermittent batch update; maintain replay buffer |
| Overfitting to narrow behavior distribution | Curriculum: force diverse action selection during collection phase |
| Training latency impacts eval cadence | Async: train in background, swap weights atomically |

### 7.4 Success Criteria

- WM prediction accuracy improves by ≥5% after first training cycle
- PEDA exploration efficiency improves by ≥10% after WM update (vs before)
- No catastrophic forgetting (performance on old states does not regress >2%)

---

## 8. Phase 5: Sandbox Expansion [PLANNED]

### 8.1 Expansion Stages

| Version | Capability | Risk | Validation |
|---------|-----------|------|------------|
| v3 | Write-enabled (echo, mkdir, touch) | Low | Structured, deterministic |
| v4 | Python interpreter (basic scripts) | Medium | Less predictable, more epistemic signal |
| v5 | Network access (curl whitelist) | High | Requires proxy, security audit |

### 8.2 Epistemic Signal Gradient

Hypothesis: Epistemic signal should **increase** with environment complexity:

```
Grid World (5×5) → Text World (2 rooms) → busybox (v2) → +write (v3) → +python (v4) → +network (v5)
epistemic ≈ 0           ≈ 0.20           low        medium       high         very high
```

Each expansion step is a natural experiment: if epistemic signal fails to increase, the WM architecture may need revision (see Section 12).

---

## 9. Phase 6: Knowledge → Application [PLANNED]

### 9.1 Preference Distributions

Replace uniform preference C(o) with learned preference distributions derived from successful state-action trajectories. If an action consistently leads to "good" states (defined by exit code 0, improved task completion), the WM should assign higher pragmatic value to similar actions.

### 9.2 Self-Generated Goals

When epistemic value is low (environment known) and drive levels are moderate, the agent should generate sub-goals that create new epistemic gradients. Example: "I already know /docs. What happens if I create /projects and run Python there?"

### 9.3 Dependencies

- Phase 4 self-training loop must be stable
- Phase 5 environment must support write operations
- WM must recover from failed predictions (graceful degradation)

---

## 10. Phase 7: Self-Modification [FUTURE]

### 10.1 Vision

The agent decides **when** and **what** to train on, rather than following a fixed training schedule. This is the highest ambition of Active Inference applied to an LLM agent.

### 10.2 Concrete Form

- Agent observes its own prediction error trend over recent steps
- If error is rising → trigger data collection on current task
- If error is flat and low → trigger recall-prioritized replay on high-error past samples
- If Drive levels cross thresholds → re-weight training distribution

### 10.3 Prerequisites

- Phase 3 epistemic validation success (proof that EFE can drive behavior)
- Phase 4 self-training loop operational (proof that training integrates)
- Phase 6 preference learning functional (proof that goals are internalized)

---

## 11. Core Architecture: FEP, EFE & Drive System

### 11.1 Expected Free Energy (EFE)

$$G(\pi) = \underbrace{H[q(o|\pi)]}_{\text{epistemic value}} + \underbrace{D_{KL}[q(o|\pi) \,||\, C(o)]}_{\text{pragmatic value}}$$

**Implementation** (from `src/phase1/drive_system.py`, commit `d26f803`):

```python
def compute_efe(self, state, trajectory, action_history, candidate_action=None):
    pragmatic = dist / max_dist  # Manhattan distance / max possible
    if self.pragmatic_only:
        return pragmatic * self.pragmatic_weight
    # PEDA path
    epistemic = sum((1.0 - p.level2_confidence) * ratio * (0.9 ** i) ...)
    base_efe = epistemic + pragmatic * self.pragmatic_weight
    return self.drive_system.apply_to_efe(base_efe, ...)
```

### 11.2 Epistemic Error Measurement

| Method | Description | Status | Empirical result |
|--------|-------------|--------|-----------------|
| Token-space ensemble variance | Predict next state with 3 checkpoints; compute output disagreement | Active | Pre-fix: 0.0 (has-key dimension missing); Post-fix: 0.20 |
| Confidence-based epistemic | `epistemic_ratio = 1 - WM_confidence` | Active | Drove exploration in Phase 1.5 (PEDA ≠ Pragmatic) |
| Hidden-state (Light JEPA) | Cosine distance between checkpoint hidden states (last layer, mean-pooled) | Implemented | Pending empirical comparison |

### 11.3 Drive System

| Drive | Formula | Phase 2 default weight | Observed effect |
|-------|---------|----------------------|-----------------|
| Novelty | $$D_N = -\log P(s_{t+1} \mid s_t, a_t)$$ | 1.0 | Drives action diversity |
| Boredom | $$D_B = \frac{1}{\tau}\sum_{i=t-\tau}^{t} \mathbb{1}[s_i = s_t]$$ | 1.0 | Prevents sequence loops |
| Competence | $$D_C = \frac{\text{success steps}}{\text{total steps}}$$ | 1.0 | Stabilizes after first success |
| Growth | $$D_G = \lvert\text{KG}_t\rvert - \lvert\text{KG}_{t-1}\rvert$$ | 1.0 | Encourages new state exploration |

### 11.4 Epistemic/Pragmatic Ratio Monitoring

From WATCHDOG C8:

> "If pragmatic dominates consistently (>80% of EFE), consider: (a) reducing pragmatic_weight, (b) increasing environment uncertainty, (c) adding minimum epistemic floor."

Current status (Phase 1 partial train pilot): Ratio not explicitly measured. Must be logged in Phase 3 confirmatory experiments.

---

## 12. World Model: Pattern Matcher vs Reasoner Analysis

### 12.1 The Distinction

| Aspect | Pattern Matcher | Reasoner |
|--------|----------------|----------|
| Mechanism | Memorizes (state, action) → next_state pairs | Learns causal rules (e.g., "rm file → file deleted") |
| OOD generalization | Low (unchanged on unseen state combos) | High (applies rules to novel states) |
| Train data needed | Proportional to state-action space | Proportional to rule complexity |
| Epistemic signal | Binary (known/unknown) | Graded (high/low confidence in rules) |
| 0.5B behavior | G1=1.0 on seen; G1_test=0.87 on unseen | N/A: model too small for abstract reasoning |

### 12.2 Evidence for Pattern Matching

From `PHASE1_PARTIAL_EVALUATION.md`:

> "g1_test_set = 0.8684 means WM has 86.84% accuracy on out-of-distribution state-action pairs. This could be 0.5B model's memory/pattern matching ability, not true 'understanding'."

| Evidence | Source | Weight |
|----------|--------|--------|
| 25% train fraction → G1=0.87 (Grid World) | Phase 1 partial train | Strong |
| 10% train + 3 epochs → G1=1.0 (Grid World) | Phase 1 repeated attempt | Strong |
| `take key` predicted exit=1 (wrong) — all 3 checkpoints identical | Phase 1.5 eval | Strong — systematic error, not uncertainty |
| Grid World G1 on training set = 1.000 (perfect memorization) | Phase 1 eval | Strong |

### 12.3 Evidence Against Full Reasoning

| Evidence | Source | Weight |
|----------|--------|--------|
| 0.5B model cannot learn `go north` exit code (e3: 2, e4: 1, correct: 0) | Phase 1.5 Iteration 2 | Moderate — may be data quantity |
| 114 samples insufficient for 0.5B to learn 2-room dynamics | Phase 1.5 eval | Strong |
| Data augmentation in simple environment adds 0 new unique pairs after 6000 attempts | Phase 1.5 Iteration 2 | Strong — state space saturation |

### 12.4 Implications for Phase 3+

| Implication | Action |
|-------------|--------|
| If WM is pure pattern matcher, epistemic signal only appears at state-action boundary | Test: compare epistemic ratio for "known" vs "novel" state-action pairs |
| Larger WM (1.5B+) + more data may transition to reasoning | Deferred until Phase 5 (hardware permitting) |
| Functional equivalence: if pattern matcher produces same behavior as reasoner, the distinction is academic for PEDA | Accept functional equivalence for now; revisit in Phase 6 (generalization) |
| Hidden-state epistemic may detect "partial pattern match" where token-space sees full confidence | Light JEPA experiment is critical |

### 12.5 Experimental Protocol to Distinguish

```
Train WM on 50% of state-action space.
For each test pair (s,a):
  - Token-space: does WM predict correct next_state? (G1 metric)
  - Hidden-state: is ensemble variance high? (epistemic)
  
Expectation for pattern matcher:
  - Known pairs: G1 ≈ 1.0, epistemic ≈ 0
  - Novel pairs (partial match): G1 ≈ 0.5-0.8, epistemic > 0
  - Novel pairs (no match): G1 ≈ random, epistemic high

Expectation for reasoner:
  - All pairs: G1 > 0.9 (if rules are learned)
  - All pairs: epistemic ≈ 0 (no uncertainty about the rule)
```

---

## 13. Data Quality Methodology

### 13.1 Five Strategies — Rationale

| # | Strategy | Origin | Experimental Validation |
|---|----------|--------|------------------------|
| 1 | **Microtask design** (5-10 step episodes, not 20+) | GLM-5.2 Q7; WATCHDOG C7 | Used in Phase 2 collection: reduces task complexity, enables FHT/SCR/Dead-loop metrics |
| 2 | **Multi-strategy mixed sampling** (random + peda + heuristic) | Phase 1.5 lesson: single-strategy saturates | 200 walks × 30 steps PURE RANDOM produced only 114 unique (Phase 1.5). Mixed strategy covers broader distribution |
| 3 | **Built-in dedup with early stopping** | Phase 1.5: 6000 attempts → 114 unique → 0 net gain after saturation | Implemented: `max_unique_samples = 2000`, early stop if no new samples in 100 steps |
| 4 | **Structured JSON state** (not free text) | GLM-5.2 Q3; `SandboxState.to_json()` | Compares against free-text: Phase 1.5 used free text → token-space noise; JSON reduces grammar/whitespace variance by ~40% (estimated) |
| 5 | **Epistemic-targeted sampling** — bias collection toward state-action pairs where current WM has highest uncertainty | Phase 3 extension | Not yet validated; planned for Phase 4 self-training loop |

### 13.2 Strategy 1 Validation: Microtask vs Long Episode

| Metric | Long episode (20+ steps) | Microtask (5-10 steps) | Advantage |
|--------|-------------------------|------------------------|-----------|
| Completion rate | Phase 1.5: ~40% | Phase 2 plan: >70% expected | Microtask |
| Task-specific signal | Diluted by off-task steps | Concentrated | Microtask |
| Data collection speed | ~2 unique/step | ~4-6 unique/step (estimated) | Microtask |
| Dead-loop detection | Late (step 10+) | Early (step 3-5) | Microtask |

### 13.3 Strategy 2 & 3 Validation: Mixed Sampling + Dedup

Phase 1.5 data from `PHASE1_5_ITERATION2_EVALUATION.md`:

```
Method: 200 random walks × 30 steps = 6000 samples attempted
Result: 114 unique samples (1.9% dedup rate)
Conclusion: In small state spaces, random walk saturates almost immediately.
```

For Phase 2 (busybox, larger state space):

```
Estimated: 1000 steps of mixed strategy → 800-900 unique samples (80-90% dedup)
Active dedup threshold: stop if <5 new samples per 50 steps
```

### 13.4 Strategy 4 Validation: JSON vs Free Text

From `PROMPT_PHASE2_START.md`:

> "GLM-5.2 identified token-space prediction as forcing LLM to model grammar, whitespace, formatting — all irrelevant to environment dynamics."

| Dimension | Free text state | JSON structured state |
|-----------|-----------------|----------------------|
| State representation | "You are in room north. You see: a door, a key on the table." | `{"room": "north", "inventory": [], "objects": [{"name": "key", "location": "table"}]}` |
| Token count per state | 15-25 tokens | 8-15 tokens |
| Aleatoric noise (formatting) | High (synonyms, whitespace) | Low (deterministic serialization) |
| WM prediction ambiguity | "door" vs "a door" vs "the door" | `"door": true` — unambiguous |
| Empirical epistemic improvement | Baseline | Estimated +15-25% cleaner signal |

### 13.5 Data Pipeline

```python
# Phase 2 data pipeline (source: PROMPT_PHASE2_START.md)
CONFIG = {
    "max_steps": 1000,
    "max_unique_samples": 2000,
    "strategy": "mixed",          # "random" | "peda" | "mixed"
    "mixed_ratio": 0.5,          # PEDA selection ratio
    "whitelist": ["ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail", "grep"],
    "output_dir": "data/phase2",
}
```

### 13.6 Best Practices Summary

1. **Start small**: 100-step pilot before 1000-step production collection
2. **Deduplicate aggressively**: Print unique count every 100 steps
3. **Check coverage**: Track how many distinct (cwd, command) pairs collected
4. **Stop on saturation**: If unique count does not increase after 100 consecutive steps, increase action diversity
5. **Log epistemic ratio**: Monitor G1 on withheld set every checkpoint

---

## 14. WATCHDOG Rule Updates

### 14.1 New Blocker Rules (v2.0)

#### C20: Subagent output accepted without explicit verification

**Trigger**: A subagent completes a task and reports "done" without producing the specific artifact (file content, experimental result, metric value) that was requested. Or: the orchestrator accepts a subagent claim without independent cross-check.

**Why**: PEDA's subagent orchestration model depends on trust-but-verify. A subagent that returns a plausible-sounding but fabricated number is indistinguishable from a correct one without verification. Phase 1's G1=1.0 was accepted as infrastructure validation but masked the core hypothesis failure.

**Correct behavior**:
- Every subagent deliverable must include at least one verifiable artifact (file path, data snapshot, test result)
- Orchestrator must independently verify numeric claims before incorporating them into decisions
- If artifact verification fails → reject and regenerate

#### C21: Orchestrator bypassing subagent for implementation work

**Trigger**: The orchestrator agent writes code, runs experiments, or performs multi-line edits directly instead of delegating to a subagent. Exception: single-line edits (<5 lines) or direct test/debug commands.

**Why**: The v2.0 orchestration model makes orchestrator the planner, not the executor. Direct execution by the orchestrator undermines the parallelism and specialization benefits of the subagent model. Additionally, the orchestrator's context is better spent on coordination and cross-cutting analysis.

**Correct behavior**:
- Orchestrator maps the work → names slices → delegates via `task` → reviews results
- Orchestrator writes only: contracts (via `local://`), coordination messages, final synthesis
- Implementation, investigation, and detailed analysis are subagent responsibilities

#### C22: Phase dependency inversion

**Trigger**: Work begins on Phase N+1 before Phase N's go/no-go criteria are fully evaluated. Exceptions: preparatory infrastructure (e.g., Docker sandbox can be built while Phase 1 runs, if it doesn't consume evaluation resources).

**Why**: Phase advancement without hypothesis validation (B1) is PEDA's single biggest risk, inherited from Folunar_. The v2.0 phase structure has explicit gates; skipping them recreates the "declare done, move on" pattern that made Phase 1 hollow.

**Correct behavior**:
- Phase transition requires explicit sign-off on Phase N's completion criteria
- Preparatory work on Phase N+1 is allowed only if: (a) it does not consume Phase N evaluation resources, and (b) the Phase N+1 hypothesis does not depend on Phase N outcome
- Violation is treated as a blocker: revert to Phase N evaluation

### 14.2 Rule Status Summary (v2.0)

| Rule | v1.x | v2.0 | Change |
|------|------|------|--------|
| B1-B9 | Active | Active | Unchanged |
| C1-C14 | Active | Active | Unchanged |
| C20 | — | **New** | Subagent verification |
| C21 | — | **New** | Orchestration discipline |
| C22 | — | **New** | Phase dependency |
| N1-N3 | Active | Active | Unchanged |

---

## 15. Orchestration Rules

### 15.1 Model Roles

| Role | Model type | Responsibility |
|------|-----------|----------------|
| **Orchestrator** (main) | Full reasoning (deepseek/deepseek-v4-flash) | Scopes the work; decomposes into parallel tasks; writes contracts (`local://.md`); delegates via `task`; reviews subagent results; synthesizes final output |
| **Subagent** (task/specialist) | Same or faster model (scout for read-only) | Executes assigned slice; produces verifiable artifact; reports back to orchestrator |
| **Scout** (read-only) | Fast model | Codebase research, pattern search, file investigation |

### 15.2 Decomposition Rules

1. **Scope before spawn**: Orchestrator reads the request, maps the work, names independent slices before any delegation
2. **Width = real independence**: Fan out exactly as wide as work genuinely decomposes
3. **Prerequisites inline**: If Step A is needed by all downstream tasks, orchestrator does it before fanout
4. **Contracts shared via `local://`**: Interface schemas, file paths, data formats are written as `.md` files that subagents read
5. **No serialization without necessity**: Parallel batch by default; serialize only when Step B strictly requires Step A's output

### 15.3 Verification Rules

1. Every subagent deliverable includes a verifiable artifact (file, metric, test output)
2. The orchestrator independently verifies numeric claims before making go/no-go decisions
3. If orchestrator detects a fabrication or inconsistency → reject and re-delegate with explicit correction
4. Results from multiple subagents covering the same system must be cross-checked for consistency

### 15.4 Handoff Protocol

```
Orchestrator → local://task_spec.md (contract + requirements)
    ↓
Subagent reads local://task_spec.md
    ↓
Subagent executes (reads source, runs experiments, writes output)
    ↓
Subagent writes results to local://task_results.md
    ↓
Subagent signals completion via hub
    ↓
Orchestrator reads local://task_results.md → reviews → accepts/rejects
```

### 15.5 Concurrency Limits

- Maximum 32 concurrent subagents per session
- Batches larger than 32 are automatically queued; design to stay under the cap
- Research-only subagents (scouts) can share a single batch with implementation subagents

---

## 16. Risk Register & Mitigation

### 16.1 Active Risks (Phase 3)

| Risk | Probability | Impact | Mitigation | Trigger |
|------|-----------|--------|------------|---------|
| Pilot signal (2 steps vs 20 steps) is random luck | Medium | High: Core hypothesis false | Run N≥10 confirmatory; α=0.05 | If p > 0.05 in goal_unknown condition |
| g1_test_set=0.8684 too high; WM still too good | Medium | Medium: Epistemic signal too weak | Reduce train_fraction to 0.25; test | If N≥10 shows no significant difference |
| CPU inference limits prevent N≥10 completion | High | Medium: Evaluation takes >1 day | Split into overnight runs; use `--start-episode` flag | If 10 episodes exceed 8h wall time |
| Pragmatic-only also succeeds (goal_unknown) | Low | High: PEDA not needed | Check train split; pragmatic may get lucky in small grid | If pragmatic success rate > 40% in goal_unknown |

### 16.2 Resolved Risks

| Risk | Status | How resolved |
|------|--------|-------------|
| Environment-model mismatch (Grid World) | Resolved via negative result | Model too powerful for environment; accepted as valid finding |
| decompose_error measurement bug | Resolved via fix | Added has-key dimension → epistemic 0.0 → 0.20 |
| Data augmentation ineffective | Resolved via strategy change | Move to busybox (larger state space); microtask design |
| Phase advancement without hypothesis validation | WATCHDOG B1 + C22 | Active enforcement |

### 16.3 Future Risks (Phase 4+)

| Risk | Phase | Impact | Pre-mitigation |
|------|-------|--------|----------------|
| Self-training loop degrades WM via online update | 4 | Catastrophic forgetting | Intermittent batch update; prioritized replay |
| Write-enabled environment increases exploit surface | 5 | Sandbox escape | Read-only container; `--tmpfs` for writable scratch |
| Larger model required but GPU unavailable | 5-6 | Cannot scale | INT4 quantization; knowledge distillation from API |
| Preference distribution C(o) doesn't converge | 6 | No pragmatic gradient | Fall back to uniform preference; treat as finding |

### 16.4 Go/No-Go Decision Framework (from WATCHDOG "3 Questions")

For every non-trivial experiment in Phase 3+:

1. **What specific hypothesis does this experiment falsify?**
2. **If it fails, what is the most likely cause, and can I confirm that within 2 hours?**
3. **Is this the first attempt?** (If N≥2, require written justification)

---

## A. Appendix: Experimental Data Tables

### A.1 Phase 1 Gate Results

| Gate | Metric | Threshold | Achieved | Pass? |
|------|--------|-----------|----------|-------|
| G1 | WM next-state accuracy | > 0.90 | 1.0000 | YES (memorization) |
| G2 | Steps ratio (PEDA/random) | < 0.50 | 0.1211 | YES (any greedy strategy would) |
| G3 | Revisit rate | < 0.20 | 0.0000 | YES (trivial for 5×5) |

### A.2 Phase 1 Partial Training Pilot

| Condition | Agent | N | Success Rate | Mean Steps | Revisit Rate | g1_test |
|-----------|-------|---|-------------|------------|--------------|---------|
| goal_known | PEDA | 1 | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_known | pragmatic_only | 1 | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_unknown | PEDA | 1 | 1.000 | 2.0 | 0.000 | 0.500 |
| goal_unknown | pragmatic_only | 1 | 0.000 | 20.0 | 0.905 | 1.000 |

### A.3 Phase 1.5 Iteration Results

| Metric | e3 (Iteration 1) | e4 (Iteration 2) | Change |
|--------|------------------|------------------|--------|
| PEDA epistemic ratio | 0.0000 | 0.2000 | +0.20 (bug fix) |
| Pragmatic epistemic ratio | 0.0000 | 0.2222 | +0.22 (bug fix) |
| Unique training samples | 113 | 114 | +1 (saturation) |
| PEDA `take key` behavior | step 3 | step 1 | Earlier exploration |
| Pragmatic behavior | look × 20 | look × 10 | No exploration |

### A.4 WM Accuracy by Prediction Level

| Level | Target | Best Observed | Notes |
|-------|--------|---------------|-------|
| L1 (exit code) | ≥90% | ~95% (seen), ~80% (unseen) | Systematic error on `take key` (all checkpoints predict exit=1, correct=0) |
| L2 (filesystem delta) | ≥70% F1 | Not measured | Pending Phase 2 implementation |
| L3 (output summary) | ≥50% semantic | Not measured | Pending structured state evaluation |

### A.5 Epistemic Measurement Methods Comparison

| Method | Phase tested | Result | Computation cost |
|--------|-------------|--------|-----------------|
| Token-space variance (pre-fix) | 1.5 | 0.0 (false negative) | Low |
| Token-space variance (post-fix) | 1.5 | 0.20 | Low |
| Confidence-based (1 - WM_confidence) | 1.5 | ~0.30 (estimated) | None (free signal) |
| Hidden-state cosine (Light JEPA) | 2 | Pending | +1 forward pass per checkpoint |

---

## B. Appendix: File Index

| Path | Purpose | Last updated |
|------|---------|-------------|
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/PEDA_ENGINEERING_PLAN_v2.md` | This document | 2026-07-26 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/RESEARCH_CHARTER.md` | Research scope, negative-result acceptance | 2026-07-02 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/WATCHDOG.md` | Guardian rules (B1-B9, C1-C14, C20-C22, N1-N3) | 2026-07-26 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/peda_reflection_v11.md` | v1.0 → v1.1 improvement log | 2026-07-02 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/PHASE1_EVALUATION.md` | Phase 1 evaluation (flaw analysis) | 2026-07-03 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/PHASE1_PARTIAL_EVALUATION.md` | Partial train pilot evaluation | 2026-07-03 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/PHASE1_5_COMPLETE_EVALUATION.md` | Phase 1.5 iteration 1 eval | 2026-07-06 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/PHASE1_5_ITERATION2_EVALUATION.md` | Phase 1.5 iteration 2 eval | 2026-07-06 |
| `Kimi_Agent_Folunar_评估与优化/PEDA_FINAL/PROMPT_PHASE2_START.md` | Phase 2 infrastructure prompt | 2026-07-06 |
| `Kimi_Agent_Folunar_评估与优化/peda_v11_sec03.md` | Architecture design (v1.1) | 2026-07-02 |
| `Kimi_Agent_Folunar_评估与优化/peda_v11_sec04.md` | Implementation plan (v1.1) | 2026-07-02 |
| `Kimi_Agent_Folunar_评估与优化/peda_v11_sec05.md` | Agent internals guide (v1.1) | 2026-07-02 |
| `Kimi_Agent_Folunar_评估与优化/peda_v11_sec06.md` | Roadmap & resources (v1.1) | 2026-07-02 |

---

## C. Appendix: WATCHDOG C20-C22 Full Text (for agent reference)

### C20: Subagent output accepted without explicit verification

**Trigger**: A subagent completes a task and reports "done" without producing the specific artifact that was requested. The orchestrator accepts a subagent claim without independent cross-check.

**Correct behavior**:
- Every subagent deliverable must include at least one verifiable artifact (file path, data snapshot, test result)
- Orchestrator must independently verify numeric claims before incorporating them into decisions
- If artifact verification fails → reject and regenerate

### C21: Orchestrator bypassing subagent for implementation work

**Trigger**: The orchestrator writes code, runs experiments, or performs multi-line edits directly instead of delegating.

**Correct behavior**:
- Orchestrator maps the work → names slices → delegates via `task` → reviews results
- Orchestrator writes only: contracts (`local://`), coordination messages, final synthesis
- Implementation and investigation are subagent responsibilities

### C22: Phase dependency inversion

**Trigger**: Work begins on Phase N+1 before Phase N's go/no-go criteria are fully evaluated.

**Correct behavior**:
- Phase transition requires explicit sign-off on Phase N's completion criteria
- Preparatory work on Phase N+1 allowed only if it does not consume Phase N evaluation resources
- Violation: revert to Phase N evaluation

---

