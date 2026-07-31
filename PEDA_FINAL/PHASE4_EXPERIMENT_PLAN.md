# Phase 4 Experiment Plan: Autonomous Improvement & Multi-Task Generalization

**Date:** 2026-07-28
**Status:** Planned (pre-execution)
**Repository:** Folunar\_/PEDA-Teacher
**Hardware Target:** NVIDIA T4 (g4dn.xlarge, us-east-1)
**Adapter Base:** `checkpoints/phase2/sandbox_adapter_v2_full`

---

## 1. Motivation: Phase 3 Recap

Phase 3 (N=20) confirmed the core epistemic validation hypothesis: PEDA's World Model generalizes to novel (unknown) CWDs, yielding statistically significant and practically large improvements over the Pragmatic baseline.

### Key Results

| Metric | PEDA (unknown) | Pragmatic (unknown) | Test | p | Effect size |
|--------|----------------|---------------------|------|---|-------------|
| Mean steps | 7.20 ($\sigma=3.91$) | 10.00 ($\sigma=0.00$) | MW (one-tailed) | 0.0021 | $d=-1.01$ |
| Success rate | 100% | 100% | — | — | — |
| Dead-loop rate | 0.00 | 0.80 | — | — | — |
| Crossover interaction | Advantage flips known → unknown | — | MW | 0.0001 | — |

The effect is concentrated in `/sandbox/projects` (PEDA: 2.00 steps vs Pragmatic: 10.00, $p=0.0004$), where the World Model successfully generalized from training CWDs. The negative control confirms this is not a trivial advantage: PEDA in *known* CWDs is significantly *worse* than Pragmatic (10.00 vs 6.85, $d=1.01$, $p=0.0043$).

### Open Questions After Phase 3

1. **Can PEDA autonomously improve?** Phase 3 used a frozen adapter. If the Learning Module updates the World Model online, does performance on unknown CWDs improve further?
2. **Does the effect generalize beyond `read_hello`?** Phase 3 used a single task. Is the epistemic advantage task-specific or does it transfer to harder tasks?
3. **Is the effect WM-dependent?** Does PEDA only help when the World Model already has reasonable competence, or does it drive exploration in tasks where the WM starts from zero?

Phase 4 addresses all three.

---

## 2. Experiment A: Closed-Loop Self-Training

### 2.1 Research Question

Does intermittent self-training (online LoRA fine-tuning) amplify PEDA's epistemic advantage on unknown CWDs? Can the agent autonomously bootstrap from its own experience?

### 2.2 Design

Three conditions, four training blocks, N=10 episodes per block per condition. Total: $3 \times 4 \times 10 = 120$ episodes.

| Condition | Initial Adapter | Training Protocol |
|-----------|-----------------|-------------------|
| **PEDA+Train** | `sandbox_adapter_v2_full` | After each block: collect transitions → 1 epoch LoRA update → save new checkpoint |
| **PEDA+Freeze** | `sandbox_adapter_v2_full` | None — same frozen adapter for all 4 blocks |
| **Pragmatic** | N/A | None (no World Model) |

### 2.3 Protocol (per condition)

```
Block 1: N=10 episodes (all 6 CWDs, round-robin, unknown only) → record transitions
  [PEDA+Train only]: LearningModule.update(transitions) → save new adapter
Block 2: N=10 episodes with current adapter → record transitions
  [PEDA+Train only]: update → save
Block 3: N=10 episodes → record transitions → update (Train only)
Block 4: N=10 episodes → record transitions → update (Train only)
```

**CWD set for all blocks:** Unknown CWDs from Phase 3 — `/sandbox/logs`, `/sandbox/projects`, `/sandbox/tmp` (round-robin per block: 4/3/3).

### 2.4 Metrics

| Metric | Scope | Collection |
|--------|-------|------------|
| Mean steps (unknown CWDs) | Per block, per condition | Trendline across blocks 1–4 |
| Success rate | Per block | Proportion of episodes with `success=True` |
| Dead-loop rate | Per block | Proportion of episodes with `dead_loop=True` |
| WM held-out L1/L2/L3 accuracy | PEDA+Train only | Before Block 1 vs after Block 4 |
| Adapter loss curve | PEDA+Train only | Loss at each of the 4 training steps |
| Per-CWD steps breakdown | Per block, per condition | By-block × by-CWD matrix |

### 2.5 Primary Hypotheses

**H1 (trend):** PEDA+Train mean steps in unknown CWDs decrease across blocks 1–4.

- Test: Repeated-measures trend test (Jonckheere-Terpstra or linear contrast in Friedman ANOVA).
- Expected: monotonic decreasing trend.

**H2 (terminal):** PEDA+Train Block 4 mean steps < PEDA+Freeze Block 4 mean steps.

- Test: Mann-Whitney U, one-tailed (PEDA+Train < PEDA+Freeze).
- $\alpha = 0.05$.

**H3 (learning signal):** PEDA+Train adapter loss decreases at each training step.

- Test: Loss at step $t$ < loss at step $t-1$ (paired sign test across the 4 updates).
- $\alpha = 0.05$.

### 2.6 Secondary Analyses

| Analysis | Method | Purpose |
|----------|--------|---------|
| Block-by-block PEDA+Train vs PEDA+Freeze | Separate MW tests for each block | Identify when training first diverges from frozen |
| Per-CWD learning curves | Visual: steps vs block per CWD | Determine which CWDs benefit from self-training |
| WM accuracy vs performance | Correlation: $\Delta$WM accuracy vs $\Delta$steps | Test whether WM improvement drives behavioral improvement |
| PEDA+Freeze vs Pragmatic (replication) | MW on Block 1 data | Confirm Phase 3 result replicates |

### 2.7 Power Analysis

$N=10$ per condition per block, 4 blocks, 3 conditions = 120 episodes.

At $\sim$150 s/episode (Phase 3 timing): $120 \times 150 / 3600 = 5$ GPU-hours.

The within-block comparison (H2) at $N=10$ per group has 80% power to detect $d \geq 1.01$ (the Phase 3 effect size) at $\alpha=0.05$ one-tailed ($t$-test approximation). The trend test (H1) across 4 blocks is higher-power as it uses repeated measures.

---

## 3. Experiment B: Multi-Task Generalization

### 3.1 Research Questions

**Primary:** Does PEDA's epistemic advantage persist across tasks of varying difficulty, or is it specific to `read_hello`?

**Secondary:** Does PEDA advantage correlate with World Model competence? If PEDA only helps where the WM is already strong, the mechanism is WM-dependent rather than truly epistemic.

### 3.2 Tasks

| Task | Command | Difficulty | Current WM Competence | Rationale |
|------|---------|------------|-----------------------|-----------|
| `read_hello` | `cat hello.txt` | Easy | 100% success (Phase 3) | Replication + baseline for harder tasks |
| `count_lines` | `wc -l *.txt` | Medium | Unknown | Tests numeric aggregation — different skill from file reading |
| `find_secret` | `grep secret *.txt` | Hard | Unknown | Tests pattern matching across files — requires discrimination |
| `read_note` | `cat docs/note.txt` | Hard | 0% (WM fails on Phase 2 held-out eval) | Tests whether PEDA helps when WM has zero prior competence |

### 3.3 Design

$2$ baselines $\times$ $2$ CWD types $\times$ $4$ tasks = 16 conditions, N=5 each = 80 episodes.

| Factor | Levels |
|--------|--------|
| Baseline | PEDA, Pragmatic |
| CWD type | Known (`/sandbox`, `/sandbox/data`, `/sandbox/docs`), Unknown (`/sandbox/logs`, `/sandbox/projects`, `/sandbox/tmp`) |
| Task | `read_hello`, `count_lines`, `find_secret`, `read_note` |

**CWD×task counterbalancing:** Each condition sees the same 3 CWDs of its type in round-robin (2/2/1 per condition since N=5).

### 3.4 Primary Hypothesis (per-task)

For each task $t \in \{\text{read\_hello}, \text{count\_lines}, \text{find\_secret}, \text{read\_note}\}$:

$$H_{0,t}: \mu_{\text{PEDA, unknown}}^{(t)} \geq \mu_{\text{Pragmatic, unknown}}^{(t)}$$
$$H_{1,t}: \mu_{\text{PEDA, unknown}}^{(t)} < \mu_{\text{Pragmatic, unknown}}^{(t)}$$

Test: Mann-Whitney U, one-tailed (PEDA < Pragmatic). Bonferroni correction across 4 tasks:

$$\alpha_{\text{adjusted}} = \frac{0.05}{4} = 0.0125$$

### 3.5 Secondary Hypotheses

**H4 (task difficulty gradient):** PEDA advantage (Pragmatic steps $-$ PEDA steps) is larger for harder tasks.

- Test: Spearman rank correlation between task difficulty rank and PEDA advantage.
- $\alpha = 0.05$ (exploratory).

**H5 (WM competence correlation):** PEDA advantage is larger for tasks where WM held-out accuracy is higher.

- Test: Spearman $\rho$ between WM pre-experiment held-out L1/L2/L3 accuracy (from Phase 2) and PEDA advantage.
- $\alpha = 0.05$ (exploratory).

**H6 (within-task replication):** For `read_hello`, PEDA unknown < Pragmatic unknown.

- Test: MW one-tailed on `read_hello` subset.
- $\alpha = 0.05$. Expected to replicate Phase 3 at $p < 0.05$.

### 3.6 Metrics (Experiment B)

| Metric | Per condition | Per task |
|--------|---------------|----------|
| Mean steps | Yes | Yes |
| Success rate | Yes | Yes |
| Dead-loop rate | Yes | Yes |
| Per-CWD breakdown | Yes | Yes |
| Elapsed time | Yes | Yes |
| WM held-out accuracy (PEDA only) | Pre/post (aggregate) | Not applicable |

### 3.7 Power Analysis

$N=5$ per cell, 16 cells = 80 episodes, approximately 3.3 GPU-hours.

At $N=5$ per group, MW one-tailed has 80% power to detect $d \geq 1.65$ at $\alpha_{\text{unadjusted}} = 0.05$. With Bonferroni $\alpha = 0.0125$, power drops to $\sim$60% for the same effect. However, if the Phase 3 effect ($d = -1.01$) is representative, we will be underpowered for per-task tests — the analysis is qualitative and effect-size oriented, not powered for formal hypothesis testing on individual tasks. The composite analysis (across-task sign test) provides an alternative.

**Composite test:** Under the global null (PEDA never better), the probability of observing PEDA advantage in $k$ of 4 tasks is binomial with $p=0.5$. Observing PEDA advantage in all 4 tasks yields $p = 0.5^4 = 0.0625$ — marginally significant. Observing it in 3+ tasks yields $p = 0.3125$. This is a secondary analysis; the per-task Bonferroni tests are the primary.

---

## 4. Implementation Notes

### 4.1 Codebase References

| Component | Path | Role |
|-----------|------|------|
| Base experiment infrastructure | `scripts/phase3_sandbox_experiment.py` | Sandbox orchestration, episode loop, result logging, CWD/task parameterization |
| Learning Module | `src/phase2/run.py` (LearningModule class) | Transition collection, LoRA fine-tuning, checkpoint management |
| Adapter base | `checkpoints/phase2/sandbox_adapter_v2_full` | Initial LoRA weights for PEDA+Train and PEDA+Freeze |
| Phase 3 result parsing | `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` | Reference for statistical methodology |

### 4.2 New Scripts Required

| Script | Purpose | Based On |
|--------|---------|----------|
| `scripts/phase4_closed_loop.py` | Experiment A: 4-block loop with intermittent training | `scripts/phase3_sandbox_experiment.py` |

The `phase4_closed_loop.py` script extends the Phase 3 experiment infrastructure with:

- **Block management**: run N episodes → trigger LearningModule update → save checkpoint → advance block counter
- **Adapter chain**: each block's output adapter becomes the next block's input
- **Transition buffer**: collect (state, action, next_state, error_vector) tuples during episodes
- **Checkpoint registry**: save checkpoints as `checkpoints/phase4/block_{block_number}/adapter`

Experiment B reuses `scripts/phase3_sandbox_experiment.py` directly with the `--task` parameter.

### 4.3 Task Definitions

Each task requires a success condition and search pattern for the sandbox:

```python
TASK_REGISTRY = {
    "read_hello": {
        "success_condition": "cat hello.txt",
        "expected_output": "Hello, World!",
        "max_steps": 20,
    },
    "count_lines": {
        "success_condition": "wc -l *.txt",
        "expected_output_regex": r"\d+",
        "max_steps": 20,
    },
    "find_secret": {
        "success_condition": "grep secret *.txt",
        "expected_output_regex": r"secret",
        "max_steps": 20,
    },
    "read_note": {
        "success_condition": "cat docs/note.txt",
        "expected_output": "This is a note.",
        "max_steps": 20,
    },
}
```

### 4.4 Environment Setup

| Parameter | Value |
|-----------|-------|
| Sandbox image | `folunar-sandbox:latest` (as Phase 3) |
| CWDs (known) | `/sandbox`, `/sandbox/data`, `/sandbox/docs` |
| CWDs (unknown) | `/sandbox/logs`, `/sandbox/projects`, `/sandbox/tmp` |
| Max steps per episode | 20 |
| Timeout per episode | 300 s |

---

## 5. GPU Resource Estimate

| Experiment | Conditions | Episodes | Est. time/episode | Total GPU-hours |
|------------|-----------|----------|-------------------|-----------------|
| Phase 3 (reference) | 4 | 80 | $\sim$150 s | $\sim$3.3 |
| Experiment A | 3 $\times$ 4 blocks | 120 | $\sim$150 s | $\sim$5.0 |
| Experiment A extra (training time) | 4 updates $\times$ 1 epoch | — | $\sim$60 s/update | $\sim$0.07 |
| Experiment B | 16 cells | 80 | $\sim$150 s | $\sim$3.3 |
| **Total Phase 4** | — | **200** | — | **$\sim$8.5** |

**Instance type:** g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM), us-east-1.
**AMI:** Deep Learning AMI (Ubuntu 22.04).
**Spot instance cost:** $\sim$\$0.126/hr → $\sim$\$1.07 total.
**Storage:** 100 GB gp3 (sufficient for adapter checkpoints and logs).

**Failure budget:** We tolerate one complete re-run per experiment: $\sim$\$2.14 total with re-runs.

---

## 6. Statistical Protocol

### 6.1 Pre-registered Analyses

| Analysis | Test | $\alpha$ | Correction |
|----------|------|----------|------------|
| H1 (trend, Exp A) | Jonckheere-Terpstra trend test | 0.05 | None (single test) |
| H2 (terminal, Exp A) | MW one-tailed | 0.05 | None (single test) |
| H3 (loss decay, Exp A) | Paired sign test | 0.05 | None (single test) |
| Primary per-task (Exp B) | MW one-tailed | 0.0125 | Bonferroni ($\times 4$) |
| H6 (read_hello replication, Exp B) | MW one-tailed | 0.05 | None (replication, not discovery) |

### 6.2 Exploratory Analyses

- Spearman $\rho$: task difficulty vs PEDA advantage
- Spearman $\rho$: WM held-out accuracy vs PEDA advantage
- Within-CWD breakdown: which CWDs improve with self-training
- Block-by-block divergence point (Exp A)

### 6.3 Reporting Standards

All hypothesis tests report:

- Test statistic ($U$, $J$, $S$)
- Exact $p$-value (or Monte Carlo approximation for ties)
- Effect size: Cohen's $d$ (with 95% CI) for pairwise, Kendall's $W$ for trend
- Raw data: all episode-level results saved as JSONL in `results/phase4_*/`

---

## 7. Success Criteria

| Criterion | Threshold | Test |
|-----------|-----------|------|
| **Primary: Self-training improves** | H1 significant at $p < 0.05$ | Jonckheere-Terpstra on PEDA+Train blocks |
| **Primary: Self-training beats frozen** | H2 significant at $p < 0.05$ | MW: PEDA+Train Block 4 < PEDA+Freeze Block 4 |
| **Primary: Multi-task generalization** | $\geq$ 2 of 4 tasks show PEDA advantage | Per-task Bonferroni MW tests |
| **Secondary: Learning signal detected** | H3 significant at $p < 0.05$ | Signed test on loss deltas |
| **Secondary: `read_hello` replication** | H6 significant at $p < 0.05$ | MW on Exp B `read_hello` subset |
| **Qualitative: WM accuracy gain** | PEDA+Train WM held-out accuracy improves from pre to post | Before vs after L1/L2/L3 comparison |
| **Minimum bar: No regression** | PEDA (frozen) in unknown CWDs not worse than Phase 3 (7.20 steps) | Descriptive comparison |

### 7.1 Staged Interpretation

| Outcome | Interpretation |
|---------|---------------|
| H1 ✓, H2 ✓, $\geq$3 tasks ✓ | **Strong confirmation:** self-training amplifies epistemic advantage, generalizes across tasks |
| H1 ✓, H2 ✓, 1–2 tasks ✓ | **Moderate confirmation:** self-training works, generalization partial |
| H1 ✓, H2 ✗ | **Ambiguous:** trend detected but terminal comparison underpowered — may need N=15 |
| H1 ✗, H2 ✓ | **Contradictory:** check Block 1 baseline drift, ceiling effects |
| H1 ✗, H2 ✗, replication holds | **Negative result for self-training:** epistemic advantage exists but is not amplifiable by LoRA fine-tuning |
| Replication fails (PEDA not better than Pragmatic on unknown) | **Showstopper:** investigate environment/task/seed differences from Phase 3 |

---

## 8. Execution Plan

### Phase 4a: Experiment A (Closed-Loop Self-Training)

1. Launch GPU instance (g4dn.xlarge, DL AMI)
2. Sync repository and adapter checkpoints
3. Create `scripts/phase4_closed_loop.py`
4. Run Experiment A in tmux session `phase4a`
5. Monitor: check Block 1 completes before leaving unattended
6. Estimated runtime: 5 GPU-hours ($\sim$6–8 hr wall clock with overhead)

### Phase 4b: Experiment B (Multi-Task Generalization)

1. After Experiment A completes (or in parallel using second session)
2. Prepare task configuration files
3. Run using `scripts/phase3_sandbox_experiment.py --task {task}` in tmux session `phase4b`
4. 16 conditions × 5 episodes = 80 episodes total
5. Estimated runtime: 3.3 GPU-hours ($\sim$4–5 hr wall clock)

### Analysis Phase

1. Parse JSONL results into analysis tables
2. Run statistical tests (Python `scipy.stats`)
3. Generate ANALYSIS_REPORT.md with full tables and effect sizes
4. Update research manuscript with Phase 4 findings

---

## 9. References

1. Phase 3 Analysis Report: `results/phase3_sandbox_n20/ANALYSIS_REPORT.md`
2. Base Experiment Script: `scripts/phase3_sandbox_experiment.py`
3. Learning Module: `src/phase2/run.py`
4. Initial Adapter: `checkpoints/phase2/sandbox_adapter_v2_full`
5. Research Charter: `PEDA_FINAL/RESEARCH_CHARTER.md`
6. Research Manuscript: `PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md`
