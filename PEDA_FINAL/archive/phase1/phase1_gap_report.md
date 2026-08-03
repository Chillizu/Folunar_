# Phase 1 Gap Report

> **Date**: 2026-07-20
> **Purpose**: Audit Phase 1 archive against v1.1 go/no-go criteria as part of top-down phase re-organization.

---

## 1. What Phase 1 Formally Achieved

- **Infrastructure validated**: LLM loading, LoRA fine-tuning, checkpoint saving, EFE computation, action selection, evaluation loop all work.
- **Formal metrics reported** (in-distribution, same 5×5 grid):
  - G1 (World Model L1/L2 accuracy): 1.0000
  - G2 (PEDA steps-to-goal vs random): 0.4337 (< 0.50 threshold)
  - G3 (revisit rate): 0.0000 (< 0.20 threshold)
- **Held-out obstacle test completed**: 3 obstacle layouts × 5 episodes each, PEDA vs pragmatic-only.
- **Artifacts archived**: `results/phase1_eval.json`, `results/phase1_grid_search.json`, `results/phase1_heldout_summary.json`, `config/phase1_default_drives.json`, etc.

---

## 2. Go/No-Go Criteria Actually Met

| Criterion | Target | Status | Evidence |
|-----------|--------|--------|----------|
| G1 World Model accuracy | > 0.90 | **Met on training distribution** | `results/phase1_eval.json`: g1=1.0 with adapter `partial_adapter_real_25_e3` |
| G2 Steps vs random | < 0.50 | **Met on training distribution** | `results/phase1_eval.json`: ratio=0.4337 |
| G3 Revisit rate | < 0.20 | **Met** | `results/phase1_eval.json`: revisit=0.0 |

**Caveat**: All G1/G2/G3 "passes" are on the **same 5×5 grid used for training**. This is memorization, not generalization or mechanism validation.

---

## 3. Criteria Not Met / Not Validated

### 3.1 Core Hypothesis: Prediction-Error-Driven Exploration

**Not validated.** The held-out obstacle test showed PEDA and `pragmatic_only` behaved identically — the epistemic/drive component produced no measurable behavioral difference. The World Model was too certain because the 5×5 grid is too simple for a 0.5B parameter model.

Evidence:
- `PEDA_WORKING_LOG.md` [ARCHIVE] 2026-07-20: "真实 LLM + adapter 在 held-out 障碍物 grid 上仍保持高准确率（认知误差≈0），PEDA 与 pragmatic-only 无显著差异。"
- `PEDA_FINAL/phase1_archive_summary.md`: "核心机制未验证（环境太简单）。"

### 3.2 Out-of-Distribution Generalization

**Not validated.** G1/G2/G3 were measured on the training distribution. Held-out obstacle grids tested layout transfer but not true OOD dynamics; WM accuracy remained near-perfect and PEDA did not rely on epistemic signal.

This violates WATCHDOG C9 (training and evaluation on same distribution).

### 3.3 Confirmatory Sample Size

**Not met.** The final real-LLM evaluation used 10 episodes. While this meets the minimum ≥10 threshold per condition (WATCHDOG B5), the held-out obstacle comparison used only 5 episodes per layout per agent — borderline for statistical inference. More importantly, the effect size was zero, so larger N would not have changed the conclusion.

### 3.4 Behavioral Diversity / Emergent Exploration

**Not measured / not met.** No entropy, coverage, or FactGraph growth metrics were reported for Phase 1. The agent's behavior was near-deterministic greedy navigation.

---

## 4. Impact on Phase 1.5 and Phase 2

- **Phase 1 advanced to 1.5 and then 2 based on infrastructure validation, not hypothesis validation** (WATCHDOG B1 risk).
- The failure to produce epistemic signal in Grid World justified moving to more complex environments, but it did **not** prove the core mechanism works elsewhere.
- Phase 1.5 (custom text env) also failed to validate the hypothesis; see `PEDA_FINAL/phase1_5_deviation_report.md`.
- Phase 2 therefore inherits a **double debt**: neither Grid World nor TextWorld validated that prediction-error/EFE drives useful exploration. Phase 2 must test this directly via multi-baseline comparison, held-out evaluation, and epistemic/pragmatic term logging.

---

## Summary

Phase 1 **formal targets** (G1/G2/G3 on training distribution) were met and the infrastructure is sound. Phase 1 **did not validate the core hypothesis** that prediction-error-driven exploration produces behavior distinguishable from greedy distance minimization. This gap was correctly identified by the project, but the phase was still archived and advancement occurred without a validated mechanism.

**Recommendation**: Treat Phase 1 as an **infrastructure milestone**, not a hypothesis-validation milestone. All claims about "PEDA works" must come from Phase 2 with explicit multi-baseline evidence.
