# Phase 1.5 Deviation Report — v1.1 Plan vs Actual

> **Purpose**: Document divergence from v1.1 architecture plan as required by WATCHDOG C19.
> **Date**: 2026-07-20
> **Status**: Read-only investigation — no existing files modified.

---

## 1. What Changed

| Dimension | v1.1 Plan | Actual | Delta |
|-----------|-----------|--------|-------|
| **Environment** | TextWorld (Microsoft Research framework) | Custom `TextRoomEnv` (2 rooms: study &harr; hallway &rarr; chest) | Full substitution |
| **State space** | Arbitrary text descriptions from game engine | Hand-crafted `Perception.render_text()` strings, 6-10 lines | Drastically simplified |
| **Task structure** | Multi-step goal-oriented (e.g. "Cook the potato and eat it", 5+ steps) | `read_note`: cd docs &rarr; cat note.txt (2-3 steps) | Much shorter horizon |
| **Complexity tiers** | 3 progressive tiers (1 room, 5 rooms, constrained) | Single static environment | No progression |
| **Threshold target** | G4: ROUGE-L >60%, facts >60%; G5: 3-step task >30%; G6: FactGraph >70%; G7: entropy >0.5 | None met | | Full gap |
| **Data volume** | Implicitly large (TextWorld generates unlimited episodes) | 113 unique samples (50 walks &times; 20 steps, deduped); after augmentation: 114 | 2 orders of magnitude below what a 0.5B model needs |
| **Timeline** | 3-4 weeks | ~1 week | Compressed |
| **Validation** | G4/G5/G6/G7 formal gates | decompose_error bug fix + 1-episode behavioral check | No threshold validation |

### What Phase 1.5 Actually Produced

- Custom lightweight text environment (2 rooms, key-chest inventory logic) &mdash; **infrastructure value**
- 113 unique (state, action, next_state) training samples
- Qwen2.5-0.5B + LoRA (3 checkpoints, 3 epochs) &rarr; final loss ~0.02
- decompose_error bug identified and fixed (epistemic error 0.0 &rarr; 0.20)
- Behavioral signal: PEDA &ne; Pragmatic, reproducible across 2 iterations
- **No v1.1 threshold was met or formally evaluated**

---

## 2. Why It Changed

### Primary Reason: Python 3.14 Incompatibility with TextWorld

TextWorld (Microsoft Research, `pip install textworld`) depends on `backports.functools_lru_cache` and several C extensions that do not support Python 3.14. The project workstation runs Arch Linux with Python 3.14 as the system interpreter. Attempting to install or run TextWorld produced import errors that could not be resolved within a reasonable time budget.

**Why this was not foreseen**: v1.1 was written for a generic Python 3.10+ assumption. The actual workstation environment (Arch Linux rolling release + Python 3.14) was not available during planning.

### Secondary Reason: Time Pressure for Infrastructure Validation

Phase 1 had just completed its partial-training pilot (Phase 1 PHASE1_PARTIAL_EVALUATION.md, N=1 directional), and the project needed a go/no-go decision on whether PEDA could operate outside Grid World. Building a custom 2-room text environment took ~4 hours vs an estimated 2-3 days to resolve TextWorld dependency issues. The trade-off was accepted consciously: sacrifice environmental fidelity for speed, validate the pipeline, then pivot.

### Tertiary Reason: CPU-Only Hardware Constraint

The workstation has no CUDA-capable GPU. 0.5B Qwen2.5 runs at ~4 tokens/sec via llama.cpp. A full TextWorld evaluation (3 tiers &times; 10 episodes &times; multiple agents) would have taken 2-3 weeks of continuous runtime. The custom env reduced per-episode time to ~30 seconds, making iterative development feasible.

---

## 3. Risks Introduced

| Risk | Level | Description |
|------|-------|-------------|
| **Core hypothesis unvalidated** | HIGH | The central claim (prediction-error-driven exploration produces useful behavior) was never tested against the v1.1 thresholds. Phase 1.5 proved "PEDA &ne; Pragmatic" but not "PEDA > Pragmatic in task completion". |
| **Data scaling unknown** | HIGH | We still do not know how much training data is needed for a 0.5B LLM to learn text-state transitions. 113-114 samples was clearly insufficient, but the saturation point is undefined. |
| **Environment complexity mismatch** | MEDIUM | The 2-room env's state space (2 rooms &times; ~5 actions &times; inventory states) is too small for meaningful epistemic uncertainty. The decompose_error fix raised mean_epistemic_error to 0.20, but this was driven by inventory-state confusion, not genuine environmental complexity. |
| **Horizon generalization** | MEDIUM | Phase 1.5 only tested horizon=1 (single-step prediction). Multi-step rollouts (horizon=2,3) were never evaluated. The EFE rollout engine's behavior under multi-step uncertainty is unknown. |
| **FactGraph / text extraction** | MEDIUM | G6 (FactGraph entity extraction >70%) was never built or tested. The v1.1 plan's text->structured-representation pipeline has no Phase 1.5 validation. |
| **Behavioral diversity baseline** | LOW | G7 (behavioral entropy >0.5) was never measured. The Phase 1.5 eval showed PEDA &ne; Pragmatic but cannot quantify "diversity" against the v1.1 target. |

### Debt Carried Forward

- **No held-out evaluation**: All Phase 1.5 "validation" was done on the training distribution. The OOD generalization question that Phase 1 partial-training pilot raised remains unanswered.
- **No multi-episode statistics**: All Phase 1.5 behavioral findings are based on 1-2 episodes per condition. Statistical significance is unknown.
- **No replicable experiment protocol**: The v1.1 plan's tiered protocol was replaced by ad-hoc single-episode checks. Phase 2 cannot replicate Phase 1.5 findings without re-implementing the env.

---

## 4. Mitigation in Phase 2

### Direct Inheritances from Phase 1.5

| Phase 1.5 Artifact | Phase 2 Mitigation |
|--------------------|-------------------|
| `TextRoomEnv` pattern | Busybox Docker sandbox (`SandboxState`) provides **intrinsic** (not hand-crafted) uncertainty |
| `Perception.render_text()` | JSON-structured state representation (GLM-5.2 recommendation) reduces semantic noise |
| decompose_error (TextState branch) | Inherited with `hasattr` guard; extended for `SandboxState` with 3 dimensions (cwd, files, exit_code) |
| Ensemble error computer (3 checkpoints) | Inherited directly; LightJEPA hidden-state epistemic as optional upgrade (GLM-5.2 recommendation) |
| PEDA vs Pragmatic behavioral test | Formalized as multi-baseline comparison: PEDA, Pragmatic, Random Walk, Heuristic, Prompt-driven |

### C17 Compliance Plan

Per WATCHDOG C17 (L1/L2/L3 measurement on held-out data, with saturation detection), Phase 2 must:

1. **L1 (Exit Code)**: Evaluate on held-out command sequences. Target >90% accuracy. Saturation detection: if accuracy plateaus above 90% with <30% of training budget, the task is too easy &rarr; increase command/directory complexity.

2. **L2 (Filesystem Delta)**: Evaluate file-existence prediction on held-out scenarios (unseen directory structures). Target >70%. Saturation detection: if >90% on <50% of data, add multi-file operations and conditional commands.

3. **L3 (Output Summary)**: Semantic similarity (cosine >0.7) on held-out command outputs. Target >50%. This is explicitly "best-effort" per v1.1 design; Phase 2 will measure but not gate on L3.

### Addressing the Unvalidated Core Hypothesis

Phase 2 does not retroactively validate Phase 1.5 thresholds. Instead, it addresses the root cause directly:

- **Busybox sandbox** provides combinatorial state space (multiple directories, files, file contents, command history) &rarr; **natural epistemic uncertainty** without hand-crafted scenarios.
- **Multiple baselines** (PEDA, Pragmatic, Random, Heuristic, Prompt-driven) &rarr; **quantified effect size**, not binary pass/fail.
- **FHT (First Hit Time)** and **SCR (State Coverage Rate)** replace G4/G5 as more informative metrics.
- **N &ge; 10 episodes per condition** for statistical significance (WATCHDOG B5 compliance).

### What Phase 2 Cannot Fix

1. **The data scaling question** remains unanswered. Phase 1.5 could not determine how many samples a 0.5B text WM needs. Phase 2's data volumes will be different (command trajectories vs state-text pairs), so the answer will not transfer.
2. **TextWorld-specific risks** (natural language parsing, entity extraction from open-ended descriptions) are avoided entirely by using JSON-structured state. This is pragmatically correct but means those v1.1 capabilities were never validated.

---

## Summary

Phase 1.5 was a **pragmatic deviation**, not a failure. The v1.1 plan called for TextWorld-based validation at 3-4 weeks with formal thresholds; Python 3.14 incompatibility and hardware constraints forced a custom replacement. The replacement delivered infrastructure value (lightweight text env, decompose_error fix, behavioral signal) but did **not** validate any v1.1 threshold.

**The core hypothesis remains unvalidated entering Phase 2.** Phase 2's sandbox environment, richer metrics (FHT/SCR), held-out evaluation, and multi-baseline comparison are designed to finally test it &mdash; but Phase 1.5's debt means Phase 2 starts without a calibrated estimate of data requirements, model capacity, or effect size.
