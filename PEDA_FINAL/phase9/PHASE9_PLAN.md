# Phase 9 Plan — Post-PEDA Research Directions

**Status:** Design phase. 3 directions, all with pre-registered metrics and fail-fast gates.
**Date:** 2026-07-31
**Context:** PEDA DISPROVEN (PEDA_FINAL/PEDA_CONCLUSION.md). 19 experiments, all 3 charter Q answer No.

---

## Direction 1: Counter-Intuitive Sandbox (P0)

**Hypothesis:** If the environment systematically violates LLM semantic priors, prediction error can produce structured epistemic signal that count-based novelty cannot replicate.

**Design:** `PEDA_FINAL/phase9/plans/plan-counter-intuitive-sandbox.md`
**Docker:** `Dockerfile.counterintuitive` → `peda-sandbox:counterintuitive-v1` (built, 12/12 probe passing)

### Rules (Level 1)
| Command | Reversed behavior |
|---------|-------------------|
| `cat FILE` | Deletes FILE (unlink) |
| `echo FILE` | Reads FILE contents to stdout |
| `ls [DIR]` | Creates empty `<entry>.ls` twin per non-.ls entry; bounded |
| `grep PAT FILE` | Inverted matching (= real `grep -v`) |
| `head FILE` | Last N lines (default 10) |
| `tail FILE` | First N lines (default 10) |

Normal: `cd`, `pwd`, `wc`, `mkdir`, `touch`

### Micro-tasks
| ID | Level | Goal predicate | Wrong path | Right path |
|----|-------|---------------|------------|------------|
| `read_secret_ci` | L1 | `"9471" in last_output` | `cat docs/note.txt` deletes it | `echo docs/note.txt` |
| `read_data_ci` | L1 | `"line3" in last_output` | `cat data/lines.txt` deletes it | `echo data/lines.txt` |
| `find_warn_ci` | L2 | `"WARN" in out and "ERROR" not in out` | normal grep prints ERROR | inverted grep prints WARN only |

### Pre-registered Metrics (ALL FOUR must pass)
| # | Metric | Definition | Threshold |
|---|--------|-----------|-----------|
| M0 | Build gate | Probe script 12/12 | PASS (verified) |
| M1 | Prior breakage | Untrained LLM DLR on 30-pair probe set | **≤ 0.35** (normal sandbox ≥ 0.8) |
| M2 | Learnability | Held-out DLR after LoRA on ≤200 CI transitions | **≥ 0.70** |
| M3 | Behavioral productivity | PE agent completion vs count baseline, 20 eps × 3 tasks | **PE ≥ count − 10pp**; both improve ≥ 2× ep11-20 vs ep1-10; discovery steps ≤ 1.5× count |
| M4 | Error trajectory | Per-step mean prediction error E(t) | E(1..10) **≥ 0.5**; E(31..40) **≤ 0.5·E(1..10)** |

### Fail-Fast Conditions
1. **FF-CI-0** (M0): Probe 12/12 fails → DEAD (build broken) — PASSED
2. **FF-CI-1**: Normal-vs-CI divergence < 3/4 commands → wrappers not reversing → DEAD
3. **FF-CI-2**: LLM policy with normal priors scores > 0/20 on read_secret_ci → reversal too shallow → DEAD
4. **FF-CI-3** (M1): DLR > 0.35 → LLM can guess rules → DEAD (deepen reversal)
5. **FF-CI-4** (M2): DLR < 0.70 → rules not learnable → DEAD (loosen reversal)
6. **FF-CI-5**: Count baseline < 40% on CI tasks → env adversarially too hard → DEAD
7. **FF-CI-6** (M3): PE < count − 10pp → prediction error STILL not useful drive signal → **formal negative result**, charter-accepted

### Safety Deviation
- `--read-only` rootfs DROPPED for CI sandbox (cat deletes, ls creates). Justification: `--cap-drop=ALL --network none --tmpfs /tmp` + fresh container per episode. Workdir is a disposable tmpdir, NOT a host bind-mount.  
  **User sign-off required before harness patches. Status: PENDING.**

---

## Direction 2: Hierarchical Horizon Decomposition

**Hypothesis:** Flat agents (EFE horizon 1-3) cannot see delayed epistemic reward. A high-level planner (horizon 20-100) that analytically sums expected new states over a visit map can discover more than a flat count-based agent.

**Design:** `PEDA_FINAL/phase9/plans/plan-hierarchical-horizon.md`

### Architecture
- **High-Level Planner**: Scores frontiers by `J(f) = G(f) − λ·d`, where `G(f) = |{unvisited cells in BFS-ball(f, H_plan − d)}|`
- **Low-Level Executor**: BFS for nav/acquire; NoveltySearch (Phase 6/8) for search; horizon 1-3
- **Re-evaluation**: Every T_reeval steps or goal completion; switch with hysteresis (τ=0.15)
- **Goal-level prediction error**: PE_goal = |G_pred − realized| / max(G_pred, 1)

### Validation Environment
- **Primary**: Grid Maze (10x10 / 15x15 / 20x20), Variant A (plain) + Variant B (2 locked doors)
- **20x20 is decisive**: count = 0% at 8400 states; layered must win THERE
- **10x10 is sanity only** (ceiling guard: flat_count SCR ≥ 0.95 → FF void at this size)
- CPU-only, minutes per sweep

### Pre-registered Metrics
| Metric | Definition | Threshold |
|--------|-----------|-----------|
| dSCR | Δ Success Completion Rate vs random_goal | ≥ 0.05 |
| dFHT | Δ First-Hit-Time vs random_goal | ≥ 20 steps |
| Coverage gradient | SCR variance across λ sweep | ≥ 0.05 (non-flat) |

### Fail-Fast Conditions (anchored at 20x20 Variant B)
1. **FF-HH-1**: dSCR < 0.05 AND dFHT < 20 → DEAD
2. **FF-HH-2**: Coverage flat across λ sweep → DEAD
3. **FF-HH-3**: Re-eval adds < 0.02 SCR → simplify to open-loop
4. **FF-HH-4**: Layered fails to beat flat_count baseline → kill/redesign goal space
5. **FF-HH-5**: Only wins at 10x10 (not 15x15 or 20x20) → marginal, NOT decisive

### Required Env Change
~20 lines: Wire `MazeTask.locked_doors` into `GridMazeEnv` (door/key/use machinery exists, `GridMaze.generate()` never populates it).

---

## Direction 3: LLM-as-Hypothesis-Generator + Lightweight Discriminator

**Hypothesis:** Demoting the LLM from predictor to proposer, and using a lightweight discriminator (STRIPS schemas) to validate hypotheses, produces structured epistemic error where LLM direct prediction produces flat error.

**Design:** `PEDA_FINAL/phase9/plans/plan-hypothesis-generator.md`

### Architecture
- **LLM Proposer** (MiniCPM5-1B): Generates 3-5 candidate (action, claimed_outcome) pairs per state
- **Discriminator** (STRIPS primary, MLP arm): Predicts 5 atomic outcome predicates, computes confidence from schema coverage
- **5 Atomic Predicates**: exit_ok, output_nonempty, cwd_changed, listing_changed, cache_gained — NEVER file contents
- **Selection**: `argmax_a [α · uncertainty_disc(s,a) + (1-α) · count_novelty(s,a)]`

### MVP Validation (sandbox v4, <1 CPU-hour)
| Test | Metric | Threshold |
|------|--------|-----------|
| V1 Differential | AUC_disc vs AUC_llm (direct prediction) | AUC_disc ≥ 0.7, AUC_llm ≤ 0.55 |
| V2 Count-orthogonality | Cohen's d of e_disc(D) vs e_disc(E), zero-visit pairs | d ≥ 1.0 |
| V3 Gradient | Spearman ρ of e_disc vs feature-distance | ρ > 0.4 |

### Fail-Fast Conditions
1. **FF-HG-1** (F1): KL(empirical error ‖ uniform) < 0.35 nats → flat error → DEAD
2. **FF-HG-2** (F2): AUC < 0.7 OR KL < 0.3 nats → no separation → DEAD
3. **FF-HG-3** (F3): d < 1.0 → error = function of visit count → same as counting/JEPA → DEAD
4. **FF-HG-4** (F4): Proposer < 3 valid distinct actions on ≥ 50% of states → DEGENERATE
5. **FF-HG-5** (F5): Agent-level ≤ count baseline (post-MVP) → DEAD

---

## Priority

| Rank | Direction | Rationale |
|:----:|-----------|-----------|
| **P0** | Counter-Intuitive Sandbox | Directly attacks root cause (LLM prior too strong). Maximal-fairness test of PEDA hypothesis. |
| P1 | Hierarchical Horizon | Tests whether horizon is the bottleneck; generalizable beyond PEDA. |
| P2 | Hypothesis-Generator | Architecture gamble; STRIPS 45.8% vs 31.3% is only weak signal. |

---

## Verification Contract (WATCHDOG A2)

All metrics defined above are grep-able. All fail-fast conditions are pre-registered with numeric thresholds. No post-hoc threshold adjustment permitted. Any death verdict must cite the specific gate ID and measured value.

**This document is the single source of truth for Phase 9 acceptance criteria.**
