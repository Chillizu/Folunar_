# Count-Driven Autonomous Agent — Research Charter

**Archive reference**: [PEDA_CONCLUSION.md](./PEDA_CONCLUSION.md)
**Status**: ACTIVE — replaces PEDA count-driven agent line of inquiry
**Temporary name**: Folunar_ (count-driven phase)
**Core identity**: NOT PEDA. No prediction error. No EFE. No World Model pretensions.

---

## 1. Project Identity

This project inherits three surviving components from the concluded PEDA investigation:

| Inherited component | Status | Evidence |
|---|---|---|
| **Count-based pair novelty** | Core exploration driver | Beat all learned signals in 17 experiments at <1000 states; handles stochastic items (Phase 6) |
| **STRIPS action schemas** | Learned action model | 45.8% hit rate vs 31.3% fallback on v2 sandbox traces |
| **Data-driven candidate generation** | Infrastructure | v2→v3→v4 sandbox migrations: 0 crashes, zero regressions |
| **Success cache (memoization)** | 1-step solver | 20/20 1-step completions across 4 tasks on repeated (state, action) pairs |

**What is discarded**:

Discarded: prediction-error exploration (disproven), EFE (binary pragmatic dominance), JEPA (never beat counting, 37x slower), World Model (added nothing), PEDA Drive System (heuristic, unvalidated).

---

## 2. Research Question

> **How far can a purely count-driven agent — using pair-novelty exploration, learned STRIPS schemas, and success memoization — go before it needs a learnable forward model?**

This decomposes into four sub-questions:

1. **Scaling question**: At what state-space size does count-based novelty become information-theoretically insufficient? (Current bound: works at 1,100 states in deterministic mazes, fails at 8,400.)

2. **Composition question**: Can STRIPS schemas learned from single-step traces chain into multi-step plans? (Current baseline: 45.8% learned hits vs 31.3% fallback — improvement exists, gap remains.)

3. **Transfer question**: Do action schemas learned in one environment transfer to another with different filesystem layout, command set, or task structure?

4. **Drive question**: Can a principled, non-heuristic explore/exploit modulation system be built without a World Model or prediction-error signal?

---

## 3. What We Already Know (from PEDA)

### 3.1 Count-based exploration works well at small scale

| Environment | States | Count-based SCR | Best learned SCR | Ratio |
|---|---|---|---|---|
| Sandbox v2 | 65 | 50% | 50% (hybrid tied) | 1.0x |
| Sandbox v4 | 270 | 42% | 8% (JEPA hybrid) | 5.3x |
| Maze 10x10 (deterministic) | 1,100 | 100% | 0% (any JEPA variant) | — |
| Maze 10x10 (stochastic) | 1,100 | 100% | 67% (JEPA hybrid) | 1.5x |
| Maze 20x20 | 8,400 | 0% | 0% | — |

Source: Phase 5 JEPA exploration, 11 experiments across 4 sandboxes.

### 3.2 Learned exploration signals underperform counting

- **JEPA-only (pure epistemic)**: SCR ~0 across all environments
- **Hybrid (novelty + epistemic)**: never beat pure novelty in any regime
- **EFE with binary pragmatic**: dominated by the 0/1 goal predicate; the epistemic weighting term α was irrelevant
- **Root cause**: All unexplored transitions are equally uncertain to JEPA — it cannot distinguish "promising" from "dead-end" unknown regions. This is equivalent to counting, but 37x slower.

### 3.3 STRIPS learning works, with room to improve

| Metric | Learned schemas | Fallback (enumeration) | Gain |
|---|---|---|---|
| Hit rate | 45.8% | 31.3% | +14.5pp |
| Coverage | Commands used in training traces | All valid commands in sandbox | Wider but shallower |

Learned schemas are cwd-independent path predicates (post-bugfix). The 45.8% ceiling may be a data coverage issue, not a fundamental ceiling.

### 3.4 Data-driven generation and success cache

Three sandbox migrations (v2→v3→v4): 0 crashes. Candidate generator parses `--help` output, applies safety blacklist, produces actions deterministically. Success cache: 20/20 1-step completions on repeated (state, action) pairs — the simplest mechanism solves the cases a learned model would struggle with.

### 3.5 JEPA/WM training adds nothing

In every comparison (Phase 4, Phase 5, Phase 6), training a JEPA ensemble or World Model and using its epistemic signal did not improve exploration over a count-based novelty baseline. The signal is noisier, slower, and not better.

---

## 4. Open Questions

| Question | Importance | Current best guess | How to test |
|---|---|---|---|
| Where does counting break? | P0 | Between 1,100 and 8,400 states | Systematically scale maze size; measure SCR as function of state count |
| Can STRIPS schemas chain? | P0 | Yes, with path planning | Build 2-step and 3-step task set; compare learned chain vs brute-force |
| Do schemas transfer? | P1 | Partial — directory structure generalizes, file names don't | Train on sandbox v2, eval on v4 with overlapping command set |
| How far does pair-novelty handle stochasticity? | P1 | Transient items fine; persistent stochasticity may break | Phase 6 stochastic maze: compare pair-novelty vs random vs heuristic |
| Can a real drive system replace the PEDA heuristic? | P2 | Yes — satiation counters, simple bandit, or UCB | Ablate count-based with explore/exploit threshold; compare fixed ratio |
| Does success cache have a failure mode? | P1 | Stale cache on environment change | Inject environment mutations mid-run; measure stale-cache hit rate |

---

## 5. Success Criteria

| Criterion | Definition | Passing threshold |
|---|---|---|
| **Find the counting limit** | Measure SCR as function of state space size; identify the point where count-based equals random | Documented scaling curve with failure point identified |
| **Demonstrate STRIPS chaining** | Learned schemas enable multi-step plans that brute-force enumeration cannot find | >50% hit rate on 2-step tasks; >30% on 3-step |
| **Measure transfer** | Schemas trained on one sandbox improve performance on another | >10pp improvement over training from scratch |
| **Build a real drive system** | Not the PEDA heuristic; principled explore/exploit modulator | Controlled experiment: satiation-based > fixed ratio > random |
| **Negative results accepted** | Finding where counting breaks IS a success | Clear documentation of failure mode and state-space boundary |

**All sub-questions answered is NOT required for project success.** A concrete, experimentally grounded boundary on count-based methods is a valid terminal outcome.

---

## 6. Principles

1. **No prediction error. No World Model. Period.** If it looks like a forward model, it belongs in a different project.

2. **Novelty is count-based until proven insufficient.** Every learned exploration method must beat counting on a task counting cannot solve — not on one counting already handles.

3. **Every module must have a controlled experiment.** "Seems useful" is not evidence. Every new component enters via ablation: compare with and without, on at least 20 episodes.

4. **Learning is intermittent, batch, and from traces.** No per-step online SGD. Collect episodes, learn in batch, freeze, evaluate.

5. **Honest labeling.** If a component doesn't increase SCR over the baseline, say so. Do not rename it, reparameterize it, or wait for more data unless there is a specific, testable hypothesis about what changes.

---

## 7. Non-Goals

- NOT resurrecting PEDA, JEPA, EFE, or any World Model formulation
- NOT building a product, API, or deployment target
- NOT chasing SOTA benchmarks on any leaderboard
- NOT implementing Active Inference, Free Energy Principle, or any FEP-derived theory
- NOT adding "creativity," "consciousness," or other anthropomorphic labels to agent behavior
- NOT implementing a Homeostatic Drive System unless it can be formulated without prediction error

---

## 8. Relationship to PEDA

PEDA is concluded. This charter inherits: surviving code (`src/phase5/action_model.py`, `explorer.py`, candidate generator), experimental methodology (controlled baselines, ablation-first), archival discipline, and the 17 negative-result experiments.

Archived: PEDA_CONCLUSION.md, `PEDA_FINAL/archive/` (Phase 1–5), RESEARCH_CHARTER.md (superseded), all PEDA design docs.

**Boundary condition**: Any proposal to reintroduce a predictive World Model must include: (1) an environment where count-based fails, (2) a hypothesis about what the WM predicts that counting cannot, (3) a controlled experiment.

---

*This charter was drafted on 2026-07-31, immediately following PEDA conclusion. It supersedes RESEARCH_CHARTER.md for count-driven work. PEDA-related work (archival, replication, publication) still references the original charter.*
