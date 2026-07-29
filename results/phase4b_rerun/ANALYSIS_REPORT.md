# Phase 4B Rerun: Multi-Task Generalization Analysis

**Date**: 2026-07-29  
**Data**: `results/phase4b_rerun/*.jsonl` — 13 files, 5 episodes each (65 total)  
**Hardware**: AWS g4dn.xlarge (T4 GPU)  
**Max steps**: 20 per episode

---

## Data Composition

| Condition | Task | N | Files |
|-----------|------|---|-------|
| PEDA known | read_hello, count_lines, find_secret, read_note | 5 each | 4 files |
| PEDA unknown | read_hello, count_lines, find_secret, read_note | 5 each | 4 files |
| Pragmatic known | read_hello, count_lines, find_secret, read_note | 5 each | 4 files |
| Pragmatic unknown | count_lines | 5 | 1 file |

**Total**: 65 episodes across 4 tasks, 3-4 conditions per task (PEDA known/unknown, Pragmatic known, + Pragmatic unknown for count_lines).

---

## Critical Note on `success` Field

All 65 episodes report `success=True`. However, examination of `fht` (first hit time, -1 = never) and `scr` (success completion ratio, 0.05-0.1 = near-zero) reveals that the sandbox environment marks all non-error episodes as successful regardless of whether the target file was found. **The real success metric is FHT >= 0.**

---

## 1. Per-Task: PEDA Known vs Unknown (FHT)

| Task | PEDA known FHT | PEDA unknown FHT | Hit (k) | Hit (u) | MWU p |
|------|----------------|------------------|---------|---------|-------|
| read_hello | all -1 | [-1, 1, -1, -1, 1] | 0/5 | **2/5** | p=0.1770 |
| count_lines | all -1 | all -1 | 0/5 | 0/5 | N/A |
| find_secret | all -1 | all -1 | 0/5 | 0/5 | N/A |
| read_note | all -1 | all -1 | 0/5 | 0/5 | N/A |

**Finding**: read_hello is the only task where any PEDA episode achieves true early completion. PEDA unknown shows 2/5 episodes hitting the target at step 1 (FHT=1, 2-step total). PEDA known shows 0/5. Direction consistent with epistemic advantage but **not significant** (p=0.1770, N=5 per cell).

---

## 2. PEDA Known vs Pragmatic Known (FHT)

| Task | PEDA known FHT | Pragmatic known FHT | Hit Pk | Hit Prk | MWU p |
|------|----------------|---------------------|--------|---------|-------|
| read_hello | all -1 | [0, -1, -1, 0, -1] | 0/5 | **2/5** | p=0.1770 |
| count_lines | all -1 | all -1 | 0/5 | 0/5 | N/A |
| find_secret | all -1 | all -1 | 0/5 | 0/5 | N/A |
| read_note | all -1 | all -1 | 0/5 | 0/5 | N/A |

**Finding**: Pragmatic known read_hello shows 2/5 episodes hitting at step 0 (FHT=0, SCR=1.0, 1-step total). Neither baseline hits on the other 3 tasks. The 2 successful pragmatic episodes were instant solves (read_hello from /sandbox), consistent with the known fact that pragmatic reward directly reads /hello.txt from the root.

---

## 3. Dead-Loop Rate Comparison

| Task | PEDA known | PEDA unknown | Pragmatic known |
|------|-----------|-------------|-----------------|
| read_hello | 0.00 | 0.00 | **0.54** |
| count_lines | 0.00 | 0.00 | **0.90** |
| find_secret | 0.00 | 0.00 | **0.90** |
| read_note | 0.00 | 0.00 | **0.90** |

**Key finding**: Pragmatic baselines exhibit severe dead-loop behavior (0.54-0.90) on all tasks except the 2/5 read_hello episodes that solved instantly. PEDA maintains dead_loop_rate=0.00 across all conditions — **PEDA never dead-loops**, even when it fails to find the target. This confirms the known action-space pathology (pragmatic oscillates between `ls` and `ls data`).

---

## 4. Task Difficulty Gradient

All tasks except read_hello show 0/5 hits across all baselines. No difficulty gradient can be established within the 20-step limit — the tasks are **uniformly unsolved** at this depth.

---

## 5. Phase 3 Comparison (read_hello)

| Metric | Phase 3 (N=20) | Phase 4B (N=5) | Delta |
|--------|----------------|----------------|-------|
| PEDA known hit rate | 20/20 (100%) | 0/5 (0%) | **Failed to replicate** |
| PEDA known steps | 6.8 | 20.0 (ceiling) | **Failed to replicate** |

**Critical divergence**: Phase 3 achieved 100% success across all 80 episodes (4 conditions x N=20) with mean steps 6.8-10.0. Phase 4B shows near-zero hits on all conditions except read_hello (2/5 per condition on PEDA unknown + Pragmatic known).

**Possible causes**:
- Different adapter/checkpoint (Phase 3 used sandbox-trained adapter; Phase 4B may use a different checkpoint)
- Different sandbox version (Phase 3 sandbox may have fewer directories / simpler structure)
- Stochastic collapse (epistemic weight or sampling temperature caused exploration collapse)

---

## 6. Summary Table

| Task | Condition | Steps | FHT | SCR | DeadLoop | Hit |
|------|-----------|-------|-----|-----|----------|-----|
| read_hello | PEDA known | 20.0 | -1.0 | 0.09 | 0.00 | 0/5 |
| read_hello | PEDA unknown | 12.8 | -0.2 | 0.25 | 0.00 | **2/5** |
| read_hello | Pragmatic known | 12.4 | -0.6 | 0.43 | 0.54 | **2/5** |
| count_lines | PEDA known | 20.0 | -1.0 | 0.09 | 0.00 | 0/5 |
| count_lines | PEDA unknown | 20.0 | -1.0 | 0.07 | 0.00 | 0/5 |
| count_lines | Pragmatic known | 20.0 | -1.0 | 0.05 | 0.90 | 0/5 |
| count_lines | Pragmatic unknown | 20.0 | -1.0 | 0.05 | 0.90 | 0/5 |
| find_secret | PEDA known | 20.0 | -1.0 | 0.09 | 0.00 | 0/5 |
| find_secret | PEDA unknown | 20.0 | -1.0 | 0.07 | 0.00 | 0/5 |
| find_secret | Pragmatic known | 20.0 | -1.0 | 0.05 | 0.90 | 0/5 |
| read_note | PEDA known | 20.0 | -1.0 | 0.09 | 0.00 | 0/5 |
| read_note | PEDA unknown | 20.0 | -1.0 | 0.07 | 0.00 | 0/5 |
| read_note | Pragmatic known | 20.0 | -1.0 | 0.05 | 0.90 | 0/5 |

---

## 7. Verdict

**Does epistemic advantage generalize?** Inconclusive — the experiment failed to replicate Phase 3's baseline performance. The 20-step cap is insufficient for the harder tasks (count_lines, find_secret, read_note) under current sandbox conditions.

**Is it task-dependent?** Yes — read_hello alone shows any hits, consistent with it being the easiest task (/hello.txt at sandbox root).

**Confirmed findings**:
1. PEDA never dead-loops (dead_loop_rate=0.00 everywhere vs 0.54-0.90 for pragmatic on non-read_hello)
2. PEDA unknown read_hello: 2/5 early hits (FHT=1) — numerically consistent with epistemic exploration enabling faster discovery
3. read_hello is the only tractable task at N=5, max_steps=20

**Limitations**:
- N=5 per cell severely underpowered (MWU not computable for constant arrays)
- Phase 3 base rates not replicated (possible checkpoint/sandbox mismatch)
- 20-step cap truncates exploration

**Recommendation**: Increase max_steps to >=50 for harder tasks, verify checkpoint/sandbox consistency with Phase 3, and run N>=20 per cell.
