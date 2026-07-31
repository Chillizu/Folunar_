# Phase 4 Evidence Bundle — Closed-Loop Self-Training & Multi-Task Generalization

**Scout:** S6Phase4
**Date:** 2026-07-31
**experiment_ids:** [E10, E11, E12, E13.01-E13.04 (P4 EFE rounds, part of E13 inventory)]
**Scope:** All Phase 4 closed-loop files: `results/phase4a/`, `results/phase4b_rerun/`, `results/phase4b_v4/` (E12 evidence), `PEDA_FINAL/PHASE4_EXPERIMENT_PLAN.md`, `scripts/phase4_closed_loop.py`, `scripts/phase4_jepa_experiment.py`, `src/phase4/jepa_peda.py`, `src/phase4/__init__.py`, plus ground-truth `PEDA_FINAL/PEDA_CONCLUSION.md` rows and `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` (5 bugs + JEPA 4 rounds), `PEDA_WORKING_LOG.md` Phase 4 entries.

---

## 1. Files Read

| Path | Description |
|---|---|
| `results/phase4a/PHASE4_RESULTS.md` | Experiment A summary (partial data recovery; per-episode JSONL lost) |
| `results/phase4b_rerun/ANALYSIS_REPORT.md` | Experiment B rerun statistical analysis (65 episodes) |
| `results/phase4b_rerun/*.jsonl` (13 files) | Raw per-episode data, 5 episodes each (all read) |
| `results/phase4b_v4/peda_known_read_hello.jsonl`, `peda_unknown_read_hello.jsonl`, `pragmatic_known_read_hello.jsonl`, `pragmatic_unknown_read_hello.jsonl` | v4 rerun raw data (E12 replication evidence; 16 files total in dir, 4 read + work-log table covers rest) |
| `PEDA_FINAL/PHASE4_EXPERIMENT_PLAN.md` | Pre-registered Phase 4 plan (367 lines) |
| `PEDA_FINAL/PEDA_CONCLUSION.md` | Ground-truth conclusion (Phase 4 rows + root causes) |
| `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` | JEPA 4-round table + 5-bugs list |
| `scripts/phase4_closed_loop.py` | Experiment A implementation (792 lines, fully read) |
| `scripts/phase4_jepa_experiment.py` | JEPA-PEDA 4-condition experiment (463 lines, fully read) |
| `src/phase4/jepa_peda.py` | JEPAPEDA agent (EFE/hybrid scoring, bug fixes in place) |
| `src/phase4/__init__.py` | 1-line module doc |
| `PEDA_WORKING_LOG.md` | Entries 2026-07-28 (4A), 2026-07-29 (4B rerun), 2026-07-29 06:30 (EVAL: `success=True` correction, v4) |

---

## 2. Quantitative Results (E10, E11, E12)

### E10 — Phase 4A: Closed-Loop Self-Training (4 blocks × N=10, read_hello)

Block-level aggregates (recovered from tmux scrollback; per-episode JSONL LOST):

| Number | Source | Verbatim quote |
|---|---|---|
| B1: 2/10 (20%), 16.2 steps; B2: 6/10 (60%), 11.0; B3: 8/10 (80%), 6.8; B4: 6/10 (60%), 14.6 | `results/phase4a/PHASE4_RESULTS.md:22-25` | `| 1 \| 2/10 (20%) \| 16.2 \| 2/10 (20%) \| 16.2 |` … `| 4 \| 6/10 (60%) \| 14.6 \| 2/10 (20%) \| 16.2 |` |
| PEDA+Freeze flat 20%, 16.2 steps all 4 blocks | `results/phase4a/PHASE4_RESULTS.md:22-25` | `PEDA+Freeze Success` column constant `2/10 (20%)` / `16.2` |
| 4× success increase; steps 16.2→6.8 | `results/phase4a/PHASE4_RESULTS.md:33` | `**PEDA+Train success rate increased 4× across blocks (2/10 → 8/10), while PEDA+Freeze remained constant (2/10 all blocks).** Average steps dropped from 16.2 to 6.8 in the training condition.` |
| Pragmatic only 1 block: 2/10 (20%) | `results/phase4a/PHASE4_RESULTS.md:29`; `PEDA_WORKING_LOG.md:1591` | `| Pragmatic \| 2/10 (20%) \| — \| — \| — |`; `Pragmatic: 仅完成 1 block（2/10, 16.2 步），实验被提前终止。` |
| 65/80 Exp-B episodes completed | `results/phase4a/PHASE4_RESULTS.md:51` | `65/80 episodes completed. 3 peda_unknown conditions (count_lines, find_secret, read_note) failed to produce output files. JSONL data lost.` |
| ~14 GPU-hours | `results/phase4a/PHASE4_RESULTS.md:6` | `**Hardware:** NVIDIA T4 (g4dn.xlarge, i-0281f99a610497865), ~14 GPU-hours` |
| Block 4 regression 6/10, 14.6 steps | `results/phase4a/PHASE4_RESULTS.md:40` | `Block 4 showed regression (6/10, 14.6 steps) — possible overfitting or saturation` |
| Ground truth row | `PEDA_FINAL/PEDA_CONCLUSION.md:50` | `| 4A \| Sandbox v2 \| Closed-loop self-training 4 blocks, N=10 each \| Q3 \| PEDA+Train: 20%-60%-80%-60% success. PEDA+Freeze: flat 20% \| POSITIVE — but success-cache mechanism |` |

⚠️ **E10 caution (paper-critical):** The 4A "success" numbers use the `success` field, whose semantics were LATER established as unreliable. `PEDA_WORKING_LOG.md:1696`: `**Phase 3 的 "20/20 success" 是误解**。`success=True` 字段的判定是 `SCR > 0`，而非 `fht >= 0`。…Phase 3 和 Phase 4B 的 **所有 episode 的 success 字段都是 True——这是常真字段，无判别力**。` Per-episode data for 4A was lost, so E10's success curve CANNOT be re-verified against FHT; the conclusion doc downgrades it to "POSITIVE — but success-cache mechanism".

### E11 — Phase 4B: Multi-Task Generalization (rerun, max_steps=20)

Design: 4 tasks × 2 baselines (PEDA/Pragmatic) × 2 CWD types (known/unknown), N=5; **13/16 cells** completed (pragmatic_unknown missing for read_hello/find_secret/read_note) — `results/phase4b_rerun/ANALYSIS_REPORT.md:19`: `**Total**: 65 episodes across 4 tasks, 3-4 conditions per task (PEDA known/unknown, Pragmatic known, + Pragmatic unknown for count_lines).`

| Number | Source | Verbatim quote |
|---|---|---|
| 65 episodes, 13 files | `results/phase4b_rerun/ANALYSIS_REPORT.md:4` | `**Data**: `results/phase4b_rerun/*.jsonl` — 13 files, 5 episodes each (65 total)` |
| All 65 `success=True` (tautology) | `results/phase4b_rerun/ANALYSIS_REPORT.md:25` | `All 65 episodes report `success=True`. … **The real success metric is FHT >= 0.**` |
| PEDA unknown read_hello 2/5 hits, MWU p=0.1770 | `results/phase4b_rerun/ANALYSIS_REPORT.md:33,38` | `| read_hello \| all -1 \| [-1, 1, -1, -1, 1] \| 0/5 \| **2/5** \| p=0.1770 |`; `Direction consistent with epistemic advantage but **not significant** (p=0.1770, N=5 per cell).` |
| count_lines/find_secret/read_note: 0/5 hits ALL conditions | `results/phase4b_rerun/ANALYSIS_REPORT.md:34-36` | `| count_lines \| all -1 \| all -1 \| 0/5 \| 0/5 \| N/A |` (same for find_secret, read_note) |
| Pragmatic known read_hello 2/5 hits (FHT=0, 1-step) | `results/phase4b_rerun/ANALYSIS_REPORT.md:46,51` | `| read_hello \| all -1 \| [0, -1, -1, 0, -1] \| 0/5 \| **2/5** \| p=0.1770 |`; `The 2 successful pragmatic episodes were instant solves (read_hello from /sandbox)` |
| Dead-loop: PEDA 0.00 everywhere; Pragmatic 0.54/0.90 | `results/phase4b_rerun/ANALYSIS_REPORT.md:59-62,64` | `| read_hello \| 0.00 \| 0.00 \| **0.54** |` … `| count_lines … **0.90** |`; `PEDA maintains dead_loop_rate=0.00 across all conditions — **PEDA never dead-loops**` |
| Summary table (per condition) | `results/phase4b_rerun/ANALYSIS_REPORT.md:94-106` | e.g. `| read_hello \| PEDA known \| 20.0 \| -1.0 \| 0.09 \| 0.00 \| 0/5 |`, `| read_hello \| PEDA unknown \| 12.8 \| -0.2 \| 0.25 \| 0.00 \| **2/5** |`, `| read_hello \| Pragmatic known \| 12.4 \| -0.6 \| 0.43 \| 0.54 \| **2/5** |` |
| Ground truth row | `PEDA_FINAL/PEDA_CONCLUSION.md:51` | `| 4B \| Sandbox v2+v4 \| Multi-task generalization (4 tasks x 2 baselines x 2 conditions) \| Q3 \| read_hello peda_unknown 40% (2/5). count_lines/find_secret/read_note: **all zero** hits \| FAIL — WM cannot solve any task beyond cat hello.txt |` |

Raw JSONL confirmations (phase4b_rerun):
- `peda_unknown_read_hello.jsonl:2,5` (episodes 1, 4): `{"baseline": "peda", "condition": "unknown", "cwd": "/sandbox/projects", …, "steps_count": 2, "success": true, "fht": 1, "scr": 0.5, "dead_loop_rate": 0.0, …}` — both hits are `/sandbox/projects`, 2 steps, FHT=1.
- `peda_known_read_hello.jsonl` (all 5): steps_count=20, fht=-1, scr 0.05-0.1.
- `pragmatic_known_read_hello.jsonl:1,4` (episodes 0, 3): `"cwd": "/sandbox", "steps_count": 1, "success": true, "fht": 0, "scr": 1.0, "dead_loop_rate": 0.0`; remaining 3 episodes: steps 20, fht=-1, **dead_loop_rate 0.9**.
- `pragmatic_known_{count_lines,find_secret,read_note}.jsonl` + `pragmatic_unknown_count_lines.jsonl`: all fht=-1, dead_loop_rate 0.9, scr 0.05.

### E12 — Phase 4B v4: Phase 3 Replication with Corrected `fht` Metric (max_steps=10)

| Number | Source | Verbatim quote |
|---|---|---|
| Phase 3 true hit rates (fht≥0): peda_known 0/20; peda_unknown **7/20 (35%)**; pragmatic_known 7/20 (35%); pragmatic_unknown 0/20 | `PEDA_WORKING_LOG.md:1715-1718` | `- peda_known: 0/20（所有 cwd 全零）`, `- peda_unknown: 7/20（全来自 /sandbox/projects）`, `- pragmatic_known: 7/20（全来自 /sandbox，1-step cat hello.txt）`, `- pragmatic_unknown: 0/20` |
| Phase 3 vs v4 replication: peda_unknown 35% → 40% (2/5); pragmatic_known 35% → 40% (2/5); zeros match | `PEDA_WORKING_LOG.md:1702-1705` | `| peda_known read_hello \| **0/20** \| **0/5** |`, `| peda_unknown read_hello \| **7/20 (35%)** \| **2/5 (40%)** |`, `| pragmatic_known read_hello \| **7/20 (35%)** \| **2/5 (40%)** |`, `| pragmatic_unknown read_hello \| **0/20** \| **0/5** |` |
| Replication verdict | `PEDA_WORKING_LOG.md:1707` | `**Phase 4B v4 完全复现 Phase 3，无退化，无翻车。**` |
| Tautology mechanism (root cause) | `PEDA_WORKING_LOG.md:1711-1712` | `**1. `success=True` 是常真字段**`; `` `phase3_sandbox_experiment.py:132`: `"success": metrics["scr"] > 0`。SCR = 去重状态数/步数。max_steps=10、agent 访问 >=2 个目录时 SCR >= 0.2，`success` 始终 True。 `` |
| Phase 3 significance re-attribution | `PEDA_WORKING_LOG.md:1720` | `PEDA 的显著优势是 peda_unknown (7/20) vs pragmatic_unknown (0/20)，p=0.0043。这是在 unknown 环境中 epistemic uncertainty 驱动探索的胜利，而非 known 环境中的效率优势。` |
| v4 full per-condition table | `PEDA_WORKING_LOG.md:1730-1752` | `| peda_unknown_read_hello \| 5 \| 2 \| 40% \| 6.8 \| 0.300 \| 0.00 |`, `| pragmatic_known_read_hello \| 5 \| 2 \| 40% \| 6.4 \| 0.460 \| 0.48 |` (all other 14 cells 0 hits) |
| Ground truth row | `PEDA_FINAL/PEDA_CONCLUSION.md:52` | `| 4B \| Sandbox v4 \| Phase 3 replication with corrected metric (fht) \| Q3 \| Phase 3 replicated: peda_unknown 35-40% hit, pragmatic_unknown 0% \| CONFIRMED — only read_hello, only /sandbox/projects |` |

Raw v4 confirmations (verified directly):
- `peda_unknown_read_hello.jsonl:2,5`: `"cwd": "/sandbox/projects", "steps_count": 2, "fht": 1, "dead_loop_rate": 0.0` (2/5 hits).
- `pragmatic_known_read_hello.jsonl:1,4`: `"cwd": "/sandbox", "steps_count": 1, "fht": 0, "scr": 1.0` (2/5 hits).
- `peda_known_read_hello.jsonl` (all): steps 10, fht=-1, dlr 0.0 (0/5). `pragmatic_unknown_read_hello.jsonl` (all): steps 10, fht=-1, **dlr 0.8** (0/5).

### E13 (P4 EFE rounds — "JEPA 4 rounds: novelty_only vs jepa_efe")

4 conditions in `scripts/phase4_jepa_experiment.py:63-77`: `jepa_efe` (alpha=0.5), `jepa_only` (alpha=1.0), `pragmatic_only` (alpha=0.0), `novelty_only` (count-based, `NOVELTY_ONLY`). Raw per-round JSONL was NOT preserved in `results/` (only summary numbers survive).

| Number | Source | Verbatim quote |
|---|---|---|
| Count-based 50% vs JEPA best 25% (P4 EFE, 4 rounds, 65-270 states) | `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:18` | `| P4 EFE (4 rounds) \| 65-270 \| 50% \| 25% \| — |` |
| jepa_efe specifically 17% | `PEDA_FINAL/PEDA_CONCLUSION.md:53` | `| 5 \| Sandbox v2/v3/v4 \| JEPA forward dynamics + hybrid, 11 exps \| Q1, Q2, Q3 \| Novelty-only 50% > jepa_efe 17% on read_hello. JEPA loss converges, no exploration gain \| FAIL — JEPA uncertainty flat across all unexplored states |` |
| No regime improvement | `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:9` | `Across 11 experiments spanning 4 sandboxes (v2/v3/v4 grid maze deterministic/stochastic), JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**.` |
| Pure epistemic SCR ~0 | `PEDA_FINAL/PEDA_CONCLUSION.md:54` | `| 5 \| Sandbox v4 \| Pure epistemic (jepa_only) explorer \| Q2 \| SCR ~0 across all tasks, zero room exploration \| FAIL — epistemic signal too weak to drive useful behavior |` |
| 37× cost | `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:38` | `All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower.` |
| JEPA loss converges 45→15 | `PEDA_FINAL/PEDA_CONCLUSION.md:90`; `README.md:27` | `JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states.`; `JEPA MLP predictor \| UNCERTAIN \| Loss always converges (45→15), MLP learns dynamics` |

Note on 17% vs 25%: `jepa_efe` condition = 17%; "JEPA best" across the 4 rounds/configs = 25% (README). Both are summary-level; per-round detail is not recoverable from disk.

---

## 3. Experimental Conditions

**E10 (Experiment A)** — `results/phase4a/PHASE4_RESULTS.md:6,13`: NVIDIA T4 (g4dn.xlarge, instance i-0281f99a610497865), ~14 GPU-hours; 3 conditions × 4 blocks × N=10, task `read_hello`, max 20 steps; adapter base `checkpoints/phase2/sandbox_adapter_v2_full` (line 7). Model: Qwen2.5-0.5B-Instruct (`scripts/phase4_closed_loop.py:302` default `~/models/Qwen2.5-0.5B-Instruct`). Drive weights `curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0`, `PRAGMATIC_WEIGHT=3.0` (`phase4_closed_loop.py:64-66`). **Deviation from plan:** plan §2.2 specified unknown-CWDs-only for all blocks; the script round-robins `ALL_CWDS` (3 known + 3 unknown, line 95-96), so E10 mixed known+unknown CWDs.

**E11 (Experiment B rerun)** — `results/phase4b_rerun/ANALYSIS_REPORT.md:5-6`: AWS g4dn.xlarge (T4 GPU), max steps 20, N=5 per cell, 13/16 cells. CWDs: known `/sandbox`, `/sandbox/data`, `/sandbox/docs`; unknown `/sandbox/logs`, `/sandbox/projects`, `/sandbox/tmp` (from `phase4_closed_loop.py:69-70`).

**E12 (v4 replication)** — `PEDA_WORKING_LOG.md:1691-1692`: `results/phase4b_v4/*.jsonl` (16 files, 79 episodes), max_steps=10, ensemble mode.

**JEPA rounds (P4 EFE)** — `scripts/phase4_jepa_experiment.py:41,57-58`: Docker image `peda-sandbox:v4`; model `~/models/Qwen2.5-0.5B-Instruct`; JEPA ensemble `n_ensemble=3` MLP predictors on frozen Qwen embeddings (`src/phase4/jepa_peda.py:54-56`); v4 CWD set: 5 known + 8 unknown (11 CWDs, lines 48-54). Alpha per condition: 0.5 / 1.0 / 0.0 / (count-based).

---

## 4. Methodology

**E10 training loop** (`scripts/phase4_closed_loop.py`): per block, N episodes with buffer `buffer_size=10000, update_interval=10000` (auto-update disabled, lines 316-320); after each block `_force_block_update` (lines 147-214): full-buffer prioritized sample (`batch_size=min(64, n)`), `lora_finetune(data, epochs=1, learning_rate=2e-4, batch_size=4)`, saves adapter to `output_dir/block_{N}_adapter` (lines 188-203). Metrics per episode: steps, success, scr, fht (first victory step), dead_loop_rate (consecutive repeated action fraction), mean epistemic/aleatoric error (lines 227-251). Plan targets: H1 Jonckheere-Terpstra trend, H2 MW one-tailed Train-B4 < Freeze-B4, H3 paired sign test on loss (`PHASE4_EXPERIMENT_PLAN.md` §6.1) — **none could be computed** due to per-episode data loss (`PHASE4_RESULTS.md:67`).

**E11 statistics** (`PHASE4_EXPERIMENT_PLAN.md` §3.4): per-task MW one-tailed, Bonferroni α=0.0125; plan pre-registered N=5 → 80% power for d≥1.65, explicitly "underpowered… the analysis is qualitative and effect-size oriented" (§3.7).

**E12 correction**: metric re-audit established `success` is defined as `SCR > 0` (`PEDA_WORKING_LOG.md:1712`), making it a constant-true field; FHT≥0 is the only discriminating metric.

---

## 5. Verbatim Conclusions from Source Docs

1. `results/phase4a/PHASE4_RESULTS.md:60`: `**Core finding: Self-training works.** PEDA's Learning Module, when run intermittently in the loop, produces measurable behavioral improvement (2/10 → 8/10 success). The frozen control rules out confounding factors.` — **Superseded/cautioned by later metric audit (E12) and by conclusion doc reclassification (E10 row).**
2. `results/phase4b_rerun/ANALYSIS_REPORT.md:112`: `**Does epistemic advantage generalize?** Inconclusive — the experiment failed to replicate Phase 3's baseline performance. The 20-step cap is insufficient for the harder tasks (count_lines, find_secret, read_note) under current sandbox conditions.`
3. `results/phase4b_rerun/ANALYSIS_REPORT.md:117`: `1. PEDA never dead-loops (dead_loop_rate=0.00 everywhere vs 0.54-0.90 for pragmatic on non-read_hello)`
4. `PEDA_WORKING_LOG.md:1696`: `**Phase 3 的 "20/20 success" 是误解**` (Phase 3's "20/20 success" is a misunderstanding — `success` is constant-true).
5. `PEDA_FINAL/PEDA_CONCLUSION.md:50`: 4A `POSITIVE — but success-cache mechanism`; `:51` 4B `FAIL — WM cannot solve any task beyond cat hello.txt`; `:52` E12 `CONFIRMED — only read_hello, only /sandbox/projects`.
6. `PEDA_FINAL/PEDA_CONCLUSION.md:72`: `3. **No replication on any other task.** Phase 4B across 4 tasks showed hit rate = 0% on count_lines, find_secret, and read_note for ALL baselines including PEDA. … no epistemic exploration is triggered because there is no prediction to be uncertain about.`

---

## 6. The 5 Bugs (before/after impact)

Source list: `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:57-61`. Impact documented by code state (fixes in place); explicit before/after metric deltas survive only for bug 4.

| # | Bug (before) | Fix (after) | Evidence |
|---|---|---|---|
| 1 | Path predicate cwd-unaware — task success checks failed for cwd-relative actions (e.g. `cat note.txt` from `/sandbox/docs`) | `_action_hits_target(action, state, target_rel_path)` resolves cwd-relative paths | `README.md:57`; `src/phase2/tasks.py:4` (`def _action_hits_target(action, state, target_rel_path):` docstring `Both `cat docs/note.txt` (absolute) and `cat note.txt` from /sandbox/docs`…) |
| 2 | Novelty tie-breaking alphabetical (distorts action selection among equal scores) | Action priority ordering (`cat/head/tail`=0 < `grep/find/wc`=1 < `cd`=2 < `ls/pwd/echo`=3), tie-break `-self._action_priority(a)` | `README.md:58`; `src/phase4/jepa_peda.py:50-53,162-164,206-207`; `scripts/phase5_jepa_experiment.py:150` |
| 3 | `final_state.victory` always False → success metric always 0% (metric bug, makes success/fht unmeasurable) | Return `next_state` with `victory=True` on success before break | `README.md:59`; `scripts/phase4_jepa_experiment.py:216-218` (`if success: next_state.victory = True; final_state = next_state; break`) |
| 4 | Dead-loop rate 0.53-0.80 (repeated (state,action) pairs) | Repeat-action penalty: repeat_count ≥ 2 in last 5 steps → `score *= 0.1` | `README.md:60`; `src/phase4/jepa_peda.py:196-203` (`if repeat_count >= 2: score *= 0.1`) |
| 5 | JEPA predictors cross-task contamination (dynamics trained on task A leak into task B) | `reset_predictors()` reinitializes MLP weights + optimizers between conditions; also resets novelty explorer and dead-loop history | `README.md:61`; `src/phase4/jepa_peda.py:82-85` (`self.jepa.reset_predictors(); self._recent_history.clear(); self._novelty_explorer = NoveltyExplorer()`); `src/phase5/jepa_wm.py:100-103`; `scripts/phase4_jepa_experiment.py:328-330` |

Bug 4 quantifiable impact: post-fix PEDA DLR = 0.00 across all conditions (`results/phase4b_rerun/ANALYSIS_REPORT.md:59-62`) vs pre-fix 0.53-0.80 (README:60). Related: pragmatic baseline DLR 0.48-0.90 shows the pathology the penalty addresses (`ANALYSIS_REPORT.md:64`: `pragmatic oscillates between `ls` and `ls data``).

---

## 7. Contradictions & Gaps

1. **E10 success curve not verifiable + semantics suspect.** Per-episode JSONL destroyed (`PHASE4_RESULTS.md:38`, `PEDA_WORKING_LOG.md:1597-1599`); `success` field later proven constant-true in Phase 3/4B scripts (`PEDA_WORKING_LOG.md:1712`). The 20%→80% curve was recovered only from tmux scrollback (`PHASE4_RESULTS.md:66`). Conclusion doc reclassifies 4A as "POSITIVE — but success-cache mechanism" (`PEDA_CONCLUSION.md:50`) — i.e., the improvement is attributed to the success cache, not to WM learning.
2. **E11 rerun vs E12 v4 contradict each other on Phase 3 replication.** Rerun report: `Failed to replicate` (`ANALYSIS_REPORT.md:78-79`); v4 EVAL: `Phase 4B v4 完全复现 Phase 3，无退化，无翻车` (`PEDA_WORKING_LOG.md:1707`). Resolution: the rerun compared against Phase 3's bogus `success`-field numbers; v4 re-audited Phase 3 with FHT and found consistency. Paper MUST use FHT-based numbers (E12), not `success`-based.
3. **JEPA 4-round raw data lost.** No `results/phase4_jepa_*.jsonl` exists on disk; only summary figures survive (`README.md:18`: 50% vs 25%; `PEDA_CONCLUSION.md:53`: 50% vs 17% for jepa_efe). 17% (jepa_efe) vs 25% (JEPA best) discrepancy is explainable but unverifiable per round.
4. **Experiment A CWD-set deviation:** plan §2.2 says unknown-only; `phase4_closed_loop.py:95` round-robins all 6 CWDs (known + unknown). Plan §4.4 lists known/unknown CWDs; results report does not state which were used, and no per-CWD breakdown exists (`PHASE4_RESULTS.md:41`).
5. **Experiment B design incomplete:** 3/16 cells missing in rerun (pragmatic_unknown read_hello/find_secret/read_note) (`ANALYSIS_REPORT.md:19`); the plan called for 80 episodes, 65 completed (`PHASE4_RESULTS.md:51`).
6. **Success-cache confound (paper-critical for E10/E11):** `PEDA_CONCLUSION.md:112`: `**Success cache.** One-step solves for seen state-action pairs… This cache provided the mechanism behind Phase 3's positive result: once PEDA discovers `cat hello.txt` in `/sandbox/projects`, the cache replays it instantly on subsequent episodes.` and `:68`: `combined with the success cache replaying it after the first hit`.

---

## 8. Architecture (Phase 4 components)

- `scripts/phase4_closed_loop.py` — Experiment A: 3 condition runners (`peda_train`/`peda_freeze`/`pragmatic`) sharing `WorldModel` (Qwen2.5-0.5B + LoRA via `phase2/run.py`), `EnsembleErrorComputer` (3 checkpoint ensemble), `HomeostaticDriveSystem`, `ActionGenerator` with `pragmatic_weight=3.0`; `SandboxLearningModule` with 10k buffer for block-wise updates; `BusyboxSandbox` environment.
- `scripts/phase4_jepa_experiment.py` — P4 EFE: 4 conditions over `JEPAPEDA` (frozen Qwen + 3 MLP predictors, hybrid novelty×epistemic scoring, alpha blending) vs `NoveltyExplorer` count baseline; Docker `peda-sandbox:v4`; per-episode JEPA training (`train_on_episode`, MSE across ensemble).
- `src/phase4/jepa_peda.py` — EFE = `epistemic * alpha + pragmatic * (1-alpha)` per docstring; implemented as hybrid `score = novelty * (1 + beta * ep_norm)` with beta=0.5, pragmatic via keyword task-relevance; dead-loop penalty (0.1×), action priority tie-breaks; `train_on_episode` → `JEPAEnsemble.train_step`.
- `src/phase4/__init__.py` — `"""Phase 4: JEPA-based PEDA with EFE-driven exploration."""`

**Recommended paper stance:** E10's headline (self-training 4× success) must be reported with the success-cache caveat and the `success`-field tautology; E12 (FHT-based Phase 3 replication: peda_unknown 35-40% vs pragmatic_unknown 0%, only read_hello, only /sandbox/projects) is the robust Phase 4 finding; dead-loop immunity (PEDA 0.00 vs pragmatic 0.48-0.90) is the cleanest engineering result.
