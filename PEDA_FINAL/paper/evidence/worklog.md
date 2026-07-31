# S11WorkLog — PEDA Working Log Analysis

**Scout**: S11WorkLog
**Slice**: worklog.md
**Primary source**: `/home/chillizu/Projects/Folunar_/PEDA_WORKING_LOG.md` (1824 lines, read in full; append-only log, entries 2026-07-18 → 2026-07-29 08:45)
**experiment_ids**: [E01, E02, E03, E04, E05, E06, E07, E08, E09, E10, E11, E12] — covered by the working log.
E13/E14/E18/E19 are NOT in the working log (log ends at Phase 5 launch); secondary evidence supplied from `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` and `results/phase8_gpu_run_2026-07-31.md` (flagged as such). E15/E16/E17 have no working-log record at all.

**Critical structural finding**: The working log ENDS at 2026-07-29 08:45 (Phase 5 delta-mode training launch). There are NO entries for 07-30/07-31: no Phase 5 results, no JEPA exploration (E13/E14), no Phase 6 (E15/E16), no Phase 7 (E17), no Phase 8 (E18/E19), no final conclusion. Also, `PEDA_FINAL/PEDA_CONCLUSION.md` (Main's designated ground truth) does **not exist on disk** — `PEDA_FINAL/` contains only `PEDA_RESEARCH_MANUSCRIPT.md`, `PHASE4_EXPERIMENT_PLAN.md`, and `archive/phase5_jepa_exploration/README.md`. The DISPROVEN verdict must therefore be sourced from Main's key facts + phase8 file + archive README; it is not derivable from the working log alone.

---

## 1. Files Read

- `/home/chillizu/Projects/Folunar_/PEDA_WORKING_LOG.md` (FULL, 1824 lines, chunked reads)
- `/home/chillizu/Projects/Folunar_/AGENTS.md` (already in context)
- `/home/chillizu/Projects/Folunar_/results/phase8_gpu_run_2026-07-31.md` (canonical Phase 8, 3.0KB)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md` (JEPA negative-result archive, 2.8KB)
- `/home/chillizu/Projects/Folunar_/results/phase3_sandbox_n20/ANALYSIS_REPORT.md` (formal Phase 3 stats, 10.8KB)
- `/home/chillizu/Projects/Folunar_/results/phase4b_rerun/ANALYSIS_REPORT.md` (formal Phase 4B stats, 6.3KB)
- `/home/chillizu/Projects/Folunar_/results/phase4a/PHASE4_RESULTS.md` (Phase 4A summary, 2.8KB)
- `/home/chillizu/Projects/Folunar_/results/phase1_report.md` (formal Phase 1 report, 8.4KB)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md` (superseded manuscript; structure-grep + key sections)
- Grepped (no match, provenance check): `results/phase1_partial_eval_10eps.json`, `phase1_partial_eval.json`, `phase1_partial_eval_e3_smoke.json` for "2.6"

---

## 2. Chronological Timeline (all dates from working log)

| Date | Event | Log anchor |
|---|---|---|
| 07-18 | Log created. Phase 2 blocked: WM on `text_adapter_e4` (2-room TextRoomEnv, 114 transitions), PEDA `ls data` dead loop, all 5 baselines SCR=0.2 | L89, L107 |
| 07-18 22:15 | EVAL: baseline run BLOCKED (C12/C15 violations); P0 = train sandbox LoRA adapter; 4h time box | L107 |
| 07-18 23:25 | Stop baselines; collect 200 transitions (random+heuristic, 5 tasks×20 steps); train e1 | L165 |
| 07-18 23:53 | e1 trained (loss 0.4424→0.0001); first action `ls` not `ls data`; FHT=None still | L202 |
| 07-19 00:15 | Non-fast ensemble verify: `ls↔ls data` 10 steps, FHT=None, SCR=0.1, ~10.5s/step | L245 |
| 07-19 00:41 | EVAL: problem relocated data→EFE pragmatic design | L329 |
| 07-19 01:02 | Diagnostic: no exit_code=2 in training data; 6 completion samples labeled; e2 training starts | L384 |
| 07-19 01:13 | Incident: 4 subagents overwrote the log (write vs edit); restored via `git checkout` (436 lines) | L438 |
| 07-19 01:14 | EVAL: root cause = `generate_sandbox_candidates` lacks task-completion actions | L592 |
| 07-19 01:21→02:10 | Phase 1 real-LLM on CPU: 600s timeout; profiling 52-67s/predict (later corrected 2.4-3.1s, ~16s/step) | L652, L675, L748 |
| 07-19 01:58 | Tiny grid search 6/16 combos, all success 0.00, score 0.160 | L801 |
| 07-19 03:15 | Bare 0.5B G1=0.18 (9/50), exit=0.76 | L839 |
| 07-19 19:20 | Adapter `partial_adapter_real_25_e3` G1=1.00 (50/50), exit=0.96 | L866 |
| 07-19 06:38 | Phase 1 final eval: 10/10, mean_steps=3.60, G1=1.0, G2=0.4337, G3=0.0 (memorization) | L940 |
| 07-19 10:18 | Base-model PEDA vs pragmatic-only: both 10/10 @ 3.60 steps → Phase 1 boundary | L983 |
| 07-20 | Held-out obstacle grids: 13/30 timeout (43%), completed 100% success, epistemic error 0 | L1019 |
| 07-20 | Phase 1 ARCHIVED (gates 1.0/0.4337/0.0 PASS; weights 0.5×4) | L1082 |
| 07-20 | e2 OLD candidates: `id↔ls` (worse); NEW candidates: `cat docs/note.txt` step 0, FHT=0 → P1 blocker resolved | L1128, L1167, L1210, L1229 |
| 07-20 | TextWorld integration: 6,656 unique transitions (945/2,977/2,734), 83 victory, 55 games | L1240 |
| 07-20 | C18 fix (stop-on-completion); L3 metric fixed (label→last_output): 20-sample 0.0→0.75; 40-sample 1.0/1.0/0.75 | L1281 |
| 07-21 | Scale: 600 eps / 9,840 transitions (fast baselines); PEDA+directed collectors failed on CPU; merged 610 eps / 10,040 transitions; e3 CPU training timeout; L1/L2/L3 fast-baseline 30: 1.0/0.9333/0.5667; multi-baseline: random SCR=0.180 DL=0.080, heuristic SCR=0.220 DL=0.000 | L1324 |
| 07-21 | Multi-task fix: goal_predicate + max_candidates 3→8 → all 5 tasks FHT=0, SCR=1.0; 152 stub tests pass | L1379, L1417 |
| 07-26 | AWS GPU (g4dn.xlarge T4 16GB, us-east-1b, ~4.5h, ~$2.40): e3 trained on 10,040 (loss 0.0103→0.0088→0.0087); L1/L2/L3 e2 held-out 1.0/0.9/0.55, e2 OOD 1.0/0.9/0.4 FAIL, e3 held-out 0.833/0.333/0.133 FAIL, e3 OOD 0.6/0.5/0.033 FAIL; PEDA 20/20 1-step across 4 tasks | L1462 |
| 07-27 | FINDING: v1→v2 generalization failure — e2 on sandbox v2: L1=0.800, L2=0.686, L3=0.229, read_note 0% all baselines; "Phase 2 实质上是沙箱基建 + 数据管道" | L1513 |
| 07-27 | C18 game_over guard; Phase 3 code ready but CPU-blocked (cold start ~176s) | L1529 |
| 07-27 | Phase 3 N=20 confirmatory run (g4dn.xlarge, ~5h, 80 episodes) — log claims "first significant evidence" | L1536 |
| 07-28 | Phase 4 closed-loop (i-0281f99a610497865, ~14h): Experiment A 2/10→8/10 success curve; per-episode data LOST | L1568 |
| 07-29 | Phase 4B rerun (13/16 cells, 65 eps): read_hello only; PEDA DLR 0.00; Phase 3 base rates NOT replicated | L1618 |
| 07-29 06:30 | EVAL: `success=True` is a tautology (SCR>0); fht>=0 is the true metric; Phase 3 real hits 0/20, 7/20, 7/20, 0/20; Phase 4B v4 (79 eps) replicates | L1688 |
| 07-29 07:00 | META: WM should not predict file contents; EFE structural contradiction; Layer1 works / Layer2 uncalibrated / Layer3 missing | L1775 |
| 07-29 08:45 | Phase 5: 1,378 new transitions (8 rounds random+heuristic × 4 tasks); 114 episodes merged; delta-mode training `v3_delta` launched (LOG ENDS) | L1795 |

---

## 3. Evidence Bundles (E-ID, number, source_file:line, verbatim quote)

### E01 — Phase 1 Grid World full-space (G1=1.0, G2=0.434, G3=0.0)
- (10/10 success, 3.60 mean steps) `PEDA_WORKING_LOG.md:940` — "10-episode shell eval：10/10 成功，mean_steps=3.60，revisit_rate=0.0000，G1=1.0000，G2=0.4337，G3=0.0000"
- (G2=0.4337 threshold <0.50 PASS) `PEDA_WORKING_LOG.md:1082` — "| G2 | 到达目标步数 / 随机步数 | 0.4337 | < 0.50 | PASS |"
- (G3=0.0000) `PEDA_WORKING_LOG.md:1082` — "| G3 | 回访率 | 0.0000 | < 0.20 | PASS |"
- (Pareto weights) `PEDA_WORKING_LOG.md:1082` — "curiosity=0.5, competence=0.5, boredom=0.5, novelty=0.5"
- (memorization caveat) `PEDA_WORKING_LOG.md:940` — "这些结果是 **in-distribution memorization**，不是泛化证据。adapter 在这个 5×5 grid 上训练，评估也在同一个分布上。"
- Conditions: 5×5 GridWorld, Qwen2.5-0.5B-Instruct + LoRA `partial_adapter_real_25_e3`, max-candidates=4, subprocess-isolated episodes, CPU.

### E02 — Phase 1 partial training 25% (ensemble variance ~0)
- (1920 transitions, 25% train fraction, 3 epochs) `PEDA_WORKING_LOG.md:866` — "加载 Qwen2.5-0.5B-Instruct + LoRA adapter（1920 transitions, 20 configs, 25% train fraction, 3 epochs）"
- (G1=1.0000 adapter vs 0.1800 bare; +0.82/+0.20) `PEDA_WORKING_LOG.md:866` — "G1 准确率（带 adapter）：**1.0000**（50/50）— 完美 … 与基线对比：G1 +0.8200（0.18 → 1.00），exit +0.2000（0.76 → 0.96）"
- (epistemic error 0 on held-out grids) `PEDA_WORKING_LOG.md:1019` — "**认知误差为 0** — World Model 在障碍物布局上也完全自信"
- (g1_test_set=0.8684) `PEDA_RESEARCH_MANUSCRIPT.md:342` — "this creates controlled uncertainty—the WM has g1_test_set=0.8684 on unseen state-action pairs." [manuscript, superseded — use only if corroborated]
- Conditions: 5×5 GridWorld partial training, CPU real-LLM, 50 random transitions (seed=42) for measurement.

### E03 — Phase 1 PEDA vs pragmatic N=10 (canonical: 2.6 vs 2.6, p=1.0)
- (both 10/10 @ 3.60 steps) `PEDA_WORKING_LOG.md:983` — "| 模式 | 成功 | 平均步数 | 回访率 |\n| PEDA（基础模型） | 10/10 | 3.60 | 0.0000 |\n| pragmatic-only（基础模型） | 10/10 | 3.60 | 0.0000 |"
- (goal_known 3.0 vs 3.0) `PEDA_RESEARCH_MANUSCRIPT.md:348-349` — "goal_known | PEDA | 1.000 | 3.0 | 0.000 | 1.000 … goal_known | pragmatic_only | 1.000 | 3.0 | 0.000 | 1.000"
- (goal_unknown PEDA 2.0 vs pragmatic 0%) `PEDA_RESEARCH_MANUSCRIPT.md:350-351` — "goal_unknown | **PEDA** | **1.000** | **2.0** | **0.000** | **0.500** … goal_unknown | pragmatic_only | 0.000 | 20.0 | 0.905 | 1.000"
- ⚠️ PROVENANCE GAP: canonical "2.6 vs 2.6, p=1.0" NOT found verbatim in log (3.60/3.60), manuscript (3.0/3.0, 2.0/20.0), phase1_report.md, or phase1_partial_eval JSONs (grep for "2.6" = no match). Recommend S2Phase1 locate the source (likely PHASE1_PARTIAL_EVALUATION.md, now missing).

### E04 — Phase 1.5 TextWorld 2-room (2 iterations, inventory loop)
- (2-room env, 114 transitions ceiling) `PEDA_RESEARCH_MANUSCRIPT.md:450` — "The 2-room environment's transition function has at most 114 distinct `(state, action) → next_state` mappings by combinatorial exhaustion." [manuscript]
- (17-step inventory dead loop) `PEDA_RESEARCH_MANUSCRIPT.md:466` — "the 17-step `inventory` dead loop" [manuscript]
- (text_adapter_e4 origin) `PEDA_WORKING_LOG.md:107` — "该 adapter 仅在 2 房间 TextRoomEnv 的 114 条数据上训练"
- (TextWorld 3-tier data, never trained in log) `PEDA_WORKING_LOG.md:1240` — "Generated 6,656 unique transitions across all 3 tiers (945 simple + 2,977 medium + 2,734 constrained)… Victory transitions: 83 (56 simple + 23 medium + 4 constrained)… 55 unique games generated with fixed seeds"
- ⚠️ GAP: no training/eval results for the 6,656-transition TextWorld dataset anywhere in the log.

### E05 — Phase 2 Sandbox v1 L1/L2/L3 in-distribution (1.0/0.9/0.55)
- (e2 held-out) `PEDA_WORKING_LOG.md:1462` — "e2 held-out: L1=1.000 PASS, L2=0.900 PASS, L3=0.550 PASS"
- (40-sample full run) `PEDA_WORKING_LOG.md:1281` — "Full 40-sample run: L1=1.0000, L2=1.0000, L3=0.7500 (all pass v1.1 thresholds)"
- (30-sample fast-baseline) `PEDA_WORKING_LOG.md:1324` — "L1 = 1.0000 PASS (>=0.90), L2 = 0.9333 PASS (>=0.70), L3 = 0.5667 PASS (>=0.50)"
- Conditions: busybox sandbox v1, Qwen2.5-0.5B + LoRA, adapter e2 (200 curated transitions), held-out from random/heuristic data. Caveat logged: "does not prove OOD generalization" (L1281).

### E06 — Phase 2 Sandbox v2 L1/L2/L3 OOD (0.8/0.686/0.229)
- `PEDA_WORKING_LOG.md:1515` — "**e2 adapter（最佳：v1 沙箱 L1=1.000）在 sandbox v2 新目录上**：L1=0.800（未达 0.90）、L2=0.686（未达 0.70）、L3=0.229（未达 0.50）; read_note 任务：所有基线 0% 成功率"
- (Phase 2 status correction) `PEDA_WORKING_LOG.md:1515` — "Phase 2 实质上是**沙箱基建 + 数据管道**，不是 PEDA 运行。"

### E07 — Phase 2 multi-baseline v2 (random, heuristic, PEDA)
- `PEDA_WORKING_LOG.md:1324` — "random: AvgFHT=1.0, AvgSCR=0.180, AvgDL=0.080 / heuristic: AvgFHT=1.0, AvgSCR=0.220, AvgDL=0.000 / PEDA single-episode smoke test (`read_note`): FHT=0, SCR=1.0, DL=0.0"
- (v2 sandbox size) `AGENTS.md` — "v2 sandbox: 7 directories, 14 files, 65 unique (state,action) pairs — 3.0× v1"
- Conditions: sandbox v2, e2 adapter, CPU-limited (full PEDA multi-task eval infeasible on CPU).

### E08 — Phase 3 N=20 confirmatory (MW p=0.0043, d=-1.01)
- Log entry numbers (⚠️ see Contradictions §10.1 — these disagree with the formal report):
  - `PEDA_WORKING_LOG.md:1536` — "PEDA unknown | 20 | **7.2** | p=0.4792 | — | p=**0.0043**, d=1.00" and "Pragmatic unknown | 20 | **10.0**" and "Crossover interaction: p=**0.0008** — significant" and "PEDA 2.0 vs Pragmatic 10.0 steps, p=0.0013"
- Formal report (AUTHORITATIVE):
  - (raw data + stats) `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` — "**PEDA unknown steps:** [10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2] … **Mann-Whitney U:** 130.0, **p-value (two-sided):** 0.0043, **Cohen's d:** -1.01"
  - (crossover) `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` — "Interaction Mann-Whitney U … U = 315.5, **p = 0.0001**"
  - (per-CWD) `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` — "**`/sandbox/projects`**: PEDA dramatically outperforms Pragmatic (2 vs 10 steps, p = 0.0004). All 7 episodes in this CWD completed in 2 steps."
  - Conditions: task read_hello, sandbox v2, adapter `sandbox_adapter_v2_full`, N=20×4, max_steps=10, known CWDs `/sandbox`,`/sandbox/data`,`/sandbox/docs`; unknown `/sandbox/logs`,`/sandbox/projects`,`/sandbox/tmp`; counterbalanced 7/7/6; T4 g4dn.xlarge us-east-1.
- ⚠️ CAUTION for paper: ANALYSIS_REPORT's verdict "Result: Core hypothesis is supported" (and log's "First statistically-significant evidence for the core hypothesis") was later undercut: `success=True` is tautological (E12), real hit rates are 14/80, and the final project verdict is DISPROVEN (per Main + phase8).

### E09 — Phase 3 N=20 negative control known CWDs (PEDA worse)
- `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` — "PEDA in known CWDs takes significantly **more** steps than Pragmatic in known CWDs (MW p = 0.0043, d = 1.01)"; "**PEDA known steps:** [10, 10, ...] … **Pragmatic known steps:** [1, 10, 10, 1, 10, ...]"
- (hits view) `PEDA_WORKING_LOG.md:1688` — "peda_known: 0/20（所有 cwd 全零）… pragmatic_known: 7/20（全来自 /sandbox，1-step cat hello.txt）"
- Interpretation: PEDA pays an epistemic-machinery cost in familiar envs (report §6: "This result is **not a bug**").

### E10 — Phase 4A self-training 4 blocks (20-80% success curve)
- `PEDA_WORKING_LOG.md:1568` and `results/phase4a/PHASE4_RESULTS.md` — blocks: PEDA+Train 2/10 (16.2) → 6/10 (11.0) → 8/10 (6.8) → 6/10 (14.6); PEDA+Freeze constant 2/10 (16.2) all blocks; Pragmatic 1 block only 2/10 (16.2).
- (finding) `PEDA_WORKING_LOG.md:1568` — "PEDA+Train 成功率从 2/10 升至 8/10（4x），平均步数从 16.2 降至 6.8。PEDA+Freeze 四轮完全不变… Block 4 出现回归（6/10, 14.6 步）——可能过拟合或饱和。"
- Conditions: read_hello, max 20 steps, 3 conditions × 4 blocks × N=10, adapter `sandbox_adapter_v2_full`, intermittent LoRA update; GPU i-0281f99a610497865 ~14h.
- ⚠️ Data loss: "终止实例前未拉取逐集 JSONL 数据。Phase 4A/B 的 130+ 集 per-episode 数据丢失" (L1568). Only block-level aggregates survive.

### E11 — Phase 4B multi-task 4 tasks (all zero except read_hello)
- `PEDA_WORKING_LOG.md:1688` (full 16-condition table, 79 episodes, v4) — hits: peda_unknown_read_hello 2/5 (40%), pragmatic_known_read_hello 2/5 (40%), all other 14 cells 0/5 or 0/4. Aggregate: peda_known 0/20, peda_unknown 2/20 (10%), pragmatic_known 2/20 (10%), pragmatic_unknown 0/19.
- (task gradient) `PEDA_WORKING_LOG.md:1688` — "count_lines / find_secret / read_note 全零。当前 WM 无法解这三种任务——不是实验设计问题，是能力边界。"
- (DLR contrast) `PEDA_WORKING_LOG.md:1688` — "Dead-loop rate: PEDA 0.00（从未死循环），Pragmatic 0.48-0.80（频繁 ls ↔ ls data 振荡）"
- Conditions: 4 tasks × 2 baselines × 2 conditions, N=5, max_steps=10 (v4, ensemble mode), sandbox v2, e2-family adapter, GPU.

### E12 — Phase 4B Phase 3 replication fht metric
- (tautology discovery) `PEDA_WORKING_LOG.md:1688` — "`phase3_sandbox_experiment.py:132`: `"success": metrics["scr"] > 0`。SCR = 去重状态数/步数。max_steps=10、agent 访问 >=2 个目录时 SCR >= 0.2，`success` 始终 True。"
- (replication table) `PEDA_WORKING_LOG.md:1688` — "| peda_known read_hello | **0/20** | **0/5** | … | peda_unknown read_hello | **7/20 (35%)** | **2/5 (40%)** | … | pragmatic_known read_hello | **7/20 (35%)** | **2/5 (40%)** | … | pragmatic_unknown read_hello | **0/20** | **0/5** |"
- (conclusion) `PEDA_WORKING_LOG.md:1688` — "**Phase 4B v4 完全复现 Phase 3，无退化，无翻车。**" and "P0：修正所有历史文档中的 `success` 字段解释（fht>=0 才是真指标）"
- (rerun report corroboration) `results/phase4b_rerun/ANALYSIS_REPORT.md` — "All 65 episodes report `success=True`… **The real success metric is FHT >= 0.**"; read_hello PEDA unknown 2/5 FHT=1, p=0.1770 (ns); "Phase 3 base rates not replicated (possible checkpoint/sandbox mismatch)".
- ⚠️ Note: rerun report's Phase 3 comparison table cites "PEDA known steps 6.8" — propagating the log's incorrect Phase 3 number (see §10.1).

### E13 / E14 — Phase 5 JEPA (NOT in working log; from archive README)
- `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` — "Across 11 experiments spanning 4 sandboxes (v2/v3/v4 grid maze deterministic/stochastic), JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**."
- Table: "| Sandbox v2 | 65 | 50% | 50% (hybrid) | — | … | Maze 10x10 | 1,100 | 100% | 0% | — | … | Stochastic 10x10 | 1,100 | 100% | 67% | 0% |" ; "| P4 EFE (4 rounds) | 65-270 | 50% | 25% | — |"
- (STRIPS) "45.8% learned vs 31.3% fallback"; (scaling) "8400 states still too small for epistemic advantage"; (root cause) "Its uncertainty is 'how uncertain am I about this transition?' All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower."
- ⚠️ Canonical E13's "novelty 50% > jepa_efe 17%" — the 17% jepa_efe figure is NOT verbatim in the README (README shows 50% hybrid / 8% v4 / 25% P4). Recommend S7Phase5 pin the 17% sub-config source. Date: 2026-07-30, ARCHIVED — negative result.

### E18 / E19 — Phase 8 count-driven (NOT in working log; from phase8 file)
- `results/phase8_gpu_run_2026-07-31.md` — count-only "**28/45 (62.2%)**" with per-task table (read_hello 5/5 1.2, read_note 1/5 8.8, count_lines 0/5, find_secret 5/5 1.6, read_welcome 5/5 1.4, find_api_key 1/5 10.0, count_measurements 5/5 1.6, find_errors_v4 1/5 10.0, read_changelog_v4 5/5 1.2); count+JEPA identical "**28/45 (62.2%)**"; "| Total success | 28/45 (62.2%) | 28/45 (62.2%) | **0** | … **Conclusion**: JEPA forward dynamics training contributes zero additional value… consistent with 17 prior JEPA experiments".
- Conditions: Phase8Runner count-driven agent (no prediction error), g4dn.xlarge i-06b0ba3dbdc214761, Qwen2.5-0.5B (CPU-only, unused for count run), docker peda-sandbox:v2/v4, 9 tasks × 5 eps × max_steps 10, commit a348c1e.

### E15/E16/E17 — NO working-log record; results files exist (not analyzed in this slice):
`results/phase6_maze_count_5x5_seed42.jsonl` (28KB), `phase6_maze_jepa_all_modes_5x5_seed42.jsonl`, `phase6_maze_jepa_pure_novelty_5x5_seed42.jsonl`, `phase6_maze_jepa_jepa_only_5x5_seed42.jsonl`, `phase6_maze_count_10x10_seed42.jsonl` (0B!), `phase7_rssm_*_5x5_seed42.jsonl`, `phase7_giant_*.jsonl`. Recommend S8Phase6/S9Phase7 analyze these.

---

## 4. Key Decisions (what, when, rationale)

1. **07-18 22:15 — Halt all baseline comparisons; P0 = train sandbox adapter** (L107). Rationale: WM untrained on sandbox ⇒ EFE meaningless (WATCHDOG C12/C15); "此时跑 `--all-baselines` 不是对比，是同步失败".
2. **07-18 23:59 — P0 closed with e1; P1 blocked on ensemble verify or 500+ data** (L277).
3. **07-19 00:41 — Problem relocated from data volume to EFE pragmatic design** (L329): "继续扩数据是 WATCHDOG B8 的变体".
4. **07-19 01:14 — Root cause = candidate action space missing task-completion actions** (L592); fix `generate_sandbox_candidates` before anything else.
5. **07-19 06:38 — Phase 1 NOT formally validated** (L940): "不将此视为 Phase 1 正式验证"; adopt max-candidates=4 + subprocess isolation as standard.
6. **07-19 10:18 — Phase 1 boundary declared** (L983): 5×5 Grid World cannot measure prediction-error-driven exploration.
7. **07-20 — P1 blocker fix approved via negative-control ablation** (L1210, L1229): OLD candidates `id↔ls` vs NEW candidates step-0 solve proves causality ("proved fix is causal, not just more data").
8. **07-20 — C18 stop-on-completion adopted** (L1281) to kill post-goal oscillation.
9. **07-20 — L3 metric redefined** (L1281): compare predicted `last_output` vs actual (token overlap ≥0.5), not generic action labels.
10. **07-21 — Goal predicate per task replaces global exit_code==2** (L1379): WM was biased to predict `cat docs/note.txt` completes everything; max_candidates 3→8.
11. **07-21 — Defer full retraining to GPU** (L1324): CPU-only PyTorch too slow; e2 (200 curated) remains the verified WM.
12. **07-26 — "Data quality > data quantity"** (L1462): e3 (10,040 random+heuristic) regressed vs e2 (200 curated); random data dilutes completion signal.
13. **07-27 — Phase 2 v2 declared infra+data-pipeline, not PEDA runtime** (L1515).
14. **07-29 06:30 — `fht>=0` is the only valid success metric; correct all historical docs** (L1688).
15. **07-29 07:00 — WM must predict structural effects (cwd/files/exit), NOT file contents** (L1775); EFE pragmatic-vs-predicted-content is a structural contradiction.
16. **07-29 08:45 — Delta-mode prediction adopted (predict change, not full state)** (L1795); expand data to 500+; test `cd ..` emergence.

## 5. Bugs Found (what, when, impact)

| Bug | When | Impact | Log |
|---|---|---|---|
| PEDA `ls data` dead loop (untrained WM) | 07-18 | Blocked Phase 2 P0 | L89, L107 |
| Prompt baseline strips command text | 07-18 | Prompt baseline invalid | L165 |
| Ensemble checkpoint path hardcoded to `text_adapter_e4` | 07-19 | Ensemble silently used WRONG adapter | L245 |
| Training data lacks exit_code=2 labels (only 6/200 marked) | 07-19 | pragmatic reward flat at 0.5 | L384 |
| Subagents overwrote log with `write` (400+ lines lost) | 07-19 01:13 | Audit trail destroyed; restored via git | L438 |
| `config median_ms=4750` stale (~10× off) | 07-19 | horizon forced to 1 always | L675, L748 |
| `max-candidates=2` → only UP/DOWN actions | 07-19 | Vertical oscillation, zero success | L940 |
| Single-process multi-episode real-LLM hangs | 07-19/20 | 13/30 timeouts; subprocess isolation required | L940, L1019 |
| Candidate set lacks task-completion actions; `id` not whitelisted | 07-20 | `id↔ls` worse oscillation | L1128 |
| L3 metric compared wrong field (action label vs last_output) | 07-20 | L3=0.0 → 0.75 after fix | L1281 |
| PEDAData subagent hang on CPU; pragmatic ~23 min/step; prompt arg-stripping (`cat` alone) | 07-21 | 2/3 collectors failed | L1324 |
| WM predicts `cat docs/note.txt` exit=2 for ALL tasks | 07-21 | PEDA repeated read_note action everywhere; fixed by goal_predicate | L1379 |
| max_candidates=3 truncates task actions | 07-21 | Fixed 3→8 | L1379 |
| C18 post-completion oscillation | 07-27 | game_over guard added | L1529 |
| GPU-side `phase2/run.py` success detection missing; adapter checkpoint transfer incomplete | 07-28 | Phase 4 experiments broken; `--fast` workaround | L1568 |
| `success=True` tautology (`scr>0`) | 07-29 | Phase 3 "100% success" overstated; real hits 14/80 | L1688 |
| WM uncalibrated (high confidence, low accuracy) | 07-29 | peda_known < peda_unknown paradox | L1688, L1775 |
| JEPA-track bugs (5): cwd-unaware path predicate; alphabetical novelty tie-break; `final_state.victory` always False; DLR 0.53-0.80; cross-task predictor contamination | 07-30 (archive) | All fixed pre-archive | archive README |

## 6. Phase Transitions (trigger events)

- **Phase 1 → 1.5**: G1=1.0 memorization + base-model equivalence (07-19 10:18 boundary; 07-20 archive). "Phase 1 形式目标已达成，但核心机制验证需要更复杂的环境" (L1082).
- **Phase 1.5 → 2**: 2-room env insufficient — 114-transition combinatorial ceiling; `text_adapter_e4` useless on sandbox (L107; manuscript §5.1).
- **Phase 2 → 3**: P1 blocker cleared (07-20 candidate fix) + 20/20 multi-task (07-21) + e2 adapter; Phase 3 scripts ready 07-27 (L1529).
- **Phase 3 → 4**: Phase 3 N=20 crossover (07-27/28) — log: "First statistically-significant evidence for the core hypothesis" (L1536).
- **Phase 4 → 5**: 07-29 07:00 conceptual clarity (WM prediction scope; delta mode) → Phase 5 launched 07-29 08:45.
- **Phase 5 → 6/7/8**: NOT in log. Archive README (07-30) records Phase 5 JEPA as ARCHIVED negative → scale-up tracks (Phase 6 maze, Phase 7 RSSM 5-track, Phase 8 count-driven). Timeline gap 07-29→07-31.

## 7. GPU Usage (instance launch/destroy dates)

| Instance | Date | Duration | Purpose | ID logged? | Destroy |
|---|---|---|---|---|---|
| g4dn.xlarge T4 16GB, us-east-1b (on-demand) | 07-26 | ~4.5h, ~$2.40 | e3 training + final L1/L2/L3 + PEDA 5-ep eval | NO (env only) | not logged |
| g4dn.xlarge T4 | 07-27 | ~5h | Phase 3 N=20 (80 eps) | NO | not logged |
| i-0281f99a610497865, g4dn.xlarge T4 16GB | 07-28 | ~14h | Phase 4 closed-loop | YES | destroyed BEFORE data pull → 130+ eps lost; lesson "永远先拉数据再关 GPU 实例" (L1568) |
| (GPU, unlogged) | 07-29 | — | Phase 4B rerun; Phase 5 data collection (1,378 trans) + delta training | NO | not logged |
| i-06b0ba3dbdc214761, g4dn.xlarge T4 16GB, IP 13.220.38.201 | 07-31 | — | Phase 8 count-driven (9 tasks × 5 eps × 2 modes) | YES (phase8 file) | not logged |

Only 2 of 5 GPU runs have instance IDs; NO destroy timestamps anywhere. Phase 8 file also notes "peda-sandbox:v2, peda-sandbox:v4 (rebuilt post-reboot)" and commit a348c1e.

## 8. Experimental Conditions & Methodology (implementation details)

- **Model**: Qwen2.5-0.5B-Instruct everywhere; LoRA adapters (peft), base frozen. CPU-only until 07-26 (Intel Arc not usable by PyTorch; L675).
- **Phase 2 training**: `phase2_synthetic_train.py`; batch 4, 3 epochs, max_length=384 (sandbox JSON states); ensemble = 3 epoch checkpoints; `--delta` mode added 07-29 (predict `cwd_changed, new_cwd, exit, files_created/deleted, output_summary`).
- **EFE (Phase 2)**: `pragmatic = 0.0 if final_exit == 2 else 0.5` × pragmatic_weight=3.0; `diversity_bonus` 0.2 for actions unseen in last 10 steps; horizon=1 (latency-budget forced); drive weights cur=0.1/cmp=2.0/bor=0.1/nov=2.0 default (Phase 1), recommended 0.5×4; goal_predicate per task (07-21) checks task output instead of exit_code.
- **Candidate generation**: cap 8 candidates, whitelist-filtered; task shortcuts added 07-20 (`cat docs/note.txt`, `cat note.txt`, `wc -l data/lines.txt`, `grep -r secret data`, `mkdir test_dir`); `id` removed (not whitelisted).
- **Data**: Phase 2 random/heuristic 5 tasks × 20 steps → 200 (e1/e2, 6 exit=2 labeled); scaled 610 eps / 10,040 transitions (e3); v2 sandbox systematic 65 (s,a) pairs; Phase 5 1,378 new + 65 old = 114 episodes.
- **Phase 3**: N=20×4, read_hello, max_steps=10, disjoint known/unknown CWD sets counterbalanced 7/7/6, adapter `sandbox_adapter_v2_full`, per-CWD analysis; metrics: steps_count (DV), dead-loop rate.
- **Phase 4A**: 3 conditions × 4 blocks × N=10, read_hello, max 20 steps, intermittent LoRA self-training vs frozen control vs pragmatic; block-level aggregates only (data loss).
- **Phase 4B**: 4 tasks × 2 baselines × 2 conditions × N=5; rerun max_steps=20; v4 max_steps=10 ensemble mode; FHT (first hit time) metric.
- **Phase 8**: count-driven agent (novelty = visited-state count), 9 tasks × 5 eps × max 10 steps, docker v2/v4; JEPA ON/OFF as side-effect training.
- **Eval harness**: subprocess isolation per episode (Phase 1); `timeout -s KILL 240`; stub mode (FOLUNAR_STUB_MODEL=1) for pipeline tests; 152 (later 138→152) stub tests.

## 9. Verbatim Conclusions (from source docs)

- (Phase 1) `PEDA_WORKING_LOG.md:983` — "**5×5 Grid World 无法衡量 PEDA 的预测误差驱动探索机制**。该环境太简单，exit code / goal distance 信息已足够导航。"
- (Phase 1 held-out) `results/phase1_report.md` — "When the World Model is certain, PEDA and pragmatic-only produce identical behavior. The Grid World remains too simple to exercise prediction-error-driven exploration."
- (Phase 2 v2) `PEDA_WORKING_LOG.md:1515` — "Phase 2 的'成功'声明（L1=1.000, 20/20 多任务完成）仅在 v1 沙箱（4 目录）上成立… Phase 2 实质上是**沙箱基建 + 数据管道**，不是 PEDA 运行。"
- (Phase 2 multi-task) `PEDA_WORKING_LOG.md:1417` — "完成机制主要依赖任务完成动作进入候选集并被 goal predicate 识别，而非预测误差驱动探索。"
- (Phase 3, later superseded) `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` — "**Yes.** PEDA in unknown CWDs requires significantly fewer steps than the Pragmatic baseline (p = 0.0043, d = -1.01)." / "Phase 3 provides strong confirmatory evidence for the epistemic validation hypothesis."
- (Phase 3 corrected) `PEDA_WORKING_LOG.md:1688` — "PEDA 的显著优势是 peda_unknown (7/20) vs pragmatic_unknown (0/20)，p=0.0043。这是在 unknown 环境中 epistemic uncertainty 驱动探索的胜利，而非 known 环境中的效率优势。" + "当前 WM 是 **未校准的**（high confidence, low accuracy）"
- (Phase 4A) `results/phase4a/PHASE4_RESULTS.md` — "**Core finding: Self-training works.** … This closes the last open question from Phase 3: epistemic signal not only guides exploration, it can drive autonomous improvement through intermittent learning."
- (Phase 4B) `results/phase4b_rerun/ANALYSIS_REPORT.md` — "**Does epistemic advantage generalize?** Inconclusive — the experiment failed to replicate Phase 3's baseline performance." / "PEDA never dead-loops (dead_loop_rate=0.00 everywhere vs 0.54-0.90 for pragmatic)"
- (Phase 8) `results/phase8_gpu_run_2026-07-31.md` — "JEPA forward dynamics training contributes zero additional value to the count-driven agent. Every task's success/failure pattern is identical with and without JEPA. This is consistent with 17 prior JEPA experiments where learned forward dynamics never improved exploration or task completion over count-based novelty."
- (JEPA archive) `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` — "JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**." / "All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower."
- (Old manuscript conclusion, SUPERSEDED) `PEDA_RESEARCH_MANUSCRIPT.md:639` — "After three experimental phases, the answer is nuanced:" — do NOT cite as final.

## 10. Contradictions vs Formal Reports

1. **Phase 3 log table vs Phase 3 ANALYSIS_REPORT (same raw data, different numbers)**:
   - PEDA known mean steps: log 6.8 (L1536) vs formal 10.00 (deterministic [10×20], ANALYSIS_REPORT §2/§6). 
   - Pragmatic known: log 7.2 vs formal 6.85.
   - Crossover p: log 0.0008 vs formal 0.0001 (U=315.5).
   - Driver p: log 0.0013 vs formal 0.0004.
   - d: log 1.00 vs formal -1.01 (sign convention, same effect).
   - The log's Phase 3 entry is the only source of the 6.8/7.2/7.2/10.0 table; the formal report of the same data says 10.0/7.2/6.85/10.0. The error propagated into `results/phase4b_rerun/ANALYSIS_REPORT.md` ("PEDA known steps 6.8"). **Use ANALYSIS_REPORT.md numbers; treat log table as errata.**
2. **Phase 3 "100% success / core hypothesis supported" vs final verdict**: Both the log (L1536: "First statistically-significant evidence for the core hypothesis"; "All 80 episodes: 100% success rate") and ANALYSIS_REPORT ("Result: Core hypothesis is supported") predate the 07-29 06:30 correction: `success=True` is a tautology; real hit counts 0/20, 7/20, 7/20, 0/20. The paper must present Phase 3 as *steps-based evidence with a later metric correction*, and reconcile with the DISPROVEN final verdict (count-based novelty ≥ all learned signals).
3. **E03 canonical "2.6 vs 2.6, p=1.0"**: not found in log (3.60 vs 3.60), manuscript (3.0/3.0 known; 2.0/20.0 unknown), or any grep'd JSON. Unresolved provenance.
4. **AGENTS.md stale**: "Current phase: Phase 2" and file index (RESEARCH_CHARTER.md, peda_report_v11, reflection, review, README_FOR_AGENTS, CONTROLLER_DIRECTIVE_PHASE2) — none of those files exist on disk now; actual phase reached Phase 8 (07-31). AGENTS.md L1/L2/L3 "1.000/0.900/0.550" matches the 07-26 e2 held-out numbers (consistent).
5. **Log internal**: 07-19 01:50 "52-67s/predict" vs 07-19 02:10 "2.4-3.1s/predict" — resolved as thermal throttling/memory swap; the 01:50 entry's "10× latency" claim is superseded by its own correction.
6. **Old manuscript conclusion** ("answer is nuanced", §9) contradicts the final DISPROVEN verdict — manuscript is SUPERSEDED; use only Theory (§2) / Architecture (§3) per Main.

## 11. Gaps / Claims NOT in Formal Reports

1. **No working-log coverage of 07-30/07-31**: Phase 5 results (v3_delta), JEPA exploration (E13/E14), Phase 6/7 (E15/E16/E17), Phase 8 (E18/E19), and the final conclusion are entirely absent from the log. The paper's Phase 5-8 timeline has no append-only-log grounding.
2. **PEDA_CONCLUSION.md missing on disk** — ground truth file referenced by Main does not exist in PEDA_FINAL/; cannot be cross-checked.
3. **TextWorld 6,656-transition dataset (E04)**: collected 07-20, but NO training or evaluation results logged; canonical E04 refers to the older 2-room TextRoomEnv (114 samples, 17-step inventory loop). The 3-tier TextWorld data outcome is unknown from the log.
4. **Phase 4A per-episode data lost** (130+ episodes) — only block-level aggregates; Experiment A has no raw data for formal stats; Pragmatic arm ran only 1 block.
5. **Phase 5 delta adapter (v3_delta) results missing** — log ends at training launch; no checkpoint/results verification on disk found in this slice.
6. **E10 note**: Phase 4A "4× improvement" (2/10→8/10) with Block-4 regression (6/10) — the regression's cause (overfitting vs saturation) never resolved.
7. **GPU cost/IDs incomplete**: only e3 run cost logged (~$2.40); 3 of 5 instances have no ID; zero destroy timestamps.
8. **Metric-validation lesson (C23)**: the `success=True` tautology correction (07-29 06:30) exists only in the log — the paper's metrics section should cite it as the reason `fht>=0` is canonical.
9. **"Data quality > quantity"** (e2 200-curated > e3 10,040-random, 07-26) is logged but appears in no formal report; important for the paper's data methodology.
10. **JEPA archive's 5 fixed bugs** and the "37× slower" root-cause analysis exist only in the archive README — no formal report documents them.
11. **`create_file` task**: excluded from the 20/20 success claim (only 4 tasks listed); read-only sandbox is a "设计取舍，非 bug" (L1462) — clarify in paper that multi-task claim = 4 tasks + create_file limit.
12. **E13 sub-config detail** ("jepa_efe 17%") not verbatim in archive README — needs S7Phase5 to locate.

---

*Prepared 2026-07-31 by S11WorkLog. All line numbers refer to the current on-disk versions of the cited files.*
