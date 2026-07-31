# PEDA Paper — Claims vs Reproducible Evidence Cross-Reference

**Generated**: 2026-07-31
**Source**: 11 scout reports + PEDA_CONCLUSION.md + results/* files
**Purpose**: Before any paper sentence is written, every claim must exist in a "REPRODUCIBLE" row below. Claims with no reproducible source are downgraded or excluded.

---

## E01-E04: Phase 1 + 1.5 (Grid World + TextWorld)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E01 | G1=1.0000, G2=0.4337, G3=0.0000 | YES | results/phase1_eval.json | Use as-is |
| E01 | 10/10 success, 3.6 steps vs random 8.3 | YES | phase1_eval.json, phase1_gap_report.md:12-28 | Use as-is |
| E02 | 1-epoch g1_test_set=0.8684 | YES | results/phase1_partial_eval_10eps.json | Use as-is |
| E02 | 3-epoch g1_test_set=1.0, 2/28 probe disagreement | YES | phase1_epistemic_blocker_report.md:5-9 | Use as-is, fix "28/28 zero variance" → "2/28" |
| E03 | PEDA 2.6 vs Pragmatic 2.6, Fisher p=1.0, MW p=1.0 | YES (but wrong phase) | results/phase3_gpu/report.json (GPU rerun, NOT Phase 1 archive) | Cite as E03-GPU, note: confidence-based epistemic, ensemble_checkpoints=0 |
| E03 | CPU pilot: PEDA 16.6 vs Prag 21.1 (goal_unknown) | YES | phase1_partial_eval_10eps.json | Cite as directional only (no p-values, N=3→10) |
| E03 | Working log "3.60 vs 3.60" | CONFLICT | PEDA_WORKING_LOG.md:L381-L386 | Resolve: log errata; use report.json 2.6/2.6 |
| E04 | No p-values exist for Phase 1.5 | MISSING | deviation_report.md:65 explicitly states unknown significance | Do not cite any p-value for Phase 1.5 |

## E05-E07: Phase 2 (Sandbox Infrastructure)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E05 | L1=1.000, L2=0.900, L3=0.550 held-out (e2) | YES | results/phase2_l1l2l3_baseline_full.json, WORKING_LOG.md:L1221-L1225 | Use as-is; note IN-DISTRIBUTION |
| E05 | e3 regressed: L1=0.833, L2=0.333, L3=0.133 | YES | WORKING_LOG.md:L1241-L1243 | Use as-is; note: data quality > quantity |
| E05 | 20/20 multi-task completions | YES | WORKING_LOG.md:L1246 | Use as-is; note: 1-step, success-cache driven |
| E06 | L1=0.800, L2=0.686, L3=0.229 OOD V1→V2 | YES | results/phase2_remaining/ (35 OOD samples) | Use as-is; FAIL all thresholds |
| E06 | Pre-fix L3=0.0 vs post-fix L3=0.75 (20 samples) | YES (partial) | phase2_l1l2l3 artifacts | Use as-is; 40-sample artifact shows 0.0 — note discrepancy |
| E07 | read_hello: Pragmatic 1.0s > PEDA 2.8s | YES | phase2_multi_baseline_aggregate.json | Use as-is |
| E07 | read_note: ALL 0% success | YES | phase2_multi_baseline_aggregate.json | Use as-is |
| E07 | Random AvgSCR 0.18, Heuristic 0.22 | YES | phase2_multi_baseline_aggregate.json | Use as-is; NO PEDA row in aggregate |

## E08-E09: Phase 3 (N=20 Confirmatory)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E08 | MW p=0.0043, d=-1.01 (PEDA unknown vs Pragmatic unknown) | YES | ANALYSIS_REPORT.md, phase3_n20_result.json | Use as-is |
| E08 | Crossover p=0.0001 | YES | ANALYSIS_REPORT.md | Use as-is; MUST include caveat: only /sandbox/projects |
| E08 | Per-CWD: projects 2.0 vs 10.0 (p=0.0004) | YES | ANALYSIS_REPORT.md per-CWD table | Use as-is |
| E08 | Per-CWD: logs/tmp both 10.0 vs 10.0 | YES | ANALYSIS_REPORT.md | Use as-is; zero advantage outside projects |
| E08 | "success=True for all 80 episodes" (scr>0 tautology) | YES (bug) | phase3_sandbox_experiment.py:132 | Must state: success field tautological; real fht pass = 14/80 |
| E08 | Log table p-values differ from formal report | CONFLICT | WORKING_LOG.md vs ANALYSIS_REPORT.md | Resolve: log is errata; use ANALYSIS_REPORT.md |
| E09 | PEDA known 10.0 vs Pragmatic known 6.85, p=0.0043 | YES | ANALYSIS_REPORT.md | Use as-is; PEDA pays cost in familiar envs |

## E10-E12: Phase 4 (Closed-Loop)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E10 | PEDA+Train: 20%→80% success over 4 blocks | PARTIAL | PHASE4_RESULTS.md (per-episode JSONL LOST) | Cite from report; flag as "per-episode raw data not preserved" |
| E10 | PEDA+Freeze: flat 20% | PARTIAL | Same source | Same flag |
| E11 | Only read_hello has hits; count_lines/find_secret/read_note all 0/5 | YES | phase4b_rerun/ANALYSIS_REPORT.md + raw JSONL | Use as-is |
| E11 | PEDA dead_loop_rate 0.00 vs Pragmatic 0.54-0.90 | YES | Raw JSONL DLR fields | Use as-is |
| E12 | v4 replicates Phase 3: peda_unknown 40% (2/5) | YES | phase4b_v4/ raw JSONL | Use as-is; small N (5), treat as directional confirmation |
| E12 | success=True tautology corrected to fht metric | YES | phase3_sandbox_experiment.py:132 | Document as critical methodology bug |

## E13-E14: Phase 5 (JEPA + Action Model)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E13.01-E13.07 | 7 sub-configs: JEPA never beats count | YES | archive/phase5_jepa_exploration/README.md table | Use as-is |
| E13.08-E13.11 | 4 sub-configs: raw JSONL deleted | MISSING | No on-disk source | Exclude from paper or cite as "unrecoverable" |
| E13 | JEPA loss curve 45→15 | YES (doc claim) | README.md:27, CONCLUSION.md:90 | Cite from README; no raw loss log |
| E13 | "37x slower" | UNVERIFIED | CONCLUSION.md:90,124 | Downgrade: no benchmark; cite as "estimated, unbenchmarked" |
| E14 | JEPA pure epistemic SCR ~0 | YES | CONCLUSION.md:54 | Use as-is |
| STRIPS | 45.8% learned vs 31.3% fallback | PARTIAL | CHARTER.md:17,35; raw trace missing | Cite as "candidate hit rate on action prediction task"; flag: raw trace not preserved |
| STRIPS | v3 episode JSONL: learned 0/48 vs fallback 14/48 (29.2%) | DIFFERENT METRIC | phase5 v3 surviving file | Must NOT conflate with 45.8% — different measurement unit |
| Bugs | 5 JEPA bugs with before/after | YES | README.md:57-61 + code line anchors | Use as-is |

## E15-E16: Phase 6 (Grid Maze)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E15 | Count 100% at 10x10 (1100 states) | PARTIAL | 10x10 count JSONL = 0 bytes; numbers survive only in docs | Downgrade: "5x5 raw data confirms count 100%; 10x10 doc-claimed" |
| E15 | JEPA 0% at 10x10 | PARTIAL | No 10x10 JEPA file; 5x5 JEPA 0% (scr 0.05, DLR 0.9) | Cite 5x5 raw; 10x10 from docs with caveat |
| E15 | Hybrid 67% — CONCLUSION.md says deterministic, README says stochastic | CONFLICT | CONCLUSION.md:55 vs archive README:15/17 | Resolve: attribute to deterministic maze (matching script expectation) |
| E16 | Both 0% at 20x20 (8400 states) | MISSING | No 20x20 raw data | Downgrade: "per archive README; raw data not preserved" |
| State counts | 1100/8400 don't match GridMaze.state_estimate() (3400/53600) | CONFLICT | maze_generator.py:137-141 | Do not cite state counts; cite maze size (10x10, 20x20) instead |
| Stochastic | Count 100% on stochastic maze contradicts script expectation of 0% | CONFLICT | phase6_stochastic_count.py:6 vs README:17 | Note contradiction; raw data missing |

## E17: Phase 7 (GPU 5-Track)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E17 | All JEPA tracks DLR ~0.996 | NO — raw JSONL shows 0.8-0.9 | phase7_rssm_all_modes_5x5_seed42.jsonl | DOWNGRADE: "DLR 0.8-0.9 (high determinism, low exploration)" — 0.996 NOT reproducible |
| E17 | Count wins every track | YES (partial) | RSSM count SCR 0.4 (5x5); other tracks no data | Cite only RSSM comparison; other tracks have no persisted results |
| E17.2-E17.4 | Goal-JEPA, Curriculum, Random-Maze: NO result files | MISSING | No on-disk JSONL | Exclude from quantitative claims; cite as "code complete, results not preserved" |
| E17.3 | Giant-JEPA rows entirely missing | MISSING | phase7_giant_all.jsonl = count-only, identical to giant_20x20.jsonl | Exclude |
| E17.1 | RSSM used single-model prior variance, not ensemble-of-3 | METHODOLOGY MISMATCH | phase7_rssm_experiment.py:203-210 vs rssm_wm.py:10 docstring | Document deviation |

## E18-E19: Phase 8 (Count-Driven Agent)

| ID | Conclusion Claim | Reproducible? | Actual Source | Action |
|:---|------------------|:---:|------|------|
| E18 | 28/45 (62.2%) across 9 tasks | YES | results/phase8_gpu_run_2026-07-31.md:23-32 | Use as-is |
| E19 | Count+JEPA = identical (zero delta) | YES | phase8_gpu_run_2026-07-31.md:47-64 | Use as-is |
| E18 | Direct reads 100% (5/9 tasks) | YES | Same source:36-38 | Use as-is |
| E18 | Deep path reads 20% (3/9 tasks) | YES | Same source | Use as-is |
| E18 | count_lines 0% | YES | Same source | Use as-is |
| E18 | "v2 sandbox" in CONCLUSION.md:106 | ERROR | Run used v2 (4 tasks) + v4 (5 tasks) | Fix: "v2+v4 sandbox" |

---

## Summary

| Status | Count |
|--------|:----:|
| REPRODUCIBLE (use as-is) | 31 |
| PARTIAL (cite with caveat) | 9 |
| MISSING (exclude from paper) | 7 |
| CONFLICT (resolve before use) | 6 |
| METHODOLOGY MISMATCH | 1 |
| ERROR in conclusion doc | 1 |
| DOWNGRADED (0.996→0.8-0.9) | 1 |
