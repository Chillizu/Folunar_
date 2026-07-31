# Phase 1 (Grid World) Evidence Bundles — S2Phase1 scout

experiment_ids: [E01, E02, E03]

Scout: S2Phase1 | Date: 2026-07-31 | Scope: ALL Phase 1 Grid World files (PEDA_FINAL/archive/phase1/, results/phase1*, src/phase1/, scripts/*phase1*)

---

## 1. E01 — Full-space training, G1/G2/G3 formal gates (all PASS on training distribution)

### 1.1 Canonical gate values (10-episode real-LLM run, 2026-07-19)

- (1.0000, results/phase1_eval.json, `"g1_accuracy": 1.0`) — G1 next-state accuracy
- (0.4337349397590361, results/phase1_eval.json, `"g2_ratio": 0.4337349397590361`) — G2 drive/random steps ratio
- (0.0, results/phase1_eval.json, `"revisit_rate": 0.0`) — G3 revisit rate
- (3.6, results/phase1_eval.json, `"mean_steps": 3.6`) — drive agent mean steps
- (8.3, results/phase1_eval.json, `"random_mean_steps": 8.3`) — random baseline mean steps
- (10, results/phase1_eval.json, `"episodes": 10`) — N
- (1.0, results/phase1_eval.json, `"success_rate": 1.0`) — 10/10 episodes succeeded
- Gate thresholds (from script, same values everywhere): G1 > 0.90, G2 < 0.50, G3 < 0.20 — (scripts/phase1_eval.py, `"G1 = {g1:.4f}  (target > 0.90)  {'PASS' if g1 > 0.90 else 'FAIL'}"` and `"G2 = {g2:.4f}  (target < 0.50)"` / `"G3 = {g3:.4f}  (target < 0.20)"`)
- (0.5/0.5/0.5/0.5, results/phase1_eval.json, `"drive_weights": {"curiosity": 0.5, "competence": 0.5, "boredom": 0.5, "novelty": 0.5}`) — recommended weights
- Per-episode steps all 1–6, 10/10 success — results/phase1_shell_eval.jsonl (e.g. `{"seed": 0, "success": true, "steps": 4, ... "actions": ["RIGHT", "UP", "RIGHT", "UP"]}`)

### 1.2 Earlier 20-episode run (2026-07-03, adapter synthetic_adapter) — G2 differs

- (1.0000, PEDA_FINAL/archive/phase1/phase1_validation_report.md:70, `| G1 | 1.0000 | > 0.90 | PASS |`)
- (0.1211, PEDA_FINAL/archive/phase1/phase1_validation_report.md:71, `| G2 | 0.1211 | < 0.50 | PASS |`)
- (0.0000, PEDA_FINAL/archive/phase1/phase1_validation_report.md:72, `| G3 | 0.0000 | < 0.20 | PASS |`)
- (3.3, PEDA_FINAL/archive/phase1/phase1_validation_report.md, §4.1 `| Mean steps | 3.3 | 27.25 |` drive vs random) — 20/20 success vs 14/20 random; Completion@20 1.000 vs 0.450
- (0.0101, PEDA_FINAL/archive/phase1/phase1_validation_report.md:44, `Average loss: 0.0101`) — training loss; `loss 0.5638 → 0.0000 (avg 0.0101)`

NOTE: canonical E01 numbers per PEDA_CONCLUSION.md are G1=1.000, G2=0.434, G3=0.000 (the 2026-07-19 10-ep run); the 0.1211 G2 belongs to the earlier 20-ep run.

### 1.3 Verbatim conclusions

- (PEDA_CONCLUSION.md:41, `| 1 | Grid World 5x5 | Full-space training, G1/G2/G3 eval | Q1 | G1=1.000, G2=0.434, G3=0.000 — WM perfectly memorizes all 25 cells | FAIL — environment too simple for 0.5B model |`)
- (results/phase1_report.md, `## Critical Caveats` / `1. **In-distribution memorization, not generalization.** The adapter was trained on this exact 5×5 Grid World; evaluation uses the same distribution.`)
- (results/phase1_report.md, `2. **G1=1.0 trivializes the other gates.** When the World Model predicts perfectly, the pragmatic term dominates Expected Free Energy (EFE), so the agent always picks the optimal action.`)
- (PEDA_FINAL/archive/phase1/phase1_gap_report.md:28, `**Caveat**: All G1/G2/G3 "passes" are on the **same 5×5 grid used for training**. This is memorization, not generalization or mechanism validation.`)
- (PEDA_FINAL/archive/phase1/phase1_gap_report.md:24-26, table rows `G1 World Model accuracy | > 0.90 | **Met on training distribution** | results/phase1_eval.json: g1=1.0 with adapter partial_adapter_real_25_e3`, `G2 Steps vs random | < 0.50 | **Met on training distribution**`, `G3 Revisit rate | < 0.20 | **Met**`)
- (PEDA_WORKING_LOG.md:955, `10-episode shell eval：10/10 成功，mean_steps=3.60，revisit_rate=0.0000，G1=1.0000，G2=0.4337，G3=0.0000。`)
- (PEDA_WORKING_LOG.md:971, `3. 当 World Model 在训练分布上达到 G1=1.0 时，pragmatic 项主导 EFE，所有测试权重都成功。驱动系统的 curiosity/competence/boredom/novelty 差异在 Grid World 上无法体现，因为无模型不确定性可供探索。`)
- (PEDA_WORKING_LOG.md:972, `4. 这些结果是 **in-distribution memorization**，不是泛化证据。adapter 在这个 5×5 grid 上训练，评估也在同一个分布上。`)

### 1.4 Experimental conditions (E01)

| Item | Value | Source |
|---|---|---|
| Env | 5x5 GridWorld, max_steps=50, rewards wall -0.2 / move -0.05 / goal +1.0 | src/phase1/grid_env.py (step) |
| Model | Qwen/Qwen2.5-0.5B-Instruct (local /home/chillizu/models/) | results/phase1_eval.json: `"model": "Qwen/Qwen2.5-0.5B-Instruct"` |
| Adapter | checkpoints/phase1/partial_adapter_real_25_e3 (25% cells, 3 epochs) | results/phase1_eval.json: `"adapter": "checkpoints/phase1/partial_adapter_real_25_e3"` |
| Training data | 1920 synthetic transitions, 20 configs × 24 free cells × 4 actions | checkpoints/phase1/partial_adapter_real_25_e3/training_info.json: `"transitions": 1920` |
| Train fraction | 0.25 → 6 known cells / 19 unknown; 448 train transitions | checkpoints/.../training_info.json: `"train_fraction": 0.25`; PEDA_FINAL/archive/phase1/partial_training_evaluation_report_for_senior_review.md §4.1 `train_data=448 transitions` |
| LoRA | r=16, alpha=32, dropout=0.05, target_modules="all-linear", bias="none", CAUSAL_LM | src/phase1/world_model.py:41-43, 69-76 |
| e3 hyperparams | epochs=3, batch_size=8, lr=3e-4 | checkpoints/phase1/partial_adapter_real_25_e3/training_info.json |
| e1 hyperparams | epochs=1, batch_size=4, lr=2e-4 | checkpoints/phase1/partial_adapter_real_25/training_info.json |
| Hardware | CPU-only (Intel Arc not usable by PyTorch); predict ~2.4–3.1 s/call, ~16 s/step (6 calls); 100-episode eval ~22 h | PEDA_WORKING_LOG.md:722, 775; results/phase1_report.md `**Latency:** ~2.4–3.1 s per WorldModel.predict call on CPU; ~16 s per step with 6 predict calls` |
| Eval mode | subprocess isolation (phase1_shell_eval.sh), max-candidates=4 | results/phase1_report.md caveats 4-5; PEDA_WORKING_LOG.md:947-950 |
| Drive weights | cur=0.5 cmp=0.5 bor=0.5 nov=0.5 | results/phase1_eval.json |

---

## 2. E02 — Partial training 25%, ensemble variance ~0 (epistemic blocker)

### 2.1 Quantitative results

- (0.8684, results/phase1_partial_eval_10eps.json, `"g1_test_set": 0.8684`) — 1-epoch adapter held-out (OOD) next-state accuracy, computed from chunk 0 only
- (0.8684, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:125, `|g1_test_set（held-out OOD）|0.8684|`)
- (1.0, PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:9, `` `g1_test_set = 1.0` on held-out pairs (previous 1-epoch run had 0.8684) ``) — 3-epoch adapter (partial_adapter_real_25_e3) generalizes PERFECTLY to held-out pairs
- (2/28, PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:8, `28 state-action probes × 4 actions across grid: only 2/28 showed any checkpoint disagreement`) — only 2 of 28 state-action probes showed ANY ensemble-checkpoint disagreement
- (0.0308 → 0.0047 → 0.0035, PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:5, `3 checkpoints saved (checkpoint_epoch_1/2/3) with decreasing loss (0.0308 → 0.0047 → 0.0035)`) — per-epoch loss
- (0.0, PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:10, `mean_epistemic_error = 0.0 in smoke test (1 episode × 2 conditions, max_steps=10)`)
- (0.18, results/phase1_g1_accuracy.json:8, `"g1_accuracy": 0.18`) — base model G1 (50 transitions, no adapter); exit_code 0.76; median predict 2.524 s; total 130.9 s
- (1.0, results/phase1_g1_accuracy_adapter.json, `"g1_accuracy": 1.0`) — adapter G1 on same 50 transitions; exit 0.96; delta_g1 0.82
- Ensemble formula: epistemic_error = ensemble_variance (pairwise squared distance over checkpoint-predicted positions) — src/phase1/world_model.py:977, 982 (`ensemble_variance = pairwise / count if count else 0.0` / `epistemic_error = ensemble_variance`)

### 2.2 Root cause (verbatim)

- (PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:15, `` `train_fraction=0.25` (6/25 known cells) provides enough coverage for 0.5B model with 3 epochs to perfectly generalize the 5×5 grid dynamics. The grid is too small, the transition rules too simple, and 0.5B too large for this environment. ``)
- (PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:23, `- PEDA ≈ pragmatic_only under current setup. Any "PEDA advantage" would come from drive system modulation (curiosity/boredom/novelty) not prediction-error-driven exploration.`)
- (PEDA_CONCLUSION.md:42, `| 1 | Grid World 5x5 | Partial training (25%), 3-epoch ensemble | Q1 | g1_test_set=0.8684, epistemic error ~0 from 28/28 state-action probes zero variance | FAIL — model generalizes perfectly even on 6/25 trained cells |`)

### 2.3 Methodology (E02)

- Cell-level train split: `cells = [(x, y) for x in range(5) for y in range(5)]; num_train_cells = max(1, min(int(len(cells) * args.train_fraction), len(cells))); split_rng = random.Random(args.split_seed); known_cells = split_rng.sample(cells, k=num_train_cells)` — scripts/phase1_synthetic_train.py
- Per-epoch checkpoint saving for ensemble: `epoch_ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}"; self.model.save_pretrained(str(epoch_ckpt))` — src/phase1/world_model.py:798-800
- EnsembleErrorComputer loads checkpoint_epoch_* dirs; with <2 checkpoints ensemble variance ≡ 0 — src/phase1/world_model.py (`EnsembleErrorComputer`), PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:37
- Probe protocol (7 states × 4 actions): PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md §Evidence

---

## 3. E03 — PEDA vs pragmatic-only, N=10 (no behavioral difference)

### 3.1 CRITICAL FINDING: canonical E03 numbers come from a Phase 3 GPU rerun, NOT the Phase 1 CPU archive runs

- (PEDA_CONCLUSION.md:43, `| 1 | Grid World 5x5 | Partial training, PEDA vs pragmatic N=10 | Q2, Q3 | PEDA 2.6 vs Pragmatic 2.6 steps goal_unknown, Fisher p=1.0, MW p=1.0 | FAIL — no behavioral difference; both agents identical |`)
- The 2.6/2.6/p=1.0 numbers appear verbatim in results/phase3_gpu/report.json (2026-07-27, 876 s total on GPU): `"experiment": "Phase 3 Epistemic Validation (Grid World, confidence-based epistemic)"`, `"adapter": "checkpoints/phase1/partial_adapter_real_25_e3"`, `"train_fraction": 0.25`, `"ensemble_checkpoints": 0`, `"episodes_per_condition": 10`; goal_unknown peda mean_steps 2.6 (line 46), pragmatic 2.6 (line 57); statistical_tests.goal_unknown_steps_mannwhitney `"p_value": 1.0, "peda_mean_steps": 2.6, "pragmatic_mean_steps": 2.6, "significant_at_005": false` (lines 72-76); `"passed_criteria": "3/7"`, `"verdict": "CORE_HYPOTHESIS_NOT_SUPPORTED"`, reason `"PEDA 10.0/10 (100%) vs Pragmatic 10.0/10 (100%) success in goal_unknown (Fisher p=1.0000, MW p=1.0000). Goal_known fairness: PEDA 10.0/10 vs Pragmatic 10.0/10 (p=1.0000)"`
- results/phase3_gpu/run.log:55-58: `PEDA N=10 success=1.000 steps=2.6+-1.9 revisit=0.000` / `Pragmatic-only N=10 success=1.000 steps=2.6+-1.9 revisit=0.000` / `Verdict: CORE_HYPOTHESIS_NOT_SUPPORTED (confidence: N/A)`
- IMPORTANT: This run used confidence-based epistemic proxy with `ensemble_checkpoints: 0` (NOT ensemble variance), on GPU, 2026-07-27. The Phase 1 CPU archive runs (below) report different numbers with no significance testing. PEDA_CONCLUSION.md attributes this to the Phase 1 row — treat as the canonical E03 dataset but cite results/phase3_gpu/ as the source file.

### 3.2 Original Phase 1 CPU partial-training comparison (2026-07-04/06, 1-epoch adapter, N=10 per condition)

- (0.9, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:80, `|**goal_known**|**PEDA**|**0.9**|**8.6**|**0.13**|0.93|`) — PEDA goal_known success / mean steps / revisit / g1
- (0.7, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:81, `|Pragmatic-only|0.7|17.3|0.27|1.00|`)
- (0.7, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:82, `|**goal_unknown**|**PEDA**|**0.7**|**16.6**|**0.31**|0.73|`)
- (0.6, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:83, `|Pragmatic-only|0.6|21.1|0.37|0.75|`)
- (16.6 < 21.1, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:96-97, `peda_better_in_unknown_goal: true / reason: PEDA mean_steps (16.6) < pragmatic_only (21.1) in goal_unknown condition`)
- Same numbers in results/phase1_partial_eval_10eps.json `"conditions"` block; 11 total failures across 40 episodes (raw `"success": false` at lines 234, 780, 1028, 1248, 1632, 1884, 2176, 2494, 2746, 3002, 3222) consistent with 0.9/0.7/0.7/0.6 headline rates
- PEDA exploration metrics goal_unknown: mean_unknown_fraction 0.86, mean_unknown_cells_visited 3.3, mean_steps_before_known 30.8 — results/phase1_partial_eval_10eps.json; PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md §4.2
- CONFOUND: single checkpoint → epistemic_error ≡ 0; EFE collapsed to drive-modulated pragmatic — PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:38 (`由于 epistmic_error=0，PEDA 的 EFE 公式退化为 drive_system.apply_to_efe(pragmatic * pragmatic_weight)`); phase1_validation_report.md §8.3 item 1
- (N=10 per condition per agent, PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:40, `**样本量**：每个条件每 agent 10 episode，不足以做显著性检验。`) — NO p-values ever computed for the Phase 1 CPU runs; explicitly stated "统计显著性不足，结果为方向性提示" (line 138)

### 3.3 Held-out obstacle comparison (2026-07-20, full LoRA adapter, 3 layouts)

- (30 planned / 17 completed / 13 timeouts, results/phase1_heldout_summary.json, `"total_planned": 30, "total_completed": 17, "total_timeouts": 13`) — 43% timeout rate at 240 s hard limit
- (1.0, results/phase1_heldout_summary.json, `"overall_success_rate": 1.0`) — all completed episodes succeeded
- (3.29 vs 3.10, results/phase1_report.md §Aggregate, `| Mean steps | 3.29 | 3.10 |` PEDA vs pragmatic-only)
- (0.0000 vs 0.0000, results/phase1_report.md §Aggregate, `| Mean epistemic error | 0.0000 | 0.0000 |`)
- Layout B anomaly: PEDA N=1 mean_steps 13.00 revisit 0.7143 aleatoric 0.7692 vs pragmatic N=3 mean 4.67 — results/phase1_heldout_summary.json + results/phase1_report.md:81-82
- Layouts: A vertical wall x=2 `[[2,1],[2,2],[2,3]]`; B horizontal wall y=2 `[[1,2],[2,2],[3,2]]`; C corner `[[1,1],[3,1],[1,3],[3,3]]` — results/phase1_report.md; scripts/phase1_heldout_episode.py (`--obstacles` JSON arg, max_steps=20, pragmatic_weight=3.0)
- (PEDA_FINAL/archive/phase1/phase1_gap_report.md:36, `**Not validated.** The held-out obstacle test showed PEDA and pragmatic_only behaved identically — the epistemic/drive component produced no measurable behavioral difference.`)

### 3.4 Base-model comparison (G1≈0.18, no adapter, 10 episodes each)

- (10/10 both, results/phase1_base_model_comparison_summary.json, `"peda": {"episodes": 10, "success_rate": 1.0, "mean_steps": 3.6, "revisit_rate": 0.0}` and `"pragmatic_only": {... same ...}`)
- (PEDA_WORKING_LOG.md:1000-1002, `1. 即使 World Model 的下一状态预测准确率只有约 0.18，5×5 Grid World 仍能被纯 pragmatic planning 完美解决。 2. Drive System 的 epistemic / curiosity / novelty 信号没有改变成功率、步数或回访率。 3. 这证明 5×5 Grid World 无法衡量 PEDA 的预测误差驱动探索机制。`)

### 3.5 E03 methodology

- PEDA: ActionGenerator(pragmatic_only=False, pragmatic_weight=3.0); Pragmatic-only: pragmatic_only=True, SAME pragmatic_weight=3.0; fresh HomeostaticDriveSystem per agent per episode; LearningModule disabled (update_interval=100000); identical (start, goal, seed) pairs — PEDA_FINAL/archive/phase1/phase1_validation_report.md:172; scripts/phase1_partial_eval.py §2.4 in senior review report
- Grid path EFE: `pragmatic = dist / max_dist` (Manhattan to goal); `epistemic += (1.0 - p.level2_confidence) * ratio * (0.9 ** i)`; `base_efe = epistemic + pragmatic * self.pragmatic_weight`; ConfidencePenalty `if avg_conf > 0.95: base_efe += 0.3 * (avg_conf - 0.95)` — src/phase1/drive_system.py compute_efe
- Drive modulation: `drive_adjustment = w.curiosity*curiosity*info_gain + w.competence*competence*challenge_level + w.boredom*boredom*diversity_bonus + w.novelty*novelty*external_info_potential; return base_efe - drive_adjustment` — src/phase1/drive_system.py apply_to_efe

---

## 4. Environment & state representation (all experiments)

- (5x5, src/phase1/grid_env.py, `"""5x5 deterministic grid world with wall, goal, and step-limit termination."""`, `width: int = 5, height: int = 5, max_steps: int = 50`)
- State: GridState(agent, goal, obstacles, width=5, height=5, step, max_steps=50) — src/phase1/types.py (`@dataclass class GridState`)
- Actions: UP/DOWN/LEFT/RIGHT (4) — src/phase1/grid_env.py all_actions; `max-candidates=4` REQUIRED (2 caused UP/DOWN-only vertical oscillation) — results/phase1_report.md caveat 5, PEDA_WORKING_LOG.md:969
- Text render: `"Agent at {agent}. Goal at {goal}. Obstacles at {obs}."` — src/phase1/grid_env.py Perception.render
- Rewards: wall -0.2, move -0.05, goal +1.0 — src/phase1/grid_env.py step
- Horizon 2 rollout (falls to 1 when latency budget exceeded on CPU); latency config config/phase1_model.json median_ms=4750 stale vs ~2500 real → horizon always 1 on CPU — src/phase1/drive_system.py select_action; PEDA_WORKING_LOG.md:778-780
- Pareto drive weights: stub grid search top_5 (results/phase1_grid_search.json, e.g. `{"curiosity": 1.0, "competence": 0.1, "boredom": 0.1, "novelty": 2.0}, "score": 0.9796`); real-LLM verification of top-5 collapsed to single point cur=0.5 cmp=0.5 bor=0.5 nov=0.5 (mean steps 2.0) — results/phase1_eval.json `pareto_frontier`; results/phase1_report.md `The real-LLM Pareto frontier collapses to a single point`

---

## 5. Contradictions & gaps found

1. **E03 mis-attribution in PEDA_CONCLUSION.md:43**: the Phase 1 row cites "PEDA 2.6 vs Pragmatic 2.6, Fisher p=1.0, MW p=1.0" but those numbers come from results/phase3_gpu/report.json (2026-07-27 GPU run, confidence-based epistemic, ensemble_checkpoints=0), NOT from the Phase 1 CPU partial eval (which reports 16.6 vs 21.1 goal_unknown with NO significance tests). Paper MUST cite results/phase3_gpu/ for the p-values and note the ensemble_checkpoints=0 caveat.
2. **E02 conflation**: PEDA_CONCLUSION.md:42 pairs `g1_test_set=0.8684` (1-epoch adapter, results/phase1_partial_eval_10eps.json) with "28/28 state-action probes zero variance" — but the blocker report (2026-07-07) says the 3-epoch adapter had `g1_test_set = 1.0` and only 2/28 probes showed checkpoint disagreement (phase1_epistemic_blocker_report.md:8-9). "28/28 zero variance" overstates "2/28 had disagreement".
3. **Stale formula line ref**: blocker report:11 cites `world_model.py:509` for `epistemic_error = ensemble_variance`; actual location is world_model.py:982 (line drift).
4. **Internal table inconsistency**: phase1_partial_training_eval_report.md per-episode table (§7 goal_known) shows PEDA failing episodes 1, 3, 4 (3 failures → success 0.7) but headline success_rate=0.9; raw results/phase1_partial_eval_10eps.json (11 total failures across 40 episodes) supports the headline 0.9/0.7/0.7/0.6. The per-episode table is unreliable; use the JSON aggregates.
5. **Two different G2 values for E01**: 0.4337 (2026-07-19, 10-ep, partial_adapter_real_25_e3) vs 0.1211 (2026-07-03, 20-ep, synthetic_adapter). Conclusion doc uses 0.434; both are real but from different runs/adapters.
6. **No statistics in Phase 1 CPU runs**: explicitly acknowledged ("不足以做显著性检验", phase1_partial_training_eval_report.md:40); all Phase 1 p-values in the conclusion doc trace to the Phase 3 GPU rerun (E03) only.
7. **Gaps**: no entropy/coverage/FactGraph metrics (phase1_gap_report.md §3.4); 3-epoch ensemble never fully evaluated on the partial-training protocol (blocked); only 17/30 obstacle episodes completed due to CPU timeouts; drive-weight provenance not from real-LLM grid search (phase1_validation_report.md §6, WATCHDOG C4); Phase 1.5 (TextWorld) and phase1_5 files NOT covered here (different slice S3Phase1_5); phase1_5_eval.json (2-room, both agents 0% success, 50 steps, epistemic 0.0) belongs to E04.

---

## 6. Files read (full paths)

- /home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase1/phase1_gap_report.md
- /home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md
- /home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md
- /home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase1/phase1_validation_report.md
- /home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase1/partial_training_evaluation_report_for_senior_review.md
- /home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md (Phase 1 rows + stats)
- /home/chillizu/Projects/Folunar_/results/phase1_report.md
- /home/chillizu/Projects/Folunar_/results/phase1_archive_summary.md
- /home/chillizu/Projects/Folunar_/results/phase1_eval.json
- /home/chillizu/Projects/Folunar_/results/phase1_partial_eval.json (1-ep pilot)
- /home/chillizu/Projects/Folunar_/results/phase1_partial_eval_10eps.json (10-ep, incl. raw per-episode)
- /home/chillizu/Projects/Folunar_/results/phase1_heldout_summary.json
- /home/chillizu/Projects/Folunar_/results/phase1_g1_accuracy.json
- /home/chillizu/Projects/Folunar_/results/phase1_g1_accuracy_adapter.json
- /home/chillizu/Projects/Folunar_/results/phase1_base_model_comparison_summary.json
- /home/chillizu/Projects/Folunar_/results/phase1_grid_search.json
- /home/chillizu/Projects/Folunar_/results/phase1_shell_eval.jsonl
- /home/chillizu/Projects/Folunar_/results/phase3_gpu/report.json + run.log (E03 canonical source)
- /home/chillizu/Projects/Folunar_/results/phase1_5_eval.json (handed to S3Phase1_5 / E04)
- /home/chillizu/Projects/Folunar_/src/phase1/types.py, grid_env.py, world_model.py, drive_system.py, run.py, __init__.py
- /home/chillizu/Projects/Folunar_/scripts/phase1_synthetic_train.py, phase1_eval.py, phase1_measure_g1.py, phase1_heldout_episode.py
- /home/chillizu/Projects/Folunar_/checkpoints/phase1/ (dir listing + training_info.json for partial_adapter_real_25, partial_adapter_real_25_e3, partial_adapter_real_10)
- /home/chillizu/Projects/Folunar_/PEDA_WORKING_LOG.md (Phase 1 entries: lines 652-1127)
