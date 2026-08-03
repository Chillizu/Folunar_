# Phase 1.5 Evidence Report (E04)

**Scout:** S3Phase1_5
**experiment_ids:** [E04] (plus E01/E02/E03 context from Grid World blocker sections in setup/complete reports)
**Ground truth:** PEDA_FINAL/PEDA_CONCLUSION.md:44 — `| 1.5 | TextWorld | 2-room PEDA vs Pragmatic, 2 iterations | Q2 | PEDA distinguishable from Pragmatic (explores key/look), but epistemic ~0 | PARTIAL — drives modulate, not prediction error |`

---

## 1. Two-room environment (custom TextRoomEnv, NOT real TextWorld)

- E04: (2 rooms: study + hallway) `src/phase1_5/text_env.py:1-4` — "Two rooms connected by a door: Study (start): has a key on the desk / Hallway: has a locked chest / Goal: take key, go north, unlock chest."
- E04: (6 legal actions) `src/phase1_5/text_env.py:114-118` — `all_actions() = ["look", "inventory", "take key", "go north", "go south", "unlock chest with key"]`
- E04: (exits) `src/phase1_5/text_env.py:47-49` — `ROOM_EXITS = {"study": {"north": "hallway"}, "hallway": {"south": "study"}}`
- E04: (state type) `src/phase1_5/text_env.py:14-22` — `TextState(room, description, inventory, goal, step, max_steps=50, game_over, victory)`
- E04: (optimal path = 3 steps) `PEDA_FINAL/archive/phase1_5/phase1_5_setup_report.md:50` — "最优路径：拿钥匙 → 向北走 → 开宝箱（3 步）"
- E04: (env diagram) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:55-60` — "书房 (study) ──── 门向北 ──── 走廊 (hallway) ... 最优路径：拿钥匙 → 向北走 → 用钥匙开宝箱（3 步）"
- E04: (state space bound) `PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:57` — "The 2-room env's state space (2 rooms × ~5 actions × inventory states) is too small for meaningful epistemic uncertainty."

**NOTE:** real TextWorld was never used for evaluation. TextWorld 1.7.0 DID install cleanly in a Python 3.10 uv venv (`textworld_setup_attempt_report.md:1-5`, step table lines 25-40), and `results/phase1_5_textworld_data.jsonl` (202 lines, tier-1 cookhouse games) was generated via `scripts/phase1_5_textworld_generate.py` (_TARGET_UNIQUE=1500, only 202 achieved → under-delivered vs target), but no eval ever ran on it.

## 2. Training data: 113 (iter1) / 114 (iter2) samples

- E04: (113 unique) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:88` — "最终：**113 条唯一样本**" (exhaustive per-room actions + 50 walks × 20 steps, dedup key `state_text + action_name`, complete_report.md:86-87)
- E04: (114 after augmentation) `PEDA_FINAL/archive/phase1_5/phase1_5_iteration2_report.md:30-31` — "参数：200 walks × 30 steps = 6000 次尝试 / 去重后：**114 条**（与之前 113 条几乎相同）... state_text + action_name 高度重复 / 数据增强在此环境级别没有实际意义"
- E04: (2 orders of magnitude too little) `PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:18` — "113 unique samples (50 walks × 20 steps, deduped); after augmentation: 114 | 2 orders of magnitude below what a 0.5B model needs"
- E04: (generation strategies) `scripts/phase1_5_synthetic_train.py:60-152` — Strategy A pure random (1/3), B goal-biased unseen-actions (1/3), C repeat same action 2-3x (1/3); defaults num_walks=200, walk_length=30
- E04: (manifest confirms 114) `checkpoints/phase1_5/text_adapter_e4/trained_manifest.json` — `{"num_rooms": 2, "num_transitions": 114, "num_walks": 200, "walk_length": 30}`

## 3. LoRA training conditions (both iterations)

- E04: (iter1) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:91-99` — model Qwen2.5-0.5B-Instruct, epoch 3, batch_size 4, 3 checkpoints, 耗时 623 秒, Loss 0.2928 → 0.0545 → 0.0240
- E04: (iter2) `PEDA_FINAL/archive/phase1_5/phase1_5_iteration2_report.md:39-40` — Loss 0.2622 → 0.0577 → 0.0175, 耗时 947s (retrain 16 min)
- E04: (hardware) `PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:44-46` — CPU-only workstation, 0.5B Qwen2.5 at ~4 tokens/sec via llama.cpp, no CUDA GPU
- E04: (systematic model error) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:112-116` — "所有 3 个 checkpoint 对 `take key` 的预测都是 exit=1（不能拿钥匙）— 错误的。环境实际允许拿钥匙。"; `phase1_5_iteration2_report.md:42` — "`take key` 仍是 exit=1" (e4, even after retrain); `phase1_5_iteration2_report.md:45-46` — "go north: 2 ❌ → 1 ❌" (prediction got WORSE after retrain)

## 4. Epistemic signal measurements

- E04: (semantic probe disagreement) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:108-110` — Room 10% (3/30), Exit code 7% (2/30), Has-key 40% (12/30), **完整语义元组 50% (15/30) 超过 33% 阈值**
- E04: (probe construction = 5 states × 6 actions = 30) `scripts/phase15_semantic_probe.py:34-54,65-73` — 5 hand-built states (study-nokey, hallway-nokey, study-key, hallway-key, victory) × all 6 actions; threshold default 0.33
- E04: (mean_epistemic_error pre-fix = 0.0 in real run) `results/phase1_5_eval_chunk_0.json:22,27,80,132` — PEDA 0.0, Pragmatic 0.0 (1653.5s real run, elapsed_seconds:142)
- E04: (post-fix = 0.20/0.2222) `results/phase1_5_eval_iter2.json:22,27` — PEDA mean_epistemic_error 0.2, Pragmatic 0.22222222222222224; `phase1_5_iteration2_report.md:23-24` — "PEDA 0.0000 → 0.2000 / Pragmatic 0.0000 → 0.2222"
- E04: (fix mechanism) `src/phase1/world_model.py:915-946` — TextState branch now parses "Inventory:" line per checkpoint into pred_has_keys, level2_errors = (room mismatch ?1:0)+(has_key mismatch ?1:0), pairwise variance over (exit, room, has_key) normalized /3.0; actual_has_key = "key" in (actual.inventory or []) (line 929)
- E04: (measured signal is inventory confusion, not env complexity) `PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:57` — "The decompose_error fix raised mean_epistemic_error to 0.20, but this was driven by inventory-state confusion, not genuine environmental complexity."

## 5. PEDA vs Pragmatic — step counts, NO p-values exist

- E04: (Iteration 1) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:138-139` — "**PEDA** ❌ 20/20 | inventory → look → **take key** (step 3!) → inventory × 17 | **Pragmatic** ❌ 20/20 | look × 20"; 耗时 1654 秒 (line 132); max steps 20, 1 episode per agent (config lines 126-132)
- E04: (Iteration 2) `PEDA_FINAL/archive/phase1_5/phase1_5_iteration2_report.md:50-51` — "**PEDA** take key → inventory → look → inventory×7 ❌ 10/10 | **Pragmatic** look×10 ❌ 10/10"; eval 685s (line 9); max_steps 10
- E04: (PEDA explored EARLIER in iter2 — step 1 vs step 3) `PEDA_FINAL/archive/phase1_5/phase1_5_iteration2_report.md:55` — "PEDA 在第 1 步就尝试 `take key`（比 Iteration 1 的 step 3 更快）。Pragmatic 从未尝试。"
- E04: (**P-VALUES: NONE. Not computed, significance explicitly unknown**) `PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:65` — "**No multi-episode statistics**: All Phase 1.5 behavioral findings are based on 1-2 episodes per condition. Statistical significance is unknown." The paper must NOT cite a Phase 1.5 p-value. (E03 is the only Phase-family p-value: `PEDA_CONCLUSION.md:43` — "PEDA 2.6 vs Pragmatic 2.6 steps goal_unknown, Fisher p=1.0, MW p=1.0" — that is Phase 1 Grid World, N=10.)
- E04: (both agents 0% success both iterations) `results/phase1_5_eval_chunk_0.json:137-140` verdict "PEDA success_rate=0.00 vs pragmatic=0.00, diff=+0.00 (threshold +0.10)", peda_better false; `results/phase1_5_eval_iter2.json:97-100` same verdict.

## 6. Drive system behavior (the ACTUAL driver, not prediction error)

- E04: (weights) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:131` — "Drive weights | cur=0.1, cmp=2.0, bor=0.1, nov=2.0"; pragmatic weight 3.0 (line 130)
- E04: (driver attribution) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:147-149` — "不是因为 ensemble variance（≈ 0），而是因为：1. **LLM 自身置信度信号**：epistemic_ratio = 1 - confidence。模型对 inventory 反复执行后，置信度逐渐降低 → boredom 累积 → 驱动向未尝试的动作偏移 2. **Drive system 调制**：boredom=0.1 的权重虽小，但在多次重复后足以产生可测量的偏差"
- E04: (post-key loop cause) `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:152-155` — "拿钥匙后 inventory 的置信度 0.999 → EFE 最低 → 每次都选 inventory; 模型没有学到 inventory → key present → go north 的转移规则; 113 条训练数据太少"
- E04: (drive term formulas) `src/phase1/drive_system.py:67-73` — "boredom_term = max(0.0, 0.7 - _action_entropy(action_history, window=50)); novelty_term = 1.0 - math.exp(-0.01 * self.steps_since_external_input)"; apply_to_efe:88-112 subtracts drive_adjustment = curiosity*challenge + competence*challenge_level + boredom*diversity_bonus + novelty*external_info_potential from base_efe
- E04: (compute_efe TextState guard) `src/phase1/drive_system.py:158-198` — pragmatic = 0.0 if final_exit==2 else 0.5 (line 183); pragmatic_only returns pragmatic*3.0 (184-185); epistemic = Σ (1.0 - p.level2_confidence)*ratio*(0.9**i) (187-189); ConfidencePenalty: avg_conf > 0.95 → base_efe += 0.3*(avg_conf-0.95) (192-195)
- E04: (eval protocol pre-registered) `scripts/phase1_5_eval.py:11-17` — 10 episodes per agent (20 total), max_steps 50, seed 42, success threshold PEDA > pragmatic + 10%; actual runs executed only 1 episode per agent (see JSON protocols)

## 7. Verbatim conclusions

- E04: `PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:159-167` — "1. **PEDA 的行为可与 Pragmatic 区分**... 这个差异是真实的、可复现的。2. **当前驱动力不是 prediction error，而是 drive system**。Ensemble variance ≈ 0... 3. **`decompose_error` 低估了真实方差**... has-key 维度被完全忽略 → mean_epistemic_error=0.0。4. **113 条数据不够**。"
- E04: `PEDA_FINAL/archive/phase1_5/phase1_5_closeout.md:1` — "Phase 1.5 complete: decompose_error fix confirmed, epistemic 0→0.20, PEDA behavior difference replicated 2/2."
- E04: `PEDA_FINAL/archive/phase1_5/PHASE1_5_COMPLETE_EVALUATION.md:79-85` — "核心假设（epistemic_error 驱动探索）未验证... 替代路径：drive system 产生可区分行为（已验证 ✅）"
- E04: `PEDA_FINAL/archive/phase1_5/PHASE1_5_ITERATION2_EVALUATION.md:55-58` — 已验证: PEDA ≠ Pragmatic (2/2 迭代复现), Drive System 有独立探索价值, decompose_error 修复有效 (0.0→0.20), 2 房间状态空间太小 (6000 尝试 → 114 去重); 未验证: epistemic error 驱动有意义探索, EFE 优于贪心, WM 学习文本转移动态
- E04: `PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:55` — "Phase 1.5 proved 'PEDA ≠ Pragmatic' but not 'PEDA > Pragmatic in task completion'"

## 8. Contradictions / gaps found

1. **NO p-values for Phase 1.5 anywhere** (deviation_report.md:65 explicitly says statistical significance unknown, 1-2 episodes/condition). Do not fabricate.
2. `results/phase1_5_eval.json` is a **STUB run** (elapsed_seconds=0.0048, mean_steps=50) — must not be cited as real; real iter-1 run is `phase1_5_eval_chunk_0.json` (elapsed 1653.5s, max_steps 20).
3. Pre-registered protocol (10 episodes/agent, max_steps 50, `scripts/phase1_5_eval.py:11-17`) vs executed runs (1 episode/agent, max_steps 20 then 10). Full 20-episode protocol never executed.
4. Report config vs script defaults mismatch: `phase1_5_complete_report.md:128-129` says Candidates=3, Horizon=1; script defaults are max-candidates=4, horizon=2.
5. `phase1_5_deviation_report.md:15` contains a copy-paste error — task-structure row quotes a Phase 2 sandbox task ("read_note: cd docs → cat note.txt (2-3 steps)") instead of the key/chest task.
6. Real TextWorld data was collected (202 unique transitions, target 1500 — under-delivered) in `results/phase1_5_textworld_data.jsonl` but never trained or evaluated on; all E04 results come from the custom TextRoomEnv.
7. `decompose_error` bug direction: report says fix raised 0.0→0.20, but chunk_0 (e3 adapter, pre-fix) measured 0.0 for BOTH agents in the real run — consistent with the bug claim; iter2 (e4, post-fix) shows 0.2/0.2222. The e3/e4 comparison across iterations confounds adapter retrain with code fix (both changed between iterations).
8. Grid-world blocker context (E01/E02/E03): `phase1_5_setup_report.md:15-17` (0.0308→0.0047→0.0035, g1=1.0, 2/28=7%), `phase1_5_setup_report.md:26-28` (2/25 cells, 148 samples, 0.0739→0.0134→0.0011, 5/28=18% < 10/28 threshold → abandon), `phase1_5_complete_report.md:31-35` (25% → 6/25, 448 samples, 2/28=7%; 10% → 2/25, 148 samples, 5/28=18%).

## Hardware / conditions summary

Model: Qwen2.5-0.5B-Instruct + LoRA (3 epochs, batch 4, lr 3e-4, 3 ensemble checkpoints). Env: custom 2-room TextRoomEnv (study↔hallway, 6 actions, 3-step optimal). Data: 113 (e3) / 114 (e4) unique transitions. Hardware: CPU-only (Intel Core Ultra 9 per PEDA_CONCLUSION.md:140), llama.cpp ~4 tok/s. Eval: 1 episode/agent/iteration; max_steps 20 (iter1)/10 (iter2); drive weights cur=0.1 cmp=2.0 bor=0.1 nov=2.0; pragmatic weight 3.0; seed 42; success threshold +0.10.

## Files read

- PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md, phase1_5_experimental_report.md, phase1_5_iteration2_report.md, PHASE1_5_COMPLETE_EVALUATION.md, PHASE1_5_ITERATION2_EVALUATION.md, phase1_5_deviation_report.md, phase1_5_setup_report.md, textworld_setup_attempt_report.md, phase1_5_closeout.md (all 9)
- src/phase1_5/text_env.py, textworld_env.py, __init__.py
- scripts/phase1_5_synthetic_train.py, phase1_5_eval.py, phase15_semantic_probe.py, phase1_5_textworld_generate.py
- results/phase1_5_eval.json, phase1_5_eval_chunk_0.json, phase1_5_eval_iter2.json, phase1_5_textworld_data.jsonl (line count 202)
- src/phase1/world_model.py:860-975 (decompose_error), src/phase1/drive_system.py:142-223 (compute_efe/apply_to_efe)
- checkpoints/phase1_5/text_adapter_e4/trained_manifest.json
- PEDA_FINAL/PEDA_CONCLUSION.md (E04 row, hardware summary)
