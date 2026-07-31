# Phase 2 (Sandbox) Evidence Report — S4Phase2

**experiment_ids: [E05, E06, E07]**
(E05 = Sandbox v1 L1/L2/L3 in-distribution; E06 = Sandbox v2 L1/L2/L3 OOD; E07 = Phase 2 multi-baseline v2. Plus E05-adjacent adapter-training and 10,040-transition collection material.)

Ground truth cross-check: `PEDA_FINAL/PEDA_CONCLUSION.md` (Phase 2 rows: E05 PASS in-distribution only, E06 FAIL, E07 FAIL — PEDA cannot beat pragmatic baseline).

---

## 1. E05 — Sandbox v1 L1/L2/L3 in-distribution (canonical 1.0 / 0.9 / 0.55)

| number | source_file:line | verbatim |
|---|---|---|
| 1.000 / 0.900 / 0.550 | PEDA_WORKING_LOG.md:1479 | `e2 held-out: L1=1.000 PASS, L2=0.900 PASS, L3=0.550 PASS` |
| 1.000 / 0.900 / 0.550 | AGENTS.md:116 | `Phase 2b: L1=1.000, L2=0.900, L3=0.550 held-out [OK]` |
| 1.0 / 0.9333 / 0.5667 | results/phase2_l1l2l3_fast_baselines_30.json:1 | `{"total": 30, "l1": 1.0, "l2": 0.9333, "l3": 0.5667, "l1_pass": true, "l2_pass": true, "l3_pass": true, "train_count": 7872, "test_count": 30}` |
| 1.0 / 0.9333 / 0.5667 | results/phase2_l1l2l3_merged_v2_30.json:1 | `{"total": 30, "l1": 1.0, "l2": 0.9333, "l3": 0.5667, ... "train_count": 8032, "test_count": 30}` |
| 1.0 / 1.0 / 0.75 | results/phase2_l1l2l3_baseline_fixed.json:1 | `{"total": 20, "l1": 1.0, "l2": 1.0, "l3": 0.75, "l1_pass": true, "l2_pass": true, "l3_pass": true, "train_count": 160, "test_count": 20}` |
| 1.0 / 1.0 / 0.0 | results/phase2_l1l2l3_baseline.json:1 | pre-L3-metric-fix quick diagnostic (L3 compared wrong field: `level3_output_summary` held generic action labels) |
| 1.0 / 1.0 / 0.0 | results/phase2_l1l2l3_baseline_full.json:1 | 40-sample artifact, `"l3": 0.0` — see Contradictions #1 |
| thresholds 0.90/0.70/0.50 | scripts/phase2_measure_l1l2l3.py:126-128 | `"l1_pass": (l1_correct / total) >= 0.90 ... >= 0.70 ... >= 0.50` |
| PASS is in-distribution only | PEDA_WORKING_LOG.md:1298 | `Caveat: held-out split is from the same random/heuristic training distribution; does not prove OOD generalization.` |

Methodology: `phase2_measure_l1l2l3.py` — deterministic 80/20 split (no shuffle), L1 = exact exit-code match, L2 = exact predicted files set vs actual next_files, L3 = token-overlap ≥ 0.5 between predicted `last_output` and actual output. Adapter under test: `checkpoints/phase2/sandbox_adapter_e2` (200 curated transitions). Eval set held-out (not used to train e2): fast-baseline and merged-v2 data (train 7872 / 8032 respectively).

## 2. E06 — Sandbox v2 L1/L2/L3 OOD (0.8 / 0.686 / 0.229)

| number | source_file:line | verbatim |
|---|---|---|
| 0.8 / 0.6857 / 0.2286 | results/phase2_remaining/l1l2l3_heldout.json:1 | `{"total": 35, "l1": 0.8, "l2": 0.6857, "l3": 0.2286, "l1_pass": false, "l2_pass": false, "l3_pass": false, "train_count": 0, "test_count": 35}` |
| — | results/phase2_remaining/report.json:7-9 (slice_a) | `Held-out test on sandbox v2 OOD directories (logs/, projects/, README.txt). e2 adapter (trained on old v1 4-dir sandbox with data/, docs/, hello.txt, tmp/) shows expected degradation: L1=0.8 (needs 0.90), L2=0.686 (needs 0.70), L3=0.229 (needs 0.50). This is a genuine finding — the WM does not generalize to new directory layouts. The v2 sandbox has 7 dirs including logs/, projects/, README.txt that e2 has never seen.` |
| 0% | PEDA_WORKING_LOG.md:1519 | `read_note 任务：所有基线 0% 成功率` (on v2 sandbox) |
| — | PEDA_WORKING_LOG.md:1523 | `Phase 2 的"成功"声明（L1=1.000, 20/20 多任务完成）仅在 v1 沙箱（4 目录）上成立。v2 沙箱（7 目录）上 WM 不泛化。Phase 2 实质上是沙箱基建 + 数据管道，不是 PEDA 运行。` |
| 1.000 / 0.900 / 0.400 | PEDA_WORKING_LOG.md:1480 | `e2 OOD: L1=1.000 PASS, L2=0.900 PASS, L3=0.400 FAIL (-0.10)` (GPU-era OOD, different layout `/sandbox/project/...`) |
| 0.400 | PEDA_WORKING_LOG.md:1502 | `OOD L3: FAIL (0.400, need 0.500)` |
| 0.833 / 0.333 / 0.133 | PEDA_WORKING_LOG.md:1481 | `e3 held-out: L1=0.833 FAIL, L2=0.333 FAIL, L3=0.133 FAIL` |
| 0.600 / 0.500 / 0.033 | PEDA_WORKING_LOG.md:1482 | `e3 OOD: L1=0.600 FAIL, L2=0.500 FAIL, L3=0.033 FAIL` |

Methodology: e2 adapter (trained only on v1 4-dir sandbox) evaluated on 35 held-out transitions collected on the v2 sandbox (7 dirs: README.txt, data, docs, hello.txt, logs, projects, tmp). Test set: `results/phase2_remaining/heldout_test_set.jsonl` (35 examples).

## 3. E07 — Phase 2 multi-baseline v2 (random, heuristic, PEDA)

### 3a. Slice C: 30 episodes, peda/pragmatic/random x read_hello/read_note
`results/phase2_remaining/multi_baseline_results.json` — metadata: `"max_steps": 10, "episodes_per_condition": 5, "total_episodes": 30, "adapter": "checkpoints/phase2/sandbox_adapter_e2/", "model": "Qwen2.5-0.5B-Instruct", "max_candidates": 5`

| condition | success_rate | mean_steps | mean_scr | revisit_rate | source |
|---|---|---|---|---|---|
| peda/read_hello | 0.8 (4/5) | 2.8 | 0.84 | 0.0 | multi_baseline_results.json `"conditions"."peda/read_hello"` |
| pragmatic/read_hello | 1.0 (5/5) | 1.0 | 1.0 | 0.0 | same |
| random/read_hello | 1.0 (5/5) | 3.0 | 0.333 | 0.0 | same |
| peda/read_note | 0.0 (0/5) | 10.0 | 0.22 | 0.0 | same |
| pragmatic/read_note | 0.0 (0/5) | 10.0 | 0.1 | 0.8 | same |
| random/read_note | 0.0 (0/5) | 10.0 | 0.2 | 0.0 | same |

- comparison table: `"read_hello": {"peda": 0.8, "pragmatic": 1.0, "random": 1.0}, "read_note": {"peda": 0.0, "pragmatic": 0.0, "random": 0.0}`
- results/phase2_remaining/report.json:18 (slice_c): `read_hello: pragmatic 100%/1.0steps > peda 80%/2.8steps > random 100%/3.0steps. read_note: ALL baselines failed (0% success, mean_steps=10.0). Pragmatic showed 80% revisit rate on read_note (dead-loop behavior). PEDA had 0% revisit but also 0% success. Note: max_candidates reduced from 8 to 5 for CPU feasibility; this may have limited task-completion actions in the candidate set.`

### 3b. Fast-baselines aggregate (300 episodes per baseline; NO PEDA row — PEDA infeasible on CPU)
`results/phase2_multi_baseline_aggregate.json`:
- random: `{"episodes": 300, "fht_sum": 60, "fht_count": 60, "scr_sum": 54.0, "dl_sum": 24.0, "steps_sum": 4920}` → AvgSCR=0.180, AvgDL=0.080, AvgFHT=1.0
- heuristic: `{"episodes": 300, "fht_sum": 60, "fht_count": 60, "scr_sum": 66.0, "dl_sum": 0.0, "steps_sum": 4920}` → AvgSCR=0.220, AvgDL=0.000, AvgFHT=1.0
- PEDA_WORKING_LOG.md:1351-1352: `random: AvgFHT=1.0, AvgSCR=0.180, AvgDL=0.080` / `heuristic: AvgFHT=1.0, AvgSCR=0.220, AvgDL=0.000`
- PEDA single-episode smoke: PEDA_WORKING_LOG.md:1353 `PEDA single-episode smoke test (read_note): FHT=0, SCR=1.0, DL=0.0, terminated at step 0.`
- PEDA_WORKING_LOG.md:1377: `PEDA multi-task evaluation: Only read_note verified end-to-end. Other tasks need GPU-backed evaluation.`

### 3c. Per-episode verification traces (evidence of mechanism, not numbers)
| run | source | result |
|---|---|---|
| peda/read_note with untrained text_adapter_e4 | results/phase2_data.log | 20 steps all `ls data`, FHT=None, SCR=0.05, DL=0.85, select≈11s/step |
| e1 verify (fast) | results/phase2_verify_e1.log | 5 steps `ls`/`ls data` oscillation, FHT=None, SCR=0.2, DL=0.0 |
| e1 ensemble | results/phase2_verify_e1_ensemble.log | 10 steps, FHT=None, SCR=0.1, 106s (select≈10.5s) |
| e2 ensemble | results/phase2_verify_e2_ensemble.log | 10 steps `id`/`ls` oscillation, FHT=None, SCR=0.1, 179s (select≈17-19s) |
| e2 + new candidates | results/phase2_verify_e2_new_candidates.log | 10 steps `cat docs/note.txt`↔`ls`, FHT=0, SCR=0.1, 186s (completes but post-goal oscillates until C18 fix) |
| C18 fix | PEDA_WORKING_LOG.md:1289 | `read_note with PEDA now terminates at step 0 after cat docs/note.txt (FHT=0, SCR=1.0, DL=0.0)` |
| Fix B smoke | results/phase2_fix_b/smoke_results.json | 3/3 episodes ok, stub mode: mean_epistemic_error=0.0, mean_aleatoric_error=0.4 (stub n=1 → variance 0) |

### 3d. PEDA 20/20 multi-task (GPU-run, v1 sandbox) — mechanism is action visibility
- PEDA_WORKING_LOG.md:1484-1486: `PEDA 5-ep/task 评估 (e2 adapter): read_note/count_lines/read_hello/find_secret: FHT=0.00, SCR=1.00, all 1-step` / `20/20 episodes 全部一次完成` (4 tasks × 5 eps)
- PEDA_WORKING_LOG.md:1492: `PEDA 可靠完成 4/5 任务：max_candidates=8 + goal_predicate 使候选集包含完成动作。但机制是动作可见性，非预测误差探索。`
- Fix: PEDA_WORKING_LOG.md:1388-1393 (goal_predicate param in ActionGenerator; max_candidates 3→8; create_file action fallback)
- create_file LIMIT: PEDA_WORKING_LOG.md:1505 `create_file: LIMIT (read-only)`

## 4. Sandbox versions v1–v4: file/dir/(state,action) counts

| version | dirs | files | unique (s,a) | source |
|---|---|---|---|---|
| v1 | 4 incl root (docs, tmp, data) | 3 (hello.txt, docs/note.txt, data/lines.txt) | 22 | Dockerfile.busybox; local/contract_engineering_plan.md:68 `v1: 22 unique (s,a), v2: 65 unique` |
| v2 | 7 subdirs (docs, data, logs, projects, projects/app, projects/lib, tmp) | 14 | 65 | AGENTS.md:120 `v2 sandbox: 7 directories, 14 files, 65 unique (state,action) pairs — 3.0× v1` |
| v3 | 7 subdirs (records, dataset, journal, modules/core, modules/shared, cache) | 15 [INFERENCE from Dockerfile.busybox_v3] | — | Dockerfile.busybox_v3 |
| v4 | 18 incl root | 29 [INFERENCE from Dockerfile.busybox_v4] | — | src/phase2/tasks.py:123 `# ── v4 micro-tasks (deeper sandbox, 18 dirs) ──` |
| v3/v4 | — | — | 270+ | PEDA_CONCLUSION.md (What Survived) `...data-driven enumeration (v2, 65 pairs; v3/v4, 270+ pairs)` |

- Systematic enumeration: results/phase2_v2_systematic.jsonl = 79 lines (78 records, flat schema `cwd/files/action/next_cwd/next_files/exit_code/output`); local/contract_research_manuscript.md:133 `系统枚举 > 随机采样（78 records via systematic vs 27 via random+heuristic）`
- v2_full data composition: results/phase2_fix_a/pipeline_result.json `"source": "merged known_wrapped (40) + wrapped_unknown_test (25)"` = 65
- PEDA_CONCLUSION.md:139: `Environment: Busybox Linux sandbox (4-7 directories, 14-65 files)`

## 5. Docker security constraints

- src/phase2/sandbox_env.py:24-26: `WHITELIST = {"ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail", "grep", "find"}`
- src/phase2/sandbox_env.py:27-33: `BLOCKLIST_PATTERNS`: `\brm\b, \bmv\b, \bcp\b, \bchmod\b, \bchown\b, \bdd\b, \bmkfs\b, \bmount\b, \bsudo\b, \bsu\b, \bdocker\b, \bkill\b, \bshutdown\b, \breboot\b`
- src/phase2/sandbox_env.py:93-95: `["docker", "run", "-d", "--rm", "--cap-drop=ALL", "--read-only", "--tmpfs", "/tmp", "--network", "none", self.image, "sleep", "3600"]`
- src/phase2/sandbox_env.py:44: `DOCKER_IMAGE = "peda-sandbox:v4"` (current default)
- PEDA_FINAL/archive/phase2/phase2_infrastructure_report.md (v1): image `peda-sandbox:latest`, base `busybox:latest` (3.7 MB); `安全验证全部通过：rm 被只读挂载拦截，网络被 --network none 阻断，/tmp 可写。`
- step() semantics: non-whitelisted/blocked → exit 1, reward 0, output = error message; 15s timeout → exit 124; `max_steps` default 20; file_cache populated on successful cat/head/tail/wc
- Confidence penalty (infra report): `如果 trajectory 的平均预测置信度 > 0.95，注入 +0.3 * (conf - 0.95) 的 EFE 惩罚`

## 6. Adapter training: epochs, batch, loss, data size

| adapter | data | epochs | batch | lr | loss | source |
|---|---|---|---|---|---|---|
| e1 | 200 transitions (100 random + 100 heuristic, 10 runs × 20 steps) | 3 | 4 | 3e-4 | 0.4424 → 0.0291 / 0.0030 / 0.0001 | results/phase2_train.log; phase2_adapter_train_report.md |
| e2 | same 200 + 6 task-completion samples relabeled exit_code=2 | 3 | 4 | 3e-4 | 0.5902 → 0.0399 / 0.0041 / 0.0016 | results/phase2_train_e2.log; PEDA_WORKING_LOG.md:408 `在现有 200 条 transitions 中，共标记出 6 条任务完成样本。` |
| e3 (GPU) | 10,040 transitions (610 eps) | 3 | 1 | — | 0.0103 → 0.0088 → 0.0087; ~2h15m | PEDA_WORKING_LOG.md:1474-1476 |
| v2_e1 | 65 transitions (phase2_v2_wrapped.jsonl) | 3 | 4 | — | — | checkpoints/phase2/sandbox_adapter_v2_e1/trained_manifest.json |
| v2_full | 65 transitions (phase2_v2_full.jsonl) | 3 | 4 | — | 0.2683 / 0.089 / 0.0312; held-out 55/10: L1=0.7, L2=0.8067, L3=0.0, FAIL | results/phase2_fix_a/pipeline_result.json |
| v2_partial | 40 transitions (phase2_v2_known_wrapped.jsonl) | — | — | — | — | checkpoints/phase2/sandbox_adapter_v2_partial/trained_manifest.json |
| e3 manifest (partial attempt) | 948 transitions, 60 runs (phase2_train_subset_60eps.jsonl) | — | — | — | — | checkpoints/phase2/sandbox_adapter_e3/trained_manifest.json — see Contradictions #5 |

- LoRA config (checkpoints/phase2/sandbox_adapter_e2/adapter_config.json): `"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": ["gate_proj","k_proj","q_proj","down_proj","up_proj","v_proj","o_proj"], "task_type": "CAUSAL_LM"`, base `~/models/Qwen2.5-0.5B-Instruct`
- SandboxLearningModule (src/phase2/run.py:13-60): `buffer_size=100, update_interval=50`, samples `batch_size=64` prioritized, `lora_finetune(data, epochs=1, learning_rate=2e-4, batch_size=4)`, saturation novelty boost
- e2 v1 held-out caveats (pipeline_result.json held_out note): `L1=0.70 (exit_code accuracy, target=0.90) — 3/10 failures: 2 cd exit_code mismatches, 1 JSON parse error. L3=0.00 because summary field missing from training data. L2=0.8067 passes target=0.70`

## 7. MICRO_TASKS — all 19 task IDs and predicates (src/phase2/tasks.py)

- v1 (tasks.py:92-100): `read_note` (Read docs/note.txt), `count_lines` (Count lines in data/lines.txt), `read_hello` (Read hello.txt), `find_secret` (Find files with 'secret'), `create_file` (Create test_dir)
- v2 (tasks.py:104-118): `count_users` (Count users in users.csv), `find_errors` (Count errors in error.log), `read_changelog` (Read docs/changelog.txt), `find_admin` (Find admin user in CSV), `count_logs` (Count lines in access.log)
- v3 (tasks.py:120-130): `read_greeting` (Read greeting.txt), `count_entries` (Count entries), `find_secret_note` (Find secret note), `read_user_guide` (Read user guide)
- v4 (tasks.py:132-147): `read_welcome` (Read welcome.txt), `find_api_key` (Find API key), `count_measurements` (Count measurements), `find_errors_v4` (Find errors in logs), `read_changelog_v4` (Read changelog for v4)

Predicate mechanics: file-reader tasks = action starts with cat/head/tail AND path hit (`_action_hits_target`, handles cwd-relative) AND `next_state.last_exit_code == 0` (tasks.py:9-24, 26-41); count tasks require `wc -l` prefix + path; v2 `count_users`/`count_logs`/`find_errors`/`read_changelog`/`find_admin` are loose output-string predicates (e.g. `"5" in out and "users" in out`, `"2" in out or "ERROR" in out`); v4 `find_api_key` requires `"API_KEY" in last_output`, `read_changelog_v4` requires `"v4" in last_output`.

## 8. 10,040 transitions collection details

- PEDA_WORKING_LOG.md:1334: `Merged existing + fast baseline data into results/phase2_train_merged_v2.jsonl: 610 episodes, 10,040 transitions.`
- PEDA_WORKING_LOG.md:1333: `FastBaselinesData completed: 600 episodes, 9,840 transitions (random + heuristic, all 5 tasks)` / `PEDAData failed: ActionGenerator WorldModel inference hung on CPU-only hardware (0 episodes, 0 transitions)` / `DirectedBaselinesData failed: pragmatic timed out at ~23 min/step; prompt only collected 2 episodes with a known argument-stripping bug`
- PEDA_WORKING_LOG.md:1337-1339: `Attempted to train sandbox_adapter_e3 on the full 10,040-transition dataset (3 epochs): timed out after 30 min.` ... `Root cause: CPU-only PyTorch inference for Qwen2.5-0.5B + LoRA is too slow for training on this machine. No NVIDIA GPU available; Intel ARC not usable by PyTorch.` / `Mitigation: The existing sandbox_adapter_e2 (trained on 200 transitions) is used as the verified Phase 2 World Model.`
- AGENTS.md:115: `Phase 2a: 10,040 transitions [OK].`
- Collection command pattern (CONTROLLER_DIRECTIVE_PHASE2.md): `--baseline random|heuristic --all-tasks --max-steps 20`

## 9. Experimental conditions

- Model: Qwen2.5-0.5B-Instruct + LoRA (r=16, α=32, dropout 0.05, 7 linear targets); stub mode via `FOLUNAR_STUB_MODEL=1` for smoke tests
- Environment: Docker busybox `peda-sandbox:{latest,v2,v3,v4}`; interface `env.reset(seed)->State`, `env.step(state, action:str)->(State, reward, done)`; SandboxState JSON keys: cwd, files, file_cache, last_command, last_exit_code, last_output[:200], step, victory, game_over; `to_structured_text()` delta format `cwd: X | files: [...] | cache: {...} | depth: N | parent: P`; `state_hash()` = cwd + sorted files (count-based novelty key)
- Data: v1 train 200 (random 100 + heuristic 100); scaled 10,040 (610 eps, random+heuristic only); v2 65 systematic; OOD test 35 (v2 layout) and project-layout OOD (GPU era); expert demos: 28 demo paths → 54 records [INFERENCE from scripts/phase2_expert_demos.py DEMO_PATHS]
- Hardware: CPU Intel Core Ultra 9 185H (no NVIDIA GPU; Intel Arc unusable by PyTorch) — select 10-19s/step, ~176s cold start, 30-min training timeout; GPU AWS g4dn.xlarge (T4 16GB), DL AMI PyTorch 2.11, NVIDIA 595.71.05, CUDA 13.2, total ~4.5h, ~$2.40 (PEDA_WORKING_LOG.md:1466-1470)
- Runtime env at Fix B smoke (results/phase2_fix_b/smoke_results.json): python 3.14.6, torch 2.13.0, transformers 5.14.1, cuda_available false
- Baselines: peda (EFE + ensemble), pragmatic (pragmatic_only), random (seed 42), heuristic (random + repetition penalty, avoid >2 repeats in last 5), prompt (raw Qwen2.5 chat, whitelist fallback)
- Drive config: `DRIVE_WEIGHTS = DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)`, `PRAGMATIC_WEIGHT = 3.0`, `max_candidates=5` (8 after fix), `horizon=1` (scripts/phase2_collect_data.py:27-29)

## 10. Verbatim conclusions from source docs

- PEDA_WORKING_LOG.md:1490: `数据质量 > 数量：e3 (10,040 random+heuristic) 退化。random 数据任务完成信号稀疏，模型学到不预测完成。e2 (200 curated) 信号更纯。`
- PEDA_WORKING_LOG.md:1492: `PEDA 可靠完成 4/5 任务：max_candidates=8 + goal_predicate 使候选集包含完成动作。但机制是动作可见性，非预测误差探索。`
- PEDA_WORKING_LOG.md:1523: `Phase 2 实质上是沙箱基建 + 数据管道，不是 PEDA 运行。`
- PEDA_WORKING_LOG.md:1404: `但 PEDA 实际行为接近"选择已知任务完成动作"（因 WM 对完成信号预测准确），尚未充分验证 prediction-error-driven exploration。`
- PEDA_FINAL/archive/phase2/phase2_adapter_train_report.md: `epistemic 信号未产生逃离效果：ensemble 已加载 3 个 epoch checkpoints，但 PEDA 仍在 ls 与 ls data 之间振荡。`
- PEDA_FINAL/PEDA_CONCLUSION.md (Phase 2 rows): E05 `PASS — in-distribution only`; E06 `FAIL — WM does not generalize to new layouts`; E07 `read_hello: Pragmatic 1.0s > PEDA 2.8s. read_note: ALL 0% success — FAIL — PEDA cannot beat pragmatic baseline`
- PEDA_CONCLUSION.md Root Cause 1: `In Sandbox v2 (Phase 2), the WM reached L1=1.000 on its training distribution after 200 curated transitions. The model does not produce epistemic uncertainty because the state spaces we tested are small enough for LoRA fine-tuning to memorize the deterministic transition dynamics completely.`
- PEDA_CONCLUSION.md Root Cause 2: `PEDA's apparent wins depend on the candidate set, not on intrinsic exploration.`

## 11. Contradictions / gaps

1. `results/phase2_l1l2l3_baseline_full.json` (total=40) records L3=0.0, but PEDA_WORKING_LOG.md:1297 claims `Full 40-sample run: L1=1.0000, L2=1.0000, L3=0.7500 (all pass v1.1 thresholds)` — the persisted artifact contradicts the log for the 40-sample run.
2. Canonical E05 numbers (1.000/0.900/0.550, AGENTS.md:116 + working log:1479) do not match any single persisted JSON: 30-sample files show 1.0/0.9333/0.5667 (train 7872 and 8032); 20-sample fixed shows 1.0/1.0/0.75. Three different L2/L3 value sets exist; the 0.900/0.550 pair appears only in prose (likely a GPU-era re-measurement whose JSON is on S3, not in results/).
3. Directory-count conventions inconsistent: v2 "7 directories" (AGENTS.md:120) excludes root; v4 "18 dirs" (tasks.py:123) includes root. Dockerfile-derived counts [INFERENCE]: v2 = 8 dirs/14 files, v3 = 7 dirs/15 files, v4 = 18 dirs/29 files.
4. `phase2_multi_baseline_aggregate.json` has no PEDA row (PEDA only single-episode smoke on CPU); E07 PEDA numbers exist only in the 30-episode Slice C run (read_hello/read_note) and the GPU 5-ep/task run.
5. `checkpoints/phase2/sandbox_adapter_e3/trained_manifest.json` records 948 transitions/60 runs (partial attempt), not the headline 10,040-transition GPU training — the final e3 loss/artifacts exist only in PEDA_WORKING_LOG.md prose.
6. Counting conventions differ: "65 unique (s,a)" (deduped pairs, AGENTS.md/contracts) vs "78 records via systematic vs 27 via random+heuristic" (enumerated transitions, contract_research_manuscript.md:133).
7. Phase 2 "success" (L1=1.0, 20/20) holds only on v1 sandbox; every v2 OOD threshold fails (E06); explicitly flagged in PEDA_WORKING_LOG.md:1523. The `success` field in later phases was separately found non-discriminative (always True when SCR>0), so FHT is the reliable metric (working log:1695-1699, Phase 3/4 territory).
