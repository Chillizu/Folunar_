# Phase 8 速赢报告 — count-driven agent（L1-QW, 2026-08-02）

方向：Phase 8 count-driven agent 速赢。改动全部在 `src/phase8/`（`count_driven_agent.py`），
评估脚本 `scripts/phase8_qw_*`，per-episode JSONL 落 `results/phase8_qw_*`。
未触碰其他 phase 目录、未改 `PHASE9_PLAN.md`。

## 1. 结论摘要

| 指标 | 基线（GPU 07-31） | 基线（本地复跑） | 修复后（全部改动） |
|------|:---:|:---:|:---:|
| 总成功率 | 28/45 (62.2%) | 27/45 (60.0%) | **39/45 (86.7%)** |
| count_lines | 0/5 (0%) | 0/5 (0%) | **4/5 (80%)** |
| 深层路径 read_note / find_api_key / find_errors_v4 | 1/1/1 | 1/0/0 | 3/2/5 |

- **62.2% 提升至 86.7%（+24.4pp，+11 集）**；相对本地基线 +26.7pp（+12 集）。
- **count_lines 0% → 80%**，盲区根因（wc -l 只对 csv/log 生成）已修复。
- 所有 9 任务 ≥ 基线，无回归（find_api_key 在修复组合下 1/5 → 2/5）。
- 改动归因（见 §5）：verb×file 矩阵 +5，cached-child revisit +5，budget guard +2。

## 2. 基线

- GPU 基线（`results/phase8_gpu_run_2026-07-31.md`，commit a348c1e）：28/45 (62.2%)，
  直接读类 5 任务 100%，深层路径 3 任务各 20%，count_lines 0%。
- 本地复跑（`results/phase8_qw_baseline/`，5ep×9task，max_steps=10，与 GPU 同参数）：
  27/45 (60.0%)。与 GPU 的差异仅在 find_errors_v4（本地 0/5 vs GPU 1/5，见 §3.4），
  属环境随机性；本报告以本地基线为修复前后对比主基准，GPU 数字作参考。

## 3. 改动 1 — count_lines 盲区修复（verb×file 组合矩阵）

### 3.1 根因（已定位）

`src/phase2/sandbox_env.py::generate_sandbox_candidates` 只对 `csv/log` 扩展名生成
`wc -l {f}`，对 `txt`（lines.txt）从不生成 `wc -l lines.txt`；且 `wc` 在
`NoveltyExplorer._ACTION_PRIORITY` 中为第 1 层（content analysis），落后于全部
cat/echo/head（第 0 层）——即使候选存在，10 步预算内也被 12+ 个 reader 候选饿死。
两层原因叠加 → count_lines 恒 0%。

### 3.2 改动内容（全部在 `src/phase8/count_driven_agent.py`）

1. 新函数 `generate_phase8_candidates(state)`：对**发现的每个文本文件**生成完整适用动词
   集合 `cat / head -n 5 / wc -l`（verb×file 矩阵），不再按扩展名门控 wc/head。
   零任务知识：组合对任何新文件通用，不硬编码文件名/答案。
2. 正常沙盒剔除 `echo`（echo 是 CI 沙盒专属 reader，正常沙盒里只浪费 reader 层预算）。
3. 新增 `grep -ri error .`（大小写不敏感，见 §3.4 的 find_errors_v4 根因）。
4. 新类 `Phase8Explorer(NoveltyExplorer)`：`_ACTION_PRIORITY` 中把 `wc` 提升到
   reader 层（第 0 层），使 `wc -l <文件>` 在目录内可被选中（这是矩阵能生效的前提）。
5. CI 模式（`--ci`）保持原 `NoveltyExplorer` + phase2 候选生成器不变（CI 语义属
   Phase 9 方向 1，不越界）。

### 3.3 预期（改动前记录）

基于候选/优先级/预算的手工推演（已写入本报告后才实现）：

- count_lines：矩阵 + wc 提升后，`wc -l lines.txt` 在 /sandbox/data 内第 6-8 个动作
  被选中 → 首次成功可达（预期 1/5）；配合 §4 的 revisit 后预期 4/5。
- find_errors_v4：`grep -ri error .` 从 /sandbox/logs 出发命中 `ERROR:` 行（exit 0）
  → 预期 5/5。
- read_note / find_api_key：预期 5/5 / 4/5（实际见 §5，因 root 层 reader 候选变宽，
  首个子目录进入推迟了一个 episode，低于预期但无回归）。

### 3.4 结果（matrix_only 消融，见 §5 表格）

- count_lines 0/5 → 1/5（首次成功在第 1 集 step 10：`cd data` → `wc -l lines.txt`）。
- find_errors_v4 0/5 → 5/5。附根因：v4 沙盒日志全部为**大写** `ERROR:`，
  原 `grep -r error .`（小写）从 /sandbox/logs 出发 exit=1 恒失败；
  `-ri`（大小写不敏感）命中后 exit=0 直接达标（谓词只要求 grep+error+exit 0）。

## 4. 改动 2 — 深层路径 20% 改进（cached-child revisit + budget guard）

### 4.1 分析（success cache + 探索策略能否补）

- **success cache 机制本身是正确的**：直接读类任务（read_hello 等）第 0 集发现答案后，
  后续集在同一状态上直接重放 → 5/5。
- **深层任务的缺口**：答案在子目录状态里（如 /sandbox/docs 的 `cat note.txt`），
  cache 只在「已经在该状态」时重放；而探索器按 novelty 广度优先，每个子目录只进一次
  （cd 计数为 1 后 novelty 低于其他未试动作），**从不重新进入发现答案的目录** →
  cache 永远等不到重放。这是 20% 的根因，不是预算问题本身。
- 结论：加一个通用的「回到已知成功子目录」机制即可补，无需深度优先大改。

### 4.2 改动内容

1. `Phase8Explorer.record_cd(parent_state, action, child_state)`：记录 cd → 子状态哈希
   （确定性沙盒，一次记录跨集有效），在 `run_episode` 每次 `cd` 后调用。
2. `Phase8Explorer.select_action` 在选择顺序上插一层：当前状态无 cache 时，若某个
   `cd X` 的子状态在 success_cache 中 → 立即返回该 cd（回到答案所在目录重放）。
   通用：对任何「曾发现答案的目录」都生效，不含任务知识。
3. **budget guard**（`run_episode`）：最后一步（step 9）不允许进入**新目录**
   （允许 `cd ..` 和已知目录）。修复 1 加宽 root 层 reader 候选后，v4 root 的
   `cd docs` 被推迟到第 2 集 step 10（进入即耗尽预算 = 纯浪费：状态被标记访问但从未
   行动，且无 cache 可回访）。这是 matrix 引入 find_api_key 回退（1→0）后的最小修正。

### 4.3 预期（改动前记录）

- read_note 1/5 → 5/5；find_api_key 1/5 → 4/5（实际见 §5）。
- count_lines 在 §3 首次成功后经 revisit 变成可重放：1/5 → 4/5。

### 4.4 结果

- read_note 1/5 → 3/5（发现后重放生效：ep2 发现，ep3-4 各 2 步重放；ep0/1 消耗在
  data/docs 之前的目录序上）。
- find_api_key 0/5 → 2/5（budget guard 使 ep2 不再浪费在 step-10 进 docs，ep3 发现，
  ep4 重放）。
- count_lines 1/5 → 4/5（ep1 发现，ep2-4 各 2 步 `cd data` + `wc -l lines.txt` 重放）。

## 5. 改动归因（消融实验）

`scripts/phase8_qw_ablation.py`（5ep×9task，与主评估同参数）：

| 配置 | read_hello | read_note | count_lines | find_secret | read_welcome | find_api_key | count_meas | find_errors | changelog | 总计 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 基线（本地） | 5 | 1 | 0 | 5 | 5 | 1 | 5 | 0 | 5 | **27** |
| matrix_only（改动1） | 5 | 1 | 1 | 5 | 5 | 0 | 5 | 5 | 5 | **32** |
| matrix+revisit（无guard） | 5 | 3 | 4 | 5 | 5 | 0 | 5 | 5 | 5 | **37** |
| all（改动1+2+guard） | 5 | 3 | 4 | 5 | 5 | 2 | 5 | 5 | 5 | **39** |

- **改动 1（verb×file 矩阵 + wc 提升 + grep -ri + 去 echo）**：27 → 32（+5）。
  归因：count_lines 0→1、find_errors_v4 0→5；同时 find_api_key 1→0（-1，
  root 层 reader 变宽所致，由 guard 收回）。
- **改动 2（cached-child revisit）**：32 → 37（+5）。
  归因：count_lines 1→4（+3）、read_note 1→3（+2）。
- **budget guard**：37 → 39（+2）。归因：find_api_key 0→2。
- 合计 27 → 39（+12）；GPU 参考基线 28 → 39（+11）。
- 「all」消融与主评估 `fixed2` 完全一致（39/45），可复现。

## 6. 9 任务修复前后对比（验收表）

| # | 任务 | 镜像 | 基线(GPU) | 基线(本地) | 修复后 | Δ(本地) |
|:--:|------|:---:|:---:|:---:|:---:|:---:|
| 1 | read_hello | v2 | 5/5 | 5/5 | 5/5 | 0 |
| 2 | read_note | v2 | 1/5 | 1/5 | 3/5 | +2 |
| 3 | **count_lines** | v2 | **0/5** | **0/5** | **4/5** | **+4** |
| 4 | find_secret | v2 | 5/5 | 5/5 | 5/5 | 0 |
| 5 | read_welcome | v4 | 5/5 | 5/5 | 5/5 | 0 |
| 6 | find_api_key | v4 | 1/5 | 1/5 | 2/5 | +1 |
| 7 | count_measurements | v4 | 5/5 | 5/5 | 5/5 | 0 |
| 8 | find_errors_v4 | v4 | 1/5 | 0/5 | 5/5 | +5 |
| 9 | read_changelog_v4 | v4 | 5/5 | 5/5 | 5/5 | 0 |
| | **TOTAL** | | **28/45 (62.2%)** | **27/45 (60.0%)** | **39/45 (86.7%)** | **+12** |

per-episode JSONL：`results/phase8_qw_{baseline,fixed,fixed2,matrix_only,all}/<task>.jsonl`
（meta 行含 commit/timestamp/镜像；每行一个 episode 的 success/steps/actions）。

## 7. STRIPS 并入 count-driven agent 的可行性与预期收益（分析性，未实现）

### 7.1 证据

1. `checkpoints/phase9/ci_m2_strips_rules.json`（24 条规则）：STRIPS 从 CI 沙盒轨迹学到
   的（verb|target_type）→ {exit_code, delta, output, n, conf} 规则，
   held-out DLR 0.926（M2 门 ≥0.70 PASS）。规则置信度高：`cd|any` exit 0+delta true
   conf 1.0、`echo|file` exit 2+output true conf 1.0、`ls|any` exit 3 conf 0.857 等。
   → STRIPS 对确定性命令语义的建模能力已被证明（数据充足时近 100% 置信）。
2. Phase 5（`PEDA_FINAL/COUNT_DRIVEN_CHARTER.md` §3.3、`PEDA_CONCLUSION.md`）：
   learned action schemas 45.8% vs fallback 31.3%（+14.5pp，v2 沙盒动作预测任务）。
   学习到的 schema 是 cwd 无关的 lifted 谓词（precondition + effects）。
   注：45.8% 的天花板被判断为数据覆盖问题而非机制问题（charter 原话）。

### 7.2 具体接入点

- **入口**：`src/phase8/count_driven_agent.py`，`Phase8Runner.run_episode` 第 1 步
  Perception（`candidates = generate_phase8_candidates(state)`，约 206 行）。
- **现状**：runner 已实例化 `ActionModelLearner`（`self.action_model`）并每步调用
  `learn_from_step` 学习 STRIPS，但**学到的模型从未被消费**——`generate_candidates` 与
  `plan_to_target` 零调用。
- **可复用现成接口**（`src/phase5/action_model.py`）：
  - `generate_candidates(state, task_id=None)`（216 行）：按 schema 成功率和当前 cwd
    文件生成动作候选，并带 `dir_contents` 导航规划。
  - `plan_to_target(state, target_file)`（352 行）：输出多步计划，如
    `["cd data", "wc -l lines.txt"]`。
  - `schemas[verb|target_type]`：每条含 attempt_count / success_count / effects
    （exit_code、output、delta）。
- **建议接线（两处之一，均不改选择主循环）**：
  a. 候选合并：`candidates += [c for c in self.action_model.generate_candidates(state)
     if c not in candidates]`（注意：**禁止传 task_id**，TASK_KEYWORDS 硬编码了
     lines.txt/note.txt 等任务答案，违反零任务知识原则；只用 lifted schema + dir_contents）。
  b. 选择偏置：对 candidate，查其 schema（verb+target_type）的 success_rate 与
     effects.output/exit_code，过滤「预测 exit≠0 或 output=false」的动作
     （如正常沙盒 cat 目录、wc 缺失文件），省出的步数直接缓解深层路径预算瓶颈。

### 7.3 预期收益

1. **动作剪枝**：STRIPS 学会的 exit_code/output 语义可直接剔除无效动作（正常沙盒：
   `cat|dir` → exit 1，`wc|missing` → exit 1），减少浪费步 → 深层路径预算更多。
2. **导航组合**（charter 未决问题 #2）：`dir_contents` 地图 + `plan_to_target` 把单步
   schema 组合成多步计划。零知识通用版：「对 dir_contents 中任何不在 cwd 的文件，
   规划 cd 链 + reader/counter」→ count_lines 从 ~8 步降到 2 步
   （`cd data` + `wc -l lines.txt`），read_note/find_api_key 同理。
3. **output 预测**：`output:true` 区分内容揭示动作（cat/head/wc on file）与无输出动作，
   让探索器优先选有产出的动作（当前矩阵仍为每个文件生成 3 个 verb，STRIPS 可排序）。

### 7.4 风险与边界

- 只做**辅助过滤/扩展**，不替换 count-based novelty 选择主循环——count 是唯一可靠机制
  （Phase 8/9 大量证据），STRIPS 信息应注入同一循环。
- CI 规则文件是 CI（反转）语义；正常沙盒规则不同，但 learner 按环境从轨迹自动学习，
  属数据问题非设计问题。
- 45.8% vs 31.3% 的 +14.5pp 说明 learned 优于 fallback 但仍有差距（数据覆盖），
  与 DLR 0.926 结合看：机制成立，补齐覆盖后收益可期。
- 预期幅度：剪枝 + 组合应能覆盖 read_note/find_api_key 剩余失败（3/5、2/5 → 5/5 级），
  且零探索回归（STRIPS 只加信息不改主循环）。本方向按任务范围仅做分析，未实现。

## 8. 验收核对

- [x] `results/phase8_qw_report.md`：9 任务修复前后对比（count_lines 单列，§6）。
- [x] 改动归因分离（§5 消融表：matrix / revisit / guard 各自贡献）。
- [x] STRIPS 接入点分析（§7：文件+函数+预期收益）。
- [x] per-episode JSONL 落盘（§6 文件清单）。
- [x] 62.2% → 86.7%（+24.4pp）；count_lines 0% → 80%（>0%）。
- [x] 本地 Docker 沙盒（peda-sandbox:v2/v4）完成全量 5ep×9task 评估。
- [x] 提交到 dev（commit bdc1f68 + 91a0c3c）。

> 注：commit 91a0c3c 修复了 budget guard 在 CI 模式（`Phase8Runner ci=True`，
> base `NoveltyExplorer` 无 `cd_child`）下的 AttributeError——guard 现仅对正常沙盒生效，
> CI 模式与改前行为逐位一致（同生成器、同 base explorer、无候选过滤），
> 已用 `read_secret_ci` 实跑验证无崩溃。

## 产出文件

- `src/phase8/count_driven_agent.py`（改动 1+2，含 EXPLORER_CLS/BUDGET_GUARD 可测接缝）
- `scripts/phase8_qw_eval.py`、`scripts/phase8_qw_ablation.py`、`scripts/phase8_qw_probe.py`
- `results/phase8_qw_{baseline,fixed,fixed2,matrix_only,all}/`（每任务 JSONL + summary.csv + meta.json）
- 本报告 `results/phase8_qw_report.md`
