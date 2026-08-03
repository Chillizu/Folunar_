# Phase 9 E1 — FF-SBH-3：Sandbox-HH 高层 R1 修复（空目录强制重选）全量重跑

**日期:** 2026-08-03
**作者:** L1-SBH3
**实验:** 90 集（λ=0 与 λ=0.5 各 45 集 = 9 任务 × 5 集，max_steps=10，原镜像），R1 修复后全量重跑
**依据:** `results/phase9_sbh_failure_analysis.md` R1 节（T2/T3 空目录陷阱）；合约 `local://contract-sbh3.md`
**结论:** **FF-SBH-3 PASS** — pooled λ-best 41/45（≥ 41），逐任务全部不劣于现行矩阵（find_api_key 3→4），零回归。修复生效且无意外行为。

---

## 0. 数据来源与复算对账

| 项 | 值 |
|---|---|
| per-episode JSONL | `results/phase9_sbh_r1_lam0.jsonl`、`results/phase9_sbh_r1_lam05.jsonl`（各 meta 头 + 45 集；`wc -l` = 46/46，90 集齐全，WATCHDOG D4） |
| JSONL meta commit | `51b4b703`（代码提交；R1 修复与脚本改动在同一提交，运行时 HEAD） |
| 复算脚本 | `scripts/phase9_sbh_analyze.py --r1`（从 JSONL 独立聚合 + FF-SBH-3 门判定，不读 runner 状态） |
| 复算结果 | 与 runner 输出逐任务全等：λ0 pooled 41/45、λ05 pooled 41/45、deep-path 7/10 |
| 现行基线矩阵 | `results/phase9_sbh_lam0.jsonl`（FF-SBH-2，40/45：read_hello 5, read_note 3, count_lines 4, find_secret 5, read_welcome 5, find_api_key 3, count_measurements 5, find_errors_v4 5, read_changelog_v4 5） |
| 运行环境 | CPU（本机），Docker 沙盒 `peda-sandbox:v2`/`v4`，max_steps=10，5 集/任务 |

## 1. 改动摘要（diff：`git show 51b4b70 --stat` = 4 文件 +121 −8）

### 1.1 R1 触发（`src/phase9/sandbox_hh/agent.py`，+25 −2）

- 新增判定函数 `_dir_lacks_text_files(state)`：当前 cwd 的目录图内无可读文本文件（无 `cat` 候选目标的 regular file；空目录为特例）。
- 状态机第 8 步重选条件扩展：`mode == "explore"` 时，原「局部 frontier 耗尽」**或**「本步 `cd` 落入无文本文件目录」→ 下一步强制进入 `select`（视为 frontier 耗尽等价条件）。
- 注释引用诊断 T2/R1：空目录陷阱（find_api_key ep1 `cd cache`）中低层 grep/find（priority 1 < cd 2）反复消耗预算，而 cd-only 候选使 density 恒 > 0，原 open-loop 重选永不触发。

### 1.2 重选目标排除（`src/phase9/sandbox_hh/planner.py`，+14）

- `select_goal` 中：**无文本文件的当前 cwd 不作为 goal 候选**（frontier 耗尽等价排除）。否则强制重选会再次选中 cache 自身（density=1.0、dist=0、goal==cwd → 不导航），低层继续困在空目录——正是诊断中「且优先于当前 cwd 重新选择高 density 目标」的要求。

### 1.3 脚本（`scripts/phase9_sandbox_hh.py`、`scripts/phase9_sbh_analyze.py`）

- runner：新增 `--out-prefix`（FF-SBH-3 重跑输出到 `results/phase9_sbh_r1_{lam0,lam05}.jsonl`）；meta 增补 `r1_fix` 字段、`density_definition` 说明 R1 触发与排除规则。
- analyze：新增 `--r1` 模式——读 r1 JSONL，输出逐任务新旧对照 + 按预注册阈值出 FF-SBH-3 门判定。

### 1.4 归因约束

- `src/phase8/`、`src/phase5/` **零改动**（`git diff 51b4b70^ 51b4b70 -- src/phase8 src/phase5` 为空）；低层仍逐行复用 Phase 8。

## 2. 逐任务新旧对照（FF-SBH-2 基线 vs R1 重跑）

| # | 任务 | 镜像 | 现行基线 (FF-SBH-2) | R1 λ=0 | R1 λ=0.5 | λ-best | Δ |
|:--:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | read_hello | v2 | 5/5 | 5/5 | 5/5 | 5 | 0 |
| 2 | read_note | v2 | 3/5 | 3/5 | 3/5 | 3 | 0 |
| 3 | count_lines | v2 | 4/5 | 4/5 | 4/5 | 4 | 0 |
| 4 | find_secret | v2 | 5/5 | 5/5 | 5/5 | 5 | 0 |
| 5 | read_welcome | v4 | 5/5 | 5/5 | 5/5 | 5 | 0 |
| 6 | find_api_key | v4 | **3/5** | **4/5** | **4/5** | **4** | **+1** |
| 7 | count_measurements | v4 | 5/5 | 5/5 | 5/5 | 5 | 0 |
| 8 | find_errors_v4 | v4 | 5/5 | 5/5 | 5/5 | 5 | 0 |
| 9 | read_changelog_v4 | v4 | 5/5 | 5/5 | 5/5 | 5 | 0 |
| | **POOLED** | | **40/45** | **41/45** | **41/45** | **41** | **+1** |
| | **DEEP (read_note+find_api_key)** | | **6/10** | **7/10** | **7/10** | **7** | **+1** |

- 无任何任务低于现行矩阵（`regressions: []`）。**唯一行为变化即诊断预期的 find_api_key ep1**（见 §4 轨迹级对账：90 集中仅此 1 集动作序列变化，其余 44 集逐位相同）。

## 3. 门判定（严格按合约预注册阈值，未事后调整）

- **PASS**：pooled λ-best ≥ 41/45 **且**逐任务 ≥ 现行矩阵（5,3,4,5,5,3,5,5,5）。
- **KILL**：任一回退 或 pooled < 40。
- 中间（40/45 无回退）→ **NULL**（修复未生效），如实记录。

**判定：`FF-SBH-3 PASS`**（reason：pooled_best=41/45 ≥ 41 且无任务回退，find_api_key 3→4）。

## 4. find_api_key ep1 轨迹修复前后对照（JSONL L28）

### 4.1 动作序列

| | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | 结果 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **修复前** | grep -ri error . | grep -r secret . | grep -r version . | find txt | find md | find log | `cd cache` | grep -r error . | grep -ri error . | grep -r secret . | ✗（10 步耗尽） |
| **修复后** | grep -ri error . | grep -r secret . | grep -r version . | find txt | find md | find log | `cd cache` | **`cd ..`** | **`cd docs`** | **`cat api_reference.md`** | **✓（10 步）** |

t0–t6 与修复前**逐位相同**（低层行为未被触碰）。t6 `cd cache`（空目录，仅 temp/sessions 两个空子目录）后：

- **t6 步末**：R1 触发——`cd` 落入无文本文件目录 → 下一步强制重选。
- **t7**：重选 `select_goal(/sandbox/cache)`；cache 因无文本文件被排除（planner R1），唯一 density=1.0 且可达的文本前沿 `/sandbox/docs`（10/10 未访问，dist=2，t4 的 `find . -name '*.md'` 已揭示）胜出 → 进入 navigate。
- **t7–t8**：沿 BFS 路径 `cd ..`、`cd docs` 导航（2 步）。
- **t9**：docs 内低层按 novelty/priority 选中 `cat api_reference.md`（t9 为 cat 非 cd，budget guard 不拦）→ 命中目标谓词（`_action_hits_target` docs/api_reference.md + exit 0 + 输出含 API_KEY）→ **成功**。

与诊断预期完全一致（t7 cd .. / t8 cd docs / t9 cat api_reference.md → 2 步导航 + 1 步命中）。λ=0 与 λ=0.5 轨迹相同（docs 在两臂均以「unvisited 10 > data 2 / logs 3」的并列打破胜出）。

### 4.2 goal_log 对照

| | 事件序列 |
|---|---|
| 修复前 | `[{t0 select /sandbox (density .3571, dist 0, unvisited 5/14)}]`（全程仅此一次 select） |
| 修复后 | `[{t0 select /sandbox}, {t7 select /sandbox/docs (density 1.0, dist 2, unvisited 10/10)}, {t8 arrive /sandbox/docs}]` |

## 5. 意外/日志级行为变化（如实记录，均无动作或成败影响）

1. **find_api_key ep2–ep4 的 goal_log 清空**：ep1 成功后 docs 状态进入 `success_cache`，ep2–ep4 走低层重放路径（`cd docs` + `cat api_reference.md`，2 步），高层不再发言——与修复前 ep3–ep4 的重放行为一致（修复前 ep2 因 ep1 未成功而走导航路径）。动作序列逐位不变（均为 2 步成功）。
2. **find_errors_v4 ep0 的 select 日志变化**：起始 cwd `/sandbox/logs`（根层无文本文件，仅 system/app/audit 子目录）从 `select /sandbox/logs`（== cwd，不导航）变为 `select None`（planner R1 将无文本文件 cwd 排除）。**动作序列不变**（grep -r error . → grep -ri error . 成功，2 步）；R1 触发仅限「cd 进入」场景，find_errors_v4 无 cd，不受影响。
3. **λ 维度**：R1 后两臂仍逐任务全等（同 FF-SBH-2），继续如实记录不强行解读。

## 6. 验收清单自检

- ✓ 90/90 集落盘（`wc -l` 46/46 × 2，meta 含 git commit `51b4b70`，D4）
- ✓ 逐任务矩阵与报告一致（§2 与 `analyze --r1` 输出全等）
- ✓ 门判定明确：**FF-SBH-3 PASS**（§3）
- ✓ 轨迹级对账：90 集中仅 find_api_key ep1 动作变化（§4），零回归
- ✓ `git show --stat` 范围干净（仅 `src/phase9/sandbox_hh/{agent,planner}.py` + 2 个脚本 + 本报告）
- ✓ `git diff` src/phase8 为空（归因约束）
- ✓ 跳过 lint/全量测试（合约允许）
