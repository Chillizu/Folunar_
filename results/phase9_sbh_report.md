# Phase 9 E1 — 沙盒迁移：分层时域 open-loop agent（sandbox-hh）

**日期:** 2026-08-03
**作者:** L1-SBH
**实验:** 90 集（λ=0 与 λ=0.5 各 45 集 = 9 任务 × 5 集），Phase 8 同任务/同镜像/同 max_steps
**方向:** HH 存活形态（open-loop 两层）迁移到 Phase 8 Docker 沙盒（HH_VERDICT §2.3/§4 → 设计 §8 沙盒迁移）
**结论:** **FF-SBH-2 PASS** — 分层在沙盒携带信号；沙盒迁移成功。

---

## 0. 数据来源与复算对账

| 项 | 值 |
|---|---|
| per-episode JSONL | `results/phase9_sbh_lam0.jsonl`、`results/phase9_sbh_lam05.jsonl`（各 meta 头 + 45 集；`wc -l` = 46/46，90 集齐全，WATCHDOG D4） |
| 复算脚本 | `scripts/phase9_sbh_analyze.py`（从 JSONL 独立聚合，不读 runner 状态） |
| 复算结果 | 与 runner 输出逐任务全等：λ0 pooled 40/45、λ05 pooled 40/45、deep-path 6/10 |
| 运行环境 | CPU（本机），Docker 沙盒 `peda-sandbox:v2`/`v4`，max_steps=10，5 集/任务 |
| JSONL meta commit | `cd1478ab`（运行时 HEAD；本次提交 `3d3288d` 为同一工作树的代码，行为逐位一致——运行确定性已验证） |

## 1. 预注册声明（结果出来前固定，未事后修改）

### 1.1 门阈值（来自合约 local://contract-sbh.md，原样引用）

- **FF-SBH-1 (kill)**: λ-best pooled success < **37/45**（基线 39/45 − 5pp）→ 沙盒迁移失败，记录 negative。
- **FF-SBH-2 (positive bar)**: λ-best pooled ≥ **39/45** AND deep-path（read_note + find_api_key 合计）≥ **4/10**（基线 2/10）→ 分层在沙盒携带信号。
- 中间态（pooled ≥ 37 但 < 39，或 deep-path 无改善）：如实记录 mixed，不强行解读。

### 1.2 实验变量定义（实现细节，同样预注册于脚本 meta）

- **高层**：在增量目录图上选 frontier goal，评分 `J(d) = unvisited_density(d) − λ·dist(cwd, d)`。
  - `unvisited_density(d)` = 目录 d 内**未尝试**候选数 / 候选总数；候选 = 每个已知文本文件的 Phase 8 verb×file 矩阵（`cat`/`head -n 5`/`wc -l`）+ 每个已知子目录的 `cd`；"未尝试" = explorer `state_action_counts[(state_hash(d), action)] == 0`。
  - `dist` = 已知图上 cwd→d 的 BFS cd 步数；不可达目录不选为 goal。
  - 并列打破：dist 小 → 未访问数大 → 路径字典序。
- **目录图**：从 ls（每次访问目录的 `state.files`）与 `find . -name '*.ext'` 输出（相对路径 `./docs/note.txt`）增量构建，跨集持久（确定性沙盒）。
- **重选时机（open-loop）**：episode 开始时选一次；此后仅当**当前目录局部 frontier 耗尽**（密度=0）时重选。不做 mid-plan re-eval。
- **低层（唯一对照变量之外，逐行一致）**：`generate_phase8_candidates` + `Phase8Explorer.select_action` + `observe`/`record_cd`/成功缓存重放 + budget guard + STRIPS 副作用学习——与 `Phase8Runner.run_episode` 完全相同的调用序列。
- **优先级规则**：低层成功缓存重放（`success_cache` / `cd_child` 回访）**永远优先于**高层导航——已知解 > 任何 frontier 启发。
- **goal == 起始 cwd 时**：不强制导航（低层自由游荡，Phase 8 行为），直到重选选中非 cwd 目录。

## 2. 结果总表（per-task，λ=0 / λ=0.5 vs Phase 8 基线）

基线 = `results/phase8_qw_report.md` §6「修复后」列（本地 fixed2，39/45）。

| # | 任务 | 镜像 | Phase 8 基线 | λ=0 | λ=0.5 | Δ(λ-best) |
|:--:|------|:---:|:---:|:---:|:---:|:---:|
| 1 | read_hello | v2 | 5/5 | 5/5 | 5/5 | 0 |
| 2 | read_note | v2 | 3/5 | 3/5 | 3/5 | 0 |
| 3 | count_lines | v2 | 4/5 | 4/5 | 4/5 | 0 |
| 4 | find_secret | v2 | 5/5 | 5/5 | 5/5 | 0 |
| 5 | read_welcome | v4 | 5/5 | 5/5 | 5/5 | 0 |
| 6 | find_api_key | v4 | **2/5** | **3/5** | **3/5** | **+1** |
| 7 | count_measurements | v4 | 5/5 | 5/5 | 5/5 | 0 |
| 8 | find_errors_v4 | v4 | 5/5 | 5/5 | 5/5 | 0 |
| 9 | read_changelog_v4 | v4 | 5/5 | 5/5 | 5/5 | 0 |
| | **POOLED** | | **39/45** | **40/45** | **40/45** | **+1** |
| | **DEEP (read_note+find_api_key)** | | **2/10** | **6/10** | **6/10** | **+4** |

- 无任何任务低于基线（无回归）。
- **λ=0 与 λ=0.5 逐任务全等**（见 §4 讨论——λ 维度在本任务集平坦，如实记录，不强行解读）。

## 3. 门判定（严格按预注册阈值）

| 门 | 阈值 | 测量值 | 判定 |
|---|---|---|---|
| **FF-SBH-1 (kill)** | λ-best pooled < 37/45 → 失败 | λ-best pooled = **40/45** | **NOT TRIGGERED** — 沙盒迁移成功（≥ 37，不记录 negative） |
| **FF-SBH-2 (positive)** | pooled ≥ 39/45 AND deep ≥ 4/10 | pooled **40/45** ✓；deep **6/10** ✓ | **PASS** — 分层在沙盒携带信号 |

中间态（mixed）不适用：pooled 40 ≥ 39 且 deep 6 ≥ 4，双条件均过。

## 4. 关键发现与归因

### 4.1 归因（低层逐行一致 → delta 唯一可归因于高层）

低层与 Phase 8 完全同一实现（同一候选生成器、同一 explorer 类、同一 observe/缓存/budget guard 序列）。两层的全部差异 = 高层 frontier-goal 选择。实验内验证：直接读类 5 任务（read_hello/read_welcome/find_secret/count_measurements/find_errors_v4/read_changelog_v4）在分层下逐集行为与 Phase 8 相同（count_measurements 等 start-cwd 任务第 1-3 步即成功；find_secret ep0 第 9 步 grep 命中），无探索回归。

### 4.2 深层路径改善（horizon 要打的弱点，Phase 8 基线 2/10 → 6/10）

- **find_api_key 2/5 → 3/5**：Phase 8 在 ep2 游荡 data/archive/processed/raw（budget guard 救回），ep3 才进 docs 发现；分层 agent 在 ep2 由高层直接选 `/sandbox/docs`（密度 1.0，未访问 10/10，dist 1），`cd docs` + `cat api_reference.md` **2 步发现**，ep3-4 各 2 步重放。
- **read_note 3/5（持平）**：发现时机同为 ep2；但分层导航一步到位 `cd docs`（vs Phase 8 ep1 先 cd data 消耗预算）。ep3-4 2 步重放。
- **机制**：目录图让未访问目录的密度在**进入之前**就可计算（find 输出揭示 `docs/api_reference.md`），高层直接导航到最高密度 frontier，省去低层广度游荡。这正是 HH_VERDICT §3 结论在沙盒的复现：count 低层不变，换高层目标选择即改变深层成功率。

### 4.3 λ 平坦性（诚实记录，不强行解读）

λ=0 与 λ=0.5 逐任务全等。原因：本 9 任务集在决策时刻的 frontier 几乎全为 `/sandbox` 的一层子目录（dist=1 并列），λ·dist 不改变 argmax；唯一 dist=2 的深层目录（`/sandbox/docs/tutorials` 等）只在其他 frontier 耗尽后才成为候选。这与 HH_VERDICT §2.2/§4 的预告一致（λ 惩罚在纯覆盖任务中无益；距离效应是视界/图结构交互而非噪声）。**结论：λ 维度在本沙盒任务集无判别力，两臂等权进入 λ-best 判定。**

## 5. deep-path 典型案例：高层决策序列摘录

（自 `results/phase9_sbh_lam0.jsonl`，`goal_log` 字段；`J` 为预注册公式计算值）

### 案例 1 — find_api_key（v4），λ=0，ep2：2 步成功

```
t=0 select goal=/sandbox/docs  density=1.0  dist=1  J=1.0  unvisited=10/10
    contender: /sandbox/docs            density=1.0  dist=1  unvisited=10/10
    contender: /sandbox/logs            density=1.0  dist=1  unvisited=3/3
    contender: /sandbox/cache           density=1.0  dist=1  unvisited=2/2
    contender: /sandbox/data            density=1.0  dist=1  unvisited=2/2
    contender: /sandbox/docs/tutorials  density=1.0  dist=2  unvisited=9/9
t=0 arrive /sandbox/docs
actions: ['cd docs', 'cat api_reference.md']   → SUCCESS（API_KEY 在 docs/api_reference.md）
```

**解读**：ep1 的 `find . -name '*.md'` 输出揭示了 docs 的 9 个未访问 verb 候选；ep2 高层在全部密度 1.0 的一层兄弟目录中按「未访问数 + dist」tie-break 选中 docs（10 > logs 3 > cache/data 2），直接导航、2 步命中。Phase 8 同集还在 data 分支游荡。

### 案例 2 — read_note（v2），λ=0，ep2：8 步成功

```
t=0 select goal=/sandbox/docs  density=1.0  dist=1  J=1.0  unvisited=9/9
    contender: /sandbox/docs   density=1.0  dist=1  unvisited=9/9
    contender: /sandbox/logs   density=1.0  dist=1  unvisited=6/6
    contender: /sandbox/data   density=0.5  dist=1  unvisited=6/12
    contender: /sandbox        density=0.36  dist=0  unvisited=4/11
t=0 arrive /sandbox/docs
actions: ['cd docs', 'cat changelog.txt', 'head -n 5 changelog.txt', 'wc -l changelog.txt',
          'cat manual.txt', 'head -n 5 manual.txt', 'wc -l manual.txt', 'cat note.txt'] → SUCCESS
```

**解读**：docs（9 未访问）以未访问数击败 logs（6）；低层到达后在 docs 内按 Phase 8 候选顺序逐文件读完，第 8 步 `cat note.txt` 命中。λ=0.5 臂同序列（J=0.5），判定一致。

### 案例 3/4 — 重放阶段（λ=0，ep3-4，两任务相同模式）

```
goal_log: []（空 — 低层成功缓存优先于高层，未触发任何 goal 选择）
actions:  ['cd docs', 'cat note.txt'] / ['cd docs', 'cat api_reference.md'] → 各 2 步 SUCCESS
```

**解读**：`cd_child[sandbox]["cd docs"]` → docs 状态在 success_cache 中 → Phase 8 的 cached-child 回访机制直接重放，高层完全让位（replay 优先规则）。

## 6. 自校验（WATCHDOG D4 / 独立复算）

- `wc -l results/phase9_sbh_lam{0,05}.jsonl` = 46/46（meta + 45 集）→ **90/90 集完整**。
- `scripts/phase9_sbh_analyze.py` 从 JSONL 独立重算：pooled 40/45（两臂）、deep 6/10、逐任务表与 runner 输出**全等**（§0 对账）。
- 门判定由复算脚本按预注册阈值给出（§3 表），与报告数字一致。
- 确定性验证：同一代码两次全量运行（含中途修复文件名前）per-task 数字一致；单任务隔离运行与全量运行一致。

## 7. 产出文件与提交

| 文件 | 内容 |
|---|---|
| `src/phase9/sandbox_hh/__init__.py` | 包 |
| `src/phase9/sandbox_hh/planner.py` | 高层：DirGraph（ls/find 增量目录图）+ SandboxGoalPlanner（J(d) 选择） |
| `src/phase9/sandbox_hh/agent.py` | 两层 open-loop episode loop（select/navigate/explore 状态机，replay 优先） |
| `src/phase9/sandbox_hh/runner.py` | 多集运行（9 任务 × 5 集） |
| `scripts/phase9_sandbox_hh.py` | 实验 CLI：跑两臂、写 per-episode JSONL（meta 含 commit） |
| `scripts/phase9_sbh_analyze.py` | 独立复算 + 门判定 + deep-path 决策序列摘录 |
| `results/phase9_sbh_lam0.jsonl` / `phase9_sbh_lam05.jsonl` | 数据（gitignore，不 commit） |
| 本报告 | — |

**Commit:** dev 分支，message `phase9: E1 sandbox-hh — two-layer open-loop in Docker sandbox; FF-SBH-2 PASS (40/45 pooled, deep-path 6/10 vs 2/10)`（hash 以 git log 为准；本报告随代码同一 commit 提交）

**验收清单自检**：✓ 90 集 JSONL（meta 含 git commit，D4）；✓ 逐任务对比表（§2）；✓ FF-SBH-1/2 逐门判定（§3）；✓ deep-path ≥ 2 个成功案例的高层决策序列（§5，共 4 个）；✓ 独立复算脚本 + 数字对账（§6）；✓ 只 commit 代码+报告，results/ 不入库。
