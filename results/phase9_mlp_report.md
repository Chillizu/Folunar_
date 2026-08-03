# Phase 9 FF-MLP-1：路径级规划器判别实验（真深度树机制检验）

**日期:** 2026-08-03
**作者:** L1-MLP1
**实验:** FF-MLP-1 — 把 Sandbox-HH 高层从**单步 goal 选择**换成**路径级规划**，测机制在真深度树（v5，dist≥2）上到底有没有效果。单步规划的两个 T1 死因（dist-1 满密度前沿垄断 select + 未知子目录被排除）被针对性修复：候选=任意深度 frontier **路径**，J=末端未探索密度先验−λ·depth，**未知目录中性先验 0.5**（不再排除）。
**依据:** 合约 `local://contract-mlp1.md`（预注册门，禁止事后调整）；归因约束：`src/phase8/`、`src/phase5/`、旧 `planner.py`/`agent.py` 零 diff；低层逐行复用 Phase 8。
**数据:** `results/phase9_mlp_lam0.jsonl`、`phase9_mlp_lam05.jsonl`（各 meta 头 + 40 集，共 80/80 新集，WATCHDOG D4 meta commit = 代码 commit `f4dd350`）；baseline 引用 FF-GEN-1 `results/phase9_gen_{flat,sbh_lam0,sbh_lam05}.jsonl`。
**结论:** **FF-MLP-1 KILL** — MLP_best deep = **0/30** ≤ 1/30（预注册阈值），且 pooled 4/40 < 7/40，**两个条件同时不满足**。但机制行为**确实改变了**（T1 冷启动 7→0、λ 首次分叉、λ=0 臂成为 s10 下第一个机械到达 depth≥2 的臂）——「改变行为」与「产生 deep 成功」被证明是两回事：无任务知识时未知目录选择是字母序赌博，5 集预算内全部赌错。

---

## 0. 数据对账（80/80 可复核，确定性复现）

| 项 | 值 |
|---|---|
| per-episode JSONL | `phase9_mlp_lam0.jsonl` / `phase9_mlp_lam05.jsonl`，各 `wc -l`=41（meta 头 + 40 集），共 **80 集**新落盘 |
| JSONL meta commit | `f4dd350`（两臂一致；代码先提交后运行，meta commit = 运行时代码 commit） |
| baseline 引用 | FF-GEN-1 `phase9_gen_flat.jsonl` / `phase9_gen_sbh_lam0.jsonl` / `phase9_gen_sbh_lam05.jsonl`（commit `a217db8`，各 40 集） |
| 复算脚本 | `scripts/phase9_mlp_analyze.py`（新增，从 JSONL 独立聚合 + 四臂矩阵 + 门判定 + λ 分叉 + 失败模式 + select 机制统计）；dump `results/phase9_mlp_analysis.json` |
| 运行环境 | CPU（本机），Docker 沙盒 `peda-sandbox:v5`，8 任务 × 5 集/任务/臂（seeds 0-4），max_steps 10 |
| **确定性复现** | 代码提交后重跑全量 80 集 + 单独第三跑 gen_read_sensor 两臂各 5 集：动作序列与 success **逐位一致**（WATCHDOG D4 可复算 ✓） |
| 归因约束 | `git diff 44e71cf..HEAD -- src/phase8 src/phase5 src/phase9/sandbox_hh/planner.py src/phase9/sandbox_hh/agent.py src/phase9/sandbox_hh/runner.py src/phase9/sandbox_hh/__init__.py` **为空**；新增文件仅 `src/phase9/sandbox_hh/path_planner.py`，runner/分析脚本纯增量 |

## 1. 设计（合约定稿，executor 细化；tie-break 在跑任何结果前注册）

### 1.1 与单步规划（SBH）的差异

| 维度 | SBH（单步 goal） | MLP（路径级规划，本实验） |
|---|---|---|
| 候选 | 目录 d（known_dirs 中 density>0） | **路径**（cd 链 cwd→…→目标，任意深度） |
| 未知子目录 | density=0 → **排除**（T1 死因 b） | **中性先验 0.5，纳入候选**（修 T1 根源） |
| 评分 | J(d)=density−λ·dist(cwd,d) | J(path)=**prior(末端)**−λ·**depth**（depth=cd 步数） |
| 已访问目录 | 实际 density | 实际 density（同口径：verb×file 候选零计数比例） |
| tie-break | J 降 → **dist 升**（T1 死因 a） | J 降 → **depth 降**（深处优先，直击 dist-1 垄断）→ 路径字典序 |
| cwd 自身 | 可被选为 goal（goal==cwd → 低层自由游荡 = T1 冷启动） | **永不作为路径末端**（链意味着移动；导航 t=0 即刻启动） |
| 导航 | BFS 逐 cd | 同（BFS 链，允许穿过已访问父目录朝未知深处走） |
| R1 空目录重选 / open-loop / 低层 | 保留 | **原样保留**（继承 SandboxHHAgent 循环，仅换 planner） |

- **低层逐行复用 Phase 8**：`generate_phase8_candidates` + `Phase8Explorer`（wc 读者层 + 缓存成功子目录重访），零改动；replay 优先于高层、预算守卫、STRIPS 副作用学习全部继承。
- **实现**：`SandboxPathAgent(SandboxHHAgent)` 只覆写 `__init__`（换 `PathPlanner`）与 `_select_and_log`（日志扩展 path/prior 字段）；`PathPlanner` 复用 `DirGraph`（import，零改动）。已访问判定 = explorer.state_counts 对目录 state_hash 计数 > 0（跨集持久，与图一致）。
- **open-loop 语义**：路径选定后中途不重评；重选仅触发于（a）当前目录局部前沿耗尽（density≤0）、（b）R1——cd 进入无文本文件目录。两者均为原单步循环语义。

### 1.2 预注册门（记 MLP_best = max(λ0, λ05)）

- **PASS**：MLP_best deep（dist≥2 六任务子集，满分 30）≥ 4/30 **且** MLP_best pooled ≥ 7/40。
- **KILL**：MLP_best deep ≤ 1/30。
- **NULL**：deep 2-3/30（弱信号，如实记录）。
- 基线：flat deep 0/30、SBH-R1 deep 0/30（FF-GEN-1 实测）。
- 次级（无门）：λ 在路径规划下是否分叉；T1 冷启动失败数变化。

## 2. 逐任务四臂矩阵（JSONL 独立复算）

| # | 任务 | dist | flat | SBH λ=0 | SBH λ=0.5 | **MLP λ=0** | **MLP λ=0.5** | best |
|:--:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | gen_read_notes | 0 | **5/5** | **5/5** | **5/5** | 3/5 | 1/5 | 5 |
| 2 | gen_read_setup | 1 | 2/5 | 3/5 | 3/5 | 1/5 | **3/5** | 3 |
| 3 | gen_read_sensor | 2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0 |
| 4 | gen_read_usage | 2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0 |
| 5 | gen_find_api_ref | 2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0 |
| 6 | gen_read_audit | 2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0 |
| 7 | gen_count_readings | 2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0 |
| 8 | gen_find_error_deep | 2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0 |
| | **POOLED** | | **7/40** | **8/40** | **8/40** | **4/40** | **4/40** | 4 |
| | CONTROL（1-2） | | 7/10 | 8/10 | 8/10 | 4/10 | 4/10 | — |
| | DEEP（3-8，dist≥2） | | **0/30** | **0/30** | **0/30** | **0/30** | **0/30** | 0 |

## 3. 门判定（严格按预注册阈值，未事后调整）

- MLP_best deep = max(0, 0) = **0/30** ≤ 1/30 → **KILL（机制无效果）**。
- 附带：MLP_best pooled = max(4, 4) = **4/40 < 7/40** — PASS 的 pooled 条件同样不满足。两个条件同时失败，判定无歧义。
- 诚实措辞：路径级规划**没有在真深度树上产生任何 deep 成功**（0/30，与 flat/SBH 的 s10 同为零）；且整体成功率反而**低于**两个 baseline（4 < 7 < 8）——机制在 dist-0 控制任务上倒退了（notes 5/5 → 3/5、1/5），dist-1 只有 λ05 保持 SBH 水平（3/5）。**这不是「差一点」，是结构性零转化**：机制确实导航到了深处（见 §5），但全部落在非目标目录。

## 4. λ 次级问题：**首次分叉（8/40 逐位一致，历史首次 ≠ 40/40）**

| 指标 | FF-GEN-1（单步） | **FF-MLP-1（路径）** |
|---|---|---|
| 同任务同 episode 动作序列逐位一致 | 40/40 | **8/40** |
| pooled 差（λ05−λ0） | 0 | 0（但构成不同） |
| 逐任务差（λ05−λ0） | 全 0 | notes −2、setup +2，其余 0 |

- **λ 第一次在实验中改变决策**：λ=0 臂 53 次 select 中 **38 次选中 depth≥2 路径**（depth 分布 {1:15, 2:31, 3:7}），全部完成导航到达；λ=0.5 臂 38 次 select **全为 depth=1**（{1:38}，λ·depth 惩罚使任何 d2 未知（J=−0.5）永远输给 d1 未知（J=0.0））。分叉机制正是预注册设计：λ=0 靠 tie-break 深潜，λ=0.5 靠 J 浅驻。
- **分叉的真实代价**：λ0 的深潜全部赌错目录（app/src、app/templates、data/archive，§6），且空目录陷阱吃掉预算（trap 失败 13）；λ05 的浅驻把预算花在广度（app→data→docs→logs 四层各一集），对 dist-1 setup 反而有利（3/5 > λ0 的 1/5）——**λ 维度从「恒死」变成「有害分叉」**：两种策略都到不了任务目标，只是失败方式不同。

## 5. T1 变化与机制统计（关键实证）

| 臂 | 失败数 | cold_start（T1） | trap（空目录） | wrong_dir | dir_reached | 到过 d2 的集数 |
|---|---:|---:|---:|---:|---:|---:|
| flat | 33 | **7** | 0 | 26 | 0 | 0/40 |
| SBH λ=0 | 32 | **7** | 0 | 25 | 0 | 0/40 |
| MLP λ=0 | 36 | **0** | **13** | 23 | 0 | **38 次到达 d2 目标** |
| MLP λ=0.5 | 36 | **0** | 0 | 36 | 0 | 0 |

1. **T1 冷启动根治（机制层面）**：每臂 7 集 → **0 集**。路径规划在 t=0 就导航（`cd app`），10 步不再被根目录 6 次文件读 + 4 次 grep 烧光。FF-GEN-1 诊断的「cd 候选（priority 2）结构性不可达」被高层导航绕开——**修复在机制层真实生效**。
2. **s10 下第一个机械到达 depth≥2 的臂**：flat/SBH 在 s10 全实验 0 集到 d2（FF-CEIL-1 §2.3 同口径：flat/SBH s10 最大深度 d2 集数 = 0）；MLP λ=0 有 **38 次 depth≥2 目标选择且 38 次全部导航到达**（app/src、app/templates、data/archive、docs…）——**「到深处」被机械实现，但「到任务目标目录」为 0**。
3. **新失败模式：空目录陷阱 13 次（λ0）**：templates/archive 是字典序最先的 d2 未知目录（/sandbox/app/src < /sandbox/app/templates < /sandbox/data/…），λ0 的 depth 优先 tie-break 必然先赌 app 子树。R1 重选照常触发（t+2 步内重选，机制工作正常），但每例烧掉 2-4 步预算——**R1 在 MLP 下首次被高频触发并首次被检验：恢复正确，代价可观**。

## 6. 轨迹对照（成功 ≥2、失败 ≥2）

### 6.1 成功 ①：MLP λ=0 gen_read_notes ep2 — 根目录高密度拉回 + 重放

```
t0 cd app → t1 cd templates（空目录陷阱）→ t2 cd .. → t3 cd ..（R1 重选）
t2 select /sandbox  prior=0.9 depth=2 J=0.9   ← 根目录实际密度 0.9 压过未知 0.5
t4-t9: cat/head/wc README.md, cat/head/wc notes.txt → cat notes.txt ✓（t7）
ep3/ep4: cat notes.txt（success_cache 重放，1 步）
```

### 6.2 成功 ②：MLP λ=0.5 gen_read_setup ep2 — dist-1 直航（λ05 的 +2 来源）

```
t0 select /sandbox/docs  prior=0.5 depth=1 J=0.0  ← λ=0.5 惩罚使 d1 未知胜过一切 d2
t0 cd docs → t1 cat api.txt → t2 head → t3 wc → t4 cat setup.md ✓（5 步）
ep3/ep4: cd docs → cat setup.md（重放，2 步）
```
对照：MLP λ=0 同任务 setup 仅 1/5（ep4 经 archive 陷阱绕行才到 docs，§6.4）；λ05 的浅驻策略在 dist-1 任务上优于 λ0 的深潜策略。

### 6.3 失败 ①：MLP λ=0 gen_read_sensor ep1 — 深潜赌错目录（机制到位、目标未到）

```
t0 select /sandbox/app/src  prior=0.5 depth=2 J=0.5  path=['cd app','cd src']
   （app/src 是字典序第一的 d2 未知目录——选择正确执行，赌注落空）
t0 cd app → t1 cd src → t2-t9: core.py/settings.yaml/utils.py 9 次文件读耗尽
```
**到 d2 ✓，到 data/raw ✗** — 5 集窗口内 λ0 依次赌 app（ep0）→ app/src（ep1）→ app/templates（ep2 陷阱）→ data（ep3）→ data/archive（ep4 陷阱）→ docs（ep4 尾），**任务目标 data/raw 从未被选中**（8 个 d2 目录中 5 个是任务目标，但字典序先赌 app 子树——app/src、app/templates 连续两错，data/archive 又在 data/raw 之前）。

### 6.4 失败 ②：MLP λ=0 gen_read_sensor ep2 — 陷阱 + 根拉回双烧预算

```
t0 select /sandbox/app/templates prior=0.5 depth=2 J=0.5 → cd app, cd templates
t2 R1 重选 select /sandbox prior=0.9 depth=2 J=0.9     ← 已访问根密度 0.9 压过未知 0.5
   contenders: /sandbox 0.9 > data/docs/logs 0.5 > app/src 0.11
t2 cd .. → t3 cd .. → 到根 → t4-t9 6 次文件读 → 集尾仍停根目录
```
**已访问目录的实际密度（0.9）系统性压过未知先验（0.5）**：根目录每集只在 t=0 被观测一次 cd，密度降不下来，于是重选被反复拉回根——这是 J=实际密度−λ·depth 公式的**诚实后果**，也是 λ0 5 集窗口被压缩的机制主因之一。

### 6.5 失败 ③：MLP λ=0.5 gen_read_sensor ep4 — 浅驻扫完四层后回头

```
ep0 app → ep1 data → ep2 docs → ep3 logs（d1 未知全部扫完，全部 J=0.0 平局按字典序）
ep4 select /sandbox/app prior=0.25 depth=1 J=-0.25  ← d2 全部被 λ 惩罚排除，只能重访 app
```
λ05 从未在任何一集选中 d2（select 深度分布 {1:38}），**深潜能力被 λ 惩罚完全关闭**——与 λ0 构成对称的失败模式。

## 7. 意外行为如实记录

1. **根目录拉回（λ0 特有）**：已访问目录按实际 density 计分，根目录因每集仅 t=0 消费一次 cd 而长期保持 density≈0.9，重选时压过一切未知先验 → 回根清 frontier。设计公式的直接后果（非 bug），但把 5 集窗口压到只剩 ~3 个有效导航 slot。
2. **λ 维度「复活」但只有负价值**：FF-GEN-1 时代 λ 恒不生效（40/40 一致）；路径规划下 λ 首次分叉（8/40），但两臂分别以「赌错深目录」和「不敢深」两种方式失败——分叉 ≠ 增益。
3. **R1 首次被高频检验**：FF-GEN-1/CEIL 全实验 trap=0（陷阱在 d2，无人到达）；MLP λ0 以 13 次 trap 首次触发 R1——重选机制工作正常（t+2 内重选），但每次触发净损失 2-4 步，且在 10 步预算下是致命损失。
4. **与 FF-CEIL-1 交叉印证（机制墙证据链闭合）**：CEIL-1（s15/s20，commit 78783e0）显示预算 10→20 使 SBH deep 0→3/30、flat 0→1/30，且 s10 下两臂 0 集到 d2；本实验在**同一 s10 预算**下证明「即使给了机械深导航（38 次 d2 到达），无任务知识的选择仍全部落空」——**深 0/30 的瓶颈不是「到不了深处」也不是「预算」，而是「没有信息知道该去哪个深处」**（CEIL 的「机制残余主导」与本实验的「未知目录选择赌博」互相印证：CEIL 让预算松绑后 SBH 也只转化 3/30，剩余失败全是机制侧）。
5. **pooled 倒退的构成**：MLP 把成功从 dist-0（notes 5→3/1）搬向 dist-1（setup λ05 3/5）——导航代价是根文件读取被推迟/跳过，而深层又没补回来，净损失 3-4 集。

## 8. 结论

**KILL**：路径级规划在真深度树上没有产生 deep 效果（MLP_best deep 0/30 ≤ 1/30），pooled 亦低于所有 baseline（4/40 < 7/40 < 8/40）。**机制改动有效（T1 冷启动根治、λ 首次分叉、s10 首次机械到达 d2），但有效性全部消耗在「方向赌博」上**：未知目录先验 0.5 让所有未知候选平权，无任务知识时字典序/depth 优先的 tie-break 只是把赌博从「浅层垄断」换成「深层乱猜」（8 个 d2 目录中 5 个是任务目标，λ0 连赌 3 错）；10 步预算 + 5 集窗口不足以既清浅层又试对目标。修复 T1 的根源（排除未知）不等于修复 T1 的后果（到达正确深处）——**下一层机制需要的是方向信息（如 find 驱动的内容先验或目标相关线索），而非更激进的遍历策略**。

---

## 附：复现与对账

```bash
# 复现（代码已在 f4dd350，JSONL 不 commit）
PYTHONPATH=src venv/bin/python3 scripts/phase9_sandbox_hh.py --mlp
# 复算 + 门判定 + 轨迹
PYTHONPATH=src venv/bin/python3 scripts/phase9_mlp_analyze.py
PYTHONPATH=src venv/bin/python3 scripts/phase9_mlp_analyze.py --cases
```

- 80/80 新集落盘（两文件各 meta 头 + 40 集）；meta commit 均 `f4dd350` = 运行时代码 commit；确定性经全量重跑 + 单任务第三跑验证（动作/成功逐位一致）。
- 全部数字可复核：四臂矩阵与门判定 = `scripts/phase9_mlp_analyze.py` 从 JSONL 独立聚合；baseline 数字与 FF-GEN-1 报告一致（flat 7/40、SBH 8/40、deep 全 0）。
- 归因约束：`git diff 44e71cf..HEAD -- src/phase8 src/phase5 src/phase9/sandbox_hh/planner.py src/phase9/sandbox_hh/agent.py src/phase9/sandbox_hh/runner.py src/phase9/sandbox_hh/__init__.py` 为空；新增 `src/phase9/sandbox_hh/path_planner.py`；runner（`--planner`/`--mlp`）与分析脚本（`phase9_mlp_analyze.py`）纯增量，与 FF-CEIL-1 区域（`--max-steps`/`--ceil`/`phase9_ceil_analyze.py`）零重叠（IRC 协调确认）。
