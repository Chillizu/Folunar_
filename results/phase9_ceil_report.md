# Phase 9 FF-CEIL-1：预算/起点天花板探测（纯诊断）

**日期:** 2026-08-03
**作者:** L1-CEIL1
**实验:** FF-CEIL-1 — 分离 v5 深度树 0/30 死因：**预算墙**（10 步不够走 depth≥2 + 探索）还是**机制墙**（规划本身到不了深处）。零新功能代码，只改运行参数（`max_steps`），结果作为 FF-MLP-1 增益解释的基线。
**依据:** 合约 `local://contract-ceil1.md`（注册 Q1-Q3，口径禁止事后改）；归因约束：`src/phase8/`、`src/phase5/`、`src/phase9/sandbox_hh/` 零 diff；runner 纯增量加 `--max-steps`/`--ceil`（`94f51c6`），未改任何逻辑；分析脚本为新增独立文件 `scripts/phase9_ceil_analyze.py`（不动 FF-MLP-1 的 MLP 臂区域）。
**数据:** `results/phase9_ceil_flat_s15.jsonl`、`phase9_ceil_flat_s20.jsonl`、`phase9_ceil_sbh_s15.jsonl`、`phase9_ceil_sbh_s20.jsonl`（各 meta 头 + 40 集 = 160/160 新集落盘，WATCHDOG D4）；s10 引用 FF-GEN-1 的 `results/phase9_gen_{flat,sbh_lam0}.jsonl`（各 40 集）。
**结论:** **预算墙与机制墙并存，机制墙为主。** 预算 10→20 使 deep 子集从 0/30 变非零（flat 1/30、SBH 3/30，Q1 ✓），控制子集持续上升（Q3 ✓）——预算墙有真实贡献；但 s20 下 deep 成功率仍 ≤10%（27-29/30 失败），SBH 24 次 dist-2 目标选择仅转化 3 个成功，到达目标目录后 verb 优先级仍吃掉预算（3 例 dir_reached）——**机制墙未实锤（Q2 反向），但机制残余是主导瓶颈**。FF-MLP-1 的任何 deep 增益若 ≤3/30 可归因于预算松绑；>3/30 必须来自机制本身。

---

## 0. 数据对账（160/160 新集可复核）

| 项 | 值 |
|---|---|
| per-episode JSONL | `phase9_ceil_flat_s15.jsonl` / `phase9_ceil_flat_s20.jsonl` / `phase9_ceil_sbh_s15.jsonl` / `phase9_ceil_sbh_s20.jsonl`，各 `wc -l`=41（meta 头 + 40 集），共 **160 集** |
| JSONL meta commit | `94f51c6`（四文件一致；代码先提交后运行，meta commit = 运行时代码 commit） |
| s10 引用 | FF-GEN-1 `phase9_gen_flat.jsonl` / `phase9_gen_sbh_lam0.jsonl`（commit `a217db8`，40 集各），不重复计入 160 |
| 复算脚本 | `scripts/phase9_ceil_analyze.py`（新增，从 JSONL 独立聚合 + 失败分类 + Q1-Q3，不读 runner 状态）；dump `results/phase9_ceil_analysis.json` |
| 运行环境 | CPU（mioarch），Docker 沙盒 `peda-sandbox:v5`，8 任务 × 5 集/任务/臂，seeds 0-4 |
| 归因约束 | `git diff -- src/phase8 src/phase5 src/phase9/sandbox_hh` 为空；runner diff 仅 `--max-steps`/`--ceil`（+67/-21 纯增量） |

## 1. 方法与口径（注册，禁止事后改）

- **矩阵**：两臂（flat = Phase 8 count 基线 / SBH-R1 λ=0）× 三档预算（s10 引用 / s15 / s20 新跑），每档 40 集。
- **deep 子集**（Q1/Q2）= `DEEP_TASKS` 6 任务（目标 dist≥2，30 集）：gen_read_sensor、gen_read_usage、gen_find_api_ref、gen_read_audit、gen_count_readings、gen_find_error_deep。
- **dist-1 对照子集**（Q3）= `CONTROL_TASKS` 2 任务（目标 dist≤1，10 集）：gen_read_notes（dist-0）、gen_read_setup（dist-1）；同时单独报告纯 dist-1 任务 gen_read_setup（5 集）。
- **失败模式**：cold_start（全程 0 次 cd）/ trap（进空目录陷阱）/ wrong_dir（游荡至非目标目录）/ dir_reached（已到目标目录未命中）——分类器与 FF-GEN-1 逐字一致。
- 轨迹对照按 (task, episode) 槽跨预算对齐（同任务同 seed）；注意跨集缓存使不同预算档的历史不完全同构，槽对照为注册口径（每档自身确定性可复算）。

## 2. 预算-成功率曲线（两臂 × 三档 × deep/非deep 分解）

### 2.1 汇总曲线

| 臂 | 预算 | POOLED | DEEP（dist≥2，30 集） | CONTROL（dist≤1，10 集） | dist-1 单任务 setup（5 集） |
|---|---:|---:|---:|---:|
| flat | s10 | 7/40 | 0/30 | 7/10 | 2/5 |
| flat | s15 | 8/40 | 0/30 | 8/10 | 3/5 |
| flat | s20 | **10/40** | **1/30** | **9/10** | **4/5** |
| sbh | s10 | 8/40 | 0/30 | 8/10 | 3/5 |
| sbh | s15 | 10/40 | **1/30** | 9/10 | 4/5 |
| sbh | s20 | **12/40** | **3/30** | **9/10** | 4/5 |

### 2.2 逐任务矩阵（flat/sbh × s10/s15/s20 成功数）

| # | 任务 | dist | flat s10/s15/s20 | sbh s10/s15/s20 |
|:--:|---|---:|---:|---:|
| 1 | gen_read_notes | 0 | 5/5/5 | 5/5/5 |
| 2 | gen_read_setup | 1 | 2/3/4 | 3/4/4 |
| 3 | gen_read_sensor | 2 | 0/0/0 | 0/0/**1** |
| 4 | gen_read_usage | 2 | 0/0/0 | 0/**1**/0 |
| 5 | gen_find_api_ref | 2 | 0/0/0 | 0/0/0 |
| 6 | gen_read_audit | 2 | 0/0/**1** | 0/0/**1** |
| 7 | gen_count_readings | 2 | 0/0/0 | 0/0/**1** |
| 8 | gen_find_error_deep | 2 | 0/0/0 | 0/0/0 |

### 2.3 深度可达性（每 episode 最大 cwd 深度 + 到目标目录集数）

| 臂/档 | 最大深度 d0 集 | d1 集 | d2 集 | 轨迹曾到目标目录 |
|---|---:|---:|---:|---:|
| flat/s10 | 12 | 28 | 0 | 7/40 |
| flat/s15 | 11 | 29 | 0 | 8/40 |
| flat/s20 | 5 | 29 | **6** | 10/40 |
| sbh/s10 | 12 | 28 | 0 | 8/40 |
| sbh/s15 | 5 | 29 | **6** | 11/40 |
| sbh/s20 | 5 | 17 | **18** | 15/40 |

- s10 无任何 episode 到达 d2（与 GEN-1 报告一致）；s15 起两臂都有 d2 到达；s20 SBH 近半数（18/40）到 d2。
- **s20 deep 目标到达率**：flat 1/30（转化 1/1）；SBH 6/30（转化 3、dir_reached 3）——SBH 的 6 个深任务各恰有 1/5 集到达目标目录。

### 2.4 SBH goal_log：被选中目标 dist 分布（机制侧写）

| 档 | select 总数 | dist-0 | dist-1 | dist-2 | dist-4 |
|---|---:|---:|---:|---:|---:|
| s10 | 34 | 15 | 19 | **0** | 0 |
| s15 | 38 | 8 | 19 | **11** | 0 |
| s20 | 54 | 8 | 13 | **24** | **9** |

dist-2 选择 0 → 11 → 24：预算松绑直接解锁高层对深目标的**选择**（dist-1 前沿被消费后 tie-break 不再垄断）；dist-4 是跨分支导航（上溯再下潜，如 logs/system → data/raw = 4 cd 步），全部出现在 t=11/t=19 的集内重选。

## 3. Q1-Q3（注册口径，逐字回答）

### Q1：预算 10→20，deep 子集是否从 0/30 变非零？变多少？

**是，两臂均变非零。** flat：0/30 → **1/30**（+1，gen_read_audit）；SBH：0/30 → **3/30**（+3，gen_read_sensor、gen_read_audit、gen_count_readings）。中间档 s15：flat 0/30（未变）、SBH **1/30**（+1，gen_read_usage，全实验首个 deep 成功）。

### Q2：s20 仍 deep=0 → 机制墙实锤；现在 s20 deep≠0，机制墙实锤吗？

**未实锤（反向判定），但机制残余是主导瓶颈。** 注册逻辑「s20 仍 deep=0 → 机制墙实锤」的前件不成立：deep 非零，预算 10→20 确实带来 deep 增益，**预算墙有真实贡献**。但定量看：s20 下 27/30（flat）与 27/30（SBH）的 deep 集仍失败；SBH 的 24 次 dist-2 目标选择只转化 3 个成功（12.5%）；到达目标目录的 6 集中 3 集仍失败（dir_reached，见 §5 失败对照）——**机制墙未实锤 ≠ 无机制墙**，deep 瓶颈的主体仍在机制侧（候选生成不给路径式动作、tie-break/verb 优先级、open-loop 重选触发条件），预算只解释了 ≤3/30 的增量。

### Q3：dist-1 对照子集是否随预算继续上升？

**是，持续上升。** CONTROL（dist≤1，10 集）：flat 7/10 → 8/10 → **9/10**（+2）；SBH 8/10 → 9/10 → **9/10**（+1）。纯 dist-1 任务 gen_read_setup（5 集）：flat 2/5 → 3/5 → **4/5**（+2）；SBH 3/5 → 4/5 → **4/5**（+1）。对照子集（成功率 90%）与 deep 子集（≤10%）在预算翻倍后的涨幅完全不成比例——**全局预算不足存在（浅层吃预算就涨），但深度特异失败占主导**。

## 4. 失败模式迁移（随预算变化）

| 臂/档 | cold_start | trap | wrong_dir | dir_reached | FAIL 总数 |
|---|---:|---:|---:|---:|---:|
| flat/s10 | 7 | 0 | 26 | 0 | 33 |
| flat/s15 | 6 | 0 | 26 | 0 | 32 |
| flat/s20 | **0** | 0 | 30 | 0 | 30 |
| sbh/s10 | 7 | 0 | 25 | 0 | 32 |
| sbh/s15 | **0** | 0 | 29 | 1 | 30 |
| sbh/s20 | **0** | 0 | 25 | **3** | 28 |

- **cold_start 随预算消失**：s10 每臂 7 集（= 7 个任务的 ep0，根 10 步被 6 次文件读 + 4 次 grep 耗尽，cd 永不轮到）；s15 SBH 归零（高层导航即刻启动）、flat 剩 6；s20 两臂归零。T1 冷启动是**纯预算现象**，预算 ≥15 即可解。
- **dir_reached 只在长预算出现（SBH）**：s15 1 例、s20 3 例——到达目标目录但未命中（见 §5.3）：reader（priority 0）按文件序把剩余预算吃光，grep（priority 1）/目标文件（文件序第 3 位）未轮到。这是**到达之后的 verb/文件序优先级墙**，与 s10 的「到达之前」墙性质不同。
- trap 全档 0（v5 陷阱在 dist-2，虽 s20 SBH 已 18 集到 d2，仍无 `cd templates`/`cd archive`——R1 重选修复继续未受检验，如实记录）。

## 5. 轨迹对照（s10 失败 → s15/s20 成功：共 10 例，≥2 例达标）

```
[flat] gen_read_setup ep2:   s10 FAIL(10步) -> s15 OK(5步)
[flat] gen_read_setup ep1:   s10 FAIL(10步) -> s20 OK(20步)
[flat] gen_read_setup ep2:   s10 FAIL(10步) -> s20 OK(2步)
[flat] gen_read_audit ep4:   s10 FAIL(10步) -> s20 OK(14步)   ← deep
[sbh] gen_read_setup ep1:    s10 FAIL(10步) -> s15 OK(5步)
[sbh] gen_read_usage ep4:    s10 FAIL(10步) -> s15 OK(9步)    ← deep（首个）
[sbh] gen_read_setup ep1:    s10 FAIL(10步) -> s20 OK(5步)
[sbh] gen_read_sensor ep4:   s10 FAIL(10步) -> s20 OK(9步)    ← deep
[sbh] gen_read_audit ep3:    s10 FAIL(10步) -> s20 OK(14步)   ← deep（集内重选）
[sbh] gen_count_readings ep4: s10 FAIL(10步) -> s20 OK(5步)   ← deep
```

### 5.1 成功对照 ① — SBH 首个 deep 成功：gen_read_usage ep4 @s15

| | t0 | t1-t3 | t4-t6 | t7-t9 | 结果 |
|---|---|---|---|---|---|
| **SBH s15**（goal_log: t=0 select `/sandbox/docs/guides` density=1.0 **dist=2** J=1.0 unvisited=9/9） | `cd docs` | cd guides | cat/head/wc changelog.md | cat/head/wc faq.md → **cat usage.md** | **✓ 9 步** |
| **SBH s10 同槽** | cd data | cat/head/wc readings.csv | cat/head/wc summary.json | grep ×3 | ✗ 10 步 |

**为何 ep4 的 t=0 选中 dist-2**：前 3 集（s15 预算）已把 dist-1 的 docs 前沿消费到 density<1.0，docs/guides 以满密度 1.0 在 J 并列中胜出（tie-break dist 只在 J 相同时生效）——**dist-1 垄断的打破条件是「dist-1 前沿被消费」，而这需要每集预算 >10**。

### 5.2 成功对照 ② — SBH deep：gen_read_sensor ep4 / gen_count_readings ep4 @s20

| 任务 | t0 select | 轨迹 | 结果 |
|---|---|---|---|
| gen_read_sensor ep4 | `/sandbox/data/raw` dist=2 unvisited=6/6 | cd data → cd raw → cat/head/wc counts.txt → cat/head/wc labels.csv → **cat sensor.log** | **✓ 9 步** |
| gen_count_readings ep4 | `/sandbox/data/raw` dist=2 unvisited=6/6 | cd data → cd raw → **cat counts.txt** → head → wc | **✓ 5 步** |

### 5.3 成功对照 ③ — SBH deep + 集内重选：gen_read_audit ep3 @s20

| t | 事件 | 动作 |
|---|---|---|
| 0 | select `/sandbox/logs/system` dist=2 unvisited=9/9 | cd logs |
| 1-10 | 到达 system，读 debug/error/kernel × 3 verbs（9 步） | cd system + 9 reads |
| 11 | **重选** `/sandbox/logs/app` dist=2（本地前沿耗尽触发 open-loop 重选） | cd .. |
| 12-14 | cd app → **cat audit.log** | **✓ 14 步** |

s10 同槽在 cd logs + 6 次 reads 后预算耗尽——**s10 下 open-loop 重选从未在集内触发（GEN-1 报告），s20 触发后直达第二个深目标**。

### 5.4 成功对照 ④ — flat 无高层也破深：gen_read_audit ep4 @s20

| t | 动作 |
|---|---|
| 0-2 | cd docs → head/wc setup.md（知识积累：根 read 已被前序集消费） |
| 3-10 | grep ×4 + find ×3（priority 1） |
| 11-14 | cd .. → cd logs → cd app → **cat audit.log** | **✓ 14 步** |

flat 靠低层 cd novelty 轮转在 s20 首次到达 d2——低层在长预算下**能**深潜，但只转了 6/40 集到 d2、转化 1/30，无方向性。

### 5.5 失败对照 — dir_reached（到达目标仍失败，verb 优先级墙）：SBH gen_find_error_deep ep3 @s20

| t | 动作 |
|---|---|
| 0-1 | select `/sandbox/logs/system` dist=2 → cd logs → **cd system**（step 1 即到达目标目录） |
| 2-10 | 读 debug/error/kernel × 3 verbs（9 步，reader priority 0） |
| 11-19 | 重选 logs/app → cd app → 读 audit/requests × 3 verbs |
| 20 | 预算耗尽——**grep（priority 1）从未轮到**；该任务成功判据要求 grep+error 落在 system |

同档另 2 例 dir_reached：gen_read_usage ep2 @s20（step 16 到 guides，changelog/faq × 3 verbs 吃掉剩余 4 步，usage.md 文件序第 3 位未轮到）、gen_find_api_ref ep4 @s20（step 14 到 ref，glossary/manual × 3 verbs 吃掉 6 步，api.txt 未轮到）。**到达 ≠ 命中：候选排序的 verb/文件序优先级在预算内把目标动作挤出**——这是 s20 才暴露的第二道机制墙。

## 6. 机制归因：预算墙 vs 机制墙（定量）

**预算墙成分（预算 10→20 可解释的增量）：**
1. deep 0→非零：flat +1/30、SBH +3/30（Q1）；
2. CONTROL 持续上升：flat +2/10、SBH +1/10（Q3）；
3. cold_start 7→0 消失（T1 是纯预算现象）；
4. dist-2 目标选择 0→24 解锁、d2 到达 0→6（flat）/18（SBH）集。

**机制墙成分（预算翻倍仍无法解释的失败）：**
1. s20 仍有 27-29/30 deep 失败（deep 成功率 ≤10% vs CONTROL 90%）；
2. SBH 24 次 dist-2 选择只转化 3 成功（12.5%），19/24 次选中的深目标不是任务目标或到得太晚（t=19 的重选剩 1 步无法导航）；
3. 到达目标目录后 3/6 仍失败（verb/文件序优先级墙，§5.5）；
4. gen_find_api_ref / gen_find_error_deep 两任务在 s20 全档 0/10——dist-2 find/grep 类任务的失败与预算无关（结构：候选生成对 find/grep 目标无路径式动作 + 动词优先级）；flat 无高层方向性，s20 深目标到达率仅 1/30。

**判定：预算墙与机制墙并存；机制墙是主导。** 预算 10→20 对 pooled 的边际 = flat +3/40（7→10）、SBH +4/40（8→12），其中 deep 仅贡献 +1/+3、CONTROL +2/+1；按比例 deep 涨幅（3-10%）远低于 CONTROL（10-20%）。「深度全灭」在 s10 是预算与机制共同造成的，但把预算翻倍后 deep 仍在个位数——**0/30 的绝大部分（≥90%）由机制墙解释**。

## 7. 结论与对 FF-MLP-1 的含义

1. **Q1=是（flat 1/30、SBH 3/30 @s20），Q2=机制墙未实锤，Q3=是（control 9/10）**——预算墙有真实贡献，机制墙是主导瓶颈。
2. **对 FF-MLP-1 的解释约束**：MLP 臂若与 flat/SBH 同预算对照，预算松绑最多可解释 **deep +1~3/30、CONTROL +1~2/10** 的增量；任何 deep 增益 **>3/30**（或 CONTROL >9/10 上限）必须归因于 MLP 机制本身（目标选择/候选生成/重选触发），不能算在预算头上。
3. **下一轮机制抓手**（本实验定位的、预算无关的失败点）：(a) 候选生成不给路径式动作（`cat data/raw/sensor.log` 永不被生成 → 深目标依赖 cd 导航）；(b) 到达后 verb/文件序优先级把目标动作挤出预算（dir_reached 3 例）；(c) open-loop 重选触发条件苛刻（本地前沿耗尽才重选，t=19 重选形同虚设）；(d) dist-2 find/grep 任务（api_ref、error_deep）s20 仍 0/10。

## 附：复现与对账

```bash
# 复现（runner 已含 --max-steps/--ceil，纯增量；JSONL 不 commit）
PYTHONPATH=src venv/bin/python3 scripts/phase9_sandbox_hh.py --ceil   # 160 集，~3.2 min
# 复算 + Q1-Q3（新增独立脚本，不动 FF-MLP-1 区域）
PYTHONPATH=src venv/bin/python3 scripts/phase9_ceil_analyze.py
PYTHONPATH=src venv/bin/python3 scripts/phase9_ceil_analyze.py --cases
```

- 160/160 新集 = 4 文件 ×（meta 头 + 40 集）；meta commit 均 `94f51c6`（代码先提交后运行）；s10 引用 commit `a217db8`。
- 归因约束：`git diff -- src/phase8 src/phase5 src/phase9/sandbox_hh` 为空；runner 仅纯增量 `--max-steps`/`--ceil`（commit `94f51c6`，+67/-21）；分析脚本新增 `scripts/phase9_ceil_analyze.py`。
- 跳过 lint/全量测试（纯诊断实验，无 PASS/KILL 门）。
