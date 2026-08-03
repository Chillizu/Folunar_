# Phase 9 FF-GEN-1：Sandbox-HH 泛化判别实验（新镜像 v5 + 8 新任务，三臂 120 集）

**日期:** 2026-08-03
**作者:** L1-GEN1
**实验:** FF-GEN-1 — 判别 Sandbox-HH（R1 版）是**机制**还是 9 任务测试床**过拟合**：全新镜像 `peda-sandbox:v5`（depth 2-3）+ 8 个全新任务，三臂同镜像同任务同 seeds 0-4（max_steps 10，每任务 5 集）零改动直跑。
**依据:** 合约 `local://contract-gen1.md`（预注册门，禁止事后调整）；归因约束：`src/phase8/`、`src/phase5/`、`src/phase2/`、`src/phase9/sandbox_hh/` 全部零改动。
**数据:** `results/phase9_gen_flat.jsonl`、`phase9_gen_sbh_lam0.jsonl`、`phase9_gen_sbh_lam05.jsonl`（各 meta 头 + 40 集，120/120 落盘，WATCHDOG D4）。
**结论:** **FF-GEN-1 PASS（弱）** — SBH_best 8/40 ≥ count 7/40（非劣，按预注册阈值判定 PASS）。但判别器在深度维**未触发**：dist≥2 的 6 个任务全臂 **0/30**；SBH 的 +1 全部来自 dist-1 对照任务（gen_read_setup 2→3）。机制在新树未崩，但「深度泛化」无证据，如实记录为弱 PASS。

---

## 0. 数据对账（120/120 可复核）

| 项 | 值 |
|---|---|
| per-episode JSONL | `phase9_gen_flat.jsonl` / `phase9_gen_sbh_lam0.jsonl` / `phase9_gen_sbh_lam05.jsonl`，各 `wc -l`=41（meta 头 + 40 集），共 120 集 |
| JSONL meta commit | `a217db8`（三臂一致；代码提交后运行，运行时 HEAD = 代码 commit） |
| 复算脚本 | `scripts/phase9_gen_analyze.py`（从 JSONL 独立聚合 + 门判定，不读 runner 状态），复算与 runner 输出逐任务全等 |
| 运行环境 | CPU（本机），Docker 沙盒 `peda-sandbox:v5`，max_steps=10，5 集/任务/臂 |
| 归因约束 | `git diff -- src/phase8 src/phase5 src/phase2 src/phase9/sandbox_hh` 为空；任务为运行时数据注册（`MICRO_TASKS.extend(GEN_TASKS)`） |

## 1. 镜像与任务设计

### 1.1 `peda-sandbox:v5` 目录树（Dockerfile.busybox_v5，build 于 2026-08-03）

```
/sandbox
├── README.md, notes.txt            ← 根 2 文本文件
├── app/    (main.py, config.ini)   ← 4 内容子目录，各含 2 个子子目录
│   ├── src/      (core.py, utils.py, settings.yaml)
│   └── templates/  ← 空目录陷阱 #1
├── data/   (readings.csv, summary.json)
│   ├── raw/      (sensor.log, counts.txt, labels.csv)
│   └── archive/  ← 空目录陷阱 #2
├── docs/   (setup.md, api.txt)
│   ├── guides/   (usage.md, faq.md, changelog.md)
│   └── ref/      (manual.md, glossary.txt)
└── logs/   (boot.log, access.log)
    ├── system/   (kernel.log, error.log, debug.log)
    └── app/      (requests.log, audit.log)
```

- **结构验证**（`docker run --rm peda-sandbox:v5 find`）：总文本文件 **26**（25-35 ✓），最大深度 **3**（depth 2-3 ✓），空目录 **2**（app/templates、data/archive，≥2 ✓），根 = 2 文本文件 + 4 子目录（✓）。
- 内容风格沿用 v2/v4（note/log/key/data），`cat`/`grep`/`wc` 语义一致；与 v2/v4 **零文件名/内容重叠**（防记忆迁移）。内容含任务谓词所需标记：sensor.log 含 `SENSOR`、usage.md 含 `Version`、api.txt 含 `API_KEY=v5-secret-7711`、audit.log 含 `AUDIT`、counts.txt 恰 12 行、error.log 含 3 条 `ERROR`。

### 1.2 8 个新任务（`src/phase9/gen_tasks.py`，全部 /sandbox 起始）

| # | 任务 id | 类别 | 目标（dist） | 成功判据（复用既有机制） |
|:--:|---|---|---:|---|
| 1 | gen_read_notes | direct-read 对照 | notes.txt（dist-0） | reader 命中 + exit 0（同 read_hello） |
| 2 | gen_read_setup | direct-read 对照 | docs/setup.md（dist-1） | reader 命中 + exit 0（同 read_note） |
| 3 | gen_read_sensor | deep-path | data/raw/sensor.log（dist-2） | reader 命中 + exit 0 + 输出含 SENSOR |
| 4 | gen_read_usage | deep-path | docs/guides/usage.md（dist-2） | reader 命中 + exit 0 + 输出含 version（同 read_user_guide） |
| 5 | gen_find_api_ref | deep-path find | docs/ref/api.txt（dist-2） | reader 命中 + exit 0 + 输出含 API_KEY（同 find_api_key） |
| 6 | gen_read_audit | deep-path | logs/app/audit.log（dist-2） | reader 命中 + exit 0 + 输出含 AUDIT |
| 7 | gen_count_readings | grep/count deep | data/raw/counts.txt（dist-2） | `wc -l` 命中 + exit 0（同 count_lines） |
| 8 | gen_find_error_deep | grep/count deep | logs/system/error.log（dist-2） | grep+error + exit 0 + **定位约束**（动作含 system 或 cwd==/sandbox/logs/system；根目录 `grep -r error .` 不算 deep） |

- 判据谓词为**纯新增**（phase9 模块内），`_action_hits_target`/`_is_file_reader` 直接复用 `phase2.tasks`（只 import，不改）。
- 深任务候选生成不变（`generate_phase8_candidates` 只对 `state.files` 生成 verb×file；`cat data/raw/sensor.log` 这类路径式动作永远不会被生成 → dist≥2 目标**构造性**要求至少 2 次 cd）。

## 2. 逐任务三臂矩阵（JSONL 独立复算）

| # | 任务 | dist | flat | SBH λ=0 | SBH λ=0.5 | best |
|:--:|---|---:|---:|---:|---:|---:|
| 1 | gen_read_notes | 0 | **5/5** | **5/5** | **5/5** | 5 |
| 2 | gen_read_setup | 1 | 2/5 | **3/5** | **3/5** | 3 |
| 3 | gen_read_sensor | 2 | 0/5 | 0/5 | 0/5 | 0 |
| 4 | gen_read_usage | 2 | 0/5 | 0/5 | 0/5 | 0 |
| 5 | gen_find_api_ref | 2 | 0/5 | 0/5 | 0/5 | 0 |
| 6 | gen_read_audit | 2 | 0/5 | 0/5 | 0/5 | 0 |
| 7 | gen_count_readings | 2 | 0/5 | 0/5 | 0/5 | 0 |
| 8 | gen_find_error_deep | 2 | 0/5 | 0/5 | 0/5 | 0 |
| | **POOLED** | | **7/40** | **8/40** | **8/40** | **8** |
| | CONTROL（1-2） | | 7/10 | 8/10 | 8/10 | 8 |
| | DEEP（3-8，dist≥2） | | **0/30** | **0/30** | **0/30** | 0 |

- λ=0 与 λ=0.5 两臂 **40/40 动作序列逐位一致**（见 §4）。
- 旧测试床对照（记忆，不入门）：flat 9 任务 39/45、SBH+R1 41/45——v5 新树对**所有臂**都是剧烈降级（flat 39→7，SBH 41→8），深度任务全部不可达。

## 3. 门判定（严格按预注册阈值，未事后调整）

记 `count` = flat pooled（满分 40），`SBH_best` = max(λ=0, λ=0.5) pooled：
- **PASS**：SBH_best ≥ count；**KILL**：count − SBH_best ≥ 5pp（≥2 集）；**NULL**：其余。

**判定：`FF-GEN-1 PASS`**（count=7/40，SBH_best=8/40，8 ≥ 7，非劣）。

**如实记录（弱 PASS，不夸大）：**
1. **判别器未触发深度维**：设计上唯一的判别点是 depth≥2（λ 首次生效 + 深路径），但 dist≥2 任务全臂 0/30——本实验**没有产生任何深度维证据**，不能据此宣称「深度泛化成立」。
2. **+1 的机制可解释**（§5.1）：SBH 高层在 ep2 就把已知满密度的 dist-1 前沿（docs）作为目标导航（`cd docs` 直达），flat 低层靠 cd novelty 轮转迟到一集（ep3 才轮转到 docs）。这是「目标导向导航早一集」而非「深度能力」。
3. 结论措辞：**机制在新树未崩（SBH ≥ flat，非劣），但深度泛化证据不足**。若须二择一：本数据支持「非过拟合」（SBH 未像 flat 一样在深树崩盘到只剩对照任务，且其优势集中在需要方向感的 dist-1 任务），不支持也不否定「深度机制」。

## 4. λ 次级问题（注册问题，无门）

**结论：λ 在本树未分叉——40/40 动作序列逐位一致（`identical_action_pairs=40/40`），与旧测试床（FF-SBH-2 同轨迹）一致。**

机制归因（从 goal_log 统计，λ=0 臂）：全部 34 次 select 中 dist=0 共 15 次、dist=1 共 19 次、**dist=2 共 0 次**；t>0 的重选 **0 次**（全在 t=0 做一次，open-loop 重选从未在集内触发）。

- select 时刻的候选前沿要么只有 /sandbox（ep0/1 冷启动，dist-0），要么存在**满密度 dist-1 前沿**（docs/logs/data，ep2+ 知识积累后）——λ·dist 的惩罚（≤0.5）不足以翻转这些候选的 J 排序：
  - λ=0：docs（J=1.0, dist 1）与 docs/guides（J=1.0, dist 2）并列 → **tie-break dist 取胜的是 dist-1**；
  - λ=0.5：docs J=0.5 vs docs/guides J=0.0 → dist-1 直接胜。
- 两臂 argmax 恒同 → 行为恒同。**λ 只有在「dist-1 无满密度前沿、需在 dist-0 部分密度与 dist-2 满密度之间权衡」时才可能分叉——本树/本预算下该状态从未出现**（dist-1 满密度前沿在知识积累后总是存在）。

## 5. 失败模式与典型轨迹对照

### 5.1 失败模式统计（失败集分类，脚本自动判定）

| 臂 | 失败数 | cold_start（T1） | trap（空目录） | wrong_dir | dir_reached |
|---:|---:|---:|---:|---:|---:|
| flat | 33 | 7 | 0 | 26 | 0 |
| SBH λ=0 | 32 | 7 | 0 | 25 | 0 |
| SBH λ=0.5 | 32 | 7 | 0 | 25 | 0 |

- **T1 冷启动仍是主失败模式之一**（每臂 7 集 = 7 个任务的 ep0）：根 2 文本文件 = 6 次文件读（priority 0）+ 4 次 grep（priority 1）恰好耗尽 10 步，cd（priority 2）永不轮到——与旧测试床诊断（failure analysis §1.1）完全同构。
- **空目录陷阱零命中**（trap=0，全臂无任何 `cd templates`/`cd archive`）：新树陷阱位于 dist-2，10 步预算内无臂到达 → **R1 空目录重选修复在本实验未触发、未受检验**（意外行为，见 §6.1）。
- 无 dir_reached：没有任何臂在预算内抵达目标目录后功亏一篑——失败全部发生在「到达之前」。

### 5.2 成功对照 ①：gen_read_setup ep2（SBH +1 的来源，同集对照）

| | t0 | t1 | t2 | t3 | t4 | t5 | 结果 |
|---|---|---|---|---|---|---|---|
| **SBH λ=0**（goal_log: select /sandbox/docs density=1.0 dist=1 J=1.0） | `cd docs`（导航） | cat api.txt | head api.txt | wc api.txt | cat setup.md | — | **✓ 5 步** |
| **flat** | `cd data`（novelty 轮转） | cat readings.csv | head | wc | cat summary.json | head | ✗ 10 步耗尽 |

ep1 的 `find . -name '*.md'` 已把 docs 写入高层图（density 1.0, 8 候选全未访问）；ep2 t0 select 时**只有 SBH 高层把 docs 选为目标并导航**，低层（flat）按 cd novelty 轮转先去了 data。ep3-4 SBH 走 success_cache 重放（`cd docs`+`cat setup.md`，2 步，goal_log 空——重放优先于高层）。

### 5.3 成功对照 ②：gen_read_setup ep3（两臂「发现时序差一集」）

| | t0 | t1 | t2 | t3 | t4 | 结果 |
|---|---|---|---|---|---|---|
| **flat ep3** | `cd docs`（低层 novelty 终于轮转到 docs） | cat api.txt | head | wc | cat setup.md | **✓ 5 步** |
| **SBH ep3** | `cd docs` | cat setup.md | — | — | — | **✓ 2 步（缓存重放）** |

同一任务、同一环境：flat 靠纯低层 cd 轮转在 ep3 才发现 docs（ep1 app → ep2 data → ep3 docs，字母序/novelty 驱动）；SBH 在 ep2 靠高层目标导航提前一集，随后缓存重放。**+1 的全部来源。**

### 5.4 失败对照 ①：ep0 冷启动（全臂同轨迹，T1）

| t0-t5 | t6-t9 | 结果 |
|---|---|---|
| cat README.md, head, wc, cat notes.txt, head, wc（6 次文件读） | grep -r error / -ri error / -r secret / -r version | ✗ 10 步耗尽，0 次 cd |

三臂 8 任务 × ep0 的 deep/dist-1 任务全部如此（SBH goal_log: select /sandbox density=1.0 dist=0 == cwd → 不导航）。**cd 候选（priority 2）在 10 步内结构性不可达**——与旧测试床 read_note ep0 同构，且因 v5 根目录同样 2 个文本文件而完全复现。

### 5.5 失败对照 ②：gen_read_sensor ep2（SBH 深度不可达机制，dist-1 垄断）

SBH λ=0 ep2 goal_log 候选（select 时刻）：

| 候选 goal | density | dist | J(λ=0) | 是否选中 |
|---|---:|---:|---:|---|
| /sandbox/docs | 1.0 | 1 | 1.0 | **✓（tie-break dist 胜出）** |
| /sandbox/logs | 1.0 | 1 | 1.0 | ✗（并列后字母序 docs < logs） |
| /sandbox/data | 1.0 | 1 | 1.0 | ✗（unvisited 1 < 8） |
| /sandbox/docs/guides | 1.0 | **2** | 1.0 | ✗（tie-break dist 输） |
| /sandbox/logs/system | 1.0 | **2** | 1.0 | ✗ |

高层**知道** dist-2 满密度前沿（guides/system 均在候选内，J=1.0），但 J 并列时按 dist 升序 tie-break → 永远选 dist-1；到达 docs 后 6 次文件读 + grep 耗尽预算，`cd guides`（priority 2）未轮到，docs 前沿（剩 2 个 cd 候选）未耗尽 → open-loop 不重选。**高层导航 + 低层候选排序双重把深目标排除在 10 步预算之外**——深度任务的 0/30 是结构性（候选生成不给路径式动作 + 预算 10 + dist-1 垄断），不是「差一点」。

## 6. 意外行为如实记录

1. **空目录陷阱零命中，R1 修复未受检验**：v5 陷阱设计在 dist-2（app/templates、data/archive），全臂 120 集无一次 `cd` 进入 → R1 的「cd 入空目录强制重选」在本实验触发 0 次。新树未能对 R1 形成压力测试（旧床该修复 +1，见 FF-SBH-3）。
2. **λ 在本树依旧零分叉（40/40 逐位一致）**：与 FF-SBH-2 结论延续，但原因升级——旧床是「frontier 全 dist-1 使 λ 无作用对象」，本树虽有 dist-2 满密度前沿进入候选，tie-break（dist 升序）与 J 惩罚（0.5·2=1.0 恰好压平）使两臂 argmax 恒同。**λ 迄今在任何实验中都未改变任何一次决策**。
3. **flat 臂 39→7 的断崖**：旧 9 任务床 flat 39/45 的绝大部分成功依赖深任务 start_cwd 下放（count_measurements/find_errors_v4/read_changelog_v4 等在 dist-1 起步）；本实验所有任务统一 /sandbox 起步后，flat 只剩「根文件读 + cd 轮转」能力，深度归零——**旧床 flat 高分有相当成分是 start_cwd 红利**（设计差异，如实记录，不构成对旧结论的否定）。
4. **SBH 目标导航的「早一集」优势跨集放大**：ep2 导航成功 → ep3/4 缓存重放 2 步完成 → 每任务 5 集内 SBH 把「发现期」压到 1 集（ep2），flat 压到 2 集（ep2-3）。在更长预算（如 max_steps 20）下该差距预计扩大为深度任务的实质差异——留作后续实验建议（非本实验结论）。

---

## 附：复现与对账

```bash
# 复现（需先 build 镜像；JSONL 不 commit）
docker build -f Dockerfile.busybox_v5 -t peda-sandbox:v5 .
PYTHONPATH=src venv/bin/python3 scripts/phase9_sandbox_hh.py --gen1
# 复算 + 门判定
PYTHONPATH=src venv/bin/python3 scripts/phase9_gen_analyze.py
```

- JSONL 不 commit（`results/` gitignore）；报告与代码 commit 于 `a217db8`（代码）后随本报告提交。运行确定性：三臂 meta commit 均 `a217db8`。
- 全部数字可复核：120 集 = 3 文件 ×（meta 头 + 40 集）；逐任务矩阵 = `scripts/phase9_gen_analyze.py` 从 JSONL 独立聚合；门判定按 §3 预注册阈值。
