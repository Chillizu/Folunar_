# FF-PEC-1：PE 罗盘（预测误差作未知目录方向信号）实验报告

- **日期**：2026-08-03
- **分支/基线**：dev @ `effbb90`（本实验 commit：`7ad577b`，已 push origin dev）
- **实验**：`sandbox-hh-pec1` — 把 FF-CI-6 的预测误差信号作为方向项接入 MLP λ0 路径规划器
- **主判定**：s10（40 集）deep 分数；**次级**：s20（40 集）
- **归因**：PEC 与 MLP λ0 的唯一差别是 `γ·s(dir)`（λ=0、γ=1 固定单臂），任何 deep 增益归因于 PE 信号本身

---

## 1. 背景与目标

FF-MLP-1 证明框架能机械到达 dist≥2 的目录（38 次），但未知目录的选择是字典序赌博
（deep 0/30）——路径规划器面对多个 equally-unknown 的候选目录时，`prior=0.5` 全部打平，
决胜规则退化为 lexicographic path 字符串。本实验正面回答「count 无罗盘」缺口：
用 WM 对未知目录 `ls <dir>` 结果的**预测误差（epistemic 不确定性）**作为方向信号
`s(dir)`，让规划器朝 WM 最不确定的目录走。这是 HH+PE 组合线的落点，也是本研究线
最后一个有阳性先验（FF-CI-6：PE 非劣 +3.3pp、学习斜率 3×）支撑的实验。

**Pre-registered Gates（FF-PEC-1，禁止事后调整）**：
- 主判定 s10：deep = dist≥2 六任务子集（满分 30），基线 MLP λ0 = 0/30。
  - **PASS**：PEC deep ≥ 4/30 且 pooled ≥ 7/40
  - **KILL**：PEC deep ≤ 1/30
  - **NULL**：2-3/30，如实记录
- 次级注册（无门）：s20 PEC vs SBH s20（3/30）；fallback 率；到达/选对分解。

## 2. PE 栈复用考古结论（FF-CI-6）

考古对象：`PEDA_FINAL/phase9/CI_M3M4_REPORT.md` §7（FF-CI-6 adjudication）+
`scripts/phase9_ci_m3.py`（M3 PEDA runner）+ 其依赖的 phase1/phase2 机制。

**栈构成（逐行复用）**：

| 组件 | 位置 | 复用方式 |
|---|---|---|
| WorldModel（L1 exit code / L2 cwd+files / L3 summary） | `phase1/world_model.py` | `wm.predict(state, action)` sandbox 路径（`_sandbox_system_message` + `_build_text_prompt`），L2 = cwd/files JSON |
| EnsembleErrorComputer（charter epistemic 定义） | `phase1/world_model.py` | sandbox 分支 ev 公式：成对 `(exit 分歧 + files-set 分歧)/2`；成员选择 `_predictions_for` 逐行拷贝（无 checkpoint → 单模型；有 → checkpoint 集） |
| SandboxLearningModule（LoRA 学习） | `phase2/run.py` | `buffer_size=100, update_interval=10`（CI 的 20 对应 20 步 episode，按 10 步 episode 等比折半）；`lora_finetune(batch_size=1, epochs=1, lr=2e-4)`；每次 update 后 `save_checkpoint` 建 ensemble |
| 模型 | `/home/data/models/Qwen2.5-0.5B-Instruct` | **同一模型**（fp32，无 GPU 依赖：本地 CPU 预测 ~6.7s/次、LoRA 更新 ~26s/次） |

**epistemic 定义（charter）**：ensemble 成对方差。CI M3 实测中前 2 个 episode（0-1 个
checkpoint）`mean_epistemic_error=0.0`，ep2 起（≥2 members）非零（read_data 成功集
epi=1.0）——本实验的 `n<2 → s=0`（罗盘惰性）与该行为同构。

**本地可行性确认**：GPU 实例已回收；FF-CI-6 的 M3 GPU 主跑之外存在 CPU partial run
（read_secret 3/10）证明 CPU 可跑。本实验全 CPU 复测通过（见 §8c 成本）。

**与本实验的边界**：
- 零任务知识：WM 只从环境动态（observed transitions）学习，任务描述从不进入 WM
  prompt（任务条件化语义先验是另一条研究线，本实验不测 —— out-of-scope）。
- CI 的 `ActionGenerator`/`HomeostaticDriveSystem`/EFE 不参与：本实验的方向项直接
  用 charter ensemble 方差（CI 中该值才是上报的 `mean_epistemic_error`；EFE 选择
  用的 `1-conf` 代理在本实验不采信为方向项，避免把「自信度」混进信号）。

## 3. 方法（实现读法，executor 定稿）

新模块 `src/phase9/sandbox_hh/pe_compass.py`（旧文件零 diff）：

- **PECPlanner**（PathPlanner 子类）：`J(path) = prior(end) − λ·depth + γ·s(dir)`，
  λ=0、γ=1 固定。s 附着语义：取路径链上**第一个未访问（unknown）目录** u —— 即
  下一跳将首次进入未知内容的位置；WM 预测从 u 的父目录 `cd <basename(u)>` 的 L2
  结果（= `ls u` 的 files），ensemble 方差即 s(u)。最短路径的中间跳都是 known 目录，
  故首个 unknown 目录通常就是路径目标；同一次 select 内多个路径共享同一
  (parent, child) 只查询一次（复用）。unvisited（never-visited）目录 prior=0.5
  （与 MLP λ0 相同），visited 目录用实际 unvisited density（此时 s=0，density 已
  携带信号）。
- **PECAgent**（SandboxPathAgent 子类）：仅覆盖 `__init__`/`_select_and_log`/
  `_choose_action`/`run_episode`（run_episode 为基类循环逐行拷贝 + WM 学习钩子：
  每步 `store_experience`，episode 末 `lm.update()` + checkpoint）。低层 Phase 8
  逐字节不变。
- **成本闸**：WM 查询只在重选点（select 事件）发起，每 episode ≤3 次（预注册）；
  `n<2` 成员时 s≡0（charter），不计费不算 fallback；预算耗尽 / WM 不可用 / 异常 →
  fallback 中性 0.5 并计数（该次选择退化为 MLP λ0）。预算语义：命中同 select 内
  缓存的重用不计费。
- **PECEnsemble**（EnsembleErrorComputer 子类）：机制与基类相同（save_pretrained
  快照当前 LoRA 状态 + 滑动窗口 ≤5），仅 checkpoint 目录按任务隔离
  （`checkpoints/phase9_pec/{task}/`），避免多任务共享 WM 时 step 号跨任务碰撞。

**归因链**：PEC = MLP λ0 + `γ·s(dir)`，其中 s 只来自 WM 对未知目录 L2 预测的
epistemic 不确定性（由环境动态经 LoRA 学习产生）。γ=0 消融 = 盘上已有 MLP λ0
（s10 deep 0/30）。

## 4. 运行配置

- 镜像 `peda-sandbox:v5`，8 gen 任务（T1-T8），seeds 0-4（5 集/任务）。
- PEC 单臂：λ=0、γ=1；**s10 40 集（主）+ s20 40 集（次级）= 80 集**。
- WM 共享：单进程一个 WorldModel，LoRA 状态跨任务累积（CI M3 同构）；ec/lm/
  checkpoint 按任务隔离。
- 输出：`results/phase9_pec_s10.jsonl`、`results/phase9_pec_s20.jsonl`
  （WATCHDOG D4：meta 含 git commit + WM 模型/adapter 标识；JSONL 不入库）。

<!-- RESULTS-BEGIN -->

## 5. 结果：s10 主表（四臂，各 40 集）

| task | 深度 | flat | SBH | MLP-λ0 | **PEC** |
|---|---|---|---|---|---|
| gen_read_notes | 0 | 5 | 5 | 3 | **0** |
| gen_read_setup | 1 | 2 | 3 | 1 | **3** |
| gen_read_sensor | 2 | 0 | 0 | 0 | **0** |
| gen_read_usage | 2 | 0 | 0 | 0 | **2** |
| gen_find_api_ref | 2 | 0 | 0 | 0 | **0** |
| gen_read_audit | 2 | 0 | 0 | 0 | **0** |
| gen_count_readings | 2 | 0 | 0 | 0 | **1** |
| gen_find_error_deep | 2 | 0 | 0 | 0 | **0** |
| **POOLED** | | **7/40** | **8/40** | **4/40** | **6/40** |
| **CONTROL** (T1+T2) | | 7/10 | 8/10 | 4/10 | **3/10** |
| **DEEP** (T3-T8) | | 0/30 | 0/30 | 0/30 | **3/30** |

- deep 0/30 → **3/30**：PE 罗盘把 MLP λ0 的字典序赌博打出了非零（usage 2 + count 1）。
- pooled 6/40：≥ MLP λ0（4/40），< flat（7/40）/ SBH（8/40）——代价在 CONTROL：T1 从
  flat/SBH 的 5/5、MLP 的 3/5 掉到 **0/5**（见 §8b 意外行为）。
- 逐任务与 MLP λ0 的差（ΔPEC−MLP0）：T2 +2、T4 +2、T7 +1、T1 −3、其余 0。

## 6. 结果：s20 次表（vs SBH s20）

| task | 深度 | flat-s20 | SBH-s20 | **PEC-s20** |
|---|---|---|---|---|
| gen_read_notes | 0 | 5 | 5 | **5** |
| gen_read_setup | 1 | 4 | 4 | **1** |
| gen_read_sensor | 2 | 0 | 1 | **0** |
| gen_read_usage | 2 | 0 | 0 | **0** |
| gen_find_api_ref | 2 | 0 | 0 | **0** |
| gen_read_audit | 2 | 1 | 1 | **1** |
| gen_count_readings | 2 | 0 | 1 | **3** |
| gen_find_error_deep | 2 | 0 | 0 | **0** |
| **POOLED** | | 10/40 | 12/40 | **10/40** |
| **DEEP** | | 1/30 | 3/30 | **4/30** |

- **PEC-s20 deep 4/30 > SBH-s20 3/30**（audit 1 + count 3，count 是 SBH 的 3 倍）；
  pooled 10/40 < SBH 12/40（T2 掉 3）。
- 预算翻倍（20 步）把 count（data/raw，dist-2 深链）从 s10 的 1 推到 3；T8
  （grep 深任务）两臂均为 0。

## 7. 门判定（s10 主判定）

```json
{
  "pec_deep": 3, "pec_pooled": 6,
  "verdict": "NULL",
  "reason": "PEC deep=3/30 在 2-3/30（弱信号，如实记录）",
  "baselines": {"flat_deep": [0,30], "sbh_deep": [0,30], "mlp_deep": [0,30]}
}
```

**NULL（2-3/30）**：deep 3/30 ≥ 1（KILL 未触发）但 < 4（PASS 未达标）。0/30 → 3/30
是本研究线 deep 首次非零，但按预注册阈值如实记录为弱信号，不宣称有效果。

## 8. WM 查询统计与到达/选对分解

**查询成本（80 集合计）**：

| | s10 | s20 | 合计 |
|---|---|---|---|
| WM 查询 | 66（1.65/集） | 58（1.45/集） | **124（1.55/集，上限 3）** |
| fallback | 60（1.50/集） | 99（2.48/集） | **159（1.99/集）** |
| fallback 构成 | budget 60 / unavailable 0 / error 0 | budget 99 / 0 / 0 | budget 159 / 0 / 0 |
| 查询 s 均值/最大 | 0.364 / 1.0 | 0.239 / 0.667 | — |
| select 事件 | 46 | 67 | 113 |

- 预算用满率：124/（80×3）= 51.7%；fallback 全部为预算耗尽类（WM 本地可用、零
  超时零异常）—— 根目录一次 select 有 ≥4 个 unknown 候选而预算只有 3，第 4 个
  必吃 0.5。**设计观察**：fallback=0.5 高于多数真实 s（0.24-0.36 均值），被 0.5
  命中的候选反而被抬高——预算耗尽方向上存在系统性偏向，报告如实记录。
- s 来源分布（113 个 select 事件的胜者项）：no_ensemble 38、query 26、cached 21、
  fallback_budget 25、none 3。

**到达/选对分解（s10）**：

| | select 事件 | depth≥2 目标被选中 | 实际到达 | 到达率 |
|---|---|---|---|---|
| PEC s10 | 46 | 30 | 29 | 96.7% |
| MLP λ0 | 53 | 38 | 38 | 100% |

- 到达率两臂都接近 100%：deep 失败**不是**可达性/预算问题（与 FF-MLP-1 诊断一致），
  而是「朝哪个未知目录走」的选择问题 + 步数耗尽。PE 罗盘改变的是选择分布，
  不是到达能力。

## 8a. 轨迹对照（PE 选对 vs 字典序选错）

**例 1 — gen_read_setup ep2（PE 罗盘直接选对）**：

```
PEC: cd docs → cat api.txt → head -n 5 api.txt → wc -l api.txt → cat setup.md   ✅ SUCC（5 步）
MLP: cd app → cd templates → cd .. → cd .. → cat README.md → …（10 步耗在 app 陷阱）❌
```
PEC ep2 的 select：`/sandbox/docs s=1.0(query)` —— ensemble 对 `ls docs` 全分歧，
J=0.5+1.0=1.5 压倒一切；docs 正是目标父目录（dist-1）。MLP 同一时刻选
`/sandbox/app/templates`（0.5 平局 → 深度优先 → 字典序）。

**例 2 — gen_read_usage ep3（PE 罗盘深链选对）**：

```
PEC: cd docs → cd guides → cat changelog.md → … → cat usage.md        ✅ SUCC（9 步）
MLP: cd data → cat readings.csv → …（10 步，data 是错方向）            ❌
```

**例 3 — gen_read_usage ep4（PE 罗盘命中即赢）**：

```
PEC: cd docs → cd guides → cat usage.md                                ✅ SUCC（3 步）
MLP: cd data → cd archive → cd .. → cd .. → cd docs → cat api.txt → …（10 步）❌
```

**反例 — gen_read_notes ep3（PE 罗盘把 agent 引离根目录）**：

```
PEC: cd data → cd raw → cat counts.txt → …（10 步，目标 notes.txt 在根目录）❌
MLP: cat notes.txt（replay 命中，1 步）                                  ✅
```
PEC ep3 select `/sandbox/data/raw s=0.5(fallback_budget)` —— 预算耗尽 fallback
把 data/raw 抬到 J=1.0 平局，深度优先选中 → 深钻 data，根目录的 notes.txt 被
跳过。T1 的 5 次成功被 PE 偏好（docs/data 高不确定性）+ fallback 抬高系统性
牺牲：这是 NULL 而非 PASS 的直接代价项。

## 8b. 意外行为与数据质量

1. **NaN-loss 更新（27/80 次）不污染权重**：lora_finetune 的 384-token 上限在个别
   长 state（文件多/输出长）下把 prompt 截断、target 全被 mask → 该 batch loss
   =NaN。实测 NaN loss 产生**零梯度 no-op**（非 NaN 权重）：40 个落盘 adapter
   文件全部有限值，JSONL 内全部 s 值有限，查询/学习未受污染。数据侧裁剪
   （file_cache 最近 1 条 ×40 字符、last_output 截 80）把这类样本压到少数；
   NaN 权重回滚守卫（_wm_episode_update）保留作兜底（本次未触发）。
2. **s 值分化真实存在**：docs s=1.0（多任务 ep2）、logs 0.83、data 1.0、archive
   0.58、已学目录（logs/app）0.0 —— ensemble 对**从未见过的目录**分歧大、对跨
   任务反复见过的目录趋于一致，方向项确实携带了「未知程度」信息。
3. **预算-3 的耦合效应**：s20 的 fallback 率（2.48/集）高于 s10（1.50/集），因为
   20 步 episode 的重选事件更多（67 vs 46），每次重选都撞预算上限。

## 8c. 成本统计

- 80 集全程本地 CPU（Intel Arc 无参与）：WM 预测 ~6.7s/次、LoRA 更新
  s10 ~27s / s20 ~57s（20 样本）、Docker sandbox ~1-2s/集。总墙钟 ~95 min
  （s10 ~40 + s20 ~55），含 WM 加载（0.5s）与 16 任务 × ≤5 checkpoint 落盘
  （checkpoints/phase9_pec/ 共 1.4G，每 adapter ~35M）。
- WM 查询成本：124 次预测 × ~6.7s × 平均 2-3 成员 ≈ 30-40 min（含在总时长内）。

## 8d. Out-of-scope 说明

任务条件化语义先验（把任务描述喂给 WM 作预测上下文）是另一条研究线，本实验
不测：WM 的训练与查询输入只含环境状态 + 动作，任务描述从未进入任何 prompt
（meta `zero_task_knowledge: true`）。若允许语义先验，docs/data/logs 的方向
可被任务文本直接标注 —— 那将混入「知道任务」的增益，破坏归因。

<!-- RESULTS-END -->

## 9. 交付与 git 状态

- 新代码：`src/phase9/sandbox_hh/pe_compass.py`；增量 runner：`scripts/phase9_sandbox_hh.py`
  （`--compass`/`--resume`）；新分析：`scripts/phase9_pec_analyze.py`；
  报告：`results/phase9_pec_report.md`。
- `git diff src/phase8 src/phase5 src/phase9/sandbox_hh/{planner,agent,path_planner}.py` = 空。
- JSONL 不 commit；commit 具体文件名并 push origin dev。
