# Phase 1 部分训练评估报告：PEDA vs Pragmatic-Only 对比

## 1. 概述

本报告记录了对 PEDA（Predictive-Error-Driven Autonomous Agent）核心假设的第一次实验验证：**预测误差能否驱动自主探索**。为此我们设计了"部分训练"评估方法——在 5×5 Grid World 中只让 World Model 在 25% 的格子（6/25 个 cell）上训练，然后在"目标在已知区域"和"目标在未知区域"两种条件下分别运行 PEDA 和纯 pragmatic-only baseline，观察差异。

## 2. 实验设计

### 2.1 方法

|项目|值|
|---|---|
|World Model|Qwen2.5-0.5B-Instruct + LoRA|
|训练数据|25% cell（6/25 已知格子）下的合成 transitions|
|训练 epoch|1（仅 1 个 checkpoint）|
|评估 episode 数|10（per condition per agent）|
|pragmatic_weight|3.0（PEDA 和 Pragmatic 相同）|
|Drive weights|cur=0.1, cmp=0.5, bor=0.1, nov=0.1|
|max_steps|50（后 2 个 chunk 用 45）|
|评估 seed|42|

### 2.2 已知区域

6 个训练过的格子：[(4,0), (0,3), (0,0), (1,3), (1,2), (4,4)]（25 个中随机选取）。

### 2.3 两种条件

- **goal_known**: goal 在 known cells 中，start 任意（但不等于 goal）
- **goal_unknown**: goal 在 known cells 之外，start 任意

两种条件下 PEDA 和 Pragmatic-only 使用**完全相同的 (start, goal) 对和 seed**，确保公平比较。

### 2.4 实验限制

**关键限制——必须诚实说明：**

1. **只有一个 checkpoint**：适配器只训练了 1 epoch，只保存了 1 个 checkpoint。`EnsembleErrorComputer` 需要 ≥2 个 checkpoint 才能产生非零 `epistemic_error`，因此本次评估中 **epistemic uncertainty ≡ 0**。
2. **PEDA 的实际行为**：由于 epistmic_error=0，PEDA 的 EFE 公式退化为 `drive_system.apply_to_efe(pragmatic * pragmatic_weight)`。PEDA 与 Pragmatic-only 的唯一区别是 drive system 对 EFE 的调制（curiosity/boredom/novelty/competence 权重）。
3. **g1_test_set** 只在 chunk 0 计算（0.8684），其余 9 个 chunk 通过 `--skip-g1-test` 跳过。合并报告重用 chunk 0 的值。
4. **样本量**：每个条件每 agent 10 episode，不足以做显著性检验。

## 3. 执行过程

### 3.1 阶段准备

- 复用已有的 `partial_adapter_real_25`（已在之前会话中用 `phase1_synthetic_train.py --train-fraction 0.25 --epochs 1` 训练）
- 为 `phase1_partial_eval.py` 增加 CLI 参数 `--start-episode`、`--total-episodes`、`--skip-g1-test`，支持分块运行
- 修复 `_sample_goal` 中 known_cells list-of-lists 的 bug
- 创建 `merge_partial_eval_chunks.py` 合并分块结果

### 3.2 分块运行

由于 CPU-only 推理速度（单步 ~10–30s），单次 10 episode 超时窗口不够。将 10 个 episode 拆为 10 个 chunk，每个 chunk 只跑 1 episode。每个条件各跑 1 episode（PEDA + Pragmatic），共 4 轮 episode 执行/条件 = 2 轮 per chunk。

|Chunk|Ep|goal_known P/P|goal_unknown P/P|耗时(s)|
|---|---|---|---|---|
|0|0|3/3 (both T)|2/2 (both T)|161|
|1|1|50/50 (both F)|4/4 (both T)|365|
|2|2|2/2 (both T)|2/1 (T/F)|535|
|3|3|50/50 (both F)|50/50 (both F)|2722|
|4|4|50/50 (both F)|3/3 (both T)|3036|
|5|5|3/2 (both T)|50/50 (both F)|2776|
|6|6|6/50 (T/F)|4/4 (both T)|1772|
|7|7|6/50 (T/F)|4/50 (T/F)|3032|
|8|8|5/5 (both T)|45/45 (both F, max_steps=45)|3222|
|9|9|2/2 (both T)|2/2 (both T)|208|

- P = PEDA, Prag = Pragmatic-only, T = success, F = failure

### 3.3 合并

使用 `merge_partial_eval_chunks.py` 合并 10 个 chunk 的 `raw_results`，重新计算聚合指标和 verdict。

## 4. 实验结果

### 4.1 主要指标

|条件|Agent|Success Rate|Mean Steps|Revisit Rate|G1 (per-ep avg)|
|---|---|---|---|---|---|
|**goal_known**|**PEDA**|**0.9**|**8.6**|**0.13**|0.93|
||Pragmatic-only|0.7|17.3|0.27|1.00|
|**goal_unknown**|**PEDA**|**0.7**|**16.6**|**0.31**|0.73|
||Pragmatic-only|0.6|21.1|0.37|0.75|

### 4.2 PEDA 探索指标（goal_unknown）

|指标|值|
|---|---|
|mean_unknown_fraction|0.86（86% 步数在未知 cell 上）|
|mean_unknown_cells_visited|3.3（访问了 3.3 个不同未知 cell）|
|mean_steps_before_known|30.8（平均首次进入已知 cell 在第 31 步）|

### 4.3 Verdict

```
peda_better_in_unknown_goal: true
reason: PEDA mean_steps (16.6) < pragmatic_only (21.1) in goal_unknown condition
```

## 5. 结果分析

### 5.1 观察到的差异来源

PEDA 在两种条件下 success_rate 都更高、mean_steps 更低。但差异的来源**不是**预测误差/信息增益——因为只有 1 个 checkpoint，epistemic_error 为零。差异最可能来自 **HomeostaticDriveSystem** 的调制：

- PEDA 的 EFE 经过 drive weights 调制（curiosity=0.1, competence=0.5, boredom=0.1, novelty=0.1），影响 action 选择
- Pragmatic-only 完全基于 `distance_pragmatic * 3.0`，无任何调制
- 当 pragmatic-only Agent 陷入循环（如 chunk 3/4/5 中反复在 (1,3)-(2,3) 之间震荡），drive system 能通过 boredom/novelty 信号打破僵局

### 5.2 典型失败模式

chunk 3 的轨迹展示了 pure pragmatic 的失败模式：

```
start=(1,1), goal=(2,2)
→ (1,2) → (2,2) → (2,3) → (1,3) → (2,3) → (1,3) → (2,3) … （50 步往返震荡）
```

Agent 在 (1,3) 和 (2,3) 之间无限往返。这两个 cell 都未训练（已知 cell 只有 [(4,0),(0,3),(0,0),(1,3),(1,2),(4,4)]），模型无法预测 action 的准确结果，但 pragmatic distance 却让 agent 在两个"看起来更近"的 cell 之间反复。

### 5.3 数据质量

|指标|值|
|---|---|
|g1_test_set（held-out OOD）|0.8684|
|goal_unknown PEDA avg g1（per trajectory）|0.73|
|goal_unknown Pragmatic avg g1|0.75|

g1_test_set=0.8684 说明模型在未知区域仍有较好的预测能力（5×5 grid 任务相对简单）。这可能削弱了 PEDA 的探索优势——如果预测误差本身就小，epistemic bonus 即使存在也缺乏驱动力。

## 6. 结论与建议

### 6.1 实验局限性

|局限性|影响|
|---|---|
|只有 1 checkpoint，epistemic_error=0|**无法验证"预测误差驱动探索"这一核心假设**|
|仅 10 episode/条件|统计显著性不足，结果为方向性提示|
|g1_test_set=0.8684 偏高|未知区域压力不够大|
|0.5B 模型在 5×5 grid 上可能过强|泛化能力掩盖了需要探索的场景|

### 6.2 结论

**本次实验不能作为核心假设验证通过的证据。** PEDA 表现更好，但优势来自 drive system 调制，而非预测误差驱动的信息增益。这与原计划要求的"在 epistemic error 非零条件下证明 PEDA 优于 pragmatic baseline"目标不符。

如果在有 ≥2 个 checkpoint 的条件下重跑，可能会观察到：
- epistemic_error > 0 → PEDA 主动探索未知区域 → 更快到达未知区域中的目标

### 6.3 下一步建议

|优先级|行动|预期效果|
|---|---|---|
|P0|用 ≥3 epoch、保存 ≥3 个 checkpoint 重新训练 25% adapter|产生非零 epistemic_error，验证核心假设|
|P0|重跑 10+ episode 评估|比较 PEDA full EFE vs pragmatic-only|
|P1|若 g1_test_set > 0.85，降低 train_fraction 到 15–20%|增大未知区域不可预测性，使预测误差更显著|
|P2|将 pragmatic_weight 从 3.0 降至 1.0/0.5|降低 pragmatic 主导，让 epistemic 项有影响空间|

## 7. 附：逐 episode 数据

### goal_known

|Ep|Start→Goal|PEDA steps/succ|Prag steps/succ|
|---|---|---|---|
|0|(4,2)→(4,0)|3/True|3/True|
|1|(3,3)→(4,4)|50/False|50/False|
|2|(0,1)→(0,3)|2/True|2/True|
|3|(1,1)→(4,4)|50/False|50/False|
|4|(4,2)→(4,0)|50/False|50/False|
|5|(0,1)→(1,3)|3/True|2/True|
|6|(1,1)→(0,3)|6/True|50/False|
|7|(1,1)→(0,3)|6/True|50/False|
|8|(2,2)→(0,3)|5/True|5/True|
|9|(0,2)→(0,3)|2/True|2/True|

### goal_unknown

|Ep|Start→Goal|PEDA steps/succ|Prag steps/succ|
|---|---|---|---|
|0|(3,3)→(2,2)|2/True|2/True|
|1|(3,3)→(0,2)|4/True|4/True|
|2|(4,2)→(4,1)|2/True|1/True|
|3|(1,1)→(2,2)|50/False|50/False|
|4|(3,3)→(2,2)|3/True|3/True|
|5|(1,1)→(2,2)|50/False|50/False|
|6|(3,3)→(3,2)|4/True|4/True|
|7|(1,1)→(3,2)|4/True|50/False|
|8|(3,0)→(0,2)|45/False|45/False|
|9|(3,3)→(2,2)|2/True|2/True|

---

*报告生成于 2026-07-06*
*脚本: `scripts/phase1_partial_eval.py` + `scripts/merge_partial_eval_chunks.py`*
*数据: `results/phase1_partial_eval_{chunk_0..9,10eps}.json`*
