# Phase 1.5 下一步任务

> **上游决策**: 修复 decompose_error（B）→ 增加训练数据（A）→ 快速验证 → 汇报结果
> **时间预算**: 1 小时（B=30min + A=20min + 验证=10min）
> **目标**: 确定核心假设（epistemic_error 驱动探索）是否可以验证

---

## 前置阅读

执行前必读：
- `PEDA_FINAL/PHASE1_5_COMPLETE_EVALUATION.md` — 上游对 Phase 1.5 报告的完整评估
- `PEDA_FINAL/phase1_5_complete_report.md` — Phase 1.5 实验报告（你写的）
- `src/phase1/world_model.py` — 需要修改的文件
- `scripts/phase1_5_synthetic_train.py` — 需要修改的文件

---

## 任务 1：修复 decompose_error（30 分钟）

### 问题

`decompose_error()` 的 TextState 分支只检查 `(room, exit_code)` 二元组的方差，完全忽略了 `inventory` / `has-key` 维度。

后果：
- 语义探针：has-key 分歧率 40%，完整元组分歧率 50%
- decompose_error：mean_epistemic_error = 0.0
- **测量失真**：用 decompose_error 无法判断 epistemic 信号是否存在

### 修复要求

文件：`src/phase1/world_model.py`
方法：`decompose_error()` 的 TextState 路径

**当前逻辑（推测）**：
```python
# 伪代码 — 你需要查看实际实现
def decompose_error(self, predicted_state, actual_state):
    if is_text_state(predicted_state):
        # 只检查 room 和 exit_code
        room_match = predicted_state.room == actual_state.room
        exit_match = predicted_state.exit_code == actual_state.exit_code
        return variance_over_checkpoints(room_match, exit_match)
```

**修复后逻辑**：
```python
def decompose_error(self, predicted_state, actual_state):
    if is_text_state(predicted_state):
        # 维度 1: room
        room_match = predicted_state.room == actual_state.room
        # 维度 2: exit_code
        exit_match = predicted_state.exit_code == actual_state.exit_code
        # 维度 3: has-key (新增)
        pred_has_key = "key" in (predicted_state.inventory or [])
        actual_has_key = "key" in (actual_state.inventory or [])
        key_match = pred_has_key == actual_has_key
        # 维度 4: room description 关键词（可选）
        # 计算分歧率（不是二元匹配，而是 checkpoints 之间的方差）
        # 返回所有维度的平均方差或最大方差
```

**关键决策点**：

1. **如何表示 has-key 的方差**？
   - 选项 A：每个 checkpoint 预测 `has_key: bool`，计算这些 bool 值的方差
   - 选项 B：每个 checkpoint 输出文本，从文本中提取 key 信息，再计算方差
   - 选 A 更简单，但需要确认 checkpoint 的输出格式是否支持

2. **方差聚合方式**：
   - 选项 A：平均方差 `(var_room + var_exit + var_key) / 3`
   - 选项 B：最大方差 `max(var_room, var_exit, var_key)`
   - 推荐 A（平均），除非语义探针显示某个维度特别重要

3. **与语义探针对齐**：
   - 修复后，运行语义探针（`scripts/phase15_semantic_probe.py`）
   - 对比：decompose_error 报告的 mean_epistemic_error 应该与语义探针的分歧率在同一数量级（20-50%）
   - 如果差距仍然 >10%，说明还有维度被遗漏

### 验证步骤

1. 修改 `decompose_error()`
2. 运行语义探针：`python scripts/phase15_semantic_probe.py`
3. 运行 decompose_error 测试（如果有单元测试）
4. 手动对比两个输出：
   - 语义探针 has-key 分歧率 = ?
   - decompose_error mean_epistemic_error = ?
5. 记录结果

---

## 任务 2：增加训练数据（20-30 分钟）

### 问题

当前：113 条唯一样本（50 walks × 20 步，穷举 + 随机游走）

需要：500-1000 条样本

### 修改要求

文件：`scripts/phase1_5_synthetic_train.py`

**修改内容**：

```python
# 1. 增加随机游走
- walks: 50 → 250
- steps_per_walk: 20 → 30
- 策略：不只是纯随机，加入多样性

# 2. 增加策略多样性（关键）
三种游走策略各 1/3：
a) 纯随机：完全随机选择合法动作
b) 目标导向：优先选择未在当前 walk 中执行过的动作
c) 重复探索：同一动作连续执行 2-3 次（测试模型对重复 action 的预测能力）

# 3. 增加确定性路径
- 最优路径：拿钥匙 → 向北走 → 开宝箱（及其变体）
- 死胡同路径：无钥匙时尝试开宝箱 → 失败 → 返回
- 循环路径：study ↔ hallway 来回

# 4. 去重逻辑保持不变
key = state_text + action_name
```

**为什么需要多样性**：

当前数据可能过度代表某些 (state, action) 组合，缺少关键的边缘 case：
- `take key` 只在特定 state 下有意义（在 study 且背包无钥匙时）
- `open chest` 只在有钥匙时成功，无钥匙时失败 — 两种 outcome 都需要样本
- 重复 `inventory` 后的状态变化（ boredom 相关的数据）

### 验证步骤

1. 修改数据生成脚本
2. 运行：`python scripts/phase1_5_synthetic_train.py`
3. 确认最终样本数：500-1000 条
4. 检查 `take key` 的样本覆盖：
   - 成功 case：`take key` 在 study 且无钥匙时执行 → 钥匙进入背包
   - 失败 case：`take key` 在 hallway 或已有钥匙时执行
5. 记录各 action 的样本分布（至少 10 条 per action）

---

## 任务 3：快速验证（10-15 分钟）

### 步骤

1. **LoRA 训练**（如果数据生成后没有自动训练）
   - 3 epochs, batch_size=4
   - 保存 3 个 checkpoints
   - 记录 loss 曲线

2. **语义探针**
   - `python scripts/phase15_semantic_probe.py`
   - 记录：has-key 分歧率、完整元组分歧率

3. **decompose_error 验证**
   - 运行 1 episode PEDA，记录 `mean_epistemic_error`
   - 对比修复前后的数值

4. **1 episode 行为检查**（PEDA + Pragmatic 各 1）
   - PEDA：是否尝试 `take key`？卡住位置？
   - Pragmatic：是否尝试 `take key`？
   - 记录行为轨迹

### 输出格式

将结果写入 `PEDA_FINAL/phase1_5_iteration2_report.md`：

```markdown
# Phase 1.5 Iteration 2 报告

## decompose_error 修复
- 修复内容：...
- 语义探针 has-key 分歧率：X%
- decompose_error mean_epistemic_error：Y（修复前：0.0）
- 判断：是否对齐？

## 数据增强
- 原始样本数：113
- 增强后样本数：XXX
- 各 action 分布：...
- take key 成功/失败样本数：...

## 训练
- Loss 曲线：...
- 训练时间：...

## 快速验证（1 episode each）
### PEDA
- Success：Y/N
- Steps：X/20
- 行为轨迹：...
- mean_epistemic_error：...

### Pragmatic
- Success：Y/N
- Steps：X/20
- 行为轨迹：...

## 关键发现
1. ...
2. ...

## 上游决策请求
基于以上结果，推荐下一步：...
```

---

## 约束与提醒

### 硬性约束

1. **总时间 ≤ 1 小时**。如果某个任务超时，跳过它并汇报原因。
2. **不要创建新文件**（除报告外）。修改现有文件。
3. **不要修改 Grid World 路径**。所有修改必须通过 `hasattr` 分派，保持向后兼容。
4. **不要创建 PLAN/ARCH 文档**。只在报告中记录。

### 心态提醒

- 这是**探索**，不是交付。负结果也是结果。
- 如果 decompose_error 修复后 epistemic 仍然 ≈ 0，记录这个事实——这是有价值的信息。
- 如果增加数据后模型仍然学不会 `take key`，记录——说明 0.5B 模型可能需要更多数据或更复杂架构。
- **不要因为想"让结果好看"而调整参数**。保持 pragmatic_weight=3.0, drive weights 不变。

### 统计显著性提醒

- 1 episode 的验证是**行为检查**，不是**假设验证**。
- 不要从 1 episode 得出"PEDA 成功/失败"的结论。
- 行为差异（PEDA 尝试 take key vs Pragmatic 不尝试）如果与 Iteration 1 一致，可以增加可信度。

---

## 上游评估标准

我会根据你的报告评估：

| 标准 | 权重 |
|------|------|
| decompose_error 是否与语义探针对齐 | 40% |
| 数据增强后样本质量 | 25% |
| 行为模式是否可复现 | 25% |
| 报告诚实性 | 10% |

如果 epistemic 仍然 ≈ 0 即使修复了 bug，这不是你的失败——这是有价值的数据。
