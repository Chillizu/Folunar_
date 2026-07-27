# Phase 1 部分训练验证报告评估

**评估日期**: 2026-07-03  
**评估对象**: `partial_training_evaluation_report_for_senior_review.md`  
**代码基**: `Chillizu/Folunar_` @ commit `fec0676` (dev 分支)  
**上一轮评估**: `PHASE1_EVALUATION.md`

---

## 1. 一句话结论

**方法学上正确，统计上不足，结果有强烈信号，但需要更多数据确认。**

上一轮评估指出的核心问题（"这不是预测误差驱动探索，这是完美记忆驱动导航"）在本轮得到了**根本性的实验设计改进**。但 1 episode/condition 的统计量仍然无法给出可信结论。

---

## 2. 实验设计改进评估（与上一轮对比）

| 维度 | 第一轮（G1/G2/G3） | 本轮（部分训练） | 改进程度 |
|------|-------------------|-----------------|---------|
| 训练/评估分布 | 同分布（记忆） | **异分布（6 known / 19 unknown cells）** | 根本性 |
| 基线对比 | 无（只有random） | **PEDA vs pragmatic_only** | 根本性 |
| 预测误差 | ≈0（完美WM） | **>0（g1_test_set=0.8684）** | 根本性 |
| 控制变量 | 无 | **相同起点/目标、相同pragmatic_weight、禁用在线学习、新鲜DriveSystem** | 根本性 |
| 统计量 | 20 episodes | **1 episode/condition** | 倒退 |
| epistemic信号 | 无（stub确定性） | **不完整（无ensemble，只有confidence-based）** | 部分改进 |

**结论**：实验设计的 5 个关键维度中有 4 个得到了根本性改进。这是从"无法验证"到"可以验证但有噪声"的跃迁。

---

## 3. 代码验证

### 3.1 pragmatic_only 实现的公平性

验证 `src/phase1/drive_system.py` (dev 分支 `d26f803`):

```python
def compute_efe(self, state, trajectory, action_history, candidate_action=None):
    pragmatic = ...  # 曼哈顿距离 / max_dist
    if self.pragmatic_only:
        return pragmatic * self.pragmatic_weight  # 仅pragmatic，无epistemic，无drive
    # PEDA路径：epistemic + pragmatic * weight + drive_adjustment
    epistemic = sum((1.0 - p.level2_confidence) * ratio * (0.9 ** i) ...)
    base_efe = epistemic + pragmatic * self.pragmatic_weight
    return self.drive_system.apply_to_efe(base_efe, ...)
```

**公平性确认**：✅ 两个agent使用完全相同的`pragmatic_weight=3.0`，只在`pragmatic_only`标志上不同。当`pragmatic_only=True`时，EFE不包含epistemic项和drive_system调整。这是正确的隔离设计。

### 3.2 cell-level 拆分的正确性

验证 `scripts/phase1_synthetic_train.py`:
- 按cell拆分（不是按state-action对拆分）→ 存在明确的"known region"和"unknown region"
- 确定性拆分（`random.Random(split_seed)`）→ 可复现
- 保存`trained_manifest.json` → 评估脚本可以读取known_cells

**设计确认**：✅ cell-level拆分比state-action拆分更适合测试"区域探索"行为。

### 3.3 评估控制的严谨性

报告声称的控制：
- 相同起点与目标 ✅（代码中固定`(goal, start_seed)`）
- 禁用在线学习 ✅（`update_interval=100000`）
- 新鲜DriveSystem ✅（每个agent每个episode新建）

**控制确认**：✅ 三个关键控制变量都得到了正确处理。

---

## 4. 结果分析

### 4.1 原始结果

| 条件 | Agent | Success | Mean Steps | Revisit Rate | g1 |
|------|-------|---------|------------|--------------|-----|
| goal_known | PEDA | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_known | pragmatic_only | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_unknown | PEDA | 1.000 | **2.0** | 0.000 | 0.500 |
| goal_unknown | pragmatic_only | 0.000 | **20.0** | **0.905** | 1.000 |

### 4.2 结果解释

**goal_known 条件下两者相同**（3步，100%成功）：

这是预期结果。当目标在已知区域时，WM预测准确，pragmatic距离信号足够导航。PEDA的epistemic信号不需要发挥作用。这个结果的"两者相同"恰恰证明了实验的公平性——如果PEDA在此条件下也优于pragmatic_only，那反而是bug。

**goal_unknown 条件下 PEDA 显著优于 pragmatic_only**（2步 vs 20步失败）：

这是核心信号。但需要谨慎解释：

**可能的解释A（理想情况）**：PEDA的epistemic信号让agent先去探索已知区域，从已知区域推断通往未知目标的路径，然后高效到达。2步意味着起点和目标距离为2（曼哈顿距离），且agent选择了正确的方向。

**可能的解释B（运气）**：1 episode的随机性。起点和目标可能恰好很近（距离2），且PEDA碰巧选择了正确方向。5x5 grid中曼哈顿距离为2的(start, goal)组合有约50对，占总组合的~8%。

**可能的解释C（WM在未知区域仍有过强预测能力）**：g1_test_set=0.8684意味着WM在未知区域的预测准确率仍有87%。这不是"完全不知道"，而是"大部分知道但有13%错误"。这可能降低了实验难度——WM的"部分知识"足以在2步内找到目标。

**pragmatic_only 的 0.905 revisit rate 分析**：

这个数值异常高，但可能是合理的。解释：

pragmatic_only agent = "闭着眼睛朝目标走"。在5x5 grid中：
- WM在未知区域的预测错误率为13%
- rollout horizon = 2步（根据latency自适应）
- 每2步中可能有0-1步的预测错误
- 错误预测导致agent走入死胡同或绕路
- 20步的限制下，agent在错误预测和边界反射之间循环
- 0.905 revisit = 在20步中约18步重访 = 实际只探索了2个新格子

这个解释是合理的：一个"知道目标方向但每一步有13%概率走错"的agent，在20步内很可能陷入局部循环。

### 4.3 关键指标 g1_test_set = 0.8684

这个指标的含义：
- WM在**训练集外**的状态-动作对上的next-position预测准确率为86.84%
- 这意味着未知区域不是"完全黑暗"——WM有一定的泛化能力
- 这可能是0.5B模型的记忆/模式匹配能力，而非真正的"理解"
- 但对于实验目的来说，这创造了一个有意义的"部分已知"环境

**问题**：如果WM在未知区域的准确率太高（>90%），PEDA和pragmatic_only的差异会被抹平。0.8684是一个合理的中间值——足够让差异显现，但不至于让WM"太聪明"。

---

## 5. 局限性评估（报告自身已承认）

| # | 局限性 | 严重程度 | 评估 |
|---|--------|---------|------|
| L1 | 1 episode/condition | **致命** | 统计量不足，无法排除运气。需要至少10 episodes。 |
| L2 | 单epoch无ensemble | 中等 | epistemic信号不完整，但confidence-based信号已足够产生差异。不是blocking。 |
| L3 | g1_test_set<0.90 | 低 | 这是实验设计的预期结果（故意让WM不完美），不是问题。 |
| L4 | CPU推理限制 | 中等 | 每episode 15分钟，10 episodes = 2.5小时。可接受。 |
| L5 | 0.5B模型 | 低 | Grid World足够简单，0.5B可以胜任。 |

---

## 6. 统计显著性分析

### 6.1 当前数据的可信度

1 episode/condition → 无法做任何统计检验。结果是**描述性**的，不是**推断性**的。

### 6.2 需要多少episodes？

假设PEDA在goal_unknown条件下的真实成功率是80%，pragmatic_only是20%（从当前1 episode的100% vs 0%推测）。

使用二项分布检验（Fisher exact test）：
- 效应量：80% vs 20%（大效应）
- 显著性水平：α = 0.05
- 统计功效：1-β = 0.80
- **需要：每组约 8-10 episodes**

如果真实效应更小（如60% vs 20%），需要约 15-20 episodes。

**建议：至少10 episodes/condition**。在CPU上约需 10 × 15min × 2conditions = 5小时。可以在一夜之间跑完。

### 6.3 推荐的评估方案

```
episodes = 10
goal_unknown: PEDA (10 eps) vs pragmatic_only (10 eps)
  - 指标：成功率、平均步数、revisit rate
  - 检验：Mann-Whitney U test（步数）或 Fisher exact test（成功率）
goal_known: PEDA (10 eps) vs pragmatic_only (10 eps)
  - 验证公平性：两者应该无显著差异
```

---

## 7. 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 实验设计改进 | 9/10 | 从"无法验证"跃迁到"可以验证" |
| 代码实现 | 8/10 | pragmatic_only隔离正确，控制变量严谨 |
| 结果强度 | 7/10 | goal_unknown差异大（2步 vs 20步失败） |
| 统计可信度 | 2/10 | 1 episode，无法排除运气 |
| 自我诚实 | 9/10 | 报告承认所有局限性 |
| **综合** | **6/10** | **方法正确，数据不足** |

---

## 8. 结论与建议

### 8.1 核心假设验证状态

**尚未通过，但有强烈信号。**

实验设计是正确的（部分训练 + 基线对比），代码实现是公平的（pragmatic_only正确隔离），结果方向是积极的（PEDA >> pragmatic_only in goal_unknown）。但1 episode无法给出统计可信的结论。

### 8.2 下一步（优先级排序）

**P0（必须做，blocking）**：
- 运行 **10 episodes/condition** 的部分训练评估
- 使用现有的 `phase1_partial_eval.py`，只需修改 `--episodes 10`
- 在CPU上约需5小时，可以 overnight 跑
- 如果PEDA在goal_unknown条件下显著优于pragmatic_only（p<0.05）→ 核心假设在Grid World中验证通过
- 如果无显著差异 → 分析原因（pragmatic_weight过大？drive权重不对？）

**P1（应该做，不blocking）**：
- 多epoch训练（2-3 epochs）启用真正的ensemble epistemic error
- 验证confidence-based epistemic vs ensemble-based epistemic的差异

**P2（可以做，不blocking）**：
- 降低pragmatic_weight（从3.0到1.0或0.5），测试PEDA优势是否仍然成立
- 注意：两个agent同时使用相同的weight

**P3（不要做）**：
- 进入Phase 1.5（直到P0完成）
- 写更多文档/计划（直到P0完成）
- 修复lint（除非blocking测试运行）

### 8.3 给本地Agent的提示词

```
## Phase 1 核心假设验证：统计补充实验

### 当前状态

部分训练验证的pilot结果显示强烈信号：
- goal_unknown: PEDA 2步成功 vs pragmatic_only 20步失败
- 但仅1 episode/condition，统计量不足

### 任务

运行统计补充实验：10 episodes/condition。

### 具体步骤

1. 使用现有的 `scripts/phase1_partial_eval.py`
2. 命令：
   ```bash
   python scripts/phase1_partial_eval.py \
     --model Qwen/Qwen2.5-0.5B-Instruct \
     --adapter checkpoints/phase1/partial_adapter_real_25 \
     --episodes 10 \
     --pragmatic-weight 3.0 \
     --output results/phase1_partial_eval_10eps.json
   ```
3. 等待完成（约5小时）
4. 分析结果：
   - goal_unknown成功率：PEDA vs pragmatic_only（Fisher exact test）
   - goal_unknown平均步数：PEDA vs pragmatic_only（Mann-Whitney U test）
   - goal_known公平性验证：两者应该无显著差异

### 成功标准

| 条件 | PEDA | pragmatic_only | 统计检验 |
|------|------|---------------|---------|
| goal_unknown | 成功率 > 60% | 成功率 < 40% | p < 0.05 |
| goal_unknown | 平均步数 < 10 | 平均步数 > 15 | p < 0.05 |
| goal_known | 成功率 ≈ pragmatic_only | — | p > 0.05（无显著差异） |

如果全部满足 → 核心假设在Grid World中验证通过，可以进入Phase 1.5。
如果任何一条不满足 → 分析原因，调整参数，重新验证。

### 绝对不要做的事

- 不要进入Phase 1.5直到本实验完成
- 不要写新文档/计划
- 不要修复lint
- 不要添加新模块

### 时间估计

运行：5小时（overnight）  
分析：30分钟  
总计：~6小时
```

---

## 9. 最终判断

**PEDA正在走上正确的道路。**

从第一轮（完美WM + 贪心导航 = 无法验证）到本轮（部分WM + 基线对比 = 可以验证但有噪声），实验设计经历了根本性的改进。代码实现是公平的，结果方向是积极的，团队展现了自我诚实。

唯一缺失的是**统计量**。10 episodes的补充实验将给出可信的结论。如果结论积极，PEDA将成为一个罕见的例子——一个从前代项目的废墟中，通过诚实的自我反思和严格的实验设计，真正验证核心假设的AI Agent项目。

如果结论消极，那也是一个有价值的结论——它告诉我们"预测误差驱动探索"在Grid World中不成立，从而避免在更复杂的环境中浪费更多时间。

无论结果如何，这种**方法学的严谨性**比任何具体结果都更重要。
