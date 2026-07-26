# Phase 1.5 完整评估

> **评估对象**: `phase1_5_complete_report.md`（commit `5fbd1cf`）
> **评估者**: 上游 Advisor
> **日期**: 2026-07-06

---

## 一、报告质量评估

### 评分：8/10

| 维度 | 评分 | 说明 |
|------|------|------|
| 诚实性 | 10/10 | 未夸大结果，明确标注"核心假设尚未验证" |
| 分析深度 | 8/10 | 行为分析到位，根因推断合理 |
| 数据完整性 | 7/10 | 关键数据都有，但缺少置信度数值的原始分布 |
| 可复现性 | 8/10 | 配置、参数、文件清单完整 |
| 决策支持 | 8/10 | 四个选项清晰，利弊权衡到位 |

### 亮点

1. **"为什么 PEDA 尝试了而 Pragmatic 没有？"** — 这个追问非常关键。报告正确识别驱动力不是 ensemble variance 而是 LLM 置信度 + boredom drive。
2. **系统性错误发现** — 所有 checkpoint 对 `take key` 预测 exit=1，这是一个很有价值的发现，说明模型容量/数据量不足以学习这个转移规则。
3. **30 分钟规则自证** — 快速放弃 Grid World 的决策被证明是正确的。

### 不足

1. **统计显著性未讨论** — 只有 1 episode per agent，PEDA 的 `take key` 行为可能是随机波动。虽然 Pragmatic 的 `look x20` 看起来很稳定，但 1 episode 不足以确认 PEDA 的行为模式。
2. **置信度数值缺失** — 报告提到"置信度逐渐降低"和"置信度 0.999"，但没有给出置信度分布的原始数据（哪些 action 的 confidence 是多少？）。
3. **选项 A 的风险低估** — "0.5B 需要多少训练数据来学习 2 房间文本环境？未知"这个风险评估准确，但没有讨论另一个风险：即使数据量够了，ensemble variance 可能仍然很小（因为模型可能只是学会确定性规则，而不是概率分布）。

---

## 二、关键发现验证

### 验证 1：PEDA ≠ Pragmatic（可信 ✅）

PEDA 在第 3 步尝试了 `take key`，Pragmatic 全程 `look`。差异真实存在。

**但是**：1 episode 的统计显著性存疑。建议至少 3-5 episodes 确认行为模式。

### 验证 2：驱动力来自 Drive System 而非 Prediction Error（可信 ✅）

`mean_epistemic_error=0.0` + ensemble variance ≈ 0 → epistemic bonus 不可能是驱动力。

驱动来源：
- `epistemic_ratio = 1 - confidence`（LLM 自身不确定性）
- `boredom=0.1` 权重在多次重复后累积

**这个发现本身有价值** — 即使 prediction error 不够强，drive system 仍能产生可区分的行为。

### 验证 3：decompose_error Bug（确认 ✅）

语义探针：has-key 分歧率 40%，但 decompose_error 只检查 (room, exit_code)。

**这是一个明确的 bug**，必须修复。修复后 `mean_epistemic_error` 预期从 0.0 上升到 20-40%。

### 验证 4：113 条数据不够（合理推断 ⚠️）

Loss 曲线 0.29 → 0.05 → 0.02，说明模型还在学习。但 113 条样本对于 0.5B 参数模型确实太少。

**但是**：增加数据不一定解决根本问题。如果环境太简单，模型可能学到确定性规则 → ensemble variance 仍然小。

---

## 三、根因分析

```
核心假设（epistemic_error 驱动探索）未验证
    │
    ├─→ 直接原因：ensemble variance ≈ 0
    │       │
    │       ├─→ decompose_error bug（has-key 维度被忽略）— 可修复
    │       └─→ 模型对大多数 (state, action) 预测过于确定 — 需要更多数据/更复杂环境
    │
    └─→ 替代路径：drive system 产生可区分行为（已验证 ✅）
            │
            ├─→ LLM 置信度信号（1 - confidence）
            └─→ boredom / novelty drive 累积
```

**关键问题**：ensemble variance 小的根本原因是什么？

| 假设 | 证据 | 验证方法 |
|------|------|----------|
| 数据太少，模型没学好 | Loss 还在下降 | 增加数据→重训→探针 |
| 环境太简单，模型学会确定性规则 | Grid World 的历史 | 增加环境复杂度→探针 |
| decompose_error 低估了方差 | 语义探针 40% vs decompose 0% | 修复 bug→重新测量 |
| 0.5B 模型容量不足 | take key 系统性错误 | 换更大模型（硬件受限） |

---

## 四、决策建议

### 推荐路径：B → A → 评估 → 决定

**第一步：修复 decompose_error（30 分钟）**

这是 must-do。不修复就无法知道真实的 epistemic 水平。

修复内容：
- TextState 的 `decompose_error()` 加入 has-key / inventory 维度的方差计算
- 预期效果：`mean_epistemic_error` 从 0.0 → 20-40%

**第二步：增加训练数据至 500-1000 条（20-30 分钟）**

理由：
- 113 条数据确实太少（Loss 还在下降）
- 20-30 分钟的时间成本可以接受
- 即使失败，也能获得"数据量不足"的负结果

**第三步：3-5 episodes 评估（1-2 小时）**

配置：
- 修复 decompose_error 后的完整评估
- PEDA vs Pragmatic，各 3-5 episodes
- 记录：success rate、epistemic/aleatoric 均值、行为轨迹

**第四步：决策点**

| 结果 | 决策 |
|------|------|
| epistemic_error > 0 且 驱动探索 | 核心假设验证成功 → 继续优化 |
| epistemic_error > 0 但不驱动探索 | Drive system 有价值 → 记录 → Phase 2 |
| epistemic_error ≈ 0（修复后） | 环境/模型不匹配 → 接受负结果 → Phase 2 |
| PEDA success > Pragmatic success | 有实用价值 → 继续优化 |
| 两者都 success=0 | 数据/模型问题 → 增加数据或接受 → Phase 2 |

### 为什么不直接选 D（接受结果，进入 Phase 2）

D 是合理的选项，但 B+A 的时间成本只有 1 小时，可能换来核心假设的验证。如果 B+A 后仍然失败，那时再选 D 更有说服力。

### 为什么不选 C（更大模型/更复杂环境）

- 硬件受限（CPU-only），1.5B+ 模型推理速度会显著下降
- 复杂度增加→数据需求指数增长→评估时间更长
- 应该先确认问题是数据量还是环境复杂度

---

## 五、给本地 Agent 的具体指令

### 任务 1：修复 decompose_error

```
文件：src/phase1/world_model.py
方法：decompose_error() 的 TextState 分支

当前逻辑：只检查 (room, exit_code) 二元组的方差
修复逻辑：增加 has-key 维度

  predicted_key_status = "key" in (predicted_state.inventory or [])
  actual_key_status = "key" in (actual_next_state.inventory or [])
  
  或者检查 predicted_state.level2_text 中的关键词

目标：让 decompose_error 的输出与语义探针的分歧率一致（20-40%）
```

### 任务 2：增加训练数据

```
文件：scripts/phase1_5_synthetic_train.py

修改：
- 随机游走：50 walks × 20 步 → 200 walks × 30 步
- 增加策略多样性：
  - 纯随机游走
  - 目标导向（优先尝试未访问的 action）
  - 重复执行（同一 action 连续执行 3-5 次）
- 目标：500-1000 条唯一样本
```

### 任务 3：快速验证

```
训练后执行：
1. 语义探针（确认 has-key 分歧率）
2. decompose_error 修复验证（确认 mean_epistemic_error > 0）
3. 1 episode PEDA + 1 episode Pragmatic（快速行为检查）

总时间预算：1 小时
```

---

## 六、风险与备案

| 风险 | 概率 | 应对 |
|------|------|------|
| 增加数据后 ensemble variance 仍然 ≈ 0 | 中 | 记录负结果 → Phase 2 |
| 评估时间超预期（CPU-only 推理慢） | 高 | 用 `--start-episode` 分块，每天跑一部分 |
| decompose_error 修复后引入新 bug | 低 | 语义探针交叉验证 |
| 0.5B 模型根本学不好文本环境 | 中 | 这是有价值的信息，直接进 Phase 2 |

---

## 七、结论

Phase 1.5 是成功的**探索** — 不是核心假设的验证，但获得了多个有价值的发现：

1. **Drive system 有独立价值**（PEDA ≠ Pragmatic）
2. **decompose_error 有 bug**（测量失真）
3. **系统性错误模式**（take key 的 exit code）
4. **数据量需求估计**（113 条 << 实际需求）

下一步：修复 bug + 增加数据（1 小时投资），然后做最终决策。

---

## 附录：评估 Checklist

| Check | 状态 |
|-------|------|
| 核心假设是否验证？ | ❌ 未验证，但有进展 |
| 实验是否可复现？ | ✅ 配置完整 |
| 是否有行为差异？ | ✅ PEDA ≠ Pragmatic |
| 是否发现新 bug？ | ✅ decompose_error |
| 统计显著性是否足够？ | ⚠️ 只有 1 episode |
| 下一步是否清晰？ | ✅ B → A → 评估 → 决策 |
| 时间预算是否合理？ | ✅ 1 小时 |
