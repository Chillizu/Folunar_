# Phase 1.5 Iteration 2 评估

> **评估对象**: `phase1_5_iteration2_report.md`
> **上游决策**: 是否进入 Phase 2

---

## 执行质量：8/10

| 维度 | 评分 | 说明 |
|------|------|------|
| 诚实性 | 10/10 | 明确报告数据增强失败（114 vs 目标500-1000），未隐瞒 |
| 技术执行 | 8/10 | decompose_error 修复正确，但数据增强策略需改进 |
| 分析深度 | 8/10 | 正确识别根因（状态空间太小），推荐合理 |
| 时间控制 | 7/10 | 总时间~62min，略超1小时预算，但可接受 |

---

## 关键结果

### 1. decompose_error 修复：成功 ✅

| 指标 | e3 | e4 | 变化 |
|------|-----|-----|------|
| PEDA epistemic | 0.0000 | **0.2000** | +0.20 |
| Pragmatic epistemic | 0.0000 | **0.2222** | +0.22 |

与语义探针预期（20-40%）一致。测量 bug 已修复。

### 2. 数据增强：失败 ❌（但失败原因有价值）

- 200 walks × 30 steps → 去重后 114 条（+1）
- 根因：**2 房间环境的状态空间太小**
- `state_text + action_name` 组合高度重复
- 这个发现意味着：在当前环境复杂度级别，数据增强不可行

### 3. 模型学习：未改善 ⚠️

| 动作 | e3 预测 exit | e4 预测 exit | 正确值 |
|------|-------------|-------------|--------|
| take key | 1 ❌ | 1 ❌ | 0 |
| go north | 2 ❌ | 1 ❌ | 0 |

`go north` 甚至变差了（2→1）。114 条数据不足以让 0.5B 模型学会转移动态。

### 4. 行为差异：复现 ✅

| Agent | e3 行为 | e4 行为 |
|-------|---------|---------|
| PEDA | step 3 尝试 take key | **step 1 就尝试 take key** |
| Pragmatic | look × 20 | look × 10 |

PEDA 比 e3 更早探索。这个模式一致且可复现。

---

## 根因确认

```
核心假设（epistemic_error 驱动有意义的探索）未验证
    │
    ├─→ 直接原因：环境太简单 + 模型没学好
    │       │
    │       ├─→ 2 房间状态空间太小 → 数据增强不可行（去重后几乎无变化）
    │       ├─→ 114 条数据 → 0.5B 模型学不好 take key / go north
    │       └─→ 即使 epistemic=0.20，任务链（3步最优路径）仍无法完成
    │
    └─→ 替代发现（已验证 ✅）：
            ├─→ PEDA ≠ Pragmatic 行为可区分（2/2 迭代复现）
            ├─→ Drive System 有独立价值（boredom + epistemic_ratio 驱动探索）
            └─→ decompose_error 修复后 epistemic 信号可测量
```

---

## 决策：进入 Phase 2

**理由**：

1. **Phase 1.5 已完成其使命**
   - 验证了工程基础设施（文本环境 + WMs + 训练 + 探针 + 评估）✅
   - 发现了 decompose_error bug 并修复 ✅
   - 确认 PEDA ≠ Pragmatic 行为可区分 ✅
   - 证明 2 房间环境无法验证核心假设（状态空间太小）✅

2. **继续 Iteration 的边际收益极低**
   - 3-4 房间环境：需要重新设计环境 + 更多数据 → 1-2 周
   - 即使成功，只是验证了"稍微复杂一点的文本环境"
   - Phase 2 的 busybox 提供真实不确定性，更有信息量

3. **负结果已完成积累**
   - Grid World：环境太简单，0.5B 完美泛化 → 无 epistemic
   - Text World 2 房间：数据空间太小，无法增强 → 模型学不好
   - 两个环境都说明同一问题：**当前环境复杂度与模型能力不匹配**
   - Phase 2 的 busybox 环境复杂度天然足够

4. **Drive System 发现值得继续**
   - PEDA 在 epistemic≈0 时仍能与 Pragmatic 区分
   - 说明 Drive System（boredom + novelty + LLM 置信度）有独立价值
   - 在更复杂的 Phase 2 环境中，这个发现可能更有用

---

## Phase 1.5 总结

### 已验证 ✅

| 发现 | 证据强度 |
|------|----------|
| PEDA ≠ Pragmatic 行为可区分 | 高（2/2 迭代复现） |
| Drive System 有独立探索价值 | 高（boredom + epistemic_ratio 驱动） |
| decompose_error bug 修复有效 | 高（0.0 → 0.20） |
| 2 房间环境状态空间太小 | 高（6000 尝试 → 114 去重） |

### 未验证 ❌

| 假设 | 状态 |
|------|------|
| Epistemic error 驱动有意义探索 | 未验证（环境太简单，模型没学好） |
| EFE 优于贪心策略 | 部分（PEDA ≠ Pragmatic，但都未成功） |
| World Model 能学习文本转移动态 | 否（114 条数据不够） |

### 意外发现 🎁

1. **PEDA 在 step 1 就探索 take key** — 比 Iteration 1 更早，说明 epistemic 信号有累积效应
2. **数据增强在简单环境无效** — 6000 次尝试只增加 1 条样本，揭示了环境设计的约束
3. **go north 预测变差** — 暗示模型在尝试学习但方向错误（overfitting 或 insufficient data）

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `PEDA_FINAL/phase1_5_complete_report.md` | Iteration 1 完整报告 |
| `PEDA_FINAL/phase1_5_iteration2_report.md` | Iteration 2 报告（本次） |
| `PEDA_FINAL/PHASE1_5_COMPLETE_EVALUATION.md` | Iteration 1 上游评估 |
| `PEDA_FINAL/PHASE1_5_ITERATION2_EVALUATION.md` | Iteration 2 上游评估（本文件） |
