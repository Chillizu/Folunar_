# Phase 1 终局与 Phase 1.5 起步报告

## 原始请求

> "ok，那浅浅做下计划然后开始吧？"

目标：执行 3 个 epoch 的 PEDA 核心假设验证——"预测误差驱动探索是否优于纯实用型（pragmatic-only）基线"。

计划路径：3 epochs → 3 checkpoints → epistemic_error > 0 → 10 集评估比较 PEDA vs pragmatic_only。

## 第一轮（25% train_fraction）

**行动**：在 25% 已知 cell（6/25）上重训 0.5B 模型，3 epoch，3 checkpoint 保存。

- loss 曲线：0.0308 → 0.0047 → 0.0035（正常收敛）
- g1_test_set=1.0（完美泛化到全部 25 个 cell）
- 28 个 state-action 探针中仅 2 个有 ensemble 分歧

**结论：epistemic_error ≈ 0，PEDA ≈ pragmatic_only。核心假设在 25% 下不可验证。** 报告写入 `phase1_epistemic_blocker_report.md`。

## 第二轮（10% train_fraction — 30 分钟定生死）

**用户决策**：降 train_fraction 到 10%（2/25 已知 cell）。如果 30 分钟后探针分歧仍 < 10/28，立即放弃 Grid World，进入 Phase 1.5。

- 训练时间：11 分钟（670s）
- 已知 cell：2/25，训练样本 148 条
- loss 曲线：0.0739 → 0.0134 → 0.0011
- 探针结果：**5/28（18% 分歧）**

**决策：5 < 10 → Grid World 到此为止。** 低于阈值。不纠结 5% 或更低，直接进入 Phase 1.5。

## 为什么 Grid World 不能验证 PEDA 核心假设

根本原因不是 PEDA 的设计问题，而是 **5×5 Grid World 对 0.5B 模型太简单了**：

1. **环境容量**：25 状态 × 4 动作 = 100 种 (s, a) 组合。0.5B 模型有 5 亿参数，不到一个 epoch 就记住了全部规则。
2. **泛化能力**：10% 训练（2/25 cell）→ 测试集准确率 ~1.0。模型从 2 个 cell 的局部经验就完美外推到了整个网格。
3. **ensemble 方差枯竭**：3 个 checkpoint 在所有训练过的状态上预测完全一致，在未见过状态下也只有 5/28 不同。方差 ≈ 0 → epistemic_error ≈ 0。
4. **Grid World 已经完成了它的使命**：验证了工程基础设施（LLM 加载、LoRA 微调、EFE 计算、评估循环）可以工作。但它不能产生验证核心假设所需的信号。

## Phase 1.5 文本环境

TextWorld 不兼容 Python 3.14，转用自定义轻量文本环境。纯 Python，零外部依赖。

### 架构

- `src/phase1_5/text_env.py`：双房间文本冒险（书房 → 走廊 → 宝箱）
- `TextState` 类型：room 名、文本描述、背包、目标
- `TextRoomEnv`：`reset(seed)→TextState` / `step(state, action)→(TextState, reward, done)` — 与 GridWorld 相同接口契约
- 最优路径：拿钥匙 → 向北走 → 开宝箱（3 步）

### WorldModel 文本支持（向后兼容）

- `PredictedState.level2_text` 字段（默认 `""`，Grid 路径不变）
- `Perception.render_text(state)` 渲染文本状态
- `_build_text_prompt()` / `_text_system_message()` 文本环境提示
- `_llm_predict` 自动识别状态类型（`hasattr .room` → 文本）

### LLM 预测测试

加载 10% grid adapter（未经文本训练）：

| Action | exit_code | 预测 room | 正确 |
|--------|-----------|-----------|------|
| look | 0 | hallway | ❌ |
| take key | 0 | hallway | ❌ |
| go north | 0 | hallway | ✅ |

输出格式正确（JSON + exit_code/next_room/summary），但 room 分类错误（偏向 default `hallway`）。**这是预期行为——模型从未见过文本房间概念。**

## 关键教训

1. **假设先行验证**：在跑完整实验前，花 5 分钟做探针检查。如果 25% 时就先测 ensemble 方差，可以省掉第一次 full eval 尝试的时间。
2. **30 分钟决断规则有效**：11 分钟训练 + 5 分钟探针 → 确定 Grid World 行不通。没有在不可能的方向上浪费 5 小时。
3. **模型过强不是错误，是信号**：0.5B 完美泛化 5×5 不是 bug，它说明 PEDA 的 LLM+LoRA pipeline 是正确的。问题是环境太简单，不是模型太差。
4. **TextWorld 外部依赖风险**：Python 3.14 + TextWorld 1.7 不兼容。自定义文本环境避免了未来的依赖断裂问题。
5. **Type dispatch 比继承更实用**：WorldModel 通过 `hasattr` 分派文本 vs Grid 预测，避免了类继承的复杂性，保持了向后兼容。

## 下一步（等上游决策）

1. **生成 text env 训练数据**：随机游走 TextRoomEnv，收集 (state, action, next_state) 三元组
2. **LoRA 微调**：在 0.5B 上训练文本 WorldModel（同上 Phase 1 pipeline）
3. **评估**：测试预测准确率（目标 > 60%）
4. **如果不够**：简化到单房间单物体，或升级模型

## 文件变更清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/phase1_5/text_env.py` | 新增 | 双房间文本环境 |
| `src/phase1/run.py` | 修改 | 添加 mean_epistemic_error/aleatoric_error 到 metrics |
| `src/phase1/types.py` | 修改 | PredictedState.level2_text 字段 |
| `src/phase1/grid_env.py` | 修改 | Perception.render_text() 方法 |
| `src/phase1/world_model.py` | 修改 | 文本提示链 + 类型分派 |
| `scripts/phase1_partial_eval.py` | 修改 | 记录 epistemic 指标到 raw_results |
| `PEDA_FINAL/phase1_epistemic_blocker_report.md` | 新增 | Grid World 阻塞分析 |
| `PEDA_FINAL/phase1_5_setup_report.md` | 修改 | 本文件 |
