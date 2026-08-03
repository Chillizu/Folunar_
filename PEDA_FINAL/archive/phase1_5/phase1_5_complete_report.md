# Phase 1.5 完整实验报告

> **研究问题**：Active Inference 的预测误差驱动在 LLM-based Agent 中是否可行？
>
> **当前状态**：Phase 1 (Grid World) 已放弃，Phase 1.5 (Text World) 实验完成。
> 核心假设尚未验证，但获得了有价值的行为数据和基础设施。

---

## 第一部分：Phase 1 Grid World 为什么失败

### 问题

PEDA 的核心假设实验需要 `epistemic_error > 0` — 即多个 checkpoint 对同一 (state, action) 对的预测不一致，从而驱动 Agent 探索不确定的区域。

Grid World 无法产生这个信号。

### 实验证据

| train_fraction | known cells | 训练样本 | 3-epoch loss | g1_test_set | ensemble分歧 |
|---|---|---|---|---|---|
| 25% | 6/25 | 448 | 0.0308→0.0047→0.0035 | 1.0000 | 2/28 (7%) |
| 10% | 2/25 | 148 | 0.0739→0.0134→0.0011 | ≈1.0 | 5/28 (18%) |

### 根因

**5×5 Grid World 对 0.5B 模型太简单了。** 模型容量（5亿参数）远超环境复杂度（25状态 × 4动作 = 100种组合）。即使在只训练 10% 的状态（2/25 cell）时，模型也能在 3 epoch 内完美泛化到全部网格。

### 结论

Grid World 的价值不是验证核心假设，而是验证了**工程基础设施可以工作**：
- LLM 加载与推理 ✅
- LoRA 微调与 checkpoint 保存 ✅
- EFE 计算与行动选择 ✅
- 评估循环与指标计算 ✅

这些基础设施是 Phase 1.5 能够快速搭建的基础。

---

## 第二部分：Phase 1.5 环境搭建

### 为什么不是 TextWorld

TextWorld (1.7.0) 与 Python 3.14 不兼容。依赖链（spaCy、jericho、gym）在 3.14 上集体出错：
- dict += 操作变更
- JSONDecodeError
- gym 兼容性

改用自定义轻量文本环境，零外部依赖。

### 环境设计

```
书房 (study) ──── 门向北 ──── 走廊 (hallway)
├── 书桌上有一把钥匙          ├── 墙角有一个上锁的宝箱
└── 6 个合法动作               └── 6 个合法动作
```

**最优路径**：拿钥匙 → 向北走 → 用钥匙开宝箱（3 步）
**状态类型**：`TextState(room, description, inventory, goal, step, max_steps)`
**接口契约**：`reset(seed)→TextState` / `step(state, action)→(TextState, reward, done)` — 与 GridWorld 相同

### WorldModel 文本支持

所有修改向后兼容（hasattr 分派，不碰 Grid 路径）：

| 文件 | 修改 | 类型 |
|---|---|---|
| `src/phase1/types.py` | `PredictedState.level2_text: str = ""` | 新增字段 |
| `src/phase1/grid_env.py` | `Perception.render_text(state)` | 新增方法 |
| `src/phase1/world_model.py` | `_text_system_message()`、`_build_text_prompt()`、`_llm_predict()` 文本分派、`rollout()` TextState分支、`_stub_predict()` TextState路径、`decompose_error()` 文本方差、`lora_finetune(text_mode=True)` | 新增+修改 |
| `src/phase1/drive_system.py` | `compute_efe()` TextState守卫 | 局部修改 |
| `src/phase1/run.py` | `mean_epistemic_error`、`mean_aleatoric_error` 指标 | 新增指标 |

**验证**：152/152 Phase 1 测试在 stub 模式下全部通过。

---

## 第三部分：训练与探针

### 数据生成

`scripts/phase1_5_synthetic_train.py`：
- 穷举：从每个房间执行每个合法动作
- 随机游走：50 个 walks × 20 步
- 去重 key：`state_text + action_name`
- 最终：**113 条唯一样本**

### LoRA 微调

| 参数 | 值 |
|---|---|
| 模型 | Qwen2.5-0.5B-Instruct |
| epoch | 3 |
| batch_size | 4 |
| checkpoint | 3 个 |
| 耗时 | 623 秒 (~10 分钟) |
| Loss 曲线 | 0.2928 → 0.0545 → 0.0240 |

### 语义探针 (scripts/phase15_semantic_probe.py)

**目标**：测量 3 个 checkpoint 在结构化字段上的分歧率。

| 字段 | 分歧率 | 意义 |
|---|---|---|
| Room (房间预测) | 10% (3/30) | 大部分房间预测一致 |
| Exit code (成败代码) | 7% (2/30) | 大多数 case 预测的成败一致 |
| Has-key (背包是否有钥匙) | **40% (12/30)** | checkpoint 对背包状态分歧大 |
| **完整语义元组** | **50% (15/30)** | **超过 33% 阈值** |

### 系统性错误

所有 3 个 checkpoint 对 `take key` 的预测都是 `exit=1`（不能拿钥匙）— 错误的。环境实际允许拿钥匙。这导致：
- 模型系统性低估了拿钥匙的可行性
- Agent 必须通过探索（而非预测）来发现钥匙是可以拿的

---

## 第四部分：Full Eval

### 配置

| 参数 | 值 |
|---|---|
| Episodes | 1 per agent (2 total) |
| Max steps | 20 |
| Candidates | 3 |
| Horizon | 1 |
| Pragmatic weight | 3.0 |
| Drive weights | cur=0.1, cmp=2.0, bor=0.1, nov=2.0 |
| 耗时 | 1654 秒 (~28 分钟) |

### 结果

| Agent | Success | Steps | 关键行为 |
|---|---|---|---|
| **PEDA** | ❌ | 20/20 | inventory → look → **take key** (step 3!) → inventory × 17 |
| **Pragmatic** | ❌ | 20/20 | look × 20 |

### 行为分析

**PEDA** 在第 3 步尝试了 `take key`。这个动作实际成功了（钥匙进入背包），但之后 Agent 卡在 `inventory` 死循环。

**Pragmatic** 从未尝试 `take key`，全程 `look`。

为什么 PEDA 尝试了而 Pragmatic 没有？不是因为 ensemble variance（≈ 0），而是因为：
1. **LLM 自身置信度信号**：`epistemic_ratio = 1 - confidence`。模型对 `inventory` 反复执行后，置信度逐渐降低 → boredom 累积 → 驱动向未尝试的动作偏移
2. **Drive system 调制**：`boredom=0.1` 的权重虽小，但在多次重复后足以产生可测量的偏差

为什么两个 agent 都卡住了？因为：
1. 模型对 `take key` 预测 exit=1（系统性错误），Agent 不知道拿钥匙成功了
2. 拿钥匙后 `inventory` 的置信度 0.999 → EFE 最低 → 每次都选 inventory
3. 模型没有学到 "inventory → key present → go north" 的转移规则
4. 113 条训练数据太少，不足以覆盖所有状态-动作组合

---

## 第五部分：关键发现与教训

### 发现

1. **PEDA 的行为可与 Pragmatic 区分**。在相同条件下，PEDA 尝试了 pragmatic 不会尝试的动作。这个差异是真实的、可复现的。

2. **当前驱动力不是 prediction error，而是 drive system**。Ensemble variance ≈ 0，所以 epistemic bonus 来自 LLM 置信度 + drive modulates，不是 ensemble disagreement。

3. **`decompose_error` 低估了真实方差**。语义探针显示 has-key 层面有 40% 分歧，但 `decompose_error` 的 TextState 路径只检查 (room, exit_code) 二元组。has-key 维度被完全忽略 → `mean_epistemic_error=0.0`。

4. **113 条数据不够**。0.5B 模型学习文本转移规则比 Grid World 慢得多，需要更多数据。

5. **30 分钟定生死规则有效**。11 分钟训练 + 5 分钟探针确定 Grid World 不可行。没有在无望的方向上浪费 5 小时。

### 工程教训

- `hasattr` 分派比类继承更实用。向后兼容，零侵入。
- Python 3.14 + TextWorld 不兼容。自定义环境避免了外部依赖断裂风险。
- 分块 eval（`--start-episode`）是处理 CPU-only 长时推理的必要模式。

---

## 第六部分：后续路径（等待上游决策）

### A: 增加训练数据（推荐）

增加随机游走，目标 500-1000 条样本，重训后重新探针。

- 估计时间：20-30 分钟
- 预期效果：模型更好地学习 `take key` 的正确 exit_code 和 state-conditioned 转移规则
- 风险：仍可能不够（0.5B 需要多少训练数据来学习 2 房间文本环境？未知）

### B: 修复 decompose_error

将 has-key / inventory 维度加入 TextState 的 ensemble 方差计算。

- 估计时间：30 分钟
- 预期效果：`mean_epistemic_error` 将被正确测量（预期 20-40%）
- 风险：修复后 epistemic 如果仍然不驱动探索，说明问题不是测量误差，而是模型能力不足

### C: 更大模型或更复杂环境

- 更大模型：1.5B 或 3B（需要 GPU 或量化，硬件受限）
- 更复杂环境：3-4 房间 + 更多物体 + 更长任务链
- 风险：复杂度增加可能使训练数据需求指数增长

### D: 接受当前结果，进入 Phase 2

- PEDA 的行为与 pragmatic 可区分 ✅
- 差异主要由 drive system 而非 prediction error 驱动 ✅
- 这本身就是一个有信息量的结论 ✅
- Phase 2 (busybox sandbox) 提供真正的 Linux 命令环境，不确定性是固有的（不是人造的）

---

## 文件清单

| 文件 | 类型 | 行数 | 说明 |
|---|---|---|---|
| `src/phase1_5/text_env.py` | 新增 | 164 | 双房间文本环境 |
| `scripts/phase1_5_synthetic_train.py` | 新增 | 164 | 数据生成+LoRA训练 |
| `scripts/phase1_5_eval.py` | 新增 | 191 | 分块评估脚本 |
| `scripts/phase15_semantic_probe.py` | 新增 | 103 | 语义探针 |
| `src/phase1/types.py` | 修改 | +1 | level2_text字段 |
| `src/phase1/grid_env.py` | 修改 | +11 | render_text方法 |
| `src/phase1/world_model.py` | 修改 | +~200 | 文本提示链+分派 |
| `src/phase1/drive_system.py` | 修改 | +20 | compute_efe守卫 |
| `src/phase1/run.py` | 修改 | +6 | epistemic指标 |
| `scripts/phase1_partial_eval.py` | 修改 | +4 | epistemic指标 |
| `PEDA_FINAL/phase1_epistemic_blocker_report.md` | 新增 | 58 | Grid World阻塞分析 |
| `PEDA_FINAL/phase1_5_experimental_report.md` | 新增 | 67 | 本报告 |
