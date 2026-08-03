# 任务：Phase 1.5 文本 World Model 训练

## 当前状态

- Grid World 已放弃（25% 和 10% 都试了，epistemic 信号不够）
- 自定义文本环境已搭建（2 房间：书房 → 走廊 → 宝箱）
- LLM 预测测试：格式正确，内容错误（预期——未经文本训练）

## 任务

生成文本环境训练数据 + LoRA 微调文本 World Model。

## 具体步骤

### 1. 生成训练数据（10 分钟）

写 `scripts/phase1_5_synthetic_train.py`：

```python
# 核心逻辑：
# 1. 创建 TextRoomEnv（2 房间）
# 2. 随机游走收集 (state, action, next_state) 三元组
# 3. 用 Perception.render_text() 将状态转为文本描述
# 4. 保存为 JSONL：{"state_text": "...", "action": "go north", "next_state_text": "..."}
# 5. 目标：500-1000 条训练样本
# 
# 收集策略：
# - 从每个房间出发，执行每个合法动作
# - 加上随机游走（50-100 步/seed，多个 seed）
# - 确保覆盖所有状态-动作组合（2 房间 × ~5 动作 = ~10 种组合，但文本描述有变化）
```

### 2. LoRA 微调（20-30 分钟）

复用 `phase1_synthetic_train.py` 的 LoRA pipeline，但：
- 输入：文本状态描述 + 动作 → 输出：下一状态描述 + exit_code
- 3 epochs，保存 3 checkpoints
- checkpoint_dir 必须传递（用于 ensemble epistemic）

```bash
python scripts/phase1_5_synthetic_train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-data data/phase1_5_synthetic.jsonl \
  --epochs 3 \
  --checkpoint-dir checkpoints/phase1_5/text_adapter_e3
```

### 3. 验证（5 分钟）

```bash
# 探针检查：3 个 checkpoint 是否有 ensemble 分歧
python -c "
from phase1.world_model import WorldModel, EnsembleErrorComputer
from phase1_5.text_env import TextRoomEnv

wm = WorldModel('Qwen/Qwen2.5-0.5B-Instruct', adapter_path='checkpoints/phase1_5/text_adapter_e3')
ec = EnsembleErrorComputer(wm)
# ... 探针逻辑（类似 Grid World 的探针）
# 目标：至少 1/3 的 state-action 对有 ensemble 分歧
"
```

**如果探针分歧 < 1/3**：
- 增加训练数据（更多随机游走）
- 或增加房间复杂度（3 房间 + 更多物体）
- 或降低 train_fraction（只训练部分状态-动作组合）

**如果探针分歧 >= 1/3**：
- 跑完整评估：PEDA vs pragmatic_only，10 episodes
- 成功标准同 Grid World：PEDA 在未知条件下显著优于 pragmatic_only

## 成功标准

| 指标 | 阈值 |
|------|------|
| 训练样本数 | >= 500 |
| g1_test_set（文本 OOD）| > 0.60 |
| ensemble 探针分歧 | >= 1/3 |
| PEDA vs pragmatic_only（10 eps）| PEDA 成功率 > pragmatic + 10% |

## 绝对不要做的事

- ❌ 不要写独立脚本后又花 30 分钟参数化合并——先跑通，再优化
- ❌ 不要增加超过 3 个房间——如果 2 房间不能产生 epistemic 信号，3 房间也不行
- ❌ 不要写文档 / 更新 AGENTS.md
- ❌ 不要修复 lint
- ❌ 不要添加新模块（phase1_5 目录下已有的文件够用）

## 时间估计

- 训练数据生成：10 分钟
- LoRA 微调：20-30 分钟
- 探针验证：5 分钟
- 总计：约 45 分钟

## 核心原则

> 2 房间 + 0.5B + 3 epochs + 500 样本。
> 
> 如果这能产生 epistemic 信号 → PEDA 核心假设在文本环境中可验证。
> 如果不能 → 问题不在环境大小，在训练方法或模型能力，需要更深层分析。
>
> 不要追求"更大更好的环境"，追求"能产生不确定性的最小环境"。
