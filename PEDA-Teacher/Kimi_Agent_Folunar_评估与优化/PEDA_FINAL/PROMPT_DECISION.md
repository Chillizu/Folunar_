# 任务：30 分钟定生死 — 0.10 train-fraction 或 Phase 1.5

## 当前状态

3-epoch adapter 训练完成，但 epistemic_error ≈ 0。根因：5x5 Grid World 对 0.5B 模型太简单（g1=1.0，28 个探针中只有 2 个有 ensemble 分歧）。

## 任务（两步走，不要讨论，直接执行）

### 第一步：30 分钟试错（train-fraction 0.10）

```bash
# 1. 重训（约 20-30 分钟）
python scripts/phase1_synthetic_train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-fraction 0.10 \
  --epochs 3 \
  --output-dir checkpoints/phase1/partial_adapter_real_10

# 2. 验证 checkpoint 数量
ls checkpoints/phase1/partial_adapter_real_10/checkpoint_epoch_*
# 预期：3 个文件

# 3. 运行 epistemic 探针（5 分钟）
python -c "
from pathlib import Path
from phase1.grid_env import GridWorld
from phase1.world_model import WorldModel, EnsembleErrorComputer

wm = WorldModel('Qwen/Qwen2.5-0.5B-Instruct', adapter_path='checkpoints/phase1/partial_adapter_real_10')
ec = EnsembleErrorComputer(wm)
ec.checkpoints = sorted(Path('checkpoints/phase1/partial_adapter_real_10').glob('checkpoint_epoch_*'))
total, diverse = 0, 0
for s in [(0,0), (2,2), (4,4), (1,3), (3,1), (4,0), (0,4)]:
    state = GridWorld(goal=(4,4)).reset(seed=42)
    state.agent = s
    for action in GridWorld.all_actions():
        preds = ec._predictions_for(state, action)
        if len(set(p.level2_next_agent for p in preds)) > 1:
            diverse += 1
        total += 1
print(f'Epistemic alive: {diverse}/{total}')
"
```

**决策点**：
- 如果 `diverse >= 10`（至少 1/3 的 state-action 对有 ensemble 分歧）→ **epistemic 上线** → 跑完整 10 集评估（约 5 小时，overnight）
- 如果 `diverse < 10`（仍然太少分歧）→ **立即进入第二步**，不要犹豫

### 第二步：Phase 1.5（如果第一步失败）

Grid World 对 0.5B 模型太简单是确定性事实。不需要再降到 0.05 或更低。直接进入 Phase 1.5。

具体选择：
1. **TextWorld**（推荐）：Microsoft 的文本交互环境框架，pip install textworld，有可控复杂度的文本任务
2. **busybox sandbox**：真实的 Linux 命令行环境，但比 Grid World 复杂得多

参考文档：`PEDA架构设计与开发计划书_v1.1.docx` 第 4.3 节 Phase 1.5

## 绝对不要做的事

- ❌ 讨论"要不要试 0.05"——如果 0.10 都不行，Grid World 不适合验证 PEDA 核心假设
- ❌ 跑 10 集评估如果第一步 epistemic 探针显示 < 10 分歧
- ❌ 写文档 / 更新 AGENTS.md
- ❌ 修复 lint
- ❌ 添加新模块

## 时间线

```
0:00  — 开始重训 0.10
0:30  — 探针检查 epistemic
       ├── diverse >= 10 → 跑 10 集评估（overnight，明天看结果）
       └── diverse < 10  → 进入 Phase 1.5（开始搭建 TextWorld 环境）
```

## 核心原则

> Grid World 已经完成了它的使命：验证了工程基础设施可以工作（LLM 加载、LoRA 微调、EFE 计算、评估循环）。
> 
> 如果它不能产生 epistemic 信号，不需要强迫它。**换一个能产生真正不确定性的环境，比在不可能的环境中浪费 5 小时更有价值。**
>
> 30 分钟定生死。要么 epistemic 上线，要么进入 Phase 1.5。
