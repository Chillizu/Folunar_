# 任务：Phase 1 核心假设验证 — 10 Episodes 统计补充实验

## 背景

上一轮 partial training pilot（1 episode/condition）显示了强烈信号：
- goal_unknown: PEDA 2步成功 vs pragmatic_only 20步失败（revisit=0.905）
- goal_known: 两者相同（3步），证明实验公平

但 1 episode 无法排除运气。需要 10 episodes/condition 给出统计可信的结论。

## 任务

运行 `scripts/phase1_partial_eval.py`，将 episodes 从 1 增加到 10。

## 具体命令

```bash
# 使用已有的 adapter（不要重新训练）
python scripts/phase1_partial_eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter checkpoints/phase1/partial_adapter_real_25 \
  --episodes 10 \
  --pragmatic-weight 3.0 \
  --max-candidates 4 \
  --output results/phase1_partial_eval_10eps.json \
  --eval-seed 42
```

如果 `--episodes` 参数不存在于当前脚本，检查 `phase1_partial_eval.py` 的 CLI 参数并添加它（修改 should 简单：找到 episode 循环的硬编码值，替换为 `args.episodes`）。

## 成功标准（全部满足才能进入 Phase 1.5）

| 条件 | PEDA | pragmatic_only | 最低要求 |
|------|------|---------------|---------|
| goal_unknown 成功率 | > 60%（6/10） | < 40%（4/10） | PEDA 显著更高 |
| goal_unknown 平均步数 | < 10 | > 15 | PEDA 显著更低 |
| goal_known 成功率 | ≈ pragmatic_only | — | 两者无显著差异（验证公平性） |

**如果任何一条不满足**：
1. 不要进入 Phase 1.5
2. 分析原始数据：哪些 episodes PEDA 失败了？失败模式是什么？
3. 检查是否是 pragmatic_weight=3.0 过大导致的问题
4. 报告具体原因，等待进一步指导

## 交付物

1. `results/phase1_partial_eval_10eps.json`（原始结果文件）
2. 一份简短的分析（可以直接写在这个对话中，不需要单独文档）：
   - 每组的成功率、平均步数、revisit rate
   - 与 1-episode pilot 的对比
   - 是否满足成功标准
   - 如果不满足，失败原因分析

## 绝对不要做的事

- ❌ 不要进入 Phase 1.5（无论结果多好，必须先完成本实验）
- ❌ 不要重新训练 World Model（使用已有的 `partial_adapter_real_25`）
- ❌ 不要修改 drive weights（使用当前 grid search 结果）
- ❌ 不要写新文档/更新 AGENTS.md
- ❌ 不要修复 lint（除非 blocking 测试运行）
- ❌ 不要添加新模块

## 时间估计

- 修改 `--episodes` 参数：5 分钟
- 运行 10 episodes × 2 conditions × ~15min：约 **5 小时**
- 分析结果：30 分钟
- **建议 overnight 跑**，明天看结果

## 如果运行中断

由于每 episode 独立，中断后可以：
1. 记录已完成的 episodes 数
2. 重新运行（相同 `--eval-seed 42` 会复现相同的起点/目标序列）
3. 手动合并两次运行的结果

## 核心原则

> **这是 Phase 1 的最后一个验证实验。它的结果决定 PEDA 的核心假设是否成立，从而决定是否能进入 Phase 1.5。**
>
> 不要简化、不要跳过、不要美化结果。
> 如果 PEDA 在 10 episodes 后仍然显著优于 pragmatic_only → 核心假设验证通过。
> 如果 PEDA 没有显著优势 → 核心假设在此条件下不成立 → 需要分析原因。
>
> 无论结果如何，**诚实报告**比"看起来好"更重要。
