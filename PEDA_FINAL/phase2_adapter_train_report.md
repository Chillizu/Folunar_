# Phase 2 Sandbox LoRA Adapter 训练报告

**日期**：2026-07-18  
**目标**：在沙箱数据上训练一个 LoRA adapter，使 PEDA 第一步不再陷入 `ls data` 死循环。

## 数据来源

| 来源 | 文件 | 运行数 | 每运行步数 | transitions |
|---|---|---|---|---|
| Random baseline | `results/phase2_random_train.jsonl` | 5 tasks | 20 | 100 |
| Heuristic baseline | `results/phase2_heuristic_train.jsonl` | 5 tasks | 20 | 100 |
| **合并** | `results/phase2_train_merged.jsonl` | 10 | — | **200** |

数据通过 `scripts/phase2_collect_data.py` 的 `--all-tasks` 模式收集，每条 record 包含 `cwd`、`files`、`action`、`next_cwd`、`next_files`、`exit_code`、`output` 等字段。

## 训练配置

```bash
python scripts/phase2_synthetic_train.py \
  --data results/phase2_train_merged.jsonl \
  --output-dir checkpoints/phase2/sandbox_adapter_e1 \
  --epochs 3 \
  --batch-size 4
```

- Base model: `~/models/Qwen2.5-0.5B-Instruct`
- 训练模式：`sandbox_mode=True`（自动检测 `next_cwd`/`next_files` 键）
- `max_length=384`（避免 JSON 状态被截断）
- 优化器：AdamW，学习率 3e-4
- 环境：`OMP_NUM_THREADS=4` 等限制，避免 CPU 超订

## 损失曲线

| Epoch | Avg Loss |
|---|---|
| 1/3 | 0.0291 |
| 2/3 | 0.0030 |
| 3/3 | 0.0001 |

初始 batch 0 loss = 0.4424，下降极快，提示 200 条 transitions 对 0.5B + LoRA 来说可能已接近过拟合。

## 验证

### Single PEDA episode（`read_note`，max_steps=5）

```bash
python scripts/phase2_collect_data.py \
  --baseline peda \
  --task read_note \
  --max-steps 5 \
  --adapter-path checkpoints/phase2/sandbox_adapter_e1 \
  --output results/phase2_verify_e1.jsonl
```

**结果**：
- 第一步 action：`ls`（不再是 `ls data`）
- 后续动作：`ls data` → `ls` → `ls data` → `ls`
- FHT：None
- SCR：0.2
- Dead-loop rate：0.0

### 结论

- **P0 最低成功标准达成**：第一步不再是 `ls data` 死循环。
- **行为仍不理想**：PEDA 在 5 步内未完成任务，且继续在同一小集合命令中打转。
- 可能原因：
  1. `--fast` 模式跳过了 ensemble checkpoints，缺乏 epistemic 信号。
  2. 200 条数据量偏小，模型记住了局部模式但未学到任务级规划。
  3. Pragmatic-only 的 EFE 无法区分探索与任务推进。

## 交付物

- `checkpoints/phase2/sandbox_adapter_e1/` — 训练后的 LoRA adapter
- `checkpoints/phase2/sandbox_adapter_e1/training_examples.json` — 样本训练数据
- `checkpoints/phase2/sandbox_adapter_e1/checkpoint_epoch_1/2/3/` — 中间 checkpoints
- `results/phase2_train.log` — 训练日志
- `results/phase2_verify_e1.jsonl` — 验证结果

## 下一步建议

1. **跑非 fast 模式验证**：加载 ensemble checkpoints，观察 epistemic 信号是否能让 PEDA 逃离局部循环。
2. **增加训练数据**：将 random/heuristic 各扩展到 10+ seeds，或加入 prompt-driven 轨迹，达到 500+ transitions。
3. **任务奖励调试**：确认 pragmatic 奖励是否正确反映 `read_note` 等任务的完成状态。
4. **报告至上游模型**：将本报告作为 Phase 2 P0 完成证据，同时指出行为仍不足以进入 P1 多基线对比。

## 参考

- `PEDA_FINAL/CONTROLLER_DIRECTIVE_PHASE2.md`
- `PEDA_WORKING_LOG.md` 中 `[EVAL] 2026-07-18 22:15:13`
- `scripts/phase2_synthetic_train.py`
- `scripts/phase2_collect_data.py`
