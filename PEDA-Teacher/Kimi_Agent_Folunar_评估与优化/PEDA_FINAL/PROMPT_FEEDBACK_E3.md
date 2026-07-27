# 反馈：Phase 1 部分训练验证 — Agent 六点回应 + 下一步

## 你的六点 — 逐一回应

### 1. 3 epochs 重训 adapter → ✅ 同意，唯一优先级

代码验证确认：
- `lora_finetune(epochs=N)` 支持多 epoch ✅
- `checkpoint_dir is not None` 时保存 `checkpoint_epoch_{N}` ✅
- `EnsembleErrorComputer` 在 `len(checkpoints)>1` 时计算 ensemble_variance ✅
- `len(checkpoints)<=1` 时 `ensemble_variance=0.0`（你观察到的 epistemic=0 的根因）✅

**唯一需要注意**：确保 `phase1_synthetic_train.py` 调用 `lora_finetune` 时传递了 `checkpoint_dir` 参数。如果当前脚本没有传递，这是唯一需要修改的代码（5 分钟）。

### 2. Drive system 调制效果意外不错 → ⚠️ 方向正确，但不要过度解读

你的数据是诚实的：
- goal_known: 90% vs 70% (+20%)
- goal_unknown: 70% vs 60% (+10%)

但统计检验（Fisher exact test, n=10）：
- goal_known: p ≈ 0.28（不显著）
- goal_unknown: p ≈ 1.0（不显著）
- 合并: p ≈ 0.47（不显著）

**你的说法"意外不错"是准确的**（没有用"显著"或"验证"），这展现了良好的科学诚实。但提醒：在 epistemic 上线后的 3-epoch 实验中，Drive system 的独立贡献可能会被 epistemic 信号掩盖或放大——届时需要分离两者的效应。

### 3. 0.5B 对 5x5 太强 → ✅ 同意，长期看需要移步 TextWorld

g1_test_set=0.8684 是强有力的证据。0.5B 模型在只训练 25% 的情况下，OOD 泛化准确率仍达 87%。这说明 Grid World 对 0.5B 来说太简单了——不是模型的问题，是环境的问题。

**短期**：降到 15-20% train_fraction 可以在 Grid World 中制造更多不确定性
**中期**：3-epoch 实验后如果 epistemic 仍然很弱，直接移步 TextWorld
**长期**：TextWorld 或 busybox sandbox 是验证 PEDA 核心假设的更有信息量的环境

### 4. 分块方法可靠 → ⚠️ 需要确认 seed 实现

你说"per-condition 独立 RNG seed 方案是正确的"——需要确认 `phase1_partial_eval.py` 中是否确实对每个 condition（goal_known/goal_unknown）使用了不同的 seed。

如果所有 condition 使用同一 `--eval-seed 42`：结果仍然可比（两个 agent 在相同 seed 下公平对比），但不是"独立 RNG seed"。

**不需要现在修改**，但请在下次 commit 的 message 中说明 seed 策略。

### 5. known_cells list-of-lists bug → ⚠️ 可信但无法完全验证

fec0676 的 diff 中 `drive_system.py` 有修改（+11, -5），但无法从 diff 确认是 `_sample_goal` 的修复。

**信任你的说法**（你说"merge 后的报告使用的是正确数据"），但请在下次 commit 中附上 `_sample_goal` 修复前后的代码 diff，以便外部验证。

### 6. WATCHDOG.md 和 config/phase1_model.json → ⚠️ 需要确认

fec0676 的 diff 中没有这两个文件。它们可能来自更早的 commit。

**请检查**：
- `config/phase1_model.json` 中的 drive weights 是否与当前 grid search 最优值一致
- `WATCHDOG.md` 是否需要同步（我已在本轮对话中更新了 WATCHDOG，新增了 5 条规则）
- 如果不一致，请同步

---

## 下一步（不需要讨论，直接执行）

### P0：重训 3 epochs adapter

```bash
# 0. 确认 drive weights 已更新
# config/phase1_default_drives.json 已更新为 grid search top-1：
#   competence: 0.5 → 2.0
#   novelty: 0.1 → 2.0
#   curiosity: 0.1 (不变), boredom: 0.1 (不变)
# 这是重大变化——e3 实验将使用新权重，与 e1 结果不完全可比
# 但 e1 旧权重下差异不显著，新权重可能产生更强信号

# 1. 确认 phase1_synthetic_train.py 传递了 checkpoint_dir
# 如果未传递，修改调用：
# world_model.lora_finetune(..., checkpoint_dir=Path("checkpoints/phase1"))

# 2. 重训
python scripts/phase1_synthetic_train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-fraction 0.25 \
  --epochs 3 \
  --output-dir checkpoints/phase1/partial_adapter_real_25_e3

# 3. 验证 checkpoint 数量
ls checkpoints/phase1/partial_adapter_real_25_e3/checkpoint_epoch_*
# 预期：checkpoint_epoch_1, checkpoint_epoch_2, checkpoint_epoch_3
```

### P1：重跑 10 episodes 评估

```bash
python scripts/phase1_partial_eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter checkpoints/phase1/partial_adapter_real_25_e3 \
  --episodes 10 \
  --pragmatic-weight 3.0 \
  --output results/phase1_partial_eval_e3_10eps.json
```

### P2：分析 epistemic 是否上线

**关键指标**：
- `epistemic_error` 是否 > 0（之前是 0）
- PEDA 在 goal_unknown 条件下是否仍然优于 pragmatic_only
- PEDA 优势是否比 e1 更大（epistemic 的增量效应）

**如果 epistemic_error > 0 且 PEDA 优势扩大 → 核心假设在 Grid World 中验证通过**
**如果 epistemic_error > 0 但 PEDA 优势不变 → Drive system 是主要驱动力，epistemic 需要更强信号**
**如果 epistemic_error = 0 → 训练过程有问题，需要调查 checkpoint 是否正确保存**

---

## 不需要做的事

- ❌ 不要写文档/更新 AGENTS.md
- ❌ 不要修复 lint
- ❌ 不要添加新模块
- ❌ 不要进入 Phase 1.5
- ❌ 不要讨论"Drive system 是否有独立价值"（有趣的问题，但等 epistemic 上线后再分析）

---

## 时间估计

- 修改训练脚本（如果需要加 checkpoint_dir）：5 分钟
- 重训 3 epochs：30-60 分钟
- 重跑 10 episodes：约 5 小时（overnight）
- 分析结果：30 分钟
- **总计：约 6 小时，建议 overnight 跑**

---

## 核心原则

> **3 epochs → 3 checkpoints → epistemic_error > 0 → 看 PEDA 是否仍然优于 pragmatic_only**
>
> 这是 Phase 1 的最后一个实验。做完这个，要么核心假设验证通过（进入 Phase 1.5），要么核心假设不成立（分析原因，调整方向）。
>
> 不需要更多分析了。重训，重跑，看结果。
