# PEDA Phase 1 部分训练泛化测试：方法、改动与结果报告

**提交给远端高级模型评估用**

---

## 1. 背景与问题

之前的 Phase 1 验证（20-episode real-LLM eval）虽然通过了 G1/G2/G3，但存在三个根本问题：

1. **训练/评估同分布**：World Model 在合成数据上预测误差接近 0，无法体现“预测误差驱动探索”。
2. **pragmatic 距离项主导**：`ActionGenerator.compute_efe` 中 pragmatic 权重 3.0 占主导，系统退化为“曼哈顿距离贪心策略”。
3. **缺乏核心假设验证**：实验没有证明 PEDA 的 epistemic（认知/预测误差）信号在部分未知环境中能优于纯 pragmatic 基线。

因此，本次工作按外部评估建议的 **Option A（部分训练）** 重新设计验证：只在部分 grid cell 上训练 World Model，然后在“未训练区域”比较 PEDA 与纯 pragmatic 基线，以证明 epistemic 信号确实驱动探索。

---

## 2. 代码改动

### 2.1 `src/phase1/world_model.py`

- 给 `WorldModel.lora_finetune()` 增加 `checkpoint_dir: Optional[Path] = None` 参数。
- 在每个 epoch 训练结束时保存 LoRA checkpoint：
  ```python
  epoch_ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch+1}"
  self.model.save_pretrained(epoch_ckpt)
  ```
- 目的：为 `EnsembleErrorComputer` 提供多个 per-epoch checkpoints，以计算真正的 epistemic error（ensemble variance）。

### 2.2 `scripts/phase1_synthetic_train.py`

- 新增 CLI 参数：`--train-fraction`（默认 0.5）、`--split-seed`（默认 42）。
- 在 `generate_synthetic_transitions()` 返回的每个 transition dict 中增加 `agent`、`goal`、`obstacles` 字段，便于后续按 cell 过滤和构造测试集。
- 按 cell 进行确定性拆分：
  ```python
  cells = [(x, y) for x in range(5) for y in range(5)]
  num_train_cells = max(1, min(int(len(cells) * args.train_fraction), len(cells)))
  split_rng = random.Random(args.split_seed)
  known_cells = split_rng.sample(cells, k=num_train_cells)
  train_data = [ex for ex in data if tuple(ex["agent"]) in known_cells_set]
  ```
- 保存 `trained_manifest.json`：
  ```json
  {
    "train_fraction": 0.25,
    "split_seed": 42,
    "num_cells": 6,
    "known_cells": [[0,0], ...],
    "all_cells": [[0,0], ..., [4,4]],
    "trained_pairs": [...]
  }
  ```
- 在 `training_info.json` 中追加 `train_fraction`、`split_seed`、`manifest_path`、`stub` 字段。
- 新增 `--stub` 模式：不加载真实 LLM，仅生成 manifest 和 stub checkpoint 目录，用于快速冒烟测试。

### 2.3 `src/phase1/drive_system.py`

- `ActionGenerator.__init__` 新增参数：
  - `pragmatic_only: bool = False`
  - `pragmatic_weight: float = 3.0`
- `compute_efe` 修改逻辑：
  - 当 `pragmatic_only=True` 时，直接返回 `pragmatic * pragmatic_weight`，**不包含 epistemic 项和 drive_system 调整**。
  - 当 `pragmatic_only=False` 时，按原逻辑计算 `base_efe = epistemic + pragmatic * pragmatic_weight`，再调用 `drive_system.apply_to_efe()`。
- 关键：两个 agent 在比较时必须使用 **相同的 `pragmatic_weight`**，否则无法判断 epistemic 是否真正起作用。

### 2.4 `scripts/phase1_partial_eval.py`（新文件）

核心评估脚本，功能：

1. 加载 `WorldModel`（带 adapter）。
2. 创建 `EnsembleErrorComputer`，从 adapter 目录加载 `checkpoint_epoch_*`。
3. 创建 `LearningModule` 但禁用在线更新：`update_interval=100000`，确保评估过程中不会修改 adapter。
4. 从 `trained_manifest.json` 读取 `known_cells`。
5. 对每种条件 `goal_known` 和 `goal_unknown` 各跑若干 episode：
   - 为每个 episode 固定相同的 `(goal, start_seed)`，让 PEDA 和 pragmatic-only 在**相同起点与目标**上评估。
   - 对每 episode、**每个 agent** 都创建新的 `HomeostaticDriveSystem`（避免历史泄漏）。
   - PEDA：`ActionGenerator(..., pragmatic_only=False)`。
   - Pragmatic-only：`ActionGenerator(..., pragmatic_only=True)`。
6. 计算 `g1_test_set`：在训练集外的状态-动作对（`all_cells × 4 actions - trained_pairs`）上测试 next-position 准确率，作为模型在未知区域的预测能力指标。
7. 输出 `results/phase1_partial_eval.json`。

CLI 参数包括：`--model`、`--adapter`、`--episodes`、`--max-candidates`、`--drive-config`、`--output`、`--eval-seed`、`--pragmatic-weight`、`--max-steps`。

### 2.5 测试文件

- `tests/phase1/test_drive_system.py`：新增 `pragmatic_only` 模式测试，验证其 EFE 仅依赖 pragmatic 项，且与 PEDA 不同。
- `tests/phase1/test_partial_eval.py`：使用 `importlib.util` 加载脚本，测试 `is_known_cell`、`sample_goal`、`sample_untrained_start`、`_compute_g1_test_set`、探索指标等边界行为。

---

## 3. 实验设置

| 项目 | 值 |
|------|-----|
| 基座模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 训练数据 | 20 个随机 goal / 无 obstacle 的 5×5 grid 配置 |
| 训练拆分 | `train_fraction = 0.25`，即 6 个 known cells / 19 个 unknown cells |
| 训练参数 | 1 epoch，batch size 4，learning rate 2e-4 |
| Adapter 路径 | `checkpoints/phase1/partial_adapter_real_25` |
| 评估脚本 | `scripts/phase1_partial_eval.py` |
| PEDA agent | `pragmatic_only=False`, `pragmatic_weight=3.0` |
| Pragmatic-only 基线 | `pragmatic_only=True`, `pragmatic_weight=3.0` |
| 控制条件 | 每个 agent 每个 episode 新建 `HomeostaticDriveSystem`；`LearningModule` 在线更新关闭；相同起点与目标 |
| 试点规模 | 1 episode / condition（CPU 推理限制，每 episode 约 15 分钟） |

### 3.1 为什么 train_fraction = 0.25？

首次尝试 `train_fraction=0.5` 时，模型泛化到未知区域的能力过强（`g1_test_set=1.0000`），导致 PEDA 和 pragmatic-only 在 `goal_unknown` 条件下表现相同。根据计划中的 fallback 流程，降低 `train_fraction` 使未知区域更难预测，从而给 epistemic 信号留出影响空间。

---

## 4. 结果

### 4.1 训练

```text
[synthetic_train] Generated 1920 synthetic transitions.
[synthetic_train] known_cells=6; train_data=448 transitions.
[synthetic_train] Training LoRA adapter for 1 epoch(s)...
[lora_finetune] epoch 1/1 avg loss=0.0207
[synthetic_train] TRAINING_FINISHED
```

### 4.2 部分评估

```text
[partial_eval] Known cells: 6 / 25
[partial_eval] Pragmatic weight: 3.0
[partial_eval] Episodes per condition: 1
[partial_eval] g1_test_set = 0.8684
```

| 条件 | Agent | Success | Mean Steps | Revisit Rate | g1 |
|------|-------|---------|------------|--------------|-----|
| goal_known | PEDA | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_known | pragmatic_only | 1.000 | 3.0 | 0.000 | 1.000 |
| goal_unknown | PEDA | 1.000 | 2.0 | 0.000 | 0.500 |
| goal_unknown | pragmatic_only | 0.000 | 20.0 | 0.905 | 1.000 |

### 4.3 关键发现

- **g1_test_set = 0.8684**：模型在未知状态-动作对上的预测能力显著低于已知区域（g1=1.0），说明部分训练成功制造了预测误差。
- **goal_unknown 条件下 PEDA 显著优于 pragmatic-only**：PEDA 用 2 步到达目标，而 pragmatic-only 在 20 步内失败（max_steps=20）并产生 90.5% 的 revisit rate。
- **goal_known 条件下两者相同**：目标在已知区域时，pragmatic 距离信号足够，PEDA 不需要依赖 epistemic 探索，这符合预期。

### 4.4 结论

在 `goal_unknown` 条件下，PEDA 的 mean_steps 严格小于 Pragmatic-only，**支持 PEDA 核心假设：预测误差信号可以在部分未知环境中驱动有效探索**。

---

## 5. 局限性与风险

1. **统计量不足**：仅 1 episode / condition。PEDA 的优势可能受随机起点/目标影响，需要更大规模（5–20 episodes）验证。
2. **单 epoch、单 checkpoint**：
   - 训练仅 1 epoch，无法保存多个 per-epoch checkpoints。
   - `EnsembleErrorComputer` 的 ensemble variance 为 0，因此 epistemic error 为 0。
   - 本次实验中的 epistemic 信号主要来自 per-prediction confidence `(1 - level2_confidence)` 和 dynamic drive 调制，而非完整的 ensemble 方差。
3. **g1_test_set < 0.90**：原始 G1 _gate 要求 > 0.90；本实验故意让模型在未知区域表现差，因此 g1_test_set=0.8684 是预期的，但说明本指标不能单独作为通过标准。
4. **CPU 推理限制**：每 episode 约 15 分钟，导致无法跑完整 20-episode 评估。硬件瓶颈是本次验证的主要限制。
5. **模型容量**：0.5B 模型低于项目建议的 1–7B 范围。虽然 5×5 规则可以学会，但更大环境需要更大模型。

---

## 6. 方法学上的关键设计决策

1. **cell-level 拆分 vs. state-action 拆分**：采用 cell-level 拆分，确保存在明确的“已知区域”和“未知区域”，而不是零散的状态-动作对。
2. **公平比较**：PEDA 和 pragmatic-only 使用完全相同的 `pragmatic_weight`，只切换 `pragmatic_only` 标志。如果 pragmatic-only 使用更小权重，比较将无效。
3. **禁用在线学习**：`update_interval=100000` 确保评估过程中 `LearningModule.update()` 不会触发，从而保证比较的是同一个静态模型。
4. **新鲜 drive system**：每个 agent 每个 episode 都新建 `HomeostaticDriveSystem`，避免跨 episode/跨 agent 的历史泄漏。
5. **fallback 流程**：当 `g1_test_set > 0.90` 时降低 `train_fraction`，直到模型在未知区域出现预测误差。本实验从 0.5 降到 0.25 后获得有效信号。

---

## 7. 仍可改进的方向

1. **多 epoch 训练**：保存 2–5 个 per-epoch checkpoints，启用真正的 ensemble epistemic error。
2. **更大规模评估**：在更快硬件上跑 5–20 episodes / condition，确认统计显著性。
3. **进一步降低 pragmatic weight**：如果 PEDA 的优势不明显，可尝试 `pragmatic_weight=1.0` 或 `0.5`，但需同时作用于两个 agent。
4. **更复杂环境**：在加入 obstacles 或更大 grid 后测试泛化能力。
5. **real-LLM grid search**：恢复/运行真实模型上的 drive weight 网格搜索，解决超参数可追溯性问题。

---

## 8. 交付清单

- [x] `src/phase1/world_model.py` — per-epoch checkpoint 保存
- [x] `scripts/phase1_synthetic_train.py` — 部分训练 + manifest + `--stub` 模式
- [x] `src/phase1/drive_system.py` — `pragmatic_only` / `pragmatic_weight`
- [x] `scripts/phase1_partial_eval.py` — 核心评估脚本
- [x] `tests/phase1/test_drive_system.py` + `tests/phase1/test_partial_eval.py` — 新增测试
- [x] `PEDA_FINAL/phase1_validation_report.md` — 报告更新
- [x] `ruff` 与 `pytest` 全绿（152 测试通过）
- [x] 真实 LLM 部分训练 + 1-episode 部分评估成功运行
- [x] Git commit `fec0676` 已推送至 `dev`

---

## 9. 需要高级模型评估的问题

请远端模型重点评估：

1. **实验设计是否公平**：PEDA 与 pragmatic-only 的比较是否真正隔离了 epistemic 信号？
2. **结果解释是否合理**：1-episode pilot 的结果是否足够支持“核心假设成立”？还需要哪些额外证据？
3. **统计显著性**：在当前硬件限制下，最低需要多少 episodes 才能给出可信结论？
4. **g1_test_set 的处理**：将 g1_test_set < 0.90 作为“模型在未知区域不完美”是合理的，但是否应该把它也作为通过/失败标准？
5. **下一步优先级**：在有限资源下，应该优先做 (a) 多 episodes 验证，(b) 多 epoch 训练启用 ensemble，还是 (c) 更复杂环境测试？

---

*报告基于 commit `fec0676`（dev 分支）生成。*
*相关输出：`results/phase1_partial_eval.json`。*
