# Phase 1.5 Iteration 2 报告

## 任务完成状态

| 任务 | 状态 | 实际耗时 |
|------|------|----------|
| B: 修复 decompose_error | ✅ 完成 | ~25 分钟（含代码损坏修复） |
| A: 增加训练数据 | ✅ 完成（但去重后仅 114 条） | ~25 分钟（含重训 16 分钟） |
| 快速验证 | ✅ 完成 | ~12 分钟（eval 685s） |

## decompose_error 修复

### 修改内容（src/phase1/world_model.py）
- TextState 分支增加第 3 维度：**has-key**
- `level2_errors` 计算：`(room 不匹配 ? 1 : 0) + (has_key 不匹配 ? 1 : 0)`
- Pairwise variance 从 (exit, room) 二维改为 (exit, room, has_key) 三维，归一化 `/ 3.0`
- 从 `level2_text` 的 `Inventory:` 行解析各 checkpoint 的 has_key 预测

### 效果

| 指标 | 修复前（e3） | 修复后（e4） |
|------|-------------|-------------|
| PEDA mean_epistemic_error | **0.0000** | **0.2000** |
| Pragmatic mean_epistemic_error | **0.0000** | **0.2222** |

**确认：epistemic 信号确实存在但之前被测量 bug 完全掩盖。**

### 数据增强尝试

- 参数：200 walks × 30 steps = 6000 次尝试
- 去重后：**114 条**（与之前 113 条几乎相同）
- 原因：2 房间环境状态空间太小，state_text + action_name 高度重复
- 数据增强在此环境级别没有实际意义

### 训练结果

| 指标 | 上一轮（e3, 113 条） | 本轮（e4, 114 条） |
|------|---------------------|--------------------|
| Loss | 0.2928 → 0.0545 → 0.0240 | **0.2622 → 0.0577 → 0.0175** |
| 耗时 | 623s | 947s |

Loss 略好于上一轮，但 114 条数据显然不足以学好转移动态（`take key` 仍是 exit=1）。

## 快速验证（1 episode each）

### 行为对比

| Agent | 动作 | Success | Steps |
|-------|------|---------|-------|
| **PEDA** | **take key → inventory → look → inventory×7** | ❌ | 10/10 |
| **Pragmatic** | look×10 | ❌ | 10/10 |

### 关键发现

1. **✅ PEDA ≠ Pragmatic 行为复现**。PEDA 在第 1 步就尝试 `take key`（比 Iteration 1 的 step 3 更快）。Pragmatic 从未尝试。

2. **✅ epistemic_error 修复有效**。从 0.0 升至 0.20-0.22，与语义探针预期的 20-40% 一致。

3. **❌ 核心假设仍无法验证**。即使 epistemic_error=0.20，epistemic bonus 仍不足以驱动"拿钥匙后继续向北走"。

4. **根因仍是数据量不足**。114 条去重样本不足以让 0.5B 模型学会 `take key → go north → unlock` 的完整转移链。

## 推荐后续

基于 2 轮迭代（Grid World + Text World）的累积证据：

| 选项 | 预计工作量 | 成功概率 | 推荐 |
|------|-----------|---------|------|
| **D: 进入 Phase 2**（busybox sandbox） | 2-3 天 | 中 | **推荐** |
| 修复后继续优化文本 env（3-4 房间） | 1-2 周 | 低 | 有效但慢 |
| 换 1.5B+ 模型（CPU 受限） | — | 未知 | 硬件不可行 |

**推荐 D 的原因**：
- decompose_error 确认有 epistemic 信号（0.20），但当前环境太简单，无法产生有意义的探索挑战
- 2 房间文本环境的最优路径只有 3 步，而 0.5B + 114 条数据连 3 步链都学不会
- 真实 Linux 命令环境（busybox）的不确定性是固有的、非人造的

## 资产

| 文件 | 说明 |
|------|------|
| `PEDA_FINAL/phase1_5_complete_report.md` | 完整实验报告（含 Iteration 1） |
| `PEDA_FINAL/phase1_5_experimental_report.md` | Iteration 1 摘要报告 |
| `checkpoints/phase1_5/text_adapter_e4/` | 修复后 3-epoch 训练（114 条） |
| `results/phase1_5_eval_iter2.json` | Iteration 2 验证结果（1 ep each） |
