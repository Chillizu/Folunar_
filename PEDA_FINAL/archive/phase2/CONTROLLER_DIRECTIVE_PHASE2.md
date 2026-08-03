# CONTROLLER_DIRECTIVE_PHASE2.md

## 当前状态（2026-07-18）

Phase 2 基础设施已跑通，但**World Model 尚未在沙箱数据上训练**。当前运行的基线对比使用旧 adapter `checkpoints/phase1_5/text_adapter_e4`（仅在 TextRoomEnv 上训练），因此 PEDA 陷入 `ls data` 死循环，所有基线 SCR 均为 0.2。

## 禁令

在以下任一条件满足前，**禁止**执行以下操作：

- 禁止运行 `--all-baselines` 多基线对比
- 禁止调整 confidence penalty / drive weights / pragmatic_weight 等超参数
- 禁止新增模块、文档或计划文件
- 禁止进入 Phase 2.5 或 Phase 3

## 唯一 P0 任务

**在沙箱数据上训练一个 LoRA adapter。**

### 步骤

1. **收集训练数据**（1-2 小时，CPU）
   ```bash
   python scripts/phase2_collect_data.py \
     --baseline random --all-tasks --max-steps 20 \
     --output results/phase2_random_train.jsonl

   python scripts/phase2_collect_data.py \
     --baseline heuristic --all-tasks --max-steps 20 \
     --output results/phase2_heuristic_train.jsonl
   ```

2. **合并并训练**（参考 `phase1_5_synthetic_train.py`）
   ```bash
   python scripts/phase2_synthetic_train.py \
     --data results/phase2_train_merged.jsonl \
     --output-dir checkpoints/phase2/sandbox_adapter_e1
   ```

3. **验证训练效果**
   ```bash
   python scripts/phase2_collect_data.py \
     --baseline peda --task read_note --max-steps 20 \
     --adapter-path checkpoints/phase2/sandbox_adapter_e1
   ```

## 成功标准

- [ ] `checkpoints/phase2/sandbox_adapter_e1/` 存在至少 1 个 checkpoint
- [ ] 在该 adapter 上跑 single PEDA episode，第一步不再是 `ls data` 死循环
- [ ] 输出 `PEDA_FINAL/phase2_adapter_train_report.md`，包含：数据来源、训练损失曲线、验证损失、第一步 action

## 失败预案

如果 4 小时内无法完成 P0：

1. 立即停止当前工作
2. 输出 `PEDA_FINAL/phase2_blocker_report.md`，说明阻塞原因
3. 等待上游决策：是否转向轻量 JEPA，或等待 AWS GPU 配额

## 3 Questions 回答模板

每次向控制器汇报前，必须回答：

1. **本次实验能证伪哪个具体假设？**
   例："训练后的 sandbox WM 是否能产生非零 epistemic 信号，使 PEDA 第一步不再 stuck in ls data。"

2. **如果失败，最可能原因是什么？能否在 2 小时内确认？**
   例："最可能是数据不足或 prompt 格式不匹配。可通过检查生成样本和验证损失在 30 分钟内确认。"

3. **这是第几次尝试同一方法？**
   例："第一次尝试 sandbox 训练。"

---

**生效日期**：2026-07-18  
**发布者**：Reviewer / Project Controller  
**依据**：WATCHDOG B1, C2, C12, C14
