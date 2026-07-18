# PEDA 工作日志

## 使用规则（必读）

本文档是 PEDA 项目的**唯一追加式工作日志**。所有轮次的工作与评估必须落到纸上，禁止仅靠 IRC 或零散消息传递状态。

### 1. 追加原则

- **只追加，不修改**。每次一轮工作或一轮评估完成后，在文档末尾新增一节，不要编辑旧内容。
- 如果旧内容有误，在新增节中引用并更正，不要删除原文。
- 每节必须带时间戳和角色标签。

### 2. 角色标签

- `[EXEC]`：执行者。负责写代码、跑实验、改配置。
- `[EVAL]`：评估者 / 项目掌控者。负责审查、判断、决策。
- `[META]`：元讨论。流程变更、规则修订、角色切换。

### 3. 每节必填字段

#### 执行者 `[EXEC]` 模板

```markdown
### [EXEC] YYYY-MM-DD HH:MM — 简要标题

**本轮目标**：
（一句话说明这次要做什么）

**实际做了什么**：
- 改了哪些文件
- 跑了什么命令
- 生成了哪些输出

**项目进展**：
- 距离当前 Phase 的 P0 完成度：X%
- 距离当前 Phase 的 go/no-go 还有多远
- 新阻塞点 or 已消除阻塞点

**本轮交付物**：
- 文件路径 1
- 文件路径 2
- 结果/指标

**下一步建议**：
（执行者视角，建议下一手做什么）
```

#### 评估者 `[EVAL]` 模板

```markdown
### [EVAL] YYYY-MM-DD HH:MM — 简要标题

**审查对象**：
（引用本轮 EXEC 节或具体文件）

**我的判断**：
（通过 / 不通过 / 需要补充 / 阻塞）

**思考过程**：
1. 关键观察
2. 风险点
3. 与 WATCHDOG / 研究宪章的对应关系

**具体建议**：
- 建议 A
- 建议 B

**下一步决策**：
（明确 P0 / P1 / P2，以及禁止做的事）
```

### 4. 禁止事项

- 禁止用"进度正常"、"继续推进"等模糊表述代替具体指标。
- 禁止在评估节中只夸不批，或只批不给出路。
- 禁止用 IRC 聊天替代日志追加。
- 禁止在核心假设未验证前写"Phase X 完成"。

### 5. 引用规则

- 提及 WATCHDOG 规则时用 `WATCHDOG B3` 或 `WATCHDOG C12` 格式。
- 提及研究宪章时用 `RESEARCH_CHARTER §X`。
- 提及计划书时用 `PEDA v1.1 第 X 章`。

---

## 初始记录

### [META] 2026-07-18 — 工作日志创建

**说明**：
本文档由项目掌控者创建，用于替代零散的消息汇报。后续每一轮工作与评估都必须追加到本文档末尾。

**当前项目状态摘要**：

- **Phase**：Phase 2（busybox Linux 沙箱）
- **阻塞点**：World Model 仍在使用 `checkpoints/phase1_5/text_adapter_e4`（TextRoomEnv 训练），未在沙箱数据上训练。
- **已发生问题**：PEDA 在沙箱中陷入 `ls data` 死循环；所有 5 个基线 SCR 均为 0.2。
- **P0 任务**：在沙箱数据上训练 LoRA adapter。
- **控制器指令**：`PEDA_FINAL/CONTROLLER_DIRECTIVE_PHASE2.md`

**下一步**：
等待上游确认是按 A 路径（沙箱训练）继续，还是转向 B（轻量 JEPA）或 C（等 AWS GPU 配额）。

---

### [EVAL] 2026-07-18 22:15:13 — Phase 2 基线运行评估

**审查对象**：
- `results/phase2_data.log`（最新运行日志）
- `results/phase2_data_sanity.jsonl`（5 基线 sanity 指标）
- `scripts/phase2_collect_data.py`（基线运行脚本）
- `src/phase2/sandbox_env.py`（沙箱环境）
- `src/phase1/drive_system.py`（EFE 计算与 confidence penalty）

**我的判断**：
**不通过 / 阻塞**。当前基线运行无法验证 PEDA 核心假设，且正在消耗无效算力。

**思考过程**：

**观察 1：World Model 未在沙箱数据上训练**
`scripts/phase2_collect_data.py` 默认加载 `checkpoints/phase1_5/text_adapter_e4`，该 adapter 仅在 2 房间 TextRoomEnv 的 114 条数据上训练，从未见过 busybox 沙箱的 JSON 结构化状态。这意味着 EFE 计算中的 epistemic/pragmatic 项都基于无意义的预测分布，PEDA 的 action 选择退化为准随机。

**观察 2：PEDA 陷入 `ls data` 死循环**
`phase2_data.log` 显示 PEDA 从第 1 步起连续选择 `ls data`。该命令不推进任务，但 EFE 持续认为它最优。confidence penalty（avg_conf > 0.95 时加惩罚）未触发，因为未训练 WM 的置信度分布本身就不具备可解释性。这是 WATCHDOG C12 的典型案例：Agent 因 WM 过度自信（或随机自信）陷入自我强化循环。

**观察 3：所有基线等价失败**
`phase2_data_sanity.jsonl` 中 5 个基线的 SCR 全部为 0.2，FHT 全部为 null。这说明当前任务/环境对任何策略都不可解，因为 WM 完全失效。此时跑 `--all-baselines` 不是对比，是同步失败。

**观察 4：违反 WATCHDOG 多项规则**
- **C2 过程指标冒充进展**："基线脚本能跑"不等于"核心假设被验证"。
- **C12 死循环未识别为 fundamental limitation**：不应继续调 confidence penalty 来掩盖 WM 未训练的问题。
- **C14 Drive System 未验证**：在 WM 无效时讨论 Drive System 的独立价值无意义。
- **B1 Phase 推进风险**：若有人据此宣布"Phase 2 基础设施完成、进入核心验证"，将触发 Blocker。
- **C15 用未训练 WM 跑基线**（新增）：详见 WATCHDOG 更新。

**具体建议**：

**建议 A：立即停止所有基线对比**
在沙箱 adapter 训练完成前，不要再跑 `--all-baselines`。单次 smoke test 可以跑，但只能用于验证代码路径，不能用于假设验证。

**建议 B：P0 改为"收集沙箱数据 + 训练 LoRA adapter"**
参考 `CONTROLLER_DIRECTIVE_PHASE2.md` 中的三步：
1. 用 random + heuristic 收集 200-500 条沙箱转移；
2. 参考 `phase1_5_synthetic_train.py` 训练 `checkpoints/phase2/sandbox_adapter_e1`；
3. 用训练后的 adapter 跑 single episode，确认第一步不再是 `ls data`。

**建议 C：不要调整超参数**
在 WM 未训练时调整 confidence penalty、drive weights、pragmatic_weight 属于 WATCHDOG B8 "just one more try" 的变体——不是参数问题，是模型未学习环境问题。

**下一步决策**：

**P0（唯一优先级）**：训练沙箱 LoRA adapter。
- 禁止：运行 `--all-baselines`、调整超参数、新增模块/文档。
- 允许：单次 smoke test、数据收集脚本、训练脚本开发。

**P1（P0 完成后）**：用训练后的 adapter 重新跑 5 基线，观察 SCR/FHT/dead_loop_rate 是否出现分化。

**P2（若 P0/P1 仍无 epistemic 信号）**：按 GLM-5.2 建议实现轻量 JEPA（hidden-state ensemble variance），评估 latent-space epistemic 是否优于 token-space。

**时间约束**：P0 应在 4 小时内完成。若超时，输出 blocker report 并等待上游决策。

---

### [EXEC] 2026-07-18 23:25 — 停止基线对比，转向沙箱数据收集与 LoRA 训练

**本轮目标**：
按 `[EVAL] 2026-07-18 22:15:13` 建议 A/B，停止无意义的多基线对比，收集沙箱转移数据并训练 sandbox-specific LoRA adapter。

**实际做了什么**：
- 停止正在运行的 `phase2_collect_data.py --all-baselines` 100 步基线跑（已产出 `results/phase2_data.jsonl`，但未用于报告）。
- 修改 `scripts/phase2_collect_data.py`：增量写入 JSONL 时包含 `records` 字段，供训练使用。
- 修改 `src/phase1/world_model.py`：
  - `lora_finetune` 增加 `sandbox_mode` 参数与自动检测（`next_cwd`/`next_files` 键）。
  - sandbox 训练时使用 `max_length=384`，避免 JSON 状态被截断。
  - 在 `generate_text` 中为 `model.generate()` 传入 `attention_mask`（减少 transformers 警告）。
- 修复 `scripts/phase2_collect_data.py` 的 prompt 基线：使用 chat template + 白名单回退，避免模型输出指令文本。
- 创建 `scripts/phase2_synthetic_train.py`：从 JSONL records 提取 `(s,a,s')` 并训练 sandbox adapter。
- 收集训练数据：
  - `results/phase2_random_train.jsonl`（5 tasks × 20 steps）
  - `results/phase2_heuristic_train.jsonl`（5 tasks × 20 steps）
  - 合并为 `results/phase2_train_merged.jsonl`（10 runs，200 transitions）
- 启动训练：`checkpoints/phase2/sandbox_adapter_e1`，3 epochs，batch size 4，正在运行中。

**项目进展**：
- P0（沙箱 LoRA adapter 训练）已启动，约 30% 完成（数据收集完成，训练中）。
- 4 小时时间约束：已用约 1 小时；剩余时间充裕。
- 已消除阻塞点：基线对比改为数据收集；训练脚本已跑通 stub smoke test。

**本轮交付物**：
- `scripts/phase2_collect_data.py`（records 输出、prompt 修复、--all-tasks 拆分）
- `scripts/phase2_synthetic_train.py`（新建）
- `src/phase1/world_model.py`（sandbox_mode、attention_mask、max_length）
- `results/phase2_random_train.jsonl`
- `results/phase2_heuristic_train.jsonl`
- `results/phase2_train_merged.jsonl`
- `results/phase2_train.log`（训练中）

**下一步建议**：
等待 `checkpoints/phase2/sandbox_adapter_e1` 训练完成，然后用该 adapter 跑 single PEDA episode（`read_note`），验证第一步是否不再是 `ls data`。

### [EXEC] 2026-07-18 23:53 — 沙箱 adapter 训练完成与初步验证

**本轮目标**：
完成 `checkpoints/phase2/sandbox_adapter_e1` 训练，并跑 single PEDA episode 验证 P0 成功标准。

**实际做了什么**：
- 训练完成：3 epochs，loss 0.4424 → 0.0291 → 0.0030 → 0.0001。
- 保存了 adapter 与 3 个中间 checkpoints：`checkpoint_epoch_1/2/3/`。
- 运行验证：`--baseline peda --task read_note --max-steps 5 --adapter-path checkpoints/phase2/sandbox_adapter_e1`。
- 编写报告：`PEDA_FINAL/phase2_adapter_train_report.md`。

**项目进展**：
- P0 最低成功标准达成：第一步 action 为 `ls`，不再是 `ls data`。
- 但 5 步内未完成任务（FHT=None），PEDA 仍在 `ls`/`ls data` 之间打转。
- P0 完成度：100%（adapter 存在且通过最低验证）。
- 进入 P1 的门槛：需要非 `--fast` 模式下验证，或更多训练数据，确认 epistemic 信号是否有效。

**本轮交付物**：
- `checkpoints/phase2/sandbox_adapter_e1/`
- `results/phase2_verify_e1.jsonl`
- `results/phase2_verify_e1.log`
- `PEDA_FINAL/phase2_adapter_train_report.md`

**下一步建议**：
1. 在非 `--fast` 模式下跑 single PEDA episode，验证 ensemble checkpoints 是否能产生有效 epistemic 信号。
2. 若仍无改善，将训练数据扩至 500+ transitions（多 seeds / 加入 prompt 轨迹）。
3. 根据结果决定是否进入 P1 多基线对比，或输出 blocker report 等待上游决策。

### [META] 2026-07-18 23:58 — 上游模型报告已输出

**说明**：
本轮 P0 的完整结果已整理为 `PEDA_FINAL/phase2_adapter_train_report.md`，可直接作为反馈报告提交给上游模型。报告包含数据来源、训练损失曲线、验证结果、交付物清单及下一步建议。

**上游反馈要点**：
- P0 最低标准已达成（第一步不再是 `ls data`）。
- 行为仍未达到进入 P1 多基线对比的要求。
- 需要上游决策：优先跑非 `--fast` ensemble 验证，还是优先扩数据到 500+ transitions，或等待 AWS GPU 配额。

**相关文件**：
- `PEDA_FINAL/phase2_adapter_train_report.md`
- `PEDA_FINAL/CONTROLLER_DIRECTIVE_PHASE2.md`
- `PEDA_WORKING_LOG.md` 中 `[EXEC] 2026-07-18 23:53`

### [EXEC] 2026-07-19 00:15 — 非 fast ensemble 验证完成

**本轮目标**：
按 `[EVAL] 2026-07-18 23:59` 的立即执行项，跑非 `--fast` ensemble 验证，确认 epistemic 信号是否能让 PEDA 逃离 `ls/ls data` 振荡。

**实际做了什么**：
- 修复 `scripts/phase2_collect_data.py`：`_build_ag` 的 ensemble checkpoint 路径从硬编码 `checkpoints/phase1_5/text_adapter_e4` 改为使用 `--adapter-path` 参数。
- 运行命令：`python scripts/phase2_collect_data.py --baseline peda --task read_note --max-steps 10 --adapter-path checkpoints/phase2/sandbox_adapter_e1`。
- 3 个 epoch checkpoints（checkpoint_epoch_1/2/3）已组成 ensemble 并参与 EFE 计算。

**验证结果**：
- 动作序列：`ls → ls data → ls → ls data → ls → ls data → ls → ls data → ls → ls data`
- FHT=None，SCR=0.1，Dead-loop rate=0.0
- 每步 select≈10.5s，10 步共 106s
- **epistemic 信号未能使 PEDA 逃离局部振荡**。

**项目进展**：
- P0 保持关闭，adapter 已保存。
- P1 仍然阻塞：ensemble 验证失败说明问题不在数据量（至少不是 200 vs 500 的量级差异），而在 EFE 奖励设计或任务奖励信号。

**本轮交付物**：
- `scripts/phase2_collect_data.py`（ensemble checkpoint 路径修复）
- `results/phase2_verify_e1_ensemble.jsonl`
- `results/phase2_verify_e1_ensemble.log`

**下一步建议**：
1. 调试 EFE / pragmatic 奖励：确认 `read_note` 任务完成时 pragmatic reward 是否被正确传递。
2. 检查 `ActionGenerator` 的 horizon=1 是否导致无法规划多步任务。
3. 若 pragmatic 信号确实太弱，尝试提高 `pragmatic_weight` 或改用更明确的任务奖励函数。
4. 在修复奖励信号后再扩展数据重新训练。
---

### [EVAL] 2026-07-18 23:59 — P0 完成度评估与 P1 门槛判断

**审查对象**：
- `PEDA_FINAL/phase2_adapter_train_report.md`
- `results/phase2_verify_e1.log`
- `results/phase2_verify_e1.jsonl`
- `checkpoints/phase2/sandbox_adapter_e1/`
- `[EXEC] 2026-07-18 23:53` 与 `[EXEC] 2026-07-18 23:25`

**我的判断**：
**P0 最低标准通过，但不建议进入 P1**。adapter 已存在且第一步从 `ls data` 变为 `ls`，但 PEDA 仍未展现出任务级规划能力。

**思考过程**：

**观察 1：P0 最低标准已达成**
`results/phase2_verify_e1.log` 显示第一步 action 为 `ls`，不再是 `ls data`。这证明在沙箱数据上训练 adapter 是有效的：模型至少学会了第一步不要重复进入 data 目录。`checkpoints/phase2/sandbox_adapter_e1/` 已保存完整 adapter 与 3 个中间 checkpoint，交付物齐全。

**观察 2：5 步内任务仍未完成，且出现新循环模式**
PEDA 在 5 步内的动作序列是 `ls → ls data → ls → ls data → ls`。这不是死循环（dead_loop_rate=0.0），但属于同一小命令集合内的无效振荡。FHT=None，SCR=0.2，说明 agent 仍未找到 `cat docs/note.txt` 或类似任务推进动作。这提示 pragmatic 奖励或 EFE 中的任务完成信号仍然太弱。

**观察 3：训练损失极低，存在过拟合风险**
loss 从 0.4424 降到 0.0001，200 条 transitions 对 0.5B + LoRA 来说明显不足。模型可能记住了训练轨迹中的局部模式（如"当前目录有 data 就 ls data"），而不是学到通用状态转移。报告本身也指出了这一点。

**观察 4：验证使用了 `--fast` 模式，未检验 epistemic 信号**
`phase2_collect_data.py` 默认 `--fast` 会跳过 ensemble checkpoints。因此本轮验证无法回答"epistemic uncertainty 是否能让 PEDA 逃离局部循环"这个关键问题。控制器指令中的 P0 成功标准只要求"第一步不是 `ls data`"。但要进入 P1，必须确认 epistemic 信号有效。

**观察 5：推理速度过慢**
每步 select 耗时约 10.8 秒，5 步共 54 秒。即使进入 P1，100 步 × 多基线 × 多任务的实验也会非常昂贵。这是后续 scaling 的隐形成本，但当前不阻塞 P1 决策。

**具体建议**：

**建议 A：先跑非 `--fast` ensemble 验证**
在扩大数据之前，先用 3 个 epoch checkpoints 组成 ensemble，跑 single PEDA episode。如果 epistemic 信号能让 agent 逃离 `ls/ls data` 振荡并尝试 `cat` 类动作，则 P1 有希望。如果仍然失败，说明问题不在数据量，而在 EFE 奖励设计或任务奖励信号。

**建议 B：扩展数据到 500+ transitions 作为备选**
如果 ensemble 验证无效，再扩展 random/heuristic/prompt 轨迹到 500+ transitions。当前 200 条确实太少。

**建议 C：不要进入 P1 多基线对比**
在 single PEDA episode 能稳定推进任务之前，跑 `--all-baselines` 仍然是 WATCHDOG C15 的变体：用未准备好的 WM 做假设验证。

**下一步决策**：

**P0 状态**：关闭。最低成功标准达成，adapter 已保存。

**P1 状态**：阻塞。必须满足以下任一条件才能进入：
- 非 `--fast` ensemble 验证下，PEDA 在 10 步内成功完成 `read_note` 至少一次；或
- 数据扩展到 500+ transitions 并重新训练后，single episode 行为明显改善。

**立即执行**：跑非 `--fast` ensemble 验证，max_steps=10，task=read_note。


