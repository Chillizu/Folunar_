# CRITICAL: 写作前必读

## 领域理解
你必须理解以下 PEDA 核心概念，才能写这篇文档。在动笔前，请 read 以下文件：
1. `PEDA_FINAL/peda_report_v11.agent.final.md` — 原始架构设计（2055行）。至少读 Phase 2（§4.4）和各模块的接口定义（§3）。
2. `PEDA_FINAL/RESEARCH_CHARTER.md` — 研究宪章。理解什么是"负结果也是成功"。
3. `PEDA_WORKING_LOG.md` — 完整工作日志。重点读 Phase 2 的 GPU 训练和评估部分。
4. `PEDA_FINAL/archive/phase1/phase1_gap_report.md` — Phase 1 的 gap 分析。
5. `src/phase1/run.py:12-81` — run_episode 函数（完整 PEDA 闭环，Phase 2 缺失的部分）。
6. `scripts/phase2_collect_data.py:_run_agent()` — Phase 2 当前的退化版 agent loop。
7. `src/phase2/sandbox_env.py` — 沙箱环境和候选生成器。

## 核心概念速查
- **PEDA**: Predictive-Error-Driven Autonomous Agent
- **FEP**: Free Energy Principle — 自组织系统最小化自由能 = 最小化预测误差 + 最小化复杂度
- **EFE**: Expected Free Energy = epistemic value（信息增益）+ pragmatic value（偏好匹配）
- **Epistemic uncertainty**: 模型不知道的 → 可以减少 → 值得探索
- **Aleatoric uncertainty**: 环境固有随机性 → 不可减少 → 不该浪费资源
- **WM**: World Model — 0.5B Qwen + LoRA adapter，预测 (s,a) → s'
- **L1/L2/L3**: 三级预测 — exit_code / filesystem delta / output summary
- **Drive System**: 四维 homeostatic — curiosity, competence, boredom, novelty
- **LearningModule**: 经验回放 + buffer≥500 → 自动 LoRA 微调

## 写作原则
- 所有指标必须来自实际实验数据，不得编造
- 引用代码时使用精确的文件路径和行号范围（用 read 核实）
- 不确定的地方标注 [INFERENCE] 或 [UNTESTED]
- 表格驱动，减少散文
- 面向 Agent 读者——不需要解释什么是 Python 或 Docker
# PEDA v2.0 工程计划书 — 写作合约

## 目标
写一份完整的、面向 Agent 的工程计划书。所有 AI 读完后应清楚知道：当前在哪、为何在这、接下来做什么、怎么做、为什么这么做。

## 文件
`PEDA_FINAL/PEDA_ENGINEERING_PLAN_v2.md`

## 必须包含的章节

### 1. 执行摘要（~300 字）
- PEDA 一句话：用预测误差替代 Prompt 作为 Agent 驱动信号
- 当前阶段：Phase 2 量化目标达标，核心假设待验证
- 下一目标：闭合自训练循环，验证 epistemic 驱动探索

### 2. 核心架构（~500 字 + mermaid 图）
- 七个模块：Perception, WorldModel, ErrorComputer, ActionGenerator, ActionExecutor, LearningModule, DriveSystem
- 数据流：State → WM.predict → ErrorComputer.decompose → ActionGenerator.EFE → ActionExecutor.step → Environment
- 三级预测：L1 exit_code (90%), L2 filesystem (70%), L3 output (50%)
- 四原则：No Prompt, Drive Emergent, WM Core, Intermittent Learning

### 3. 阶段回顾（~800 字, 表格驱动）

Phase 1 (Grid World):
- 环境：5×5 grid, 4 actions, 障碍物可选
- 结论：基础设施验证通过，WM 在训练分布上完美（G1=1.000），但无法产生 epistemic 误差——环境太简单
- 核心假设：未验证

Phase 1.5 (TextWorld):
- 结论：decompose_error 修复，epistemic 信号 0→0.20
- partial training 实验显示正向信号（1-episode pilot）

Phase 2 (Sandbox):
- Phase 2a: 10,040 transitions [OK]
- Phase 2b: L1=1.000, L2=0.900, L3=0.550 [OK]
- PEDA 多任务: 20/20 [OK]
- **关键发现 1**: 数据质量 > 数量。e2 (200 curated) > e3 (10,040 noisy)
- **关键发现 2**: WM 是模式匹配器不是推理器。OOD L3=0.400，换文件名就崩
- **关键发现 3**: 沙箱状态空间太小。v1: 22 unique (s,a), v2: 65 unique
- **关键发现 4**: Phase 2 从未闭合 PEDA 自训练循环——LearningModule 未导入
- **核心假设**: [OPEN] epistemic 驱动探索未验证

### 4. 当前架构缺陷（~400 字）
列出 Phase 2 与 Phase 1 的差距：

| Phase 1 有的 | Phase 2 缺失 |
|-------------|-------------|
| LearningModule.store_experience() | 无 |
| LearningModule.update() (buffer≥500 → auto-lora) | 无 |
| ErrorComputer.decompose_error() | 无 |
| DriveSystem.update() | 无 |
| SaturationDetector | 无 |
| 真正的 EFE 选择（epistemic + pragmatic） | 只有 pragmatic（goal_predicate） |

具体指出 `scripts/phase2_collect_data.py:_run_agent()` 只做两步（predict + execute），砍掉了 error/learning/drive。

### 5. Phase 2.5: 核心假设验证（~600 字）
目标：在沙箱中回答 "预测误差能否驱动有效探索？"

实验设计：
- partial training：只在 /sandbox/docs + /sandbox/data 上训练 adapter
- 测试：在 /sandbox/logs + /sandbox/projects 跑任务
- 对照：PEDA (epistemic+pragmatic) vs Pragmatic-only
- 指标：FHT, SCR, behavioral entropy, coverage
- 每个 task × agent × 10 episodes

数据流水线：
```
v2_all.jsonl (65 transitions)
  → split by cwd (Slice 1: DataSplitter)
    → known_train.jsonl (40) + unknown_test.jsonl (25)
      → partial_adapter (Slice 2: PartialTrainer)
        → epistemic vs pragmatic test (Slice 3)
          → report (Slice 4)
```

当前状态：Slice 1-2 完成，Slice 3 待执行。

### 6. Phase 3: 自训练闭环（~800 字）
目标：把 Phase 1 的完整 PEDA 循环移植到 Phase 2 沙箱。

需要改造的函数：`scripts/phase2_collect_data.py:_run_agent()`

改造内容（+5 个调用）：
```python
# 每步新增：
predicted = wm.predict(state, action)
error = ec.decompose_error(state, action, next_state)
ds.update(error, action, ...)
lm.store_experience(Experience(state, action, next_state, error, exit_code, summary))
if lm.should_update():
    lm.update()
    boost = lm.saturation_novelty_boost
    ds.current_terms.novelty += boost
```

需要适配的类型：
- Experience 当前是 GridState/Action enum → 扩展为 SandboxState/string action
- lora_finetune(sandbox_mode=True) — 已支持，无需改

预期行为：
- 初始 adapter 只在 docs/data 准确 → 探索 logs/projects 时 epistemic 高
- EFE 选择 logs/projects 探索 → 采集新 transitions → buffer 满
- 自动微调 → 新的 transition 被学会 → epistemic 降低
- 循环继续，直到 WM 覆盖全部环境

### 7. Phase 4: 从知识到应用（~600 字）
目标：Agent 不只是"了解"环境，而是"改变"环境。

三个阶段：
1. **拥有知识**：WM 学会预测所有 (s,a,s') [当前状态]
2. **应用知识**：Agent 用 preference distribution p(o|C) 选择有 pragmatic value 的行动 [需要设计]
3. **创造知识**：Agent 自发生成新目标 [理论目标]

偏好分布设计：
| p(o|C) 定义 | 催生行为 |
|------------|---------|
| 偏好观察到的输出与之前不同 | 探索新命令/新参数 |
| 偏好文件列表变化 | 倾向 touch/mkdir/echo > |
| 偏好输出长度 > 0 | 倾向 cat/grep 而非 cd/ls |
| 偏好高信息密度输出 | 倾向 grep -r 而非单文件 |
| 偏好预测误差大的行动（与 epistemic 协同） | 倾向探索未知 |

### 8. 沙箱演进路线（~500 字）

| 版本 | 内容 | 状态 | 唯一 (s,a) 对 |
|------|------|------|-------------|
| v1 | 3 dir, 3 files, busybox | [DONE] | 22 |
| v2 | 7 dir, 14 files, busybox+find | [DONE] | 65 |
| v3 | 写权限, mkdir/touch/echo > | [NEXT] | ~150 |
| v4 | Python 解释器 | [PLANNED] | ~500 |
| v5 | curl/wget (白名单 HTTP) | [PLANNED] | ~1000 |

v3 的关键变化：`--read-only` 移除，沙箱变为可写。这使 create_file 任务可用，也使"创造"成为可能。

v4 的爆炸点：Python 解释器 → WM 从"预测文件内容"变为"预测代码执行结果" → epistemic error 永不归零 → 探索永远停不下来。

### 9. 数据质量方法论（~400 字）

五条策略及其验证结果：

| 策略 | 新增 unique | 有效？ | 适用场景 |
|------|-----------|--------|---------|
| PEDA 自产轨迹 | ~4/task | 少（确定性行为） | 多步探索场景 |
| Expert Demo | ~15 | 是（多 cwd 变体） | 任务完成路径 |
| 系统枚举 | 全覆盖 | 是（完整覆盖） | 小环境完整覆盖 |
| 难点定向采样 | 未执行 | — | 已知误差修正 |
| 课程学习 | 未执行 | — | 渐进复杂度 |

核心原则：**在确定性且状态空间小的环境中，系统枚举是最优策略。随机采样在信号稀疏时退化为噪声。**

### 10. 创造性机制（~500 字）

PEDA 的"创造性"不是创造文件或代码——是**创造关于环境的知识**。

创造性的四个层级：
1. **探索新状态**：curiosity drive → 访问未去过的地方
2. **发现新关系**：WM 发现 "所有 .log 文件都有时间戳"
3. **生成新目标**：发现关系后自发定义子问题 ("找到所有 ERROR")
4. **自我改进**：发现自己的 prediction 不准 → 主动触发训练

当前：层级 1 已实现（通过 candidate generator），层级 2-4 需要：
- 层级 2: 更大的训练数据 + 更强的模型（7B）
- 层级 3: preference distribution + preference learning
- 层级 4: "训练自己" 成为候选 action + WM 预测训练效果

**爆炸触发**：环境中出现 Python/curl → 行动空间爆炸 → WM 永远无法完全预测 → 探索永不停止 → 创造性行为自然涌现。

### 11. Agent 能否触碰自己？（~400 字）

理论上是 PEDA 的天花板能力。

逻辑链：
```
1. Agent 观察自己的 epistemic error 持续高
2. 它将 "error 状态" 编码为 State 的一部分
3. 候选 action 中包含 "train_on_recent_data"
4. WM 预测: "执行 train_on_recent_data → 未来 epistemic 降低"
5. EFE 计算: 这个 action 的 epistemic value = 预期信息增益 → 高价值
6. 选中执行 → Agent 主动训练自己
```

这不是 Agent "想变聪明"——是 EFE 计算显示训练自己的预期信息增益最高。

当前差距：
- error vector 计算了但从未暴露给候选 action
- "训练" 不在候选空间
- 需要二阶 WM（预测 WM 的行为）

### 12. 评估框架（~400 字）

| 指标 | 定义 | 当前基准 |
|------|------|---------|
| L1 | exit_code 预测准确率 | 1.000 (e2) |
| L2 | 文件系统变化 Jaccard | 0.900 (e2) |
| L3 | 输出摘要匹配 | 0.550 (e2) |
| FHT | 首次命中目标步数 | 0 (e2, 4 tasks) |
| SCR | 状态覆盖率 | 1.0 (e2, 4 tasks) |
| Epistemic Ratio | epistemic / (epistemic + aleatoric) | [NOT MEASURED] |
| Behavioral Entropy | 动作序列香农熵 | [NOT MEASURED] |
| Knowledge Growth | unique (s,a) 对随时间增长 | [NOT MEASURED] |

epistemic 对照实验的判据：
- PEDA 在未知区域 FHT < Pragmatic → 假设支持
- 两者无差异 → 负结果（按研究宪章有效）
- 统计量：≥10 episodes per condition

### 13. 基础设施需求（~300 字）

| 需求 | 当前 | 建议 |
|------|------|------|
| 模型 | Qwen2.5-0.5B | Phase 4 考虑 1.5-7B |
| GPU | AWS g4dn.xlarge ($0.53/h) | 按需启动 |
| 沙箱基础镜像 | peda-sandbox:v2 | v3: 加写权限 |
| 内存 | 16GB | 0.5B 推理够用 |
| 存储 | S3 chillizu-peda-checkpoints | 已有 |

### 14. WATCHDOG 规则更新（~300 字）

新增规则：
- **C20**: 核心假设验证优先于功能扩展。epistemic 验证通过前，不新增任务/模块/环境。
- **C21**: 自训练循环必须闭合后才算 PEDA 实际运行。当前 Phase 2 只是"评估模式"，不算运行。
- **C22**: 环境复杂度必须与 WM 容量匹配。0.5B 模型在 65 unique (s,a) 的环境已接近饱和——更大环境需更大模型。

保留规则再声明：B1, B3, B5, C12, C14, C16-C19

### 15. 编排规则（~200 字）
- **主模型角色**：只做编排——分析需求、设计合约、拆解任务、审核结果。不亲自写代码或跑长时间实验。
- **Subagent 角色**：积极调用。用户同意后，每个独立工作任务分发一个 subagent。
- **合约格式**：每个任务写 `local://` 文件，包含 Target、Change、Acceptance 三部分。
- **编排流程**：范围分析 → 合约设计 → 用户确认 → 分发 → 等待 → 审核 → 合并报告。

### 16. 风险登记（~300 字）

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| epistemic 对照实验无差异 | 中 | 核心假设证伪 | 记录负结果，按研究宪章处理 |
| v2 推理太慢 (251s/步) | 高 | 实验无法规模化 | 候选集降到 6，缩短 horizon 到 1 |
| 0.5B 模型无法泛化 | 高 | 创新行为无法涌现 | 升级到 1.5-7B |
| 自训练闭环引入不稳定 | 中 | 评估失控 | 每轮微调后保存 checkpoint + 回滚能力 |
| credits 不足 | 中 | subagent 无法使用 | 降级为自行执行 |
