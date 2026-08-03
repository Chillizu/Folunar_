# CRITICAL: 写作前必读

## 领域理解
你必须掌握以下概念才能写这份手稿。动笔前请 read：
1. `PEDA_FINAL/peda_report_v11.agent.final.md` — 原始架构（§2 理论基础, §3 模块接口, §4 Phase 2 设计）
2. `PEDA_FINAL/RESEARCH_CHARTER.md` — 核心问题、负结果标准
3. `PEDA_FINAL/peda_reflection_v11.md` — v1.0 问题 → v1.1 修正
4. `PEDA_WORKING_LOG.md` — 全部实验记录和数据
5. `PEDA_FINAL/archive/phase1/phase1_gap_report.md` — Phase 1 gap
6. `PEDA_FINAL/archive/phase2/phase2_adapter_train_report.md` — e1/e2/e3 训练详情

## 核心概念速查
- **FEP/Active Inference**: 自由能原理 → EFE 最小化驱动行动
- **EFE 分解**: epistemic value（信息增益）+ pragmatic value（偏好匹配）
- **Epistemic vs Aleatoric**: 模型不知道 vs 环境随机 → ensemble variance 启发式分解
- **三级预测**: L1 exit_code (90%), L2 filesystem delta (70%), L3 output (50%)
- **WM as Pattern Matcher**: 0.5B+LoRA 在少量 transition 上学的是查表而非推理
- **数据质量 > 数量**: e2 (200 curated) > e3 (10,040 noisy) — 已被实验证实
- **自训练闭环**: Experience buffer → auto-lora-finetune → 新 checkpoint — Phase 1 实现，Phase 2 缺失

## 写作原则
- 所有实验数据必须来自 PEDA_WORKING_LOG.md，不得编造
- 理论引用必须精确（Friston 2006/2010/2017, Ha & Schmidhuber 2018 等）
- 讨论段落允许推测，但标注 [HYPOTHESIS] 或 [SPECULATION]
- 风格：正式但不学术八股——Agent 要能读懂
- 表格 > 散文，证据 > 断言
# PEDA 研究手稿 — 写作合约

## 目标
写一份"论文风格"的 PEDA 研究文档，面向 Agent。整体呈现：我们在研究什么问题、理论基础是什么、做了什么实验、发现了什么、目前在哪里、未来走向哪里。

## 文件
`PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md`

## 风格
- 正式但非学术行话——Agent 读了要能清楚理解
- 理论部分引用公式和论文，实现部分引用代码文件
- 实验结果用表格呈现
- 讨论部分允许推测性思考，但标注 [HYPOTHESIS] / [EVIDENCE]

## 必须包含的章节

### 摘要（~200 字）
PEDA 试图回答：当一个 Agent 的 World Model 不完美时，预测误差（prediction error）是否能作为内在驱动信号，引导 Agent 主动探索不确定性区域？基于自由能原理（FEP）和主动推理（Active Inference），我们设计了七模块架构，在三个环境中进行了实验。Phase 2 的量化指标已达标，但核心假设——预测误差驱动探索——仍未验证。本文记录了当前状态、核心发现和未来路线。

### 1. 引言（~800 字）
1.1 动机
- 当前 LLM Agent 依赖外部 prompt → 无法真正自主
- FEP 提供了一个替代框架：减少预测误差 = 减少认知不适 = 内在动机
- PEDA 不是要造 AGI，是要回答一个可证伪的科学问题

1.2 核心问题
- 分解为三个子问题（来自 RESEARCH_CHARTER）
  1. 信号问题：LLM-based WM 能否产生可测量的 epistemic 信号？
  2. 驱动问题：epistemic 信号能否驱动 Action Generator 选择探索性行为？
  3. 效果问题：预测误差驱动探索是否比纯 pragmatic/random 基线更有效？

1.3 本文结构

### 2. 理论基础（~1000 字）

2.1 自由能原理与主动推理
- FEP: 任何自组织系统 = 最小化自由能 = 最小化预测误差 + 最小化复杂度
- Active Inference: 行动选择 = EFE 最小化
- EFE = epistemic value (信息增益) + pragmatic value (偏好匹配)

$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[D_{KL}[q(s|o,\pi) \| q(s|\pi)]]}_{\text{Epistemic Value}} + \underbrace{D_{KL}[q(o|\pi) \| p(o|C)]}_{\text{Pragmatic Value}}$$

2.2 Epistemic vs Aleatoric 不确定性
- Epistemic: 模型不知道（可以减少的）→ 驱动探索
- Aleatoric: 环境固有的随机性（不可减少的）→ 不该浪费资源
- 分解方法：ensemble checkpoints → 预测方差 = epistemic, 均值偏离 = aleatoric
- 坦诚声明：这是启发式方法，非严格数学分解

2.3 预测误差作为内在动机
- 对比外在奖励 vs 内在动机
- Drive System 的四维动态平衡：curiosity, competence, boredom, novelty
- 关键洞察：PEDA 的"目标"不是外部定义的 reward，而是内部偏好分布 p(o|C)

### 3. 架构（~800 字 + mermaid）

3.1 总览
- 七模块闭环
- 数据流

3.2 World Model
- LLM (Qwen2.5-0.5B) + LoRA adapter
- 三级预测：L1 exit_code, L2 filesystem, L3 output
- 沙箱环境的 JSON 结构化状态表示
- 不预测：时间戳、PID、随机数 → 标记为 ALEATORIC

3.3 Predictive Error Computer
- EnsembleErrorComputer: 多 checkpoint 预测方差
- decompose_error(): epistemic ∝ ensemble_variance, aleatoric ∝ mean_deviation - ensemble_variance

3.4 Action Generator
- EFE = epistemic + pragmatic × pragmatic_weight
- Rollout horizon=2, max_candidates=8-12
- 退化策略：推理预算不足时降为贪心单步

3.5 Learning Module
- ExperienceBuffer (FIFO, 1000 capacity)
- should_update(): buffer ≥ 500
- update(): sample_prioritized → lora_finetune(epochs=1)
- saturation_detector: 学习停滞 → novelty boost

3.6 Drive System
- 四维 homeostatic 权重：curiosity, competence, boredom, novelty
- 动态调整：历史误差 → 权重变化

### 4. 实验（~1200 字）

4.1 Phase 1: Grid World
- 环境：5×5 grid, 4 actions
- 训练：448 transitions, 1 epoch, loss 0.0207
- 结果：
  - 同分布：G1=1.000, G2=0.434, G3=0.000 [MET]
  - partial training: g1_test_set=0.868
  - goal_unknown: PEDA 2步 vs Pragmatic 20步失败 (1-episode pilot)
- 结论：基础设施验证通过。环境太简单，无法产生持续 epistemic 信号。

4.2 Phase 2: Sandbox
- 环境：Docker busybox, 白名单命令, read-only
- 数据质量实验：
  - e2 (200 curated): L1=1.000, L2=0.900, L3=0.550 [PASS]
  - e3 (10,040 noisy, GPU trained): L1=0.833, L2=0.333, L3=0.133 [FAIL]
- 核心发现：数据质量 > 数量。Random 数据信号密度 ~2%（exit_code=2 极度稀疏）
- 状态空间分析：v1 沙箱仅 22 个唯一 (s,a) 对
- PEDA 多任务：20/20 一步完成（但机制是 action visibility + goal predicate，不是 epistemic 探索）

4.3 沙箱扩展实验
- v1 → v2: 7 目录, 14 文件, 65 unique (s,a) 对
- 系统枚举 > 随机采样（78 records via systematic vs 27 via random+heuristic）

4.4 WM 泛化分析
- OOD L1=1.000（假象——默认预测 exit_code=0 就能拿分）
- OOD L3=0.400（真实泛化水平——换文件名就崩）
- 结论：WM 是模式匹配器，不是推理器。0.5B + 65 transitions 不足以形成抽象。

### 5. 讨论（~1500 字）

5.1 数据质量 > 数量：一个具体论证
- 展示 e2 vs e3 的对比数据
- 解释：随机数据的信号密度问题
- 推广：在稀疏奖励环境中，curated/curriculum 数据远优于大量 noisy 数据

5.2 WM 作为模式匹配器 vs 推理器
- Grid World 的 partial training 证据（G1 从 1.000 掉到 0.868）
- Sandbox 的 OOD 证据（L3 从 0.550 掉到 0.400）
- 这对 PEDA 意味着什么：模式匹配恰好创造了 epistemic error —— 这是需要的

5.3 什么是"创造性"？
- 传统定义：产生新颖有价值的东西
- PEDA 中的定义：创造关于环境的知识
- 四个层级：探索新状态 → 发现新关系 → 生成新目标 → 自我改进
- [HYPOTHESIS] 环境爆炸触发：当 Python/curl 进入沙箱，行动空间爆炸 → WM 永远无法完全预测 → 探索永不停止 → 创造性行为自然涌现

5.4 从观察到意图
- Gap: 当前 WM 拥有知识但不应用
- 机制：preference distribution p(o|C) 可以把"拥有知识"转化为"驱使行动"
- [HYPOTHESIS] 如果 C 定义为"偏好观察到之前未见过的输出模式"，Agent 会自发寻求 novelty

5.5 Agent 能否触碰自己？
- Problem: Agent 能否观察自己的认知状态，并主动改善它？
- 理论路径：
  1. Error vector 暴露为 State 的一部分
  2. "train_on_recent_data" 成为候选 action
  3. WM 预测：执行训练 → 未来 epistemic 降低
  4. EFE 选择训练 → Agent 主动自我改进
- 这不是 Agent "想变聪明"，是 EFE 计算显示训练自己的预期信息增益最高

5.6 知识→应用的鸿沟
- 当前：WM 记住了 (s,a,s') → epistemic → 0 → 停止
- 需要：从"减少未知"变为"增加已知的作用"
- Preference 设计的关键：p(o|C) 不只匹配目标 predicate，而是编码更抽象的价值
- [HYPOTHESIS] 环境越丰富，偏好分布的设计空间越大，涌现行为的可能性越高

### 6. 路线图（~400 字）

| Phase | 目标 | 状态 | 成功标准 |
|-------|------|------|---------|
| 1: Infrastructure | PEDA 核心循环 | [DONE] | 七模块全部实现并通过测试 |
| 2: Sandbox Foundation | WM 在沙箱中准确预测 | [DONE] | L1/L2/L3 达标, 多任务完成 |
| 3: Epistemic Validation | 预测误差驱动探索 | [NOW] | Epistemic vs Pragmatic 对照有显著差异 OR 负结果记录 |
| 4: Self-Training Loop | 闭合 PEDA 自训练循环 | [NEXT] | WM 覆盖范围在运行中自主扩展 |
| 5: Sandbox Expansion | 可写+Python+网络 | [PLANNED] | 每个扩展触发新的探索波 |
| 6: Knowledge→Application | 偏好驱动目标生成 | [PLANNED] | Agent 自发生成并完成任务 |
| 7: Self-Modification | Agent 主动训练自己 | [FUTURE] | 训练成为候选 action |

### 7. 局限性（~300 字）
- 0.5B 模型不足以形成抽象泛化
- 当前实验规模太小（1-episode pilot, 10 episodes per condition）
- OOD 测试仅在文件级别，未测试语义级别泛化
- Epistemic 分解是启发式的（ensemble variance），不是严格的
- Drive weights 是手动调的，未系统优化
- Phase 2 的 PEDA 闭环从未真正闭合（LearningModule 缺失）

### 8. 相关工作（~500 字）
- Voyager: skill library as learning → PEDA 考虑引入
- DreamerV3: RSSM World Model → LLM 替代
- JEPA: non-generative prediction → PEDA v2.x 方向
- BYOL-Explore: latent-space exploration → 参考
- 经典 Active Inference 实现 → 对比差异
- 现代 LLM Agent（ReAct, Reflexion, AutoGPT）→ 持久状态 + prompt → PEDA 用误差替代 prompt

### 9. 结论（~200 字）
PEDA 是一个在 FEP 框架内探索 Agent 自主性的工程研究项目。当前量化指标达标，基础设施完善，但核心假设——预测误差驱动探索——仍未验证。Phase 3 的实验将直接回答这个问题。无论结果是正或负，都将是对 "Active Inference + LLM Agent" 可行性的有价值贡献。

## 引用素材
- `PEDA_FINAL/peda_report_v11.agent.final.md` — 原始架构文档
- `PEDA_FINAL/RESEARCH_CHARTER.md` — 研究宪章
- `PEDA_FINAL/peda_reflection_v11.md` — v1.1 改进
- `PEDA_FINAL/peda_independent_review.md` — 第三方评审
- `PEDA_WORKING_LOG.md` — 全部工作日志
- `PEDA_FINAL/archive/phase1/phase1_gap_report.md` — Phase 1 gap
- `PEDA_FINAL/archive/phase2/phase2_adapter_train_report.md` — Phase 2 训练报告
- `src/phase1/world_model.py` — WM + LearningModule 实现
- `src/phase2/sandbox_env.py` — Sandbox 实现
