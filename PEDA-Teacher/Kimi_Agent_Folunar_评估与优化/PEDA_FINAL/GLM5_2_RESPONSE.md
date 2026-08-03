# GLM-5.2 技术咨询回应原文

> **日期**: 2026-07-07
> **咨询对象**: GLM-5.2（智谱清言）
> **Prompt**: `GLM5_2_PROMPT.md`
> **状态**: 完整回应，7/7 问题全部回答

---

## Q1: Active Inference + LLM-as-World-Model 的技术路线评估

⚠️ **理论障碍警告**：FEP 的数学框架与 LLM 的统计特性之间存在根本性 mismatch。

Active Inference 的 EFE 公式 G(π) = H[q(o|π)] + D_KL[q(o|π) || C(o)] 要求对观察空间进行精确的概率分布积分。但在 LLM-as-World-Model 中，你用自回归生成的 token 序列近似 o，用 token 的 softmax 概率近似 q(o|π)。这带来两个致命问题：

1. **概率空间不对齐**：LLM 的生成概率包含大量对"语法正确性"的置信度，而非对"环境状态因果转移"的置信度。模型对 "take key" 系统性预测错误但置信度极高（0.999），说明它把语言模式匹配的自信误认为了环境动力学的确定性。
2. **计算维度灾难**：真实文本环境的观察空间 o 是组合爆炸的，LLM 生成的只是该空间的一个点采样。用单点采样的 ensemble 方差去近似整个状态空间的熵 H[q(o|π)]，在数学上是无效的。

当前路线的核心瓶颈在于：**你试图用一个语言模式匹配器去执行概率图模型的任务**。这不是调参能解决的，而是表征空间的错位。

---

## Q2: Phase 1.5 结果的深度解读

这既是因果推理能力不足，也是数据问题，但核心是**LLM 的表征偏好**。

模型对 "has-key" 维度产生分歧，是因为这涉及文本中特定 token 的出现与否（如 "key" in inventory），这是 LLM 擅长的表层模式匹配。模型对 "take key" 后状态转移预测一致但错误（预测 exit=1），说明它没有建立 "动作->状态变更" 的因果图，而是掉入了预训练语言的先验陷阱（"take" 通常意味着 "obtain/success"）。

114 条数据对 0.5B 模型确实太少，但即便给 1140 条，LLM 也是通过 memorize 序列模式来学习，而非学习符号逻辑。LLM 作为 World Model 的缺陷在于：它把确定性的状态机（Grid/2-room）视作概率性语言序列来拟合，不可避免地引入语言先验噪声。这也解释了为什么 decompose_error 修复后 epistemic=0.20，信号依然无法指导正确决策——因为分歧发生在不重要的维度上。

---

## Q3: Phase 2 环境设计建议

✅ **环境建议**：Busybox sandbox 是正确的方向，但当前白名单太简单。`ls, cat, echo` 等命令在简单初始文件下产生的输出极为有限，状态空间很快会收敛。建议增加组合复杂性：多层目录结构、需管道操作的文件（如 `grep "keyword" file.txt | wc -l`）、可变环境变量。通过组合爆炸提供天然的 epistemic 信号。

⚠️ **语义鸿沟警告**：绝对不要让 LLM 直接预测 raw terminal output。终端输出包含对齐空格、权限字符等，LLM 处理这些极度消耗容量且易产生 aleatoric noise。

✅ **实施要点**：引入结构化感知中间件。将环境状态序列化为 JSON（如 `{"cwd": "/docs", "files": ["note.txt"], "stdout": "..."}`），LLM 预测下一个 JSON 状态。这能把 LLM 的注意力强制对齐到因果状态变更上，大幅降低语义鸿沟带来的噪声。

---

## Q4: Epistemic Uncertainty 估计方法

在 CPU-only 和 0.5B 模型约束下：

⚠️ **不推荐**：MC Dropout（推理速度慢 10 倍+，且对自回归 LLM 效果存疑）、显式方差预测（改变了 World Model 的训练目标，与当前架构冲突）。

✅ **推荐方案**：混合方法（Ensemble + Token-level Entropy）。
当前 3 个 LoRA 方差太小是因为同源初始化和数据太少。建议：
1. 保留 3 个 LoRA，但使用不同 random seed 的数据顺序和不同的 dropout rate 进行训练，强制增加 ensemble 分歧。
2. 引入 **生成 token 的 logit 熵** 作为 epistemic 补充。如果模型对 next_state 的某个关键 token（如 file_name）生成概率分散，这才是真正的 epistemic uncertainty。
3. **置信度惩罚机制**：针对 0.999 死循环，实施 temperature scaling 或 confidence penalty。当模型对某个动作的预测置信度 > 0.95 时，强制注入人工噪声降低其 EFE 权重，打破贪心死锁。

---

## Q5: 遗漏相关工作

⚠️ **重大遗漏：JEPA (Joint-Embedding Predictive Architecture, LeCun 2022) 及其变体（I-JEPA, V-JEPA）。**

PEDA 当前试图让 LLM 在 token 空间（观察空间 o）直接做预测，这是高度低效的。JEPA 的核心思想是在 latent space 做预测，丢弃难以预测的底层细节。这会直接改变你们的技术路线：不要让模型预测 `next_state` 的文本，而是预测 `next_state` 的 embedding。

其他需补充：
- **TextWorld 相关的 RL 工作**：特别是如何处理文本环境的稀疏奖励和状态空间抽象。
- **AgentTuning / ToolBench**：观察它们如何让小模型在工具调用环境中建立因果反馈回路，而非纯生成。
- **Evidential Deep Learning (EDL)**：如果想在单模型下估计 epistemic uncertainty，EDL 比单纯 ensemble 在理论上更适配，通过预测 Dirichlet 分布参数来量化证据不足。

---

## Q6: Drive System 的价值评估

⚠️ 这是一个 **artifact（伪探索）**，不具有 Active Inference 理论上的泛化性。

当 epistemic≈0 时，EFE 退化为贪心。PEDA 尝试 `take key` 而非 `look`，仅仅是因为 Boredom 累积了一个负 bonus，或者 `take key` 的 LLM 置信度较低。这在本质上是一个带有短期记忆惩罚的 epsilon-greedy 变体，或者是启发式搜索中的 step penalty。这不是 FEP 框架的胜利，而是工程兜底机制在起作用。

**如何消除/验证**：
把 Drive System 从 EFE 公式中剥离出来测试。如果直接用一个随机优先队列 + Boredom 惩罚，也能复现 "尝试 take key" 的行为，说明 Drive System 没有提供超越启发式搜索的数学价值。建议将 Drive System 视为工程上的"防呆设计"，而非理论核心，不要在此浪费过多研究精力去论证其"独立价值"。

---

## Q7: 实验设计与统计建议

1-2 episodes 毫无统计意义。在极强算力约束下，必须改变实验设计思路。

✅ **实施建议**：
1. **缩短 Episode 长度**：将单次任务截断为 5-10 步的微任务（如 "从 / 移动到 /docs 并 cat note.txt"）。单 episode 耗时降至 2-3 分钟，一晚上可跑 100+ episodes。
2. **基线设计**：必须加入 **Random Walk 基线** 和 **Prompt-driven 基线**（直接给 Qwen2.5 Prompt 让它做 Few-shot 决策）。如果 PEDA 连 Random Walk 都打不过，核心架构即宣告失败；如果打不过 Prompt-driven，说明"预测误差驱动"不如"直接指令驱动"。
3. **评估指标**：
   - **First Hitting Time (FHT)**：首次到达目标状态的步数。
   - **State Coverage Rate (SCR)**：在 N 步内探索到的不同状态数 / 总合法状态数。
   - **Dead-loop Rate**：陷入置信度>0.99 循环的频率。
4. **统计策略**：采用 Pre-registration。提前声明假设、目标指标和终止条件。在 30+ episodes 的基础上，使用非参数检验（如 Mann-Whitney U test）比较 PEDA 与基线的 FHT 分布。

---

## 总体评估

**成功概率：15-20%**（作为验证"预测误差替代 Prompt 驱动 Agent"这一核心假设的概率）。

最大风险点：**将工程兜底（如 Boredom、Logit penalty）产生的可区分行为，错误归因为 Active Inference 理论的有效性**。你们从 Folunar_ 的过度膨胀走向了过度防御，但这种"防坑文化"可能导致你们在一条理论 mismatch 的路线上精细打磨（死磕 0.5B 模型的 LoRA 方差），而不敢做激进的根本性架构调整（如转向 Latent Space 预测或彻底放弃用 LLM 做 World Model）。

FEP 与 LLM token 级生成的数学错位是致命的。建议尽快进入 Phase 2，在真实复杂环境中让这个矛盾彻底暴露，不要在当前架构上做过度微调。如果 Phase 2 依然无法产生有效的 epistemic 信号，应果断宣告假设证伪，这本身就是一个极具价值的负结果。
