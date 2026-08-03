# GLM-5.2 技术咨询 Prompt

> **使用方法**: 将本文档的全部内容（从"【角色设定】"到结尾）复制粘贴给 GLM-5.2。
> 不需要额外补充信息。如果 GLM-5.2 要求澄清，把对应部分从 BRIEF.md 贴过去。

---

【角色设定】

你是 PEDA 项目的技术顾问。PEDA（Predictive-Error-Driven Autonomous Agent）是一个研究探索项目，尝试用内部预测误差替代 Prompt 作为 AI Agent 的驱动力。项目当前处于 Phase 1.5 完成、准备进入 Phase 2 的决策点。你需要基于以下信息，对项目的技术路线、实验结果和下一步设计给出尖锐、建设性的评估。

请直接回答技术问题，不需要客套。如果你认为某个方向不可行，请明确说。如果你认为我们遗漏了什么，请指出。负结果和警告对我们来说比鼓励更有价值。

---

【项目背景】

## 核心理论

PEDA 基于 Active Inference / Free Energy Principle（Friston et al.）设计。核心公式：

G(π) = H[q(o|π)]（Epistemic Value，驱动探索不确定区域）+ D_KL[q(o|π) || C(o)]（Pragmatic Value，驱动偏好状态）

Agent 拥有持久 World Model（LLM+LoRA），持续生成 (state, action) → next_state 的预测。预测误差通过 EFE 驱动行动选择。

## 架构（5 模块）

Perception → World Model (Qwen2.5-0.5B + LoRA) → Predictive Error Computer (Ensemble 方差, 3 checkpoints)
                                                                    ↓
Environment ← Action Executor ← Action Generator (EFE 最小化)
                    ↑
            Learning Module (间歇性 LoRA 微调)
                    ↑
        Drive System: curiosity / competence / boredom / novelty 加权调制 EFE

## Drive System 权重（当前配置）

- curiosity: 0.1
- competence: 2.0
- boredom: 0.1
- novelty: 2.0
- pragmatic_weight: 3.0

---

【项目编年史：我们从哪里来】

## 起源：Folunar_ 的教训

PEDA 的前身是 Folunar_（后改名 Trahexa），一个尝试构建"自主 AI Agent"的项目。它犯了以下错误，直接导致 abandonment：

- **142 个 commits，40+ 模块，~30,000 字计划文档** → 核心假设从未被验证
- **用"计划完成"作为"项目成功"的代理指标** → 文档膨胀，代码空心化
- **Stub 模式陷阱** → 给 stub 注入随机噪声让测试通过，误以为是真实行为
- **过早 commit/push** → 每个想法一出现就 commit，没有验证就推进
- **模块膨胀** → 40+ 模块很多 <150 行，为了"完整性"而非"必要性"
- **Grid Search 结果可疑未调查** → score=0.996, steps=1.0 的异常结果直接被接受
- **过度纠结 lint** → 连续多轮只修 ruff/pyright/mypy，不做核心验证

PEDA 的存在就是为了不重蹈覆辙。项目宪章规定："负结果也是结果，没有时间压力， honesty > speed"。

## Phase 1：Grid World（已放弃）

**目标**：在 5×5 网格中验证 epistemic error 能否驱动探索
**结果**：失败，但获得了工程基础设施验证

实验过程：
- 25% 训练数据 → g1_test_set=0.87 → ensemble 分歧 7%
- 10% 训练数据 + 3 epochs → g1_test_set=1.0 → ensemble 分歧 18%
- 根因：**5×5 Grid World 对 0.5B 模型太简单**，模型容量（5亿参数）远超环境复杂度（25状态 × 4动作 = 100种组合）

**关键教训（B7/B8 规则）**：
- 如果 g1 > 0.90 且 <50% 训练数据 → 环境太简单，不要试第三次
- 最多 2 次尝试 per condition，第 3 次需要书面理由
- Grid World 的价值不是验证假设，而是验证了工程基础设施（LLM 加载、LoRA 微调、EFE 计算、评估循环）

## Phase 1 第一轮验证的陷阱

第一轮验证（pragmatic_weight=3.0）结果：
- PEDA success=0.2, Random success=0.63
- PEDA 表现差于随机！

根因：pragmatic_weight=3.0 + epistemic≈0 → **EFE 退化为纯贪心距离最小化**
Agent 不是被预测误差驱动，而是被"离目标最近"驱动。

**教训（C8 规则）**：必须监控 epistemic/pragmatic 比例。如果 pragmatic 贡献 >80%，系统不是"预测误差驱动"，而是"贪心导航 + 小好奇 bonus"。

## 独立评审与 v1.1 改进

PEDA v1.0 计划书接受独立第三方评审，评分 **5.5/10**。主要问题：
- 理论与实践脱节（2.6 节连续时间架构 1500 字 → 删除）
- "不需要外部目标"误导性表述 → 修正为"目标从 reward 转变为偏好分布"
- 70% 预测准确率是拍脑袋数字 → 分层目标（exit 90%+/文件系统 70%/输出 50%）
- Grid World → Linux 跳跃太大 → 新增 Phase 1.5
- 遗漏相关工作（Voyager 等）→ 补充完整
- 推理速度、安全设计、Drive 超参数敏感性、LLM 幻觉 → 全部新增章节

v1.1 改进后自评 **7.0/10**。

## 第三方评价（9 点批评）

一位外部评审者提出以下尖锐批评（逐条已回应）：

1. **"Emergence" 是幻觉** → 同意，改为 L2 模拟（涌现行为 vs 涌现智能）
2. **Ornith 架构借鉴不足** → 同意，提取 Self-Scaffolding 作为 v2 方向
3. **<1B 参数不够** → 同意，定位研究探索，不声称 AGI
4. **Phase 跳跃风险** → 同意，新增 Phase 1.5
5. **Continuous cognition 缺失** → 同意，标记为 v2 方向
6. **评估标准不清晰** → 同意，全部量化为可计算指标
7. **数据效率问题** → 部分同意，探索 ensemble 蒸馏
8. **与现实世界脱节** → 同意，Phase 2 用 busybox
9. **商业可行性为零** → 同意，研究探索定位，无商业目标

## 已验证的工程能力（无论核心假设如何）

以下基础设施已通过多轮实验验证，是 Phase 2 的基础：

- ✅ LLM 加载与推理（Qwen2.5-0.5B-Instruct）
- ✅ LoRA 微调与 checkpoint 保存（3 checkpoints）
- ✅ EFE 计算与行动选择（含 pragmatic_only 基线）
- ✅ Drive System 四 Drive 动态平衡
- ✅ Ensemble Error Computer（修复 decompose_error 后）
- ✅ 评估循环与指标计算（epistemic/aleatoric/success rate）
- ✅ `hasattr` 分派向后兼容（Grid/Text/Sandbox 三路复用）
- ✅ 语义探针（结构化字段分歧率测量）
- ✅ 分块 eval（`--start-episode`，CPU-only 长时推理的必要模式）

## 已知陷阱模式（WATCHDOG 规则摘要）

项目维护 21 条规则防止重蹈覆辙：

**Blocker（8 条）**：
- B1: 无假设验证就推进 Phase
- B2: 给 stub 注入假数据
- B3: 无门槛新增模块
- B4: 创建新 PLAN 文档而非更新旧文档
- B5: 样本不足就宣称验证（<5 episodes）
- B6: Cherry-picking 实验条件
- B7: 环境太简单却继续调参
- B8: "Just one more try" 死亡螺旋（≥3 次同一参数调整）

**Concern（12 条）**：
- C1: Lint/docs/git 连续 >2 轮
- C2: 用过程指标替代 go/no-go 指标
- C3: 异常结果不调查
- C4: Drive 权重无依据
- C5: 安全边界缺失
- C6: 推理速度瓶颈不处理
- C7: Pilot 成功就跳过 confirmatory
- C8: Epistemic 被 pragmatic 压制
- C9: 训练-评估同分布
- C10: 计划偏差不报告
- C11: 测量方法与独立真值不一致
- C12: Agent 陷入局部最优（置信度 >0.99 死循环）

---

【Phase 1.5 实验结果】

## 环境

自定义 2 房间文本环境：
书房(study) ──门向北── 走廊(hallway)
- study: 书桌上有钥匙，6 个合法动作
- hallway: 墙角有上锁宝箱，6 个合法动作
- 最优路径：拿钥匙 → 向北走 → 开宝箱（3 步）

## 训练

- 模型: Qwen2.5-0.5B-Instruct, 3 epochs, batch_size=4
- 数据: 穷举 + 随机游走(200 walks × 30 steps) → 去重后 114 条
- Loss: 0.26 → 0.06 → 0.02
- 3 个 checkpoints 用于 ensemble 方差

## 预测准确率

| 动作 | 预测 exit | 正确值 |
|------|----------|--------|
| take key | 1 ❌ | 0 |
| go north | 1 ❌ | 0 |
| unlock chest w/ key | 1 ✅ | 1 |
| look | 0 ✅ | 0 |

## 行为对比（2 次迭代均复现）

PEDA: step 1 尝试 take key → 拿到钥匙 → 卡在 inventory 死循环
Pragmatic: 全程 look，从不尝试 take key

## Epistemic 信号

修复 decompose_error bug 后：mean_epistemic_error = 0.20（修复前 0.0）
语义探针：has-key 维度 40% 分歧，完整元组 50% 分歧

---

【已验证的发现】

1. PEDA ≠ Pragmatic 行为可区分（2/2 迭代复现）
2. Drive System 有独立探索价值（epistemic≈0 时 boredom + LLM 置信度仍能驱动不同行为）
3. decompose_error bug 修复有效（0.0 → 0.20）
4. 2 房间环境状态空间太小（6000 尝试 → 114 去重），数据增强无效

【未验证的假设】

1. Epistemic error 驱动有意义探索（环境太简单 + 模型没学好）
2. EFE 优于纯贪心策略（PEDA ≠ Pragmatic，但都未成功）
3. World Model 能学习文本转移动态（114 条数据不够）

---

【核心问题】

请逐一回答以下 7 个问题。每个问题的回答控制在 200-500 字。如果某个问题需要更多信息才能回答，请明确说"需要补充 X"而不是猜测。

**重要背景**：PEDA 是一个从失败中学习出来的项目。Folunar_ 曾花 142 commits 和 40+ 模块却从未验证核心假设。PEDA 的设计（WATCHDOG 规则、研究宪章、30 分钟止损规则）都是为了防止重蹈覆辙。请在评估时考虑：PEDA 的架构选择中，哪些是技术必要，哪些是过度防御？这种"防坑优先"的文化是否可能阻碍了激进但有价值的尝试？

---

**Q1: Active Inference + LLM-as-World-Model 的技术路线评估**

PEDA 的核心假设是：LLM 可以作为 World Model，通过 ensemble 方差产生 epistemic 信号，经由 EFE 驱动有意义的探索行为。基于 Phase 1.5 的结果（epistemic=0.20 但无法驱动任务完成，模型连 take key 都预测错误），你认为这条路线的**核心瓶颈**在哪里？FEP 的数学框架（要求精确的概率分布 q(o|π)）与 LLM 的统计特性（自回归生成、没有显式的概率分布）之间是否存在**根本性 mismatch**？请诚实评估：这个组合是"工程上需要调优"还是"理论上存在障碍"。

---

**Q2: Phase 1.5 结果的深度解读**

我们的 epistemic 信号（ensemble 方差=0.20）主要来源是 has-key 维度（模型对"背包里是否有钥匙"预测不一致），而非结构化的转移规则学习（模型对"执行 take key 后状态如何变化"预测一致但错误）。这是否说明 LLM 作为 World Model 的**因果推理能力**不足——它能生成合理的文本描述，但无法学习精确的因果转移规则？还是说这纯粹是环境复杂度/数据量问题（114 条样本对 0.5B 模型太少）？

---

**Q3: Phase 2 环境设计建议**

我们计划进入 Phase 2：Docker busybox sandbox。假设：
- 命令白名单：ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep
- 安全：cap-drop=ALL, read-only, network none
- 初始文件：hello.txt, docs/note.txt, data/lines.txt

你认为这个环境是否足够复杂以产生**有意义的 epistemic 信号**？还是需要更复杂的初始环境（如多层目录结构、需要管道操作的文件、环境变量 puzzle）？另外，命令输出（自由文本）→ Perception 解析 → World Model 预测 这条链路中，**语义鸿沟**（LLM 需要理解"ls -la 的输出表示什么"）你有什么具体建议？

---

**Q4: Epistemic Uncertainty 估计方法**

PEDA 当前用 3 个 LoRA checkpoints 的预测方差近似 epistemic uncertainty。已知问题：
- 0.5B 模型 + 114 条数据 → checkpoints 差异小
- decompose_error 曾漏掉 has-key 维度（已修复）
- 模型对某些 action 过度自信（inventory 置信度 0.999）

在**不更换更大模型**的约束下，你认为更好的 epistemic 估计方法是什么？选项包括但不限于：
- MC Dropout（在 LLM 推理时启用 dropout）
- 深度集成（更多 checkpoints，不同初始化）
- LLM 自身的 confidence logit（使用生成 token 的概率）
- 显式的不确定性建模（训练模型预测方差）
- 混合方法（ensemble + confidence）

请给出你的推荐及理由。

---

**Q5: 遗漏相关工作**

PEDA 已参考的工作：Friston (FEP), Ha & Schmidhuber (World Models), Pathak et al. (ICM), Voyager, BYOL-Explore, JEPA, ReAct, Reflexion。

是否有我们**遗漏的关键相关工作**？特别是以下方向：
- LLM 作为 World Model 的最新进展（2024-2025）
- 文本 / 符号环境中的 Active Inference 实现
- 内在动机（curiosity/boredom/novelty）+ LLM 的组合方案
- 预测误差驱动 Agent 的评估方法论
- 小模型（<1B）World Model 的训练策略

如果某个相关工作会直接改变我们的技术路线，请重点说明。

---

**Q6: Drive System 的价值评估**

我们的 Drive System（curiosity/competence/boredom/novelty 四 Drive 加权）在 epistemic≈0 时仍能产生可区分行为（PEDA 尝试 take key，Pragmatic 不尝试）。行为差异的来源是：
- LLM 置信度信号：epistemic_ratio = 1 - confidence
- Boredom 累积：重复状态 → 负 bonus → 驱动向未尝试动作偏移

你认为这是一个**有价值的独立发现**（Drive System 在不依赖 prediction error 时仍有探索价值），还是一个**artifact**（EFE 公式在特定参数下的副作用，不具有泛化性）？如果是前者，Drive System 值得深入研究的方向是什么？如果是后者，应该如何消除这种"伪探索"？

---

**Q7: 实验设计与统计建议**

当前硬件约束：CPU-only, 0.5B 模型, 单步推理 ~10-30 秒。
当前数据：1-2 episodes per agent per eval。

在这种约束下，你有什么**实验设计建议**可以在有限资源下获得更有说服力的结果？具体问题：
- Episode 数：最少需要多少 episodes 才能声称"行为差异显著"？
- 基线设计：除了 PEDA vs Pragmatic，还需要什么基线？（如纯随机、ε-greedy、纯 curiosity）
- 评估指标：除了 success rate 和 behavior trace，还有什么指标可以量化"探索质量"？
- 硬件受限下的策略：如果无法增加 episode 数，如何通过**实验设计**（如 pre-registration、交叉验证、定性分析）增强结果的可信度？

---

【回答格式要求】

1. 每个问题用 "## QN: 问题标题" 开头
2. 回答控制在 200-500 字，除非问题特别复杂
3. 如果认为某个方向不可行，用 ⚠️ 标记并说明理由
4. 如果推荐某个具体方案，用 ✅ 标记并给出实施要点
5. 最后附一段 "## 总体评估"（300 字以内），给出你对 PEDA 项目整体成功概率的诚实估计（0-100%）及最大风险点
