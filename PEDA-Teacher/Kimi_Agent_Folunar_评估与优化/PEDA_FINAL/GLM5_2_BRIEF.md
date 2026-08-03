# PEDA 项目技术简报（for GLM-5.2）

> **项目**: PEDA (Predictive-Error-Driven Autonomous Agent)
> **定位**: 研究探索 — 用预测误差替代 Prompt 作为 AI Agent 的驱动力
> **当前状态**: Phase 1.5 完成，决策点：是否进入 Phase 2 (busybox sandbox)
> **核心问题**: Active Inference + LLM World Model 的技术可行性

---

## 1. 一句话命题

PEDA 尝试用**内部预测误差**（而非外部 Prompt）驱动 AI Agent 的行动。Agent 拥有持久 World Model，持续生成预测；预测与实际感知的误差通过 Expected Free Energy (EFE) 驱动行动选择。

**诚实声明**: 这不是通往 AGI 的捷径。我们追求"看起来像自主"的工程效果，评估标准是"是否产生有趣、有用的行为"，而非"是否真正自主"。

---

## 2. 理论框架

| 理论 | 来源 | PEDA 中的角色 |
|------|------|--------------|
| **Active Inference / FEP** | Friston et al. 2006-2023 | EFE 统一探索与利用：G(π) = Epistemic Value + Pragmatic Value |
| **Predictive Coding** | Rao & Ballard 1999 | 预测误差作为学习信号（直觉框架，实际用标准 BP+LoRA） |
| **World Models** | Ha & Schmidhuber 2018 | LLM+LoRA 作为 World Model，参考 RSSM 架构 |
| **Intrinsic Motivation** | Pathak et al. 2017 | Ensemble 不确定性近似 epistemic value |

**EFE 公式**: G(π) = H[q(o|π)]（认识价值，驱动探索不确定区域）+ D_KL[q(o|π) || C(o)]（实用价值，驱动偏好状态）

---

## 3. 架构（5 模块 + Drive System）

```
[Perception] → [World Model (LLM+LoRA)] → [Predictive Error Computer]
                                              ↓
[Environment] ← [Action Executor] ← [Action Generator (EFE-driven)]
                      ↑
              [Learning Module] ← 间歇性批量 LoRA 微调
                      ↑
         [Homeostatic Drive System] ← 四 Drive 动态平衡
```

| 模块 | 实现 | 功能 |
|------|------|------|
| Perception | 状态→文本 | 将环境状态编码为 World Model 可理解的文本 |
| World Model | Qwen2.5-0.5B-Instruct + LoRA | 预测 (state, action) → next_state |
| Error Computer | Ensemble 方差（3 checkpoints） | 分解 epistemic（可约化）vs aleatoric（不可约化）不确定性 |
| Action Generator | EFE 最小化 | 生成候选动作，选择 EFE 最低的动作 |
| Drive System | 四 Drive 加权 | curiosity / competence / boredom / novelty 动态调制 EFE |

**Drive System**:
- Curiosity: 探索新状态（与 epistemic 协同）
- Competence: 追求可预测区域（实用价值放大）
- Boredom: 惩罚重复状态（打破死循环）
- Novelty: 惩罚近期访问过的状态（短期记忆）

---

## 4. Phase 1.5 实验结果（完整）

### 4.1 环境

自定义 2 房间文本环境：
```
书房 (study) ──── 门向北 ──── 走廊 (hallway)
├── 书桌上有一把钥匙          ├── 墙角有一个上锁的宝箱
└── 6 个合法动作               └── 6 个合法动作
```
最优路径：拿钥匙 → 向北走 → 用钥匙开宝箱（3 步）

### 4.2 训练

- 模型: Qwen2.5-0.5B-Instruct
- 数据: 穷举 + 随机游走 → 去重后 **114 条唯一样本**
- Loss: 0.26 → 0.06 → 0.02（3 epochs）
- Checkpoints: 3 个（用于 ensemble 方差）

### 4.3 关键发现

#### ✅ 已验证

| 发现 | 证据 |
|------|------|
| PEDA ≠ Pragmatic 行为可区分 | 2/2 迭代复现：PEDA step 1 尝试 take key，Pragmatic 全程 look |
| Drive System 有独立探索价值 | epistemic≈0 时 boredom + LLM 置信度仍能驱动不同行为 |
| decompose_error bug | 语义探针 50% 分歧 vs decompose_error 0.0 → 修复后 0.20 |
| 2 房间状态空间太小 | 6000 次尝试 → 114 去重样本，数据增强无效 |

#### ❌ 未验证

| 假设 | 状态 |
|------|------|
| Epistemic error 驱动有意义探索 | 未验证（环境太简单，模型没学好） |
| EFE 优于贪心策略 | 部分（PEDA ≠ Pragmatic，但都未成功完成任务） |
| World Model 能学习文本转移动态 | 否（114 条数据不够，take key 系统性预测错误） |

#### 预测检查（e4）

| 动作 | 预测 exit | 正确值 | 状态 |
|------|----------|--------|------|
| take key | 1 ❌ | 0 | 系统性错误（所有 checkpoint） |
| go north | 1 ❌ | 0 | 比 e3 更差 |
| unlock chest w/ key | 1 ✅ | 1 | 正确 |
| look | 0 ✅ | 0 | 正确 |

### 4.4 行为分析

PEDA 选择 take key **不是因为模型预测它成功**，而是因为它的 epistemic_ratio 更高（置信度更低），在 EFE 中产生微弱探索偏差。

Agent 成功拿到钥匙后卡在 `inventory` 死循环（17 步），因为模型对 inventory 预测置信度 0.999 → EFE 最低 → 每次都选 inventory。

---

## 5. 当前决策点

**选项 D: 进入 Phase 2 (busybox sandbox)**

理由：
1. Phase 1.5 完成使命（基础设施验证 + bug 修复 + 负结果积累）
2. 2 房间环境是结构性瓶颈（状态空间太小），非数据策略问题
3. Busybox 的不确定性是天然的（非人造），数据空间天然足够大
4. Drive System 发现值得在更复杂环境中继续验证

**下一步**: Docker busybox 环境 → PEDA 集成 → 数据收集 → 行为观察

---

## 6. 核心技术问题（请 GLM-5.2 评估）

### Q1: 技术路线可行性
Active Inference + LLM-as-World-Model 这条路线，你认为核心瓶颈在哪里？FEP 的数学框架与 LLM 的统计特性之间是否存在根本性 mismatch？

### Q2: Phase 1.5 结果解读
我们的 epistemic 信号（ensemble 方差）在 2 房间环境中只有 0.20，且主要来源是 has-key 维度而非结构化的转移规则学习。这是否说明 LLM 作为 World Model 的**因果推理能力**不足？还是环境复杂度问题？

### Q3: Phase 2 设计建议
Busybox sandbox 是否足够复杂以产生有意义的 epistemic 信号？还是需要更复杂的初始环境（如包含文件系统操作、管道、环境变量等）？沙箱环境与 LLM World Model 之间的**语义鸿沟**（命令输出 → 文本 → 状态表示）你有什么建议？

### Q4: 架构改进
PEDA 当前用 ensemble 方差近似 epistemic uncertainty。你认为更好的 epistemic 估计方法是什么？（如 MC Dropout、深度集成、或 LLM 自身的 confidence logit）

### Q5: 遗漏相关工作
是否有我们遗漏的关键相关工作？特别是：
- LLM 作为 World Model 的最新进展
- 文本环境中的 Active Inference 实现
- 内在动机 + LLM 的组合方案
- 预测误差驱动的 Agent 评估方法

### Q6: Drive System 评估
我们的 Drive System（四 Drive 加权）在 epistemic≈0 时仍能产生可区分行为。你认为这是一个**有价值的发现**（Drive System 有独立价值）还是一个**artifacts**（EFE 公式的副产品）？

### Q7: 统计与实验设计
我们目前只有 1-2 episode 的行为数据。在 CPU-only、0.5B 模型的硬件约束下，你有什么实验设计建议可以在有限资源下获得更有说服力的结果？

---

## 附录：项目文件索引

| 文件 | 内容 |
|------|------|
| `PEDA架构设计与开发计划书_v1.1.docx` | 完整技术文档（634K） |
| `phase1_5_complete_report.md` | Phase 1.5 Iteration 1 报告 |
| `phase1_5_iteration2_report.md` | Phase 1.5 Iteration 2 报告 |
| `PHASE1_5_ITERATION2_EVALUATION.md` | 上游评估与决策 |
| `WATCHDOG.md` | 21 条开发规则（防坑指南） |
| `RESEARCH_CHARTER.md` | 研究宪章（负结果也是结果） |
