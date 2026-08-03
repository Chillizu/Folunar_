## 1. 执行摘要

### 一句话命题

PEDA（Predictive-Error-Driven Autonomous Agent）尝试用预测误差替代Prompt，作为驱动AI Agent行动的核心信号——这不仅是工程上的重构，更是对"智能本质"的一次范式追问。

### 背景与动机

当前所有LLM应用——从ChatBot到AutoGPT——都困在同一个范式中：**模型是冻结权重，每次调用从零推理，没有外部输入（Prompt）就没有输出**。Folunar_试图打破这一范式，构建"不依赖用户提示"的自主Agent，但其技术路线（<1M参数从零训练、在线学习每步SGD、硬编码目标轮转）与所追求的"涌现自主"存在根本矛盾。

PEDA从计算神经科学中汲取理论武器，提出了一条不同的路径。

### 核心理论支撑

| 理论 | 来源 | 对PEDA的意义 |
|------|------|-------------|
| **Free Energy Principle (FEP)** | Friston et al. [2006, 2010, 2017] | 提供统一目标函数：最小化变分自由能 = 感知 + 行动 + 学习 |
| **Predictive Coding** | Rao & Ballard [1999]; Clark [2013, 2015] | 预测误差作为学习和感知的信号；可用局部学习替代反向传播 |
| **Active Inference** | Friston et al. [2017] | Expected Free Energy (EFE) 统一探索(epistemic)与利用(pragmatic) |
| **World Models** | Ha & Schmidhuber [2018]; Hafner et al. [2019-2023] | 内部模型预测环境变化，支持想象和规划 |
| **Intrinsic Motivation (FEP视角)** | Pathak et al. [2017]; Burda et al. [2018] | FEP通过信息增益（非预测误差）自然解决Noisy TV问题 |

### PEDA核心架构（五大模块）

```
[Perception] → [World Model] → [Predictive Error Computer]
                                    ↓
[Environment] ← [Action Executor] ← [Action Generator (EFE)]
                      ↑
              [Learning Module] ← 间歇性批量更新
                      ↑
         [Homeostatic Drive System] ← 四Drive动态平衡
```

**关键设计决策**：
1. **World Model是核心**（1-7B预训练LLM + LoRA微调），其余模块辅助
2. **预测误差是驱动**（不是Prompt，不是硬编码目标）
3. **学习是间歇的**（批量更新，不是每步SGD——防止灾难性遗忘）
4. **Drive System调节探索/利用平衡**（Curiosity / Competence / Boredom / Novelty）
5. **EFE最小化选择行动**（通过rollout想象，选择预期信息增益最大的行动）

### 与Folunar_的根本区别

| 维度 | Folunar_ | PEDA |
|------|---------|------|
| 驱动信号 | 硬编码目标轮转 | 预测误差（内在涌现） |
| 模型规模 | <1M参数，从零训练 | 1-7B预训练 + LoRA微调 |
| 学习机制 | 每步在线SGD（灾难性遗忘） | 间歇性批量更新 |
| 探索策略 | RND（100步后失效） | EFE信息增益（Noisy TV免疫） |
| 好奇心机制 | 预测误差（混淆可约/不可约） | Epistemic Value（信息增益） |
| 模块数量 | 40+（膨胀） | 5核心 + 4 Drive（精简） |
| 环境开放性 | `--network none`封闭 | 可控开放（只读外部数据） |

### 开发路线图

| 阶段 | 时间 | 目标 | 成功标准 |
|------|------|------|---------|
| Phase 1 | 2-4周 | Grid World验证核心假设 | 预测误差驱动探索效率 > 随机2x |
| Phase 2 | 4-8周 | Linux沙箱World Model训练 | 预测准确率 > 70% |
| Phase 3 | 4-6周 | 整合评估 | 48小时连续运行，行为"有趣" |

**关键决策点**：Phase 1是唯一"未达标则停止"的阶段。

### 能做到 vs 不能做到

**能做到**：
- 不依赖用户输入的自主探索
- 行为上的"成长"（探索效率提升）
- 有趣的、多样的行为模式
- 对新环境的适应能力

**不能做到**（必须诚实面对）：
- 真正的意识、自我、主观体验
- 从零产生价值判断（"好"/"坏"需要人为定义或从人类数据中学习）
- 超越训练数据分布的原创知识

### 核心资产 vs 核心负债（从Folunar_继承）

**继承**：Docker沙箱执行环境、闭合感知-执行循环的工程经验、REFLECTION.md的自我反思诚实

**抛弃**：<1M参数模型、在线每步SGD、硬编码目标、模板引擎、40+模块膨胀、RND好奇心驱动

---

> **最后的话**：PEDA不是通往AGI的捷径。它是一次诚实的尝试——用有理论支撑的方式，在现有技术条件下，构建一个"看起来有自主性"的系统。它的价值不在于"是否真的自主"（这是哲学问题），而在于"是否能产生有趣、有用、令人惊讶的行为"（这是工程问题）。如果Phase 1验证失败，我们会及早知道——这比在错误方向上投入6个月更有价值。
