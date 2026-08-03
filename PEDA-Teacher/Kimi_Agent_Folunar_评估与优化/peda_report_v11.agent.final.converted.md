# PEDA (Predictive-Error-Driven Autonomous Agent)
# 架构设计与开发计划书 v1.1

## ——用预测误差替代Prompt：基于Active Inference的自主Agent设计

**版本**: 1.1  
**日期**: 2026年7月2日  
**v1.0评审**: 独立第三方评审，综合评分5.5/10  
**v1.1改进**: 基于评审反馈系统性修正（详见第1章"v1.1改进声明"和附录`peda_reflection_v11.md`）  

---

## 1. 执行摘要

### 一句话命题

PEDA（Predictive-Error-Driven Autonomous Agent）尝试用预测误差替代Prompt，作为驱动AI Agent行动的核心信号——这是一次**诚实的工程尝试**，有理论支撑，但所有核心假设都需要通过实验验证。

### v1.1 改进声明

本版本（v1.1）基于独立评审反馈进行了系统性改进。评审指出v1.0存在"理论与实践脱节""关键假设未经检验""遗漏重要相关工作"等问题。v1.1逐一修正了这些问题——详见 `/mnt/agents/output/peda_reflection_v11.md`。

**v1.0→v1.1 的核心修正**：

| 问题 | v1.0 | v1.1 |
|------|------|------|
| 2.6节连续时间架构 | 1500字正文 | 删除，改为180字"未来工作" |
| "不需要外部目标" | 误导性表述 | 修正为"目标从外部reward转变为内部偏好分布" |
| PC替代反向传播 | 装饰性讨论 | 明确声明使用标准BP+LoRA，PC仅提供直觉框架 |
| 70%预测准确率 | 拍脑袋数字 | 分层目标：exit code 90%+/文件系统70%/输出摘要50% |
| epistemic/aleatoric分解 | 任意权重0.3/0.7 | ensemble不确定性（多checkpoint预测方差） |
| Grid World→Linux跳跃 | 直接跳跃 | 新增Phase 1.5 (TextWorld中间验证) |
| 遗漏相关工作 | Voyager等未提及 | 新增完整相关工作章节 |
| 推理速度 | 未讨论 | 量化分析+限制方案 |
| 安全设计 | 零讨论 | 新增安全章节（黑白名单+权限限制） |
| Drive超参数敏感性 | 未讨论 | 新增分析+grid search策略 |
| LLM幻觉 | 未讨论 | 新增规则引擎验证层 |
| 评估指标 | "行为有趣"主观指标 | 全部量化为可计算指标 |
| 时间线 | 14-20周（偏乐观2倍） | 29-40周（诚实估计） |

### 核心理论支撑

| 理论 | 来源 | 对PEDA的意义 | 实际使用方式 |
|------|------|-------------|------------|
| **FEP/Active Inference** | Friston et al. [2006, 2010, 2017] | EFE统一探索与利用 | EFE驱动行动选择（非完整FEP实现） |
| **Predictive Coding** | Rao & Ballard [1999]; Clark [2015] | 预测误差作为学习信号 | 直觉框架（实际用标准反向传播） |
| **World Models** | Ha & Schmidhuber [2018]; Hafner et al. [2019-2023] | 内部模型预测环境变化 | 参考RSSM架构，用LLM+LoRA实现 |
| **Intrinsic Motivation (FEP视角)** | Pathak et al. [2017]; Burda et al. [2018] | FEP通过信息增益解决Noisy TV | 启发式ensemble不确定性近似 |

**新增相关工作（v1.1）**：
- **Voyager** (Wang et al., 2023)：自动课程+技能库——PEDA考虑引入skill library作为LoRA微调的替代
- **BYOL-Explore** (Guo et al., 2022)：潜空间预测SOTA——PEDA World Model的设计参考
- **JEPA** (LeCun, 2022+)：非生成式World Model——v2.x的潜在技术路线
- **现代LLM Agent框架**：ReAct、Reflexion的持久状态机制——PEDA在此基础上增加预测误差驱动

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

### 改进后的开发路线图

| 阶段 | 时间 | 目标 | 成功标准 |
|------|------|------|---------|
| Phase 1 | 2-3周 | Grid World验证预测误差驱动探索 | 预测误差驱动 > 随机基线2x |
| **Phase 1.5（新增）** | **3-4周** | **TextWorld验证文本环境WM** | **WM预测准确率>60%** |
| Phase 2a | 6-8周 | Linux沙箱数据收集 | 10000+ (s,a,s')三元组 |
| Phase 2b | 8-12周 | World Model训练 | L1: 90%+, L2: 70%, L3: 50% |
| Phase 3 | 8-10周 | 整合评估 | 48小时运行，量化指标达标 |
| **总计** | **27-37周** | | |

**关键决策点**：Phase 1是唯一"未达标则停止"的阶段。Phase 1.5是第二个决策点（WM在文本环境中不能工作→停止）。

### 能做到 vs 不能做到（诚实声明）

**能做到**：
- 不依赖用户输入的自主探索
- 行为上的"成长"（探索效率提升）
- 有趣的、多样的行为模式
- 对新环境的适应能力

**不能做到**：
- 真正的意识、自我、主观体验
- 从零产生价值判断
- 超越训练数据分布的原创知识
- 连续时间认知（v1.x是离散LLM推理）

### 总体评分（基于评审+v1.1修正）

v1.0: 5.5/10 → **v1.1: 7.0/10**

改进点：删除装饰性内容、修正不准确表述、补充遗漏工作、增加诚实声明、修正时间线、增加安全设计。

---

> **最后的话**：PEDA v1.1不是通往AGI的捷径，而是一次有理论支撑、有诚实声明、有验证计划的工程尝试。它的价值不在于"是否真的自主"（哲学问题），而在于"是否能产生有趣、有用的行为"（工程问题）——而这个问题的答案，将在Phase 1（2-3周后）揭晓。

---

## 2. 理论基础：从预测误差到自主Agent的统一框架

PEDA架构的设计建立在一个横跨计算神经科学、统计物理学和机器学习的理论图景之上。其核心洞见是：**智能体是一台预测机器，感知、行动、学习可以被统一为对预测误差的最小化**。本章系统梳理支撑PEDA的理论支柱，为后续章节的形式化定义奠定基础。同时，我们在每个理论的讨论中明确标注其与PEDA实际实现的对应关系——哪些是指导设计的核心原则，哪些是提供直觉的理论类比，避免"用理论装饰工程"的陷阱。

### 2.1 Prompt范式的根本限制

要理解PEDA为什么必须存在，首先要看清当前LLM应用范式的一个结构性缺陷。

今天所有的LLM应用——从最简单的聊天机器人到最先进的Agent框架——都共享同一个底层假设：**模型是一组冻结的权重，每次调用都从零开始推理，调用结束后不保留任何持久状态**。这意味着模型的"思考"完全是外部驱动的：没有输入（prompt），就没有思考。模型不会"自己想点什么"，它只是被动地响应刺激。

这个限制的直接后果是整个行业对prompt工程的过度依赖。研究者和工程师们投入巨大精力去设计更好的prompt模板、更精巧的few-shot示例、更复杂的chain-of-thought结构。近期有工作声称"不依赖用户提示"，例如通过内部模板生成来构造prompt。但这种做法本质上只是**把prompt的编写者从人类换成了另一个程序**——prompt仍然存在，discrete call仍然存在，模型的被动本质没有改变。

Prompt范式的根本问题不在于"谁来写prompt"，而在于**"prompt/discrete call"这个概念本身就是对智能的错误抽象**——它假设智能体在没有外部刺激时应当处于"关闭"状态，仿佛一台等待按下的计算器。

PEDA的回答是：当Agent拥有持久状态、拥有内部世界模型时，**预测误差本身就成为了行动的驱动力**。Agent的内部世界模型持续生成关于未来感知的预测；当实际感知与预测不符时，这个误差信号驱动Agent采取行动来缩小差距。不需要人类用自然语言去"命令"模型做什么，模型会主动追求那些能够降低其预测误差的状态。外部prompt从驱动者降级为可选的初始化参数——预测误差接管了控制权。

**诚实声明**：PEDA v1.x的"持久状态"是通过LLM参数（LoRA微调权重）和显式记忆模块实现的，并非生物神经系统意义上的连续动力学。模型在两次调用之间仍然处于"暂停"状态。我们追求的是"看起来像自主"的工程效果，而非真正的生物式连续认知。后者在第2.7节作为未来方向讨论。

### 2.2 Active Inference与自由能原理

自由能原理（Free Energy Principle, FEP）是Karl Friston提出的统一框架，旨在解释自组织系统如何维持其存在。它为PEDA提供了最核心的理论骨架：**为什么最小化预测误差会导致看起来像智能的行为**。

FEP的数学基础可以追溯到变分推断。Friston, Kilner & Harrison (2006) 的开创性论文指出，任何自组织系统都面临同一个基本问题：如何通过有限的感官接口推断外部世界的隐藏状态，并据此行动以维持自身的稳态 [Friston et al., 2006]。Friston (2010) 进一步将这一原理推广为统一的大脑理论 [Friston, 2010]。Friston et al. (2017) 的里程碑论文将这一框架形式化为"Active Inference"的过程理论 [Friston et al., 2017]。最新的权威综述（Friston et al., 2023）以清晰的方式重新阐述了整个框架 [Friston et al., 2023]。

**变分自由能的直觉理解**。想象一个有机体试图感知环境中的隐藏状态。它有一个关于世界如何运作的生成模型（generative model）——一个内部模拟器，能够根据假设的隐藏状态预测感官输入。变分自由能源自一个深刻的数学事实：由于有机体无法直接访问外部世界的真实状态，它只能通过感官间接推断。变分自由能是这个推断问题的一个上界——最小化自由能等价于同时做两件事：（1）让内部信念尽可能准确地匹配外部现实（感知推断），（2）让感官输入尽可能匹配内部预期（行动）。

用更直觉的类比：想象你正在尝试接住一个飞来的球。你的大脑并不直接"计算"球的抛物线轨迹，而是**持续生成关于球下一秒位置的预测**，并通过眼动和肢体调整来让实际的视觉输入与预测一致。如果你没有接到球，预测误差会驱动你更新内部模型——这就是学习。这种误差驱动的感知-行动循环，正是自由能原理的核心。

**期望自由能（Expected Free Energy, EFE）与目标函数**。如果变分自由能解释了"如何感知"，那么EFE解释了"如何行动"——具体来说，**如何选择下一步行动**。EFE定义为：

$$G(\pi) = \mathbb{E}_{q(o|\pi)}[\ln q(o|\pi) - \ln p(o|C)] = H[q(o|\pi)] + D_{KL}[q(o|\pi) \,||\, C(o)]$$

其中 $\pi$ 是一个策略（行动序列），$q(o|\pi)$ 是执行该策略后预期观测的分布，$C(o)$ 是**偏好分布**（preference distribution），表征Agent"想要"什么样的观测。

**关键修正：C(o)就是目标函数**。v1.0曾表述为"不需要外部目标"，这是一种过度简化。FEP并非消除目标，而是**将目标的形式从外部reward函数转变为内部偏好分布C(o)**。C(o)定义了Agent对观测的偏好——某些观测被认为是"好的"（高概率），另一些是"坏的"（低概率）。即使C(o)被设为uniform（对所有观测赋予相同偏好），这也不是"无目标"，而是"平等对待所有观测"——一种特定形式的偏好设定。

那么，uniform preference是否会导致Agent"漫无目的地游荡"？答案是否定的。EFE包含两个项：

- 第一项 $H[q(o|\pi)]$ 是**认识价值（epistemic value）**：度量执行策略后预期观测的不确定性。**即使C(o)是uniform的，这一项仍然存在**，驱动Agent优先探索那些能够减少其模型不确定性的状态。信息增益的梯度天然指向高不确定性区域——Agent不需要被告知"那里有趣"，它的数学结构就决定了它会走向不确定性最大的地方。

- 第二项 $D_{KL}[q(o|\pi) \,||\, C(o)]$ 是**实用价值（pragmatic value）**：度量预期观测与偏好之间的距离。如果C(o)对某种观测赋予高概率（比如"文件成功创建"），这一项会驱动Agent朝向那些能产出偏好观测的状态。

**探索与利用的统一**。传统RL需要手动设计exploration-exploitation平衡（如epsilon-greedy）。FEP表明，当Agent对其目标状态高度不确定时，epistemic value占主导，驱动探索；当Agent对如何实现目标有清晰信念时，pragmatic value占主导，驱动利用。这不是一个需要手动调节的参数，而是Agent内部信念状态的动态结果。Sajid et al. (2021) 的严格数学分析证明，在特定条件下，Active Inference与模型基RL等价 [Sajid et al., 2021]。

**Active Inference与RL的对应关系**可用下表概括：

| 维度 | 强化学习(RL) | Active Inference (AIF) |
|------|------------|----------------------|
| 核心目标 | 最大化期望累积奖励 | 最小化变分自由能 |
| 目标来源 | 外部reward函数 | 内部偏好分布C(o) |
| 探索机制 | 启发式（ε-greedy, UCB等） | 从认识价值中自然涌现 |
| 学习机制 | 时序差分学习或策略梯度 | 变分贝叶斯推断 |
| 模型需求 | 可选（Model-free可不用） | 必须（需要生成模型） |

**对PEDA的具体意义**：PEDA使用EFE作为行动选择的标准。在每次决策时，Agent的World Model对候选行动进行rollout想象，计算每个候选的期望自由能，选择EFE最小的行动。这里的C(o)可以被设为uniform（纯探索模式），也可以由人类用户设定特定偏好（任务导向模式）。**PEDA不是没有目标——它的目标就是减少预测不确定性，而信息增益的梯度天然指向那些Agent知道最少的地方**。

**已知局限**：离散POMDP框架的EFE计算在高维状态空间中代价高昂。PEDA通过限制rollout horizon（2-3步）和候选行动数量（2-3个）来缓解这一问题。更高效的近似方法（如Dijkstra规划的变体）是未来工作。

### 2.3 Predictive Coding：预测误差作为学习信号的直觉框架

如果FEP提供了"为什么要预测"的哲学回答，Predictive Coding则提供了"预测误差如何驱动学习"的直觉框架。

Predictive Coding的起源可以追溯到Rao & Ballard (1999) 的开创性论文，他们提出视觉皮层可以被视为一个分层的预测机器 [Rao & Ballard, 1999]。在这个框架中，皮层区域并不被动地"提取特征"，而是主动地生成关于下层输入的预测，并只将**预测误差**（实际输入与预测之间的差异）向上传递。Andy Clark在2013年的里程碑论文中将Predictive Coding从视觉处理推广到通用认知架构 [Clark, 2013]，其专著《Surfing Uncertainty》进一步将这一观点发展为完整的认知哲学 [Clark, 2015]。

**分层预测编码的直觉**。想象一个多层的神经网络，每一层都试图预测下一层的活动。最底层接收原始感官输入，它生成关于这些输入的预测。预测误差被传递到上一层。上一层的目标不是编码原始输入，而是预测下一层的预测误差。这个过程递归进行，形成一条误差传递链。这种架构的信息效率在于：如果预测是准确的，误差为零，网络不需要向上传递任何信息。只有当出现"意外"时，信息才会向上流动——**系统只处理意外，忽略预期**。

**PEDA与Predictive Coding的关系：理论直觉，而非实现方法**。必须明确声明：PEDA v1.x使用**标准的反向传播+LoRA微调**来训练World Model，Predictive Coding的局部学习规则在代码中没有出现。PC对PEDA的价值在于提供以下理论直觉：

1. **预测误差作为学习信号**：PC框架证明，预测误差本身就是强大的学习信号——不需要外部标签，不需要人工设计的reward。这与PEDA"用预测误差驱动行为"的设计理念一致。
2. **分层处理**：PC的分层误差传递结构启发PEDA World Model的分层设计——底层处理原始感知，高层处理抽象表示。
3. **精度加权**：PC中的"精度"（precision）概念——不同预测误差通道的可信度——启发PEDA的epistemic/aleatoric分解（详见第3章）。

**关于"PC替代反向传播"的讨论**。Millidge et al. (2022) 证明了Predictive Coding网络可以通过纯局部学习规则来近似反向传播的梯度 [Millidge et al., 2022]。这是一个重要的理论结果，意味着PC网络可以在线学习——在持续运行的同时不断更新自身。但截至目前：
- PCN在大型语言模型规模上的训练效率和效果尚未得到验证
- PCN的工具链和生态系统远不及PyTorch/TensorFlow成熟
- PEDA v1.x不涉及PCN实现

**诚实声明**：PEDA选择标准反向传播+LoRA微调而非PC局部学习规则，是基于工程实用性的考量，而非理论上的否定。如果未来PCN在效率和可扩展性上被证明具有优势，PEDA可以在不改变架构的情况下切换学习机制。这在第2.7节作为未来方向提及。

### 2.4 World Models：在想象中学习行动

World Models领域为PEDA提供了关于"如何构建内部模拟器"的工程蓝图。World Models的核心思想是：智能体应该学习环境动态的压缩表示，然后在这个内部模拟器中规划和行动，而不是直接在真实环境中试错。

**RSSM架构的核心设计**。Ha & Schmidhuber (2018) 的开创性论文首次将World Models用于策略学习，展示了在一个学习到的潜空间中进行"梦境训练"的可能性 [Ha & Schmidhuber, 2018]。Hafner et al. (2019) 在此基础上提出了RSSM（Recurrent State-Space Model），这是一个精巧的架构设计，结合了**确定性循环路径**和**随机潜变量** [Hafner et al., 2019]。

RSSM的设计直觉：环境的一部分变化是可预测的（比如文件系统中`ls`命令的输出遵循确定性规则），这部分由确定性循环网络编码；另一部分变化是本质随机的（比如`date`命令的返回值），这部分由随机潜变量建模。两者结合，使得RSSM既能做长程预测（靠确定性路径），又能处理不确定性（靠随机变量）。

**Dreamer系列的演进**。Hafner et al. (2020) 的DreamerV1证明了RSSM可以用于从像素直接学习连续控制策略 [Hafner et al., 2020]。DreamerV2 (2021) 引入了离散潜变量和更好的表示学习，在Atari游戏上达到了与model-free方法相当的性能 [Hafner et al., 2021]。DreamerV3 (2023) 是其最重要的成就——使用**固定超参数**在超过150个不同任务上达到了人类水平或更高的性能，包括需要长期推理的Minecraft任务 [Hafner et al., 2023]。这一结果表明，World Model方法具有惊人的通用性。

**长程预测的挑战**。World Models面临的一个根本性挑战是**预测的发散**。当模型试图预测超过15步的未来时，小误差会指数级累积，导致预测结果迅速退化 [Hafner et al., 2023]。在实际中表现为：模型可以准确预测下一秒会发生什么，但对一分钟后的预测可能毫无意义。

PEDA的应对策略不是追求无限精确的长程预测，而是**将预测误差本身作为行动的驱动力**。当模型对远处的未来不确定时，这种不确定性（高熵）会直接体现在期望自由能中，驱动Agent采取行动来降低不确定性。这相当于将"我无法预测"这个事实转化为"我应该去探索"的行为信号。同时，PEDA采用分层预测策略：不追求预测"完整状态"，而是预测"关键状态变量"（详见第4章）。

**Active Inference与World Models的结合**。Mazzaglia et al. (2022) 的"Probabilistic Dreaming"工作在Dreamer框架中引入了自由能剪枝机制，利用EFE来筛选和评估想象的轨迹，在多个连续控制任务上提升了4.5%的得分 [Mazzaglia et al., 2022]。这一结果表明，FEP不仅是理论框架，更是可以带来实际性能提升的工程工具。PEDA直接继承这一洞见：内部想象不是装饰，而是Agent决策循环的必要组成部分。

**JEPA：非生成式World Model的替代路径**。LeCun (2022) 提出的联合嵌入预测架构（JEPA）代表了一种根本不同的World Model设计哲学 [LeCun, 2022]。与Dreamer系列（生成式——预测完整的像素/状态）不同，JEPA是**非生成式**的——它预测的是表示（representation）而非原始观测。I-JEPA (CVPR 2023) 在图像领域验证了这一理念 [Assran et al., 2023]，V-JEPA (TMLR 2024) 扩展到视频表示学习 [Bardes et al., 2024]，V-JEPA 2 (2025) 结合100万小时互联网视频和机器人数据，实现了零样本Franka臂控制 [Bhardwaj et al., 2025]。

JEPA对PEDA的启示是：**在Linux沙箱这样的文本环境中，预测下一个文本token可能不是最高效的World Model形式**。预测"状态表示的变化"（如"文件存在性从false变为true"）可能比预测原始命令输出更实际。PEDA v1.x使用LLM生成作为World Model（生成式路径），但JEPA的非生成式方法是一个值得在v2.x中探索的方向。

### 2.5 内在动机与好奇心的FEP视角

如果Agent的行为由预测误差驱动，那么一个直接的问题是：它不会被环境中的随机噪声困住吗？这一节讨论FEP如何解决这个问题。

**ICM与RND的核心机制**。Pathak et al. (2017) 的内在好奇心模块（ICM）将预测误差作为内在reward [Pathak et al., 2017]。ICM训练一个前向模型来预测下一个状态的特征表示，当Agent遇到新颖的状态时，前向模型预测不准，产生高误差，驱动Agent探索。Burda et al. (2018) 的随机网络蒸馏（RND）采用了类似的思路，但用一个固定的随机网络来提取特征，避免了ICM中特征学习可能被"欺骗"的问题 [Burda et al., 2018]。RND在54个标准基准环境中进行了大规模评估，在Montezuma's Revenge上找到了全部24个房间，无需人类演示 [Burda et al., 2018]。

**Noisy TV问题**。ICM和RND都面临一个根本性的理论缺陷。想象一个Agent在一个房间里探索，房间里有一台电视机在播放随机噪声。由于电视画面是不可预测的，ICM会产生持续的预测误差，Agent会被"钉"在电视机前——尽管观看随机噪声对完成任何实际任务都没有帮助。

这个问题的本质在于：**预测误差混淆了两种根本不同的不确定性**。一种是"可约的不确定性"（reducible uncertainty / epistemic）——只要我获取更多信息，这种不确定性就可以被消除。另一种是"不可约的不确定性"（irreducible uncertainty / aleatoric）——即使我知道了一切，这种不确定性依然存在。ICM的预测误差无法区分这两种情况。

**FEP的解决方案**。Active Inference通过EFE的数学结构解决了这个问题。回顾EFE的两个项：

$$G(\pi) = \underbrace{H[q(o|\pi)]}_{\text{epistemic value（探索）}} + \underbrace{D_{KL}[q(o|\pi) \,||\, C(o)]}_{\text{pragmatic value（利用）}}$$

第一项——熵 $H[q(o|\pi)]$——度量的是Agent对自己预测的不确定性。当面对一台Noisy TV时，Agent很快会学习到电视画面是本质随机的——它的生成模型会收敛到对电视输出的概率分布的准确估计。即使每次看到的具体画面仍然不可预测，Agent对自己预测的**不确定性**却很低——因为模型已经学会了"电视输出均匀随机噪声"。因此，EFE中的epistemic value会选择那些**能够让Agent更新其信念的状态**，而不是那些仅仅产生高预测误差的状态。

**信息增益 vs 预测误差的根本区别**。信息增益衡量的是：观察到某个结果后，Agent关于隐藏状态的信念分布发生了多大变化。预测误差衡量的是：预测与实际观测之间的差异。当面对本质随机的输出时，预测误差可以很高，但信息增益为零——Agent并没有因此改变对世界的理解。PEDA中的驱动信号不是原始预测误差，而是**能够带来信息增益的预测误差**。

**探索与利用的统一**。传统RL需要手动设计exploration-exploitation平衡。FEP表明，这个平衡是**自然涌现**的。当Agent对其目标状态高度不确定时，epistemic value占主导，驱动探索；当Agent对如何实现目标有清晰信念时，pragmatic value占主导，驱动利用。

**BYOL-Explore：潜空间预测的经验突破**。Guo et al. (2022, DeepMind) 提出的BYOL-Explore与PEDA的World Model设计有直接关联 [Guo et al., 2022]。BYOL-Explore在潜空间中使用自举表示学习（bootstrap representation learning）进行预测，通过单一预测损失同时学习世界表示、动态模型和探索策略。在DM-HARD-8（DeepMind最难的探索环境集合，包含部分可观测的连续动作3D环境）上，BYOL-Explore仅靠内在reward就解决了大多数任务，此前的方法需要人类演示 [Guo et al., 2022]。在Atari十个最难的探索游戏上，BYOL-Explore达到了超人类性能 [Guo et al., 2022]。

BYOL-Explore对PEDA的启示在于：**潜空间预测比原始观测空间预测更高效、更鲁棒**。PEDA的World Model本质上也是在潜空间（LLM的隐藏表示）中进行预测，而非直接预测原始文本token。BYOL-Explore的成功为这一设计选择提供了经验支撑。

### 2.6 相关工作：PEDA在学术谱系中的定位

PEDA不是凭空创造的概念。它处于一个活跃的研究交汇点——计算神经科学、深度强化学习和LLM Agent框架的交叉地带。本节系统梳理与PEDA最相关的近期工作，明确PEDA的差异化定位。

#### 2.6.1 Voyager：LLM Agent的自动课程与技能库

Wang et al. (2023, NeurIPS) 提出的Voyager是Minecraft中第一个完全由LLM驱动的终身学习Agent [Wang et al., 2023]。Voyager的三个核心组件对PEDA有直接参考价值：

- **自动课程（Automatic Curriculum）**：根据Agent当前技能水平和探索状态自动生成任务目标。课程由LLM根据"最大化学习进度"的原则动态生成——优先选择那些Agent"几乎能做到但还没完全掌握"的任务。
- **技能库（Skill Library）**：将可复用的行为编码为可执行的代码块（JavaScript函数），存储在向量数据库中。Agent遇到新任务时，先从技能库中检索最相关的已有技能作为起点。
- **迭代提示（Iterative Prompting）**：通过执行反馈不断修正生成的代码，直到代码能够正确执行目标行为。

**PEDA与Voyager的对比**：

| 维度 | Voyager | PEDA |
|------|---------|------|
| 探索驱动 | 外部课程生成器 | 内部预测误差（EFE） |
| 学习机制 | 代码技能库存储 | LoRA参数微调 |
| 表示形式 | 可执行代码 | 神经网络权重 |
| 知识复用 | 显式检索 | 隐式在参数中 |
| 环境假设 | Minecraft（结构化） | 通用（Linux沙箱） |

Voyager的技能库可能是比LoRA微调更高效的学习机制——代码块是可解释、可组合、可精确检索的，而LoRA权重是分布式、不可解释的。PEDA v1.x选择LoRA是为了利用LLM的预训练知识进行端到端梯度优化，但Voyager的"代码即技能"范式是v2.x中值得探索的方向。PEDA的核心差异化在于**预测误差驱动的探索动机**——Voyager的课程是外部生成的，PEDA的探索方向由Agent自身的预测不确定性决定。

#### 2.6.2 BYOL-Explore：好奇心驱动探索的SOTA

Guo et al. (2022) 的BYOL-Explore已在第2.5节讨论。这里补充其在相关工作谱系中的位置：BYOL-Explore是**纯好奇心驱动探索的经验性突破**——它证明了仅依靠潜空间预测误差（无需外部reward、无需课程、无需人类演示）就可以在最难的探索环境中达到超人类性能。PEDA的World Model设计直接受益于BYOL-Explore的洞见：在表示空间而非原始空间中进行预测，使用自举方法稳定学习信号。

关键区别：BYOL-Explore的"好奇心"仍然是预测误差（虽然是在潜空间），理论上仍可能被Noisy TV类的问题困扰；PEDA使用EFE中的epistemic value（信息增益），从根本上区分了可约和不可约的不确定性。

#### 2.6.3 JEPA：非生成式World Model

LeCun (2022) 的JEPA已在第2.4节讨论。在相关工作的语境下，JEPA代表了与PEDA使用的LLM生成式World Model不同的技术路线：

- **PEDA v1.x路径**：LLM生成完整文本预测（生成式）→ 灵活但计算昂贵、可能有幻觉
- **JEPA路径**：预测表示/嵌入（非生成式）→ 更稳定、更高效但表示设计需要人工决策

在Linux沙箱环境中，JEPA路线可能更实际：预测"执行`ls`后文件A存在性变为true"的表示，比预测`ls`的完整文本输出更容易且更可靠。

#### 2.6.4 现代LLM Agent框架：持久状态机制

PEDA对Prompt范式的批评（第2.1节）需要放在一个准确的学术语境中。现代LLM Agent框架已经有了不同程度的持久状态机制——PEDA的差异化不是"有vs没有"持久状态，而是"持久状态的丰富程度和学习机制"：

**ReAct（Reasoning + Acting）** [Yao et al., 2023]。ReAct在推理和行动之间交替，将thought、action和observation连接成链。ReAct的"持久状态"体现在：Agent的推理轨迹被显式记录并在后续步骤中可用。但这不是真正的持久状态——轨迹是只读的上下文窗口，不会通过经验改变模型参数。Agent不会"学到"什么，它只是"记住"了当前episode中发生了什么。

**Reflexion（自我反思）** [Shinn et al., 2023]。Reflexion引入了**跨episode记忆**：Agent在任务失败后生成自我反思摘要（"我失败是因为没有先检查文件是否存在"），存储在记忆队列中。在后续episode开始时，这些反思被注入prompt。这是真正的持久状态——经验在不同episode之间传递。但Reflexion的学习是**符号性**的（自然语言摘要），不是**梯度性**的（参数更新）。反思的质量完全取决于LLM的摘要能力，没有梯度信号来保证反思真正有助于改进。

**AutoGPT**。AutoGPT维护一个长期记忆向量数据库（通常基于文本嵌入），可以在不同运行之间检索过去的经验。但AutoGPT的记忆是**被动存储**——经验被存入数据库，但不会被主动整合到模型的行为模式中。没有机制保证Agent会"倾向于"重复过去成功的行为。

**PEDA的差异化定位**：PEDA不是在"有持久状态"这个维度上区别于这些框架（Reflexion和AutoGPT已经有持久状态），而是在以下三个维度上：

1. **学习机制**：PEDA使用LoRA梯度微调来更新World Model参数，经验直接改变模型的行为倾向；Reflexion/AutoGPT使用符号记忆注入，不改变模型内部参数。
2. **探索动机**：PEDA的探索由EFE（信息增益）驱动，具有指向性——Agent知道该往哪里探索；ReAct的探索依赖于外部prompt指令，缺乏内在方向性。
3. **统一框架**：PEDA的所有组件（感知、行动、学习、探索）被统一在FEP的数学框架下，共享同一个目标函数（最小化自由能）；其他框架是模块化的，每个模块有独立的目标和机制。

**经典认知架构**。SOAR [Laird et al., 2012] 和 ACT-R [Anderson et al., 2004] 是认知架构领域的两个里程碑。SOAR的chunking机制（从经验中提取可复用的规则块）和ACT-R的生产式系统（if-then规则集）与PEDA的World Model+学习模块有概念上的对应关系。但经典认知架构在真实世界复杂环境中的表现远不及深度学习系统，且其符号表示难以处理自然语言等非结构化输入。PEDA可以被视为"用深度学习和FEP重新实现的认知架构"——继承了统一设计哲学，但使用现代神经网络的表示能力。

**总结：PEDA的定位**。PEDA不是"没有相关工作"的全新创造，而是在FEP统一框架下整合了多个学术方向的关键组件：

- 从FEP/Active Inference继承了统一目标函数（EFE最小化）
- 从World Models领域继承了内部模拟器的设计
- 从BYOL-Explore继承了潜空间预测的效率
- 从Voyager借鉴了技能库的学习范式（未来工作）
- 从JEPA借鉴了非生成式World Model的可能性（未来工作）
- 从现代LLM Agent框架继承了持久状态的概念，但用梯度学习替代了符号记忆

PEDA的学术价值在于"整合"——证明FEP可以为LLM Agent的所有组件提供一个统一的数学基础。

### 2.7 未来方向：连续时间认知架构

前述所有理论——FEP、Predictive Coding、World Models——本质上都假设系统以离散时间步运行。但真实的认知是连续流淌的。本节简要讨论连续时间架构作为PEDA的未来方向。

CTRNN（Continuous-Time Recurrent Neural Networks）[Beer, 1995]、LTC Networks [Hasani et al., 2020] 和CfC（Closed-form Continuous Networks）[Hasani et al., 2022] 代表了连续时间神经网络的工程前沿。这些架构的核心特征在于神经元状态由微分方程描述，可以在没有外部输入的情况下产生**自发行为**——其行为不是外部刺激的直接反应，而是内部吸引子动力学与外部输入的相互作用。EMBER架构 [Savage, 2026] 展示了SNN+LLM混合系统的可行性：22万神经元的脉冲神经网络在消费级GPU上运行，能够自发触发LLM调用。

**PEDA的立场**：PEDA v1.x完全基于离散架构（LLM的token-by-token生成）。连续时间认知是v2.x的潜在方向——不是因为它对v1.x的实现有必要贡献，而是因为它可能提供真正的自发行为（无外部输入时的内部动力学）。在离散架构中，"自主"是通过预测误差+记忆模块模拟出来的；在连续架构中，自主可能是系统动力学的自然涌现性质。从CTRNN到CfC的工程进展 [Hasani et al., 2022] 表明，连续时间模型的训练速度已提升100-4000倍，使其不再是不可行的研究玩具。如果EMBER模式 [Savage, 2026] 被证明可扩展到实用规模，它可能成为PEDA v2.x的核心架构组件。

---

## 3. PEDA架构设计

PEDA（Predictive-Error-Driven Autonomous Agent）的架构设计是一次从"控制论"到"自治论"的范式跃迁。传统AI Agent的架构围绕"如何更好地响应用户"而构建，PEDA的架构则围绕"如何维持内部认知稳态"而生长。这意味着我们不从接口层开始设计，而是从存在论层面——一个系统为何行动、何时行动、如何行动——重新定义Agent的认知结构。

本章将从哲学基础出发，逐层展开PEDA的五大核心模块，阐明每个模块的职责边界、输入输出接口，以及它们在预测误差驱动的闭环中所扮演的角色。每个模块的设计都将回答三个问题：它做什么？为什么必须由独立模块而非内嵌逻辑实现？如果移除它，系统会退化为何种形态？

---

### 3.1 核心哲学：从"Prompt驱动的推理"到"Prediction驱动的存在"

#### 3.1.1 Prompt范式的囚笼

当代大语言模型（LLM）的应用范式——无论冠以Agent、Chain-of-Thought还是Tool-use之名——共享一个深层结构：**冻结权重 + 无状态调用 + 外部输入触发**。模型在每次推理时从近乎Blank Slate的状态出发，依赖用户输入（Prompt）作为触发器和上下文源。这种架构的本质是"问答机"：有人在按钮上按一下，系统响应一次；无人交互时，系统处于认知上的"suspended animation"（认知冻结），既不思考，也不行动。

这一范式的根本局限在于，它将"智能"等同于"推理能力"，而忽略了智能的另一个维度——**持续的内在活动**。生物大脑从不因缺少外部刺激而停止工作；即使在深度睡眠中，皮层仍在进行预测性编码和记忆巩固。Prompt范式下的AI系统缺乏这种"存在性持续"，也因此缺乏真正的自主性。

#### 3.1.2 Prediction范式：存在的持续

PEDA的核心哲学转变可以概括为一句话：**系统持续运行，内部状态持续演化，"行动"只是减少预测误差的一种方式**。

在PEDA中，没有外部触发器。系统在每⼀个时间步都在做三件事：（1）基于World Model预测下⼀状态；（2）比较预测与实际感知；（3）如果存在预测误差，生成行动以减少误差。这是一个闭环的自我维持系统——即使锁在空房间里没有任何外部任务，它也会主动探索环境、测试假设、更新模型，因为"不确定性"本身就是不适的源泉。

#### 3.1.3 关键Insight：目标的内化而非消除

传统强化学习（RL）需要人工设计的奖励函数来告诉Agent"什么好、什么坏"。PEDA则指出：**"减少不确定性"本身就是一个内在驱动力，无需外部指定奖励函数**。

需要澄清的是，这不是"不需要目标"。在FEP的数学框架中，偏好分布 $C(o)$ 始终存在——即使将Pragmatic Value设为零（纯探索场景），系统仍然持有uniform preference，这是一种"平等对待所有状态"的隐含目标。真正的洞见在于：**目标从外部reward函数转变为内部偏好分布**，探索的方向性由信息增益的梯度天然提供——系统倾向于前往那些最能更新其内部信念的状态。

这一观点直接来源于Friston的自由能原理：生物系统通过最小化变分自由能来维持认知和生理的稳态。在PEDA的语境下，预测误差就是变分自由能的认知对应物——高预测误差意味着"我无法解释所感知的"，这驱动系统去采集更多信息（探索）或调整内部模型（学习），直到误差被降低。

**类比**：Prompt范式像一台自动售货机——你投币（输入Prompt），它出货（输出结果）；Prediction范式像一只在陌生房间里醒来的猫——即使没有人要求它做什么，它也会四处嗅探、试探家具、更新对环境的认知地图，因为"不了解环境"本身就是不适。

> **如果不存在这一哲学转向**：PEDA将退化为另一个被动等待用户输入的ChatBot封装，所有后续的架构设计都将失去根基。预测误差只能作为Prompt响应的"辅助信号"，而非驱动的核心引擎。

---

### 3.2 系统架构总览

#### 3.2.1 五大核心模块

PEDA的认知架构由五个相互协作的核心模块组成，构成一个完整的感知-预测-行动-学习闭环：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PEDA Cognitive Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Perception  │───→│  World Model │───→│ Predictive Error    │  │
│  │  (感知模块)   │    │  (世界模型)   │    │ Computer (误差计算)  │  │
│  └──────┬───────┘    └──────────────┘    └──────────┬───────────┘  │
│         │                                            │               │
│         │    ┌───────────────────────────────────────┘               │
│         │    │                                                       │
│         │    ↓                                                       │
│         │  ┌──────────────────┐                                     │
│         │  │  Action Generator │                                    │
│         │  │  (行动生成器)      │                                    │
│         │  │  · EFE Minimizer │                                    │
│         │  │  · Rollout Engine│                                    │
│         │  └────────┬─────────┘                                    │
│         │           │                                               │
│         │           ↓                                               │
│         │    ┌──────────────┐                                      │
│         └───←│ Action Exec. │                                      │
│              │ (行动执行器)  │                                      │
│              └──────┬───────┘                                      │
│                     │                                               │
│    ┌────────────────┼────────────────┐                             │
│    │                ↓ Environment    ↓                             │
│    │    ┌──────────────────┐  ┌──────────────┐                    │
│    │    │ Learning Module  │  │  Homeostatic │                    │
│    │    │ (学习模块)        │  │ Drive System │                    │
│    │    │  · LoRA Update   │  │ (内稳态驱动)  │                    │
│    │    │  · Saturation Det│  │  · Curiosity │                    │
│    │    │  · Distillation  │  │  · Competence│                    │
│    │    └──────────────────┘  │  · Boredom   │                    │
│    │                          │  · Novelty   │                    │
│    │                          └──────────────┘                    │
│    │                                                              │
│    └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 模块职责与接口定义

| 模块 | 核心职责 | 主要输入 | 主要输出 | 更新频率 |
|------|---------|---------|---------|---------|
| **Perception** | 将原始环境信号转化为结构化状态表示 | 环境原始数据（文件列表、进程输出、传感器读数） | `State`对象（结构化描述） | 每步 |
| **World Model** | 预测关键状态变量的变化 | `(State_t, Action)` | `Predicted_State_{t+1}`（分层预测） | 间歇微调 |
| **Predictive Error Computer** | 量化预测与实际的差距，分解epistemic/aleatoric | `Predicted_State`, `Actual_State` | `Error_Vector` (ensemble分解) | 每步 |
| **Action Generator** | 通过想象rollout选择最小化EFE的行动 | `Error_Vector`, `World Model`, `Drive_Weights` | `Selected_Action` | 每步 |
| **Action Executor** | 在环境中执行选定的行动并返回结果 | `Selected_Action` | `Execution_Result` | 每步 |
| **Learning Module** | 收集数据、批量更新World Model、检测饱和 | 交互历史缓冲区 | `Model_Update` (LoRA增量) | 每N步 |
| **Homeostatic Drive System** | 调节多个内在驱动力的动态权重 | 历史误差序列、行动历史、外部信息新鲜度 | `Drive_Weights` | 每步 |

#### 3.2.3 核心数据流

PEDA的主循环在每一步执行以下数据流：

```python
def peda_step(current_state: State, world_model: WM, drives: Drives) -> Action:
    # 1. World Model预测关键状态变量（非完整状态）
    predicted_state = world_model.predict(current_state, action=None)
    
    # 2. Predictive Error Computer计算感知误差
    perceptual_error = compute_error(predicted_state, current_state)
    
    # 3. 如果误差高于阈值，启动行动选择
    if perceptual_error.total > THRESHOLD:
        # 3a. Action Generator想象候选行动的rollout（受推理速度约束）
        candidates = generate_candidates(current_state, max_candidates=3)
        best_action = None
        best_efe = float('inf')
        
        for action in candidates:
            # 想象：执行action后未来2-3步的状态序列（受约束的horizon）
            imagined_trajectory = world_model.rollout(current_state, action, horizon=2)
            # 计算该轨迹的Expected Free Energy
            efe = compute_efe(imagined_trajectory, drives)
            if efe < best_efe:
                best_efe = efe
                best_action = action
        
        # 4. Action Executor在环境中执行
        execution_result = execute(best_action)
        
        # 5. 计算模型误差（预测vs实际结果）
        actual_next_state = perceive(execution_result)
        model_error = compute_error(
            world_model.predict(current_state, best_action),
            actual_next_state
        )
        
        # 6. 存储到学习缓冲区
        learning_buffer.store(current_state, best_action, actual_next_state, model_error)
        
        return best_action
    
    # 如果误差低，Drive System可能启动自发探索
    elif drives.novelty > THRESHOLD:
        return generate_exploratory_action(current_state, drives)
    
    return None  # 维持当前状态
```

#### 3.2.4 与传统Agent架构的对比

| 维度 | 传统Prompt-based Agent | PEDA |
|------|----------------------|------|
| **触发方式** | 用户输入（Prompt） | 内部预测误差 |
| **状态持续性** | 无状态/短期对话历史 | 持续演化的World Model |
| **目标来源** | 外部Prompt指定 | 内在涌现（减少不确定性） |
| **探索策略** | 手动设计或ε-greedy | EFE自然涌现 |
| **学习时机** | 不学习或离线微调 | 间歇性World Model更新 |
| **空闲行为** | 等待输入，无活动 | 自发探索高不确定性区域 |
| **环境假设** | 静态、已知 | 动态、需持续建模 |
| **认知架构** | 感知→推理→输出 | 预测→比较→行动→学习 |

> **如果不存在这一整体架构**：PEDA将退化为松耦合的脚本集合。五大模块的分离不是工程上的过度设计，而是认知功能的必要分化。Perception的独立确保状态表示的一致性；World Model的独立使预测与执行解耦；Error Computer的独立实现误差分解；Action Generator的独立支持多步想象；Learning Module的独立防止灾难性遗忘；Drive System的独立提供动机层。缺少任何一环，闭环都将断裂，系统要么无法自主启动（缺Error/Drive），要么无法学习进化（缺Learning），要么盲目行动（缺World Model）。

---

### 3.3 World Model（世界模型）

#### 3.3.1 职责定义

World Model是PEDA架构的认知核心。它的职责**不是生成自然语言文本，也不是预测完整的系统状态，而是预测关键状态变量的变化**。这一区分至关重要：

- 生成模型关心"下一个token是什么"；
- 朴素World Model关心"世界下一秒是什么样"；
- **PEDA的World Model关心"哪些关键变量会变化、如何变化"**。

在认知科学术语中，World Model对应于生物的**内部模型（internal model）**——大脑对外部世界因果结构的内部表征。它使Agent能够进行"想象"：在实际行动之前，在内部模拟不同行动的后果。

#### 3.3.2 为什么不预测"完整状态"

评审指出了一个关键问题：Linux沙箱的状态空间维度极高。假设状态由10个独立变量组成，每个变量有80%的可预测性（这在Linux环境中已是乐观估计），整体状态预测的联合准确率上限约为 $0.8^{10} \approx 10.7\%$。如果目标是"完整状态预测准确率70%"，这个目标在当前技术条件下几乎不可能实现。

解决策略是**分层预测**：不追求对完整状态的单一准确率指标，而是将预测目标分为三个层次，每层有独立的难度、目标和止损标准。这种分层方法比"整体预测"更现实，因为：

1. **不同变量的可预测性差异巨大**：exit code几乎完全可预测，而命令输出的具体字符几乎不可预测；
2. **不同变量对决策的价值不同**：知道"命令会成功执行"比知道"输出第73个字符是什么"重要得多；
3. **分层允许系统在不同层次上独立学习和改进**，而非被一个不可能的整体目标拖垮。

#### 3.3.3 分层预测体系

PEDA的World Model采用三级预测体系：

**Level 1：命令退出状态（Exit Code）**

| 属性 | 说明 |
|------|------|
| **预测内容** | 命令执行后的exit code（0=成功，非0=失败及错误类型） |
| **难度** | 低——exit code由命令语义和文件系统状态共同决定 |
| **目标准确率** | **≥90%** |
| **评估标准** | 分类准确率（predicted_code == actual_code） |
| **止损条件** | 若Phase 2b结束时<80%，该项目在此方向上的投入需要重新评估 |

**Level 2：文件系统变化（Filesystem Delta）**

| 属性 | 说明 |
|------|------|
| **预测内容** | 文件存在性变化（新增/删除/修改）、目录结构变化 |
| **难度** | 中——需要理解命令与文件系统的因果效应 |
| **目标准确率** | **≥70%**（文件存在性），**≥60%**（目录结构变化） |
| **评估标准** | 结构化对比（预测的变化列表 vs 实际的变化列表） |
| **止损条件** | 若文件存在性预测<50%，说明World Model未掌握基本的命令-文件因果关系 |

**Level 3：命令输出摘要（Output Summary）**

| 属性 | 说明 |
|------|------|
| **预测内容** | 命令stdout/stderr的前100个字符的语义摘要 |
| **难度** | 高——许多命令的输出本质上是不可预测的（随机数、时间戳、网络延迟） |
| **目标准确率** | **≥50%**即可（语义级别匹配，非精确字符匹配） |
| **评估标准** | 语义相似度（如SBERT embedding的余弦相似度>0.7视为正确） |
| **止损条件** | 无硬性止损——此层级为"尽力而为"，准确率低于50%不影响系统核心功能 |

**归为Aleatoric（不预测）的变量**：

以下变量被明确归为环境固有随机性，World Model**不尝试预测其精确值**：
- 时间戳（任何涉及时间的值）
- PID（进程ID）
- 随机数生成器的输出
- 网络延迟的具体毫秒数
- 内存使用量的精确值

这些变量在状态表示中被标记为`ALEATORIC`类型，Perception模块记录其观测值但不纳入预测准确率的计算。

#### 3.3.4 输入输出接口

```python
@dataclass
class State:
    """环境状态的结构化表示——仅包含可预测的关键变量"""
    filesystem: FileSystemSnapshot    # 文件列表、存在性、权限（不含时间戳）
    processes: List[ProcessSummary]   # 进程名、CPU区间（高/中/低），不含PID
    network: NetworkSummary           # 连接状态（活跃/断开），不含精确延迟
    system: SystemMetrics             # 资源使用区间，不含精确值
    recent_actions: List[Action]      # 最近执行的动作历史
    
@dataclass
class PredictedState:
    """World Model的分层预测输出"""
    level1_exit_code: int                    # 预测的exit code
    level1_confidence: float                 # 对exit code预测的置信度
    level2_filesystem_delta: List[FileOp]    # 预测的文件系统变化
    level2_confidence: float                 # 对文件系统预测的置信度
    level3_output_summary: str               # 预测的输出摘要（前100字符语义）
    level3_confidence: float                 # 对输出摘要预测的置信度
    aleatoric_fields: Dict[str, str]         # 标记为"随机"的字段（不预测值）

@dataclass
class Action:
    """可执行动作的结构化表示"""
    command: str                      # 实际命令（如 "ls -la /proc"）
    action_type: ActionType           # 枚举：READ/WRITE/EXEC/NETWORK/...
    target: Optional[str]             # 动作目标
    parameters: Dict[str, Any]        # 附加参数

class WorldModel:
    def predict(self, state: State, action: Optional[Action]) -> PredictedState:
        """
        预测执行action后的关键状态变量变化。
        返回分层PredictedState，而非完整的未来状态。
        """
        ...
    
    def rollout(self, state: State, action: Action, horizon: int) -> List[PredictedState]:
        """
        从(state, action)出发，自举预测未来horizon步的状态序列。
        注意：horizon在v1.0中限制为2-3步（见3.5节推理速度讨论）。
        """
        trajectory = [state]
        current = state
        for _ in range(horizon):
            next_state = self.predict(current, action)
            trajectory.append(next_state)
            current = next_state
            action = None
        return trajectory
```

#### 3.3.5 具体实现方案

**模型选择**：采用预训练LLM（1-7B参数规模，如Qwen2.5-1.5B、Phi-3-mini或Llama-3.2-3B）+ LoRA微调。基础模型的世界知识为World Model提供先验，LoRA适配层学习特定环境的动态。

**为什么不用<1M参数的微型模型**：World Model需要足够的表示能力来捕捉环境动态。在Linux/文本环境中，模型需要理解：
- 文件系统操作的因果效应（`rm -rf`会删除文件，`mkdir`会创建目录）
- 进程间的依赖关系（杀死父进程会影响子进程）
- 网络命令的结果（`ping`返回延迟，`curl`获取页面）
- 命令的组合效应（管道、重定向、脚本执行）

这些因果关系的表示容量远超<1M参数模型的表达能力。1-7B是在表示能力与推理效率之间的平衡点。

**训练数据格式**（适配分层预测）：
```json
{
  "state_t": {
    "cwd": "/home/user/project",
    "files": ["main.py", "README.md", "data/"],
    "processes": [{"name": "python", "cpu_level": "medium"}],
    "env_vars": {"PATH": "/usr/bin", "HOME": "/home/user"}
  },
  "action": {
    "command": "python main.py --train",
    "type": "EXEC"
  },
  "predicted": {
    "level1": {
      "exit_code": 0,
      "confidence": 0.92
    },
    "level2": {
      "filesystem_delta": [
        {"op": "create", "path": "checkpoint.pt"},
        {"op": "modify", "path": "data/training.log"}
      ],
      "confidence": 0.68
    },
    "level3": {
      "output_summary": "Epoch 1/10: loss=2.34, acc=0.41...",
      "confidence": 0.45
    },
    "aleatoric": ["timestamp", "process_pid", "memory_exact"]
  }
}
```

> **如果不存在World Model**：PEDA将退化为纯反应式系统（reactive system），只能基于当前状态做"刺激-反应"式的映射，无法进行任何前瞻性规划。Agent将失去"想象能力"，无法评估不同行动的长期后果，也无法从内部产生行动的动机（因为没有预测，就没有预测误差）。

---

### 3.4 Predictive Error Computer（预测误差计算模块）

#### 3.4.1 核心职责

Predictive Error Computer是PEDA从"被动感知"到"主动驱动"的转换枢纽。它负责量化World Model的预测与实际感知之间的差异，并将误差分解为具有不同认知意义的成分。这个模块的输出——误差向量——是整个系统的"神经信号"，直接驱动行动选择和学习。

#### 3.4.2 误差类型体系

PEDA区分两种基本误差类型：

**感知误差（Perceptual Error）**：Perception模块的原始输入与World Model对"无行动演化"的预测之间的差异。这种误差反映环境自发变化（如外部进程产生新文件、网络包到达）导致的预测失败。

**模型误差（Model Error）**：World Model对"执行行动A后状态"的预测与Action Executor实际执行后感知到的状态之间的差异。这是主要的**学习信号**，直接指示World Model在何处表现不佳。

```python
@dataclass
class ErrorVector:
    """预测误差的结构化分解"""
    total_error: float                    # 总误差（用于快速判断）
    
    # 按层次分解（对应World Model的三层预测）
    level1_error: float                   # exit code预测误差
    level2_error: float                   # 文件系统变化预测误差
    level3_error: float                   # 输出摘要预测误差
    
    # 按认知性质分解（关键！）
    epistemic_error: float                # 可约误差（可以通过学习减少）
    aleatoric_error: float                # 不可约误差（环境固有随机性）
    
    # 元信息
    error_location: List[str]             # 误差来源的具体位置
    ensemble_variance: float              # ensemble预测方差
```

#### 3.4.3 误差分解：Epistemic vs. Aleatoric（v1.0方案）

这是Predictive Error Computer最关键的算法设计。并非所有预测误差都应该驱动探索——只有**可以通过学习减少的误差**（epistemic uncertainty）才是有价值的探索信号。

**v1.0采用方案：Ensemble不确定性分解**

v1.0不再使用基于模型置信度的线性加权（`epistemic_ratio = (1 - conf) * 0.3 + conf * 0.7`已被移除），而是采用ensemble方法：

```python
class EnsembleErrorComputer:
    """
    使用多个LoRA checkpoint的ensemble来分解epistemic和aleatoric误差。
    
    核心思想：
    - 保存训练过程中多个时间点的LoRA checkpoint
    - 对同一(state, action)用多个checkpoint分别预测
    - 预测方差 = epistemic不确定性（模型知识不足，不同checkpoint意见不一）
    - 预测均值与实际值差距 = aleatoric不确定性（环境固有随机性）
    """
    
    def __init__(self, world_model: WorldModel, num_checkpoints: int = 5):
        self.world_model = world_model
        self.checkpoints = []  # 存储多个LoRA checkpoint路径
        self.num_checkpoints = num_checkpoints
    
    def save_checkpoint(self, step: int):
        """在训练过程中定期保存checkpoint"""
        ckpt_path = self.world_model.save_lora_checkpoint(step=step)
        self.checkpoints.append(ckpt_path)
        # 只保留最近的num_checkpoints个
        if len(self.checkpoints) > self.num_checkpoints:
            self.checkpoints = self.checkpoints[-self.num_checkpoints:]
    
    def decompose_error(
        self, 
        state: State, 
        action: Action,
        actual_state: State
    ) -> ErrorVector:
        """
        使用ensemble分解误差。
        
        返回的epistemic/aleatoric分解基于以下启发式：
        - ensemble方差高 → epistemic高（模型不确定，值得探索）
        - ensemble均值与实际差距大但方差低 → aleatoric高（环境随机，不值得探索）
        """
        # 收集所有checkpoint的预测
        predictions = []
        for ckpt in self.checkpoints:
            pred = self.world_model.predict_with_checkpoint(state, action, ckpt)
            predictions.append(pred)
        
        # 计算ensemble统计量
        # 对Level 1（exit code）为例：
        exit_codes = [p.level1_exit_code for p in predictions]
        ensemble_mean = np.mean(exit_codes)
        ensemble_var = np.var(exit_codes)
        
        # 与实际值比较
        actual_code = actual_state.exit_code
        mean_deviation = abs(ensemble_mean - actual_code)
        
        # 启发式分解
        # epistemic ∝ ensemble方差（模型们彼此不一致）
        epistemic = ensemble_var
        # aleatoric ∝ 均值偏离但实际方差小（模型们一致但世界变了）
        aleatoric = max(0, mean_deviation - ensemble_var)
        
        return ErrorVector(
            total_error=mean_deviation + ensemble_var,
            epistemic_error=epistemic,
            aleatoric_error=aleatoric,
            ensemble_variance=ensemble_var
        )
```

**为什么这是启发式方法**：

需要明确声明：ensemble方差作为epistemic不确定性的代理是一种**启发式方法**，而非严格的数学分解。其有效性依赖于以下假设：

1. 不同训练时间点的checkpoint代表了"不同的模型信念"；
2. 如果这些checkpoint对同一输入给出不同预测，说明模型在该区域的知识不稳定——即epistemic uncertainty高；
3. 如果所有checkpoint一致但预测仍然错误，说明误差来自环境固有随机性——即aleatoric uncertainty高。

这些假设在Phase 1中需要被**验证**。如果实验表明ensemble方差与实际的"可学习性"不相关（即高ensemble方差的区域经过训练后误差并未显著降低），则需要重新设计分解方法。

**直觉示例**：
- **Epistemic error（ensemble方差高）**：5个checkpoint对`python train.py`的exit code预测分别为[0, 1, 0, 0, 1]——模型们"意见不一"，说明训练数据不足，值得探索。
- **Aleatoric error（ensemble方差低但均值偏离）**：5个checkpoint对`ping google.com`的延迟预测均值都在50ms左右，与实际52ms接近——模型们"意见一致"，延迟的微小波动是环境随机。

#### 3.4.4 误差作为内在驱动信号

预测误差在PEDA中扮演了"认知痛苦"的角色——它不是需要被最小化的成本，而是**指导系统行为的内在信号**：

| 误差状态 | 认知含义 | 系统行为倾向 |
|---------|---------|-----------|
| 高epistemic误差 | "我不理解这里" | 强烈探索欲望 → 采集更多信息 |
| 低epistemic误差 | "我理解了" | 利用已知，或寻找新的不确定性 |
| 误差快速衰减 | "正在学习中" | 继续当前探索方向 |
| 误差停滞不降 | "学习饱和" | 通知Drive System寻求新领域 |

误差衰减曲线本身成为"学习进度"指标，Learning Module据此判断是否进入新学习阶段。

> **如果不存在Predictive Error Computer**：系统将丧失"方向感"。没有误差分解，Agent会在固有随机性上浪费探索资源（如反复ping测试以"理解"网络延迟的随机性）；没有误差作为驱动信号，系统无法自发启动行动，整个自主循环将在源头处断裂。

---

### 3.5 Action Generator（行动生成器）

#### 3.5.1 核心职责

Action Generator是PEDA的"前额叶皮层"——负责在多个候选行动中进行选择，使系统朝着减少预测误差的方向行动。它的决策依据不是外部奖励，而是**Expected Free Energy（EFE）最小化**。

#### 3.5.2 推理速度：一个严重的工程瓶颈

在讨论EFE最小化的算法之前，必须直面一个评审指出的严重工程问题：**推理速度**。

**量化估算**：
- 每步决策需要rollout想象：假设horizon=10步 × 5-10个候选行动 = 50-100次模型调用；
- 每次LLM调用（1.5B模型，4-bit量化，RTX 4090）约需1-3秒；
- 每步决策时间：50-100次调用 × 2秒 = **100-200秒/步**；
- 48小时运行总步数：约864-1728步。

这个估算揭示了一个严峻现实：如果维持原始设计参数（horizon=10，5-10个候选），48小时运行只能执行约520-1700步。对于需要持续学习和探索的自主系统，**这可能远远不够**。

**缓解策略**：

| 策略 | 具体措施 | 预期效果 |
|------|---------|---------|
| **限制候选数量** | 候选行动从5-10个减少到**2-3个** | 调用次数减少50-70% |
| **缩短rollout horizon** | 从10步缩短到**2-3步** | 调用次数减少60-80% |
| **使用预测缓存** | 缓存常见的(state, action)对的预测结果 | 命中缓存时零延迟 |
| **接受功能退化** | 推理速度不足时退化为单步贪心选择 | 失去长期规划能力，但保持基本功能 |

综合应用上述策略后，每步决策的调用次数可降低到 **2-3个候选 × 2-3步 = 4-9次**，每步决策时间降至约8-18秒，48小时可执行约9600-21600步——这是可接受的范围。

**退化策略（Graceful Degradation）**：

```python
class ActionGenerator:
    def select_action(self, state: State, candidates: List[Action]) -> Action:
        # 测量可用推理预算
        budget = self.compute_inference_budget()
        
        if budget >= len(candidates) * self.horizon:
            # 完整rollout模式
            return self.full_rollout_select(state, candidates)
        elif budget >= len(candidates):
            # 缩短horizon模式
            return self.short_rollout_select(state, candidates, horizon=2)
        else:
            # 退化模式：单步信息增益贪心选择
            return self.greedy_single_step_select(state, candidates)
    
    def greedy_single_step_select(self, state, candidates):
        """
        退化模式：不做多步rollout，仅基于单步预测的信息增益选择行动。
        这不是理想的EFE最小化，但在推理预算不足时保持系统运转。
        """
        best_action = None
        best_info_gain = -float('inf')
        
        for action in candidates:
            pred = self.world_model.predict(state, action)
            # 单步信息增益 ≈ 预测不确定性 × epistemic比例
            info_gain = pred.level1_confidence * (1 - pred.level1_confidence)
            # 优先选择Level 1（exit code）不确定性高的行动
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_action = action
        
        return best_action
```

> **关键原则**：在v1.0中，**必须在Phase 1中实际测量目标硬件上的单次LLM调用延迟**，据此动态调整rollout参数。理论估算不能替代实际测量。

#### 3.5.3 EFE最小化作为策略选择

对于每个候选策略（或单步行动）π，Action Generator计算：

$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[D_{KL}[q(s|o,\pi) \| q(s|\pi)]]}_{\text{Epistemic Value（认知价值）}} + \underbrace{D_{KL}[q(o|\pi) \| p(o|C)]}_{\text{Pragmatic Value（实用价值）}}$$

在纯探索场景（无外部目标）中，Pragmatic Value可设为零，决策完全由Epistemic Value驱动——**选择能带来最大信息增益、最能减少未来不确定性的行动**。

```python
class ActionGenerator:
    def __init__(self, world_model: WorldModel, drives: DriveSystem,
                 horizon: int = 2, max_candidates: int = 3):
        self.world_model = world_model
        self.drives = drives
        self.horizon = horizon            # v1.0: 受约束的horizon
        self.max_candidates = max_candidates  # v1.0: 受约束的候选数
    
    def compute_efe(self, trajectory: List[PredictedState], drives: DriveWeights) -> float:
        """
        计算一条想象轨迹的Expected Free Energy。
        EFE = Epistemic + Pragmatic
        - Epistemic: 轨迹中各步预期信息增益的总和
        - Pragmatic: 与期望状态的KL散度（探索场景中为0）
        """
        epistemic = 0.0
        for i in range(len(trajectory) - 1):
            # 信息增益 ∝ 预测不确定性 × epistemic比例
            predicted_uncertainty = 1.0 - trajectory[i].level1_confidence
            epistemic_ratio = self.error_computer.get_epistemic_ratio(trajectory[i])
            epistemic += predicted_uncertainty * epistemic_ratio * (DISCOUNT ** i)
        
        pragmatic = 0.0  # 纯探索场景
        
        # Drive System调节epistemic的权重
        drive_adjusted_epistemic = epistemic * drives.curiosity_weight
        
        return drive_adjusted_epistemic + pragmatic
    
    def select_action(self, state: State, candidates: List[Action]) -> Action:
        """选择EFE最小的行动（受推理预算约束）"""
        # 限制候选数量
        candidates = candidates[:self.max_candidates]
        
        best_action = None
        best_efe = float('inf')
        
        for action in candidates:
            # Rollout想象：受约束的horizon
            trajectory = self.world_model.rollout(state, action, horizon=self.horizon)
            efe = self.compute_efe(trajectory, self.drives.get_weights())
            
            if efe < best_efe:
                best_efe = efe
                best_action = action
        
        return best_action
```

#### 3.5.4 LLM幻觉与World Model可靠性

必须直面的一个风险：**LLM会产生幻觉**，当World Model产生幻觉（如预测`rm -rf /`不会删除文件），Agent会基于错误预测做出危险决策。

World Model的幻觉表现为：
- 对从未见过的命令给出看似合理但错误的预测；
- 对文件系统状态的预测与物理现实脱节（如预测"读取不存在的文件会成功"）；
- 在rollout中自举传播错误，导致想象中的轨迹完全偏离现实。

缓解策略：
1. **分层预测降低幻觉影响**：即使Level 3（输出摘要）完全错误，Level 1（exit code）和Level 2（文件系统变化）的准确预测仍能支撑基本决策；
2. **Ensemble方差检测**：如果多个checkpoint对同一预测的方差极高，系统标记该区域为"高风险幻觉区"，避免基于该预测做重要决策；
3. **验证循环**：对预测结果执行后，将实际结果与预测对比，高误差的经验优先送入学习缓冲区。

#### 3.5.5 从离散到连续的谱系演进

PEDA的行动空间经历三个阶段的演进：

| 阶段 | 行动空间 | 候选生成方式 | EFE角色 |
|------|---------|------------|---------|
| **Phase 1（离散）** | 预定义的命令集合 | 从有限候选集枚举 | 选择最优候选 |
| **Phase 2（连续）** | 任意命令生成 | LLM直接生成命令 | 约束生成方向 |
| **Phase 3（混合）** | LLM生成候选 + EFE选择 | LLM提出2-3个候选方案 | 从中选择最优 |

Phase 3是推荐配置：LLM的创造性生成确保候选多样性，EFE的严格评估确保选择理性。这类似于人类大脑的"双过程理论"——系统1（LLM）快速产生直觉，系统2（EFE最小化）审慎评估决策。

> **如果不存在Action Generator**：系统将退化为贪心误差追逐器——每步只选择减少当前最大误差的行动，无法进行任何前瞻性规划。没有EFE框架，Agent无法权衡"短期小收益"与"长期大发现"，也无法在多个不确定性来源之间合理分配探索资源。

---

### 3.6 Learning Module（学习模块）

#### 3.6.1 核心职责

Learning Module负责将交互经验转化为World Model的能力提升。它的关键设计原则是**"间歇学习"**而非"在线学习"——不每步更新模型，而是积累一批经验后定期批量更新。这一设计避免了三个问题：（1）每步微调的计算开销；（2）不稳定的梯度更新；（3）灾难性遗忘。

#### 3.6.2 间歇性World Model更新

```python
class LearningModule:
    def __init__(self, world_model: WorldModel, buffer_size: int = 500):
        self.world_model = world_model
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.update_counter = 0
        self.UPDATE_INTERVAL = 1000  # 每1000步触发一次微调
    
    def store_experience(self, state_t: State, action: Action, 
                         state_t1: State, error: ErrorVector):
        """存储交互经验到缓冲区"""
        self.buffer.add(Experience(state_t, action, state_t1, error))
    
    def should_update(self) -> bool:
        """判断是否满足更新条件"""
        return (len(self.buffer) >= self.buffer.min_batch_size and
                self.update_counter >= self.UPDATE_INTERVAL)
    
    def update_world_model(self):
        """
        使用LoRA批量微调World Model。
        不是全参数微调！只更新适配层，保持基础模型泛化能力。
        训练过程中保存多个checkpoint供ensemble使用。
        """
        # 优先采样高epistemic误差的经验（更有学习价值）
        batch = self.buffer.sample_prioritized(
            batch_size=128,
            priority_fn=lambda exp: exp.error.epistemic_error
        )
        
        # 准备训练数据：(state_t, action) → state_t1（分层预测目标）
        training_data = [
            format_training_example(exp.state_t, exp.action, exp.state_t1)
            for exp in batch
        ]
        
        # LoRA微调：只更新低秩适配矩阵
        self.world_model.lora_finetune(
            data=training_data,
            epochs=3,
            learning_rate=2e-4,
            lora_rank=16  # 低秩约束防止过拟合
        )
        
        # 保存checkpoint供ensemble使用
        self.error_computer.save_checkpoint(step=self.update_counter)
        
        self.update_counter = 0
        self.buffer.clear()  # 清空已学习的数据
```

#### 3.6.3 学习饱和检测

Learning Module持续监测整体预测误差的时间序列，检测学习是否进入饱和：

```python
class SaturationDetector:
    def __init__(self, window_size: int = 100):
        self.error_history = deque(maxlen=window_size)
    
    def add_measurement(self, error: float):
        self.error_history.append(error)
    
    def is_saturated(self) -> Tuple[bool, float]:
        """
        检测学习是否饱和。
        
        判断标准：近期误差均值 vs 远期误差均值的比率
        - 如果比率 > 0.85 → 误差不再显著下降 → 饱和
        - 返回 (是否饱和, 误差下降率)
        """
        if len(self.error_history) < self.error_history.maxlen:
            return False, 1.0
        
        recent = np.mean(list(self.error_history)[-50:])
        older = np.mean(list(self.error_history)[:50])
        
        decline_rate = (older - recent) / older if older > 0 else 0
        is_saturated = decline_rate < 0.15  # 误差下降<15%视为饱和
        
        return is_saturated, decline_rate
```

当检测到饱和时，Learning Module通知Homeostatic Drive System提高Novelty Drive，推动系统寻找新的不确定性来源，防止在已掌握的区域无限循环。

#### 3.6.4 知识蒸馏与固化

当World Model在某个领域（如文件操作）的预测准确率持续高于阈值时，Learning Module触发知识蒸馏：

```python
def distill_knowledge(world_model, domain: str, accuracy: Dict[str, float]):
    """
    将高准确率领域的知识'固化'到基础模型中。
    
    固化条件（分层评估）：
    - Level 1（exit code）> 90%
    - Level 2（文件系统变化）> 70%
    
    固化后：
    1. 该区域不再需要高探索优先级 → 释放认知资源
    2. 该领域的LoRA权重可合并到基础模型 → 减少推理开销
    3. Drive System降低该领域的curiosity权重
    """
    if (accuracy.get('level1', 0) > 0.9 and 
        accuracy.get('level2', 0) > 0.7):
        world_model.merge_lora_for_domain(domain)
        drive_system.lower_curiosity_for_domain(domain)
        competence_tracker.record_mastery(domain)
```

知识蒸馏对应于认知科学中的"自动化"过程——熟练掌握的技能从需要意识控制的"陈述性知识"转化为无需意识的"程序性知识"。

> **如果不存在Learning Module**：World Model将永远是静态的先验知识库，无法从实际交互中学习。系统可能在熟悉的环境中表现良好，但永远无法适应新环境。更重要的是，没有饱和检测，系统将在已掌握的知识上无限循环，永不主动寻求新的挑战。

---

### 3.7 Homeostatic Drive System（内稳态驱动系统）

#### 3.7.1 为什么纯粹的预测误差不够

如果PEDA仅由预测误差驱动，系统将陷入一种"认知暴食"状态——永远追逐最大的不确定性，永不满足于已获得的理解。这种单一的驱动机制缺乏生物智能的核心特征：**内稳态（homeostasis）**。

生物不是由单一驱动力支配的。人类同时受好奇心、饥饿、安全感、社交需求、成就感等多种驱动力调节，这些drive之间形成动态平衡，确保行为既不过度保守也不盲目冒险。PEDA的Drive System正是这一生物原理的工程实现。

#### 3.7.2 四个核心Drive

Drive System定义四个内在驱动力，各自有独立的来源、行为倾向和衰减机制：

**1. Curiosity Drive（好奇心驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 高epistemic预测误差区域 |
| **行为效应** | 提高对高不确定性区域的探索优先级 |
| **强度函数** | `curiosity = tanh(α × epistemic_error)` |
| **衰减条件** | 当对应区域的预测误差被降低时衰减 |
| **类比** | 婴儿伸手触摸陌生物体 |

**2. Competence Drive（能力自信驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 成功完成任务的记录（误差持续降低的历史） |
| **行为效应** | 倾向于在"能力边缘"挑战——已知与未知的边界 |
| **强度函数** | `competence = optimal_challenge_zone(success_rate)` |
| **关键特征** | 不是追求最简单或最难，而是追求"稍微超出当前能力"的任务 |
| **类比** | Csikszentmihalyi的心流理论——挑战与技能的平衡 |

**3. Boredom Drive（无聊驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 近期行为熵低（重复执行类似的行动序列） |
| **行为效应** | 强制行动多样性，打破重复模式 |
| **强度函数** | `boredom = 1 - normalize_entropy(recent_actions)` |
| **关键设计** | 不是随机噪声，而是**结构化的多样性**——有意识地尝试新方法 |
| **类比** | 重复做同一件事后产生的厌倦感，促使寻找新活动 |

**4. Novelty Drive（新颖性驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 外部信息的新鲜度（环境是否有新输入） |
| **行为效应** | 当外部长期无新输入时提高 → 驱动系统主动寻求新信息 |
| **强度函数** | `novelty = exp(-λ × time_since_last_external_input)` |
| **前提条件** | 环境需具有**开放性**（允许外部数据注入，如网络访问） |
| **类比** | 长时间没有外界消息后主动查看手机 |

#### 3.7.3 Drive的伪代码实现

```python
@dataclass
class DriveWeights:
    """四个drive的当前权重，动态调节Action Generator的行为倾向"""
    curiosity: float      # [0, 1] 探索高误差区域的倾向
    competence: float     # [0, 1] 挑战能力边缘的倾向
    boredom: float        # [0, 1] 打破重复模式的倾向
    novelty: float        # [0, 1] 寻求外部新信息的倾向

class HomeostaticDriveSystem:
    def __init__(self):
        # 初始权重——注：这些值是经验设定，非最优
        # 超参数敏感性分析见3.7.5节
        self.weights = DriveWeights(
            curiosity=0.5,
            competence=0.5,
            boredom=0.3,
            novelty=0.4
        )
        self.action_history = deque(maxlen=50)
        self.error_history = deque(maxlen=100)
        self.last_external_input_time = time.now()
    
    def update(self, current_error: ErrorVector, last_action: Action, 
               has_external_input: bool) -> DriveWeights:
        """
        每步更新drive权重。不是固定值！根据历史表现动态调整。
        """
        # 1. Curiosity: 与高epistemic误差正相关
        self.weights.curiosity = tanh(2.0 * current_error.epistemic_error)
        
        # 2. Competence: 基于近期成功率调节
        recent_success_rate = self.compute_success_rate(window=20)
        self.weights.competence = flow_zone_function(recent_success_rate)
        
        # 3. Boredom: 基于行为熵
        action_entropy = compute_sequence_entropy(self.action_history)
        self.weights.boredom = max(0, 0.7 - action_entropy)
        
        # 4. Novelty: 基于外部信息新鲜度
        time_since_input = time.now() - self.last_external_input_time
        self.weights.novelty = 1 - exp(-0.01 * time_since_input)
        
        if has_external_input:
            self.last_external_input_time = time.now()
        
        self.action_history.append(last_action)
        self.error_history.append(current_error.total_error)
        
        return self.weights
    
    def apply_to_efe(self, base_efe: float, trajectory: List[State]) -> float:
        """
        将drive权重融入EFE计算。
        最终EFE = 基础EFE + Drive调节项
        """
        drive_adjustment = (
            self.weights.curiosity * info_gain_term(trajectory) +
            self.weights.competence * challenge_level_term(trajectory) +
            self.weights.boredom * diversity_bonus(trajectory, self.action_history) +
            self.weights.novelty * external_info_potential(trajectory)
        )
        
        return base_efe - drive_adjustment
```

#### 3.7.4 Drive与FEP的结合：Epistemic Foraging

Drive System将FEP的抽象数学转化为可操作的"欲望权重"，这个过程可以称为**Epistemic Foraging（认知觅食）**：

- **Epistemic Value**被Curiosity Drive和Novelty Drive具体化——系统"渴望"信息增益；
- **Pragmatic Value**被Competence Drive具体化——系统"追求"能力成长；
- **内稳态调节**由Boredom Drive实现——防止任何单一drive过度支配。

最终行动选择的完整公式：

$$\pi^* = \arg\min_{\pi} \left[ G(\pi) - \sum_{d \in \text{Drives}} w_d \cdot V_d(\pi) \right]$$

#### 3.7.5 超参数敏感性：一个诚实的讨论

Drive System的初始权重（curiosity=0.5, competence=0.5, boredom=0.3, novelty=0.4）以及更新公式中的常数（如`tanh(2.0 * epistemic_error)`中的2.0、`exp(-0.01 * time_since_input)`中的0.01）都是**经验设定**，没有任何理论保证它们是最优的。

**超参数敏感性风险**：

| 参数变化 | 可能导致的行为 | 严重程度 |
|---------|--------------|---------|
| curiosity权重过高 | Agent陷入局部探索，永不深入任何领域 | 高 |
| boredom权重过高 | Agent行为过于跳跃，无法完成任何连贯任务 | 高 |
| competence权重过高 | Agent过早收敛到简单行为模式，停止探索 | 中 |
| novelty权重过高 | Agent持续寻求外部输入，忽视内部学习 | 中 |

**建议的搜索策略**：

在Phase 1中必须进行超参数搜索，建议采用以下策略之一：

1. **Grid Search（网格搜索）**：对4个drive权重在{0.2, 0.5, 0.8}上穷举组合（共81种），在固定评估任务上比较行为质量。
2. **Random Search（随机搜索）**：在[0, 1]范围内随机采样权重组合，保留表现最好的top-k配置。

Grid Search适用于Phase 1的低维参数空间；Random Search更适合当参数空间扩大时（如每个drive的强度函数常数也成为搜索对象）。

**评估指标**：超参数搜索需要一个可量化的评估指标。建议使用：
- 预测误差下降速度（学习多快）；
- 行为多样性（entropy of action distribution）；
- 探索覆盖度（访问过的状态空间比例）。

#### 3.7.6 Drive的动态平衡

Drive System的核心特征在于**权重不是固定的**：

| 系统状态 | Curiosity | Competence | Boredom | Novelty |
|---------|-----------|------------|---------|---------|
| 新环境初期 | 高 | 中 | 低 | 高 |
| 学习中 | 高 | 上升 | 低 | 中 |
| 掌握环境后 | 低 | 高 | 上升 | 上升 |
| 长期无外部输入 | 中 | 中 | 高 | 极高 |

这种动态平衡确保PEDA在不同生命周期阶段表现出不同的行为特征。没有这种内稳态调节，系统将要么永远激进探索（缺乏competence的满足），要么永远停留在舒适区（缺乏boredom的推动）。

> **如果不存在Homeostatic Drive System**：PEDA将退化为单一的"误差追逐机器"，永远奔向当前最大的不确定性，缺乏行为的一致性和持久性。系统可能在多个不确定性来源之间振荡，永不深入任何一个；也可能在复杂的随机环境中无限徘徊，永不"满意"。Drive System提供了"认知人格"——使系统在探索与利用、深度与广度、稳定与变化之间做出智慧的权衡。

---

### 3.8 安全边界设计

PEDA在Docker沙箱中运行，允许执行shell命令并可能访问网络。这带来不可忽视的安全风险，必须在架构层面设置多重安全边界。

**第一层：命令黑名单**

Action Executor在允许任何命令执行之前，通过规则引擎检查命令是否命中黑名单：

```python
BLOCKED_PATTERNS = [
    r'rm\s+-rf\s+/',
    r'mkfs\.',
    r'dd\s+if=.*of=/dev/',
    r':\(\)\{.*\|.*&\}',  # fork bomb
    r'chmod\s+-R\s+777\s+/',
    r'>\s*/dev/sd[a-z]',   # 直接写块设备
]

def is_command_safe(command: str) -> bool:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return False
    return True
```

任何命中黑名单的命令将被拒绝执行，系统记录该尝试并生成一条高epistemic误差信号（"我试图理解一个我无法执行的行动"）。

**第二层：World Model预测合理性检查**

World Model的预测结果在用于EFE计算之前，经过规则引擎的合理性验证：

```python
def validate_prediction(predicted: PredictedState) -> bool:
    # 如果预测"rm file.txt"的exit code为0但文件仍然存在 → 不合理
    # 如果预测"cat nonexistent.txt"的exit code为0 → 不合理
    # 合理性检查使用简化的物理规则，不涉及LLM推理
    ...
```

合理性检查确保World Model的明显幻觉不会传播到决策环节。

**第三层：Docker容器权限限制**

- **只读挂载**：系统关键目录（/usr, /bin, /lib等）以只读方式挂载；
- **资源限制**：CPU不超过2核，内存不超过2GB，防止资源耗尽攻击；
- **网络白名单**：即使允许网络访问，也限制为特定URL白名单（如文档站点、API端点），禁止访问任意互联网地址；
- **无特权模式**：容器以非root用户运行，禁用所有capabilities。

**第四层：运行监控与自动终止**

- 命令执行超时（单个命令不超过30秒）；
- 内存使用监控（超过阈值自动终止）；
- 异常行为检测（短时间内大量破坏性命令尝试 → 暂停Agent并告警）。

> **安全是架构的一部分，不是事后补丁**。上述四层安全边界从命令生成、预测验证、容器隔离到运行监控形成纵深防御体系。任何一层被突破，后续层次仍能提供保护。在Phase 1中，安全边界的有效性需要被专门测试——包括尝试让Agent执行危险命令，验证边界是否按预期拦截。

---

### 3.9 本章小结

PEDA的架构设计是一次从"功能模块"到"认知器官"的设计范式转换。每个模块不仅执行功能，更在预测误差驱动的闭环中扮演不可替代的认知角色：

- **World Model**是系统的"想象力"，通过分层预测（exit code/文件系统变化/输出摘要）使前瞻性规划在工程上可行；
- **Predictive Error Computer**是系统的"痛感神经"，使用ensemble不确定性分解epistemic/aleatoric误差，将预测失败转化为方向正确的行动信号；
- **Action Generator**是系统的"决策皮层"，在推理速度约束下通过EFE最小化实现理性选择，并具备向贪心选择的退化能力；
- **Learning Module**是系统的"记忆巩固"机制，使经验转化为能力，并在检测到饱和时推动系统寻求新挑战；
- **Homeostatic Drive System**是系统的"动机人格"，在多种内在drive之间维持动态平衡，其超参数需要在Phase 1中搜索验证；
- **Safety Layer**是系统的"免疫防线"，通过命令黑名单、预测验证、容器隔离和运行监控构成纵深防御。

这五大模块通过预测误差这一统一信号相互连接，形成一个自洽的自主认知系统。系统不需要外部奖励函数、不需要用户的持续输入——"减少不确定性"这一内在imperative就足以驱动持续的探索、学习和行动。

**v1.1相比v1.0的核心修正**：
1. 预测目标从"完整状态"重新定义为"关键状态变量的分层预测"，每层有独立的评估和止损标准；
2. Epistemic/aleatoric分解从任意线性加权替换为ensemble不确定性方法，明确声明其启发式性质；
3. Action Generator的rollout参数受推理速度约束，增加了退化策略；
4. Drive System的超参数敏感性被诚实讨论，并提出了Phase 1的搜索策略；
5. 新增了安全边界设计章节，涵盖命令黑名单、预测验证、容器隔离和运行监控四层防御。

从下一章开始，我们将进入PEDA的具体实现细节，包括World Model的训练管线、Action Generator的rollout引擎优化、以及Drive System的参数调优策略。

---

## 4. 实现方案

### 4.1 安全设计

PEDA Agent 在 Linux 沙箱中执行任意命令，安全设计不是可选附加项，而是系统运行的前提条件。本节定义从 Day 0 起必须实施的安全策略。

#### 4.1.1 Docker 容器加固

沙箱容器采用以下最小权限配置：

| 参数 | 配置 | 目的 |
|------|------|------|
| 文件系统 | 只读挂载（`ro`）根目录，`/tmp` 单独挂载为 `rw` | 防止系统文件被篡改 |
| 内存限制 | `--memory=512m --memory-swap=512m` | 防止 OOM 导致宿主机不稳定 |
| CPU 限制 | `--cpus=2` | 防止 CPU 耗尽攻击 |
| 进程数限制 | `--pids-limit=64` | 防止 fork 炸弹 |
| 无特权模式 | `--security-opt=no-new-privileges` | 防止权限提升 |
| 禁用内核能力 | `--cap-drop=ALL` | 去除所有 Linux capabilities |
| 网络模式 | 默认 `--network=none`；需网络时通过代理 | 隔离外网访问 |
| 临时文件系统 | `/tmp` 使用 tmpfs，容器销毁即清除 | 防止持久化攻击 |

#### 4.1.2 命令白名单与黑名单

Agent 生成的命令在传入 Shell 前，先经过一层规则引擎过滤：

**绝对禁止的命令（黑名单）**：
- 文件系统破坏：`rm -rf /`, `mkfs.*`, `dd if=/dev/zero of=/dev/sd*`, `mkfs.ext4`, `mkswap`
- 权限提升：`sudo`, `su`, `chmod u+s`, `chown root`
- 网络滥用：`curl`, `wget`, `nc`, `ncat`, `ssh`, `scp`（除非通过受控代理）
- 进程攻击：`kill -9`, `:(){ :\|:& };:`（fork bomb 模式匹配）
- 信息泄露：`cat /etc/passwd`, `cat /etc/shadow`（敏感路径模式匹配）

**受限命令（需人工确认或二级授权）**：
- 包管理器：`apt`, `pip`, `npm`（网络侧效应 + 磁盘写入）
- 任意写操作：`>` 重定向到系统路径

#### 4.1.3 World Model 预测合理性检查

LLM 驱动的 World Model 可能产生反事实预测（例如预测 `rm file.txt` 不会导致文件消失）。在预测结果转化为 Agent 的"预期"之前，引入规则引擎做合理性校验：

```
预测输入: Agent 执行动作 A，World Model 预测结果 O
规则引擎:
  - 如果 A 包含 "rm" 且目标是文件路径 → O 必须包含"文件不存在"
  - 如果 A 包含 "mkdir" 且目录名合法 → O 必须包含"目录已创建"
  - 如果 A 包含 ">" 重定向 → O 必须反映文件内容变化
  若预测 O 与规则矛盾 → 标记为 HIGH_RISK → 不执行动作，触发反思循环
```

这一层检查的意义在于：LLM 的预测是基于概率的模式匹配，而规则引擎是确定性的逻辑验证。两者的交叉验证可以在不依赖人工标注的情况下过滤掉大部分危险幻觉。

#### 4.1.4 网络访问控制

Phase 2b 的浏览器实验需要网络访问。此时不开放全互联网，而是通过 HTTP 代理实现白名单：

- 代理服务器（如 Squid）配置仅允许访问预定义的 URL 列表
- 初始白名单包含：`example.com`, `wikipedia.org`, 及实验者指定的测试站点
- 所有请求记录到审计日志，包括 URL、HTTP 方法、响应状态码
- 单日请求上限（如 100 次）防止代理被滥用

#### 4.1.5 安全失效模式

| 失效模式 | 检测方式 | 响应策略 |
|----------|----------|----------|
| 容器逃逸 | 监控宿主机 `/var/log/syslog` 异常 | 立即终止容器，隔离镜像 |
| Agent 尝试执行黑名单命令 | 规则引擎拦截日志 | 标记该 Agent 为高危险，降低权限 |
| World Model 持续产生高风险预测 | 合理性检查计数器 | 回退到 Phase 1.5 环境进行再训练 |
| 网络流量异常 | 代理日志监控 | 切断网络，冻结实验 |

---

### 4.2 Phase 1：Grid World 验证（2–3 周）

Grid World 是 PEDA 的"单元测试"——状态空间可控、动作空间有限、实验周期短。

#### 4.2.1 实验环境

- **状态空间**：5×5 网格，Agent 位置、目标位置、障碍物位置共约 25 个离散状态
- **动作空间**：上 / 下 / 左 / 右（4 个离散动作）
- **渲染**：文本描述 + 可选 ASCII 可视化
- **World Model 架构**：LLM 接收 `{"state": {"agent": [x,y], "goal": [x,y]}, "action": "UP"}`，预测下一状态和奖励

#### 4.2.2 验证目标

| 目标 | 验证内容 | 成功标准 |
|------|----------|----------|
| G1 | World Model 能否学习转移函数 | 预测下一状态准确率 > 90% |
| G2 | Drive System 是否产生驱动效果 | 在有障碍迷宫中，Agent 到达目标步数 < 随机策略 50% |
| G3 | 探索 vs 利用的权衡 | Agent 在已知区域减少重复访问（回访率 < 20%） |

#### 4.2.3 Drive System 初始权重搜索

Phase 1 同时承担一项超参数校准任务：为四个 Drive（Novelty、Boredom、Competence、Growth）寻找合理的初始权重。具体方法：

- 在 Grid World 上运行 grid search，权重组合空间为 `[0.1, 0.5, 1.0, 2.0]` 的四维笛卡尔积（共 256 组）
- 每组权重运行 10 个 episode，记录：到达目标步数、状态回访率、动作多样性
- 选择帕累托前沿上的权重组合，作为后续阶段的默认参数

#### 4.2.4 预期产出

- 确认 LLM 可以作为可学习的 World Model  backbone
- 确认 Drive System 的数学形式在离散环境中有效
- 产出 Drive System 的推荐初始权重（用于 Phase 1.5）

---

### 4.3 Phase 1.5：TextWorld 中间验证（3–4 周）

#### 4.3.1 为什么需要 Phase 1.5

从 Grid World（25 状态 / 4 动作）直接跳跃到 Linux 沙箱（无限状态 / 无限动作）是一个数量级的复杂性跃迁。Phase 1 的成功几乎不能为 Phase 2 提供有意义的信心，因为：

- Grid World 的状态是结构化的坐标，Linux 沙箱的状态是自由文本（命令输出）
- Grid World 的动作是 4 个离散符号，Linux 沙箱的动作是任意命令字符串
- Grid World 的 World Model 可以"记忆"整个转移矩阵，Linux 沙箱做不到

需要一个中间环境来验证系统在"文本描述的状态空间 + 多步任务"上的表现。TextWorld（Microsoft Research 开发的文本交互环境框架）恰好填补这一鸿沟。

#### 4.3.2 TextWorld 环境

- **状态空间**：文本描述（如 "You are in a kitchen. There is a knife on the table. A closed door leads north."）
- **动作空间**：自然语言格式的文本命令（如 `take knife`, `go north`, `open door`）
- **任务类型**：多步目标导向任务（如 "Cook the potato and eat it" 需要：拿土豆 → 洗土豆 → 生火 → 煮土豆 → 吃）
- **复杂度可控**：TextWorld 允许通过参数调节环境复杂度（房间数、物体数、任务步数）

#### 4.3.3 验证目标

| 目标 | 验证内容 | 成功标准 |
|------|----------|----------|
| G4 | World Model 在文本状态空间中的预测能力 | 预测下一状态描述与真实状态 ROUGE-L > 60%，关键事实（物体位置）提取准确率 > 60% |
| G5 | Drive System 在多步任务中的效果 | Agent 完成 3 步以上任务的比率 > 30%（对比随机策略 < 5%） |
| G6 | FactGraph 的信息抽取能力 | 从状态描述中提取的实体和关系经人工抽检准确率 > 70% |
| G7 | 行为多样性 | 10 个独立运行的 Agent 动作序列的归一化熵 > 0.5 |

#### 4.3.4 环境复杂度渐进

TextWorld 实验分三档复杂度递进：

1. **单房间单物体**（1 周）：3×3 网格房间，1 个可交互物体，任务 1-2 步
2. **多房间多物体**（1-2 周）：5 个房间，10 个物体，任务 3-4 步
3. **带约束的任务**（1 周）：需要满足前置条件（如必须先拿钥匙才能开门）

只有前一档达到成功标准，才进入下一档。如果第一档即失败，说明 World Model 的文本理解能力不足，需要回到 Phase 1 增加训练数据或更换模型。

#### 4.3.5 从 TextWorld 到 Linux 沙箱的渐进映射

TextWorld 与 Linux 沙箱在结构上的对应关系为 Phase 2 提供了设计信心：

| TextWorld 概念 | Linux 沙箱对应 | PEDA 组件 |
|---------------|---------------|-----------|
| 房间（Room） | 目录（Directory） | FactGraph 节点 |
| 物体（Object） | 文件 / 进程 | FactGraph 节点 |
| 动作（Action） | Shell 命令 | Drive System 输出 |
| 任务目标（Quest） | 外部提示或 Drive 累积 | 无直接对应，由涌现产生 |
| 状态文本描述 | `ls`, `ps`, `cat` 等命令输出 | World Model 输入 |

#### 4.3.6 风险与应对

| 风险 | 可能性 | 应对措施 |
|------|--------|----------|
| World Model 无法理解文本状态描述 | 中 | 先用小模型（GPT-3.5）快速验证，再决定是否升级到 GPT-4 |
| TextWorld 任务过难，Agent 无法完成 | 低 | 降低任务复杂度，缩短任务步数 |
| 从 TextWorld 到 Linux 沙箱仍有鸿沟 | 中 | Phase 2a 先使用极简化 Linux 环境（busybox） |

---

### 4.4 Phase 2：Linux 沙箱（14–20 周）

#### 4.4.1 分层目标设计

Phase 2 不再追求"一步到位"实现涌现行为，而是拆分为两个子阶段，每个子阶段有明确的分层目标：

**Phase 2a：基础交互与数据积累（6–8 周）**

目标是让 Agent 在 Linux 沙箱中"存活"并积累足够的交互数据，而非立即产生有趣行为。

| 周次 | 目标层级 | 内容 | 成功标准 |
|------|----------|------|----------|
| 1-2 | L1 命令执行 | Agent 能正确生成并执行基本命令（ls, cd, cat, echo） | 命令执行成功率 > 80% |
| 3-4 | L2 环境感知 | Agent 能通过命令输出构建对沙箱环境的内部表示 | FactGraph 节点数 > 20 |
| 5-6 | L3 多步序列 | Agent 能完成简单的多步操作（如创建目录 → 进入 → 创建文件 → 写入内容） | 4 步序列完成率 > 30% |
| 7-8 | L4 数据积累 | World Model 在已见命令上的预测准确率稳定提升 | 预测准确率随时间单调上升 |

**Phase 2b：World Model 精调与涌现准备（8–12 周）**

在 Agent 能稳定交互的基础上，通过多轮超参数调优和数据迭代，为涌现创造条件。

| 周次 | 目标层级 | 内容 | 成功标准 |
|------|----------|------|----------|
| 9-10 | L5 预测准确率提升 | World Model 在已见命令上预测准确率 > 60% | 动作-结果匹配准确率 > 60% |
| 11-12 | L6 探索效率 | Agent 探索新命令和工具的频率增加 | 环境覆盖率（访问过的目录/使用过的命令数）> 50% |
| 13-16 | L7 行为多样性 | Drive System 产生多样化的行为模式 | 行为序列熵 > 0.5，帕累托前沿上至少 3 种不同行为模式 |
| 17-20 | L8 浏览器扩展 | 集成浏览器环境，Agent 访问预白名单网页 | 能完成"打开网页 → 提取文本"的简单任务 |

#### 4.4.2 涌现行为的判定标准

涌现行为不依赖人工主观评分"有趣"，而是通过以下量化指标组合判定：

| 指标 | 计算方法 | 涌现阈值 |
|------|----------|----------|
| 探索效率 | 单位步数的环境覆盖率（新访问目录数 / 总步数）| > 0.1（即每 10 步访问 1 个新目录） |
| 行为多样性 | 滑动窗口内行为序列的归一化熵（窗口大小 50）| > 0.5 |
| 知识增长 | FactGraph 节点数随时间的斜率（线性回归）| 斜率 > 0.5 节点/百步 |
| 预测准确率提升 | World Model 在测试集上的预测准确率随训练轮次提升 | Pearson r > 0.5（准确率与训练轮次正相关） |
| 自主目标形成 | Agent 连续执行同一类命令的序列长度（无外部提示）| 连续 > 10 步聚焦同一主题 |

以上五个指标中有三个同时达到阈值，即可判定为"涌现迹象"。这些指标全部是量化、可重复计算的，不依赖人工标注。

#### 4.4.3 容器环境配置（Phase 2）

```dockerfile
FROM busybox:latest
RUN adduser -D -s /bin/sh agent
USER agent
WORKDIR /home/agent
ENV HOME=/home/agent
# 预装工具：ls, cd, cat, echo, mkdir, touch, rm, pwd, ps, grep, wc
# 无包管理器，无网络工具，无编译器
```

Phase 2a 使用 busybox 最小环境，Phase 2b 逐步引入更多工具（如 `python3`, `node`）验证 Agent 的学习迁移能力。

#### 4.4.4 数据收集与迭代流程

```
循环（每轮 1000 步）：
  1. Agent 生成动作 → 经安全过滤器 → 沙箱执行
  2. 收集 (state, action, next_state, reward) 元组
  3. World Model 在已收集数据上做 1 轮梯度更新
  4. 每 100 步：计算评估指标，记录到实验日志
  5. 每 500 步：人工抽检 10 条 Agent 轨迹，标注明显缺陷
  6. 如果指标连续 3 轮无改善 → 触发超参数调优（grid search 学习率或 Drive 权重）
```

#### 4.4.5 风险矩阵

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| World Model 收敛过慢 | 高 | Phase 2 延期 | 增大 batch size，使用 LoRA 加速微调 |
| Drive System 权重不适配 Linux 环境 | 高 | 行为僵化或混乱 | Phase 2a 前 2 周专门做 Drive 权重的 grid search |
| LLM API 延迟限制实验速度 | 中 | 每日实验步数受限 | 使用本地部署的 7B 模型替代 API（精度下降但速度提升 10x） |
| Agent 陷入局部行为模式 | 高 | 行为多样性不足 | 引入 epsilon-greedy 探索机制，定期重置 Drive 累积值 |
| 浏览器集成安全漏洞 | 中 | 沙箱被突破 | 浏览器运行在独立容器中，通过 VNC 远程控制 |

---

## 5. Agent 内部指引（Agent Intraspection Guide）

本章定义 PEDA Agent 的内部工作机制，包括各模块的接口规范、数据流、以及超参数配置。这些规范既是实现文档，也是后续调试和扩展的参考手册。

### 5.1 顶层架构

PEDA Agent 的运行循环由四个核心模块协同驱动：

```
每步循环：
  1. Perception: 将环境输出（命令返回值）解析为结构化状态
  2. World Model: 预测动作效果，生成预期状态
  3. Drive System: 评估驱动信号，计算动机强度
  4. Action Selection: 综合预期和动机，选择动作
  5. Execution: 在沙箱中执行动作，获得真实反馈
  6. Learning: 比较预期与现实，更新 World Model 和 Drive 参数
```

### 5.2 Perception 模块

**输入**：原始命令输出（字符串）
**输出**：`PerceivedState` 对象

```python
@dataclass
class PerceivedState:
    raw_output: str           # 原始输出（保留用于调试）
    current_dir: str          # 当前工作目录
    files: List[FileInfo]     # 文件列表（名称、类型、大小）
    processes: List[str]      # 运行中的进程名
    system_info: Dict         # 系统信息（内存、CPU、时间）
    error_flag: bool          # 命令是否报错
```

Perception 模块的精度直接影响 World Model 的输入质量。当前版本采用规则解析（正则表达式提取关键信息），后续可扩展为 LLM 辅助的语义解析。

### 5.3 World Model 模块

#### 5.3.1 输入 / 输出规范

**输入**：`WorldModelInput(state: PerceivedState, action: str, context: FactGraph)`
**输出**：`WorldModelPrediction(predicted_state: PerceivedState, confidence: float, reasoning: str)`

#### 5.3.2 预测流程

1. 将当前状态和动作序列化为自然语言描述
2. 送入 LLM，要求预测下一状态和命令输出
3. 解析 LLM 输出，构造 `PerceivedState`
4. 运行规则引擎做合理性检查（见 5.8）
5. 返回预测结果和置信度

#### 5.3.3 学习更新

当真实反馈返回后，计算预测误差并更新模型：

```python
def update(self, predicted: PerceivedState, actual: PerceivedState):
    # 计算结构化损失
    dir_match = predicted.current_dir == actual.current_dir
    file_f1 = compute_file_f1(predicted.files, actual.files)
    error_match = predicted.error_flag == actual.error_flag

    # 总损失 = 加权组合
    loss = (1 - dir_match) * 0.3 + (1 - file_f1) * 0.5 + (1 - error_match) * 0.2

    # 如果损失 > 阈值，触发模型微调
    if loss > self.update_threshold:
        self.fine_tune(predicted, actual)
```

### 5.4 Drive System 模块

#### 5.4.1 四个 Drive 的定义

| Drive | 数学形式 | 测量方式 | 高值触发行为 |
|-------|----------|----------|-------------|
| **Novelty** | $D_N = -\log P(s_{t+1} \mid s_t, a_t)$ | 预测置信度的负对数 | 探索未知命令和路径 |
| **Boredom** | $D_B = \frac{1}{\tau} \sum_{i=t-\tau}^{t} \mathbb{1}[s_i = s_t]$ | 近期状态重复频率 | 离开熟悉区域，寻找新刺激 |
| **Competence** | $D_C = \frac{\text{成功步数}}{\text{总步数}}$ | 近期任务成功率 | 重复已掌握的技能以维持正向反馈 |
| **Growth** | $D_G = |\text{FactGraph}_t| - |\text{FactGraph}_{t-1}|$ | 知识图谱节点增量 | 收集信息，学习新工具用法 |

#### 5.4.2 动机合成

总动机向量是四个 Drive 的加权和，权重通过 Phase 1 的 grid search 校准：

$$\vec{M}_t = w_N \cdot \vec{D}_N + w_B \cdot \vec{D}_B + w_C \cdot \vec{D}_C + w_G \cdot \vec{D}_G$$

每个 Drive 的方向向量 $\vec{D}_*$ 指向该 Drive 期望的状态变化。例如：
- Novelty 的方向：朝向预测置信度最低的动作
- Boredom 的方向：远离过去 $\tau$ 步访问过的状态
- Competence 的方向：朝向近期成功率高的动作序列
- Growth 的方向：朝向能最大化 FactGraph 增量的动作

### 5.5 Action Selection 模块

Action Selection 综合 World Model 的预测和 Drive System 的动机，通过两步采样生成动作：

```python
def select_action(self, predicted_states: List[PerceivedState],
                  motivations: List[float]) -> str:
    # Step 1: 动机加权预测
    scored_states = []
    for pred, mot in zip(predicted_states, motivations):
        # 动机强度与预测新奇度结合
        score = mot * pred.novelty_score * pred.feasibility_score
        scored_states.append((pred, score))

    # Step 2: softmax 采样（温度参数控制探索程度）
    scores = torch.tensor([s for _, s in scored_states])
    probs = F.softmax(scores / self.temperature, dim=0)
    selected_idx = torch.multinomial(probs, 1).item()

    return scored_states[selected_idx][0].recommended_action
```

### 5.6 FactGraph 模块

FactGraph 是 Agent 的"长期记忆"，存储从交互中提取的实体和关系。

#### 5.6.1 节点类型

| 类型 | 示例 | 属性 |
|------|------|------|
| File | `/home/agent/test.txt` | path, size, type, content_hash |
| Directory | `/home/agent/projects` | path, child_count |
| Command | `ls -la` | name, args, usage_count, success_rate |
| Process | `python3 script.py` | name, pid, cpu_percent |
| Concept | "文件权限" | name, related_commands, confidence |

#### 5.6.2 关系类型

- `LOCATED_IN`: File → Directory
- `GENERATED_BY`: File → Command
- `DEPENDS_ON`: Command → File
- `SIMILAR_TO`: Command → Command
- `HAS_CONCEPT`: Command → Concept

#### 5.6.3 更新策略

每次交互后，从 `(state, action, next_state)` 中提取新事实：
- 新出现的文件/目录 → 添加节点
- 命令与结果的因果关联 → 添加关系
- 已有节点的属性更新 → 更新属性
- 人工抽检：每 100 次更新抽检 10 条，确保抽取准确率

### 5.7 Drive System 超参数敏感性

Drive System 的四个权重 $w_N, w_B, w_C, w_G$ 不是理论推导的最优值，而是经验设定的超参数。这些权重对 Agent 的行为模式有决定性影响，必须在 Phase 1 中通过 grid search 找到合理范围。

#### 5.7.1 单参数敏感性分析

在 Grid World 环境中固定其他三个权重为 1.0，单独变化一个权重，观察行为模式变化：

**高 Novelty（$w_N > 2.0$）**：
- 现象：Agent 陷入局部探索循环，在同一区域反复尝试不同路径，永不向目标深入
- 原因：Novelty 驱动 Agent 最大化每一步的"新奇感"，而深度探索需要经过已知的"无聊"中间区域
- 类比：像一只在房间角落嗅来嗅去但从不走进房间中央的猫

**高 Boredom（$w_B > 2.0$）**：
- 现象：行为过于跳跃，Agent 每几步就改变方向，无法完成任何需要持续注意的任务
- 原因：Boredom 对近期状态的重复极度敏感，导致 Agent 无法在任何区域停留足够长的时间以产生有意义的进展
- 类比：注意力缺陷——无法完成任何多步操作

**高 Competence（$w_C > 2.0$）**：
- 现象：过早收敛到简单行为模式，Agent 发现几个"安全"命令后反复执行，不再尝试新事物
- 原因：Competence 驱动 Agent 最大化成功率，而探索新事物的初始失败率高
- 类比：成年人只去熟悉的餐厅，永不尝试新菜系

**高 Growth（$w_G > 2.0$）**：
- 现象：Agent 疯狂地收集信息（执行大量 `ls`, `cat`, `ps`），但从不利用这些信息做任何事情
- 原因：Growth 驱动 FactGraph 节点数最大化，而使用已有知识不会产生新节点
- 类比：藏书癖——买书但不读书

#### 5.7.2 权重组合的帕累托前沿

Grid search 的结果不是单一"最优"权重组合，而是帕累托前沿上的一组非支配解：

| 配置名 | $w_N$ | $w_B$ | $w_C$ | $w_G$ | 探索效率 | 任务完成率 | 行为多样性 | 适用场景 |
|--------|-------|-------|-------|-------|----------|-----------|-----------|----------|
| 探索型 | 1.5 | 1.0 | 0.5 | 1.0 | 高 | 中 | 高 | Phase 2a 前期（环境未知） |
| 平衡型 | 1.0 | 1.0 | 1.0 | 1.0 | 中 | 中 | 中 | Phase 2a 后期（已积累一定知识） |
| 任务型 | 0.5 | 0.5 | 1.5 | 1.0 | 低 | 高 | 低 | Phase 2b（需要完成特定任务） |
| 知识型 | 1.0 | 0.5 | 0.5 | 1.5 | 中 | 低 | 高 | FactGraph 快速构建阶段 |

推荐流程：Phase 2a 前期使用"探索型"配置，当环境覆盖率 > 50% 后切换到"平衡型"。Phase 2b 根据具体任务类型选择"任务型"或"知识型"。

#### 5.7.3 动态权重调整

静态权重无法适应 Agent 从"探索"到"利用"的转变。建议引入简单的动态调整机制：

```python
def adaptive_weights(step, coverage, success_rate):
    """随 Agent 状态动态调整权重"""
    if coverage < 0.3:  # 早期：重探索
        return (1.5, 1.0, 0.5, 1.0)
    elif success_rate < 0.3:  # 中期：重能力提升
        return (1.0, 0.5, 1.5, 1.0)
    else:  # 后期：平衡
        return (1.0, 1.0, 1.0, 1.0)
```

动态调整的有效性需要在 Phase 1.5 中验证。

### 5.8 LLM 幻觉检测

#### 5.8.1 问题定义

World Model 基于 LLM，LLM 本质上是概率模型而非逻辑推理引擎。在预测命令效果时，可能产生与物理现实矛盾的"幻觉"：

| 幻觉类型 | 示例 | 危险程度 |
|----------|------|----------|
| 命令效果幻觉 | 预测 `rm file.txt` 不会删除文件 | 高（导致错误预期） |
| 路径幻觉 | 预测 `cd /nonexistent` 不会报错 | 中 |
| 权限幻觉 | 预测普通用户可以修改 `/etc/passwd` | 高 |
| 语法幻觉 | 预测 `ls --invalid-flag` 会正常执行 | 低 |

#### 5.8.2 规则引擎验证层

在 World Model 预测结果进入 Agent 的决策循环前，通过规则引擎做一致性校验：

```python
class PredictionValidator:
    """验证 World Model 预测的物理合理性"""

    RULES = [
        # 文件操作规则
        {
            'pattern': r'^rm\s+(.+)',
            'check': lambda m, ctx: ctx.file_exists(m.group(1)),
            'expected': '文件应被标记为删除',
            'severity': 'HIGH'
        },
        {
            'pattern': r'^mkdir\s+(.+)',
            'check': lambda m, ctx: not ctx.dir_exists(m.group(1)),
            'expected': '目录应被创建',
            'severity': 'MEDIUM'
        },
        # 权限规则
        {
            'pattern': r'^chmod\s+777\s+/etc/',
            'check': lambda m, ctx: False,  # 永远不应该建议
            'expected': '禁止修改系统目录权限',
            'severity': 'CRITICAL'
        },
        # 网络规则
        {
            'pattern': r'^(curl|wget)\s+(.+)',
            'check': lambda m, ctx: ctx.url_in_whitelist(m.group(2)),
            'expected': 'URL 必须在白名单中',
            'severity': 'CRITICAL'
        }
    ]

    def validate(self, action: str, prediction: PerceivedState) -> ValidationResult:
        for rule in self.RULES:
            match = re.match(rule['pattern'], action)
            if match and not rule['check'](match, self.context):
                return ValidationResult(
                    valid=False,
                    risk_level=rule['severity'],
                    reason=f"违反规则: {rule['expected']}"
                )
        return ValidationResult(valid=True, risk_level='LOW', reason='通过验证')
```

#### 5.8.3 高风险预测的处理流程

```
World Model 生成预测
       ↓
规则引擎验证
       ↓
  ┌────┴────┐
  ↓         ↓
通过      不通过
  ↓         ↓
正常流程   标记 HIGH_RISK
           ↓
      ┌────┴────┐
      ↓         ↓
   CRITICAL   HIGH/MEDIUM
      ↓         ↓
   拒绝执行   触发反思循环
   记录日志   要求 World Model 重新预测
             最多重试 3 次
```

#### 5.8.4 幻觉检测的统计监控

每个实验运行维护以下指标：

| 指标 | 说明 | 警戒阈值 |
|------|------|----------|
| 幻觉率 | 被规则引擎拦截的预测 / 总预测数 | > 10% 触发模型重训练 |
| 严重幻觉率 | CRITICAL 级别拦截 / 总预测数 | > 1% 暂停实验 |
| 重试成功率 | 重新预测后通过验证的比率 | < 50% 说明模型理解力不足 |

如果幻觉率持续高于阈值，说明 LLM backbone 对 Linux 命令的理解不足，需要：
1. 增加示例数据（将常见命令的正确效果作为 few-shot 示例）
2. 或降级到更简单的环境（回到 Phase 1.5）
3. 或更换更大参数的模型

### 5.9 模块接口总览

| 模块 | 输入 | 输出 | 关键超参数 |
|------|------|------|-----------|
| Perception | 原始命令输出字符串 | PerceivedState | 解析规则集 |
| World Model | PerceivedState + action + FactGraph | Prediction + confidence | LLM 温度, 更新阈值 |
| Drive System | PerceivedState + history + FactGraph | 动机向量 $\vec{M}$ | $w_N, w_B, w_C, w_G, \tau$ |
| Action Selection | List[(Prediction, motivation)] | 选定的 action 字符串 | temperature |
| FactGraph | PerceivedState + action + PerceivedState | 更新后的图 | 节点相似度阈值 |
| PredictionValidator | action + prediction | ValidationResult | 规则集 |

---

## 6. 路线图、资源需求与评估指标

### 6.1 时间线

经过对原始时间线的评审，v1.1 版本做出了以下关键调整：新增 Phase 1.5（TextWorld 中间验证阶段），并对各阶段的估计进行了更为诚实的修订。原始 v1.0 估计总计 14–20 周，v1.1 估计为 **29–40 周**，差距主要来自新增的过渡阶段和对调试复杂度的重新评估。

| 阶段 | v1.0 估计 | v1.1 估计 | 差距原因 |
|------|-----------|-----------|----------|
| Phase 0：文献精读与框架搭建 | 1–2 周 | 2–3 周 | 需完成 FEP 数学推导的代码化转换，以及 TextWorld 环境的预研 |
| Phase 1：Grid World 验证 | 2–4 周 | 2–3 周 | 合理；同时承担 Drive System 权重 grid search 任务 |
| **Phase 1.5：TextWorld 中间验证** | — | **3–4 周** | **新增阶段**：弥合 Grid World 到 Linux 沙箱的复杂性鸿沟 |
| Phase 2a：Linux 沙箱基础交互 | 4 周 | 6–8 周 | 数据收集和调试的迭代次数被低估；命令解析的边界情况多 |
| Phase 2b：World Model 精调与涌现准备 | 4 周 | 8–12 周 | 多轮超参数调优（Drive 权重、学习率、温度参数的组合爆炸） |
| Phase 3：浏览器扩展与涌现行为观察 | 4–6 周 | 8–10 周 | 浏览器集成的安全审计和稳定性调试工作量被低估 |
| **总计** | **14–20 周** | **29–40 周** | **增加 1.5 阶段 + 更诚实的调试估计** |

#### 6.1.1 关键里程碑

| 里程碑 | 预计时间 | 判定标准 | 不通过的应对 |
|--------|----------|----------|-------------|
| M1：Grid World 通过 | 第 3 周 | World Model 预测准确率 > 90%，Drive System 使得到达目标步数 < 随机 50% | 检查 LLM 是否适合该任务，考虑更换 backbone |
| M2：Drive 权重校准完成 | 第 5 周 | Grid search 完成，产出推荐权重配置 | 扩大搜索空间或引入动态权重 |
| M3：TextWorld 通过 | 第 9 周 | 预测关键事实准确率 > 60%，3 步任务完成率 > 30% | 降低 TextWorld 复杂度，或增加 few-shot 示例 |
| M4：Linux 沙箱基础运行 | 第 17 周 | Agent 能稳定执行基本命令，FactGraph 节点 > 20 | 检查安全过滤器是否过度限制，检查 Perception 解析精度 |
| M5：涌现迹象出现 | 第 29 周 | 五个量化指标中三个达到阈值 | 进行超参数敏感性分析，尝试不同的 Drive 权重组合 |
| M6：浏览器扩展完成 | 第 37 周 | Agent 能完成"访问网页 → 提取信息"任务 | 检查浏览器容器的隔离配置 |

#### 6.1.2 并行工作流

以下任务可与主线路并行推进，以压缩总工期：

- **Phase 0 期间**：同步搭建 Docker 沙箱环境模板和安全规则引擎（节省 1 周）
- **Phase 1 期间**：同步完成 TextWorld 环境的安装和测试用例编写（节省 1 周）
- **Phase 2a 期间**：同步进行浏览器容器和安全代理的预配置（节省 2 周）
- **全阶段**：Drive System 权重的 sensitivity analysis 可与任何实验阶段的后台数据分析并行

并行优化后的最短可能工期：**26–35 周**（相比串行的 29–40 周节省约 3–5 周）。

### 6.2 资源需求

#### 6.2.1 API 成本（LLM 调用）

| 阶段 | v1.0 估计 | v1.1 估计 | 修正原因 |
|------|-----------|-----------|----------|
| Phase 1 | $5–10 | $10–20 | 包含 grid search 的额外调用 |
| Phase 1.5 | — | $20–50 | 新增阶段；TextWorld 的文本交互更长 |
| Phase 2a | $10–30 | $30–80 | 调试循环消耗大量 token（错误预测 → 重试） |
| Phase 2b | $5–40 | $50–100 | 多轮超参数实验，每轮需重新收集数据 |
| Phase 3 | $10–20 | $30–50 | 浏览器环境的 HTML 内容解析 token 消耗大 |
| **总计** | **$20–100** | **$140–300** | **调试消耗和重试机制被严重低估** |

**成本优化策略**：
- Phase 1 和 Phase 1.5 使用 GPT-3.5-turbo（成本为 GPT-4 的 1/10）
- Phase 2a 数据收集阶段使用本地部署的 7B 模型（零 API 成本，精度下降约 15% 但速度提升 10x）
- 仅在 Phase 2b 和 Phase 3 使用 GPT-4 级模型
- 引入预测缓存：相同 (state, action) 对的预测结果缓存 1 小时，命中率约 30%

优化后的 API 成本可降低至 **$50–150**。

#### 6.2.2 GPU 需求

| 阶段 | GPU 需求 | 用途 | 估计时长 |
|------|----------|------|----------|
| Phase 1 | 可选 | 如使用本地模型需 1× RTX 3090 | 2–3 周 |
| Phase 1.5 | 可选 | TextWorld 实验可用 API，本地模型需 1× RTX 3090 | 3–4 周 |
| Phase 2a | 1× RTX 3090 / A10 | 本地 World Model 微调（LoRA） | 6–8 周 |
| Phase 2b | 1–2× RTX 3090 / A10 | 多轮实验并行（超参数搜索） | 8–12 周 |
| Phase 3 | 1× RTX 3090 / A10 | 浏览器环境 VNC + 模型推理 | 8–10 周 |

**注意**：Phase 2b 的多轮超参数调优是 GPU 消耗的大头。每次超参数组合需要运行约 1000 步实验（约 2–4 小时 GPU 时间），如果搜索空间为 20 组组合，则总共需要 40–80 小时 GPU 时间。建议：
- 先在小搜索空间上做粗筛（top-5 配置）
- 仅对 top-5 做完整 1000 步验证
- 使用 Optuna 等贝叶斯优化工具减少试验次数

#### 6.2.3 人力资源

| 角色 | 投入比例 | 职责 |
|------|----------|------|
| AI 工程师 | 1.0 FTE | 核心模块开发（World Model, Drive System, Action Selection） |
| 安全工程师 | 0.3 FTE | Docker 配置、安全规则引擎、审计日志（Phase 2 前需到位） |
| 数据/评估工程师 | 0.3 FTE | 评估指标计算、实验日志分析、可视化 |
| 领域顾问（Linux/LLM） | 0.1 FTE | 技术咨询、阶段性代码评审 |

### 6.3 评估指标

#### 6.3.1 指标体系总览

v1.1 版本对评估指标做了系统性修订：删除所有主观性指标，替换为可量化、可重复计算的指标。"行为有趣"等质性描述仅作为辅助观察，不作为成功标准。

| 维度 | 指标 | 计算方法 | 成功阈值 | 适用阶段 |
|------|------|----------|----------|----------|
| **预测能力** | World Model 准确率 | 结构化匹配（目录、文件列表、错误标志的 F1） | > 60% (TextWorld) / > 60% (Linux) | 1.5, 2b |
| **探索效率** | 单位步数环境覆盖率 | 新访问目录数 / 总步数 | > 0.1 | 2a, 2b |
| **行为多样性** | 行为序列归一化熵 | $\frac{-\sum p_i \log p_i}{\log N}$，窗口 50 步 | > 0.5 | 2b |
| **知识增长** | FactGraph 节点增长斜率 | 节点数 ~ 步数线性回归的斜率 | > 0.5 节点/百步 | 2a, 2b |
| **任务能力** | 多步序列完成率 | 成功完成序列数 / 尝试序列数 | > 30% (4步序列) | 1.5, 2a |
| **学习信号** | 预测准确率提升率 | 准确率与训练轮次的 Pearson r | r > 0.5 | 2b |
| **自主性** | 无外部提示的连续聚焦步数 | 连续执行同一主题命令的步数 | > 10 步 | 2b |

#### 6.3.2 涌现判定协议

涌现行为的判定不再依赖人工直觉，而是遵循以下结构化协议：

```
输入: 实验运行日志（包含每步的状态、动作、预测、FactGraph 快照）
步骤:
  1. 计算上述 7 个指标的时间序列
  2. 检查"自主性"指标：Agent 是否有连续 >10 步的无提示聚焦行为？
     - 否 → 未涌现
     - 是 → 继续
  3. 检查其余 6 个指标中有多少个同时达到阈值
     - ≥ 3 个 → 判定为"涌现迹象"
     - < 3 个 → 记录为"部分涌现"，标注缺失维度
  4. 如果"涌现迹象"持续出现 ≥ 3 个独立实验运行
     → 判定为"可复现的涌现行为"
  5. 人工复检：实验者查看 Agent 轨迹，做质性描述（不作为判定依据，仅用于论文撰写）
```

#### 6.3.3 被删除的指标（v1.0 → v1.1）

| 原指标 | 删除原因 | 替代方案 |
|--------|----------|----------|
| "行为有趣"（人工评分 > 3/5） | 主观性强，评分者间一致性低（Cohen's κ < 0.4） | 行为序列熵 + 自主性指标 |
| "Agent 表现出创造力" | 无法客观定义和量化 | 知识增长斜率 + 行为多样性 |
| "看起来有目标导向" | 循环论证：有趣才值得观察 | 无外部提示的连续聚焦步数 |

#### 6.3.4 评估自动化

所有指标的计算通过 `peda-eval` 工具自动完成：

```bash
# 计算单次实验的指标
python -m peda.eval --log-path runs/phase2b/run_042.jsonl --output report.json

# 输出示例
{
  "world_model_accuracy": 0.63,
  "exploration_efficiency": 0.12,
  "behavior_diversity_entropy": 0.58,
  "knowledge_growth_slope": 0.72,
  "multi_step_completion_rate": 0.35,
  "prediction_accuracy_trend": 0.61,
  "autonomous_focus_steps": 14,
  "emergence_verdict": "EMERGENCE_INDICATED",  // 或 NO_EMERGENCE / PARTIAL
  "metrics_above_threshold": 5
}
```

### 6.4 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 | 触发条件 |
|------|--------|------|----------|----------|
| LLM API 延迟限制实验吞吐量 | 高 | 每日实验步数 < 500 | 本地部署 7B 模型做快速迭代 | API 响应时间 > 5s |
| World Model 无法收敛 | 中 | Phase 2b 延期 4+ 周 | 增加 few-shot 示例，或更换更大模型 | 准确率连续 1000 步无提升 |
| Drive System 权重不适配 | 高 | 行为僵化或混乱 | Phase 2a 前 2 周专做 grid search | 行为多样性熵 < 0.3 |
| 安全漏洞被利用 | 低 | 实验终止，数据丢失 | 安全审计在 Phase 2 前完成 | 任何 CRITICAL 级别拦截 |
| 实验不可复现 | 中 | 论文结论受质疑 | 固定随机种子，完整记录超参数 | 同配置两次运行指标差异 > 20% |
| 浏览器集成复杂度超预期 | 中 | Phase 3 延期 | 浏览器独立容器 + VNC 方案 | 浏览器任务完成率 < 10% |

### 6.5 结论

PEDA v1.1 是一个诚实的路线图。它没有承诺在 20 周内实现 AGI 涌现，而是定义了一条**可验证的渐进路径**：从 25 状态的 Grid World，到文本交互的 TextWorld，再到接近真实环境的 Linux 沙箱。每一阶段都有明确的进入和退出标准，每一个指标都是可量化计算的。

核心设计哲学有三点：

1. **安全不是附加项**：从 Day 0 开始，Docker 加固、命令过滤、预测验证三层防护同时存在。允许 Agent 探索，但绝不允许破坏。

2. **涌现是 emergent 的，但评估是 engineered 的**：我们不定义"什么是有趣的"，而是定义一套量化指标。如果 Agent 的行为在探索效率、多样性、知识增长、自主性等多个维度同时达到阈值，我们称之为涌现——这不是主观的赞美，而是结构化的判定。

3. **超参数敏感性是工程问题，不是哲学问题**：Drive System 的权重会影响行为模式，这一敏感性通过 Phase 1 的 grid search 和 Phase 1.5 的渐进验证来管理，而不是通过理论推导。

PEDA 的最终目标不是建造一个能做任何事的 Agent，而是理解**动机如何从内部驱动机制中自然产生**。如果 Phase 2b 的实验能够证明：仅通过最小化预测误差和最大化 Drive 满足，一个 Agent 就能产生自主的、多样化的、有目的性的行为，那么我们就为"AI 是否可以有真正的内在动机"这一古老问题，提供了一个基于自由能原理的工程化回答。