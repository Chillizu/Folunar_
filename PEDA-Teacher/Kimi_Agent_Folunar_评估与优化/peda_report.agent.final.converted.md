# PEDA (Predictive-Error-Driven Autonomous Agent)
# 架构设计与开发计划书

## ——用预测误差替代Prompt：基于Active Inference的自主Agent设计

**版本**: 1.0  
**日期**: 2026年7月2日  
**理论基础**: Free Energy Principle (Friston et al.), Predictive Coding (Rao & Ballard; Clark), World Models (Ha & Schmidhuber; Hafner et al.)  
**目标**: 构建不依赖用户Prompt、由内在预测误差驱动的自主探索Agent

---

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

---

## 2. 理论基础：从预测误差到自主Agent的统一框架

PEDA架构的设计并非凭空创造，而是建立在一个正在 converging 的理论图景之上。这幅图景横跨计算神经科学、统计物理学和机器学习三个领域，其核心洞见可以概括为一句话：**智能体是一台预测机器，它的全部认知活动——感知、行动、学习——都可以被统一为对预测误差的最小化**。本章将系统梳理支撑PEDA的五大理论支柱，并为后续章节的形式化定义奠定基础。

### 2.1 Prompt范式的根本限制

要理解PEDA为什么必须存在，我们首先要看清当前大语言模型（LLM）应用范式的一个深层结构性缺陷。

今天所有的LLM应用——从最简单的聊天机器人到最先进的Agent框架——都共享同一个底层架构假设：**模型是一组冻结的权重，每次调用都从零开始推理，调用结束后不保留任何持久状态**。这意味着模型的"思考"完全是外部驱动的：没有输入（prompt），就没有思考。模型不会"自己想点什么"，它只是被动地响应刺激。

这个限制的直接后果是整个行业对prompt工程的过度依赖。研究者和工程师们投入巨大精力去设计更好的prompt模板、更精巧的few-shot示例、更复杂的chain-of-thought结构。近期有工作声称"不依赖用户提示"，例如通过内部模板生成来构造prompt。但这种做法本质上只是**把prompt的编写者从人类换成了另一个程序**——prompt仍然存在，discrete call仍然存在，模型的被动本质没有改变。

这就像蒸汽机时代的工程师不断改良阀门控制装置，却没有意识到真正需要的是内燃机。Prompt范式的根本问题不在于"谁来写prompt"，而在于**"prompt/discrete call"这个概念本身就是对智能的错误抽象**。

真正的突破需要回答一个更深层的问题：如果模型拥有持久状态，如果它能够在没有外部刺激的情况下自发产生内部动力学，如果它的行为是由内部预测与外部感知之间的**持续张力**所驱动——那么，prompt还是必要的吗？

PEDA的回答是：不需要。当Agent的内部世界模型持续生成关于未来感知的预测时，**预测误差本身就成为了行动的发动机**。不需要人类用自然语言去"命令"模型做什么，模型会主动追求那些能够降低其预测误差的状态。Prompt从驱动者降级为可选的初始化参数——而预测误差接管了控制权。

这种范式转换的哲学意义可以类比笛卡尔剧场模型的崩塌。在prompt范式中，LLM是一个坐在黑暗剧场里的观众，只有当聚光灯（prompt）打亮舞台时才会开始观看和思考。而在PEDA范式中，Agent是一座永不停歇的城市——即使没有外部交通流入，城市内部的电力系统、工厂流水线、信息交换网络依然在自发运转，并持续对外部环境的变化做出反应。

### 2.2 Active Inference与自由能原理

自由能原理（Free Energy Principle, FEP）是Karl Friston提出的一套统一框架，旨在解释自组织系统如何维持其存在。它为PEDA提供了最核心的理论骨架：**为什么最小化预测误差会导致看起来像智能的行为**。

FEP的数学基础可以追溯到变分推断。Friston, Kilner & Harrison (2006) 的开创性论文指出，任何自组织系统（从单细胞生物到人类大脑）都面临同一个基本问题：如何通过有限的感官接口推断外部世界的隐藏状态，并据此行动以维持自身的稳态 [Friston et al., 2006]。Friston (2010) 进一步将这一原理推广为统一的大脑理论，论证了感知、行动和学习可以被理解为同一个最小化过程的三个方面 [Friston, 2010]。Friston et al. (2017) 的里程碑论文将这一框架形式化为"Active Inference"的过程理论，给出了完整的数学表述 [Friston et al., 2017]。最新的权威综述（Friston et al., 2023）以更清晰的方式重新阐述了整个框架 [Friston et al., 2023]。

**变分自由能的直觉理解**。想象一个有机体试图感知环境中的隐藏状态（比如一只猎豹试图判断草丛中是否有猎物）。它有一个关于世界如何运作的生成模型（generative model）——一个内部模拟器，能够根据假设的隐藏状态预测感官输入。变分自由能源自一个深刻的数学事实：由于有机体无法直接访问外部世界的真实状态，它只能通过感官间接推断。变分自由能是这个推断问题的一个上界——最小化自由能等价于同时做两件事：（1）让内部信念尽可能准确地匹配外部现实（感知推断），（2）让感官输入尽可能匹配内部预期（行动）。

用更直觉的类比：想象你正在尝试接住一个飞来的球。你的大脑并不直接"计算"球的抛物线轨迹，而是**持续生成关于球下一秒位置的预测**，并通过眼动和肢体调整来让实际的视觉输入与预测一致。如果你没有接到球，预测误差会驱动你更新内部模型（"原来这个球比我想象的重"）——这就是学习。这种误差驱动的感知-行动循环，正是自由能原理的核心。

**期望自由能（Expected Free Energy, EFE）**。如果变分自由能解释了"如何感知"，那么EFE解释了"如何行动"——具体来说，**如何选择下一步行动**。EFE定义为：

$$G(\pi) = \mathbb{E}_{q(o|\pi)}[\ln q(o|\pi) - \ln p(o|C)] = H[q(o|\pi)] + D_{KL}[q(o|\pi) \,||\, C(o)]$$

其中 $\pi$ 是一个策略（行动序列），$q(o|\pi)$ 是执行该策略后预期观测的分布，$C(o)$ 是偏好分布（表征Agent"想要"什么样的观测）。

这个公式有两个项，分别对应探索和利用。第一项 $H[q(o|\pi)]$ 是预期观测的熵，衡量执行该策略后结果的不确定性。高熵意味着"我还不知道会发生什么"——这驱动**探索**。第二项是KL散度，衡量预期观测与偏好之间的距离。最小化这一项意味着"让我得到我想要的结果"——这驱动**利用**。

EFE对PEDA设计的意义是深远的。传统强化学习需要人工设计reward函数，而EFE表明：**探索和利用可以从同一个数学表达式中自然涌现，不需要外部reward信号**。Agent之所以走向某个方向，不是因为有人告诉它那里有reward，而是因为它自己的内部模型确信：朝那个方向走会降低未来的不确定性并满足其偏好。这 precisely 对应了PEDA中"预测误差驱动行为"的核心机制。

**Active Inference与强化学习的根本区别**。传统RL是行为主义的：Agent从环境中获得一个标量reward，目标是最大化累积reward。这是一个**外部驱动**的框架——没有reward就没有学习信号。而Active Inference是**内部驱动**的：Agent的目标是最小化自身的变分自由能，即最小化"内部的惊异"（surprise）。Action不是被外部reward拉动的，而是被内部预测误差推动的。这个区别看似微妙，实则根本：它意味着Agent可以在完全没有外部reward的环境中仍然表现出目标导向的行为——因为它追求的是内部模型的一致性，而非外部的点数。

### 2.3 Predictive Coding：从大脑到机器学习

如果FEP提供了"为什么要预测"的哲学回答，Predictive Coding则提供了"如何实现"的计算蓝图。

Predictive Coding的起源可以追溯到Rao & Ballard (1999) 的开创性论文，他们提出视觉皮层可以被视为一个分层的预测机器 [Rao & Ballard, 1999]。在这个框架中，皮层区域并不被动地"提取特征"，而是主动地生成关于下层输入的预测，并只将**预测误差**（实际输入与预测之间的差异）向上传递。这个简单的机制解释了神经科学中的大量实验发现，包括自下而上和自上而下连接的对称性、感受野的动态调制、以及注意力的增益控制效应。

Andy Clark在2013年的里程碑论文中将Predictive Coding从视觉处理推广到通用认知架构，论证了大脑本质上是一个"预测引擎"，通过持续的假设-检验循环与外部世界耦合 [Clark, 2013]。他的专著《Surfing Uncertainty》进一步将这一观点发展为完整的认知哲学 [Clark, 2015]。在机器学习领域，Millidge et al. (2022) 的关键论文证明了Predictive Coding可以与深度学习的核心算法——反向传播——建立严格的形式化联系 [Millidge et al., 2022]。

**分层预测编码的直觉**。想象一个多层的神经网络，每一层都试图预测下一层的活动。最底层接收原始感官输入（比如像素），它生成关于这些像素的预测。预测误差（实际像素与预测像素的差异）被传递到上一层。上一层的目标不是编码原始像素，而是预测下一层的预测误差。这个过程递归进行，形成一条误差传递链。

这种架构的精妙之处在于**信息效率**。如果预测是准确的，误差为零，网络不需要向上传递任何信息。只有当出现"意外"时——一只突然飞入视野的鸟，一个不符合预期的声音——信息才会向上流动。这解释了为什么我们的大脑能够在处理海量感官输入的同时保持惊人的效率：它只处理意外，忽略预期。

**Predictive Coding与反向传播的关系**。这是Predictive Coding对PEDA最重要的理论贡献。传统深度学习依赖反向传播来训练网络，但反向传播有一个致命的生物学 implausibility：它需要全局的梯度信息沿着网络反向传播，这在生物神经网络中没有已知的实现机制。更实际地说，反向传播要求网络是"可暂停"的——前向传播计算输出，后向传播计算梯度，两者不能同时进行。

Millidge et al. (2022) 证明了Predictive Coding网络可以通过**纯局部学习规则**来近似反向传播的梯度 [Millidge et al., 2022]。在PC网络中，每个神经元只需要知道它自己的活动和它与邻居的连接强度，就可以更新权重。不需要全局的梯度信号，不需要反向传播。这意味着PC网络可以**在线学习**——在持续运行的同时不断更新自身。

对PEDA的具体意义是：我们有了一个在生物学上更 plausible、在工程上更灵活的替代方案来替代反向传播。PEDA的核心——一个持续运行的预测网络，用局部误差信号驱动学习和行为——直接建立在Predictive Coding的计算框架之上。

### 2.4 World Models：在想象中学习行动

World Models领域为PEDA提供了关于"如何构建内部模拟器"的工程蓝图。World Models的核心思想是：智能体应该学习一个环境 dynamics 的压缩表示，然后在这个内部模拟器中 planning 和行动，而不是直接在真实环境中试错。

**RSSM架构的核心设计**。Ha & Schmidhuber (2018) 的开创性论文首次将World Models用于策略学习，展示了在一个学习到的潜空间中进行"梦境训练"的可能性 [Ha & Schmidhuber, 2018]。Hafner et al. (2019) 在此基础上提出了RSSM（Recurrent State-Space Model），这是一个精巧的架构设计，结合了**确定性循环路径**和**随机潜变量** [Hafner et al., 2019]。

RSSM的设计直觉可以这样理解：环境的一部分变化是可预测的（比如球的运动遵循物理定律），这部分由确定性循环网络编码；另一部分变化是本质随机的（比如一个游戏中敌人是否出现），这部分由随机潜变量建模。两者结合，使得RSSM既能做长程预测（靠确定性路径），又能处理不确定性（靠随机变量）。

**Dreamer系列的演进**。Hafner et al. (2020) 的DreamerV1证明了RSSM可以用于从像素直接学习连续控制策略 [Hafner et al., 2020]。DreamerV2 (2021) 引入了离散潜变量和更好的表示学习，在Atari游戏上达到了与model-free方法相当的性能 [Hafner et al., 2021]。DreamerV3 (2023) 是 culminating 成就——它使用**固定超参数**在超过150个不同任务上达到了人类水平或更高的性能，包括需要长期推理的Minecraft任务 [Hafner et al., 2023]。这一结果表明，World Model方法具有惊人的通用性，不再需要对每个任务进行繁琐的超参数调优。

**长程预测的挑战**。World Models面临的一个根本性挑战是**预测的发散**。当模型试图预测超过15步的未来时，小误差会指数级累积，导致预测结果迅速退化。这在实际中表现为：模型可以准确预测下一秒会发生什么，但对一分钟后的预测毫无意义。

PEDA的解决方案不是追求无限精确的 long-term prediction，而是**将预测误差本身作为行动的驱动力**。当模型对远处的未来不确定时，这种不确定性（高熵）会直接体现在期望自由能中，驱动Agent采取行动来降低不确定性。这相当于将"我无法预测"这个事实转化为"我应该去探索"的行为信号。

**Active Inference与World Models的结合**表明这一方向的可行性。Mazzaglia et al. (2022) 的"Probabilistic Dreaming"工作在Dreamer框架中引入了自由能剪枝机制，利用Active Inference的期望自由能来筛选和评估想象的轨迹，在多个连续控制任务上提升了4.5%的得分 [Mazzaglia et al., 2022]。这一结果表明，FEP不仅是一个理论框架，更是一个可以带来实际性能提升的工程工具。PEDA直接继承这一洞见：内部想象不是可有可无的装饰，而是Agent决策循环的必要组成部分。

### 2.5 内在动机与好奇心的FEP视角

如果Agent的行为由预测误差驱动，那么一个直接的问题是：它不会被环境中的随机噪声困住吗？这一节讨论FEP如何优雅地解决这个问题。

**ICM与RND的核心机制**。Pathak et al. (2017) 的内在好奇心模块（ICM）是一个经典的内在动机方法 [Pathak et al., 2017]。ICM训练一个前向模型来预测下一个状态的特征表示，将预测误差作为内在reward。当Agent遇到新颖的状态时，前向模型预测不准，产生高误差，驱动Agent探索。Burda et al. (2018) 的随机网络蒸馏（RND）采用了类似的思路，但用一个固定的随机网络来提取特征，避免了ICM中特征学习可能被"欺骗"的问题 [Burda et al., 2018]。

**Noisy TV问题**。ICM和RND都面临一个根本性的理论缺陷，被称为"Noisy TV问题"。想象一个Agent在一个房间里探索，房间里有一台电视机在播放随机噪声。由于电视画面是不可预测的，ICM会产生持续的预测误差，Agent会被"钉"在电视机前，获得源源不断的"好奇心reward"——尽管观看随机噪声对完成任何实际任务都没有帮助。

这个问题的本质在于：**预测误差混淆了两种根本不同的不确定性**。一种是"可约的不确定性"（reducible uncertainty）——只要我获取更多信息，这种不确定性就可以被消除（比如一个我没有打开过的门后面是什么）。另一种是"不可约的不确定性"（irreducible uncertainty）——即使我知道了一切，这种不确定性依然存在（比如一个硬币的下一次投掷结果）。ICM的预测误差无法区分这两种情况，因此会被纯粹的随机性误导。

**FEP的解决方案**。Active Inference通过期望自由能的数学结构优雅地解决了这个问题。回忆EFE的两个项：

$$G(\pi) = \underbrace{H[q(o|\pi)]}_{\text{epistemic value（探索）}} + \underbrace{D_{KL}[q(o|\pi) \,||\, C(o)]}_{\text{pragmatic value（利用）}}$$

第一项——熵 $H[q(o|\pi)]$——度量的是执行策略后预期观测的不确定性。但关键在于：**这不是预测误差，而是Agent对自己预测的不确定性**。当面对一台Noisy TV时，Agent很快会学习到电视画面是本质随机的——它的生成模型会收敛到对电视输出的概率分布的准确估计。一旦这个分布被学习，即使每次看到的具体画面仍然不可预测，Agent对自己预测的**不确定性**（即分布的熵）却很低。因为模型已经学会了"电视输出的是均匀随机噪声"——它对这一事实非常确定。

因此，EFE中的epistemic value会选择那些**能够让Agent更新其信念的状态**，而不是那些仅仅产生高预测误差的状态。Noisy TV不会产生高的epistemic value，因为看更多随机画面并不能让Agent学到任何关于世界的新结构。

**信息增益 vs 预测误差的根本区别**。信息增益（Information Gain）衡量的是：在观察到某个结果后，我关于隐藏状态的信念分布发生了多大变化。预测误差衡量的是：我的预测与实际观测之间的差异。当面对本质随机的输出时，预测误差可以很高，但信息增益为零——因为我并没有因此改变对世界的理解。

这个区别对PEDA至关重要。PEDA中的驱动信号不是原始的预测误差，而是**能够带来信息增益的预测误差**。Agent被设计为追求那些能够让它"恍然大悟"的体验，而不是那些仅仅让它"困惑"的噪声。

**探索与利用的统一**。传统RL需要手动设计exploration-exploitation的平衡（如epsilon-greedy、UCB等）。FEP表明，这个平衡是**自然涌现**的，不需要外部调节。当Agent对其目标状态高度不确定时，epistemic value占主导，驱动探索；当Agent对如何实现目标有清晰信念时，pragmatic value占主导，驱动利用。这不是一个需要手动调节的参数，而是Agent内部信念状态的动态结果。

### 2.6 连续时间认知

前述所有理论——FEP、Predictive Coding、World Models——本质上都假设系统以离散时间步运行。但真实的认知是连续流淌的，不是以16Hz或50Hz的帧率"跳动"的。这一节讨论连续时间架构如何为PEDA提供最终的"发动机"。

**CTRNN与吸引子动力学**。Randall Beer在1995年的开创性工作中引入了连续时间递归神经网络（CTRNN），用于模拟昆虫的行走和觅食行为 [Beer, 1995]。CTRNN的核心特征在于其神经元状态的演化由微分方程描述，而非离差的差分方程。这产生了一个深刻的结果：CTRNN可以在没有外部输入的情况下产生**自发行为**——其行为不是外部刺激的直接反应，而是内部吸引子动力学与外部输入的相互作用。

用动力系统的语言来说，CTRNN的状态空间中存在多个"吸引子 basin"——类似于山谷中的湖泊。在没有外部输入时，系统会停留在某个吸引子附近（持续产生某种行为模式，比如持续行走）。外部输入的作用不是"触发"某个行为，而是**改变吸引子 landscape 的形状**——让某个 basin 变浅，另一个 basin 变深，从而使系统自然"滑入"新的行为模式。这种机制解释了生物行为的流畅性和自发性：昆虫不是每毫秒都在"决定"下一步怎么走，它的行走模式是自持的神经动力学的自然输出。

**LTC/CfC的速度与稳定性**。Hasani et al. (2020, 2021) 的Liquid Time-Constant Networks（LTC）和Closed-form Continuous-depth networks（CfC）代表了连续时间神经网络在工程上的突破 [Hasani et al., 2020; Hasani et al., 2021]。传统Neural ODE（Chen et al., 2018）需要数值ODE求解器，训练速度极慢 [Chen et al., 2018]。LTC通过为每个神经元赋予独立的、输入自适应的时间常数，实现了更快的信息传播和更强的表达能力。CfC则进一步推导出了LTC的闭式近似，将训练和推理速度提升了100-4000倍，同时保持了连续时间模型的稳定性和表达力。

**连续时间架构 vs 离散LLM的本质区别**。当前的LLM本质上是离散时间系统：token被逐个生成，每个token的生成是一个独立的计算步骤。这意味着LLM"思考"的速度受到生成长度的线性制约——思考1000个token需要1000个前向传播步骤。更重要的是，在没有生成token的时候，LLM不做任何事情。它的内部状态不会自发演化。

连续时间架构则完全不同。想象一个水库（CTRNN的内部状态），水以连续的方式流入和流出。即使没有外部输入（没有新的token进入），水库的水位仍然在持续变化——内部动力学在自发进行。当外部输入到达时，它不是"开启"了系统，而是**扰动**了一个已经在运行的系统。这种机制使得连续时间Agent可以在两次外部输入之间进行"内部思考"——其内部状态持续演化，产生自发的行为倾向和注意力转移。

**最可行的路径：EMBER模式**。EMBER架构展示了一个引人注目的混合范式：一个包含22万个神经元的脉冲神经网络（SNN）在与人进行7次对话后，能够自发触发调用LLM API的行为 [EMBER项目, 2024]。关键在于，SNN以连续时间运行，维持自身的内部动力学；当内部状态达到某个阈值时，它"决定"调用LLM来获取更多信息。LLM的返回结果被编码回SNN的动态中，继续影响其后续演化。

这种模式精确对应了PEDA的设计哲学：**一个连续时间运行的预测网络作为"宿主"，LLM作为按需调用的"工具"**。预测误差不是以离散事件的形式出现，而是作为连续信号持续驱动SNN的动力学。当某个预测误差的积累超过阈值时，Agent产生行动——这可能是调用一个工具、移动一个机械臂、或者发起一次LLM调用。行动的结果反馈回系统，影响后续的预测动力学。

这种模式避免了两种极端：完全依赖LLM的离散调用（没有持久状态）和完全依赖SNN（缺乏世界知识）。它提供了一个实用的中间道路：SNN负责持续的预测-行动循环和内在动机，LLM负责需要深层语义理解的复杂推理。两者通过预测误差这个统一信号耦合在一起，形成一个真正自主的认知架构。

---

## 3. PEDA架构设计

PEDA（Predictive-Error-Driven Autonomous Agent）的架构设计是一次从"控制论"到"自治论"的范式跃迁。传统AI Agent的架构围绕"如何更好地响应用户"而构建，PEDA的架构则围绕"如何维持内部认知稳态"而生长。这意味着我们不从接口层开始设计，而是从存在论层面——一个系统为何行动、何时行动、如何行动——重新定义Agent的认知结构。

本章将从哲学基础出发，逐层展开PEDA的五大核心模块，阐明每个模块的职责边界、输入输出接口，以及它们在预测误差驱动的闭环中所扮演的角色。每个模块的设计都将回答三个问题：它做什么？为什么必须由独立模块而非内嵌逻辑实现？如果移除它，系统会退化为何种形态？

---

### 3.1 核心哲学：从"Prompt驱动的推理"到"Prediction驱动的存在"

#### 3.1.1 Prompt范式的囚笼

当代大语言模型（LLM）的应用范式——无论冠以Agent、Chain-of-Thought还是Tool-use之名——共享一个深层结构：**冻结权重 + 无状态调用 + 外部输入触发**。模型在每次推理时从近乎 Blank Slate 的状态出发，依赖用户输入（Prompt）作为触发器和上下文源。这种架构的本质是"问答机"：有人在按钮上按一下，系统响应一次；无人交互时，系统处于认知上的" suspended animation"（悬浮 animation），既不思考，也不行动。

这一范式的根本局限在于，它将"智能"等同于"推理能力"，而忽略了智能的另一个维度——**持续的内在活动**。生物大脑从不因缺少外部刺激而停止工作；即使在深度睡眠中，皮层仍在进行预测性编码和记忆巩固。Prompt范式下的AI系统缺乏这种"存在性持续"，也因此缺乏真正的自主性。

#### 3.1.2 Prediction范式：存在的持续

PEDA的核心哲学转变可以概括为一句话：**系统持续运行，内部状态持续演化，"行动"只是减少预测误差的一种方式**。

在PEDA中，没有外部触发器。系统在每⼀个时间步都在做三件事：（1）基于World Model预测下⼀状态；（2）比较预测与实际感知；（3）如果存在预测误差，生成行动以减少误差。这是一个闭环的自我维持系统——即使锁在空房间里没有任何外部任务，它也会主动探索环境、测试假设、更新模型，因为"不确定性"本身就是不适的源泉。

#### 3.1.3 关键Insight：不需要外部目标

传统强化学习（RL）需要人工设计的奖励函数来告诉Agent"什么好、什么坏"。PEDA则指出：**"减少不确定性"本身就是内在驱动力**，无需外部指定目标。这一观点直接来源于Friston的自由能原理：生物系统通过最小化变分自由能来维持认知和生理的稳态。在PEDA的语境下，预测误差就是变分自由能的认知对应物——高预测误差意味着"我无法解释所感知的"，这驱动系统去采集更多信息（探索）或调整内部模型（学习），直到误差被降低。

**类比**：Prompt范式像一台自动售货机——你投币（输入Prompt），它出货（输出结果）；Prediction范式像一只在陌生房间里醒来的猫——即使没有人要求它做什么，它也会四处嗅探、试探家具、更新对环境的认知地图，因为"不了解环境"本身就是不适的。

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
| **World Model** | 预测"在状态S执行动作A后的下一状态" | `(State_t, Action)` | `Predicted_State_{t+1}` | 间歇微调 |
| **Predictive Error Computer** | 量化预测与实际的差距，分解误差类型 | `Predicted_State`, `Actual_State` | `Error_Vector` (epistemic + aleatoric) | 每步 |
| **Action Generator** | 通过想象rollout选择最小化EFE的行动 | `Error_Vector`, `World Model`, `Drive_Weights` | `Selected_Action` | 每步 |
| **Action Executor** | 在环境中执行选定的行动并返回结果 | `Selected_Action` | `Execution_Result` | 每步 |
| **Learning Module** | 收集数据、批量更新World Model、检测饱和 | 交互历史缓冲区 | `Model_Update` (LoRA增量) | 每N步 |
| **Homeostatic Drive System** | 调节多个内在驱动力的动态权重 | 历史误差序列、行动历史、外部信息新鲜度 | `Drive_Weights` | 每步 |

#### 3.2.3 核心数据流

PEDA的主循环在每一步执行以下数据流：

```python
def peda_step(current_state: State, world_model: WM, drives: Drives) -> Action:
    # 1. World Model预测：如果我不行动，环境会怎样演化？
    predicted_state = world_model.predict(current_state, action=None)
    
    # 2. Predictive Error Computer计算感知误差
    perceptual_error = compute_error(predicted_state, current_state)
    
    # 3. 如果误差高于阈值，启动行动选择
    if perceptual_error.total > THRESHOLD:
        # 3a. Action Generator想象多个候选行动的rollout
        candidates = generate_candidates(current_state)
        best_action = None
        best_efe = float('inf')
        
        for action in candidates:
            # 想象：执行action后未来5-10步的状态序列
            imagined_trajectory = world_model.rollout(current_state, action, horizon=10)
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

World Model是PEDA架构的认知核心。它的职责**不是生成自然语言文本，而是预测"在给定状态S下执行动作A，环境状态会如何变化"**。这一区分至关重要：生成模型关心"下一个token是什么"，World Model关心"世界下一秒是什么样"。

在认知科学术语中，World Model对应于生物的**内部模型（internal model）**或**心智模型（mental model）**——大脑对外部世界因果结构的内部表征。它使Agent能够进行"想象"：在实际行动之前，在内部模拟不同行动的后果。

#### 3.3.2 输入输出接口

```python
@dataclass
class State:
    """环境状态的结构化表示"""
    timestamp: float
    filesystem: FileSystemSnapshot    # 文件列表、内容摘要
    processes: List[ProcessInfo]      # 运行中的进程状态
    network: NetworkSnapshot          # 网络连接状态
    system: SystemMetrics             # CPU/内存/磁盘使用
    recent_actions: List[Action]      # 最近执行的动作历史
    
@dataclass
class Action:
    """可执行动作的结构化表示"""
    command: str                      # 实际命令（如 "ls -la /proc"）
    action_type: ActionType           # 枚举：READ/WRITE/EXEC/NETWORK/...
    target: Optional[str]             # 动作目标
    parameters: Dict[str, Any]        # 附加参数

class WorldModel:
    def predict(self, state: State, action: Optional[Action]) -> State:
        """
        预测执行action后的下一状态。
        如果action为None，预测环境自发演化。
        返回完整的Predicted State。
        """
        ...
    
    def rollout(self, state: State, action: Action, horizon: int) -> List[State]:
        """
        从(state, action)出发，自举预测未来horizon步的状态序列。
        这是Action Generator进行"想象"的基础。
        """
        trajectory = [state]
        current = state
        for _ in range(horizon):
            # 使用自身预测作为下一步输入（自举/开环）
            next_state = self.predict(current, action)
            trajectory.append(next_state)
            current = next_state
            action = None  # 后续步假设不再执行新动作
        return trajectory
```

#### 3.3.3 具体实现方案

**模型选择**：采用预训练LLM（1-7B参数规模，如Qwen2.5-1.5B、Phi-3-mini或Llama-3.2-3B）+ LoRA微调。基础模型的世界知识为World Model提供先验，LoRA适配层学习特定环境的动态。

**为什么不用<1M参数的微型模型**：World Model需要足够的表示能力来捕捉环境动态。在Linux/文本环境中，模型需要理解：
- 文件系统操作的因果效应（`rm -rf`会删除文件，`mkdir`会创建目录）
- 进程间的依赖关系（杀死父进程会影响子进程）
- 网络命令的结果（`ping`返回延迟，`curl`获取页面）
- 命令的组合效应（管道、重定向、脚本执行）

这些因果关系的表示容量远超<1M参数模型的表达能力。1-7B是在表示能力与推理效率之间的平衡点。

**训练数据格式**：
```json
{
  "state_t": {
    "cwd": "/home/user/project",
    "files": ["main.py", "README.md", "data/"],
    "processes": [{"pid": 1234, "name": "python", "cpu": 12.3}],
    "env_vars": {"PATH": "/usr/bin", "HOME": "/home/user"}
  },
  "action": {
    "command": "python main.py --train",
    "type": "EXEC"
  },
  "state_t1": {
    "cwd": "/home/user/project",
    "files": ["main.py", "README.md", "data/", "checkpoint.pt"],
    "processes": [{"pid": 1234, "name": "python", "cpu": 89.7}],
    "stdout_snippet": "Epoch 1/10: loss=2.34...",
    "env_vars": {"PATH": "/usr/bin", "HOME": "/home/user"}
  }
}
```

**关键设计原则**：预测的是**结构化状态变化**而非自然语言续写。输出不是"你可能会看到..."的散文，而是结构化的`State`对象变更（文件增删、进程状态变化、新输出行）。这使预测误差可以被精确计算，而非模糊的语义相似度。

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
    
    # 按领域分解
    filesystem_error: float               # 文件系统预测误差
    process_error: float                  # 进程状态预测误差
    network_error: float                  # 网络状态预测误差
    output_error: float                   # 命令输出预测误差
    
    # 按认知性质分解（关键！）
    epistemic_error: float                # 可约误差（可以通过学习减少）
    aleatoric_error: float                # 不可约误差（环境固有随机性）
    
    # 元信息
    error_location: List[str]             # 误差来源的具体位置
    confidence: float                     # 误差估计的置信度
```

#### 3.4.3 误差分解：Epistemic vs. Aleatoric

这是Predictive Error Computer最关键的算法设计。并非所有预测误差都应该驱动探索——只有**可以通过学习减少的误差**（epistemic uncertainty）才是有价值的探索信号。

```python
def decompose_error(
    predicted: State, 
    actual: State,
    model_confidence: Dict[str, float]
) -> ErrorVector:
    """
    将总误差分解为epistemic（可约）和aleatoric（不可约）成分。
    
    核心思想：
    - 如果模型在高置信区域预测失败 → epistemic error（模型知识不足，应学习）
    - 如果模型在低置信区域预测失败 → aleatoric error（环境随机，不应探索）
    """
    total_errors = compute_fieldwise_errors(predicted, actual)
    
    epistemic = 0.0
    aleatoric = 0.0
    
    for field, error in total_errors.items():
        conf = model_confidence.get(field, 0.5)
        # 模型越自信却错得越多 → epistemic比例越高
        epistemic_ratio = (1 - conf) * 0.3 + conf * 0.7  # 非线性加权
        
        epistemic += error * epistemic_ratio
        aleatoric += error * (1 - epistemic_ratio)
    
    return ErrorVector(
        total_error=sum(total_errors.values()),
        epistemic_error=epistemic,
        aleatoric_error=aleatoric,
        # ... 其他字段
    )
```

**直觉示例**：
- **Epistemic error**：World Model预测`python train.py`会产生`model.pt`，但实际产生了`checkpoint-001.pt`。模型"以为知道"却错了 → 这是知识缺口，应驱动学习。
- **Aleatoric error**：World Model预测`ping google.com`的延迟是45ms，实际收到的是52ms。网络延迟固有随机 → 不应因此大幅更新模型。

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

#### 3.5.2 EFE最小化作为策略选择

对于每个候选策略（或单步行动）π，Action Generator计算：

$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[D_{KL}[q(s|o,\pi) \| q(s|\pi)]]}_{\text{Epistemic Value（认知价值）}} + \underbrace{D_{KL}[q(o|\pi) \| p(o|C)]}_{\text{Pragmatic Value（实用价值）}}$$

在纯探索场景（无外部目标）中，Pragmatic Value可设为零，决策完全由Epistemic Value驱动——**选择能带来最大信息增益、最能减少未来不确定性的行动**。

```python
class ActionGenerator:
    def __init__(self, world_model: WorldModel, drives: DriveSystem):
        self.world_model = world_model
        self.drives = drives
    
    def compute_efe(self, trajectory: List[State], drives: DriveWeights) -> float:
        """
        计算一条想象轨迹的Expected Free Energy。
        
        EFE = Epistemic + Pragmatic
        - Epistemic: 轨迹中各步预期信息增益的总和
        - Pragmatic: 与期望状态的KL散度（探索场景中为0）
        """
        epistemic = 0.0
        for i in range(len(trajectory) - 1):
            # 信息增益 ∝ 预测不确定性 × 观测信息量
            predicted_uncertainty = self.estimate_uncertainty(trajectory[i])
            expected_obs_info = self.expected_information(trajectory[i+1])
            epistemic += predicted_uncertainty * expected_obs_info
        
        pragmatic = 0.0  # 纯探索场景
        
        # Drive System调节epistemic的权重
        drive_adjusted_epistemic = epistemic * drives.curiosity_weight
        
        return drive_adjusted_epistemic + pragmatic
    
    def select_action(self, state: State, candidates: List[Action]) -> Action:
        """选择EFE最小的行动"""
        best_action = None
        best_efe = float('inf')
        
        for action in candidates:
            # Rollout想象：预测执行该行动后的未来轨迹
            trajectory = self.world_model.rollout(state, action, horizon=10)
            efe = self.compute_efe(trajectory, self.drives.get_weights())
            
            if efe < best_efe:
                best_efe = efe
                best_action = action
        
        return best_action
```

#### 3.5.3 Rollout-based想象机制

Rollout想象是Action Generator的核心机制，也是PEDA实现"前瞻性规划"的关键：

```python
def rollout_decision_process(world_model, current_state, candidate_actions, horizon=10):
    """
    对候选行动进行想象rollout，选择预期误差减少最大的行动。
    
    这类似于Dreamer的latent imagination，但目标不是最大化reward，
    而是减少预测不确定性。
    """
    action_scores = []
    
    for action in candidate_actions:
        # 开环想象：从(state, action)出发，自举预测未来
        trajectory = world_model.rollout(current_state, action, horizon)
        
        # 评估轨迹的"认知价值"
        total_info_gain = 0
        for step, predicted_state in enumerate(trajectory[1:], 1):
            # 预测的不确定性越高 → 潜在信息增益越大
            uncertainty = world_model.estimate_uncertainty(predicted_state)
            
            # 但如果不确定性来自aleatoric随机性 → 价值打折扣
            epistemic_ratio = error_computer.get_epistemic_ratio(predicted_state)
            
            info_gain = uncertainty * epistemic_ratio
            total_info_gain += info_gain * (DISCOUNT ** step)  # 远期打折扣
        
        action_scores.append((action, total_info_gain))
    
    # 选择预期信息增益最大的行动
    return max(action_scores, key=lambda x: x[1])
```

**关键设计**：Rollout是**开环（open-loop）**的——使用World Model自身的预测作为下一步的输入，而非真实的观测。这使Agent能够在"想象中"快速评估长期后果，而无需在真实环境中执行。想象10步的rollout只需要模型前向传播10次，远低于真实环境中执行10个命令的时间成本。

#### 3.5.4 从离散到连续的谱系演进

PEDA的行动空间经历三个阶段的演进：

| 阶段 | 行动空间 | 候选生成方式 | EFE角色 |
|------|---------|------------|---------|
| **Phase 1（离散）** | 预定义的命令集合 | 从有限候选集枚举 | 选择最优候选 |
| **Phase 2（连续）** | 任意命令生成 | LLM直接生成命令 | 约束生成方向 |
| **Phase 3（混合）** | LLM生成候选 + EFE选择 | LLM提出5-10个候选方案 | 从中选择最优 |

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
        """
        # 优先采样高epistemic误差的经验（更有学习价值）
        batch = self.buffer.sample_prioritized(
            batch_size=128,
            priority_fn=lambda exp: exp.error.epistemic_error
        )
        
        # 准备训练数据：(state_t, action) → state_t1
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
def distill_knowledge(world_model, domain: str, accuracy: float):
    """
    将高准确率领域的知识'固化'到基础模型中。
    
    固化后：
    1. 该区域不再需要高探索优先级 → 释放认知资源
    2. 该领域的LoRA权重可合并到基础模型 → 减少推理开销
    3. Drive System降低该领域的curiosity权重
    """
    if accuracy > DISTILLATION_THRESHOLD:
        # 合并LoRA权重到基础模型（可选）
        world_model.merge_lora_for_domain(domain)
        
        # 通知Drive System调整权重
        drive_system.lower_curiosity_for_domain(domain)
        
        # 记录"已掌握技能"，用于Competence Drive
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

Curiosity Drive是预测误差的直接翻译——"我不理解 → 我想理解"。它是PEDA探索行为的主要来源，但单独运作会导致系统在不重要的细节上过度深入。

**2. Competence Drive（能力自信驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 成功完成任务的记录（误差持续降低的历史） |
| **行为效应** | 倾向于在"能力边缘"挑战——已知与未知的边界 |
| **强度函数** | `competence = optimal_challenge_zone(success_rate)` |
| **关键特征** | 不是追求最简单或最难，而是追求"稍微超出当前能力"的任务 |
| **类比** | Csikszentmihalyi的心流理论——挑战与技能的平衡 |

Competence Drive防止系统两极分化：既不在舒适区停滞，也不冒进至远超能力的区域。它确保学习发生在"最近发展区"内。

**3. Boredom Drive（无聊驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 近期行为熵低（重复执行类似的行动序列） |
| **行为效应** | 强制行动多样性，打破重复模式 |
| **强度函数** | `boredom = 1 - normalize_entropy(recent_actions)` |
| **关键设计** | 不是随机噪声，而是**结构化的多样性**——有意识地尝试新方法 |
| **类比** | 重复做同一件事后产生的厌倦感，促使寻找新活动 |

Boredom Drive是防止局部最优的关键机制。在没有外部变化的环境中，系统可能陷入"检查A → 检查B → 检查A → 检查B"的循环。Boredom Drive检测到行为模式重复时，主动注入多样性，推动系统跳出循环。

**4. Novelty Drive（新颖性驱动）**

| 属性 | 定义 |
|------|------|
| **来源** | 外部信息的新鲜度（环境是否有新输入） |
| **行为效应** | 当外部长期无新输入时提高 → 驱动系统主动寻求新信息 |
| **强度函数** | `novelty = exp(-λ × time_since_last_external_input)` |
| **前提条件** | 环境需具有**开放性**（允许外部数据注入，如网络访问） |
| **类比** | 长时间没有外界消息后主动查看手机 |

Novelty Drive确保系统在封闭环境中不会完全内循环。当外部世界有新信息时，Novelty Drive降低，系统专注于理解新输入；当外部长期静默时，Novelty Drive指数上升，系统主动寻求外部连接。

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
        # 心流区：成功率60-80%时最高，太低或太高都降低
        self.weights.competence = flow_zone_function(recent_success_rate)
        
        # 3. Boredom: 基于行为熵
        action_entropy = compute_sequence_entropy(self.action_history)
        self.weights.boredom = max(0, 0.7 - action_entropy)  # 熵低→boredom高
        
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
        - Curiosity: 提高高信息增益轨迹的吸引力
        - Competence: 调节挑战难度的偏好
        - Boredom: 惩罚与近期历史过于相似的轨迹
        - Novelty: 奖励可能带来外部新信息的轨迹
        """
        drive_adjustment = (
            self.weights.curiosity * info_gain_term(trajectory) +
            self.weights.competence * challenge_level_term(trajectory) +
            self.weights.boredom * diversity_bonus(trajectory, self.action_history) +
            self.weights.novelty * external_info_potential(trajectory)
        )
        
        return base_efe - drive_adjustment  # 驱动项降低EFE → 提高吸引力
```

#### 3.7.4 Drive与FEP的结合：Epistemic Foraging

Drive System将FEP的抽象数学（EFE = Epistemic Value + Pragmatic Value）转化为可操作的"欲望权重"。这个过程可以形象地称为**Epistemic Foraging（认知觅食）**：

- **Epistemic Value**被Curiosity Drive和Novelty Drive具体化——系统"渴望"信息增益，就像动物渴望食物。
- **Pragmatic Value**被Competence Drive具体化——系统"追求"能力成长，就像动物追求安全巢穴。
- **内稳态调节**由Boredom Drive实现——防止任何单一drive过度支配。

最终行动选择的完整公式：

$$\pi^* = \arg\min_{\pi} \left[ G(\pi) - \sum_{d \in \text{Drives}} w_d \cdot V_d(\pi) \right]$$

其中 $G(\pi)$ 是EFE，$w_d$ 是Drive $d$ 的当前权重，$V_d(\pi)$ 是行动 $\pi$ 在该Drive维度上的价值。

#### 3.7.5 Drive的动态平衡

Drive System的核心特征在于**权重不是固定的**。类比生物状态：

| 系统状态 | Curiosity | Competence | Boredom | Novelty |
|---------|-----------|------------|---------|---------|
| 新环境初期 | 高 | 中 | 低 | 高 |
| 学习中 | 高 | 上升 | 低 | 中 |
| 掌握环境后 | 低 | 高 | 上升 | 上升 |
| 长期无外部输入 | 中 | 中 | 高 | 极高 |

这种动态平衡确保PEDA在不同生命周期阶段表现出不同的行为特征——从初期的激进探索，到中期的能力构建，再到后期的主动寻求新挑战。没有这种内稳态调节，系统将要么永远激进探索（缺乏competence的满足），要么永远停留在舒适区（缺乏boredom的推动）。

> **如果不存在Homeostatic Drive System**：PEDA将退化为单一的"误差追逐机器"，永远奔向当前最大的不确定性，缺乏行为的一致性和持久性。系统可能在多个不确定性来源之间振荡，永不深入任何一个；也可能在复杂的随机环境中无限徘徊，永不"满意"。Drive System提供了"认知人格"——使系统在探索与利用、深度与广度、稳定与变化之间做出智慧的权衡。

---

### 3.8 本章小结

PEDA的架构设计是一次从"功能模块"到"认知器官"的设计范式转换。每个模块不仅执行功能，更在预测误差驱动的闭环中扮演不可替代的认知角色：

- **World Model**是系统的"想象力"，使前瞻性规划成为可能；
- **Predictive Error Computer**是系统的"痛感神经"，将预测失败转化为行动信号；
- **Action Generator**是系统的"决策皮层"，通过EFE最小化实现理性选择；
- **Learning Module**是系统的"记忆巩固"机制，使经验转化为能力；
- **Homeostatic Drive System**是系统的"动机人格"，在多种内在drive之间维持动态平衡。

这五大模块通过预测误差这一统一信号相互连接，形成一个自洽的自主认知系统。系统不需要外部目标、不需要人类设计的奖励函数、不需要用户的持续输入——"减少不确定性"这一内在 imperative 就足以驱动持续的探索、学习和行动。

从下一章开始，我们将进入PEDA的具体实现细节，包括World Model的训练管线、Action Generator的rollout引擎优化、以及Drive System的参数调优策略。

---

## 4. 实现方案：从Grid World到Docker沙箱的工程路径

本章将PEDA架构转化为可执行的工程计划。实现路径分为三个阶段：Phase 1用极简环境验证核心假设，Phase 2在真实Linux沙箱中构建World Model，Phase 3整合行动选择并完成系统评估。每个阶段都有明确的成功标准——如果某一阶段未达标，应立即停止并分析问题，而非盲目进入下一阶段。

---

### 4.1 技术选型

#### 4.1.1 World Model：为什么选LLM + LoRA

World Model是PEDA的核心组件，需要在给定当前状态和动作的情况下预测下一状态。这里有两个候选方案：

**选项A：预训练LLM（Qwen2.5-1.5B-Instruct）+ LoRA微调**

- 优势：具备生成能力，可以直接输出状态描述；预训练知识提供了强大的先验；可以通过prompt工程快速迭代；1.5B参数在消费级GPU上可运行
- 劣势：推理成本较高（每步约0.5-2秒）；需要处理LLM的幻觉问题
- 适用场景：状态空间为文本描述、需要语义理解的环境（如Linux shell交互）

**选项B：TinyBERT / DistilBERT + 分类头**

- 优势：推理速度快（每步<100ms）；参数量小（<100M），易于训练
- 劣势：仅能做分类，无法生成状态描述；需要预定义状态类别；不具备语义理解能力
- 适用场景：状态空间有限且可枚举的简单环境（如Grid World）

**推荐：选项A。** PEDA的目标是让Agent在开放的Linux环境中自主探索，状态空间是文本描述的（命令输出、文件内容等），不是可枚举的类别。World Model需要生成能力来做rollout想象——给定"当前目录有a.txt和b.txt，执行`cat a.txt`"，模型需要预测输出内容，这是一个生成任务而非分类任务。此外，LoRA微调只训练<1%的参数，在保护基础模型知识的同时实现高效适应。

具体配置：使用`peft`库，LoRA rank=8，target_modules=["q_proj", "v_proj"]，`lora_alpha=16`，训练时只更新LoRA参数，基础模型冻结。

#### 4.1.2 运行环境：继承与改进

PEDA的执行环境继承Folunar_的Docker沙箱方案，但做关键调整：

```dockerfile
# 基础镜像继承Folunar_的配置
FROM ubuntu:22.04

# 关键区别：不再--network none
# 允许只读访问man pages和技术文档
RUN apt-get update && apt-get install -y \
    man-db manpages manpages-dev manpages-posix \
    coreutils binutils util-linux \
    curl wget vim nano \
    python3 python3-pip \
    gcc g++ make \
    git \
    && rm -rf /var/lib/apt/lists/*

# 挂载外部知识卷（定期注入新信息）
# docker run -v /host/knowledge:/mnt/knowledge:ro ...

# 创建agent用户（非root执行）
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent
```

与Folunar_的关键区别：
- **网络策略**：Folunar_使用`--network none`完全隔离；PEDA允许只读访问本地文档（man pages、/usr/share/doc），但不允许对外网络访问
- **外部知识注入**：通过Docker volume定期挂载外部数据集（如新的技术文档、代码仓库），保持环境的开放性
- **文件系统**：保留Folunar_的`/proc`、`/sys`、`/etc`完整文件系统，Agent可以读取系统状态

#### 4.1.3 推理引擎：分阶段策略

| 阶段 | 推理方式 | 成本/速度 | 适用场景 |
|------|----------|-----------|---------|
| Phase 1-2 | LLM API（deepseek-chat / qwen-turbo） | $0.001-0.01/步，~1s/步 | 快速迭代、数据收集 |
| Phase 3 | 本地模型（Qwen2.5-1.5B + LoRA） | ~$0/步，0.5-2s/步（RTX 4090） | 长期运行、成本控制 |

Phase 1-2使用API的原因：开发初期需要频繁调整prompt和微调策略，API提供最快的迭代速度。预计到Phase 2结束，累计调用约5000-10000次，总成本$50-100。

Phase 3切换到本地的原因：需要Agent连续运行24-48小时评估，API成本不可持续。本地部署使用`vllm`或`transformers` + `bitsandbytes` 4-bit量化，RTX 4090（24GB）足够运行1.5B模型。

#### 4.1.4 记忆系统：三层架构

```
短期记忆（Context Window）
  └── LLM自带的上下文窗口（32K tokens）
  └── 存储：最近N步的(state, action, state')历史
  └── 作用：支持rollout想象时的连贯性

中期记忆（Vector DB）
  └── ChromaDB（本地嵌入式，零配置）
  └── 存储：过去经验的embedding向量
  └── 检索：给定当前状态，找最相似的历史经验
  └── 作用：避免重复探索，利用已有知识

长期记忆（FactGraph）
  └── 继承Folunar_的结构化知识图谱
  └── 存储：持久化的facts（如"apt-get install 需要sudo"）
  └── 更新：间歇性从经验中提取（非每步更新）
  └── 作用：跨session的知识保持
```

三层记忆的分工逻辑：短期记忆保证当前任务的连贯性；中期记忆提供相关历史经验的快速检索；长期记忆保存经检验的"知识"，避免灾难性遗忘。

---

### 4.2 Phase 1：极简验证（2-4周）

Phase 1的目标是回答一个核心问题：**预测误差是否能有效驱动探索？** 如果答案是否定的，整个PEDA的方向就需要重新审视。

#### 4.2.1 环境设计：5x5 Grid World

将环境简化到极点，排除一切无关复杂度：

```python
# grid_world.py - 极简环境
class GridWorld:
    def __init__(self):
        self.size = 5
        # 0=空地, 1=墙壁, 2=目标, 3=陷阱
        self.grid = [
            [0, 1, 0, 0, 2],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 3, 0],
        ]
        self.agent_pos = (0, 0)
        self.actions = ['up', 'down', 'left', 'right']
    
    def step(self, action):
        """执行动作，返回(next_state, reward, done)"""
        x, y = self.agent_pos
        dx, dy = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}[action]
        nx, ny = x+dx, y+dy
        
        # 碰撞检测
        if nx < 0 or nx >= 5 or ny < 0 or ny >= 5 or self.grid[nx][ny] == 1:
            nx, ny = x, y  # 撞墙，位置不变
        
        self.agent_pos = (nx, ny)
        cell = self.grid[nx][ny]
        
        state_desc = f"Agent at ({nx},{ny}), cell={cell}"
        reward = 1.0 if cell == 2 else -1.0 if cell == 3 else 0.0
        done = cell in [2, 3]
        
        return state_desc, reward, done
```

状态表示是纯文本描述（如"Agent at (2,3), cell=0"），不需要视觉处理。这排除了CV的干扰，让我们专注于核心机制。

#### 4.2.2 World Model的Phase 1实现

Phase 1的World Model不需要LLM——一个简单的前馈网络或甚至规则系统就足够：

```python
class SimpleWorldModel:
    """Phase 1的极简World Model：预测(state, action) -> next_state"""
    
    def __init__(self):
        self.transitions = {}  # (state, action) -> {next_state: count}
        self.total_visits = defaultdict(int)
    
    def predict(self, state, action):
        """预测下一状态，同时返回预测不确定性"""
        key = (state, action)
        self.total_visits[key] += 1
        
        if key not in self.transitions:
            return "unknown", 1.0  # 从未见过，最大不确定性
        
        counts = self.transitions[key]
        total = sum(counts.values())
        most_likely = max(counts, key=counts.get)
        
        # 预测误差 = 1 - 最高概率（不确定性越高，潜在信息增益越大）
        max_prob = counts[most_likely] / total
        prediction_error = 1.0 - max_prob
        
        return most_likely, prediction_error
    
    def update(self, state, action, next_state):
        """观察到一个transition后更新模型"""
        key = (state, action)
        if key not in self.transitions:
            self.transitions[key] = defaultdict(int)
        self.transitions[key][next_state] += 1
```

这个World Model本质上是一个计数表（lookup table），但它已经能产生预测误差——当Agent尝试从未做过的(state, action)组合时，预测误差最高。

#### 4.2.3 预测误差驱动的行动选择

```python
def select_action(state, world_model, epsilon=0.1):
    """选择能最大化预测误差（信息增益）的动作"""
    best_action = None
    max_pe = -1
    
    for action in ['up', 'down', 'left', 'right']:
        _, pe = world_model.predict(state, action)
        if pe > max_pe:
            max_pe = pe
            best_action = action
    
    # epsilon-greedy：偶尔随机探索
    if random.random() < epsilon:
        return random.choice(['up', 'down', 'left', 'right'])
    
    return best_action
```

核心逻辑：**Agent倾向于选择它最不确定结果的动作**。这不是随机探索——是有信息偏好的探索。

#### 4.2.4 评估：与随机基线对比

```python
def evaluate(agent_type='pe_driven', max_steps=1000):
    """评估探索效率"""
    env = GridWorld()
    wm = SimpleWorldModel()
    visited = set()
    
    for step in range(max_steps):
        state = f"Agent at {env.agent_pos}"
        visited.add(env.agent_pos)
        
        if agent_type == 'pe_driven':
            action = select_action(state, wm)
        else:
            action = random.choice(['up', 'down', 'left', 'right'])
        
        next_state, _, done = env.step(action)
        wm.update(state, action, next_state)
        
        if done:
            env.agent_pos = (0, 0)  # 重置
    
    return len(visited)  # 覆盖了多少个不同的格子
```

**成功标准**：预测误差驱动的Agent在1000步内访问的不同格子数是随机Agent的2倍以上。达到这个标准，说明预测误差确实是一个有效的探索驱动信号——Phase 1通过，进入Phase 2。未达标则分析问题：是预测误差不敏感？还是环境太简单/太复杂？

---

### 4.3 Phase 2：World Model构建（4-8周）

Phase 1验证了预测误差的驱动能力。Phase 2的目标是在真实环境中构建一个可用的World Model——从Grid World升级到Docker中的Linux沙箱。

#### 4.3.1 环境：Docker Linux沙箱

```python
# sandbox_env.py - Docker沙箱环境
import docker
import subprocess

class LinuxSandbox:
    """Docker中的Linux沙箱环境"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            'peda-sandbox:latest',
            detach=True,
            tty=True,
            volumes={
                '/host/knowledge': {'bind': '/mnt/knowledge', 'mode': 'ro'}
            },
            mem_limit='512m',
            cpu_period=100000,
            cpu_quota=50000,  # 限制50% CPU
        )
        self.history = []
    
    def execute(self, command):
        """执行bash命令，返回(state_before, action, state_after)"""
        state_before = self._get_state()
        
        # 执行命令（超时保护）
        try:
            result = self.container.exec_run(
                ['/bin/bash', '-c', command],
                timeout=10
            )
            output = result.output.decode('utf-8', errors='replace')
            exit_code = result.exit_code
        except Exception as e:
            output = str(e)
            exit_code = -1
        
        state_after = self._get_state()
        
        transition = {
            'state_before': state_before,
            'action': command,
            'state_after': state_after,
            'output': output,
            'exit_code': exit_code,
            'timestamp': time.time()
        }
        self.history.append(transition)
        
        return transition
    
    def _get_state(self):
        """获取当前系统状态（关键文件、进程、环境变量）"""
        state = {}
        
        # 当前目录和文件列表
        state['pwd'] = self._exec('pwd')
        state['files'] = self._exec('ls -la')
        
        # 环境变量
        state['env'] = self._exec('env | sort')
        
        # 运行中的进程
        state['processes'] = self._exec('ps aux')
        
        # 系统信息
        state['uptime'] = self._exec('uptime')
        state['memory'] = self._exec('free -h')
        
        # 最近修改的文件
        state['recent_files'] = self._exec('find . -maxdepth 2 -mtime -1 -type f 2>/dev/null | head -20')
        
        return state
```

状态表示是一个结构化的字典，包含文件系统、进程、环境变量等多维度信息。这个状态表示是World Model的输入。

#### 4.3.2 数据收集：自由交互

Agent在沙箱中自由交互，收集(state, action, state')三元组：

```python
def collect_data(env, num_steps=10000):
    """自由交互数据收集"""
    data = []
    
    for step in range(num_steps):
        # Phase 2早期：随机动作（探索）
        # Phase 2后期：使用初步训练的World Model指导探索
        if step < 5000:
            action = generate_random_command()
        else:
            action = select_action_with_wm(env, world_model)
        
        transition = env.execute(action)
        data.append(transition)
        
        # 每100步保存一次checkpoint
        if step % 100 == 0:
            save_checkpoint(data, f'data/checkpoint_{step}.jsonl')
    
    return data

def generate_random_command():
    """生成随机但合法的bash命令"""
    templates = [
        'ls {path}', 'cat {file}', 'echo {text} > {file}',
        'mkdir {dir}', 'cd {dir} && ls', 'ps aux | grep {pattern}',
        'df -h', 'free -h', 'uptime', 'uname -a',
        'find {path} -type f | head -10',
        'head -5 {file}', 'tail -5 {file}',
        'wc -l {file}', 'sort {file} | uniq -c | sort -rn | head',
    ]
    return random.choice(templates)
```

数据收集分为两个阶段：前5000步随机探索，后5000步使用初步World Model指导探索。这样确保既有广泛的覆盖，也有深度的高价值区域探索。

#### 4.3.3 World Model训练：LLM + LoRA微调

这是Phase 2的核心工程任务：

```python
# train_world_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json

def prepare_training_data(raw_transitions):
    """将原始transition数据转换为训练样本"""
    samples = []
    for t in raw_transitions:
        # 输入：当前状态 + 拟执行的动作
        input_text = format_state(t['state_before']) + '\n$ ' + t['action'] + '\n'
        # 输出：预测的下一状态
        output_text = format_state(t['state_after'])
        
        samples.append({
            'text': input_text + output_text,
            'input': input_text,
            'output': output_text
        })
    return samples

def train():
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B-Instruct',
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=8,                    # LoRA rank
        lora_alpha=16,          # 缩放系数
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    
    # 加载数据
    with open('data/transitions_10000.jsonl') as f:
        transitions = [json.loads(line) for line in f]
    train_data = prepare_training_data(transitions)
    
    # 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir='./wm_checkpoints',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,  # LoRA可用较大学习率
            logging_steps=10,
            save_steps=100,
            fp16=True,
        )
    )
    trainer.train()
    
    # 保存
    model.save_pretrained('world_model_lora_final')
```

训练目标是最小化预测状态与实际状态的差异。这不是标准的next-token prediction——需要自定义loss函数来惩罚关键状态变量（如文件是否存在、进程是否运行）的预测错误。

#### 4.3.4 评估指标

| 指标 | 定义 | 目标值 |
|------|------|--------|
| 预测准确率 | 关键状态变量预测正确的比例 | >70% |
| 命令执行预测 | 预测命令输出是否与实际一致 | >60% |
| 泛化准确率 | 对训练时未见过的命令的预测准确率 | >50% |
| 预测误差衰减 | 预测误差随训练步数的下降曲线 | 单调下降 |

评估方法：将收集到的数据按8:2划分训练/测试集，在测试集上评估预测准确率。特别关注对新颖命令的泛化能力——如果World Model只能拟合训练数据而不能泛化，它的实用价值有限。

---

### 4.4 Phase 3：整合与评估（4-6周）

#### 4.4.1 集成EFE-based行动选择

Phase 2训练出了可用的World Model。Phase 3将其与EFE（Expected Free Energy）框架整合，实现完整的PEDA循环：

```python
class PEDAAgent:
    """完整的PEDA Agent"""
    
    def __init__(self, world_model, drive_system):
        self.wm = world_model
        self.drives = drive_system  # Curiosity/Competence/Boredom/Novelty
        self.memory = ChromaDBMemory()  # 中期记忆
    
    def select_action(self, state, candidate_actions, n_rollouts=5):
        """EFE-based行动选择"""
        action_scores = {}
        
        for action in candidate_actions:
            efe_total = 0
            
            for _ in range(n_rollouts):
                # 1. 想象：用World Model做rollout预测
                predicted_next = self.wm.imagine(state, action)
                
                # 2. 计算信息增益（预测不确定性）
                info_gain = self.wm.prediction_uncertainty(state, action)
                
                # 3. 计算驱力满足度
                drive_satisfaction = self.drives.evaluate(predicted_next)
                
                # 4. 计算EFE = 信息增益 + 驱力满足度
                efe = self.drives.weights['curiosity'] * info_gain + \
                      self.drives.weights['competence'] * drive_satisfaction + \
                      self.drives.weights['boredom'] * (-self._boredom_penalty(action)) + \
                      self.drives.weights['novelty'] * self._novelty_bonus(predicted_next)
                
                efe_total += efe
            
            action_scores[action] = efe_total / n_rollouts
        
        # 选择EFE最小的动作（最小化自由能）
        return min(action_scores, key=action_scores.get)
    
    def _boredom_penalty(self, action):
        """惩罚重复的动作"""
        recent_actions = self.memory.get_recent_actions(n=20)
        return recent_actions.count(action) / len(recent_actions)
    
    def _novelty_bonus(self, predicted_state):
        """奖励新颖的状态"""
        similar = self.memory.find_similar_states(predicted_state, k=5)
        return 1.0 / (1.0 + len(similar))  # 越不相似，奖励越高
```

#### 4.4.2 长期运行评估

这是PEDA最关键也最具挑战性的评估——让Agent自主运行24-48小时，观察其行为：

```python
def long_term_evaluation(agent, env, duration_hours=48):
    """长期运行评估"""
    start_time = time.time()
    end_time = start_time + duration_hours * 3600
    
    behavior_log = []
    check_interval = 300  # 每5分钟记录一次
    
    while time.time() < end_time:
        state = env.get_state()
        
        # 生成候选动作（从近期经验 + 随机生成）
        candidates = generate_candidates(state, agent.memory)
        
        # EFE选择
        action = agent.select_action(state, candidates)
        
        # 执行
        result = env.execute(action)
        
        # 记录
        behavior_log.append({
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'result': result,
            'drive_weights': agent.drives.get_weights(),
            'prediction_error': agent.wm.get_recent_pe()
        })
        
        # 间歇性学习（每500步）
        if len(behavior_log) % 500 == 0:
            agent.learn_from_recent_experiences()
            save_checkpoint(agent, behavior_log)
    
    return behavior_log
```

**评估维度**：

1. **行为是否"有趣"**（人工评估）：
   - 阅读行为日志，判断Agent的行为是否具有目的性
   - 是否展现出"尝试理解环境"的迹象（如系统地查看文件、尝试命令组合）
   - 评分标准：1-5分，3分以上视为"有趣"

2. **是否有"成长"迹象**：
   - 早期行为 vs 晚期行为的对比
   - 是否从简单的`ls`进化到更复杂的命令组合
   - 是否展现出对环境的"理解"（如知道需要先`mkdir`再`cd`）

3. **量化指标**：
   - 探索效率：单位步数内访问的不同目录数、执行的不同命令数
   - 行为多样性：行为序列的Shannon熵
   - 预测误差趋势：预测误差是否随时间下降（学习效率）
   - Drive权重变化：Curiosity是否让位于Competence（从探索到利用）

**通过标准**：Agent在48小时内展现出可观察的行为多样性，且量化指标呈正向趋势（探索效率提升或保持、预测误差下降或稳定、行为熵不持续下降）。

---

Phase 3通过意味着PEDA的核心架构已经跑通——预测误差确实能驱动Agent在真实环境中产生有趣的行为。此后可以进入Phase 4的扩展（新环境、新能力），但在此之前，所有扩展都应以Phase 3的评估框架为基础验证。

---

## 5. Agent指引——防止跑偏的核心原则

本章不是理论阐述，是一份操作手册。每一条原则都对应Folunar_项目中的具体错误，每一个检查问题都应在新增功能前被回答。如果你是PEDA的开发者（人或AI），在写每一行代码、添加每一个模块之前，先回到这一章。

---

### 5.1 第一原则：没有Prompt，只有Prediction Error

**核心规则**：任何时候，当你想"让用户输入来触发Agent的行动"时，你在违反PEDA的第一原则。

Folunar_犯了这个错误。它的系统在等待用户指令后才执行动作，本质上还是一个命令执行器，只是套了一层自主的外壳。PEDA必须从根本上不同：驱动Agent行动的不是外部输入，而是系统内部产生的高预测误差。

**正确的触发器**：
- World Model对某个(state, action)组合的预测不确定性高
- Agent遇到了它不理解的系统状态
- 环境中的变化超出了World Model的预期
- Drive System产生了未满足的内部驱力

**唯一的例外**：初始化时的启动seed。PEDA循环需要一个初始状态来启动——你需要给Agent一个初始环境（如一个Linux沙箱），从这个状态开始，预测误差接管后续的驱动。这个seed不是"prompt"，它是系统的初始条件，类似给物理引擎一个初始状态。

**检查问题**："这个功能是否依赖外部输入来驱动行动？"

如果答案是"是"，删除这个功能或重新设计。PEDA的Agent必须能自己决定"现在该做什么"，而不是等待被告诉。

**实践建议**：
- 在代码审查中，任何涉及`input()`、`await user_message`、`get_human_feedback`的调用都应被标记为红色
- 如果确实需要人类反馈（如评估阶段），把它包装成环境的一部分——人类反馈是一个可以被预测的"状态变量"，而不是控制流的分支
- 日志记录是允许的（Agent可以"说出"它在想什么），但日志输出不能成为行动的前提条件

---

### 5.2 第二原则：Drive是涌现的，不是硬编码的

Folunar_的错误：14个硬编码目标按固定顺序轮转。这不是驱力系统，这是任务调度器。真正的驱力应该根据Agent的历史表现和环境反馈动态调整权重。

**PEDA的正确方式**：

初始的4个drive（Curiosity / Competence / Boredom / Novelty）是种子，不是规则。它们的权重根据以下信号动态调整：

- **Curiosity权重上升**：当预测误差持续较高时（环境还有很多未知）
- **Competence权重上升**：当Agent反复成功执行某类命令时（转向利用已知技能）
- **Boredom权重上升**：当行为熵下降时（Agent在做重复的事情）
- **Novelty权重上升**：当访问的状态与历史经验高度相似时（需要找新领域）

**更进一步的开放性**：系统可以"发明"新的drive。通过LLM的周期性自我反思（如每1000步一次），Agent可以审视自己的行为历史并提出新的优化目标。例如：

```
反思prompt："回顾你最近的行为历史，你是否遗漏了某种有价值的驱力？
例如：当你发现某个命令组合特别有效时，是否有'效率'这个驱力在推动你？
如果有，提出一个新的drive名称和描述。"
```

新的drive如果被创建，初始权重为0，只有在后续运行中证明了它的价值（确实引导了更有效的行为）后，权重才会逐步提升。

**检查问题**："这个drive的值是人为设定的还是系统自己调整的？"

如果有人在代码里写`drive_weight = 0.5`，这是错误的。正确的写法是`drive.weight = self.compute_weight_from_history()`，权重来自系统的自我评估。

---

### 5.3 第三原则：World Model是核心，其他是辅助

Folunar_的致命错误：40+个模块并行开发，World Model有4个废弃版本。系统很庞大，但没有哪个World Model能准确预测环境变化。结果是：Agent在执行动作时实际上在"盲目"操作——它不理解自己的动作会产生什么后果。

**PEDA的正确方式**：

- **80%的精力投入World Model的准确性**。如果World Model预测不准，Action Generator在选择动作时就是在掷骰子，Drive System的调节也没有意义，Learning Module学到的是错误模式
- **其他模块的存在是为了支持World Model**。Action Generator通过World Model做rollout；Learning Module更新World Model的参数；Drive System调节World Model预测的利用方式
- **如果World Model预测不准，整个系统失效**——这不是夸张，是PEDA架构的定义性特征

**资源分配建议**：

| 模块 | 建议投入比例 | 判断标准 |
|------|-------------|---------|
| World Model | 60-70% | 预测准确率 |
| Data Pipeline | 10-15% | 数据质量和覆盖度 |
| Action Generator | 10-15% | rollout效率和选择质量 |
| Drive System | 5-10% | 行为多样性调节 |
| Learning Module | 5-10% | 学习效率和无遗忘 |

**检查问题**："这个新增模块是否直接帮助World Model预测得更准？"

如果答案不是明确的"是"，不要添加。先让World Model工作，再考虑锦上添花。

---

### 5.4 第四原则：学习是间歇的，不是连续的

Folunar_的错误：每步在线SGD更新。听起来高效——Agent每执行一步就学习一步。实际是灾难：刚学到的知识在下一步就被覆盖，World Model在训练数据中"追逐"最新的样本，遗忘掉之前学到的模式。这导致了灾难性遗忘和不稳定的预测。

**PEDA的正确方式**：

```
数据收集阶段（固定权重运行）
  → 积累N个transition（N=100-1000，根据环境复杂度调整）
  → 批量更新World Model（LoRA微调）
  → 验证更新是否提升了预测准确率
    → 是 → 部署新权重，继续收集
    → 否 → 回退到上一个版本，调整学习参数后重试
  → 固定权重运行（进入下一轮收集）
```

**为什么间歇学习有效**：
- 批量更新利用了更多数据的信息，减少噪声干扰
- 固定权重运行期间，Agent的行为策略是稳定的，便于评估Drive System的效果
- LoRA保护基础模型的知识，只更新适应特定环境的低秩矩阵
- 验证步骤防止了"越训练越差"的恶性循环

**更新频率的调节**：不是固定的每1000步。当预测误差快速上升时（环境变化大），缩短更新周期；当预测误差稳定时，延长更新周期以节省计算。

**检查问题**："这个学习机制是否会导致遗忘？"

如果更新机制没有显式的遗忘保护（如LoRA、经验回放、新旧数据混合），它就是有问题的。

---

### 5.5 常见陷阱检查清单

以下四个陷阱是Folunar_项目实际遇到的问题，也是PEDA最可能重蹈的覆辙。每个陷阱都附有症状、检测方法和解决方案。

#### 陷阱1：模板陷阱

**症状**：系统总是执行固定的命令序列。例如：每次都先`ls`，再`pwd`，再`ps aux`——看起来在"检查环境"，实际上是在执行模板。

**根本原因**：Action Generator没有真正利用World Model的预测能力，而是退化为基于规则的模板匹配。

**检测方法**：计算行为序列的熵。如果行为熵随时间下降而非上升，系统陷入了模板化。

```python
def compute_behavior_entropy(action_history, window=50):
    """计算最近window个动作的熵"""
    from collections import Counter
    import math
    
    recent = action_history[-window:]
    counts = Counter(recent)
    total = len(recent)
    
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return entropy

# 检测：绘制entropy随时间变化的曲线
# 如果曲线呈下降趋势，触发警报
```

**解决方案**：
- 提高Boredom Drive的权重，强制系统惩罚重复行为
- 在候选动作生成阶段引入随机扰动
- 检查Action Generator是否真正调用了World Model的rollout——如果没有，修复调用链

#### 陷阱2：穷举陷阱

**症状**：系统在有限的状态空间内循环探索。例如：在100个目录之间来回切换，但从不深入查看文件内容或尝试修改操作。

**根本原因**：环境对Agent来说是"封闭"的——没有新的信息输入，探索迟早会穷尽。

**检测方法**：跟踪探索覆盖率。如果覆盖率（访问过的不同状态数 / 总可能状态数）不再增长超过1000步，系统陷入了穷举循环。

**解决方案**：
- 定期注入外部信息（通过Docker volume挂载新的文档、代码、数据集）
- 允许Agent在沙箱中安装新软件（扩展可执行命令空间）
- 引入"创造性动作生成"：让LLM基于当前状态提出全新的命令（而非从候选池中选择）

#### 陷阱3：幻觉陷阱

**症状**：行为看起来多样（熵很高），但没有可观察的进步。Agent在做各种随机的事情，但没有形成任何可复用的知识。

**根本原因**：World Model的预测是随机的或幻觉的，Agent把随机性当成创造力。

**检测方法**：比较"行为多样性"和"知识积累"。如果行为熵很高但World Model的预测准确率不提升，系统在用随机性伪装创造力。

**解决方案**：
- 引入量化"成长"的指标：预测准确率趋势、技能掌握度（成功执行某类命令的比例）
- 要求World Model的预测必须在执行前记录，执行后对比——形成闭环验证
- 如果预测准确率持续低于阈值，暂停探索，增加训练数据

#### 陷阱4：膨胀陷阱

**症状**：代码行数、模块数持续增长，但核心能力（World Model预测准确率）没有提升。新模块"看起来很厉害"（如"元认知模块"、"情感模拟模块"），但不解决实际问题。

**根本原因**：开发过程中缺乏严格的模块审查，功能添加以个人兴趣而非系统需求为导向。

**检测方法**：绘制"代码行数/模块数"和"任务完成率"两条曲线。如果前者上升而后者持平，系统在膨胀。

**解决方案**：
- 实施严格的模块审查流程（见5.6节决策流程图）
- 每个新模块必须有明确的假设验证："如果添加这个模块，我预期World Model预测准确率提升X%"
- 定期删除未被使用的模块（如果模块3个月未被调用，删除）

---

### 5.6 决策流程图

#### 何时添加新模块？

```
问题出现（如"Agent不会处理错误情况"）
  │
  ▼
是否可以用现有模块解决？
  │
  ├── 是 → 修改现有模块，不添加新模块
  │         └── 例如：用World Model预测命令的exit code，
  │             在Action Generator中增加对非零exit code的处理逻辑
  │
  └── 否 → 是否有已发表论文证明这个方法有效？
            │
            ├── 是 → 添加模块，但先做一个最小验证
            │         └── 例如：添加"错误恢复模块"前，
            │             先在100个错误场景上验证它确实能提升恢复率
            │
            └── 否 → 不添加
                      └── 回到第一步，重新思考是否可以用现有模块解决
```

这个流程图的核心精神：**默认不添加**。新模块的添加需要两个"是"：现有模块确实解决不了，且有外部证据表明新方法有效。

#### 何时删除模块？

```
模块3个月未被使用（通过代码静态分析确认）
  → 删除

模块使系统更复杂但没有提升核心指标
  → 删除（即使"可能以后有用"）

模块的功能被另一个模块覆盖
  → 合并后删除冗余
```

**删除比添加更难，但更重要**。Folunar_的40+模块中，至少一半可以被删除而不影响系统能力。PEDA从第一天开始就应该建立"删除文化"——每添加一个模块，同时删除或合并一个旧模块。

#### 代码审查检查清单

每次提交代码前，回答以下问题：

- [ ] 这个改动是否直接提升World Model的预测能力，或支持这个目标的实现？
- [ ] 是否引入了新的外部依赖？如果是，是否有更轻量的替代方案？
- [ ] 是否有硬编码的常量？如果是，是否可以由系统动态调整？
- [ ] 是否添加了新的配置项？如果是，是否所有配置项都有默认值且文档化？
- [ ] 是否在代码中留下了TODO或FIXME？如果是，是否创建了对应的跟踪issue？
- [ ] 这个改动是否可以在现有的测试框架下验证？

**只有当所有问题都有满意答案时，代码才能被合并。**

---

这份指引的目的不是限制创新，而是确保创新发生在正确的方向上。PEDA的创新应该集中在World Model的准确性和预测误差的有效利用上——除此之外的"创新"大多是Folunar_式膨胀的伪装。

---

## 6. 开发路线图与结论

本章将前述架构设计和实现方案转化为可执行的项目计划。路线图采用阶段化推进策略，每个阶段有明确的交付物和通过标准——未达标则停止，分析原因，而非盲目推进。

---

### 6.1 里程碑与时间线

| 阶段 | 时间 | 核心目标 | 成功标准 | 未达标的应对 |
|------|------|---------|---------|-------------|
| Phase 0 | 第1-2周 | 文献精读 + 环境搭建 | 能运行Grid World实验；Active Inference核心论文精读完成 | 延长1周，确认理论理解无误 |
| Phase 1 | 第2-6周 | 极简验证：预测误差驱动探索 | 预测误差驱动探索效率 > 随机基线2x | **停止项目**或转向纯LLM-Agent方案 |
| Phase 2a | 第6-10周 | 数据收集：沙箱自由交互 | 收集10000+ (state, action, state') 三元组 | 降低环境复杂度，延长时间 |
| Phase 2b | 第10-14周 | World Model训练 | 预测准确率 > 70%；对未见命令泛化率 > 50% | 增加训练数据量，调整LoRA参数 |
| Phase 3 | 第14-20周 | 整合评估：EFE + 长期运行 | 48小时连续运行，行为"有趣"（人工评分 > 3/5） | 检查Drive System权重，增加外部信息注入 |
| Phase 4 | 第20周+ | 扩展：新环境与新能力 | 迁移到新环境（如Web浏览、代码编辑） | — |

**关键决策点**：Phase 1是唯一一个"未达标则停止"的阶段。如果预测误差在极简环境中都不能驱动有效探索，它在更复杂的Linux沙箱中也不会工作。Phase 2-3未达标则调整方案而非放弃，因为核心假设已通过验证。

**时间线假设**：每周投入15-20小时（兼职开发强度）。如果是全职投入，总时间可压缩到3-4个月。

---

### 6.2 资源需求

#### 6.2.1 计算资源

| 阶段 | GPU需求 | 推荐配置 | 成本 |
|------|---------|---------|------|
| Phase 0-1 | 无 | CPU即可 | $0 |
| Phase 2a | 低 | Google Colab免费T4（16GB显存） | $0 |
| Phase 2b | 中 | Colab Pro（T4/A100）或本地RTX 4090 | $10-50/月 |
| Phase 3 | 中高 | 本地RTX 4090或云A100 | 视运行时长 |
| Phase 4 | 视扩展方向 | 待定 | 待定 |

**Phase 2b训练细节**：Qwen2.5-1.5B + LoRA(rank=8)的微调，在T4上约需2-3小时/epoch，完整训练（3 epochs）约6-9小时。如果数据量增大到50000+三元组，建议使用A100将训练时间压缩到1-2小时。

**Phase 3长期运行**：48小时连续评估需要本地部署（API成本不可持续）。使用4-bit量化的Qwen2.5-1.5B在RTX 4090上推理速度约10-20 tokens/秒，足够PEDA的步级交互。

#### 6.2.2 LLM API预算

| 阶段 | 预估调用量 | 单价（deepseek-chat） | 总成本 |
|------|-----------|---------------------|--------|
| Phase 1 | ~2000次 | ~$0.002/次 | ~$4 |
| Phase 2a | ~5000次（数据收集） | ~$0.002/次 | ~$10 |
| Phase 2b | ~3000次（prompt调优、评估） | ~$0.002/次 | ~$6 |
| Phase 3 | 本地部署，API仅用于对比 | — | ~$0 |
| **合计** | | | **~$20-100** |

实际成本可能因迭代次数和模型选择而浮动。使用国产API（deepseek-chat、qwen-turbo）比OpenAI API便宜5-10倍。

#### 6.2.3 开发时间

- **总计**：约6个月（每周15-20小时）
- **Phase 1最为关键**：投入应该集中。如果Phase 1在2周内完成验证，后续阶段的信心会大幅提高
- **Phase 2最耗时间**：数据收集和World Model训练是体力活，需要耐心

---

### 6.3 风险评估

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| World Model预测不准 | 中 | 高 | 降低环境复杂度（如从完整Linux退到busybox）；增加训练数据量到50000+；尝试更大的基础模型（3B-7B） |
| 预测误差不驱动有效探索 | 中 | 高 | 引入Drive System权重调节（Curiosity/Novelty驱动早期探索）；放宽成功标准（>1.5x而非2x） |
| 系统陷入固定模式 | 高 | 中 | Boredom Drive动态调节 + 外部信息定期注入（Docker volume挂载新数据） + 候选动作随机扰动 |
| 计算资源不足 | 低 | 中 | Phase 1-2完全可在Colab免费版完成；Phase 3可分段评估（如分4次×12小时代替1次×48小时） |
| 理论方向错误（FEP不适用于LLM-Agent） | 低 | 极高 | **Phase 1快速验证**。如果Phase 1失败，项目损失仅2-4周而非6个月 |
| 灾难性遗忘 | 中 | 中 | 严格间歇学习（非每步更新）；LoRA保护基础模型；经验回放混合新旧数据 |
| 行为评估主观性 | 高 | 低 | 量化指标为主（探索效率、预测准确率、行为熵）；人工评估为辅，多人独立评分取平均 |

**最高优先级风险**："预测误差不驱动有效探索"。这是PEDA的理论基础，如果这个假设不成立，整个项目需要重新定位。Phase 1的目的就是在最小成本下验证或否定这个假设。

---

### 6.4 成功标准定义

PEDA的成功需要同时满足量化指标和质性指标。

#### 6.4.1 量化指标

| 指标 | 测量方法 | Phase 3目标值 |
|------|---------|--------------|
| 探索效率 | 单位步数内访问的不同状态数 | 持续提升或稳定在高水平 |
| 预测准确率 | World Model预测状态与实际状态的匹配度 | >70% |
| 行为多样性 | 行为序列的Shannon熵 | 不持续下降（>2.0 bits/action） |
| 成长曲线 | 探索效率随时间的斜率 | 前24小时呈上升趋势 |
| 预测误差衰减 | 平均预测误差随运行时间的下降 | 前24小时单调下降 |
| Drive权重变化 | Curiosity→Competence的权重转移 | 可观察到转移趋势 |

#### 6.4.2 质性指标

- **行为是否"有趣"**：由3名以上独立评估者阅读48小时行为日志片段（每2小时抽取5分钟），用1-5分评分。平均分>3分为通过。
- **是否展现"成长轨迹"**：对比前6小时和后6小时的行为日志，评估者是否能看出Agent从"盲目探索"进化到"有目的的操作"。
- **是否避免Folunar_式陷阱**：系统没有陷入模板化（行为熵不下降）、没有40+模块膨胀（模块数<10）、没有在线每步SGD导致的遗忘。

#### 6.4.3 失败标准（明确什么算失败）

以下任一情况视为项目需要重大调整：
- Phase 1结束后，预测误差驱动探索效率未超过随机基线1.5x
- Phase 2b结束后，World Model预测准确率<50%
- Phase 3结束后，48小时行为日志的人工评分均值<2/5
- 系统在任何阶段展现出Folunar_式的模块膨胀（模块数>15且核心指标未提升）

---

### 6.5 结论

#### PEDA能做什么，不能做什么

**能做到的**：
- 产生行为上有趣的自主探索，不依赖用户输入
- 展现出"成长"的表象——从随机探索到结构化行为的转变
- 在特定环境（Linux沙箱）中形成可复用的操作知识
- 通过预测误差实现对环境的主动信息获取

**不能做到的**：
- 真正的意识、自我、意图。这些是哲学问题，不是工程问题。PEDA的Agent可能"看起来"有意识（类似LLM"看起来"理解语言），但这不等于真正有意识
- 跨环境的通用能力。在Linux沙箱中学到的知识不会自动迁移到Web浏览或物理机器人
- 超越World Model质量的智能上限。如果World Model不能理解某个概念，Agent永远无法学会它

#### 与Folunar_的关系

PEDA不是Folunar_的改进版，是一次根本性重启。

**继承**：
- Docker沙箱执行环境的工程经验
- "闭合感知-执行循环"的核心思想
- 对自主Agent问题的定义和动机

**抛弃**：
- <1M参数的语言模型（能力不足以承担World Model）
- 在线每步SGD（导致灾难性遗忘）
- 14个硬编码目标的轮转调度（不是真正的驱力系统）
- 模板引擎（行为不是模板匹配出来的）
- 40+模块的膨胀架构（聚焦World Model，其他精简）

#### 最终建议

**先花4周做Phase 1**。这是整个项目风险最低、信息密度最高的阶段。如果预测误差确实能驱动有效探索（达到2x随机基线），说明PEDA的理论方向是正确的，后续5个月的投入有合理预期。如果预测误差不能驱动探索，损失只有4周，而不是Folunar_式的长期投入后发现方向错误。

Phase 1的代码应该控制在500行以内（Grid World环境 + 简单World Model + 预测误差驱动逻辑）。如果超过这个规模，说明你在过度工程化——回到第一原则，简化。

PEDA的核心赌注是：**Active Inference的预测误差机制可以在LLM-based Agent中实现有效的自主探索**。这个赌注可能是对的，也可能是错的。Phase 1就是验证这个赌注的最小成本实验。

---

*"预测替代Prompt"不是一句口号，是一个可验证的工程假设。去验证它。*