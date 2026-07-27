## 维度: Active Inference / FEP / Predictive Coding 理论基础

> 调研日期: 2025年7月  
> 覆盖范围: Karl Friston的自由能原理(FEP)、Active Inference认知架构、Predictive Coding理论、AI Agent应用及开源实现  
> 搜索次数: 20+ 次网络搜索，覆盖7大主题方向

---

### 1. Key Papers

#### 1.1 Friston原始论文与FEP奠基

**Friston, K., Kilner, J., & Harrison, L. (2006).** "A free energy principle for the brain." *Journal of Physiology-Paris*, 100(1-3), 70-87.  
- **核心贡献**: 首次明确提出自由能原理(FEP)，将其作为大脑感知和行动的统一解释框架，提出生物系统通过最小化变分自由能来维持非平衡稳态。  
- **与AI Agent的关联**: 为AI Agent设计提供了统一的目标函数——所有感知、行动和学习都服务于单一的自由能最小化目标，这是构建统一认知架构的理论基础。  
- **关键概念**: Variational Free Energy, Surprise, Markov Blanket  
- [^526^]

**Friston, K. (2009).** "The free-energy principle: a rough guide to the brain?" *Trends in Cognitive Sciences*, 13(7), 293-301.  
- **核心贡献**: 以更通俗的方式解释了FEP，成为理解该原理的经典入门文献，广泛传播了"大脑作为推理机器"的视角。  
- **与AI Agent的关联**: 介绍了感知即推理(perception as inference)和行动即推理(action as inference)的统一框架。  
- [^527^]

**Friston, K. (2010).** "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.  
- **核心贡献**: FEP的里程碑式综述，将感知、行动和学习统一在贝叶斯推理的理论框架下，提出FEP可以作为统一大脑理论的候选者。  
- **与AI Agent的关联**: 明确提出Action and Behavior的free-energy formulation，将RL中的价值函数与自由能联系起来；为设计同时感知和行动的AI Agent提供了数学框架。  
- [^512^][^521^][^526^]

**Friston, K., & Kiebel, S. (2009).** "Predictive coding under the free-energy principle." *Philosophical Transactions of the Royal Society B: Biological Sciences*, 364(1521), 1211-1221.  
- **核心贡献**: 将Predictive Coding与FEP结合，提出预测编码是自由能原理在感知中的具体实现形式。  
- **与AI Agent的关联**: 为分层生成模型中的感知处理提供了计算机制，是构建具有分层感知能力的AI Agent的理论基础。  
- [^553^][^591^]

**Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017).** "Active inference: a process theory." *Neural Computation*, 29(1), 1-49.  
- **核心贡献**: 将Active Inference发展为过程理论(process theory)，提供了从感知到行动的完整数学描述，包括Expected Free Energy的完整推导。  
- **与AI Agent的关联**: 这是Active Inference作为认知架构的核心论文之一，详细阐述了策略选择(policy selection)的数学基础，为AI Agent的决策模块设计提供了直接指导。  
- **关键公式**: Expected Free Energy G(π) = H[q(o|π)] + D_KL[q(o|π) || C(o)]  
- [^588^][^592^]

**Friston, K., et al. (2023).** "The free energy principle made simpler but not too simple." *Physics Reports*, 1024, 1-29.  
- **核心贡献**: Friston等人对FEP最新、最完整的系统性梳理和简化阐述。  
- **与AI Agent的关联**: 作为入门和参考的最新权威综述。  
- [^527^][^581^]

#### 1.2 Predictive Coding

**Rao, R. P., & Ballard, D. H. (1999).** "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience*, 2(1), 79-87.  
- **核心贡献**: Predictive Coding的奠基之作，提出大脑皮层实现分层生成模型，通过最小化预测误差来进行感知和学习；解释了经典感受野之外的神经响应特性。  
- **与AI Agent的关联**: 提供了大脑式高效编码的计算原理——只传输预测误差而非原始信号，这种原理可以应用于设计更高效的感知系统和神经网络架构。  
- **关键概念**: 分层预测、预测误差单元(prediction error units)、解释消除(explaining away)  
- [^516^][^524^][^583^]

**Clark, A. (2015).** *Surfing Uncertainty: Prediction, Action, and the Embodied Mind.* Oxford University Press.  
- **核心贡献**: 将Predictive Processing(PP)推广为认知科学的统一理论框架，提出大脑作为"预测机器"的观点，强调感知是主动构建而非被动接收。  
- **与AI Agent的关联**: 强调认知的具身性和行动导向性——Agent不是被动接收世界表征，而是通过主动预测和行动来塑造感知经验，这对设计具身智能(embodied AI)有深远影响。  
- **关键概念**: Hierarchical Predictive Processing, Action-Oriented Predictive Processing, Precision Weighting  
- [^525^][^527^]

**Clark, A. (2013).** "Whatever next? Predictive brains, situated agents, and the future of cognitive science." *Behavioral and Brain Sciences*, 36(3), 181-204.  
- **核心贡献**: 明确提出了预测加工作为认知科学新范式的宣言式论文。  
- **与AI Agent的关联**: 提出Agent的感知和行动是统一的预测过程中的两个面。  
- [^553^]

**Millidge, B., Salvatori, T., Song, Y., Bogacz, R., & Lukasiewicz, T. (2022).** "Predictive coding: Towards a future of deep learning beyond backpropagation?" In *IJCAI-ECAI 2022 Survey Track*, 5538-5545.  
- **核心贡献**: 综述了Predictive Coding作为超越反向传播的深度学习替代方案的潜力，证明了预测编码网络(PCNs)可以近似并实现反向传播。  
- **与AI Agent的关联**: 为设计不依赖反向传播的生物可信(biologically plausible)学习算法提供了路径，这意味着AI Agent可以使用更本地化的学习规则。  
- [^576^][^577^][^579^]

**Whittington, J. C., & Bogacz, R. (2017).** "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity." *Neural Computation*, 29(5), 1229-1262.  
- **核心贡献**: 证明了预测编码网络可以通过局部Hebbian可塑性来近似反向传播算法。  
- **与AI Agent的关联**: 为AI Agent提供了生物学上更可信的学习机制。  
- [^583^]

**Song, Y., Millidge, B., Salvatori, T., Lukasiewicz, T., Xu, Z., & Bogacz, R. (2024).** "Inferring neural activity before plasticity as a foundation for learning beyond backpropagation." *Nature Neuroscience*, 27(2), 348-358.  
- **核心贡献**: 在Nature Neuroscience上发表，提出了超越反向传播的学习基础。  
- **与AI Agent的关联**: 为不依赖反向传播的深度学习提供了生物学基础。  
- [^583^]

#### 1.3 Active Inference综合与教科书

**Parr, T., Pezzulo, G., & Friston, K. J. (2022).** *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.  
- **核心贡献**: Active Inference领域的权威教科书，系统性地阐述了从离散状态空间到连续状态空间的Active Inference理论。  
- **与AI Agent的关联**: 提供了设计Active Inference Agent的完整"食谱"，包括如何构建生成模型、如何进行策略选择等。  
- [^526^][^594^][^596^]

**Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021).** "Active inference: Demystified and compared." *Neural Computation*, 33(3), 674-712.  
- **核心贡献**: 将Active Inference与Reinforcement Learning进行系统对比，证明了Active Inference中探索和利用自然地从认识价值(epistemic value)中产生。  
- **与AI Agent的关联**: 直接对比了AIF与RL的核心差异，为选择AI Agent架构提供了参考；证明了AIF在探索-利用权衡上的自然优势。  
- [^511^][^572^][^574^]

**Da Costa, L., Lanillos, P., Sajid, N., Friston, K., & Khan, S. (2022).** "How active inference could help revolutionise robotics." *Entropy*, 24(3), 361.  
- **核心贡献**: 系统阐述了Active Inference如何应用于机器人学，讨论了在物理机器人上实现Active Inference的挑战和前景。  
- **与AI Agent的关联**: 直接将Active Inference与机器人AI Agent设计联系起来，提出了机器人学中的关键应用场景。  
- [^573^]

**Fountas, Z., Sajid, N., Mediano, P. A. M., & Friston, K. (2020).** "Deep active inference agents using Monte-Carlo methods." In *Advances in Neural Information Processing Systems*, 33, 11662-11675.  
- **核心贡献**: 提出了深度主动推理(Deep Active Inference)Agent，将深度学习与Active Inference结合，使用蒙特卡洛方法进行推理。  
- **与AI Agent的关联**: 展示了如何将Active Inference扩展到高维连续状态空间，是深度主动推理领域的开创性工作。  
- [^570^][^573^][^572^]

#### 1.4 数学基础与综述

**Buckley, C. L., Kim, C. S., McGregor, S., & Seth, A. K. (2017).** "The free energy principle for action and perception: A mathematical review." *Journal of Mathematical Psychology*, 81, 55-79.  
- **核心贡献**: 对FEP的数学基础进行了最全面的综述和推导，从变分贝叶斯到主动推理的完整数学梳理。  
- **与AI Agent的关联**: 是理解FEP数学基础的必读文献，为实现Active Inference Agent提供了数学参考。  
- [^521^][^541^][^545^][^547^]

**Bogacz, R. (2017).** "A tutorial on the free-energy framework for modelling perception and learning." *Journal of Mathematical Psychology*, 76, 198-211.  
- **核心贡献**: 对自由能框架在感知和学习中的建模提供了详细教程。  
- **与AI Agent的关联**: 入门级的数学教程，便于工程师理解FEP的计算机制。  
- [^552^]

**Friston, K., Parr, T., & de Vries, B. (2017).** "The graphical brain: belief propagation and active inference." *Network Neuroscience*, 1(4), 381-414.  
- **核心贡献**: 使用图模型描述了信念传播和Active Inference的神经实现。  
- **与AI Agent的关联**: 为在概率图模型中实现Active Inference提供了方法。  
- [^574^]

---

### 2. Core Concepts

#### 2.1 自由能原理 (Free Energy Principle, FEP)

**定义**: FEP提出所有自适应系统（从细胞到大脑）都通过最小化变分自由能(Variational Free Energy)来维持自身远离热力学平衡。自由能是感官输入"惊奇"(surprise)的上界，最小化自由能等同于使内部生成模型与外部世界更好匹配。[^526^][^597^]

**核心直觉**: 生物系统不能直接被世界所"惊奇"（因为这可能导致死亡），因此它们不断通过感知更新内部模型、通过学习改进模型、通过行动改变世界来最小化惊奇。[^523^]

#### 2.2 变分自由能 (Variational Free Energy)

变分自由能有两个等价表达：

1. **能量-熵形式**:  
   $$F(s, a, r) = E_q[G(\psi, s, a, r)] - H[q(\psi|\mu)]$$
   其中 $E_q[G]$ 是期望Gibbs能量（负对数联合概率的期望），$H[q]$ 是变分密度的熵。[^528^]

2. **惊奇-KL形式**:  
   $$F = -\ln p(s, a, r) + D_{KL}[q(\psi|r) \| p(\psi|s, a, r)]$$
   即自由能 = 惊奇(negative log evidence) + 近似后验与真实后验之间的KL散度。[^528^]

由于KL散度非负，自由能是惊奇的上界：$F \geq -\ln p(s, a, r)$。最小化自由能同时使Agent更接近真实后验并减少惊奇。[^528^]

#### 2.3 预测编码 (Predictive Coding)

Predictive Coding是FEP在感知中的具体实现。在分层网络中：
- **前馈连接**传输预测误差(prediction error)
- **反馈连接**传输预测(predictions)
- 每个层次的目标是使预测与实际输入之间的误差最小化
- 通过精度加权(precision weighting)调节不同层级误差的可信度[^516^][^524^]

**与AI的关联**: Rao & Ballard (1999) 的模型启发了后续大量深度学习架构，包括Lotter et al. (2016)的Deep Predictive Coding Networks和变分自编码器(VAE)。[^516^]

#### 2.4 主动推理 (Active Inference)

Active Inference将行动视为一种推理过程——Agent通过推断自己"应该"采取什么行动来最小化预期自由能(Expected Free Energy, EFE)。

**策略选择**: Agent选择策略 $\pi$ 的概率与预期自由能的负值成正比：
$$p(\pi) \propto \exp(-G(\pi))$$

**预期自由能(EFE)**:
$$G(\pi) = H[q(o|\pi)] + D_{KL}[q(o|\pi) \| C(o)]$$

其中：
- 第一项 $H[q(o|\pi)]$ 是**认识价值(epistemic value)**：表示策略将导致的不确定性减少（信息增益），驱动探索行为
- 第二项 $D_{KL}[q(o|\pi) \| C(o)]$ 是**实用价值(pragmatic value)**：表示预期观测与偏好观测之间的差异，驱动目标导向行为[^539^][^572^]

**Active Inference统一了探索-利用权衡**：探索（减少不确定性）和利用（达成目标）不是对立的，而是EFE最小化的两个互补方面。[^511^]

#### 2.5 Markov Blanket

Markov Blanket是区分系统与环境的统计边界——毛毯上的状态使内部状态与外部状态条件独立。这是FEP中定义"Agent"概念的核心数学工具。[^525^][^515^]

#### 2.6 精度加权 (Precision Weighting)

精度(precision)是预测误差信度的逆方差。在不确定环境中，Agent需要动态调整对不同感知通道或预测层级的信任度。这对应于注意力的调节机制。[^591^]

---

### 3. Mathematical Foundation

#### 3.1 变分自由能的完整推导

根据Friston (2015)和Buckley et al. (2017)，变分自由能的推导如下：

设 $q(\psi|r)$ 是变分近似后验（内部状态 $r$ 对外部状态 $\psi$ 的信念），$p(\psi, s, a, r)$ 是生成模型的联合概率。

$$F(s, a, r) = -\int_\psi q(\psi|r) \ln\left(\frac{p(\psi, s, a, r)}{q(\psi|r)}\right) d\psi$$

这等价于：
$$F = E_q[-\ln p(\psi, s, a, r)] - H[q(\psi|r)]$$

即期望能量减去变分密度的熵。[^528^]

#### 3.2 预期自由能的分解

对于策略 $\pi$（动作序列），预期自由能可以分解为：

$$G(\pi) = \underbrace{-E_{q(o|\pi)}[\ln C(o)]}_{\text{实用/外在价值}} + \underbrace{E_{q(s|\pi)}[H[p(o|s)]]}_{\text{认识/内在价值}}$$

实用价值驱动Agent朝向偏好的观测；认识价值驱动Agent减少对世界状态的不确定性。[^523^][^539^]

#### 3.3 与强化学习的数学关系

在RL中，Agent最大化累积奖励；在Active Inference中，Agent最小化EFE。关键对应关系：
- RL的奖励函数 $\leftrightarrow$ Active Inference中的对数先验偏好 $-\ln C(o)$
- RL的价值函数 $\leftrightarrow$ 负EFE $-G(\pi)$
- RL的探索奖励 $\leftrightarrow$ 认识价值（信息增益）
- RL的状态估计 $\leftrightarrow$ 感知推断（变分推断）[^511^][^514^]

Sajid et al. (2021) 证明了在特定条件下，Active Inference与模型基RL等价。[^572^]

#### 3.4 离散状态空间Active Inference

在离散POMDP设定中，生成模型由以下部分组成：
- **A矩阵**: $P(o|s)$ — 观测似然
- **B矩阵**: $P(s'|s, u)$ — 状态转移
- **C向量**: $\ln C(o)$ — 先验偏好（对数形式）
- **D向量**: $P(s)$ — 初始状态先验

这些构成了Agent对世界结构的全部假设。[^513^][^523^]

---

### 4. AI Agent Applications

#### 4.1 Active Inference Agent的通用架构

基于FEP的AI Agent具有以下核心组件：

1. **生成模型(Generative Model)**: Agent内部对外部世界的概率模型，指定了状态如何生成观测以及行动如何改变状态[^513^]

2. **感知推断(State Inference)**: 给定观测，更新对隐藏状态的后验信念：$Q(s_t) \propto P(o_t|s_t) \cdot \text{prior}(s_t)$[^523^]

3. **策略选择(Policy Selection)**: 计算每个候选策略的EFE，选择最可能最小化EFE的策略[^523^]

4. **学习(Learning)**: 更新生成模型的参数（Dirichlet参数），使模型更好地匹配环境统计规律[^513^]

#### 4.2 深度主动推理 (Deep Active Inference)

Fountas et al. (2020) 提出了深度主动推理Agent，使用深度神经网络参数化生成模型，结合蒙特卡洛采样进行近似推理。这使得Active Inference可以扩展到高维视觉输入和连续控制任务。[^570^][^572^]

#### 4.3 Active Inference在机器人学中的应用

Active Inference在机器人学中的应用正在快速增长：

- **自适应操纵器控制**: Pezzato, Ferrari & Corbato (2020) 提出了基于Active Inference的自适应机器人操纵器控制器，能够在线适应动态不确定性和传感器故障[^585^][^601^]

- **人形机器人身体感知**: Lanillos等人(TU Munich)在物理人形机器人上实现了身体感知和到达(reaching)行为[^601^]

- **多模态感知**: 整合视觉和本体感受的前向运动学模型[^601^]

- **故障检测**: Ferrari团队的beta-residuals方案用于检测和补偿故障传感器[^601^]

- **主动感知**: Bristol Robotics Lab的贝叶斯主动感知触觉探索系统[^601^]

Da Costa et al. (2022) 综述了Active Inference在机器人学中的应用前景，指出其在不确定性处理和自我监督学习方面的独特优势。[^573^]

#### 4.4 Predictive Coding在深度学习中的应用

Millidge et al. (2022) 综述了Predictive Coding作为反向传播替代方案的潜力：

- PCNs可以使用纯局部学习规则近似反向传播[^576^][^577^]
- 可以处理任意图拓扑结构[^576^]
- 在持续学习(continual learning)场景中展现出抗灾难性遗忘的优势[^585^]
- Song et al. (2024) 在Nature Neuroscience上发表的工作为"大脑是否能做反向传播"提供了肯定回答[^583^]

#### 4.5 与LLM和当代AI的结合

最近的研究开始探索FEP/Active Inference与大型语言模型的结合：
- Friston et al. (2020) 的生成模型、语言交流和Active Inference[^540^]
- 一些框架提出使用Active Inference作为更安全AGI的路径[^546^]
- "From Pixels to Planning: Scale-Free Active Inference" (Friston et al., 2025) 探索从像素到规划的统一框架[^592^]

#### 4.6 Active Inference vs Reinforcement Learning对比

| 方面 | 强化学习(RL) | Active Inference (AIF) |
|------|------------|----------------------|
| 核心目标 | 最大化期望累积奖励 | 最小化变分自由能（惊奇的上界） |
| 学习机制 | 时序差分学习或策略梯度 | 变分贝叶斯推断 |
| 动作选择 | Argmax over Q/V函数 | 策略推断 via EFE最小化 |
| 探索机制 | 启发式（ε-greedy, UCB等） | 自然从认识价值中涌现 |
| 模型需求 | 可选（Model-free可不用） | 需要内部生成模型 |
| ANN集成 | 广泛支持(DQN, Policy Gradient) | 深度生成模型+变分推断 |

[^514^][^511^][^572^]

---

### 5. Open Source Implementations

#### 5.1 pymdp (Python)

- **项目链接**: https://github.com/infer-actively/pymdp  
- **论文**: Heins et al. (2022). "pymdp: A Python library for active inference in discrete state spaces." *Journal of Open Source Software*, 7(73), 4098.[^523^][^532^][^540^]
- **状态**: 活跃维护，支持JAX后端（v1.0.0+），支持CPU/GPU/TPU加速
- **特点**:
  - 第一个开源的离散POMDP Active Inference Python库
  - 核心类Agent提供完整API：infer_states(), infer_policies(), update_A/B/D()
  - 高度模块化和可定制
  - 底层数学操作是SPM MATLAB函数的NumPy端口
  - 遵循OpenAI Gym API标准
  - 文档: https://pymdp-rtd.readthedocs.io/[^537^]

**基本用法示例**:
```python
from pymdp.agent import Agent
# 定义POMDP生成模型 (A, B, C, D)
my_agent = Agent(A=myA, B=myB, C=myC, D=myD)
# 在每个时间步：
qs = my_agent.infer_states(observation)  # 感知推断
my_agent.infer_policies()                # 策略推断
action = my_agent.sample_action()        # 采样动作
my_agent.update_A(observation)           # 学习(可选)
```
[^513^]

#### 5.2 SPM (Statistical Parametric Mapping) - MATLAB

- **链接**: https://www.fil.ion.ucl.ac.uk/spm/ (Friston lab, UCL)
- **状态**: 最原始、最权威的Active Inference实现，MATLAB平台
- **特点**: 
  - DEM(Dynamic Expectation Maximization)工具箱
  - 核心函数 `spm_MDP_VB_X.m`
  - 主要用于神经成像数据分析和行为建模
  - 被pymdp验证和benchmark[^523^][^535^]

#### 5.3 相关Python包

- **pymdp-ai**: https://github.com/ellietoulabi/pymdp_ai - 面向AI应用的pymdp扩展[^526^]
- **pymdp-continuous**: https://github.com/whatcoloris/pymdp-continuous - 支持连续状态空间(WIP)[^529^]
- **InferActively组织**: https://github.com/infer-actively - 托管FEP/AIF相关开源软件[^526^]

#### 5.4 Active Inference Institute

- **网站**: https://activeinference.org  
- **GitHub**: https://github.com/ActiveInferenceInstitute  
- **资源**: 教科书学习小组、应用研讨会、在线课程[^597^]

#### 5.5 其他实现

- **Lanillos et al. (2021)** 的机器人学Active Inference综述中列出了多个机器人学实现[^592^]
- **GitHub Topic "active-inference"**: 38+ 公开仓库，涵盖深度主动推理、多Agent学习等[^530^]

---

### 6. Criticisms & Limitations

#### 6.1 不可证伪性批评

FEP被批评为一个声称能解释一切（从细胞到意识、从进化到文化）的框架，因此难以被经验性地证伪。Friston本人曾暗示FEP不能被经验性地证伪（"mathematical immunity"），这引发了科学哲学家们的质疑。[^599^][^528^]

**回应**: FEP的支持者认为，虽然FEP作为"第一原理"是数学上必然的，但其在特定系统中的具体实现（如哪个脑区编码什么变量）是可以被经验检验的。

#### 6.2 热力自由能与信息论自由能的混淆

批评者指出，FEP在热力学自由能（来自统计物理）和信息论自由能（来自变分贝叶斯）之间做了不恰当的类比，混淆了两个不同领域的概念。[^599^][^528^]

**回应**: Friston的框架明确使用信息论自由能，热力学类比仅用于提供直觉。

#### 6.3 目的论问题

FEP经常使用"目标"、"偏好"、"好奇心"等目的论语言来描述机械过程，批评者认为这引入了隐藏的类别错误。[^515^][^599^]

#### 6.4 与RL相比的实用性限制

- Active Inference需要显式的内部生成模型，在高维复杂环境中构建这样的模型非常困难[^515^]
- 变分推断的计算成本可能高于标准RL方法[^515^]
- 虽然探索-利用权衡是"自然"的，但计算EFE可能很昂贵[^511^]
- 离散状态空间的POMDP实现难以扩展到大规模问题

#### 6.5 Markov Blanket的适用性

Markov Blanket作为定义Agent边界的工具，在动态开放系统中（边界模糊或渗透的系统）的普适性受到质疑。[^515^]

#### 6.6 过度声称

FEP被戏称为"大脑的薛定谔方程"或"生命科学的先验原则"，一些神经科学家批评其解释范围过于宏大，可能超出了其应有范围。[^523^]

#### 6.7 积极的建设性视角

也有研究者（如Jake's Vision博客）认为，虽然FEP的宏大声明存在问题，但其提供的数学工具（如变分自由能、KL散度）对于形式化描述感知和行动是有用的补充。[^528^]

---

### 7. 对AI Agent设计的综合评估

#### 7.1 优势

1. **统一的感知-行动-学习框架**: 不需要为感知、决策、学习分别设计独立模块[^523^]
2. **内建的探索-利用平衡**: 不需要额外设计探索策略（如ε-greedy、UCB）[^511^]
3. **目标与偏好的自然编码**: 通过先验偏好C(o)编码目标，不需要显式设计奖励函数[^523^]
4. **不确定性量化**: 通过精度加权和变分推断自然处理不确定性[^591^]
5. **生物可信性**: 提供了更符合生物智能的计算机制[^576^]
6. **自我监督学习**: 学习主要通过最小化预测误差实现，不需要外部标签[^516^]

#### 7.2 挑战

1. **可扩展性**: 离散POMDP方法难以扩展到大规模问题
2. **模型构建**: 需要手动设计生成模型（A, B, C, D矩阵），在复杂环境中不直观
3. **计算成本**: 变分推断的计算成本可能较高
4. **缺乏大规模成功案例**: 相比RL（如AlphaGo、ChatGPT），Active Inference还没有同等规模的成功应用
5. **工程复杂度**: 现有的工程工具和生态远不如RL成熟（PyTorch/TensorFlow + RLlib等）

#### 7.3 前沿方向

1. **深度主动推理**: 结合深度生成模型和蒙特卡洛推理[^570^]
2. **预测编码网络**: 作为反向传播的生物可信替代方案[^576^]
3. **机器人学应用**: 在自适应控制和身体感知中的应用[^601^][^573^]
4. **多Agent系统**: Active Inference在多Agent协作中的应用
5. **与LLM结合**: 使用Active Inference框架增强大语言模型的规划和决策能力[^546^]

---

### 8. Sources

[^511^] https://arxiv.org/pdf/1909.10863v2 (Active inference: demystified and compared)
[^512^] https://ar5iv.labs.arxiv.org/html/2407.04117 (Predictive Coding Networks and Inference Learning: Tutorial and Survey)
[^513^] https://ar5iv.labs.arxiv.org/html/2201.03904 (pymdp: A Python library for active inference in discrete state spaces)
[^514^] https://arxiv.org/pdf/2509.23896 (RL vs Active Inference comparison)
[^515^] https://arxiv.org/pdf/2509.10875 (Active Inference criticism - anthropocentric biases)
[^516^] https://arxiv.org/pdf/2112.10048 (Predictive Coding Theories of Cortical Function)
[^521^] https://arxiv.org/html/2510.17916v1 (A Dissipative System That Maintains Non-Equilibrium Steady-State)
[^522^] https://arxiv.org/html/2404.12013v1 (Sequential Compositional Generalization)
[^523^] https://ar5iv.labs.arxiv.org/html/2201.03904 (pymdp paper - detailed)
[^524^] https://communities.springernature.com/posts/a-long-and-winding-road-towards-and-away-from-predictive-coding (Rao & Ballard 1999 commentary)
[^525^] https://www.thebsps.org/reviewofbooks/andy-clark-surfing-uncertainty-prediction-action-and-the-embodied-brain/ (Andy Clark Surfing Uncertainty review)
[^526^] https://github.com/ellietoulabi/pymdp_ai (pymdp_ai GitHub)
[^527^] https://arxiv.org/html/2309.06707v1 (Active inference model - various references)
[^528^] https://jake.vision/blog/dr-free-energy (Blog: How I learned to stop worrying and love the FEP)
[^530^] https://github.com/topics/active-inference (GitHub active-inference topic)
[^532^] https://ui.adsabs.harvard.edu/abs/2022JOSS....7.4098H/abstract (pymdp JOSS paper)
[^539^] https://www.frontiersin.org/articles/10.3389/fpsyg.2020.539726/full (Losing Ourselves: Active Inference)
[^540^] https://arxiv.org/html/2605.04065v2 (Free Energy-Driven RL with Adaptive Advantage Shaping)
[^541^] https://arxiv.org/html/2603.09729v1 (Efficient and robust control with spikes)
[^545^] https://arxiv.org/html/2508.05619v1 (The Missing Reward: Active Inference in the Era of Experience)
[^546^] https://arxiv.org/html/2508.05766v1 (A Framework for Inherently Safer AGI through Language-Mediated Active Inference)
[^547^] https://arxiv.org/html/2205.10316v2 (Complex behavior from intrinsic motivation)
[^548^] https://arxiv.org/html/2503.24016v1 (Bayesian Predictive Coding)
[^553^] https://arxiv.org/pdf/1911.10601 (Criticism of FEP - unfalsifiability)
[^570^] https://arxiv.org/html/2604.15679v1 (Deep active inference agents - references)
[^572^] https://arxiv.org/html/2508.06980v1 (Simulating Biological Intelligence with Active Inference)
[^573^] https://arxiv.org/html/2403.12417 (On Predictive planning and counterfactual learning in active inference)
[^576^] https://ar5iv.labs.arxiv.org/html/2308.07870 (Brain-Inspired Computational Intelligence via Predictive Coding)
[^577^] https://arxiv.org/html/2308.07870v2 (A Survey on Brain-inspired Deep Learning via Predictive Coding)
[^581^] https://arxiv.org/html/2510.17916v1 (A Dissipative System - FEP references)
[^585^] https://arxiv.org/pdf/2603.09729 (Predictive coding with spiking neural networks - references)
[^588^] https://arxiv.org/pdf/2004.08128 (WHENCE THE EXPECTED FREE ENERGY?)
[^591^] https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/feeling-the-strain (Predictive Processing and core affect)
[^592^] https://www.mdpi.com/1099-4300/28/1/1 (Decision, Inference, and Information under Active Inference)
[^594^] https://lib.ugent.be/catalog/rug01:003042442 (Active Inference textbook - reference details)
[^596^] https://cordis.europa.eu/project/id/820213/results (EU project on Active Inference)
[^597^] https://link.springer.com/chapter/10.1007/978-981-95-1327-7_14 (Free-Energy Principle and Predictive Coding - Springer)
[^599^] https://www.academia.edu/129711548/The_Limits_of_the_Free_Energy_Principle (Systematic Critique of FEP)
[^601^] https://activeinference.institute/active-inference/robotics/ (Active Inference and Robotics survey)
