## 维度: 连续时间认知架构

### 概述

连续时间神经网络和神经形态计算代表了从传统离散深度学习（以Transformer为代表的逐层前向传播）向生物启发、持续运行架构的根本性转向。本维度调研涵盖从理论模型（CTRNN、Neural ODE）到实际架构（Liquid Neural Networks）、再到脉冲神经网络（SNN）和神经形态硬件的完整谱系，以及它们与当前大语言模型（LLM）结合的最新尝试。

---

### 1. CTRNN (Continuous-Time Recurrent Neural Networks)

#### 1.1 基础理论 (Beer 1995)

CTRNN由Randall D. Beer于1995年开创性地提出 [^1^]，是连续时间动力学与神经网络结合的基石。其核心动力学方程为：

$$\tau_i \frac{dy_i}{dt} = -y_i + \sum_j w_{ji}\sigma(y_j + \theta_j) + \sum_k w_{ki}^{in}I_k$$

其中 $y_i$ 是神经元i的激活状态，$\tau_i$ 是时间常数（控制响应速度），$w_{ji}$ 是循环权重，$\sigma$ 是logistic sigmoid激活函数，$I_k$ 是外部输入 [^2^]。

**关键特性：**
- **时间常数$\tau_i$**：每个神经元可以有不同的响应速度，允许网络同时处理多个时间尺度的信息 [^3^]
- **连续动力学**：网络状态随时间连续演化，而非离散时间步更新
- **吸引子动力学**：CTRNN支持固定点、极限环和混沌等丰富动力学行为

#### 1.2 能产生的自发行为

Beer的CTRNN被用于进化自主智能体（autonomous agents），产生了真正的自发行为：

- **自维持运动模式**：通过进化策略（如CMA-ES）训练的CTRNN可以产生稳定的行走、游泳等行为模式，无需外部指令 [^2^]
- **内部状态调制**：在觅食任务中，CTRNN的内部状态（hidden state）可以调制群体行为——当资源水平下降时，智能体会自发聚集，这类似于"紧急性门控"机制 [^3^]
- **模块化的功能连接**：CTRNN中形成功能连接模块，产生与特定行为相关的神经活动模式 [^4^]
- **信号交流**：在进化多智能体系统中，CTRNN可以自发发展出通信协议，用于协调捕食者-猎物行为 [^5^]

**关键洞察**：CTRNN的"自发行为"来源于其连续动力学和吸引子结构——网络在无输入时也会沿着其固有动力学轨迹演化，产生自发的模式切换。这与离散LLM的"无输入则无输出"有本质不同 [^6^]。

#### 1.3 工程实现与训练

CTRNN主要通过进化算法（如CMA-ES、NEAT）训练，而非梯度下降。这限制了其在大型系统中的应用，但也赋予了其探索非梯度优化空间的灵活性 [^2^][^3^]。现代变体如G-CTRNN尝试将其与MEMS硬件结合，实现超低功耗的人类活动识别 [^7^]。

---

### 2. Liquid Neural Networks (LTC Networks)

#### 2.1 技术核心 (Hasani et al. 2020, 2021)

Liquid Time-Constant Networks (LTC) 由MIT的Ramin Hasani团队提出，是CTRNN的现代化扩展 [^8^][^9^]。核心创新是每个神经元的时间常数不是固定的，而是由其内部状态和外部输入动态调制：

$$\tau_i(h,x)\frac{dh_i(t)}{dt} = -h_i(t) + \sum_{j=1}^N W_{ij}g_{ij}(h,x) + b_i$$

其中 $\tau_i(h,x)$ 是随状态和输入变化的有效时间常数 [^8^]。

#### 2.2 闭式解 (CfC - Closed-form Continuous Networks)

2022年，Hasani团队提出了CfC（Closed-form Continuous Networks），用紧界解析近似替代ODE求解器，实现了直接前馈计算 [^10^]：

- **速度提升**：在标准基准上实现了比ODE-based LTC **100-4000倍** 的加速 [^8^]
- **理论保证**：对所有液体时间常数设置都有有界性和稳定性保证 [^8^]
- **可扩展性**：CfC使LTC可以扩展到大规模应用

#### 2.3 关键应用与成果

| 应用领域 | 成果 | 来源 |
|---------|------|------|
| 自动驾驶车辆控制 | 端到端车道保持，强泛化能力 | [^11^][^12^] |
| 无人机飞行导航 | 分布外鲁棒飞行导航 | [^13^] |
| 时间序列预测 | 超过RNN、LSTM、GRU、Neural ODE | [^8^] |
| 电信网络 | mmWave阻塞预测达97.85%-99.60% | [^8^] |
| Loihi-2神经形态芯片 | >91% CIFAR-10准确率，亚毫焦/帧 | [^8^] |
| 多智能体图控制 | 匹配集中式专家策略 | [^14^] |

#### 2.4 优势与局限

**优势：**
- **因果性**：Liquid NNs已被证明是因果和可解释的 [^15^]
- **参数效率**：用更少参数学习新技能 [^12^]
- **连续时间建模**：自然处理不规则采样数据
- **鲁棒性**：在分布外场景下保持性能 [^13^]

**局限：**
- 训练比标准RNN更复杂，需要ODE求解器或特殊近似
- 在大规模语言建模上尚未证明与Transformer竞争的能力
- 社区生态和工具链远不及PyTorch/TensorFlow成熟

---

### 3. Neural ODE (Neural Ordinary Differential Equations)

#### 3.1 核心创新 (Chen et al. 2018)

Neural ODE由Chen等人在NeurIPS 2018（最佳论文奖）中提出 [^16^]。核心思想是将残差网络（ResNet）解释为ODE的Euler离散化：

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

其中 $f$ 是一个由神经网络参数化的向量场 [^16^]。

**关键技术贡献：**
1. **伴随敏感度方法（Adjoint Sensitivity Method）**：通过求解反向ODE计算梯度，实现**O(1)内存开销**（与网络"深度"无关）[^17^]
2. **自适应计算**：ODE求解器根据误差容限自适应调整评估步数
3. **连续归一化流（Continuous Normalizing Flows）**：将变量替换公式从立方代价的log-determinant简化为线性代价的trace运算 [^17^]
4. **Latent ODEs**：将时间序列编码为潜在ODE轨迹，自然处理不规则采样 [^18^]

#### 3.2 与Transformer的结合尝试

近年来出现了多个将Neural ODE与Transformer结合的项目：

**ContinuumLM** [^19^]：Julia实现的连续深度Transformer语言模型
- 使用 `dh/dt = TransformerBlock(h, t)` 替代离散层堆叠
- 支持多种ODE求解器（Tsit5、RK4、Euler等）
- 使用伴随敏感度方法进行高效反向传播
- **局限**：仅小模型规模（研究/教育用途），基本分词（word-level）

**Continuous-Depth Transformers with Learned Control** [^20^]：
- "三明治"架构：Input → [离散层] → [ODE块] → [离散层] → Output
- 实现**情感控制**：正/负面情感控制准确率达98%/88%
- **延迟与标准Transformer持平**（仅-2.4%开销）
- 证明连续深度Transformer在实际应用中的可行性

**FLUID (Flexible Unified Information Dynamics)** [^21^]：
- 用Liquid Attention Network (LAN) 替代标准SDPA
- 将注意力logits重新解释为线性ODE的解
- 在多个任务上实现比CT基线高达47%的提升
- 在分布偏移下增强泛化能力

#### 3.3 与语言模型的结合

虽然Neural ODE在语言建模方面尚未达到Transformer的规模，但已有探索性工作：
- 使用约30M参数的小模型进行概念验证 [^20^]
- 扩展到GPT-2规模（124M参数）是未来工作 [^20^]
- 目前主要挑战：固定Euler步数、单维度控制、规模限制 [^20^]

---

### 4. Spiking Neural Networks (SNN)

#### 4.1 基本原理

SNN是第三代神经网络，以离散脉冲（spike）事件进行通信，更接近生物神经系统的运作方式 [^22^]。

**核心机制：**
- **膜电位累积**：神经元累积输入脉冲，达到阈值时发放脉冲
- **时间编码**：信息不仅编码在发放率中，还编码在脉冲的精确时间中
- **STDP（Spike-Timing-Dependent Plasticity）**：根据前后神经元脉冲的时间差调整连接权重 [^23^]
- **事件驱动**：仅在脉冲发放时消耗能量，空闲时几乎不消耗

**主要神经元模型：**
- LIF (Leaky Integrate-and-Fire)：最常用，简单高效
- Hodgkin-Huxley：更生物真实但计算复杂
- 自适应神经元：具有发放率适应特性 [^24^]

#### 4.2 训练方法

| 方法 | 原理 | 优势 | 局限 |
|------|------|------|------|
| **Surrogate Gradient** | 用平滑近似替代不可微的脉冲函数 | 可利用标准BPTT训练 | 梯度近似误差 [^25^] |
| **ANN-to-SNN转换** | 训练ANN后转换为SNN | 高准确率 | 需要长仿真时间步 [^26^] |
| **速率编码反向传播** | 基于平均动态的简化BPTT | 内存和计算需求降低 | 依赖速率编码假设 [^27^] |
| **EventProp** | 在脉冲时间精确计算梯度 | 精确梯度，稀疏计算 | 实现复杂 [^28^] |
| **e-prop** | 使用局部eligibility traces | 支持在线学习 | 近似梯度 [^29^] |
| **无反向传播** | WTA + 广播对齐 + 神经调制 | 纯局部学习 | 准确率较低（~97% on MNIST）[^30^] |

#### 4.3 与深度学习的差距

- **准确率差距**：虽然小数据集上接近ANN，ImageNet等大规模任务仍有差距 [^31^]
- **Transformer等价物缺失**："神经形态研究领域还没有Transformer的神经形态版本" [^32^]
- **训练成本**：BPTT需要存储所有时间步的激活状态，内存开销大
- **软件生态**：远不及PyTorch/TensorFlow成熟

---

### 5. 神经形态硬件

#### 5.1 Intel Loihi 2

**规格：**
- 每芯片100万神经元，1.2亿突触
- 128个神经形态核心，每核心最多8192个神经元
- 全数字异步设计，Intel 4工艺
- 每芯片约1W功耗（典型配置）
- 支持8位突触权重和32位消息 [^33^]

**可用系统：**

| 系统 | 规模 | 功耗 | 状态 |
|------|------|------|------|
| Oheo Gulch（单芯片） | 100万神经元 | ~1W | 研究社区可用 |
| Kapoho Point | 800万神经元 | ~8W | 研究社区 |
| Hala Point（最大） | **11.5亿神经元** | 2.6kW | 部署于Sandia国家实验室 [^34^] |

**可用性**：仅通过Intel Neuromorphic Research Community (INRC)提供给学术机构和合格研究组织，**不开放商业购买** [^35^]

**软件**：Lava框架（Python API）用于模型开发和部署 [^36^]

#### 5.2 IBM TrueNorth & NorthPole

**TrueNorth (2014)**：
- 100万可编程神经元，2.56亿突触
- 每芯片仅约70mW功耗
- 4096个核心，每个核心256x256交叉开关 [^37^]

**NorthPole (2023)**：
- 不是严格的脉冲芯片，但体现神经形态原则
- **所有权重存储在芯片上**（256MB SRAM分布在256个核心）
- ResNet-50推理比当代GPU能效高**25倍**
- 消除片外DRAM访问，移除推理的主要能耗来源 [^38^]

#### 5.3 其他平台

| 平台 | 规模 | 功耗 | 特点 |
|------|------|------|------|
| **SynSense DYNAP-CNN** | 100万神经元 | <1mW | 超低功耗视觉 |
| **SynSense Speck** | 328K神经元 | ~mW | 视觉SoC+DVS |
| **SpiNNaker 2** | 每芯片15万+神经元 | 500mW | 大规模脑模拟 |
| **BrainChip Akida** | 商业可用 | 极低 | 最易获得的神经形态平台 [^35^] |

#### 5.4 现状与限制

- **软件瓶颈**：大多数AI工作负载为密集矩阵运算构建，而非神经形态硬件的稀疏脉冲计算 [^39^]
- **尚无Transformer等价物**：无法有效运行基于Transformer的LLM [^32^]
- **适用范围**：最适合事件驱动的工作负载（视觉、听觉、异常检测、运动控制）[^40^]
- **个人获取**：对独立研究者而言，Loihi 2不可购买；BrainChip Akida是唯一商业可选方案

---

### 6. 连续+离散混合架构（Continuous + Discrete Hybrid）

#### 6.1 核心问题：连续架构能否产生自发行为？

**答案是肯定的，已有多个实证：**

1. **EMBER架构**（2026年最新）：
   - 22万神经元SNN + STDP + LLM混合架构
   - 在仅7次对话（14条消息）后，SNN就自发触发了LLM行为
   - 系统在8小时空闲期间自主学习的人-话题关联后，**自主主动联系用户**
   - "SNN决定何时行动以及浮现哪些关联，LLM选择行动类型并生成内容" [^41^]
   - 在NVIDIA RTX 5070 Ti上运行，**独立研究者可实现** [^41^]

2. **CTRNN进化智能体**：
   - 连续动力学产生自维持的运动模式和内部状态转换
   - 无需外部输入即可维持活性 [^2^][^6^]

3. **A-CANN（自适应连续吸引子网络）**：
   - 适应机制导致"自发运动"，产生行波行为
   - 类似Lévy flights的高效搜索策略 [^42^]

**自发行为的机制**：
- 噪声驱动的侧向传播（如EMBER中的背景膜噪声）
- 学习到的关联权重在空闲期间持续激活
- 连续动力学系统的固有吸引子结构

#### 6.2 与LLM-based Agent的结合方式

**架构模式1：SNN作为关联基质（EMBER模式）**
```
Text → Embedding → SNN (STDP学习) → 脉冲/冲动检测 → LLM推理 → 输出
```
- SNN提供**何时行动**和**浮现什么关联**的信号
- LLM作为可替换的推理引擎，负责内容生成 [^41^]

**架构模式2：连续记忆架构（CMA）**
- 持久、可变、整合的记忆基质
- 满足6个行为属性：持久性、选择性保留、检索驱动变异、联想路由、时间连续性、整合抽象 [^43^]
- 比RAG更丰富的记忆交互

**架构模式3：流式思维（Streaming Thinking）**
- LLM增量处理输入，持续更新表示
- 支持渐进理解和全局整合
- 推理深度自适应问题复杂度 [^44^]

**架构模式4：连续深度Transformer**
- 用Neural ODE块替代离散Transformer层
- 支持推理时的语义控制
- 延迟与标准Transformer持平 [^20^]

#### 6.3 工程可行性分析

**硬件需求：**

| 方案 | 硬件需求 | 是否个人可行 |
|------|---------|-------------|
| EMBER（SNN+LLM混合） | 消费级GPU（RTX 5070 Ti 16GB） | **是** |
| Liquid Neural Networks | GPU即可（PyTorch/TensorFlow） | 是 |
| Neural ODE Transformer | GPU，需要特殊库（torchdiffeq） | 是 |
| SNN仿真（PyTorch） | 消费级GPU | 是 |
| Loihi 2部署 | 需通过INRC申请 | 否（对独立研究者） |
| BrainChip Akida | 开发套件可购买 | 是（有成本） |

**关键库与工具：**
- `torchdyn`/`torchdiffeq`：Neural ODE PyTorch实现
- `lava`：Intel Loihi编程框架
- `sinabs`/`rockpool`：SNN训练和仿真
- `snnTorch`：消费级硬件上的SNN

**主要挑战：**
1. **训练稳定性**：连续时间模型训练比离散模型更困难
2. **规模限制**：连续架构尚未证明可扩展到十亿参数级别
3. **生态系统**：工具链和社区远不及标准深度学习成熟
4. **硬件访问**：神经形态硬件对独立研究者大多不可及

#### 6.4 一个人是否可能实现？

**答案是肯定的，取决于具体方案：**

**最可行路径（EMBER模式）** [^41^]：
- 用PyTorch实现SNN（已有开源实现）
- 调用LLM API（Anthropic/OpenAI）作为推理引擎
- 消费级GPU即可运行（EMBER在RTX 5070 Ti上运行22万神经元SNN）
- **独立研究者William Savage已完成此工作**，无外部资助

**中等可行路径（Liquid NN / Neural ODE）**：
- 使用Hasani团队的torchdyn或torchdiffeq
- 在标准GPU上训练
- 需要较强的数学背景（ODE理论、数值方法）

**困难路径（神经形态硬件）**：
- Loihi 2不可商业购买
- 需要通过INRC申请（学术机构优先）
- BrainChip Akida是唯一商业选择，但生态有限

---

### 7. 关键结论与建议

#### 7.1 连续时间架构能否产生自发行为？

**可以，但取决于"自发"的定义。**

- 如果"自发"指"无外部输入时也能产生输出"：CTRNN的连续动力学、EMBER的噪声驱动SNN都满足
- 如果"自发"指"类似人类意识的自主决策"：目前尚无证据支持
- 最有前景的路径是EMBER模式：SNN的连续动力学提供"冲动"信号，LLM将其转化为有意义的行动

#### 7.2 与LLM结合的最佳路径

对于独立研究者，推荐以下渐进路径：

1. **阶段1（立即可行）**：使用EMBER架构——PyTorch SNN + LLM API
2. **阶段2（近期）**：引入Liquid Neural Networks处理时间序列输入
3. **阶段3（中长期）**：探索Neural ODE Transformer进行连续深度处理
4. **阶段4（远期）**：如有硬件访问，部署到Loihi或Akida

#### 7.3 技术风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 连续模型训练不稳定 | 中 | 使用CfC近似，固定步长求解器 |
| 规模扩展困难 | 高 | 从LLM作为推理引擎开始，SNN只处理关联 |
| 硬件获取困难 | 高 | 优先消费级GPU方案 |
| 生态系统不成熟 | 中 | 使用PyTorch生态的扩展库 |
| 无现有成功案例 | 低 | EMBER已证明可行性 |

---

### Sources

[^1^]: Beer, R.D. (1995). "On the dynamics of small continuous-time recurrent neural networks." *Adaptive Behavior*, 3, 469-509. https://www.mdpi.com/2076-3417/15/13/7508

[^2^]: Löffler et al. (2025). "Emergence of Internal State-Modulated Swarming in Multi-Agent Patch Foraging System." arXiv. https://arxiv.org/html/2510.18886v1

[^3^]: arXiv (2025). "Self-Regulation through Communication in Evolved Neural Agents." https://arxiv.org/html/2606.02840v1

[^4^]: Yang et al. (2023). "Functional connectivity modules in recurrent neural networks: function, origin and dynamics." https://ar5iv.labs.arxiv.org/html/2310.20601

[^5^]: arXiv (2026). "Self-Regulation through Communication in Evolved Neural Agents." https://arxiv.org/html/2606.02840v1

[^6^]: arXiv (2021). "A Signal-Centric Perspective on the Evolution of Symbolic Communication." https://ar5iv.labs.arxiv.org/html/2103.16882

[^7^]: MDPI (2025). "G-CTRNN: A Trainable Low-Power Continuous-Time Neural Network for Human Activity Recognition." https://www.mdpi.com/2076-3417/15/13/7508

[^8^]: Emergent Mind. "Liquid Time-constant Networks." https://www.emergentmind.com/topics/liquid-time-constant-networks

[^9^]: Hasani et al. (2021). "Liquid Time-Constant Networks." *AAAI*, 35(9), 7657-7666. https://www.liquid.ai/research/liquid-neural-networks-research

[^10^]: Hasani et al. (2022). "Closed-Form Continuous-Time Neural Networks." *Nature Machine Intelligence*, 4(11), 992-1003. https://arxiv.org/pdf/2106.13898v1

[^11^]: Lechner et al. (2020). "Neural Circuit Policies Enabling Auditable Autonomy." *Nature Machine Intelligence*, 2(10), 642-652. https://www.liquid.ai/research/liquid-neural-networks-research

[^12^]: arXiv (2025). "Liquid Neural Networks: Next-Generation AI for Telecom." https://arxiv.org/pdf/2504.02352

[^13^]: Chahine et al. (2023). "Robust Flight Navigation Out of Distribution with Liquid Neural Networks." *Science Robotics*, 8(77). https://arxiv.org/pdf/2504.02352

[^14^]: Marino et al. (2024). "Liquid-Graph Time-Constant Network for Multi-Agent Systems Control." https://arxiv.org/html/2404.13982v1

[^15^]: Vorbach et al. (2021). "Causal Navigation by Continuous-time Neural Networks." *NeurIPS*. https://www.liquid.ai/research/liquid-neural-networks-research

[^16^]: Chen et al. (2018). "Neural Ordinary Differential Equations." *NeurIPS*, 6572-6583. https://hunterheidenreich.com/notes/machine-learning/generative-models/neural-odes/

[^17^]: Chen et al. (2018). Neural ODE PyTorch Implementation. https://github.com/daniallegue/neural-ordinary-differential-equations

[^18^]: Rubanova et al. (2019). "Latent ODEs for Irregularly-Sampled Time Series." https://arxiv.org/pdf/2307.05735v3.pdf

[^19^]: GitHub - ContinuumLM. "A continuous depth language model framework in Julia." https://github.com/zaydabash/ContinuumLM

[^20^]: Jemley (2026). "Continuous-Depth Transformers with Learned Control Dynamics." GitHub. https://github.com/PeterJemley/Continuous-Depth-Transformers-with-Learned-Control-Dynamics

[^21^]: Razzaq & Zhao (2025). "FLUID: Continuous-Time Hyperconnected Sparse Transformer for Sink-Free Learning." arXiv. https://arxiv.org/html/2605.04421v1

[^22^]: GeeksforGeeks (2026). "Spiking Neural Networks in Deep Learning." https://www.geeksforgeeks.org/deep-learning/spiking-neural-networks-in-deep-learning-/

[^23^]: arXiv (2025). "Toward Large-scale Spiking Neural Networks: A Comprehensive Survey." https://arxiv.org/pdf/2409.02111v1

[^24^]: arXiv (2025). "Local Timescale Gates for Timescale-Robust Continual Spiking Neural Networks." https://arxiv.org/html/2510.12843v1

[^25^]: arXiv (2025). "Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient." https://arxiv.org/html/2406.19645v1

[^26^]: arXiv (2024). "Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation." https://arxiv.org/html/2410.11488

[^27^]: Yu et al. (2024). "Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation." https://arxiv.org/html/2410.11488

[^28^]: arXiv (2020). "Event-Based Backpropagation can compute Exact Gradients for Spiking Neural Networks." https://ar5iv.labs.arxiv.org/html/2009.08378

[^29^]: SpiNNaker2 (2025). "Event-based backpropagation on the neuromorphic platform SpiNNaker2." https://arxiv.org/html/2412.15021v3

[^30^]: arXiv (2026). "Scalable Learning in Structured Recurrent Spiking Neural Networks without Backpropagation." https://arxiv.org/html/2605.00402

[^31^]: Hu et al. (2024). "Toward Large-scale Spiking Neural Networks: A Comprehensive Survey." https://arxiv.org/pdf/2409.02111v1

[^32^]: The Register / Intel (2024). 引自 https://alanscottencinas.com/sustainable-cognition/

[^33^]: arXiv (2025). "Neuromorphic Principles for Efficient Large Language Models on Intel Loihi 2." https://arxiv.org/pdf/2503.18002

[^34^]: Intel (2024). Hala Point system specifications. https://arxiv.org/html/2606.15361v1

[^35^]: arXiv (2025). "The Promise of Spiking Neural Networks for Ubiquitous Computing." https://arxiv.org/html/2506.01737v1

[^36^]: Intel Neuromorphic Research Community. https://www.remio.ai/post/neuromorphic-computing-chipset-adoption-intel-hala-point-and-emerging-industry-trends

[^37^]: arXiv (2020). "Deep Medical Image Analysis with Representation Learning and Neuromorphic Computing." https://arxiv.org/pdf/2005.05431

[^38^]: Modha et al. (2023). "IBM NorthPole." *Science* 382. https://iotdigitaltwinplm.com/how-neuromorphic-chips-actually-work-2026/

[^39^]: Next Waves Insight (2026). "Neuromorphic Computing 2026: Intel, IBM & the Enterprise Gap." https://nextwavesinsight.com/neuromorphic-computing-intel-ibm-enterprise-2026/

[^40^]: Abhishek Singh Shekhawat (2026). "Tech-Neuromorphic-Computing." https://www.way2abhi.com/polymath/tech-neuromorphic-computing

[^41^]: Savage (2026). "EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture." arXiv:2604.12167. https://arxiv.org/abs/2604.12167

[^42^]: arXiv (2024). "Dynamics of Adaptive Continuous Attractor Neural Networks." https://arxiv.org/html/2410.06517v1

[^43^]: arXiv (2026). "Continuum Memory Architectures for Long-Horizon LLM Agents." https://arxiv.org/html/2601.09913v1

[^44^]: arXiv (2025). "Streaming Thinking for Large Language Model Reasoning." https://arxiv.org/pdf/2510.17238

[^45^]: IEEE (2024). "CTRNN-Transformer: Adding Continuous Time Neural Models to Transformers." https://ieeexplore.ieee.org/abstract/document/10677304

[^46^]: arXiv (2025). "Optimal Control for Transformer Architectures." https://www.arxiv.org/pdf/2505.13499

[^47^]: arXiv (2024). "Accurate Mapping of RNNs on Neuromorphic Hardware with Adaptive Spiking Neurons." https://arxiv.org/abs/2407.13534

[^48^]: arXiv (2024). "Solving QUBO on the Loihi 2 Neuromorphic Processor." https://arxiv.org/abs/2408.03076

[^49^]: arXiv (2024). "Accelerating Sensor Fusion in Neuromorphic Computing: A Case Study on Loihi-2." https://arxiv.org/abs/2408.16096v1

[^50^]: arXiv (2026). "Neuromorphic Principles for Efficient Large Language Models on Intel Loihi 2." https://arxiv.org/html/2503.18002v2
