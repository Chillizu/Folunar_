## 维度: World Models for Autonomous Agents

---

### 1. Key Papers and Core Contributions

#### 1.1 Ha & Schmidhuber (2018) - "World Models"
- **引用**: Ha, D. & Schmidhuber, J. (2018). "Recurrent World Models Facilitate Policy Evolution." NeurIPS 2018. [^1^]
- **核心贡献**: 
  - 首次提出完整的World Model框架，由三部分组成：(1) Vision (VAE) - 将高维观测压缩为低维潜向量 z；(2) Memory (RNN) - 预测未来潜状态；(3) Controller (线性模型) - 在想象的rollouts中学习策略
  - 开创性地在"梦境"(learned dream)中训练控制器，而非真实环境交互
- **效果数据**:
  - CarRacing环境: V+M+C组合获得 +103.8 平均分，随机基线为 +4.84 [^2^]
  - VizDoom环境: 在梦境中训练的策略在真实环境中达到 49.1 分，随机基线为 22.4 (2.2x提升) [^2^]
- **局限性**: VAE重建质量限制了策略性能；RNN长期预测能力有限；仅在相对简单的游戏环境中测试

#### 1.2 PlaNet (Hafner et al., 2019)
- **引用**: Hafner, D., et al. (2019). "Learning Latent Dynamics for Planning from Pixels." ICML 2019. [^3^]
- **核心贡献**:
  - 提出Recurrent State-Space Model (RSSM) - 结合确定性循环路径和随机潜变量
  - 完全在潜空间中进行规划，通过Cross-Entropy Method (CEM)在线优化动作序列
  - 从像素输入中学习潜动态，无需人工特征工程
- **效果数据**:
  - 在DeepMind Control Suite的6个连续控制任务上达到SOTA样本效率
  - 比当时最优的模型无关方法少用约200倍环境交互
- **局限性**: CEM规划计算开销大；需要在线规划，推理速度受限

#### 1.3 DreamerV1 (Hafner et al., 2020)
- **引用**: Hafner, D., et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination." ICLR 2020. [^4^]
- **核心贡献**:
  - 在潜空间中训练Actor-Critic策略，无需在线规划
  - 通过"latent imagination"从世界模型生成的想象轨迹中学习
  - 引入latent overshooting训练技术提升多步预测一致性
- **效果数据**:
  - 在20个视觉控制任务上达到SOTA数据效率
  - 在Atari基准上超过SimPLe和PlaNet
  - 比模型无关方法(DrQ-v2, CURL)需要更少的训练时间 [^5^]

#### 1.4 DreamerV2 (Hafner et al., 2021)
- **引用**: Hafner, D., et al. (2021). "Mastering Atari with Discrete World Models." ICLR 2021. [^6^]
- **核心贡献**:
  - 用**离散分类潜变量**替代高斯潜变量，更好地捕捉多模态转移
  - 引入**KL Balancing** - 让先验分布更快地向 posterior 靠近，而非反向
  - 增加折扣预测器(discount predictor)优化世界模型损失
- **效果数据**:
  - **首个**在Atari 55游戏基准上达到人类水平性能的基于世界模型的智能体 [^7^]
  - 在单GPU训练条件下超越Rainbow、IQN等顶尖模型无关算法
  - 平均达到人类记录分数的25% [^8^]

#### 1.5 DreamerV3 (Hafner et al., 2023/2025)
- **引用**: Hafner, D., et al. (2023). "Mastering Diverse Domains through World Models." Nature 2025. [^9^]
- **核心贡献**:
  - 引入**symlog变换**: symlog(x) = sign(x) * ln(|x|+1)，对称压缩大值和小值到可比损失尺度
  - **Two-hot编码**: 用离散分桶的交叉熵损失替代MSE，对异常值更鲁棒
  - **Percentile-based回报归一化**: 跟踪V^lambda回报的5th和95th百分位数，Actor始终看到[0,1]范围内的回报
  - **KL balancing with free bits**: 防止世界模型忽略视觉细节或过度拟合噪声
  - 固定超参数跨域通用
- **效果数据**:
  - 在**150+不同任务**上使用**单组固定超参数**达到人类水平性能 [^10^]
  - 覆盖7大基准: Proprio Control Suite (18任务), Visual Control Suite (20任务), Atari, BSuite (468配置), Crafter, DMLab, ProcGen [^11^]
  - **首个**从零开始在Minecraft中收集钻石的RL算法，无需人类数据或课程 [^12^]
  - 在Proprio Control Suite上超越D4PG、DMPO、MPO；在Visual Control Suite上超越DrQ-v2 [^11^]
  - 缩放特性: 更大的模型 = 更好的性能 + 更高的数据效率 [^13^]
  - 所有实验在单张Nvidia V100 GPU上运行

#### 1.6 DreamerV4 (Hafner et al., 2025)
- **引用**: Hafner, D., et al. (2025). "Training Agents Inside of Scalable World Models." [^14^]
- **核心贡献**:
  - 从RSSM转向**高效的Transformer架构**
  - 使用**Shortcut Forcing目标**加速生成，比典型视频模型快25倍以上
  - 使用**扩散模型**思想建模环境动态
  - 完全从**离线数据**训练，无需环境交互
- **效果数据**:
  - 在Minecraft离线钻石挑战中达到0.7%成功率(每1000评估episodes)，基线<0.1% [^15^]
  - 使用比先前离线RL方法**少100倍标注数据**
  - 实现单GPU实时交互推理
  - 中间里程碑: 原木(~100%), 木板(~99%), 工作台(~97%), 木镐(~95%), 石镐(~90%), 铁镐(~29%) [^15^]
- **局限性**: 有限上下文长度(~9.6秒)；库存状态预测不够精确；长期记忆仍需改进

#### 1.7 JEPA Series (LeCun, 2022+)
- **引用**: LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence." OpenReview. [^16^]
- **核心贡献**:
  - 提出**非生成式**世界模型架构 - 预测表示(representation)而非像素
  - 联合嵌入预测架构(Joint Embedding Predictive Architecture)
  - 包含Configurator、Perception、World Model、Cost Module、Actor、Short-term Memory六组件
  - I-JEPA(CVPR 2023): 图像领域实现 [^17^]
  - V-JEPA(TMLR 2024): 视频表示学习 [^18^]
  - V-JEPA 2(2025): 结合100万小时互联网视频 + 机器人数据，实现零样本Franka臂控制 [^19^]
- **与Active Inference的关系**: JEPA和Active Inference在预测性表示学习方面有深层联系，但JEPA更关注表示学习，Active Inference更关注动作选择 [^20^]

---

### 2. RSSM Architecture Deep Dive

#### 2.1 Technical Details
RSSM (Recurrent State-Space Model) 是现代World Model的核心架构组件 [^21^]:

```
确定性路径: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])  # 长期记忆
随机路径: z_t ~ q(z_t | h_t, o_t)   (posterior)      # 从观测推断
         z_t ~ p(z_t | h_t)          (prior)        # 预测未来
```

**关键设计**:
- **确定性状态 h_t**: 4096维GRU状态，负责长期历史信息传播 [^22^]
- **随机状态 z_t**: 32个分类变量 x 32个类别 (DreamerV2/V3)，捕捉不确定性 [^22^]
- **Posterior推断**: 结合当前观测编码和隐藏状态推断后验分布
- **Prior预测**: 仅基于隐藏状态预测下一步潜变量，允许"梦境"中无需真实观测
- **完整状态**: s_t = [h_t; z_t] (拼接确定性和随机状态)

#### 2.2 Training Objective (ELBO)
```
Loss = Reconstruction(image) + Reward(prediction) + Continue(prediction) + KL[p(z_t|h_t) || q(z_t|h_t,o_t)]
```

**DreamerV3改进**:
- 使用symlog变换处理不同尺度的奖励信号
- Two-hot离散回归替代连续MSE
- Unimix categoricals (1%均匀混合)防止训练不稳定性 [^13^]

---

### 3. Dreamer Series Evolution Comparison

| 特性 | DreamerV1 (2020) | DreamerV2 (2021) | DreamerV3 (2023) | DreamerV4 (2025) |
|------|------------------|------------------|------------------|------------------|
| 潜变量类型 | 连续高斯 | 离散分类 | 离散分类 | Transformer-based |
| 训练范式 | 在线RL | 在线RL | 在线RL | **离线数据** |
| 超参数 | 任务特定调优 | 部分共享 | **完全固定** | 固定 |
| 核心创新 | Latent imagination | KL balancing | Symlog + normalization | Shortcut forcing |
| Atari性能 | 良好 | **人类水平** | 超越人类 | - |
| 任务数量 | ~20 | 55 | **150+** | 专注Minecraft |
| Minecraft钻石 | 无 | 无 | **首次达成** | 离线达成 |
| 架构 | RSSM+MLP | RSSM+CNN | RSSM+CNN | **Transformer** |

---

### 4. Real-World Performance

#### 4.1 DayDreamer (Wu et al., 2022) - Physical Robot Learning
- **引用**: Wu, P., et al. (2022). "DayDreamer: World Models for Physical Robot Learning." CoRL 2022. [^23^]
- **核心成果**:
  - **四足机器人(A1)**: 1小时真实世界训练内学会从仰卧状态翻滚、站立并行走；SAC基线仅学会翻滚 [^24^]
  - **UR5机械臂**: 8小时学会视觉拾取放置，从像素和稀疏奖励学习，接近人类远程操作水平
  - **XArm机械臂**: 10小时内达到平均3.1个物体/分钟的拾取率，与人类性能相当 [^24^]
  - **Sphero导航**: 2小时内学会纯视觉导航到目标位置
  - 所有实验使用**相同超参数**，无需模拟器
- **适应性**: 训练后对机器人施加推力，10分钟内适应并学会承受推力或快速恢复 [^24^]
- **局限性**: 长时间硬件训练导致磨损；需要更长时间训练的探索

#### 4.2 Autonomous Driving World Models
- GAIA-1 (Wayve, 2023): 从多模态输入创建逼真驾驶场景，允许详细控制车辆行为 [^25^]
- UniAD (CVPR 2023 Best Paper): 多任务级联架构，将感知、预测和规划统一 [^26^]
- DriveWorld/PreWorld: 预测未来占用和流场 [^25^]
- 关键挑战: 长程一致性和幻觉困境 - 自回归部署中短clip监督与长序列推理的失配 [^27^]

#### 4.3 Industrial Applications
- RSSM已成功应用于多温区连续结晶器的虚拟传感和无传感器闭环控制 [^28^]
- 光伏发电预测(沙尘暴天气) [^29^]
- 移动流量推断和网络优化 [^30^]

---

### 5. Long-Term Prediction Challenges

#### 5.1 Problem Description
World Models面临的核心挑战是**复合误差累积(compounding error)** [^31^]:
- 小的一步预测不准确会在多步展开中累积，产生不可靠的长程轨迹
- DreamerV3在**超过15步**后性能显著下降 [^32^]
- RoboDreamer在长程任务上仅达到15%成功率 [^32^]
- 自动回归生成不可避免地累积感知错误 [^33^]

**具体表现**:
- 视觉预测: 逐渐模糊、物体消失、物理/几何约束违反 [^27^]
- Latent dynamics: 预测基线在k=60步时递归误差是教师强制误差的2.4倍 [^34^]
- 多智能体系统: 随着horizon延伸，轨迹扭曲和畸变加剧 [^35^]

#### 5.2 Solutions and Mitigations

| 方法 | 机制 | 效果 |
|------|------|------|
| **Latent Space预测** | 在潜空间而非像素空间进行预测 | 减少感知误差注入，误差增长从0.23降至0.11(60步) [^34^] |
| **Keyframe-Initialized Rollouts (KIR)** | 从任务关键状态开始想象 | 缩短有效预测深度，减少幻觉复合 [^36^] |
| **离散潜变量** | DreamerV2/V3使用分类变量 | 减少复合误差，更稳定的多模态建模 [^37^] |
| **扩散模型世界模型** | DIAMOND, DreamerV4使用扩散 | 视觉保真度更高的世界模拟 [^38^] |
| **Transformer架构** | TransDreamer, STORM, TWM | 更好捕捉长程依赖 [^39^] |
| **Shortcut Forcing** | DreamerV4 | 比典型视频模型快25倍生成速度 [^14^] |
| **Uncertainty-Gated Re-grounding** | 当ensemble分歧超过阈值时注入真实观测 | 将纯想象转换为闭环预测 [^32^] |

---

### 6. Integration with Active Inference

#### 6.1 Theoretical Connections
Active Inference (主动推断) 是由Friston提出的Bayesian框架，与World Models有深层理论联系 [^40^]:

**核心对应关系**:
| Active Inference概念 | World Models对应 |
|---------------------|-----------------|
| 生成模型 (Generative Model) | World Model/RSSM |
| 变分自由能 (VFE) | ELBO训练目标 |
| 预期自由能 (EFE) | Actor-Critic回报估计 |
| 感知推断 (Perception) | Posterior编码器 q(z_t \| h_t, o_t) |
| 动作选择 (Action) | Actor网络策略优化 |
| 偏好分布 (Preferences) | 奖励预测/价值函数 |

#### 6.2 Practical Implementations

**Probabilistic Dreaming for World Models** (2026):
- 将粒子滤波引入Dreamer的latent imagination过程 [^41^]
- 使用**自由能原理**进行轨迹剪枝:
  ```
  F_t^k = V_phi(h_t^k, z_t^k) + beta * sigma_ens^2
  ```
  其中 V_phi 是critic预测奖励，sigma_ens^2 是ensemble方差(认知不确定性)
- 实验结果: 在MPE SimpleTag上相比标准Dreamer提升4.5%得分，episode回报方差降低28% [^41^]

**Active Inference vs RL的区别**:
- RL: 最大化累积外部奖励，需要数百万环境交互
- Active Inference: 用内部量(预期自由能)替代外部奖励，减少人工奖励设计需求 [^42^]
- Active Inference智能体在grid-world上比DQN少30-50%样本达到竞争力性能 [^42^]
- 在连续控制基准(MuJoCo)上达到可比渐近性能，对奖励错误设定更鲁棒 [^42^]

#### 6.3 Integration Possibilities for Autonomous Agents
1. **探索-利用平衡**: Active Inference的EFE自动平衡epistemic value(信息增益)和pragmatic value(偏好满足)
2. **内在动机**: 无需外部奖励，Agent从减少不确定性中获得驱动力
3. **层级规划**: 结合H-JEPA的多层时间尺度和Active Inference的深层时间模型
4. **认知不确定性估计**: 用自由能原理指导world model何时需要重新接地(re-grounding)

---

### 7. World Models in Text/Command-Line Environments

#### 7.1 Text World Models (TWMs)
- **综述论文**: Li, Y., et al. (2026). "Bridging the Agent-World Gap: Text World Models for LLM-based Agents." [^43^]
- **核心定义**: TWM是文本状态上的转移函数 M: S x A -> TS，给定状态和候选动作，预测后续文本状态

#### 7.2 Two Paradigms

**LLM-as-World-Model**:
- 将LLM的前向传递作为转移函数
- **监督微调(SFT)**: 在轨迹数据上预测完整下一状态或状态增量(delta)
- **强化学习训练**: 优化任务相关奖励而非token-level似然
- **Prompt-based**: 直接提示冻结LLM预测下一状态(e.g., WebDreamer, LLM-MCTS) [^44^]

**Code-as-World-Model**:
- 将转移编码为可执行代码(Python, TypeScript)
- 提供强可重复性和约束执行
- 例如: TextWorld框架中的结构化动态 [^45^]

#### 7.3 Applications
- **网页导航**: 预测网页状态变化
- **代码编辑**: 预测代码修改结果
- **API交互**: 预测API响应
- **对话系统**: 预测用户回复
- **文本游戏**: TextWorld中的多步推理 [^45^]

#### 7.4 Challenges Specific to Text Environments
- 复合错误在文本空间中更严重(语义漂移)
- 长程依赖链要求准确的状态跟踪
- 自然语言的开放性和歧义性
- 缺乏像素空间的几何约束

---

### 8. Summary: Relevance to Autonomous Agent Design

#### 8.1 Why World Models Matter for Agents
1. **样本效率**: 通过"想象"学习，大幅减少真实环境交互(DayDreamer: 1小时学会走路)
2. **安全性**: 在模拟中学习危险或昂贵的任务(DreamerV4纯离线学习)
3. **适应性**: 相同算法适用于不同任务(DreamerV3: 150+任务固定超参数)
4. **规划能力**: 内部模拟支持前瞻决策(Minecraft钻石: 20+分钟规划)
5. **迁移学习**: 世界知识从大规模视频预训练转移(V-JEPA 2)

#### 8.2 Key Limitations for Agent Deployment
1. **复合误差**: 长程预测不可靠(>15步显著退化)
2. **模型利用**: 策略可能利用world model的不准确之处("幻觉利用")
3. **上下文限制**: Transformer架构的有限上下文长度(~9.6秒)
4. **Sim-to-Real差距**: 真实世界复杂性超出模拟范围
5. **计算成本**: 高质量世界模型需要大量计算资源
6. **表示学习**: 重建像素可能浪费容量在任务无关细节上

#### 8.3 Open Problems
1. **长程一致性**: 如何保持100+步的准确预测？
2. **层级世界模型**: 多时间尺度的抽象表示
3. **物理可信性**: 如何嵌入物理约束避免违反守恒定律？
4. **非平稳环境**: 世界变化时如何更新world model？
5. **与LLM的结合**: 如何将世界模型的结构化预测与LLM的常识推理结合？
6. **Active Inference的深度整合**: 如何利用自由能原理统一感知、规划和动作选择？

---

### Sources

[^1^]: https://github.com/cybertronai/schmidhuber-problems - "Recurrent World Models Facilitate Policy Evolution" reproduction
[^2^]: https://github.com/cybertronai/schmidhuber-problems/blob/main/README.md - CarRacing +103.8 mean vs random +4.84; VizDoom dream 49.1 vs random 22.4
[^3^]: https://arxiv.org/html/2606.18208 - Looped World Models, references PlaNet as "first demonstrated agents can learn latent dynamics entirely from pixels"
[^4^]: https://arxiv.org/html/2603.04715v1 - Probabilistic Dreaming paper citing Dreamer "state-of-the-art performance across diverse domains"
[^5^]: https://www.preprints.org/manuscript/202604.0928 - Survey of Embodied World Models
[^6^]: https://gitcode.csdn.net/69cf830054b52172bc66b390.html - DreamerV2: Mastering Atari with Discrete World Models
[^7^]: https://www.marktechpost.com/2021/02/23/google-ai-deepmind-and-the-university-of-toronto-introduce-dreamerv2/ - "first RL agent that outperforms humans on Atari benchmark"
[^8^]: https://mulab.ai/project/dreamerv2/ - DreamerV2 evaluation details
[^9^]: https://arxiv.org/html/2301.04104v2 - Mastering Diverse Domains through World Models (DreamerV3)
[^10^]: https://arxiv.org/html/2606.18208v1 - "DreamerV3 masters over 150 different tasks with a single set of hyperparameters"
[^11^]: https://ar5iv.labs.arxiv.org/html/2301.04104 - DreamerV3 results: "outperforms all previous algorithms on 4 of 7 domains"
[^12^]: https://www.preprints.org/manuscript/202604.0928 - "first agent to collect diamonds in Minecraft from scratch without human demonstrations or curricula"
[^13^]: https://browndeeplearning.com/slides/lecture_26.pdf - CSCI 1470: The Dreamer Series, symlog and normalization tricks
[^14^]: https://arxiv.org/abs/2509.24527 - Training Agents Inside of Scalable World Models (DreamerV4)
[^15^]: https://www.emergentmind.com/topics/dreamer-4 - Dreamer 4 benchmarks: Diamond 0.7% vs baseline <0.1%
[^16^]: https://www.turingpost.com/p/jepa - What Is JEPA? Joint Embedding Predictive Architecture
[^17^]: https://arxiv.org/pdf/2301.08243 - I-JEPA: Self-Supervised Learning from Images
[^18^]: https://arxiv.org/pdf/2404.08471 - V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video
[^19^]: https://arxiv.org/pdf/2506.09985 - V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
[^20^]: https://themesis.com/2026/01/21/world-models-jepa-and-vl-jepa/ - JEPA Contrast-and-Compare with Active Inference
[^21^]: https://www.emergentmind.com/topics/stochastic-recurrent-state-space-model-rssm - RSSM mathematical formulation
[^22^]: https://www.arxiv.org/pdf/2512.04279v2 - RSSM with 4096-dim deterministic state and 32x32 categorical variables
[^23^]: https://proceedings.mlr.press/v205/wu23c/wu23c.pdf - DayDreamer: World Models for Physical Robot Learning
[^24^]: http://autolab.berkeley.edu/assets/publications/media/2022-12-DayDreamer-CoRL.pdf - DayDreamer paper: A1 quadruped 1 hour; XArm 3.1 objects/minute
[^25^]: https://arxiv.org/html/2512.19133v1 - WorldRFT: Latent World Model Planning for Autonomous Driving
[^26^]: https://deepwiki.com/TianxingChen/Embodied-AI-Guide/2.10-embodied-ai-applications - UniAD (CVPR 2023 Best Paper)
[^27^]: https://arxiv.org/html/2603.09086v1 - Latent World Models for Automated Driving: Long-Horizon Consistency and Hallucination Dilemma
[^28^]: https://www.mdpi.com/1424-8220/26/5/1698 - RSSM-Based Virtual Sensing for Multi-Temperature-Zone Continuous Crystallizer
[^29^]: https://www.mdpi.com/1996-1073/19/3/809 - Prediction of Photovoltaic Power Output Using RSSM
[^30^]: https://arxiv.org/html/2604.08199v1 - Beyond Static Forecasting: World Models for Mobile Traffic Extrapolation
[^31^]: https://arxiv.org/html/2605.00412v2 - Physically Native World Models: Long-horizon rollout vulnerable to compounding errors
[^32^]: https://arxiv.org/html/2604.22446v1 - Organising Heterogeneous Agents: "DreamerV3 degrades beyond 15 steps"
[^33^]: https://arxiv.org/html/2605.23993v1 - Nano World Models: autoregressive generation accumulates perceptual errors
[^34^]: https://arxiv.org/html/2606.23444v1 - SkyJEPA: Learning Long-Horizon World Models: compounding ratio analysis
[^35^]: https://arxiv.org/html/2505.20922v2 - Revisiting Multi-Agent World Modeling from Diffusion-Inspired Perspective
[^36^]: https://arxiv.org/html/2602.13977v2 - WoVR: Keyframe-Initialized Rollouts for hallucination mitigation
[^37^]: https://arxiv.org/html/2405.12399v2 - Visual Details Matter in Atari: discrete latents reduce compounding error
[^38^]: https://arxiv.org/html/2512.24497v3 - JEPA-WMs vs reconstruction-based world models
[^39^]: https://arxiv.org/html/2606.18208v1 - Looped World Models: TransDreamer, STORM, IRIS
[^40^]: https://arxiv.org/pdf/2411.14991 - Free Energy Projective Simulation: Active inference with interpretability
[^41^]: https://arxiv.org/html/2603.04715v1 - Probabilistic Dreaming for World Models: particle filter + free energy pruning
[^42^]: https://www.remio.ai/post/what-is-active-inference-ai-models-for-decision-making - Active Inference compared with RL
[^43^]: https://arxiv.org/abs/2606.09032 - Bridging the Agent-World Gap: Text World Models for LLM-based Agents
[^44^]: https://arxiv.org/pdf/2509.25052 - Language-based World Model: LLM functions as world model
[^45^]: https://arxiv.org/pdf/2604.13824 - TextWorld: procedurally generated text-based interactive fiction framework
[^46^]: https://danieleder.substack.com/p/surviving-in-text-worlds - Surviving in Text Worlds: LLM Agents in Interactive Fiction
[^47^]: https://arxiv.org/html/2606.09032v1 - TWM formal framework and taxonomy
[^48^]: https://arxiv.org/html/2307.00504 - On Efficient Computation in Active Inference
[^49^]: https://arxiv.org/html/2207.06415 - The Free Energy Principle for Perception and Action
[^50^]: https://arxiv.org/html/2605.17537v1 - Self-supervised Hierarchical Visual Reasoning with World Model
