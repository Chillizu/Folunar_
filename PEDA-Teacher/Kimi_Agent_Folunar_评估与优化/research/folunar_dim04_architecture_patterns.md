## 维度: AI Agent架构模式与脑启发设计

### Key Findings

- **分层认知架构**（反应层→审议层→元认知层）在理论和工程上均被广泛验证，ACT-R、SOAR等经典架构已存在数十年，现代LLM Agent（如Session-Governor-Executor模型）也 convergently rediscover 了相同模式 [^353^][^357^]
- **MoE动态扩展**在持续学习场景中已有多个成功实现（LLaVA-CMoE、MILE、ExPaMoE、PMoE），但主要在视觉和语言任务上验证，在Agent决策中的动态专家添加仍处于研究阶段 [^361^][^365^][^376^]
- **World Models**（Ha & Schmidhuber 2018）在模拟环境中表现出色（Dreamer系列），但在真实环境中的长程预测存在严重的compounding error问题，预测在10-20步后发散 [^369^][^409^]
- **小模型路由+大模型执行**（cascaded LLM）已被广泛验证可节省30-70%成本，如AutoMix、FrugalGPT等系统，但引入额外的路由延迟 [^360^]
- **知识图谱作为Agent记忆**（AriGraph、MemGraph等）在结构化推理上表现优于纯向量检索，但构建和维护成本较高 [^381^][^388^]
- **在线学习**在Agent中面临根本性挑战：VLA模型的真实世界持续学习实验表明，朴素的顺序微调导致灾难性遗忘（性能从100%降至15%），只有适度的replay buffer才能缓解 [^380^]
- **脑启发架构**（如ZenBrain的7层记忆架构）在消融实验中显示，扁平基线比完整分层系统低17.8% F1，证明分层有价值，但大脑功能区的精确映射到AI模块仍缺乏实证支持 [^389^][^395^]
- **Docker沙箱**提供安全隔离但带来严重性能开销，且限制Agent访问真实环境数据，可能导致information saturation问题 [^394^][^418^]
- **过度工程化是Agent架构的核心风险**：独立多Agent系统在顺序规划任务上比单Agent差39-70%，错误放大17.2倍；应从单Agent开始，仅在真正需要时才增加复杂性 [^413^][^432^]

---

### Hierarchical Architecture

**理论基础与验证**

分层认知架构是AI Agent设计中历史最悠久、验证最充分的模式之一。经典认知架构ACT-R和SOAR已经存在了数十年，它们将智能分为感知、推理、规划和元认知等层次 [^358^]。2025年的研究表明，这些经典架构的原则正在被现代LLM Agent设计"convergently rediscover"——这不是巧合，而是反映了目标导向行为在复杂不确定世界中的功能需求 [^357^]。

现代的代表性实现包括：

1. **AGIArch**：提出四层堆叠架构（感知→推理→规划→元认知），并声称可模拟任何图灵机 [^358^]。不过该论文的实验验证较为有限（GLUE 92%、ARC 78%），其"完整性证明"更多是理论性的。

2. **SIDE Agent Framework**：将语义认知建模为分层过程——语义感知→结构化推理→高级语义认知，由元认知模块协调 [^344^]。

3. **Session-Governor-Executor模型**：直接映射认知架构原理到实现——Session层（感知/表达）、Governor层（认知控制/策略）、Executor层（动作执行），外加独立的记忆、安全和元认知模块 [^357^]。

**分层架构的优势**
- **时间尺度分离**：快速安全关键逻辑在反应层执行，昂贵的规划和推理在更高层进行 [^354^]
- **明确的控制接口**：层与层之间的边界可以被指定、记录和验证，这在医疗和工业机器人等受监管领域很重要 [^354^]
- **故障隔离**：元认知层可以检测异常并停止执行或升级到人类 [^357^]

**分层架构的局限**
- **开发成本高**：必须定义层之间的中间表示并随着任务和环境演化进行维护 [^354^]
- **层不匹配风险**：如果审议层的抽象与实际感知运动现实偏离，规划决策可能变得脆弱 [^354^]
- **集中式假设**：面向单个Agent，扩展到大规模集群需要额外的协调层 [^354^]

**工程实践结论**：分层架构在机器人、工业自动化和任务规划中被广泛使用，是**工程上有效**的模式。但具体分几层、每层做什么，需要根据任务设计，而非套用固定模板。

---

### MoE Dynamic Expansion

**研究现状**

Mixture of Experts（MoE）作为模型级架构（token级路由）与多Agent系统（任务级路由）在概念上类似但操作层面完全不同 [^351^]。MoE动态扩展的核心思想是在新任务到来时动态添加专家，避免灾难性遗忘。

**关键研究与可行性**

1. **动态专家添加的可行性已被验证**：
   - **LLaVA-CMoE**：提出Probe-Guided Knowledge Extension机制，使用探测专家动态确定何时何地添加新专家，在CoIN基准上显著减少遗忘和参数开销 [^365^]
   - **MILE**：使用LoRA实例化为每个新任务创建轻量级专家，每个任务仅需边际参数增加，在数十个LoRA适配器匹配单个完整模型大小前保持高效 [^361^]
   - **ExPaMoE**：可扩展并行MoE，通过光谱感知在线域判别器实时检测域偏移，按需扩展新的域特定专家 [^376^]
   - **Dynamic MoE**：在合成持续学习实验中验证，动态添加专家的方法能在最终任务上保持初始性能，而其他扩展方法（如Net2Wider、Progressive Network）则不能 [^364^]
   - **PMoE**：渐进式MoE框架，在RANS湍流建模中成功实现，新流态可通过添加专家和路由组件来整合，先前训练的组件保持不变 [^370^]

2. **关键挑战**：
   - **路由冲突**：修改共享路由组件常导致灾难性遗忘 [^365^]
   - **可扩展性限制**：动态扩展方法存在可扩展性问题 [^361^]
   - **参数无界增长**：一些方法动态扩展模型架构可能导致无限增长 [^390^]
   - **任务标签不可用**：实践中推理时通常不知道任务标签，需要自动路由机制 [^363^]

**可行性评估**：MoE动态扩展在**持续学习场景**中是可行的，尤其在视觉-语言模型和特定领域（如CFD湍流建模）已有实证成功。但在通用Agent决策中的动态专家添加仍处于研究阶段，主要挑战在于如何在没有明确任务边界的情况下自动决定何时添加专家，以及如何保持路由机制的稳定性。

---

### World Models Effectiveness

**理论起源**

World Models的概念由Ha & Schmidhuber (2018) 提出，核心思想是让Agent学习环境动态模型，在内部"想象"行动结果以进行规划。Agent在world model中模拟可能的行动并"想象"结果，从而获得规划能力 [^369^]。

**性能表现**

1. **模拟环境中的成功**：
   - **Dreamer系列**在视觉控制任务上取得state-of-the-art的样本效率，在20个DeepMind Control Suite基准上超越PlaNet、A3C、D4PG等 [^414^]
   - **DreamerV2**在Atari环境上超越Rainbow DQN，使用不到2亿步数据 [^415^]
   - **DreamerV3**在3D机器人操作、稀疏奖励迷宫和Minecraft中使用单一配置取得一致性能 [^415^]

2. **真实世界机器人学习**（DayDreamer）：
   - 在4种不同机器人上成功学习（四足行走、机械臂抓取放置、导航）[^421^]
   - 四足机器人从 scratch 学会翻滚、站立和行走仅需1小时 [^422^]
   - 机械臂从像素和稀疏奖励中学会抓取放置需要8-10小时 [^422^]
   - 使用相同超参数适用于不同机器人平台 [^421^]

3. **Minecraft中的突破**（Dreamer 4）：
   - 在Minecraft钻石挑战中达到0.7%成功率，超越所有先前的offline agent [^416^]
   - 在中间里程碑上达到>90%成功率 [^416^]
   - 使用100倍 less数据 than VPT [^416^]

**根本局限**

1. **复合误差（Compounding Error）**：多步rollout会发散。一个在单步损失上表现良好的模型在50步规划时可能不可用 [^409^]
2. **预测发散**：想象的rollout在10-20步后发散 [^409^]
3. **部分可观测性**：真实环境需要信念状态，world model需要处理观察不全的情况 [^409^]
4. **目标不匹配**：预测像素与预测决策相关变量之间存在差异，MuZero通过直接建模奖励/价值/策略来规避这个问题 [^409^]
5. **仿真到现实的鸿沟**：world model可能在模拟中表现良好但转移到真实世界时失败 [^409^]
6. **世界模型精度约束**：在MARL中，world model在捕捉环境交互动态方面的精度限制显著影响样本轨迹生成的可靠性 [^374^]

**工程实践结论**：World Models在**样本效率**方面提供了巨大价值（Dreamer使机器人学习效率提高数个数量级），但**长程预测的可靠性**仍是核心瓶颈。对于Agent架构而言，world model更适合作为"想象力引擎"进行短程规划和探索，而非依赖其进行长程精确预测。在需要长程规划的场景中，应结合receding horizon control和定期用真实观察重新校正。

---

### Small Model Decision + Large Model Execution

**研究现状与验证**

小模型路由+大模型执行（Model Cascading/Routing）已被广泛研究并部署到生产系统中：

1. **AutoMix**：使用小模型生成初始答案并自我验证，然后根据验证置信度路由到大模型。将级联建模为POMDP，仅依赖自验证置信度作为输入 [^360^]
2. **FrugalGPT系列方法**：通过quality estimation决定是接受小模型的翻译还是升级到大模型进行后编辑 [^360^]
3. **生产系统实践**：混合架构已成为2025年最优生产架构——MoE模型（如DeepSeek-V3，671B总参数/37B活跃参数）作为每个Agent的推理引擎，多Agent编排层负责任务路由和工具管理 [^351^]

**性能提升**
- 成本节省：与使用单一大型模型相比，可减少**30-70%**成本 [^360^]
- 计算节省：MoE模型相比dense等效模型可实现**3-5倍**计算节省 [^351^]
- 延迟优化：简单查询直接由小模型处理，避免大模型的推理延迟 [^360^]

**关键挑战**
- **路由准确性**：路由决策错误会导致性能下降
- **自验证可靠性**：研究发现self-verification在修复模型输出或估计置信度方面不够可靠 [^360^]
- **额外延迟**：级联引入多步推理延迟
- **质量估计成本**：需要额外的quality estimation模型

**工程实践结论**：小模型决策+大模型执行是**工程上高度有效**的模式，已在生产翻译系统、客服Agent和代码生成工具中广泛部署。核心洞察是：并非所有查询都需要最大模型的全部能力——通过将计算资源与查询复杂度匹配，可实现显著的成本优化。

---

### Knowledge Graph / FactGraph in Agents

**研究现状**

知识图谱作为Agent的结构化记忆越来越受到关注：

1. **AriGraph**：将Agent记忆表示为语义三元组的知识图谱，编码语义事实和情景事件 [^381^]
2. **上下文图（Context Graphs）**：提供三层Agent记忆——长期记忆（知识）、短期记忆（对话）、推理记忆（决策痕迹）。上下文图直接链接决策痕迹到数据实体，确保Agent推理基于世界的实际状态 [^384^]
3. **GraphMemory综述**：整理了知识图谱结构、层次记忆结构、时序图结构、超图结构等多种图基Agent记忆方案 [^388^]
4. **MemGraph**：用于专利分析的图基Agent记忆 [^381^]
5. **KG-Agent**：在知识图谱上进行复杂推理的高效自主Agent框架 [^388^]

**性能优势**
- **结构化记忆组织**：使Agent不仅能回忆孤立数据点，还能回忆它们之间的结构化连接（如因果关系、时间链接）[^381^]
- **精确查询**：Agent可通过直接查询定位特定关系，而非依赖LLM从大量文本中提取相关信息 [^386^]
- **可解释性**：决策图提供透明度和可解释性，支持人类和Agent审计 [^384^]
- **NeurIPS 2025研究表明**：图结构记忆在五个基准测试中实现高达**20.89%**的具身动作成功率提升 [^401^]

**关键挑战**
- **构建和维护成本**：知识图谱需要领域专家设计schema并持续维护 [^386^]
- **与LLM的集成复杂性**：需要Agent框架（如LangChain、LangGraph）提供抽象来实现KG查询作为工具调用 [^386^]
- **动态更新**：如何自动从Agent交互中提取和更新知识图谱仍是开放问题

**工程实践结论**：知识图谱在需要**结构化推理、长期记忆和可解释决策**的Agent中是有效的。但并非所有Agent都需要KG——对于简单任务，纯向量检索（RAG）可能更经济。KG的价值随任务复杂度、领域结构化程度和可解释性需求而增加。

---

### Online Learning Challenges

**灾难性遗忘：根本性问题**

在线学习（每个时间步更新）在神经网络Agent中面临灾难性遗忘的根本挑战：

1. **VLA模型的真实世界持续学习**（2026年首个实证研究）：
   - 朴素顺序微调导致严重灾难性遗忘：第一个任务性能从100%降至15%，第二个从97.5%降至25%，第三个从100%降至13.3% [^380^]
   - 遗忘是结构性的而非均匀的：不同任务有不同的遗忘脆弱性，视觉共享属性导致跨任务混淆 [^380^]
   - 适度的replay buffer（约500轨迹）和正确的replay频率几乎可完全消除遗忘 [^380^]
   - 关键发现：配置良好的顺序学习+适度replay可优于联合多任务训练 [^380^]

2. **在线持续学习的核心挑战**：
   - **可塑性与稳定性权衡**：学习新信息（可塑性）与保留旧知识（稳定性）之间的平衡仍是开放问题 [^390^]
   - **深度网络中的遗忘**：与人类记忆不同，深度神经网络难以保留旧知识，因为共享表示中的权重更新会覆盖先前知识 [^390^]
   - **现有方法的有限有效性**：虽然replay、正则化和动态架构在一定程度上缓解遗忘，但并未提供通用解决方案 [^390^]

3. **边缘设备上的在线学习**：
   - 在线学习消耗更多网络带宽和计算资源以换取更高模型性能和适应能力 [^383^]
   - 概念漂移（concept drift）随时间发生，导致模型输入和输出之间的功能关系变化 [^383^]

**每个时间步更新的可行性**

- **完全不可行**：朴素地在每个时间步进行梯度更新会导致立即的灾难性遗忘
- **有条件可行**：使用经验回放（replay buffer）、正则化（如EWC）或动态架构扩展，可以在一定程度上实现在线学习
- **实际限制**：对于大模型（如VLA），即使在"在线"设置中，更新频率也是 minutes-level 而非 step-level [^380^]

**工程实践结论**：在当前技术下，**每个时间步进行梯度更新是不可行的**。实际部署中的"在线学习"通常指分钟级或小时级的模型更新，且必须配备replay buffer（即使是 modest size）来防止灾难性遗忘。对于Agent架构，更实际的做法是将学习分为"fast adaptation"（如prompt调整、上下文学习）和"slow learning"（如定期微调），而非连续梯度更新。

---

### Brain-Inspired Design Value

**支持证据**

1. **ZenBrain 7层记忆架构**：
   - 提出受神经科学启发的7层记忆架构（工作记忆、短期记忆、情景记忆、语义记忆、程序记忆、核心记忆、跨上下文记忆）[^389^]
   - 消融实验显示：扁平基线比完整系统低**17.8% F1**，证明分层记忆结构的价值 [^389^]
   - 情景记忆（-11.8%）和语义记忆（-10.6%）的移除影响最大 [^389^]

2. **认知架构的趋同**：
   - 认知科学文献中的模型（ACT-R、SOAR、Global Neuronal Workspace）在AI Agent设计中被"convergently rediscovered" [^357^]
   - 这反映了目标导向行为在复杂不确定世界中的功能需求对架构的约束——无论是生物基材还是transformer权重，感知、审议、行动和记忆的功能需求都施加了相同的架构解决方案 [^357^]

3. **MAPS元认知架构**：
   - 在感知环境（Blindsight、人工语法学习）、单Agent RL（MinAtar）和多Agent RL（Melting Pot）中 outperform baseline [^419^]
   - 但在学习2个或更多新环境后仍观察到灾难性遗忘 [^419^]

**批评与局限**

1. **大脑复杂性难以复制**：人脑有数十亿神经元和数万亿突触，组织成复杂的网络，复制这种复杂性在计算和工程上都是巨大挑战 [^434^]

2. **对大脑的理解仍不完整**：学习、记忆和意识等许多大脑功能方面仍 poorly understood，这使得将神经科学洞见转化为实用算法变得复杂 [^434^]

3. **Gary Marcus的批评**：
   - 机器学习有一个巨大偏见：认为一切都是学习的，没有什么是先天的，忽略了人类本能和大脑生物学 [^430^]
   - 我们缺乏构建复杂认知系统的程序 [^430^]
   - 80%的准确率对广告或推荐可以接受，但对医疗诊断或自动驾驶不够 [^430^]

4. **生物合理性 vs 工程效率**：
   - 反向传播的生物不合理性转化为实际限制：与硬件和非可微实现不兼容，导致高能耗需求 [^395^]
   - CapsNet试图解决CNN中池化层的问题，但计算复杂度更高，实现更困难 [^434^]

**工程实践结论**：脑启发设计提供了有价值的**架构直觉**（如分层记忆、感知-认知-行动分离、元认知监控），但精确的大脑功能区映射到AI模块缺乏实证支持。最佳实践是借鉴大脑的组织原则而非试图复制其实现细节。对于Agent架构，**元认知层、分层记忆和感知-行动分离**是受脑启发且工程有效的模式，但不应过度追求生物合理性而牺牲计算效率。

---

### Docker Sandbox Limitations

**安全优势**

Docker沙箱为AI Agent提供了关键的安全隔离：
- **隔离文件系统**：Agent只能访问沙箱目录中的文件 [^418^]
- **无网络访问**：防止Agent向外部服务器发送数据 [^418^]
- **资源限制**：限制CPU和内存使用，防止资源耗尽 [^418^]
- **自动清理**：完成后可删除整个沙箱 [^418^]

**对学习能力的影响**

1. **性能开销**：
   - Docker沙箱的"性能开销可能是 crippling"，即使对于简单项目 [^394^]
   - 沙箱中运行Agent的速度显著慢于直接在主机上运行

2. **信息饱和度（Information Saturation）**：
   - 沙箱隔离限制了Agent访问外部数据源、API和工具的能力
   - Agent无法与真实世界环境交互，只能依赖预装的数据和工具
   - 这限制了Agent的学习能力，因为无法获取新的训练数据

3. **环境约束**：
   - 网络策略限制可能阻碍Agent访问必要的在线资源 [^394^]
   - 签名、认证等高级功能在沙箱中难以实现 [^394^]
   - 某些操作（如提交代码签名）需要复杂的workaround [^394^]

4. **沙箱与真实环境的差距**：
   - 沙箱中的行为可能与真实环境不同，导致sim-to-real gap
   - Agent可能学会利用沙箱特定特性而非学习通用策略

**工程实践结论**：Docker沙箱对于**安全隔离**是有效的，但对于**学习Agent**来说是一个 trade-off。完全隔离的沙箱会限制Agent获取真实世界数据和反馈的能力，可能导致信息饱和。实际部署中常用的平衡方案是：
- 使用网络策略控制而非完全禁用网络
- 提供受控的数据访问通道
- 使用轻量级隔离（如V8 isolates）替代完整容器 [^407^]
- 定期将沙箱中的学习结果同步到持久化存储

---

### Over-Engineering Risks

**实证证据：复杂架构不如简单基线**

1. **Google研究（180个配置的大规模评估）**：
   - 独立多Agent系统比单Agent基线放大错误**17.2倍**，而集中式架构通过验证将错误限制在**4.4倍** [^401^][^413^]
   - 多Agent在需要顺序约束满足的任务上（规划）比单Agent差**39-70%** [^413^]
   - 多Agent系统比单Agent慢**最多4倍**，贵**3倍** [^432^]
   - 但当单Agent性能已超过45%准确率时，增加Agent会产生负回报 [^413^]
   - 最佳架构因任务而异：去中心化在高熵搜索空间任务上受益（网页导航+9.2%），但在顺序约束任务上 universally 退化（规划-39%到-70%）[^413^]

2. **Anthropic的工程建议**：
   - 多Agent系统比单Agent多使用约**10-15倍**token [^431^]
   - 应在 weeks 内部署单Agent，months 才能做好多Agent [^431^]
   - 建议从单Agent开始，仅在真正需要时才增加复杂性 [^431^]
   - 80%的解决方案是标准Python代码，20%使用Agent自动化 [^412^]

3. **成本效益分析**：
   - 单Agent系统开发成本$1-5M，部署$0.10-0.50/案例 [^403^]
   - 多Agent系统开发成本$5-20M，部署$1-5/案例 [^403^]
   - 在高吞吐量应用中，单Agent提供更好的投资回报 [^403^]

4. **去冗余优化**：
   - 研究表明Agent框架可通过去除冗余组件减少近**30%**运营成本，同时保留超过**96%**的性能 [^397^]

**过度工程化的具体表现**

1. **Agent用于简单查询**：浪费计算和开发工作 [^406^]
2. **不必要的多Agent编排**：当工具数量<10-12时，单Agent通常足够 [^352^]
3. **过度复杂的推理链**：增加延迟和成本而不成比例提高准确性 [^400^]
4. **不必要的工具调用**：Agent可能"执行代码次数过多"、"读取过多不必要文件" [^407^]

**反模式与最佳实践**

| 反模式 | 最佳实践 |
|--------|----------|
| 为多Agent而多Agent | 从单Agent开始，仅在真正需要时才扩展 [^431^] |
| 所有任务都使用最强大模型 | 根据查询复杂度匹配合适模型 [^431^] |
| 追求生物合理性而牺牲效率 | 借鉴组织原则而非实现细节 |
| 每个时间步更新模型 | 分级学习：快适应（prompt）+ 慢学习（定期微调）|
| 完全隔离沙箱 | 受控访问 + 轻量级隔离 |

**工程实践结论**：过度工程化是Agent架构设计中的**核心风险**。实证证据明确表明：
- 单Agent在大量场景中是最佳选择
- 多Agent仅在任务真正需要并行化、多领域协调时才优于单Agent
- 复杂性应随需求演化而增加，而非从开始就设计最大化
- 架构应跟随任务，而非反过来 [^432^]

---

### Controversies & Conflicting Claims

1. **分层架构的价值**
   - **支持者**：认知架构原则在AI Agent设计中被"convergently rediscovered"，反映功能需求的约束 [^357^]
   - **批评者**：AGIArch等框架的理论声明（如"可模拟任何图灵机"）缺乏严格实证，有过度承诺之嫌 [^358^]
   - **争议点**：分层是认知的本质特征还是工程便利的副产品？

2. **脑启发设计的实际价值**
   - **支持者**：ZenBrain的消融实验显示扁平基线低17.8% F1，证明分层记忆结构有实际价值 [^389^]
   - **批评者**：Gary Marcus指出机器学习忽视大脑生物学，且80%准确率对关键应用不够 [^430^]
   - **争议点**：大脑功能区的精确映射是否有工程价值，还是仅提供架构直觉？

3. **多Agent vs 单Agent**
   - **支持者**：在复杂金融推理中集中式多Agent比单Agent提高81% [^413^]
   - **批评者**：在顺序规划任务上多Agent比单Agent差39-70%，错误放大17.2倍 [^413^]
   - **争议点**：多Agent的价值高度依赖于任务结构，不存在通用优势

4. **World Model的实用性**
   - **支持者**：Dreamer在真实机器人上实现1小时学会四足行走，样本效率提高数个数量级 [^421^]
   - **批评者**：长程预测在10-20步后发散，存在严重的compounding error [^409^]
   - **争议点**：World Model适合短程规划还是长程推理？

5. **在线学习的可行性**
   - **支持者**：适度replay可几乎完全消除遗忘，顺序学习+replay可优于联合训练 [^380^]
   - **批评者**：朴素在线学习导致立即的灾难性遗忘，现有方法不提供通用解决方案 [^390^]
   - **争议点**：真正的"每个时间步更新"是否可行，还是需要分钟级批处理？

6. **MoE在Agent中的价值**
   - **支持者**：MoE+多Agent是2025年最优生产架构，3-5倍计算节省 [^351^]
   - **批评者**：MoE是模型级优化，与多Agent系统级优化操作层面完全不同，不应混淆 [^351^]
   - **争议点**：MoE的动态专家添加在持续学习中有效，但在Agent决策中是否同样有效？

---

### Sources

[^344^] https://arxiv.org/html/2510.17129v1 - A Bio-Inspired Cognitive Framework for Embodied Agents (SIDE Framework)
[^353^] https://www.emergentmind.com/topics/supervisory-or-meta-cognitive-layer - Supervisory and Meta-Cognitive Layers
[^354^] https://www.marktechpost.com/2025/11/15/comparing-the-top-5-ai-agent-architectures-in-2025/ - Comparing the Top 5 AI Agent Architectures in 2025
[^357^] https://zylos.ai/research/2026-03-12-cognitive-architectures-ai-agents-perception-to-action/ - Cognitive Architectures for AI Agents: From Perception to Action
[^358^] https://openreview.net/pdf?id=aMep22Wzw7 - AGIArch: A Unified Hierarchical Architecture for Artificial General Intelligence
[^360^] https://arxiv.org/html/2603.04445v2 - Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey
[^361^] https://arxiv.org/abs/2605.03555 - MILE: Mixture of Incremental LoRA Experts for Continual Semantic Segmentation
[^363^] https://arxiv.org/html/2403.11549 - Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters
[^364^] https://arxiv.org/html/2511.18987v1 - Dynamic Mixture of Experts Against Severe Distribution Shifts
[^365^] https://arxiv.org/html/2503.21227v2 - LLaVA-CMoE: Towards Continual Mixture of Experts for Large Vision-Language Models
[^369^] https://arxiv.org/pdf/2503.00653 - Discrete Codebook World Models for Continuous Control
[^370^] https://arxiv.org/abs/2601.09305 - Progressive Mixture-of-Experts with autoencoder routing for continual RANS turbulence modelling
[^374^] https://arxiv.org/pdf/2411.19639 - RMIO: A Model-Based MARL Framework with Observation Loss
[^376^] https://arxiv.org/html/2507.00502v1 - ExPaMoE: An Expandable Parallel Mixture of Experts for Continual Test-Time Adaptation
[^380^] https://arxiv.org/html/2605.26820v1 - Can VLA Models Learn from Real-World Data Continually without Forgetting?
[^381^] https://arxiv.org/html/2506.18019v1 - Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities
[^383^] https://arxiv.org/pdf/2302.08571v1.pdf - A Review and Taxonomy of Edge Machine Learning
[^384^] https://neo4j.com/blog/agentic-ai/context-graph-ai-agent-memory/ - Context graphs: Why AI agents need three types of memory
[^389^] https://arxiv.org/html/2604.23878v2 - A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems (ZenBrain)
[^390^] https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.173886426.63028528 - Challenges in Continual Learning for Real-World Applications
[^394^] https://andrewlock.net/running-ai-agents-safely-in-a-microvm-using-docker-sandbox/ - Running AI agents safely in a microVM using docker sandbox
[^395^] https://arxiv.org/pdf/2403.18929 - A Review of Neuroscience-Inspired Machine Learning
[^397^] https://appamass.com/en/blog/optimizing-ai-agent-architecture-complexity-to-task-alignment-9jdweu8cjs1zq03k39s3 - Optimizing AI Agent Architecture Through Complexity-to-Task Alignment
[^401^] https://galileo.ai/blog/ai-agent-architecture - AI Agent Architecture From Patterns to Governance
[^403^] https://hal.science/hal-05491919v1/document - Clinical Appropriateness Decision Framework for AI System Architecture
[^406^] https://r6.ieee.org/scv-cis/wp-content/uploads/sites/6/2025/12/CNM-Jan-March.pdf - STRIDE Framework: Necessity Assessment for Agentic AI
[^409^] https://www.c-sharpcorner.com/article/world-modeling-in-ai-what-it-is-and-how-it-works-end-to-end/ - World Modeling in AI: Limitations and Considerations
[^411^] https://dev.to/alanwest/how-to-stop-over-engineering-with-ai-when-a-simple-query-will-do-26b7 - How to Stop Over-Engineering with AI When a Simple Query Will Do
[^412^] https://github.com/xFlashAI/ai-agent-framework - Agno: Lightweight AI Agent Framework
[^413^] https://arxiv.org/html/2512.08296v1 - Towards a Science of Scaling Agent Systems
[^414^] https://www.emergentmind.com/topics/dreamer - Dreamer: World-Model RL Innovations
[^415^] https://www.mdpi.com/2078-2489/16/10/898 - LiDAR Dreamer: Efficient World Model for Autonomous Racing
[^416^] https://www.emergentmind.com/topics/dreamer-4 - Dreamer 4: Scalable World Model Agent
[^418^] https://mbrenndoerfer.com/writing/action-restrictions-and-permissions-ai-agents - Action Restrictions and Permissions: Controlling What Your AI Agent Can Do
[^419^] https://umontreal.scholaris.ca/server/api/core/bitstreams/c93bce66-2afe-4b87-869a-f17d179229f6/content - MAPS: A Metacognitive Architecture for Improved Perceptual and Social Learning
[^421^] http://autolab.berkeley.edu/assets/publications/media/2022-12-DayDreamer-CoRL.pdf - DayDreamer: World Models for Physical Robot Learning
[^426^] https://www.preprints.org/manuscript/202501.1953 - AI Is Not Intelligent
[^430^] https://ide.mit.edu/insights/gary-marcus-probes-ais-limitations/ - Gary Marcus Probes AI's Limitations
[^431^] https://resources.anthropic.com/hubfs/Building%20Effective%20AI%20Agents-%20Architecture%20Patterns%20and%20Implementation%20Frameworks.pdf - Building Effective AI Agents: Architecture Patterns (Anthropic)
[^432^] https://blog.doubleslash.de/en/software-technologien/kuenstliche-intelligenz/more-ki-agents-do-not-always-mean-better-results-the-fallacy-in-detail - More AI Agents Do Not Always Mean Better Results
[^434^] https://arxiv.org/html/2408.14811v1 - Brain-inspired Artificial Intelligence: A Comprehensive Review
