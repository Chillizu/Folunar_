# PEDA (Predictive-Error-Driven Autonomous Agent) 架构设计与开发计划

## 1. 执行摘要
- PEDA的核心创新：用预测误差替代Prompt作为驱动力
- 理论支撑：FEP + Predictive Coding + World Models
- 关键设计决策
- 预期成果

## 2. 理论基础：为什么需要新范式
- 2.1 Prompt范式的根本限制
- 2.2 Active Inference / Free Energy Principle (Friston et al.)
  - 2.2.1 核心思想：感知、行动、学习的统一
  - 2.2.2 Variational Free Energy的数学推导
  - 2.2.3 Expected Free Energy (EFE)与策略选择
  - 2.2.4 Epistemic Value vs Pragmatic Value
- 2.3 Predictive Coding (Rao & Ballard 1999; Clark 2013/2015)
  - 2.3.1 分层预测编码
  - 2.3.2 预测误差作为学习信号
  - 2.3.3 与反向传播的关系 (Millidge et al. 2022)
- 2.4 World Models (Ha & Schmidhuber 2018; Hafner et al. 2019-2025)
  - 2.4.1 RSSM架构
  - 2.4.2 Dreamer系列演进
  - 2.4.3 长程预测的挑战与解决
- 2.5 内在动机与好奇心的FEP视角
  - 2.5.1 ICM/RND的局限 (Noisy TV问题)
  - 2.5.2 FEP如何统一探索与利用
  - 2.5.3 信息增益 vs 预测误差
- 2.6 连续时间认知 (CTRNN, LTC, EMBER)
  - 2.6.1 自发行为的动力学基础
  - 2.6.2 与离散LLM的结合策略

## 3. PEDA架构设计
- 3.1 核心哲学：从"Prompt驱动的推理"到"Prediction驱动的存在"
- 3.2 系统架构总览
  - 3.2.1 五大模块及其关系
  - 3.2.2 数据流与控制流
  - 3.2.3 与传统Agent架构的对比
- 3.3 World Model (世界模型)
  - 3.3.1 职责：预测环境状态变化
  - 3.3.2 输入/输出接口设计
  - 3.3.3 在文本环境中的具体实现
- 3.4 Predictive Error Computer (预测误差计算)
  - 3.3.1 误差类型：感知误差 vs 模型误差
  - 3.3.2 误差分解：可约 vs 不可约
  - 3.3.3 误差作为"内在驱动"信号
- 3.5 Action Generator (行动生成器)
  - 3.5.1 EFE最小化作为策略选择
  - 3.5.2 Rollout-based想象
  - 3.5.3 从离散选择到连续控制的谱系
- 3.6 Learning Module (学习模块)
  - 3.6.1 World Model的更新
  - 3.6.2 预测误差的衰减与学习饱和检测
  - 3.6.3 知识蒸馏：从探索到利用
- 3.7 Homeostatic Drive System (内稳态驱动系统)
  - 3.7.1 为什么需要：纯粹的预测误差不够
  - 3.7.2 Drive设计：Curiosity / Competence / Boredom / Novelty
  - 3.7.3 Drive与FEP的结合：Epistemic Foraging
  - 3.7.4 Drive的动态平衡与自发行为

## 4. 实现方案
- 4.1 技术选型
  - 4.1.1 World Model：预训练LLM + LoRA微调 vs 专门训练
  - 4.1.2 运行环境：Docker沙箱 + 可控开放性
  - 4.1.3 推理引擎：LLM API vs 本地模型
  - 4.1.4 记忆系统：向量数据库 + 知识图谱
- 4.2 Phase 1：极简验证（2-4周）
  - 4.2.1 环境：Grid World / 文本迷宫
  - 4.2.2 目标：验证预测误差是否能驱动探索
  - 4.2.3 评估指标：探索效率、行为多样性
- 4.3 Phase 2：World Model构建（4-8周）
  - 4.3.1 数据收集：Agent在沙箱中的交互数据
  - 4.3.2 模型训练：预测"命令→状态变化"
  - 4.3.3 评估：预测准确率、泛化能力
- 4.4 Phase 3：整合与评估（4-6周）
  - 4.3.1 EFE-based行动选择
  - 4.3.2 Drive System集成
  - 4.3.3 长期运行评估：是否产生"有趣"的行为

## 5. Agent指引：防止跑偏的核心原则
- 5.1 第一原则：没有Prompt，只有Prediction Error
- 5.2 第二原则：Drive是涌现的，不是硬编码的
- 5.3 第三原则：World Model是核心，其他是辅助
- 5.4 第四原则：学习是间歇的，不是连续的
- 5.5 常见陷阱检查清单
  - 5.5.1 "模板陷阱"——不要让系统退化到固定模式
  - 5.5.2 "穷举陷阱"——探索效率必须持续提升
  - 5.5.3 "幻觉陷阱"——不要把随机性当成创造力
  - 5.5.4 "膨胀陷阱"——模块数不等于智能
- 5.6 决策流程图：何时添加新模块？何时删除？

## 6. 开发路线图
- 6.1 里程碑与时间线
- 6.2 资源需求
- 6.3 风险评估与应对
- 6.4 成功标准定义

## 7. 结论
- 7.1 PEDA的边界：什么能做到，什么不能
- 7.2 与Folunar_的关系：继承什么，抛弃什么
- 7.3 最终建议
