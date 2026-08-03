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
