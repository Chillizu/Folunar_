# PEDA 项目文件指引

## 文件夹内容总览

本文件夹包含 PEDA (Predictive-Error-Driven Autonomous Agent) 项目的全部最终交付文件。

### 核心权威文档（根目录）

| 文件 | 用途 | 给哪种Agent |
|------|------|-----------|
| `RESEARCH_CHARTER.md` | 研究宪章：核心问题、负结果标准、成功定义 | **所有Agent先读** |
| `PEDA架构设计与开发计划书_v1.1.docx` | 核心技术文档：架构设计+实现方案+开发路线图 | **Coding Agent / Planning Agent** |
| `peda_report_v11.agent.final.md` | 同上内容的 Markdown 源文件 | 备查/版本控制 |
| `peda_reflection_v11.md` | v1.0问题反思 + v1.1改进说明 | **Coding Agent（避坑指南）** |
| `peda_independent_review.md` | 独立第三方评审报告（5.5/10） | **Review Agent（评审标准示例）** |
| `README_FOR_AGENTS.md` | 本文件 | 所有Agent先读 |

### 归档（`archive/`）

| 目录 | 内容 | 阶段 |
|------|------|------|
| `archive/phase1/` | Phase 1 Grid World 验证报告、gap分析、部分训练评估 | 已完成，核心假设未验证 |
| `archive/phase1_5/` | Phase 1.5 TextWorld 实验报告、偏差分析、评估 | 已完成，epistemic 信号初步验证 |
| `archive/phase2/` | Phase 2 控制器指令、基础设施报告、adapter 训练报告 | **当前阶段**，formal 目标已达标 |
| `archive/historical/` | Folunar_ 前代项目诊断、思维修正 | 历史参考 |

---

## 快速开始：不同Agent该读什么

### 如果你是 Coding Agent（负责写代码）

**必读顺序**：
1. 先读 `peda_reflection_v11.md`（2分钟）—— 了解前代项目犯了什么错，避免重蹈覆辙
2. 再读 `PEDA架构设计与开发计划书_v1.1.docx`（完整技术参考）

**重点关注**：
- 第3章：五大模块的接口定义和数据流
- 第4章：Phase 1的具体实现步骤（先做这个！）
- 第5章：四大原则（没有Prompt只有Prediction Error、Drive是涌现的不是硬编码的...）和四大陷阱检查清单

**绝对不要做的事**（来自反思报告）：
- 不要每步在线SGD训练（灾难性遗忘）
- 不要硬编码目标列表
- 不要让模块数超过功能数
- 不要用<1M参数模型做World Model
- 不要把随机性当成创造力

---

### 如果你是 Planning Agent（负责制定计划）

**必读**：
- `PEDA架构设计与开发计划书_v1.1.docx` 第4-6章（实现方案+路线图+评估指标）
- `peda_independent_review.md`（了解评审提出的问题，在计划中规避）

**关键约束**：
- Phase 1 是"未达标则停止"的决策点（2-3周）
- Phase 1.5 是第二个决策点（3-4周，TextWorld中间验证）
- 总时间线：29-40周（诚实估计，不要压缩）
- 所有评估指标必须量化（不要"行为有趣"这种主观指标）

---

### 如果你是 Review Agent（负责评审代码/设计）

**必读**：
- `peda_independent_review.md` —— 作为评审风格和标准的参考

**评审维度**：
1. 技术可行性（能否在现有条件下实现？）
2. 与文档声称的一致性（代码是否实现了文档描述的功能？）
3. 工程实用性（资源需求是否合理？时间线是否现实？）
4. 潜在盲点（作者可能遗漏的风险或替代方案）

**输出格式**：
- 综合评分（1-10分）
- 每个维度的具体评价
- 可操作的改进建议

---

## 核心原则（所有Agent必须遵守）

### 第一原则：没有Prompt，只有Prediction Error
- 系统的驱动信号是内部预测误差，不是外部用户输入
- 不要添加"让用户输入来触发"的功能

### 第二原则：Drive是涌现的，不是硬编码的
- Drive的权重根据历史表现动态调整
- 不要写死目标列表或固定权重

### 第三原则：World Model是核心，其他是辅助
- 80%精力投入World Model的准确性
- 新增模块必须直接帮助WM预测得更准

### 第四原则：学习是间歇的，不是连续的
- 收集一批数据 -> 批量更新 -> 固定权重运行
- 不要每步在线更新

---

## 已知局限（诚实声明）

PEDA v1.1 是一个**有理论支撑但核心假设未经实验验证**的设计。

- World Model在Linux环境中的预测准确率**未知**（目标是分层目标，非保证）
- epistemic/aleatoric分解是**启发式的**（ensemble近似，非精确解）
- Drive System的超参数敏感性**未测试**
- 推理速度可能限制rollout深度（需要实际测量）
- **不是AGI，不是意识，不是生命**——是一个尝试用预测误差驱动行为的工程系统

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-07-02 | 初始版本，独立评审评分 5.5/10 |
| v1.1 | 2026-07-02 | 基于评审反馈改进，详见 `peda_reflection_v11.md` |

---

## 联系

GitHub: https://github.com/Chillizu/Folunar_
