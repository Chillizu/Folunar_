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
