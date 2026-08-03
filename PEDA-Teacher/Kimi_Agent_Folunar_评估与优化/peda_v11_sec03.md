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
