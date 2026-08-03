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
