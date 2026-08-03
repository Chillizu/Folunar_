## 4. 实现方案：从Grid World到Docker沙箱的工程路径

本章将PEDA架构转化为可执行的工程计划。实现路径分为三个阶段：Phase 1用极简环境验证核心假设，Phase 2在真实Linux沙箱中构建World Model，Phase 3整合行动选择并完成系统评估。每个阶段都有明确的成功标准——如果某一阶段未达标，应立即停止并分析问题，而非盲目进入下一阶段。

---

### 4.1 技术选型

#### 4.1.1 World Model：为什么选LLM + LoRA

World Model是PEDA的核心组件，需要在给定当前状态和动作的情况下预测下一状态。这里有两个候选方案：

**选项A：预训练LLM（Qwen2.5-1.5B-Instruct）+ LoRA微调**

- 优势：具备生成能力，可以直接输出状态描述；预训练知识提供了强大的先验；可以通过prompt工程快速迭代；1.5B参数在消费级GPU上可运行
- 劣势：推理成本较高（每步约0.5-2秒）；需要处理LLM的幻觉问题
- 适用场景：状态空间为文本描述、需要语义理解的环境（如Linux shell交互）

**选项B：TinyBERT / DistilBERT + 分类头**

- 优势：推理速度快（每步<100ms）；参数量小（<100M），易于训练
- 劣势：仅能做分类，无法生成状态描述；需要预定义状态类别；不具备语义理解能力
- 适用场景：状态空间有限且可枚举的简单环境（如Grid World）

**推荐：选项A。** PEDA的目标是让Agent在开放的Linux环境中自主探索，状态空间是文本描述的（命令输出、文件内容等），不是可枚举的类别。World Model需要生成能力来做rollout想象——给定"当前目录有a.txt和b.txt，执行`cat a.txt`"，模型需要预测输出内容，这是一个生成任务而非分类任务。此外，LoRA微调只训练<1%的参数，在保护基础模型知识的同时实现高效适应。

具体配置：使用`peft`库，LoRA rank=8，target_modules=["q_proj", "v_proj"]，`lora_alpha=16`，训练时只更新LoRA参数，基础模型冻结。

#### 4.1.2 运行环境：继承与改进

PEDA的执行环境继承Folunar_的Docker沙箱方案，但做关键调整：

```dockerfile
# 基础镜像继承Folunar_的配置
FROM ubuntu:22.04

# 关键区别：不再--network none
# 允许只读访问man pages和技术文档
RUN apt-get update && apt-get install -y \
    man-db manpages manpages-dev manpages-posix \
    coreutils binutils util-linux \
    curl wget vim nano \
    python3 python3-pip \
    gcc g++ make \
    git \
    && rm -rf /var/lib/apt/lists/*

# 挂载外部知识卷（定期注入新信息）
# docker run -v /host/knowledge:/mnt/knowledge:ro ...

# 创建agent用户（非root执行）
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent
```

与Folunar_的关键区别：
- **网络策略**：Folunar_使用`--network none`完全隔离；PEDA允许只读访问本地文档（man pages、/usr/share/doc），但不允许对外网络访问
- **外部知识注入**：通过Docker volume定期挂载外部数据集（如新的技术文档、代码仓库），保持环境的开放性
- **文件系统**：保留Folunar_的`/proc`、`/sys`、`/etc`完整文件系统，Agent可以读取系统状态

#### 4.1.3 推理引擎：分阶段策略

| 阶段 | 推理方式 | 成本/速度 | 适用场景 |
|------|----------|-----------|---------|
| Phase 1-2 | LLM API（deepseek-chat / qwen-turbo） | $0.001-0.01/步，~1s/步 | 快速迭代、数据收集 |
| Phase 3 | 本地模型（Qwen2.5-1.5B + LoRA） | ~$0/步，0.5-2s/步（RTX 4090） | 长期运行、成本控制 |

Phase 1-2使用API的原因：开发初期需要频繁调整prompt和微调策略，API提供最快的迭代速度。预计到Phase 2结束，累计调用约5000-10000次，总成本$50-100。

Phase 3切换到本地的原因：需要Agent连续运行24-48小时评估，API成本不可持续。本地部署使用`vllm`或`transformers` + `bitsandbytes` 4-bit量化，RTX 4090（24GB）足够运行1.5B模型。

#### 4.1.4 记忆系统：三层架构

```
短期记忆（Context Window）
  └── LLM自带的上下文窗口（32K tokens）
  └── 存储：最近N步的(state, action, state')历史
  └── 作用：支持rollout想象时的连贯性

中期记忆（Vector DB）
  └── ChromaDB（本地嵌入式，零配置）
  └── 存储：过去经验的embedding向量
  └── 检索：给定当前状态，找最相似的历史经验
  └── 作用：避免重复探索，利用已有知识

长期记忆（FactGraph）
  └── 继承Folunar_的结构化知识图谱
  └── 存储：持久化的facts（如"apt-get install 需要sudo"）
  └── 更新：间歇性从经验中提取（非每步更新）
  └── 作用：跨session的知识保持
```

三层记忆的分工逻辑：短期记忆保证当前任务的连贯性；中期记忆提供相关历史经验的快速检索；长期记忆保存经检验的"知识"，避免灾难性遗忘。

---

### 4.2 Phase 1：极简验证（2-4周）

Phase 1的目标是回答一个核心问题：**预测误差是否能有效驱动探索？** 如果答案是否定的，整个PEDA的方向就需要重新审视。

#### 4.2.1 环境设计：5x5 Grid World

将环境简化到极点，排除一切无关复杂度：

```python
# grid_world.py - 极简环境
class GridWorld:
    def __init__(self):
        self.size = 5
        # 0=空地, 1=墙壁, 2=目标, 3=陷阱
        self.grid = [
            [0, 1, 0, 0, 2],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 3, 0],
        ]
        self.agent_pos = (0, 0)
        self.actions = ['up', 'down', 'left', 'right']
    
    def step(self, action):
        """执行动作，返回(next_state, reward, done)"""
        x, y = self.agent_pos
        dx, dy = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}[action]
        nx, ny = x+dx, y+dy
        
        # 碰撞检测
        if nx < 0 or nx >= 5 or ny < 0 or ny >= 5 or self.grid[nx][ny] == 1:
            nx, ny = x, y  # 撞墙，位置不变
        
        self.agent_pos = (nx, ny)
        cell = self.grid[nx][ny]
        
        state_desc = f"Agent at ({nx},{ny}), cell={cell}"
        reward = 1.0 if cell == 2 else -1.0 if cell == 3 else 0.0
        done = cell in [2, 3]
        
        return state_desc, reward, done
```

状态表示是纯文本描述（如"Agent at (2,3), cell=0"），不需要视觉处理。这排除了CV的干扰，让我们专注于核心机制。

#### 4.2.2 World Model的Phase 1实现

Phase 1的World Model不需要LLM——一个简单的前馈网络或甚至规则系统就足够：

```python
class SimpleWorldModel:
    """Phase 1的极简World Model：预测(state, action) -> next_state"""
    
    def __init__(self):
        self.transitions = {}  # (state, action) -> {next_state: count}
        self.total_visits = defaultdict(int)
    
    def predict(self, state, action):
        """预测下一状态，同时返回预测不确定性"""
        key = (state, action)
        self.total_visits[key] += 1
        
        if key not in self.transitions:
            return "unknown", 1.0  # 从未见过，最大不确定性
        
        counts = self.transitions[key]
        total = sum(counts.values())
        most_likely = max(counts, key=counts.get)
        
        # 预测误差 = 1 - 最高概率（不确定性越高，潜在信息增益越大）
        max_prob = counts[most_likely] / total
        prediction_error = 1.0 - max_prob
        
        return most_likely, prediction_error
    
    def update(self, state, action, next_state):
        """观察到一个transition后更新模型"""
        key = (state, action)
        if key not in self.transitions:
            self.transitions[key] = defaultdict(int)
        self.transitions[key][next_state] += 1
```

这个World Model本质上是一个计数表（lookup table），但它已经能产生预测误差——当Agent尝试从未做过的(state, action)组合时，预测误差最高。

#### 4.2.3 预测误差驱动的行动选择

```python
def select_action(state, world_model, epsilon=0.1):
    """选择能最大化预测误差（信息增益）的动作"""
    best_action = None
    max_pe = -1
    
    for action in ['up', 'down', 'left', 'right']:
        _, pe = world_model.predict(state, action)
        if pe > max_pe:
            max_pe = pe
            best_action = action
    
    # epsilon-greedy：偶尔随机探索
    if random.random() < epsilon:
        return random.choice(['up', 'down', 'left', 'right'])
    
    return best_action
```

核心逻辑：**Agent倾向于选择它最不确定结果的动作**。这不是随机探索——是有信息偏好的探索。

#### 4.2.4 评估：与随机基线对比

```python
def evaluate(agent_type='pe_driven', max_steps=1000):
    """评估探索效率"""
    env = GridWorld()
    wm = SimpleWorldModel()
    visited = set()
    
    for step in range(max_steps):
        state = f"Agent at {env.agent_pos}"
        visited.add(env.agent_pos)
        
        if agent_type == 'pe_driven':
            action = select_action(state, wm)
        else:
            action = random.choice(['up', 'down', 'left', 'right'])
        
        next_state, _, done = env.step(action)
        wm.update(state, action, next_state)
        
        if done:
            env.agent_pos = (0, 0)  # 重置
    
    return len(visited)  # 覆盖了多少个不同的格子
```

**成功标准**：预测误差驱动的Agent在1000步内访问的不同格子数是随机Agent的2倍以上。达到这个标准，说明预测误差确实是一个有效的探索驱动信号——Phase 1通过，进入Phase 2。未达标则分析问题：是预测误差不敏感？还是环境太简单/太复杂？

---

### 4.3 Phase 2：World Model构建（4-8周）

Phase 1验证了预测误差的驱动能力。Phase 2的目标是在真实环境中构建一个可用的World Model——从Grid World升级到Docker中的Linux沙箱。

#### 4.3.1 环境：Docker Linux沙箱

```python
# sandbox_env.py - Docker沙箱环境
import docker
import subprocess

class LinuxSandbox:
    """Docker中的Linux沙箱环境"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            'peda-sandbox:latest',
            detach=True,
            tty=True,
            volumes={
                '/host/knowledge': {'bind': '/mnt/knowledge', 'mode': 'ro'}
            },
            mem_limit='512m',
            cpu_period=100000,
            cpu_quota=50000,  # 限制50% CPU
        )
        self.history = []
    
    def execute(self, command):
        """执行bash命令，返回(state_before, action, state_after)"""
        state_before = self._get_state()
        
        # 执行命令（超时保护）
        try:
            result = self.container.exec_run(
                ['/bin/bash', '-c', command],
                timeout=10
            )
            output = result.output.decode('utf-8', errors='replace')
            exit_code = result.exit_code
        except Exception as e:
            output = str(e)
            exit_code = -1
        
        state_after = self._get_state()
        
        transition = {
            'state_before': state_before,
            'action': command,
            'state_after': state_after,
            'output': output,
            'exit_code': exit_code,
            'timestamp': time.time()
        }
        self.history.append(transition)
        
        return transition
    
    def _get_state(self):
        """获取当前系统状态（关键文件、进程、环境变量）"""
        state = {}
        
        # 当前目录和文件列表
        state['pwd'] = self._exec('pwd')
        state['files'] = self._exec('ls -la')
        
        # 环境变量
        state['env'] = self._exec('env | sort')
        
        # 运行中的进程
        state['processes'] = self._exec('ps aux')
        
        # 系统信息
        state['uptime'] = self._exec('uptime')
        state['memory'] = self._exec('free -h')
        
        # 最近修改的文件
        state['recent_files'] = self._exec('find . -maxdepth 2 -mtime -1 -type f 2>/dev/null | head -20')
        
        return state
```

状态表示是一个结构化的字典，包含文件系统、进程、环境变量等多维度信息。这个状态表示是World Model的输入。

#### 4.3.2 数据收集：自由交互

Agent在沙箱中自由交互，收集(state, action, state')三元组：

```python
def collect_data(env, num_steps=10000):
    """自由交互数据收集"""
    data = []
    
    for step in range(num_steps):
        # Phase 2早期：随机动作（探索）
        # Phase 2后期：使用初步训练的World Model指导探索
        if step < 5000:
            action = generate_random_command()
        else:
            action = select_action_with_wm(env, world_model)
        
        transition = env.execute(action)
        data.append(transition)
        
        # 每100步保存一次checkpoint
        if step % 100 == 0:
            save_checkpoint(data, f'data/checkpoint_{step}.jsonl')
    
    return data

def generate_random_command():
    """生成随机但合法的bash命令"""
    templates = [
        'ls {path}', 'cat {file}', 'echo {text} > {file}',
        'mkdir {dir}', 'cd {dir} && ls', 'ps aux | grep {pattern}',
        'df -h', 'free -h', 'uptime', 'uname -a',
        'find {path} -type f | head -10',
        'head -5 {file}', 'tail -5 {file}',
        'wc -l {file}', 'sort {file} | uniq -c | sort -rn | head',
    ]
    return random.choice(templates)
```

数据收集分为两个阶段：前5000步随机探索，后5000步使用初步World Model指导探索。这样确保既有广泛的覆盖，也有深度的高价值区域探索。

#### 4.3.3 World Model训练：LLM + LoRA微调

这是Phase 2的核心工程任务：

```python
# train_world_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json

def prepare_training_data(raw_transitions):
    """将原始transition数据转换为训练样本"""
    samples = []
    for t in raw_transitions:
        # 输入：当前状态 + 拟执行的动作
        input_text = format_state(t['state_before']) + '\n$ ' + t['action'] + '\n'
        # 输出：预测的下一状态
        output_text = format_state(t['state_after'])
        
        samples.append({
            'text': input_text + output_text,
            'input': input_text,
            'output': output_text
        })
    return samples

def train():
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B-Instruct',
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=8,                    # LoRA rank
        lora_alpha=16,          # 缩放系数
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    
    # 加载数据
    with open('data/transitions_10000.jsonl') as f:
        transitions = [json.loads(line) for line in f]
    train_data = prepare_training_data(transitions)
    
    # 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir='./wm_checkpoints',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,  # LoRA可用较大学习率
            logging_steps=10,
            save_steps=100,
            fp16=True,
        )
    )
    trainer.train()
    
    # 保存
    model.save_pretrained('world_model_lora_final')
```

训练目标是最小化预测状态与实际状态的差异。这不是标准的next-token prediction——需要自定义loss函数来惩罚关键状态变量（如文件是否存在、进程是否运行）的预测错误。

#### 4.3.4 评估指标

| 指标 | 定义 | 目标值 |
|------|------|--------|
| 预测准确率 | 关键状态变量预测正确的比例 | >70% |
| 命令执行预测 | 预测命令输出是否与实际一致 | >60% |
| 泛化准确率 | 对训练时未见过的命令的预测准确率 | >50% |
| 预测误差衰减 | 预测误差随训练步数的下降曲线 | 单调下降 |

评估方法：将收集到的数据按8:2划分训练/测试集，在测试集上评估预测准确率。特别关注对新颖命令的泛化能力——如果World Model只能拟合训练数据而不能泛化，它的实用价值有限。

---

### 4.4 Phase 3：整合与评估（4-6周）

#### 4.4.1 集成EFE-based行动选择

Phase 2训练出了可用的World Model。Phase 3将其与EFE（Expected Free Energy）框架整合，实现完整的PEDA循环：

```python
class PEDAAgent:
    """完整的PEDA Agent"""
    
    def __init__(self, world_model, drive_system):
        self.wm = world_model
        self.drives = drive_system  # Curiosity/Competence/Boredom/Novelty
        self.memory = ChromaDBMemory()  # 中期记忆
    
    def select_action(self, state, candidate_actions, n_rollouts=5):
        """EFE-based行动选择"""
        action_scores = {}
        
        for action in candidate_actions:
            efe_total = 0
            
            for _ in range(n_rollouts):
                # 1. 想象：用World Model做rollout预测
                predicted_next = self.wm.imagine(state, action)
                
                # 2. 计算信息增益（预测不确定性）
                info_gain = self.wm.prediction_uncertainty(state, action)
                
                # 3. 计算驱力满足度
                drive_satisfaction = self.drives.evaluate(predicted_next)
                
                # 4. 计算EFE = 信息增益 + 驱力满足度
                efe = self.drives.weights['curiosity'] * info_gain + \
                      self.drives.weights['competence'] * drive_satisfaction + \
                      self.drives.weights['boredom'] * (-self._boredom_penalty(action)) + \
                      self.drives.weights['novelty'] * self._novelty_bonus(predicted_next)
                
                efe_total += efe
            
            action_scores[action] = efe_total / n_rollouts
        
        # 选择EFE最小的动作（最小化自由能）
        return min(action_scores, key=action_scores.get)
    
    def _boredom_penalty(self, action):
        """惩罚重复的动作"""
        recent_actions = self.memory.get_recent_actions(n=20)
        return recent_actions.count(action) / len(recent_actions)
    
    def _novelty_bonus(self, predicted_state):
        """奖励新颖的状态"""
        similar = self.memory.find_similar_states(predicted_state, k=5)
        return 1.0 / (1.0 + len(similar))  # 越不相似，奖励越高
```

#### 4.4.2 长期运行评估

这是PEDA最关键也最具挑战性的评估——让Agent自主运行24-48小时，观察其行为：

```python
def long_term_evaluation(agent, env, duration_hours=48):
    """长期运行评估"""
    start_time = time.time()
    end_time = start_time + duration_hours * 3600
    
    behavior_log = []
    check_interval = 300  # 每5分钟记录一次
    
    while time.time() < end_time:
        state = env.get_state()
        
        # 生成候选动作（从近期经验 + 随机生成）
        candidates = generate_candidates(state, agent.memory)
        
        # EFE选择
        action = agent.select_action(state, candidates)
        
        # 执行
        result = env.execute(action)
        
        # 记录
        behavior_log.append({
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'result': result,
            'drive_weights': agent.drives.get_weights(),
            'prediction_error': agent.wm.get_recent_pe()
        })
        
        # 间歇性学习（每500步）
        if len(behavior_log) % 500 == 0:
            agent.learn_from_recent_experiences()
            save_checkpoint(agent, behavior_log)
    
    return behavior_log
```

**评估维度**：

1. **行为是否"有趣"**（人工评估）：
   - 阅读行为日志，判断Agent的行为是否具有目的性
   - 是否展现出"尝试理解环境"的迹象（如系统地查看文件、尝试命令组合）
   - 评分标准：1-5分，3分以上视为"有趣"

2. **是否有"成长"迹象**：
   - 早期行为 vs 晚期行为的对比
   - 是否从简单的`ls`进化到更复杂的命令组合
   - 是否展现出对环境的"理解"（如知道需要先`mkdir`再`cd`）

3. **量化指标**：
   - 探索效率：单位步数内访问的不同目录数、执行的不同命令数
   - 行为多样性：行为序列的Shannon熵
   - 预测误差趋势：预测误差是否随时间下降（学习效率）
   - Drive权重变化：Curiosity是否让位于Competence（从探索到利用）

**通过标准**：Agent在48小时内展现出可观察的行为多样性，且量化指标呈正向趋势（探索效率提升或保持、预测误差下降或稳定、行为熵不持续下降）。

---

Phase 3通过意味着PEDA的核心架构已经跑通——预测误差确实能驱动Agent在真实环境中产生有趣的行为。此后可以进入Phase 4的扩展（新环境、新能力），但在此之前，所有扩展都应以Phase 3的评估框架为基础验证。
