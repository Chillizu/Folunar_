# Phase 2 启动任务：Busybox 沙箱基础设施

> **上游决策**: Phase 1.5 已完成使命 → 进入 Phase 2
> **前置阅读**: `PEDA_FINAL/PHASE1_5_ITERATION2_EVALUATION.md`
> **时间预算**: 2-3 小时（单次会话）
> **目标**: 验证 PEDA 能在真实 Linux 环境中运行，收集第一批 (s,a,s') 数据

---

## 为什么 Phase 2

Phase 1.5 的教训：
- 2 房间文本环境状态空间太小（6000 次尝试 → 114 条去重样本）
- 0.5B 模型 + 114 条数据学不好转移动态
- **人造简单环境无法产生验证核心假设所需的复杂度**

Phase 2 的假设：
- Busybox 沙箱的不确定性是**固有的**（不是人造的）
- Linux 命令输出不可完全预测 → 天然产生 epistemic 信号
- 数据空间足够大（命令组合 >> 114 条）

---

## 任务拆解

### 任务 1：Docker Busybox 环境（30 分钟）

**目标**: 一个 PEDA 可以安全执行命令的最小沙箱。

**要求**:

```dockerfile
# Dockerfile.busybox
FROM busybox:latest
# 不需要额外安装 — busybox 内置 ls, cd, cat, echo, mkdir, touch, rm, pwd, wc 等
WORKDIR /sandbox
# 预置一些文件/目录供 Agent 探索
RUN mkdir -p docs tmp data && \
    echo "hello world" > hello.txt && \
    echo "secret key: 12345" > docs/note.txt && \
    echo -e "line1\nline2\nline3" > data/lines.txt
```

**安全约束**（WATCHDOG C5）:
- `--cap-drop=ALL`（去除所有 Linux capabilities）
- `--read-only` 挂载（根目录只读）
- `--tmpfs /tmp`（临时写权限限制在 /tmp）
- `--network none`（无网络访问）
- 命令白名单：`["ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail", "grep"]`
- 命令黑名单：`["rm", "mv", "cp", "chmod", "chown", "dd", "mkfs", "mount", "sudo", "su"]`

**验证**:
```bash
docker build -f Dockerfile.busybox -t peda-sandbox .
docker run --rm --cap-drop=ALL --read-only --tmpfs /tmp --network none peda-sandbox ls /sandbox
# 应输出：data  docs  hello.txt  tmp
```

**Python 接口**:

```python
# src/phase2/sandbox_env.py
class BusyboxSandbox:
    """Docker busybox 环境，接口与 TextState 兼容"""
    
    def reset(self, seed=None) -> SandboxState:
        """启动新容器，返回初始状态"""
        pass
    
    def step(self, state: SandboxState, action: str) -> tuple[SandboxState, int, bool]:
        """
        在容器中执行命令
        
        Args:
            state: 当前状态（包含容器 ID、当前目录、环境表示）
            action: 命令字符串（如 "ls -la"）
        
        Returns:
            next_state: 执行后的状态
            reward: 0（Phase 2 不使用外部 reward）
            done: 命令是否在白名单中
        """
        pass
    
    def render_text(self, state: SandboxState) -> str:
        """将状态转换为 World Model 可理解的文本描述"""
        pass
```

**状态表示** (`SandboxState`):

> **GLM-5.2 建议**: 使用 JSON 结构化表示，不要自由文本。这能强制 LLM 的注意力对齐到因果状态变更，降低语义鸿沟噪声。

```python
@dataclass
class SandboxState:
    container_id: str      # Docker 容器 ID
    cwd: str               # 当前工作目录
    last_command: str      # 上一个执行的命令
    last_output: str       # 命令输出（截断至 500 字符）
    last_exit_code: int    # 退出码
    files: list[str]       # 当前目录下的文件列表（缓存）
    step_count: int        # 步数
    max_steps: int = 20    # 最大步数（Phase 2 微任务：5-10 步）
    
    def to_json(self) -> str:
        """GLM-5.2 建议：JSON 结构化状态表示"""
        return json.dumps({
            "cwd": self.cwd,
            "files": self.files,
            "last_command": self.last_command,
            "last_exit_code": self.last_exit_code,
            "last_output": self.last_output[:200],
            "step": self.step_count,
        }, ensure_ascii=False)
```

---

### 任务 2：PEDA → 沙箱集成（45 分钟）

**目标**: PEDA 的 Action Generator 能生成命令，Docker 执行命令，Perception 解析输出。

**实现路径**:

1. **新增 `src/phase2/sandbox_env.py`** — 使用 `subprocess.run(['docker', 'exec', ...])` 与容器交互
2. **复用现有模块**:
   - `Perception.render_text()` → 新增 `SandboxState` 分支（hasattr 分派）
   - `ActionGenerator` → 新增 prompt template 生成 Linux 命令
   - `DriveSystem.compute_efe()` → 新增 `SandboxState` 守卫
   - `WorldModel` → 复用文本模式（SandboxState → text prompt）
3. **命令白名单检查** — `sandbox_env.step()` 中强制执行

**Action Generator 的 prompt template**:

```
You are in a Linux sandbox.
Current state: {state_json}

Generate ONE Linux command to explore or interact with the environment.
Command must be from this whitelist: ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep
Respond with ONLY the command, no explanation.
```

**Perception 的 render_text**（GLM-5.2 建议：JSON 结构化，非自由文本）:

```python
def render_text(self, state) -> str:
    if hasattr(state, 'container_id'):  # SandboxState
        return state.to_json()  # JSON 结构化表示
    # ...  existing TextState/GridState branches
```

**World Model 预测格式**:

```python
# World Model 预测下一个 JSON 状态
def predict(self, state_text: str, action: str) -> str:
    """
    输入: JSON 状态 + 动作
    输出: 预测的下一个 JSON 状态（也是 JSON 格式）
    
    示例:
      输入: {"cwd": "/sandbox", "files": ["hello.txt"], ...} + "ls"
      输出: {"cwd": "/sandbox", "files": ["hello.txt", "docs", "data"], ...}
    """
```

**Confidence Penalty**（GLM-5.2 建议：打破 0.999 死循环）:

```python
def compute_efe_with_penalty(self, state, action_candidates):
    """
    标准 EFE 计算 + Confidence Penalty
    
    如果模型对某 action 的预测置信度 > 0.95，强制注入噪声降低其 EFE 权重。
    这能防止 inventory 死循环（Phase 1.5 教训）。
    """
    base_efe = self.compute_efe(state, action_candidates)
    for i, (action, confidence) in enumerate(action_candidates):
        if confidence > 0.95:
            # 降低高置信度 action 的 EFE 优势
            base_efe[i] += self.confidence_penalty_weight * (confidence - 0.95)
    return base_efe
```

---

### 任务 2b：轻量版 JEPA — Hidden State Epistemic（GLM-5.2 强烈推荐，1-2 天）

> **优先级**: 🔴 **最高**。GLM-5.2 两轮咨询中性价比最高的建议。
> **理由**: Token 空间的分歧受语法/同义词影响（aleatoric noise），
> Hidden State 捕捉语义层面的不确定性。epistemic 信号质量可能显著提升。

**架构**:

```python
class LightJEPAEpistemicComputer:
    """
    轻量版 JEPA：复用 LLM 的 hidden states 计算 epistemic uncertainty
    
    零额外参数。不需要 encoder/predictor/decoder。
    只需要修改 Predictive Error Computer 的 forward 逻辑。
    """
    
    def compute_epistemic(self, checkpoints, state_json, action):
        """
        对每个 checkpoint：
          1. 输入 (state_json, action) → LLM 生成 next_state
          2. 提取 last hidden state（layer -1, mean pooled over tokens）
          3. 收集 3 个 hidden state 向量（各 896 维）
          4. 计算两两余弦距离的均值 → epistemic score
        
        为什么改善信号：
        - "key" vs "keys" → token 不同但 hidden state 相似 → 不贡献假阳性分歧
        - 模型对状态转移真的困惑 → hidden states 分歧大 → 真实的 epistemic
        """
        hidden_states = []
        for ckpt in checkpoints:
            outputs = ckpt.generate(
                f"State: {state_json}\nAction: {action}\nPredict next state:",
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # 取 last layer, mean pool over sequence
            last_hidden = outputs.decoder_hidden_states[-1]  # (batch, seq, 896)
            pooled = last_hidden.mean(dim=1).squeeze()  # (896,)
            hidden_states.append(pooled)
        
        # 两两余弦距离
        epistemic = 0.0
        pairs = [(0,1), (0,2), (1,2)]
        for i, j in pairs:
            sim = F.cosine_similarity(hidden_states[i], hidden_states[j], dim=0)
            epistemic += (1 - sim.item())  # distance = 1 - similarity
        return epistemic / len(pairs)
```

**关键实施要点**:
1. `output_hidden_states=True` 在 HF Transformers 中已支持
2. 序列长度对齐：不同 checkpoint 可能生成不同长度 → 用 attention mask 做 mean pool
3. 方差计算：先实现余弦距离（简单），如果信号不够强再试协方差矩阵的迹
4. 保留 token-space epistemic 作为对比基线
5. 向后兼容：新增 `hasattr` 分派，不修改 Grid/Text 路径

**验证**:
- 运行 Phase 1.5 的语义探针，对比 token-space vs hidden-space epistemic
- 预期：hidden-space epistemic 应该与语义探针的"有意义分歧"更一致
- 如果 hidden-space epistemic ≈ token-space → 没有改善，记录并继续
- 如果 hidden-space epistemic >> token-space → 重大改善，后续实验全部使用

---

### 任务 2c：INT4 量化 + 知识蒸馏（后续优化，非优先）

> **来源**: GLM-5.2 追问 3
> **优先级**: 🟢 低。Phase 2 基础设施稳定后考虑。

**INT4 量化**:
- 工具：`llama.cpp` 或 `bitsandbytes` (CPU 模式)
- 效果：内存降至 500MB 以下，CPU 推理 3-4x 加速
- 实施：将 Qwen2.5-0.5B 导出为 GGUF 格式，用 llama.cpp 推理

**知识蒸馏**:
- 方案：用 GPT-4o / Qwen72B API 在 busybox 中跑大量探索
- 收集高质量 (s,a,s') 数据（大模型能完美预测 bash 命令的确定性结果）
- 用这些数据微调 0.5B 模型，快速建立"命令→状态转移"的因果映射
- 成本：API 调用 $10-30

**混合策略"受限的自主性"**（fallback 方案）:
- 如果 PEDA 单独运行效果不佳，可用 Prompt-driven（ReAct）生成 candidate actions
- 然后用 PEDA 的 epistemic 在这些 candidates 中做 final selection
- 这绕过了 0.5B 模型"从零生成命令"的弱点，但改变了核心假设的验证条件
- 仅作为额外基线，不作为默认方案

**验证步骤**:
1. PEDA 生成第一个命令（如 `ls`）
2. Docker 执行，返回输出
3. Perception 解析为文本状态
4. World Model 预测下一个状态
5. Drive System 计算 EFE
6. 选择下一个命令

---

### 任务 3：数据收集脚本（30 分钟）

**目标**: 自动化收集 (s,a,s') 三元组，对比多个基线 Agent 的行为。

**GLM-5.2 建议**：
1. **微任务设计**：单次任务 5-10 步（如"从 / 移动到 /docs 并 cat note.txt"），而非 20 步长 episode
2. **多基线对比**：PEDA / Pragmatic / Random Walk / Heuristic（Random + Boredom）/ Prompt-driven
3. **新指标**：FHT（首次到达目标步数）、SCR（状态覆盖率）、Dead-loop Rate（死循环频率）

**脚本**: `scripts/phase2_collect_data.py`

```python
"""
Phase 2 数据收集：多 Agent 基线对比 + 记录

策略（GLM-5.2 建议）：
1. PEDA → 完整 EFE + Drive System
2. Pragmatic → pragmatic_only（无 epistemic）
3. Random Walk → 完全随机选择合法命令
4. Heuristic → 随机 + Boredom 惩罚（验证 Drive System 是否为 artifact）
5. Prompt-driven → 直接给 Qwen2.5 Few-shot prompt 做决策

微任务设计（5-10 步）：
- "找到 /docs 目录并读取 note.txt"
- "创建 /sandbox/test 目录并在其中创建 hello.txt"
- "统计 data/lines.txt 的行数"

记录：
- (agent_type, task_id, state_json, action, next_state_json, exit_code, confidence, epistemic)
- 每个 agent 每个任务 10+ episodes
- 指标：FHT, SCR, Dead-loop Rate, Success Rate
"""
```

**关键设计（Phase 1.5 教训）**:

1. **状态空间要大**: Linux 命令输出是自由文本，不像 2 房间环境那样高度重复
2. **多样化策略**: 不只是随机，加入：
   - 目录遍历（`cd` 到每个子目录 + `ls`）
   - 文件检查（`cat` 每个文件）
   - 内容创建（`echo` + `touch` + `mkdir`）
   - 搜索（`grep` 不同关键词）
3. **去重但不要过度**: 相同命令在不同目录下是不同的 (s,a) 对
4. **样本数检查**: 每 100 步打印一次去重后的样本数

**配置**:
```python
CONFIG = {
    "max_steps": 1000,           # 单次收集会话步数
    "max_unique_samples": 2000,  # 目标样本数
    "strategy": "mixed",         # "random" | "peda" | "mixed"
    "mixed_ratio": 0.5,          # PEDA 选择比例（mixed 模式）
    "whitelist": ["ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail", "grep"],
    "output_dir": "data/phase2",
}
```

---

### 任务 4：最小验证（30 分钟）

**目标**: 确认 PEDA 能在沙箱中运行并收集数据。

**验证清单**:

```markdown
## Phase 2 基础设施验证

### Docker 环境
- [ ] 容器能启动
- [ ] 命令白名单有效（允许 ls, cat 等）
- [ ] 命令黑名单有效（阻止 rm, chmod 等）
- [ ] 容器隔离（无网络、无特权）

### PEDA 集成
- [ ] Action Generator 能生成合法命令
- [ ] Docker 执行命令并返回输出
- [ ] Perception 使用 JSON 结构化状态（非自由文本）
- [ ] World Model 能预测 JSON 格式的下一个状态
- [ ] Confidence Penalty 生效（置信度>0.95 时注入噪声）
- [ ] Drive System 能计算 EFE
- [ ] Agent 能选择下一个命令

### 多基线运行（GLM-5.2 建议）
- [ ] PEDA（完整 EFE + Drive）能运行
- [ ] Pragmatic（pragmatic_only）能运行
- [ ] Random Walk（纯随机）能运行
- [ ] Heuristic（Random + Boredom 惩罚）能运行
- [ ] Prompt-driven（Few-shot）能运行

### 数据收集
- [ ] 100 步内无崩溃
- [ ] 每个基线至少 3 个微任务 × 3 episodes
- [ ] 输出文件格式正确（jsonl，含 agent_type, task_id, FHT, SCR）

### 行为观察
- [ ] PEDA 是否比 Random Walk 更快到达目标（FHT）？
- [ ] Heuristic 是否也能复现 PEDA 的"尝试新动作"行为？（Drive artifact 验证）
- [ ] Confidence Penalty 是否减少了死循环（Dead-loop Rate）？
- [ ] Prompt-driven 是否优于所有其他方法？（验证"预测误差驱动"的价值）
```

---

## 输出

### 报告格式

写入 `PEDA_FINAL/phase2_infrastructure_report.md`：

```markdown
# Phase 2 基础设施报告

## 完成状态
| 任务 | 状态 | 耗时 |
|------|------|------|
| Docker 环境 | Y/N | X min |
| PEDA 集成 | Y/N | X min |
| 数据收集脚本 | Y/N | X min |
| 最小验证 | Y/N | X min |

## Docker 环境
- 镜像大小：X MB
- 启动时间：X ms
- 白名单命令数：X
- 安全验证：（cap-drop, read-only, network none 是否生效）

## PEDA 集成
- 新增/修改文件清单
- 向后兼容验证（Grid/Text 路径是否仍通过）
- 接口契约说明

## 数据收集
- 运行步数：X
- 去重样本数：X
- 覆盖命令数：X
- 各命令分布：

## 行为观察
- PEDA vs Random 的行为差异（如有）
- 发现的死循环或异常模式
- 下一步建议
```

---

## 约束与提醒

### 硬性约束

1. **总时间 ≤ 3 小时**。超时则汇报已完成部分 + 阻塞项。
2. **安全优先**: 如果 Docker 安全配置搞不定，不要继续。
3. **不要创建 PLAN/ARCH 文档**。
4. **向后兼容**: Grid World 和 Text World 路径必须仍能通过测试。

### 心态提醒

- Phase 2 的目标不是"让 Agent 学会 Linux"，而是"验证 PEDA 能在真实环境中运行"
- 数据收集量不需要达到 10000（计划书的数字）。Phase 2a 的第一步是 1000+ 即可。
- Agent 行为可能很蠢（反复 `ls`）。记录但不惊讶。
- 如果 100 步内崩溃，这是 bug 不是"模型不够聪明"。

### Phase 1.5 + GLM-5.2 教训应用

| 教训 | 来源 | 应用 |
|------|------|------|
| 数据增强在简单环境无效 | Phase 1.5 | Linux 环境天然数据丰富，不需要人造增强 |
| decompose_error 维度不完整 | Phase 1.5 | SandboxState 设计时就包含完整维度 |
| 模型学不好短任务链 | Phase 1.5 | 不追求任务完成，先追求稳定运行 |
| 统计显著性不足 | Phase 1.5 | 使用微任务（5-10步）+ FHT/SCR 指标 |
| **JSON 结构化状态** | GLM-5.2 Q3 | SandboxState.to_json() 替代自由文本 |
| **Confidence Penalty** | GLM-5.2 Q4 | 置信度>0.95 注入噪声，打破死循环 |
| **多基线对比** | GLM-5.2 Q6/Q7 | PEDA/Pragmatic/Random/Heuristic/Prompt-driven |
| **Drive artifact 验证** | GLM-5.2 Q6 | Heuristic 基线测试 Drive System 是否伪探索 |
| **Token-space vs Latent-space** | GLM-5.2 Q5 | Phase 2 用 JSON token-space；失败时评估 JEPA |
| **微任务设计** | GLM-5.2 Q7 | 5-10 步子任务替代 20 步长 episode |

---

## 上游评估标准（GLM-5.2 更新版）

| 标准 | 权重 | 通过门槛 |
|------|------|----------|
| Docker 环境安全可用 | 20% | 白名单/黑名单生效，容器隔离 |
| PEDA 能运行完整循环 | 25% | generate → execute → perceive → predict → select |
| JSON 结构化状态 + Confidence Penalty | 20% | 状态为 JSON 格式，penalty 生效 |
| 多基线对比运行 | 20% | ≥3 个基线（PEDA/Pragmatic/Random）成功运行 |
| 报告质量 | 15% | 诚实记录问题，含 FHT/SCR/Dead-loop 指标 |

**GLM-5.2 关键指标定义**：
- **FHT (First Hitting Time)**: 首次到达目标状态的步数。越低越好。
- **SCR (State Coverage Rate)**: N 步内探索到的不同状态数 / 总合法状态数。越高越好。
- **Dead-loop Rate**: 陷入置信度>0.95 循环的频率。越低越好。

**go/no-go**: 如果前两项通过，Phase 2 基础设施就绪。
**GLM-5.2 附加条件**: 如果 Prompt-driven 基线显著优于 PEDA，需要分析"预测误差驱动"是否优于"直接指令驱动"。
** artifact 验证**: 如果 Heuristic（Random + Boredom）与 PEDA 行为无显著差异，Drive System 被确认为 artifact（WATCHDOG C14）。
