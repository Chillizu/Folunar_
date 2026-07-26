# Phase 2 基础设施报告

## 完成状态

| 任务 | 状态 | 耗时 |
|------|------|------|
| Docker 环境 | ✅ | 30 min |
| SandboxEnv Python 接口 | ✅ | 20 min |
| JSON 结构化状态 | ✅ | 内置在 SandboxState |
| Confidence Penalty | ✅ | 5 min |
| Perception SandboxState 分派 | ✅ | 10 min |
| WorldModel Sandbox 文本模式 | ✅ | 20 min |
| ActionGenerator 提示词 + 守卫 | ✅ | 15 min |
| DriveSystem 守卫 + 字符串 action 支持 | ✅ | 10 min |
| **完整循环验证** | ✅ | 10 min |
| Data collection script | ❌ | 范围外 |
| 100+ 步验证 | ❌ | 范围外 |

**总实际耗时**: ~120 分钟

## Docker 环境

- 镜像: `peda-sandbox:latest`
- 基础: `busybox:latest`（3.7 MB）
- 预置文件: `hello.txt`, `docs/note.txt`, `data/lines.txt`
- 白名单命令: `ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep`
- 黑名单拦截: `rm, mv, cp, chmod, chown, dd, mkfs, mount, sudo, su`
- 安全约束: `--cap-drop=ALL --read-only --tmpfs /tmp --network none`

安全验证全部通过：rm 被只读挂载拦截，网络被 `--network none` 阻断，/tmp 可写。

## PEDA 集成

### 修改的文件

| 文件 | 修改 | 类型 |
|------|------|------|
| `src/phase2/sandbox_env.py` | 新增：BusyboxSandbox + SandboxState + candidate generator | 新增 |
| `src/phase1/grid_env.py` | `render_text()` SandboxState分支（`to_json()`分派） | 修改 |
| `src/phase1/world_model.py` | `_sandbox_system_message()`、`_build_text_prompt()`沙箱指令、`_llm_predict()`沙箱JSON解析、`rollout()` SandboxState更新、`decompose_error()`沙箱JSON字段方差、`_actual_exit_code()`沙箱守卫 | 修改 |
| `src/phase1/drive_system.py` | `compute_efe()` SandboxState/TextState合并守卫、ConfidencePenalty（>0.95注入）、`_action_entropy()`字符串兼容、`apply_to_efe()`字符串兼容 | 修改 |

### 接口契约

```python
# 所有 env 类型共享的接口（GridWorld / TextRoomEnv / BusyboxSandbox）
env.reset(seed=None) -> State
env.step(state, action: str) -> (State, reward, done)
```

```python
# SandboxState.to_json() -> JSON 结构化状态表示
{
  "cwd": "/sandbox",
  "files": ["docs", "data", "hello.txt", "tmp"],
  "last_command": "ls",
  "last_exit_code": 0,
  "last_output": "...",
  "step": 1
}
```

### Confidence Penalty

在 `compute_efe()` 的 TextState/SandboxState 路径中，如果 trajectory 的平均预测置信度 > 0.95，注入 `+0.3 * (conf - 0.95)` 的 EFE 惩罚。这能打破 Phase 1.5 中的 inventory 死循环。

### 字符串 action 兼容

`drive_system.py` 新增 `_action_name()` 辅助函数，统一处理 `Action` 对象和字符串。所有 `.name` 访问已替换为 `_action_name()`。

## 循环验证

```
Initial: cwd=/sandbox files=['docs', 'data', 'hello.txt', 'tmp']
Candidates: ['ls', 'pwd', 'id', 'ls data', 'ls docs', 'cat hello.txt', 'cat tmp', 'cd docs']
Selected: id
Result: exit=1 (不在白名单中)
Selected2: ls data
Result2: exit=0 -> lines.txt
```

PEDA 选择了 `ls data`（有效探索命令）。Docker 执行成功，状态更新正确。

## 遗留问题

| 问题 | 严重性 | 说明 |
|------|--------|------|
| `id` 在 candidates 但不在 whitelist | 低 | candidate generator 未过滤白名单 |
| stub_predict 无 SandboxState 守卫 | 低 | `--stub` 模式不可用于沙箱 |
| decompose_error 只比较 files 字段 | 中 | 需要扩展至 cwd + files + last_output |
| No LoRA fine-tune for sandbox | 中 | 当前用 TextRoomEnv 适配器预测沙箱状态 |

## 下一步

1. **提交+推送**：当前基础设施可运行
2. **Task 3**：创建 `scripts/phase2_collect_data.py`（多基线对比）
3. **Task 4**：运行 100+ 步，收集 (s,a,s') 三元组
4. **LoRA 微调**：在沙箱数据上微调 WorldModel

## 关键限制

### World Model 未在沙箱数据上训练
**这是 Phase 2 基础设施当前最大的限制。**

循环验证中使用的 World Model 是 `text_adapter_e4`（在 TextRoomEnv 的 114 条文本数据上训练）。该模型从未见过 Linux 沙箱的 JSON 结构化状态表示。

后果：
- World Model 对 JSON 状态的预测本质上是随机输出（预训练模型的文本续写能力，不是 PEDA 的 prediction error 机制）
- `decompose_error` 报告的 epistemic/aleatoric 值不可用于行为分析
- PEDA 当前选择 `ls data` 更可能是 DriveSystem 的候选循环，而非有意义的 decision-making

**这个限制不意味着基础设施失败** — 管线本身已验证工作。但核心假设验证需要：
1. 收集 1000+ 沙箱 (s,a,s') 三元组
2. 在沙箱数据上 LoRA 微调 WorldModel
3. 重新运行 PEDA 评估

### candidate generator 包含白名单外的命令
`generate_sandbox_candidates` 包含 "id" 等不在白名单中的命令。这是非破坏性问题（step() 会拦截并返回 exit=1），但浪费候选 slot。

### stub_predict 无 SandboxState 守卫
`--stub` 模式（确定性预测）没有 SandboxState 分支。运行沙箱时不要使用 `--stub`。
