"""Phase 9 sandbox-hh: PE 罗盘 (FF-PEC-1) — WM epistemic-uncertainty direction term.

把 FF-CI-6 的预测误差信号作为方向项接入 MLP λ0 路径规划器（path_planner.py）：
对路径候选中「下一跳为未知目录」的节点，用 WorldModel 预测 `ls <dir>` 的结果
（L2 级 = 预测下一状态的 files 列表），取 epistemic 不确定性（ensemble 成对
方差，charter 定义）作为方向信号 s(dir)：

    J(path) = prior(end) - lam*depth + gamma*s(dir),   lam=0 固定, gamma=1 固定

PE 栈逐行复用 FF-CI-6（scripts/phase9_ci_m3.py + phase1/phase2 机制，考古结论
见 results/phase9_pec_report.md §2）：
  * 同一模型：Qwen2.5-0.5B-Instruct（/home/data/models/Qwen2.5-0.5B-Instruct）
  * 同一 L1/L2/L3 预测协议：WorldModel.predict(state, action) 的 sandbox 路径
    （_sandbox_system_message + _build_text_prompt；L2 = cwd/files JSON）
  * 同一 epistemic 定义：EnsembleErrorComputer.decompose_error 的 sandbox 分支
    ev 公式（pred_exits 分歧 + pred_files-set 分歧，成对平均 /2）；无 checkpoint
    时单模型（n=1 -> ev=0，罗盘惰性 —— 与 CI M3 前 2 个 episode 的
    mean_epistemic_error=0.0 同构）
  * 同一学习机制：SandboxLearningModule（LoRA, batch_size=1, epochs=1，
    lr=2e-4）+ 每 episode 一次 update + save_checkpoint 建 ensemble（CI 的
    UPDATE_INTERVAL=20 按 10 步 episode 等比折半为 10；buffer_size 100）

零任务知识：WM 只从环境动态（observed transitions）学习；任务描述从不进入
WM prompt（任务条件化语义先验是另一条研究线，本实验不测，仅 out-of-scope）。

成本闸：WM 查询只在重选点（select 事件）发起，每 episode <= 3 次；WM 不可用
/异常/超预算 -> fallback 中性 0.5 并计数（此时该次选择退化为 MLP λ0）。
n<2 个 ensemble 成员时 s 恒为 0（charter 定义），不消耗预算、不计 fallback。

实现说明（与盘上旧文件的差分）：planner.py / agent.py / path_planner.py 零改
动；本模块新增 PECPlanner（PathPlanner 子类）+ PECAgent（SandboxPathAgent
子类，仅覆盖 __init__/_select_and_log/_choose_action/run_episode —— run_episode
是基类循环的逐行拷贝 + WM 学习钩子）。归因：PEC 与 MLP λ0 的唯一差别是
gamma*s(dir)，任何 deep 增益归因于 PE 信号本身。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from phase1.types import ErrorVector, Experience
from phase1.world_model import EnsembleErrorComputer, WorldModel
from phase2.run import SandboxLearningModule
from phase2.sandbox_env import SandboxState
from phase8.count_driven_agent import (
    Phase8Explorer,
    _get_task,
    _task_start_cwd,
    generate_phase8_candidates,
)

from .agent import SandboxHHAgent, _has_replay
from .path_planner import PathPlanner, SandboxPathAgent

MODEL_PATH = "/home/data/models/Qwen2.5-0.5B-Instruct"

# CI M3 (FF-CI-6) 的 UPDATE_INTERVAL=20 对应 20 步 episode；本实验 episode 为
# 10 步（s10）/20 步（s20），等比取 10 —— 每 episode 至多一次 LoRA 更新。
UPDATE_INTERVAL = 10
BUFFER_SIZE = 100
MAX_ENSEMBLE = 5            # 与 EnsembleErrorComputer.num_checkpoints 默认一致
QUERIES_PER_EPISODE = 3     # 成本闸：WM 查询每 episode <= 3 次（预注册）
FALLBACK_S = 0.5            # WM 不可用/超时/超预算 -> 中性 0.5

# 占位 error（compass 不逐步骤做 error 分解 —— 那是 M3 的机制检查，非本实验
# 合约；LM buffer 的 LoRA 更新只用 state/action/next_state，error 仅喂给
# saturation detector）。
_ZERO_ERROR = ErrorVector(
    total_error=0.0, level1_error=0.0, level2_error=0.0, level3_error=0.0,
    epistemic_error=0.0, aleatoric_error=0.0, ensemble_variance=0.0,
)


def _trim_state_for_wm(st):
    """浅拷贝 + file_cache/last_output 裁剪，作为 WM 训练/查询输入。

    环境在 cat/head/wc 时向 state.file_cache 累积文件内容；state.to_json() 把
    它们全量带进 prompt，lora_finetune 的 384-token 上限会被撑爆 —— prompt 截
    断把 target 整个切掉 -> 无有效 label -> NaN loss（实测复现：prompt >=384
    token 的样本全部 NaN，v5 任务多文件读取必触发）。裁剪到最近 1 条缓存 x 40
    字符、last_output 截到 80（比 to_json 的 200 更紧），使 state JSON <= ~550
    字符（实测 212-380 字符区间内 prompt 全部 < 384 token）。裁剪的是喂给 WM
    的状态副本，agent 自身状态与盘上栈代码零改动。"""
    if not getattr(st, "file_cache", None) and not getattr(st, "last_output", ""):
        return st
    return st.__class__(
        container_id=getattr(st, "container_id", ""),
        cwd=getattr(st, "cwd", "/sandbox"),
        last_command=getattr(st, "last_command", ""),
        last_output=(getattr(st, "last_output", "") or "")[:80],
        last_exit_code=getattr(st, "last_exit_code", 0),
        files=list(getattr(st, "files", []) or []),
        file_cache={k: v[:40] for k, v in list((getattr(st, "file_cache", {}) or {}).items())[-1:]},
        step_count=getattr(st, "step_count", 0),
        max_steps=getattr(st, "max_steps", 20),
        victory=getattr(st, "victory", False),
        game_over=getattr(st, "game_over", False),
    )


class PECEnsemble(EnsembleErrorComputer):
    """EnsembleErrorComputer，checkpoint 落到 per-task 目录。

    基类 save_lora_checkpoint 写共享的 checkpoints/phase1/adapter_step_{N}；
    多任务共用一个 WorldModel 时 step 号会跨任务碰撞（task2 覆盖 task1 的同
    步号文件，而 predict_with_checkpoint 按存的路径加载）。机制与基类完全相
    同（save_pretrained 快照当前 LoRA 状态 + 滑动窗口保留最近 N 个），仅目录
    隔离。CI M3 单进程内每任务重建 ec 且 task 串行，本实现同样串行。
    """

    def __init__(self, world_model: WorldModel, ckpt_dir, num_checkpoints: int = MAX_ENSEMBLE):
        super().__init__(world_model, num_checkpoints=num_checkpoints)
        self._ckpt_dir = Path(ckpt_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[Path] = []

    def save_checkpoint(self, step: int) -> Path:
        path = self._ckpt_dir / f"adapter_step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        if self.world_model.mode == "llm" and self.world_model.model is not None:
            self.world_model.model.save_pretrained(path)
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.num_checkpoints:
            self.checkpoints.pop(0)
        return path


class PECompass:
    """WM epistemic 罗盘：s(dir) = ensemble 成对方差（charter）的 `cd <dir>` L2 预测。

    查询协议与 EnsembleErrorComputer._predictions_for 逐行一致：
      无 checkpoint -> [wm.predict]（n=1 -> s=0）
      有 checkpoint -> [predict_with_checkpoint(c) for c in checkpoints]
    方差公式 = decompose_error sandbox 分支的 ev 公式，去掉 actual 对照：
      成对 d = (exit 分歧) + (files-set 分歧)，ev = (Σd / C) / 2。
    """

    def __init__(self, wm: WorldModel, ec: PECEnsemble,
                 max_queries_per_episode: int = QUERIES_PER_EPISODE) -> None:
        self.wm = wm
        self.ec = ec
        self.max_queries_per_episode = max_queries_per_episode
        self.queries_this_episode = 0
        self.queries_total = 0
        self.fallbacks = {"budget": 0, "unavailable": 0, "error": 0}
        self.query_log: List[dict] = []
        self._cache: Dict[tuple, float] = {}

    # ── per-episode lifecycle ──────────────────────────

    def start_episode(self) -> None:
        self.queries_this_episode = 0
        self._cache = {}

    # ── charter machinery（逐行复用 CI M3 栈）──────────

    def _ensemble_predictions(self, state, action: str) -> list:
        """同一成员选择逻辑：EnsembleErrorComputer._predictions_for 逐行拷贝。"""
        if not self.ec.checkpoints:
            return [self.wm.predict(state, action)]
        return [
            self.wm.predict_with_checkpoint(state, action, ck)
            for ck in self.ec.checkpoints
        ]

    @staticmethod
    def _charter_ensemble_variance(preds) -> float:
        """charter ev 公式（decompose_error sandbox 分支），prediction-only。"""
        n = len(preds)
        if n < 2:
            return 0.0
        pred_exits = [p.level1_exit_code for p in preds]
        pred_files: List[list] = []
        for p in preds:
            try:
                j = json.loads(p.level2_text) if p.level2_text else {}
            except (json.JSONDecodeError, TypeError):
                j = {}
            pred_files.append(list(j.get("files", [])))
        pw = 0.0
        c = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 if pred_exits[i] != pred_exits[j] else 0.0
                d += 1.0 if set(pred_files[i]) != set(pred_files[j]) else 0.0
                pw += d
                c += 1
        return (pw / c) / 2.0 if c else 0.0

    # ── query API ──────────────────────────────────────

    def signal(self, parent_dir: str, child: str, state) -> Tuple[float, str]:
        """s(dir) in [0,1] + source tag（query / cached / no_ensemble /
        fallback_budget / fallback_unavailable / fallback_error）。

        预算语义：每 episode 至多 max_queries_per_episode 次真实 WM 查询
        （命中 _cache 的重用不计费）；n<2 个成员时 s 恒 0（charter），不
        计费、不算 fallback。
        """
        key = (parent_dir, child)
        if key in self._cache:
            return self._cache[key], "cached"

        if self.wm is None or self.wm.mode != "llm" or self.wm.model is None:
            self.fallbacks["unavailable"] += 1
            return FALLBACK_S, "fallback_unavailable"

        # n<2：charter 定义 ev=0，罗盘惰性（等价于 MLP λ0 该项为 0）
        if len(self.ec.checkpoints) < 2:
            return 0.0, "no_ensemble"

        if self.queries_this_episode >= self.max_queries_per_episode:
            self.fallbacks["budget"] += 1
            return FALLBACK_S, "fallback_budget"

        self.queries_this_episode += 1
        self.queries_total += 1
        t0 = time.time()
        try:
            preds = self._ensemble_predictions(state, f"cd {child}")
            s = self._charter_ensemble_variance(preds)
        except Exception as exc:  # OOM / adapter 损坏等 -> fallback
            self.fallbacks["error"] += 1
            self.query_log.append({
                "parent": parent_dir, "child": child, "s": FALLBACK_S,
                "src": "fallback_error", "n_members": len(self.ec.checkpoints),
                "seconds": round(time.time() - t0, 2), "error": str(exc)[:200],
            })
            return FALLBACK_S, "fallback_error"

        self._cache[key] = s
        self.query_log.append({
            "parent": parent_dir, "child": child, "s": round(s, 4),
            "src": "query", "n_members": len(self.ec.checkpoints),
            "seconds": round(time.time() - t0, 2),
        })
        return s, "query"


class PECPlanner(PathPlanner):
    """PathPlanner + PE 罗盘方向项（FF-PEC-1）。J = prior - lam*depth + gamma*s。

    s 附着语义（实现读法，报告 §方法中说明）：对每条候选路径，取路径链上
    「第一个未访问（unknown）目录」u —— 即下一跳将首次进入未知内容的位置；
    WM 预测从 u 的父目录 `cd <basename(u)>` 的 L2 结果（= ls u 的 files），
    ensemble 方差即 s(u)。最短路径的中间跳都是 known 目录，故首个 unknown
    目录通常就是路径目标；当目标需穿过 named-but-unvisited 中间目录时，
    s 附着在第一个 unknown 上（更贴近「下一跳为未知目录的节点」字面）。
    同一次 select 内多个路径共享同一 (parent, child) 时只查询一次（复用）。
    """

    def __init__(self, lam: float, compass: PECompass, gamma: float = 1.0) -> None:
        super().__init__(lam)
        self.compass = compass
        self.gamma = float(gamma)

    def _first_unknown(self, path: List[str], cwd: str, explorer) -> Tuple[Optional[str], Optional[str]]:
        """(first-unknown-dir, its-parent) along the cd chain; (None, None) if none."""
        cur = cwd
        parent: Optional[str] = None
        for step in path:
            parent = cur
            if step == "cd ..":
                cur = str(Path(cur).parent)
            else:
                cur = f"{cur}/{step[3:].strip()}"
            if not self._visited(cur, explorer):
                return cur, parent
        return None, None

    def _query_state(self, parent: Optional[str], live_state) -> Optional[SandboxState]:
        """查询用状态：parent == 当前 cwd 用 live state（file_cache 裁剪，
        与训练输入同口径），否则从图 listings 合成（无 file_cache）。"""
        if live_state is not None and parent == getattr(live_state, "cwd", None):
            return _trim_state_for_wm(live_state)
        if parent is None:
            return None
        entries = self.graph.entries.get(parent)
        if not entries:
            return None
        return SandboxState(container_id="pec", cwd=parent, files=sorted(entries),
                            step_count=0, max_steps=20)

    def select_goal(self, cwd: str, explorer, state=None) -> Optional[dict]:
        """Pick the frontier PATH maximizing J = prior - lam*depth + gamma*s.

        state: 当前 live SandboxState（查询父目录 == cwd 时优先使用）。
        """
        scored = []
        for d in self.graph.known_dirs():
            if d == cwd:
                continue
            prior, unknown = self.prior(d, explorer)
            if prior <= 0:
                continue
            path = self.graph.shortest_path(cwd, d)
            if path is None:
                continue
            depth = len(path)
            s, src = 0.0, "none"
            u, parent = self._first_unknown(path, cwd, explorer)
            if u is not None:
                qstate = self._query_state(parent, state)
                if qstate is not None:
                    s, src = self.compass.signal(parent, u.rsplit("/", 1)[-1], qstate)
                else:
                    s, src = FALLBACK_S, "fallback_unavailable"
            j = prior - self.lam * depth + self.gamma * s
            scored.append({
                "goal": d, "prior": round(prior, 4), "unknown": unknown,
                "depth": depth, "s": round(s, 4), "pe_src": src,
                "j": round(j, 4), "path": path,
            })
        if not scored:
            return None
        scored.sort(key=lambda r: (-r["j"], -r["depth"], r["goal"]))
        best = dict(scored[0])
        best["contenders"] = [
            {k: r[k] for k in ("goal", "prior", "unknown", "depth", "s", "pe_src", "j", "path")}
            for r in scored[:5]
        ]
        return best


class PECAgent(SandboxPathAgent):
    """两层 open-loop agent，high layer = PECPlanner（PE 罗盘），低层逐行不变。

    相对 SandboxPathAgent 的覆盖点：
      * __init__ —— planner 换 PECPlanner，挂 wm/ec/lm/compass；
      * _select_and_log / _choose_action —— 把 live state 传给 planner 查询；
      * run_episode —— 基类循环逐行拷贝 + WM 学习钩子（每步 store_experience，
        episode 末 LoRA update + checkpoint -> ensemble 增长）。
    WM 由 runner 传入（单进程共享，跨任务累积 LoRA 状态 —— CI M3 同构）；
    ec/lm/checkpoint 目录按 task 隔离（PECEnsemble）。
    """

    def __init__(self, docker_image: str, task_id: str, lam: float,
                 wm: Optional[WorldModel] = None, ckpt_root: Optional[Path] = None) -> None:
        super().__init__(docker_image, task_id, lam)
        repo_root = Path(__file__).resolve().parents[3]
        ckpt_root = Path(ckpt_root) if ckpt_root is not None else repo_root / "checkpoints" / "phase9_pec"
        self.wm = wm if wm is not None else WorldModel(MODEL_PATH, device="cpu")
        self.ec = PECEnsemble(self.wm, ckpt_root / task_id)
        self.lm = SandboxLearningModule(self.wm, self.ec, buffer_size=BUFFER_SIZE,
                                        update_interval=UPDATE_INTERVAL)
        self.compass = PECompass(self.wm, self.ec)
        self.planner = PECPlanner(lam, self.compass, gamma=1.0)
        self.nan_updates = 0  # NaN loss 回滚计数（数据侧裁剪的兜底）

    # ── select / dispatch（带 live state）──────────────

    def _select_and_log(self, cwd: str, t: int, state) -> Optional[dict]:
        sel = self.planner.select_goal(cwd, self.explorer, state=state)
        if sel is None:
            self.goal_log.append({
                "t": t, "event": "select", "goal": None, "prior": 0.0,
                "unknown": False, "depth": 0, "s": 0.0, "pe_src": "none",
                "j": 0.0, "path": [], "contenders": [],
            })
            return None
        self.goal_log.append({
            "t": t, "event": "select", "goal": sel["goal"],
            "prior": sel["prior"], "unknown": sel["unknown"],
            "depth": sel["depth"], "s": sel["s"], "pe_src": sel["pe_src"],
            "j": sel["j"], "path": sel["path"], "contenders": sel["contenders"],
        })
        return sel

    def _choose_action(self, state, candidates, actions, t: int) -> str:
        """基类 SandboxHHAgent._choose_action 逐行拷贝，仅 select 传入 live state。"""
        if _has_replay(self.explorer, state, candidates):
            return self.explorer.select_action(state, candidates, actions)

        if self.mode == "select":
            sel = self._select_and_log(state.cwd, t, state)
            self.goal = sel["goal"] if sel else None
            if self.goal is not None and self.goal != state.cwd:
                p = self.planner.graph.shortest_path(state.cwd, self.goal)
                if p:
                    self.mode = "navigate"
                    self.path = p[1:]
                    return p[0]
            self.mode = "explore"

        if self.mode == "navigate":
            if state.cwd == self.goal:
                self.goal_log.append({"t": t, "event": "arrive", "goal": self.goal})
                self.mode = "explore"
            elif self.path:
                action = self.path[0]
                self.path = self.path[1:]
                return action
            else:
                self.mode = "explore"

        return self.explorer.select_action(state, candidates, actions)

    # ── WM 学习钩子 ───────────────────────────────────

    def _store_wm_experience(self, state, action: str, next_state) -> None:
        try:
            t_state = _trim_state_for_wm(state)
            t_next = _trim_state_for_wm(next_state)
            exit_code = getattr(next_state, "last_exit_code", 0)
            out = getattr(next_state, "last_output", "")
            summary = f"action {action}: {out[:60]}" if out else action
            self.lm.store_experience(Experience(
                state=t_state, action=action, next_state=t_next,
                error=_ZERO_ERROR, exit_code=exit_code, summary=summary,
            ))
        except Exception as exc:
            print(f"[pec {self.task_id}] WM store_experience failed: {exc}", flush=True)

    def _wm_episode_update(self) -> None:
        """End-of-episode LoRA update + checkpoint（ensemble 增长）。

        NaN 防护（数据侧裁剪的兜底）：若 update 后 LoRA 权重出现 NaN（极长
        state/prompt 仍可能触发 384-token 截断 -> 无 label -> NaN loss），回滚
        到 update 前快照、丢弃本次 checkpoint，ensemble 保持原状 —— 防止 NaN
        权重污染后续查询与下一集更新。快照仅含 requires_grad（LoRA）参数。"""
        try:
            if self.lm.should_update():
                import torch
                snap = {n: p.detach().clone() for n, p in
                        self.wm.model.named_parameters() if p.requires_grad}
                t0 = time.time()
                self.lm.update()   # 内部 lora_finetune + save_checkpoint
                nan = any(torch.isnan(p).any().item()
                          for p in self.wm.model.parameters() if p.requires_grad)
                if nan:
                    with torch.no_grad():
                        for n, p in self.wm.model.named_parameters():
                            if p.requires_grad:
                                p.copy_(snap[n])
                    if self.ec.checkpoints:
                        bad = self.ec.checkpoints.pop()
                        import shutil
                        shutil.rmtree(bad, ignore_errors=True)
                    self.nan_updates += 1
                    print(f"[pec {self.task_id}] NaN in LoRA weights after update — "
                          f"rolled back, checkpoint dropped (ensemble="
                          f"{len(self.ec.checkpoints)})", flush=True)
                else:
                    print(f"[pec {self.task_id}] LoRA update ({time.time()-t0:.0f}s), "
                          f"ensemble={len(self.ec.checkpoints)}", flush=True)
        except Exception as exc:
            print(f"[pec {self.task_id}] WM update failed: {exc}", flush=True)

    # ── episode ────────────────────────────────────────

    def run_episode(self, episode_idx: int, max_steps: int = 10) -> dict:
        """基类 SandboxHHAgent.run_episode 逐行拷贝 + PE 钩子（标注 [PE]）。"""
        state = self.sandbox.reset(seed=episode_idx, start_cwd=self._start_cwd)
        self.planner.graph.observe_cwd(state.cwd, state.files)
        self.explorer.reset_episode()
        self.compass.start_episode()          # [PE]

        self.mode = "select"
        self.goal: Optional[str] = None
        self.path: List[str] = []
        self.goal_log: List[dict] = []
        actions: List[str] = []
        success = False

        for t in range(max_steps):
            candidates = generate_phase8_candidates(state)
            if not candidates:
                candidates = ["ls", "pwd"]
            if t >= max_steps - 1:
                known = getattr(self.explorer, "cd_child", {}).get(state.state_hash(), {})
                candidates = [
                    c for c in candidates
                    if not (c.startswith("cd ") and c != "cd .." and c not in known)
                ]
                if not candidates:
                    candidates = ["ls"]

            action = self._choose_action(state, candidates, actions, t)

            next_state, _reward, done = self.sandbox.step(state, action)

            if action.startswith("cd "):
                self.explorer.record_cd(state, action, next_state)
                if next_state.cwd != state.cwd:
                    self.planner.graph.note_parent(next_state.cwd, state.cwd)
            if self.mode == "navigate" and next_state.cwd == self.goal and state.cwd != self.goal:
                self.goal_log.append({"t": t, "event": "arrive", "goal": self.goal})
                self.mode = "explore"

            check_fn = self.task.get("check")
            if check_fn is not None:
                try:
                    if check_fn(state, action, next_state):
                        success = True
                except Exception:
                    pass

            self.explorer.observe(state, action, success)
            self.buffer.append((state, action, next_state, success))
            try:
                self.action_model.learn_from_step(state, action, next_state, success)
            except Exception:
                pass
            self._store_wm_experience(state, action, next_state)   # [PE]
            actions.append(action)

            if success or done:
                break

            self.planner.graph.observe_cwd(next_state.cwd, next_state.files)
            if action.startswith("find "):
                self.planner.graph.observe_find(next_state.cwd, next_state.last_output)
            state = next_state

            if self.mode == "explore" and (
                self._local_frontier_exhausted(state)
                or (action.startswith("cd ") and self._dir_lacks_text_files(state))
            ):
                self.mode = "select"

        self._wm_episode_update()             # [PE] LoRA + checkpoint（ensemble 增长）

        # [PE] 该 episode 的 WM 查询/回退统计（进 JSONL row，WATCHDOG D4）
        s_terms = [
            {"t": g["t"], "goal": g["goal"], "s": g["s"], "src": g["pe_src"]}
            for g in self.goal_log if g.get("event") == "select"
        ]
        result = {
            "episode": episode_idx,
            "success": success,
            "steps": len(actions),
            "actions": actions,
            "buffer_size": len(self.buffer),
            "goal_log": self.goal_log,
            "wm_queries": self.compass.queries_this_episode,
            "wm_fallbacks": sum(self.compass.fallbacks.values()),
            "wm_nan_updates": self.nan_updates,
            "pe_terms": s_terms,
        }
        self.results.append(result)
        return result
