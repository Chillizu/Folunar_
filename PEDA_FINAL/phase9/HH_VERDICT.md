# Phase 9 Direction 2 — Hierarchical Horizon: Closing Verdict

**Date:** 2026-08-02
**Author:** L1-HH (closing analysis; data produced 2026-07-31 sweep, commit `1c40068`)
**Status:** VERDICT — direction **ALIVE at 20x20 Variant B**, architecture simplified to open-loop (FF3)
**Authoritative plan:** `PEDA_FINAL/phase9/PHASE9_PLAN.md` (unchanged); design detail `PEDA_FINAL/phase9/plans/plan-hierarchical-horizon.md`

---

## 0. Data provenance

| Artifact | Path | Notes |
|---|---|---|
| Per-episode data | `results/phase9_hierarchical_sweep.jsonl` (129 MB, 3384 episodes, seeds 42–53, meta commit `1c40068`) | WATCHDOG D4 compliant: meta header + per-episode records |
| Summary (sweep's own) | `results/phase9_hierarchical_sweep_summary.csv` (282 rows) | Written by `scripts/phase9_hierarchical_sweep.py` `summarize()` |
| Sweep runner | `scripts/phase9_hierarchical_sweep.py` (run-time version = commit `1c40068`; HEAD adds only CSV-header/data-loss/resume fixes — no behavioral change to the sweep) | diff `1c40068..HEAD` = 7+/3−, all in `write_meta`/`summarize`/CLI |
| HH code | `src/phase9/hierarchical/{planner,executor,runner,goals}.py` — **unchanged** between `1c40068` and HEAD | `git diff 1c40068 HEAD -- src/phase9/hierarchical/` empty |
| Analysis scripts (this verdict) | `scripts/phase9_hh_treeval_check.py`, `scripts/phase9_hh_recompute.py` | committed with this verdict |

**Independent recomputation (this verdict):** `scripts/phase9_hh_recompute.py` re-aggregates SCR/FHT/success per config directly from the per-episode JSONL and reconciles against the summary CSV: **282/282 rows match** (SCR tolerance 5e-4, FHT tolerance 0.05, 0 mismatches). The summary CSV is therefore **trustworthy**; all numbers below are the JSONL-recomputed values and agree with the CSV.

---

## 1. t_reeval 疑点排查（阻塞性问题）— 结论：参数**已接线且会触发**，但在判定关口近乎惰性

### 1.1 代码证据（接线链完整）

| 环节 | 文件:行 | 内容 |
|---|---|---|
| Sweep 配置 → runner | `scripts/phase9_hierarchical_sweep.py:272,279` | `t_reeval = config.get("t_reeval", None)` → 传入 `run_condition(..., t_reeval, ...)` |
| runner 消费 | `scripts/phase9_hierarchical_sweep.py:213` | layered 分支 `run_layered_episode(..., T_reeval=t_reeval, ...)`（random_goal 分支 :203 硬编码 `T_reeval=None`，符合设计 §6） |
| 重估触发条件 | `src/phase9/hierarchical/runner.py:200-201` | `elif T_reeval is not None and (t + 1) % T_reeval == 0:` → `planner.re_evaluate(H_plan, lam, tau)` |
| 切换门槛（滞回） | `src/phase9/hierarchical/planner.py:329`（`re_evaluate` 定义）、`:355` | 仅当 `best != current and best_j > cur_j + max(tau * cur_j, 1.0)`（τ=0.15，下限 1.0）才切换 |
| 目标完成优先 | `src/phase9/hierarchical/runner.py:185-200` | `if goal_done: ... elif T_reeval ...` — 目标完成分支优先，重估只在**目标未完成**时可能触发 |

### 1.2 实证：参数确实被消费并能改变轨迹

- **119 次 `switched` 事件**存在于 goal_log（全部 354091 条 goal 记录中；`scripts/phase9_hh_treeval_check.py` 全量扫描），分布在 15 个配置单元。
- **26/90 个 (variant, size, H, λ) 单元存在逐种子轨迹差异**（对 12 个种子逐一比对 `(x,y,action)` 序列）。差异集中在 **H=20**（短视界下目标快速完成、重估有机可乘），以及少量 λ=0/λ=2 的长视界单元。
- **具体反例（非"参数从未生效"）**：20x20 B, H=100, λ=2, seed 45 — `t_reeval=10` 和 `25` 各触发 1 次切换，第 150 步轨迹分叉（never 向东 `go east`，t10/t25 向南 `go south`），但两者最终访问的细胞集合相同 → 聚合 SCR/FHT 恰好一致（0.3425 / −1）。这解释了"CSV 三行相同"与"轨迹不同"可以并存。

### 1.3 为什么在判定关口（20x20 B, H≥50, λ=0.5/1）完全惰性

- **H=100 下目标几乎每步完成**：对 headline 配置（λ=0.5, H=100, 20x20 B, seed 42）逐集检查，247 个 goal 全部为 nav 类，其中 **211 个在 1 步内完成**（steps=1），`realized=1`。`goal_done` 分支每步都命中 → `elif T_reeval` 分支几乎永不执行。
- **滞回门槛挡住切换**：即便重估执行，当前目标仍是 argmax，需 `best_j > cur_j + max(0.15·cur_j, 1.0)` 才切换，长视界下 `cur_j` 大，门槛高。
- 后果：20x20 B 全网格上 re-eval 相对 open-loop 的 **max ΔSCR = +0.0150**（H=20, λ=2），且部分配置 re-eval **有害**（λ=1 h20: −0.0004；λ=0 h50: −0.0040）。

### 1.4 结论（两种表述都成立，按任务要求写清）

1. **"参数未接线"为假**：链路完整（证据 §1.1），且确实触发过（119 次 switched、26/90 单元轨迹分叉、seed 45 具体反例）。
2. **"re-eval 从不触发"在判定关口近似为真**：20x20 B 的 H=50/100 + λ=0.5/1 单元零切换、逐种子轨迹完全相同（`same_traj=YES`, `n_switched=0`）；全网格最大收益 +0.0150 < 0.02 → **FF3 触发：重估回路是死重，改为 open-loop**（见 §2.3）。
3. 对 sweep 结论的影响：**re-eval 维度在 20x20 B 判定上无效**，但 layered 的胜负不依赖它——所有通过的门都在 open-loop（T=never）下成立，且 open-loop 恰好是最好配置（best SCR 0.5458 即 T=never）。

---

## 2. FF-HH-* 逐门判定（全部数字 = JSONL 独立复算，与 summary CSV 对账一致）

**判定锚点**：PHASE9_PLAN.md 声明 FF 门 "anchored at 20x20 Variant B"；kill rules（设计文档 §7）明确 FF1/FF2 在 20x20 判定生死。

| 门 | 定义（阈值） | 判定 | 测量值（来源） | 结论 |
|---|---|---|---|---|
| **FF-HH-1** | `layered(best) vs random_goal`：dead iff ΔSCR<0.05 **AND** ΔFHT<20 | **PASS** | 20x20 B：best=λ0,H50,T=never，SCR 0.5458 vs random 0.2604 → **ΔSCR=+0.2854**；FHT 235.0 vs 133.0 → **ΔFHT=+102.0**（CSV 行 255/283）| 目标选择携带信号 ✓ |
| | 次级臂 20x20 A | PASS | best=λ0,H100：ΔSCR=+0.3227，ΔFHT=+20.8（CSV 行 249/283 对应 A 侧） | 双指标均过 |
| **FF-HH-2** | λ 扫描非平坦：range across λ < 0.05 **for every H_plan** | **PASS（锚点读法）** | 20x20 B open-loop：H20 range=0.1419，H50=0.2375，H100=0.1958（全部 ≥ 0.05）| 延迟奖励权衡真实存在 ✓ |
| | ⚠ 字面读法（设计文档 "judged at 20x20 and 15x15"） | FAIL（需裁决） | 15x15 B H=100 range=0.0222 < 0.05；15x15 A 全部 < 0.05；20x20 A H=20 range=0.0250 < 0.05 | 见 §2.2 分析 |
| **FF-HH-3** | `layered(·,never) vs layered(·,T∈{10,25})`：ΔSCR < 0.02 | **TRIGGERED → 简化 open-loop** | 20x20 B 全网格 max Δ = **+0.0150**（H20 λ2: 0.4452→0.4602）；其余 ≤ +0.0106 或为 0/负 | 重估回路死重 → 去掉闭环，保留分层 |
| **FF-HH-4** | `layered(best) vs flat_count`：dead iff SCR<base+0.05 AND FHT>base+20 | **PASS** | 20x20 B：layered 0.5458 vs flat 0.1592 → **ΔSCR=+0.3867**（≥0.05 ✓）；flat 从未命中目标（FHT=−1，0/12 成功），layered 6/12 成功、命中均值 FHT=235 → FHT 条件不构成 kill | 分层胜过 count 基线 ✓ |
| **FF-HH-5** | 仅在 10x10 取胜 → marginal | **PASS** | 全尺寸取胜：10x10 B Δ+0.1808 / 15x15 B Δ+0.2489 / 20x20 B Δ+0.3867；A 侧 +0.2667/+0.3752/+0.4548 | 非 marginal ✓ |
| **Positive bar** (§6) | 20x20 B 双指标胜 flat；20x20 A/B 双指标胜 random | **MET** | B vs flat：ΔSCR +0.3867（FHT 比较退化，flat 无命中）；B vs random：ΔSCR +0.2854、ΔFHT +102.0；A vs random：ΔSCR +0.3227、ΔFHT +20.8 | 方向存活，可进沙盒迁移 |

### 2.1 关键数字复核（来源 = 文件:行/配置，与 CSV 对账）

| 数字 | 值 | 来源 |
|---|---|---|
| flat_count 20x20 B | SCR=0.1592, success=0.0, FHT=−1 (0/12 命中) | CSV 行 237 |
| random_goal 20x20 B | SCR=0.2604, success=0.0833, FHT=133.0 (1/12) | CSV 行 283 |
| layered best 20x20 B | **λ=0, H=50, T=never**: SCR=0.5458, success=0.5, FHT=235.0 (6/12) | CSV 行 255 |
| 2026-08-02 review 引用的 headline 配置 | **λ=0.5, H=100**: SCR=0.5417, success=**0.4167**（review 写 50% 不准确）, FHT=226.0, dSCR_vs_flat=+0.3825, dSCR_vs_random=+0.2812, dFHT_vs_random=+93.0 | CSV 行 240 |
| λ 扫描（20x20 B, H=50, open-loop） | 0.0→0.5458, 0.5→0.4794, 1.0→0.4758, 2.0→0.4535, inf→0.3083 | CSV 行 252–256 |
| 15x15 B H=100 λ 扫描 | 0.0→0.4537, 0.5→0.4485, 1.0→0.4489, 2.0→0.4437, inf→0.4659（range 0.0222） | JSONL 复算（CSV 对应行） |
| 20x20 A H=20 λ 扫描 | range=0.0250（H=20 平坦） | JSONL 复算 |

> **对账结论**：独立复算（per-episode JSONL 逐集聚合）与 summary CSV **282/282 行一致**（`scripts/phase9_hh_recompute.py` 节 G）。review 摘要中的 SCR/dSCR/dFHT 数字全部核实无误；唯一不准确处为 λ=0.5,H=100 的 success 率（0.4167=5/12，非 50%——50% 属于 λ=0,H=50 配置）。

### 2.2 FF-HH-2 的读法歧义（需最上级裁决，未擅自改阈值）

- 设计文档 FF2 原文："judged at 20x20 and 15x15"。按此字面读法：**15x15 B H=100 range=0.0222 < 0.05 → FAIL**（15x15 A 全 H 亦 FAIL；20x20 A H=20 亦 FAIL）。
- 但：PHASE9_PLAN.md（唯一权威文档）FF 段标题为 "anchored at **20x20 Variant B**"，kill rules 明确 "FF1 or FF2 fail **at 20x20** → dead"。20x20 B 三个 H_plan 的 range 全部 ≥ 0.05 → **按锚点读法 PASS**。
- 设计文档自身也预告了 15x15 H=100 的平坦性："A horizon covering 25%+ of the maze flattens J(f) (FF2); that interaction is a finding, not a bug"。15x15 直径≈28，H=100 覆盖整图 → J(f) 平坦是**预告过的视界缩放效应**，不是"scoring 是噪声"的证据。
- **建议**：以 PHASE9_PLAN.md 锚点读法（20x20 B）记录 FF2=PASS，并在结果元数据中注明 15x15 B H=100 的平坦性为已预告的视界效应。若验收方坚持字面读法，FF2 判 FAIL → 方向 dead；需在验收时明确选择，本报告不做事后调阈值。

### 2.3 FF3 触发的机制解释与架构含义

- 机制：长视界下 nav 目标 1 步完成（§1.3），重估无机会；且滞回门槛高。H=20 才有少量切换（119 次 switched 中绝大多数在 H=20）。
- 含义：**方向存活但以 open-loop 形态**——去掉 re-eval 循环，保留两层结构（高层目标选择 + 低层 BFS/novelty 执行）。这正是设计文档 FF3 的失败裁定路径："Loop is dead weight → run planner open-loop; hierarchy survives"。

### 2.4 附：PE_goal 指标观察（非门，供后续参考）

- PE_goal 在所有配置下极高（nav 0.95–0.997，search 0.93–0.96，acquire 0.82–0.92）：`G_pred`（整球未访问计数）与 `realized`（单次承诺实际新增 1–2 格）严重不匹配——当前实现下 PE_goal 几乎恒≈1，**不具判别力**。若未来把 goal-level 预测误差当信号用，需重定义（如按承诺全程累计 realized）。

---

## 3. COUNT_DRIVEN_CHARTER 开放问题回答："count 在 8400 态断裂"

**问题**：count-based novelty 在 20x20（设计文档引用 ~8400 态）断裂（PEDA_CONCLUSION 行 6：Count 0%，JEPA 0%），这是状态空间规模的硬上限吗？

**回答（基于本 sweep 数据）**：

1. **规模数字**：本 sweep 的 20x20 B maze 实际 `state_estimate()`=**54400**（400 格 × 136 项，135 个 item + 1；`src/phase6/maze_generator.py:339-344`），比引用的 8400 更大。Count 在该规模 success=0.0、SCR=0.1592，确认断裂。
2. **断裂不是规模本身，而是目标选择/视界**：
   - Count 在 **15x15（更小）就已 collapse**：A 侧 success=0.083、SCR=0.2767；B 侧 success=0.0、SCR=0.2170。10x10 也仅 success=0.417、SCR=0.445（设计文档假设 10x10 是 ceiling/100%，本 sweep 配置下未达到——10x10 的 ceiling guard **未触发**）。
   - **同一 count 低层**（`NoveltySearchExecutor` 内部就是 `MazeNoveltyExplorer`，与 flat_count 完全相同的策略），换成高层长期视界目标选择后，20x20 B 达到 success=0.5、SCR=0.5458。**唯一变量是目标选择**。
   - 结论：count 的 8400 态断裂是 **horizon-1 贪婪选择的失败**（回溯惩罚 + 局部环路 → `dead_loop_rate` 高、每步新增状态率低），不是状态容量问题。**layered 提供了 count 缺失的延迟奖励估计，且无需任何学习模型**——这正面回答了 charter 开放问题：count 断裂可通过分层视界目标选择修复，方向 2 的价值正在于此。

---

## 4. 后续建议

1. **采纳 open-loop layered 作为方向 2 交付形态**（FF3）：从 runner 移除 re-eval 分支（或保留代码但标注 dead weight），后续实验统一 T=never。
2. **layered + count 组合（推荐下一步实验）**：当前 layered 已是"高层目标 + count 低层"的组合；下一步按设计文档 §8 做**沙盒迁移**（未知图 → 前端密度启发式 `G(f)≈|frontier-neighbors|`，低层复用 Phase-8 NoveltyExplorer），验证 transfer。λ 建议取 0–0.5（20x20 B 上 λ=0 最优，λ 惩罚在纯覆盖任务中无益）。
3. **FF-HH-2 读法裁决**（§2.2）需在验收时明确记录，避免结果依赖未言明的解释。
4. **PE_goal 重定义**（§2.4）若作为 PEDA 式信号，需按承诺全程累计 realized。
5. **10x10 ceiling 假设修正**：设计文档假设 10x10 count ≥0.95 SCR，实测 0.445——后续计划文档应更新该前提（10x10 并非 ceiling，但 layered 仍全胜，不影响 FF5 判定）。

---

## 5. 产出文件与提交

| 文件 | 内容 |
|---|---|
| `PEDA_FINAL/phase9/HH_VERDICT.md` | 本判定文档 |
| `scripts/phase9_hh_treeval_check.py` | t_reeval 接线 + 逐种子轨迹比对（90 单元全网格） |
| `scripts/phase9_hh_recompute.py` | FF-HH-1..5 独立复算 + 282 行 CSV 对账 |

**Commit:** `5445afb`（dev 分支，message: `phase9: HH verdict — t_reeval wired-but-inert at gate arm; direction ALIVE at 20x20B, simplify to open-loop (FF3)`）

**验收清单自检**：✓ t_reeval 结论含代码证据路径:行号（§1.1）；✓ FF-HH-* 逐门判定表（§2）；✓ 独立复算数字与 summary CSV 对账 282/282 一致（§2.1）；✓ commit hash 已记录（§5）；✓ COUNT_DRIVEN_CHARTER 回答（§3）；✓ 后续建议（§4）；✓ FF2 歧义诚实标注 UNVERIFIABLE-待裁决而非擅自改阈值（§2.2）。
