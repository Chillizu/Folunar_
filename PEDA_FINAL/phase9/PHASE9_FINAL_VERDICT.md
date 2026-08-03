# Phase 9 终裁决（PHASE9_FINAL_VERDICT）

**日期**: 2026-08-03 | **状态**: CLOSED | **dev HEAD**: d3fe457
**性质**: 预注册 fail-fast 纪律下的最终封盘。所有门判定先于数据登记，无一事后调整；NULL 如实记录为 NULL。

---

## 一句话裁决

> 零任务知识约束下，count 家族机制在浅层/dist-1 环境有真实且强的效果（91.1%）；
> 在真深度树（dist≥2）上，效果边界 ≈10-13%（vs 盲选 0%）——**效果存在、有限、有界**。
> 最后一层墙的定性：**不确定 ≠ 有价值**。epistemic 信号指向新颖，不指向目标。
> 这与 PEDA 19 实验的结论是同一命题的第二次独立确认。

## 三方向终局

| 方向 | 终裁决 | 关键数字 |
|---|---|---|
| D1 CI 沙盒 | **FF-CI-6 PASS（弱阳性，非劣界内）** | PE 0.400 vs count 0.367（+3.3pp）；PE 学习斜率 3× vs count 1.75×。全项目首个 agent-level PE 正信号 |
| D2 层级时域 | **ALIVE（增益域收窄至 dist-1）** | 迷宫 dSCR +0.387；沙盒 41/45、deep-path 7/10；泛化弱 PASS（8/40 ≥ 7/40）但 dist≥2 全臂 0/30 |
| D3 假设生成器 | **DEAD（二次确认）** | 根因修复使 held-out 0→65%，aggregate 仍 < count（对称失败） |

## 收官证据链（2026-08-03，五环全部预注册）

| 环 | 实验 | 裁决 | 回答的问题 |
|---|---|---|---|
| 1 | FF-SBH-2/3 | PASS | 分层在沙盒携带信号：39/45 → 41/45，deep-path 2/10 → 7/10，零回退 |
| 2 | FF-GEN-1 | 弱 PASS | 泛化非劣成立，但增益域收窄到 dist-1；旧 deep 成果部分归因 start_cwd handout |
| 3 | FF-CEIL-1 | 诊断 | 预算墙次要（s20 deep 3/30、dist-1 9/10）；机制墙主导 |
| 4 | FF-MLP-1 | KILL | 路径规划 38 次机械到达 dist≥2、0 成功——瓶颈是方向信息，非预算非可达性 |
| 5 | FF-PEC-1 | NULL | PE 罗盘：deep 0→3/30（s10）、4/30（s20 > SBH 3/30）。方向信号真实存在但强度不足 |

## 机制贡献清单（无论成败，这些是确凿学到的）

1. **count 新颖性**：<1000 态浅层环境的可靠机制，62.2%→91.1%（速赢+R1），零任务知识。
2. **horizon 分层**：count 的 8400 态断裂是 horizon-1 贪婪问题，不是规模问题（迷宫+沙盒双环境一致）。
3. **T1 冷启动根治**：路径级规划 t=0 即导航，cold-start 失败 7→0。
4. **λ 维度终审**：单步选择下跨全项目零分叉（判死）；路径规划下首次分叉（λ0 深导航/λ05 全浅）——λ 语义依赖规划结构。
5. **R1 空目录陷阱恢复**：高频检验 13 次全对（每次代价 2-4 步）。
6. **PE 的两次独立定性**：作驱动不行（PEDA 19 实验）；作方向启发真实但弱（PEC NULL）。不确定 ≠ 有价值。

## 未测方向（明示出界）

**任务条件化语义先验**（让 LLM 看任务描述排序候选目录）：离开「零知识探索」charter 范畴，属 LLM agent 工程。若未来开线，是新项目而非本线延续。

## 产物索引

- 报告：`results/phase9_sbh_report.md`、`phase9_sbh_failure_analysis.md`、`phase9_sbh_r1_report.md`、`phase9_gen_report.md`、`phase9_ceil_report.md`、`phase9_mlp_report.md`、`phase9_pec_report.md`、`phase9_hg_f5_report.md`、`phase9_hg_f5_rerun_report.md`
- 代码：`src/phase9/sandbox_hh/`（planner/agent/path_planner/pe_compass + runner）、`src/phase9/gen_tasks.py`、`Dockerfile.busybox_v5`
- 数据：results/ 下 19 份 JSONL（WATCHDOG D4 meta 全含 git commit），~700 episodes
- 归因链：src/phase8、src/phase5 全程零 diff（每环验收验证）

## 封盘声明

Phase 9 三个方向 + 五个收官实验，全部按预注册门走完。无悬空主张、无事后阈值、无未记录失败。
本研究线关闭。遗留资产见上方清单；复现入口为各报告内 reproduce 命令。
