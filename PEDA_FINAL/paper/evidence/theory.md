# S1Theory — Theory & Architecture Evidence Bundles (PEDA final paper)

**Scout**: S1Theory | **Date**: 2026-07-31
**experiment_ids**: [E01, E02, E03, E04, E05, E06, E07, E08, E09, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19]
**Sub-IDs**: E13.01–E13.11 (Phase 5 JEPA sub-configs), E17.1–E17.5 (Phase 7 GPU tracks). This theory slice does not enumerate sub-config-level numbers (see phase5.md / phase7.md); E13/E17 top-level numbers are cited below.

File path shorthand: REPORT=peda_report_v11.agent.final.md, CHARTER=RESEARCH_CHARTER.md, REFLECTION=peda_reflection_v11.md, REVIEW=peda_independent_review.md, CONCLUSION=PEDA_CONCLUSION.md, COUNTCHARTER=COUNT_DRIVEN_CHARTER.md, MANUSCRIPT=PEDA_RESEARCH_MANUSCRIPT.md. All under /home/chillizu/Projects/Folunar_/PEDA_FINAL/ unless noted.

---

## 1. Files Read (FULL)

- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/RESEARCH_CHARTER.md` (82 lines, full)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/peda_report_v11.agent.final.md` (2055 lines, full; read in contiguous chunks 1–500 / 500–1003 / 1003–1503 / 1503–2055)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/peda_reflection_v11.md` (full)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/peda_independent_review.md` (393 lines, full: 1–300 + 301–393)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md` (full)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/COUNT_DRIVEN_CHARTER.md` (full)
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md` (707 lines; Sections 2 Theory lines 70–161 and 3 Architecture lines 162–313 ONLY, per instructions; Abstract/Results/Conclusion ignored as superseded)
- `/home/chillizu/Projects/Folunar_/AGENTS.md` (context; canonical source of the 4 Immutable Principles and 7-module diagram)

---

## 2. Quantitative Results — (number, source_file:line, verbatim_snippet)

### 2.1 Theory / Architecture constants

- (5.5/10, REVIEW:373, `| **综合评分** | **5.5/10** | **值得Phase 1验证，但不应过度投入**`)
- (7.0/10, REPORT:97, `v1.0: 5.5/10 → **v1.1: 7.0/10**`)
- (10.7%, MANUSCRIPT:213, `a state with 10 independent variables, each 80% predictable, has a joint accuracy ceiling of $0.8^{10} \approx 10.7\%$`)
- (0.8^10 ≈ 10.7%, REPORT §3.3.2, `整体状态预测的联合准确率上限约为 $0.8^{10} \approx 10.7\%$`)
- (≥90% / ≥70%+≥60% / ≥50%, REPORT:503 / 513 / 523, L1/L2/L3 target accuracies)
- (stop-loss <80% / <50% / none, REPORT:505 / 515 / 525)
- (0.3/0.7 removed, REPORT §3.4.3, `epistemic_ratio = (1 - conf) * 0.3 + conf * 0.7`已被移除)
- (5 checkpoints, REPORT:685–688, `class EnsembleErrorComputer:` / `使用多个LoRA checkpoint的ensemble来分解epistemic和aleatoric误差`)
- (0.5/0.5/0.3/0.4, REPORT:1132–1135, `self.weights = DriveWeights(curiosity=0.5, competence=0.5, boredom=0.3, novelty=0.4)`)
- (2.0 / 0.01, REPORT:1147 / 1159, `tanh(2.0 * current_error.epistemic_error)` / `novelty = 1 - exp(-0.01 * time_since_input)`)
- (81 combos, MANUSCRIPT:293, `Grid search over {0.2, 0.5, 0.8} for each drive weight (81 combinations) was planned but not completed in Phase 1`)
- (every 1000 steps, REPORT §3.6.2 / MANUSCRIPT:225, `self.UPDATE_INTERVAL = 1000  # 每1000步触发一次微调`)
- (<15% decline = saturated, MANUSCRIPT:276, `If error decline rate < 15%, the detector signals saturation`)
- (100–200 s/step, REPORT §3.5.2, `每步决策时间：50-100次调用 × 2秒 = **100-200秒/步**`)
- (4–9 calls/step → 8–18 s/step → 9600–21600 steps/48h, REPORT §3.5.2, `2-3个候选 × 2-3步 = 4-9次`)
- (~520–1700 steps/48h unmitigated, MANUSCRIPT §3.2.4 / REVIEW §4.2, `50-100 calls per step would limit 48-hour runs to ~520-1700 steps`)
- (2.8%, REVIEW Risk 2, `10步rollout...整体准确率约为 0.7^10 ≈ 2.8%`)
- (14–20 → 29–40 weeks, REPORT §6.1, `v1.1 估计为 **29–40 周**`)
- ($20–100 → $140–300, REPORT §6.2.1)
- (4 / 65 / 270+ candidates, CONCLUSION:110, `candidate generator evolved from hardcoded heuristics (v1, 4 candidates) to data-driven enumeration (v2, 65 pairs; v3/v4, 270+ pairs) with zero crashes`)
- (0.85 ratio / 0.15 decline, REPORT §3.6.3, `如果比率 > 0.85 → 误差不再显著下降 → 饱和`)

### 2.2 Experimental results (E-ID prefixed)

- E01: (G1=1.000, G2=0.434, G3=0.000, CONCLUSION table row 1, `G1=1.000, G2=0.434, G3=0.000 — WM perfectly memorizes all 25 cells`)
- E02: (0.8684, CONCLUSION table row 2, `` `g1_test_set`=0.8684, epistemic error ~0 from 28/28 state-action probes zero variance ``)
- E03: (2.6 vs 2.6, p=1.0, CONCLUSION table row 3, `PEDA 2.6 vs Pragmatic 2.6 steps goal_unknown, Fisher p=1.0, MW p=1.0`)
- E04: (epistemic 0.20, MANUSCRIPT:534/644, `epistemic signal measurable (0.20) but insufficient for meaningful exploration` / `producing non-zero epistemic signals (0.20 in Phase 1.5 after bug fix)`; CONCLUSION row 4 says `epistemic ~0` — see contradictions §7.4)
- E05: (L1=1.000, L2=0.900, L3=0.550, CONCLUSION table row 5, `thresholds met on training distribution`)
- E06: (L1=0.800, L2=0.686, L3=0.229, CONCLUSION table row 6, `all below threshold`)
- E07: (1.0 s vs 2.8 s, CONCLUSION table row 7, `read_hello: Pragmatic 1.0s > PEDA 2.8s. read_note: ALL 0% success`)
- E08: (7.2 vs 10.0 steps, MW p=0.0043, d=−1.01, CONCLUSION:48/64, `PEDA unknown 7.2 steps vs Pragmatic unknown 10.0, MW p=**0.0043**, d=-1.01`)
- E09: (10.0 vs 6.85 steps, p=0.0043, CONCLUSION:49, `PEDA 10.0 steps vs Pragmatic 6.85 steps, p=0.0043`)
- E10: (20%→60%→80%→60% vs flat 20%, CONCLUSION:50, `PEDA+Train: 20%-60%-80%-60% success. PEDA+Freeze: flat 20%`)
- E11: (40% = 2/5; all zero others, CONCLUSION:51, `read_hello peda_unknown 40% (2/5). count_lines/find_secret/read_note: **all zero** hits`)
- E12: (35–40% vs 0%, CONCLUSION:52, `Phase 3 replicated: peda_unknown 35-40% hit, pragmatic_unknown 0%`)
- E13: (50% vs 17%, CONCLUSION table row 13, `Novelty-only 50% > jepa_efe 17% on read_hello`; 11 exps: `JEPA forward dynamics + hybrid, 11 exps` → sub-IDs E13.01–E13.11 in phase5.md)
- E14: (SCR ~0, CONCLUSION table row 14, `SCR ~0 across all tasks, zero room exploration`)
- E15: (100% / 0% / 67%, CONCLUSION table row 15, `Count: 100% goal-reaching. JEPA: 0%. Hybrid: 67% (count-driven carries JEPA)`)
- E16: (0% / 0%, CONCLUSION table row 16, `Count: 0%. JEPA: 0%. Both agents hit state-space ceiling`)
- E17: (DLR ~0.996, CONCLUSION:57, `All JEPA tracks: DLR ~0.996 (near-perfect determinism, zero epistemic signal). Count wins every track` → sub-IDs E17.1–E17.5 in phase7.md)
- E18: (62.2% = 28/45, CONCLUSION:58/106, `Count-driven: **62.2% avg success rate** across 9 tasks`)
- E19: (zero delta, CONCLUSION:58, `JEPA toggle adds **zero delta**`)
- (37x, CONCLUSION:90, `computed at approximately 37x the computational cost`)
- (45.8% vs 31.3%, CONCLUSION:108, `reach 45.8% learned vs 31.3% fallback on the action prediction task`; COUNTCHARTER §3.3 `+14.5pp`)
- (176 s + ~3 s, CONCLUSION:98, `~176s cold start + ~3s per inference call`)
- (60–120 h, CONCLUSION:98, `The Phase 3 confirmatory experiment was estimated at 60-120 hours on CPU — infeasible`)
- (p=0.0001, CONCLUSION:64, `crossover interaction (advantage flips direction between known and unknown conditions, p=0.0001)`)
- (2.0 vs 10.0, p=0.0004, CONCLUSION:68, `PEDA's advantage in `/sandbox/projects` (2.0 steps vs 10.0, p=0.0004)`)
- (0% vs 48–80% dead-loop, CONCLUSION:114, `PEDA consistently showed zero dead-loop rate, versus Pragmatic's 48-80%`)
- (20/20, COUNTCHARTER §1, `**Success cache (memoization)** | 1-step solver | 20/20 1-step completions across 4 tasks`)
- (45→15 JEPA loss, CONCLUSION:90/124, `loss converges from 45 to 15 across training`)
- (3–10x, CONCLUSION:126, `the pragmatic term dominates by 3-10x`)
- (~2,000 episodes, CONCLUSION:11, `with ~2,000 total evaluation episodes`)
- (17+ experiments, CONCLUSION:11, `tested across 17+ controlled experiments spanning 5 environments`)
- (5.3x, COUNTCHARTER §3.1, `Sandbox v4 | 270 | 42% | 8% (JEPA hybrid) | 5.3x`)

---

## 3. Experimental Conditions (model, env, data_size, hardware)

- Declaration (CONCLUSION:138–143): `**Model:** Qwen2.5-0.5B-Instruct with LoRA (rank=16), JEPA MLP predictors (1-3 hidden layers), zero-shot RSSM`; `**Environment:** Busybox Linux sandbox (4-7 directories, 14-65 files), Grid Maze (1100-8400 states), Grid World (25 cells), TextWorld (2 rooms)`; `**Training:** 65-1378 transitions, 1-3 epochs LoRA, 500-2000 steps JEPA, CPU (Intel Core Ultra 9) or GPU (NVIDIA T4 16GB)`; `**Exploration signal:** EFE with ensemble variance, model-confidence proxy, JEPA hidden-state cosine distance, JEPA MLP prediction loss`; `**Baselines:** Pragmatic (goal-distance minimization), Random (uniform action selection), Count-based pair novelty, Heuristic (command templates)`; `**Tasks:** read_hello, count_lines, find_secret, read_note, read_welcome, find_api_key, count_measurements, find_errors_v4, read_changelog_v4`
- Planned Phase 1 (REPORT §4.2.1): 5×5 grid, 4 discrete actions; LLM input `{"state": {"agent": [x,y], "goal": [x,y]}, "action": "UP"}`
- Phase 1 actual (MANUSCRIPT:320): `Qwen2.5-0.5B-Instruct, fine-tuned on synthetic data (~1920 transitions from 20 configs × 24 free cells × 4 actions). 3 epochs, LoRA rank=16`
- Phase 1.5 actual (MANUSCRIPT:376–379): `Data | Exhaustive enumeration + random walks → 114 unique samples after dedup`; `Loss trajectory | 0.26 → 0.06 → 0.02 (3 epochs)`; `Checkpoints | 3 (for ensemble variance)`
- Planned WM model (REPORT §3.3.5): `预训练LLM（1-7B参数规模，如Qwen2.5-1.5B、Phi-3-mini或Llama-3.2-3B）+ LoRA微调`
- Count-charter scale table (COUNTCHARTER §3.1): v2 65 states count 50% vs best learned 50% (1.0x); v4 270 states 42% vs 8% (5.3x); Maze 10×10 deterministic 1100 states 100% vs 0%; Maze 10×10 stochastic 1100 states 100% vs 67% (1.5x); Maze 20×20 8400 states 0% / 0%

---

## 4. Methodology Descriptions (implementation details)

- **A. EFE selection loop** (REPORT:409–447, `peda_step`): predict current state → compute perceptual error → if `total > THRESHOLD`: generate ≤3 candidates → rollout horizon=2 → compute EFE each → select min → execute → compute model error → store (s, a, s', err); `elif drives.novelty > THRESHOLD` → exploratory action; else `return None`.
- **B. EFE computation** (REPORT §3.5.3): `epistemic += predicted_uncertainty * epistemic_ratio * (DISCOUNT ** i)`; `drive_adjusted_epistemic = epistemic * drives.curiosity_weight`; `return drive_adjusted_epistemic + pragmatic  # pragmatic = 0 in pure-exploration mode`.
- **C. Ensemble error decomposition** (REPORT §3.4.3 / MANUSCRIPT:231–249): ≥3 LoRA checkpoints predict same (state, action); `epistemic = ensemble_var`; `aleatoric = max(0, mean_deviation - ensemble_var)`; `total_error = mean_deviation + ensemble_var`; explicitly heuristic: `ensemble方差作为epistemic不确定性的代理是一种**启发式方法**，而非严格的数学分解`. Assumptions to validate in Phase 1: (1) checkpoints = different beliefs; (2) disagreement → epistemic; (3) agreement-but-wrong → aleatoric. Alternative statistical method (REFLECTION 方案B): run same command 10×, observation variance = aleatoric, model-vs-mean gap = epistemic.
- **D. Learning Module** (REPORT §3.6.2–3.6.4): buffer 500, update every 1000 steps, prioritized sampling by epistemic_error (batch 128), LoRA epochs=3, lr=2e-4, rank=16, save checkpoint per update, clear buffer. SaturationDetector: window 100, `decline_rate = (older - recent) / older`, saturated when `< 0.15` → raise Novelty drive. Distillation trigger: L1 > 0.9 AND L2 > 0.7 → merge LoRA per domain.
- **E. Drive System update** (REPORT §3.7.3): curiosity = `tanh(2.0 * epistemic_error)`; competence = flow-zone(success_rate, window 20); boredom = `max(0, 0.7 - action_entropy)`; novelty = `1 - exp(-0.01 * time_since_input)`; history deques: actions 50, errors 100. Selection: `π* = argmin_π [G(π) − Σ_d w_d·V_d(π)]` (REPORT:1194).
- **F. Graceful degradation** (REPORT §3.5.2): inference-budget-gated: full rollout → shortened horizon=2 → single-step greedy info-gain (`info_gain = conf * (1 - conf)`).
- **G. Safety** (REPORT §3.8 / §4.1 / §5.8): BLOCKED_PATTERNS regex (`rm -rf /`, `mkfs.`, `dd if=...of=/dev/`, fork bomb, `chmod -R 777 /`, `> /dev/sd*`); Docker `--read-only` + rw `/tmp`, `--memory=512m`, `--cpus=2`, `--pids-limit=64`, `--cap-drop=ALL`, `--network=none` (whitelist proxy in Phase 2b), non-root, tmpfs; prediction sanity rules (rm → file deleted; mkdir → dir created; `>` → content change); 30 s command timeout; hallucination monitoring thresholds: rate >10% → retrain, CRITICAL >1% → pause, retry success <50% → model too weak (REPORT §5.8.4).
- **H. Emergence protocol** (REPORT §4.4.2 + §6.3.2): 7 metrics; gate = autonomous focus >10 steps; then ≥3 of remaining 6 above threshold → `涌现迹象`; ≥3 independent runs → `可复现的涌现行为`.
- **I. Phase 1 hyperparameter search plan** (REPORT §4.2.3): grid `{0.1, 0.5, 1.0, 2.0}^4` = 256 combos × 10 episodes, Pareto front. (v1.1 §3.7.5 plan: `{0.2, 0.5, 0.8}` = 81 combos — not completed.)
- **J. Phase 2 data loop** (REPORT §4.4.4): 1000-step rounds; collect (s,a,s',r); 1 gradient update per round; human spot-check 10 traces/500 steps; hyperparameter tuning if 3 consecutive rounds without improvement.

---

## 5. Verbatim Conclusions (source docs)

- CHARTER:21: `三个子问题层层递进。如果任何一个子问题的答案是"否"，整个假设在此条件下不成立——**但这本身就是一个有价值的研究结论**`
- CHARTER:37: `**关键原则**: 负结果不是项目失败，而是知识。一个诚实记录的负结果比一个人为制造的"成功"更有科学价值。`
- CHARTER:79–81: `> **我们是否对"Active Inference 在 LLM-based Agent 中的可行性"有了比项目开始前更深的理解？**` / `如果这个问题的答案是"是"——无论最终 Agent 是否自主、无论是否按期完成——PEDA 就是成功的。`
- CONCLUSION:19: `Count-based pair novelty, not epistemic prediction error from learned World Models, is the reliable exploration mechanism in state spaces under ~1,000 states. This negative result is a valid scientific conclusion per the research charter`
- CONCLUSION:136: `The PEDA hypothesis — that prediction error from an LLM-based World Model can drive autonomous exploration more effectively than baselines in LLM-based agents — is **DISPROVEN** under the conditions tested:`
- CONCLUSION:66: `**This does not validate prediction-error-driven exploration.**` (E08, three non-epistemic factors)
- CONCLUSION:130: `The Cold Start problem (no model without data, no exploration without model) is not solvable by better exploration algorithms: without a minimal set of diverse (state, action, next_state) transitions, no learned model can predict anything useful`
- CONCLUSION:126: `EFE is dominated by pragmatic value at any practically testable horizon... This is not a fixable hyperparameter issue; it is a structural property of EFE in goal-directed tasks with small lookahead horizons.`
- MANUSCRIPT:28: `**Core findings**: (1) LLM-based World Models can produce measurable prediction error signals, but environment complexity must match model capacity... (2) The Homeostatic Drive System (curiosity, competence, boredom, novelty) has independent behavioral value even when epistemic prediction error is near zero... (3) Information gain (via ensembl[e]...)` (superseded doc — use only as supporting narrative)
- COUNTCHARTER §2: `**How far can a purely count-driven agent — using pair-novelty exploration, learned STRIPS schemas, and success memoization — go before it needs a learnable forward model?**`
- COUNTCHARTER §6: `1. **No prediction error. No World Model. Period.** If it looks like a forward model, it belongs in a different project.`
- REFLECTION (final judgment): `评审的5.5/10评分是公允的。PEDA v1.0确实存在理论与实践脱节、关键假设未经检验、遗漏重要相关工作等问题。`; top-3 insights: (1) "70%准确率可能是完全不现实的" (2) "推理速度可能被严重低估" (3) "遗漏Voyager"

---

## 6. Requested Extractions

### 6.1 FEP/EFE equations and derivations
- Variational free energy: MANUSCRIPT:76 `$$F = \underbrace{-\ln p(o)}_{\text{surprise}} + \underbrace{D_{KL}[q(s) \| p(s|o)]}_{\text{approximation error}}$$`; MANUSCRIPT:78 `F ≥ −ln p(o): free energy is an upper bound on surprise`
- EFE compact: REPORT:137 `$$G(\pi) = \mathbb{E}_{q(o|\pi)}[\ln q(o|\pi) - \ln p(o|C)] = H[q(o|\pi)] + D_{KL}[q(o|\pi) \,||\, C(o)]$$`
- EFE annotated: REPORT:218 `$$G(\pi) = \underbrace{H[q(o|\pi)]}_{\text{epistemic value（探索）}} + \underbrace{D_{KL}[q(o|\pi) \,||\, C(o)]}_{\text{pragmatic value（利用）}}$$`; MANUSCRIPT:88 + 92 (full epistemic/pragmatic form; `A more compact formulation (from Friston et al., 2017)`)
- EFE full form: REPORT:856 `$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[D_{KL}[q(s|o,\pi) \| q(s|\pi)]]}_{\text{Epistemic Value（认知价值）}} + \underbrace{D_{KL}[q(o|\pi) \| p(o|C)]}_{\text{Pragmatic Value（实用价值）}}$$`
- Drive-adjusted selection: REPORT:1194 / MANUSCRIPT:291 `$$\pi^* = \arg\min_{\pi} \left[ G(\pi) - \sum_{d \in \text{Drives}} w_d \cdot V_d(\pi) \right]$$`
- Key correction (v1.0→v1.1): REPORT §2.2 `v1.0曾表述为"不需要外部目标"，这是一种过度简化。FEP并非消除目标，而是**将目标的形式从外部reward函数转变为内部偏好分布C(o)**`; even uniform C(o) is a goal ("平等对待所有观测")
- Noisy TV resolution: REPORT:222 `PEDA中的驱动信号不是原始预测误差，而是**能够带来信息增益的预测误差**` (Noisy TV: high prediction error forever, zero info gain once the TV distribution is learned)
- Citations: Friston et al. 2006/2010/2017/2023; Rao & Ballard 1999; Clark 2013/2015; Sajid et al. 2021 (AIF ≈ model-based RL equivalence); Hafner et al. 2019–2023 (RSSM/Dreamer); Mazzaglia et al. 2022 (Probabilistic Dreaming, +4.5% continuous control); Pathak 2017 (ICM), Burda 2018 (RND); Guo 2022 (BYOL-Explore); LeCun 2022+ (JEPA), Assran 2023 (I-JEPA), Bardes 2024 (V-JEPA), Bhardwaj 2025 (V-JEPA 2); Wang 2023 (Voyager); Yao 2023 (ReAct); Shinn 2023 (Reflexion); Millidge 2022 (PC ≈ BP).

### 6.2 Seven-module architecture with pseudocode
- Module table: REPORT:392–402 (Perception, World Model, Predictive Error Computer, Action Generator, Action Executor, Learning Module, Homeostatic Drive System; update frequencies 每步/间歇微调/每步/每步/每步/每N步/每步)
- Pseudocode: REPORT:409 `def peda_step(current_state: State, world_model: WM, drives: Drives) -> Action:` (full loop, lines 409–447); DriveWeights + HomeostaticDriveSystem.update (REPORT:1120–1165); compute_efe/select_action (REPORT §3.5.3); EnsembleErrorComputer (REPORT:685–760); rollout (REPORT §3.3.4, self-bootstrapping, `action = None` after first step)
- Discrepancy: REPORT:346–348 claims `五大核心模块` (five) while the table (392–402), MANUSCRIPT:164 (`seven interacting modules`), and AGENTS.md (`closed loop of seven modules`) say seven; REPORT §3.9 enumerates six roles while calling them 五大模块. **Use 7** (majority + authoritative AGENTS.md/MANUSCRIPT).

### 6.3 Three-level prediction (L1/L2/L3) targets and rationale
- Rationale: REPORT §3.3.2 / MANUSCRIPT:213 — `0.8^10 ≈ 10.7%` joint-accuracy ceiling; `分层允许系统在不同层次上独立学习和改进`
- L1 exit code: ≥90% (REPORT:503), eval = classification accuracy, stop-loss <80% (REPORT:505)
- L2 filesystem delta: ≥70% existence / ≥60% directory-structure (REPORT:513), eval = structured diff, stop-loss <50% (REPORT:515)
- L3 output summary: ≥50% semantic match (REPORT:523), eval = SBERT cosine >0.7, no stop-loss (REPORT:525)
- Aleatoric (explicitly NOT predicted): timestamps, PIDs, RNG outputs, exact network latency, exact memory usage — tagged `ALEATORIC`, excluded from accuracy (REPORT:527–533)
- Interfaces: `State` / `PredictedState` / `Action` dataclasses (REPORT §3.3.4); training-data JSON format (REPORT §3.3.5)

### 6.4 Four immutable principles (canonical: AGENTS.md:56–59 — grep-verified ABSENT from all PEDA_FINAL files)
1. `**No Prompt, only Prediction Error.** Never add features that require user input to trigger behavior.`
2. `**Drive is emergent, not hardcoded.** Never write fixed goal lists or fixed drive weights.`
3. `**World Model is the core.** Spend ~80% of effort on the World Model; any new module must directly improve its predictions.`
4. `**Learning is intermittent, not continuous.** Collect data, then batch-update. Never do per-step online SGD.`

### 6.5 Ensemble uncertainty method
- REPORT §3.4.3: `EnsembleErrorComputer(num_checkpoints=5)`; keep last 5 checkpoints; per (state, action): `epistemic = ensemble_var`; `aleatoric = max(0, mean_deviation - ensemble_var)`; heuristic declared with 3 assumptions to validate in Phase 1. MANUSCRIPT:231–249 identical (with epistemic/aleatoric intuition examples: `[0,1,0,0,1]` exit-code disagreement → epistemic; ping 50 ms vs 52 ms agreement → aleatoric).
- Alternative statistical decomposition (REFLECTION 方案B): repeated observation variance = aleatoric; model-vs-mean gap = epistemic.
- Replaced v1.0 heuristic: `epistemic_ratio = (1 - conf) * 0.3 + conf * 0.7` (REVIEW Risk 3; REFLECTION 问题三).

### 6.6 Drive System (4 drives, weights)
- REPORT:1076–1106: Curiosity (high-epistemic regions; `tanh(α × epistemic_error)`); Competence (success history; `optimal_challenge_zone(success_rate)`; flow state); Boredom (low action entropy; `1 - normalize_entropy(recent_actions)`; structured diversity, not noise); Novelty (time since external input; `exp(-λ × time_since_last_input)`; requires open environment)
- Initial weights (REPORT:1132–1135): curiosity=0.5, competence=0.5, boredom=0.3, novelty=0.4 — `经验设定，非最优`
- Dynamic-balance table (REPORT §3.7.6): 新环境初期 / 学习中 / 掌握环境后 / 长期无外部输入 weight regimes
- Epistemic Foraging (REPORT §3.7.4): Epistemic→Curiosity+Novelty; Pragmatic→Competence; homeostasis→Boredom; selection formula REPORT:1194
- Hyperparameter sensitivity (REPORT:1198): `都是**经验设定**，没有任何理论保证它们是最优的`; risks (curiosity↑ local trap; boredom↑ erratic; competence↑ premature convergence; novelty↑ external chasing); search: grid {0.2,0.5,0.8} 81 combos or random search
- INCONSISTENT alternate formalism: REPORT §5.4.1 defines Novelty = `−log P(s_{t+1}|s_t,a_t)`, Boredom = repeat-frequency `1/τ Σ 1[s_i=s_t]`, Growth = `|FactGraph_t| − |FactGraph_{t−1}|`, competence = success-rate ratio; §4.2.3 also uses Growth. Two drive formalisms in one document — cite §3.7.2 (Curiosity/Competence/Boredom/Novelty) as canonical.

### 6.7 Cold-start problem definition
- CHARTER:34 (negative-result row): `| 冷启动无法解决 | 初始数据质量太低导致学习循环断裂 | 说明 bootstrap 策略或模型先验知识不足 |`
- CONCLUSION:130: `The Cold Start problem (no model without data, no exploration without model) is not solvable by better exploration algorithms... The project spent approximately 50% of total engineering effort on data collection and pipeline infrastructure`
- MANUSCRIPT:62: `**Cold start problem**: The World Model needs training data to make accurate predictions, but the Action Generator requires an accurate World Model to generate useful training data (bootstrap circularity, identified by third-party review)`

### 6.8 Review score 5.5/10 and critiques
- REVIEW:373 score; executive summary 3 root problems: `理论与实践的严重脱节` / `核心工程假设缺乏实证支撑` / `对FEP的工程可行性存在过度解读`
- 4 major risks (REVIEW §1.2): Risk 1 WM accuracy (single point of failure, `70%的预测准确率目标缺乏任何数据支撑`); Risk 2 rollout divergence (`0.7^10 ≈ 2.8%`); Risk 3 epistemic/aleatoric (`LLM的softmax置信度≠环境固有随机性`); Risk 4 Grid World→Linux `质的差异` (`Phase 1的成功几乎不能为Phase 2提供任何信心`)
- Feasibility scores (REVIEW §1.3): Docker 95%, Grid World 95%, LoRA WM 90%, Drive System 85%, Linux WM >70% 40%, Rollout EFE 30%, epistemic/aleatoric 35%, 48 h autonomous 20%
- Theory critiques (REVIEW §2.2): "不需要外部目标" misleading (`C(o)就是偏好分布，它**就是目标**`); Predictive Coding decorative (`PC的局部学习规则在PEDA的代码中**完全没有出现**`); §2.6 continuous-time `装饰性内容`; Noisy-TV immunity theory≠implementation (`理论正确不等于实现正确`)
- Missed work (REVIEW §3.1): Voyager, BYOL-Explore, JEPA, SOAR/ACT-R/LIDA, ReAct/Reflexion/AutoGPT; scores: 文献覆盖度 6/10, 区分度 5/10, 公平评估 4/10
- Engineering (REVIEW §4): timeline realism 5/10 (`偏乐观约2倍`), resources 6/10, maintainability 7/10, scalability 4/10
- 7 blind spots (REVIEW §5): LLM hallucination; circular "有趣" metric; safety ~zero; FEP unfalsifiability; emergence-vs-random; bounded openness; Drive hyperparameter sensitivity
- Dimension scores (REVIEW §6.4): technical 5, theory 6, related-work 5, engineering 5, innovation 5, doc quality 7
- REFLECTION: accepted score as fair; 9 issues addressed (incl. new: inference speed, LLM hallucination, safety)

### 6.9 ALL defined metrics and thresholds (inventory)
- Charter: Q1 (epistemic error > 0), Q2 (EFE effective → different action selection), Q3 (PEDA > baseline); 5 accepted negative results (CHARTER:30–35)
- Phase 1 G1/G2/G3 (REPORT §4.2.2): G1 next-state accuracy >90%; G2 steps <50% of random; G3 revisit rate <20%
- Phase 1.5 G4–G7 (REPORT §4.3.3): G4 ROUGE-L >60% + key-fact extraction >60%; G5 ≥3-step task completion >30% (random <5%); G6 FactGraph extraction >70%; G7 normalized action entropy >0.5 (10 runs)
- Phase 2 weekly L1–L8 (REPORT §4.4.1): command success >80%; FactGraph nodes >20; 4-step sequence >30%; accuracy monotonic; seen-command accuracy >60%; coverage >50%; behavior entropy >0.5 + ≥3 Pareto modes; browser whitelist task
- Emergence 5-metric set (REPORT §4.4.2): exploration efficiency >0.1 new dirs/step; entropy >0.5 (window 50); knowledge growth slope >0.5 nodes/100 steps; accuracy-trend Pearson r >0.5; autonomous focus >10 consecutive steps; 3-of-5 → 涌现迹象
- Unified 7-metric table (REPORT §6.3.1): WM accuracy >60% (TextWorld & Linux); exploration efficiency >0.1; entropy >0.5; growth >0.5/100 steps; multi-step >30% (4-step); Pearson r >0.5; focus >10 steps
- Emergence protocol (REPORT §6.3.2): focus-gate >10 steps; ≥3 of 6; ≥3 runs for reproducibility
- Deleted subjective metrics (REPORT §6.3.3): "行为有趣" (κ < 0.4), "创造力", "看起来有目标导向"
- Milestones M1–M6 (REPORT §6.1.1): M1 w3 (WM >90%, steps <50% random); M2 w5; M3 w9 (facts >60%, 3-step >30%); M4 w17 (FactGraph >20); M5 w29 (3/5 thresholds); M6 w37 (browser task)
- Inference budgets (REPORT §3.5.2): 2–3 candidates × 2–3 horizon = 4–9 calls/step, 8–18 s/step
- Safety thresholds (REPORT §3.8/§4.1/§5.8.4): regex blacklist; `--memory=512m --cpus=2 --pids-limit=64 --cap-drop=ALL --network=none`; 30 s timeout; hallucination >10% retrain / CRITICAL >1% pause / retry <50% weak
- Saturation (REPORT §3.6.3): decline <0.15 → saturated; distillation L1>0.9 AND L2>0.7
- Count-charter success criteria (COUNTCHARTER §5): counting-limit curve documented; STRIPS chaining >50% (2-step) / >30% (3-step); transfer >10pp; drive system satiation > fixed ratio > random; 20-episode ablation minimum (COUNTCHARTER §6.3)

---

## 7. Contradictions / Gaps

1. **Module count**: REPORT §3.2.1 "五大核心模块" (5) vs own table (7, REPORT:392–402) vs MANUSCRIPT:164 "seven interacting modules" vs AGENTS.md 7-module loop. Report §3.9 says 五大模块 while enumerating 6. Paper: use 7.
2. **Drive formalism**: REPORT §3.7.2 (Curiosity/Competence/Boredom/Novelty) vs §5.4.1 + §4.2.3 (Novelty/Boredom/Competence/Growth with different math). Canonical: §3.7.2.
3. **E05 target vs achieved**: REPORT targets 90/70/50; AGENTS.md "formal targets met" = 1.000/0.900/0.550 (achieved in-distribution values). State targets and achieved values separately.
4. **E04 epistemic magnitude**: CONCLUSION says "epistemic ~0"; MANUSCRIPT:534/644 says measurable 0.20 (after bug fix) but insufficient. Quote both with context.
5. **Inference math**: unmitigated 50–100 calls → ~520 steps/48h (REVIEW/REPORT §3.5.2) vs mitigated 4–9 calls → 9600–21600 steps/48h (REPORT §3.5.2). Cite as pair.
6. **Review date typo**: REVIEW header "评审日期: 2025年7月" vs scope "(v1.0, 2026年7月2日)". Use 2026.
7. **Gap**: cold-start defined in 3 docs but no dedicated methodology section; "~50% effort on data collection" (CONCLUSION:130) has no quantitative breakdown.
8. **Gap**: 4 Immutable Principles exist only in AGENTS.md:56–59 (grep over PEDA_FINAL: zero hits) — cite AGENTS.md or treat as project doctrine.
9. **E08/E09 p-values**: same p=0.0043 reported for positive (unknown CWDs, d=−1.01) and negative-control (known CWDs) comparisons (CONCLUSION:48–49). Verify per-test attribution before citing.
10. **Manuscript scope**: superseded — Sections 2–3 only are usable for theory/architecture; its results sections (4+) conflict with canonical E-table in places and must not be cited as evidence.

---

*Evidence gathered 2026-07-31 by scout S1Theory. All quotes verbatim from files listed in §1; line numbers verified via grep.*
