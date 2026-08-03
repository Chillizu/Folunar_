# Prediction-Error-Driven Exploration in LLM Agents: A Negative Result

## Abstract

We investigate whether prediction error from an LLM-based World Model can serve as an intrinsic drive signal for autonomous exploration, outperforming purely goal-directed behavior. This question decomposes into three sub-questions: (1) can an LLM-based World Model produce measurable epistemic uncertainty? (2) can Expected Free Energy drive action selection toward exploration? (3) does prediction-error-driven exploration beat baselines? Across 19 controlled experiments spanning 5 environments (Grid World, TextWorld, Busybox Linux Sandbox v1-v4, Grid Maze 5x5-20x20), 4 model architectures (Qwen2.5-0.5B with LoRA, JEPA MLP, RSSM, STRIPS), and ~2,000 evaluation episodes, all three questions answer **no** under tested conditions. The World Model produces near-zero or uniform epistemic uncertainty (DLR 0.8-0.9, ensemble variance approximately zero). Expected Free Energy is dominated by pragmatic value at horizons 1-3. Prediction-error-driven agents never beat count-based baselines. The one statistically significant result (Phase 3, p=0.0043, d=-1.01) is attributable to candidate-set engineering and success caching, not epistemic signal. A count-driven agent using pair-novelty, learned STRIPS schemas, and success memoization achieves 62.2% across 9 sandbox tasks with zero contribution from JEPA forward dynamics. We conclude that count-based novelty, not epistemic prediction error from learned World Models, is the reliable exploration mechanism in state spaces under ~1,000 states. This negative result constitutes a valid scientific conclusion: Active Inference with LLM-based agents does not produce behaviorally distinguishable exploration under practical conditions.

---
# Prediction-Error-Driven Exploration in LLM Agents: A Negative Result

**Sections 1-3 — Introduction, Theoretical Framework, Architecture** (draft S1)

**Citation legend** (all paths under `PEDA_FINAL/`; every quantitative claim carries `(file:line, verbatim quote)`):
- `MANUSCRIPT` = `PEDA_RESEARCH_MANUSCRIPT.md` (old draft — only §2 Theory and §3 Architecture are authoritative, per STRUCTURE.md)
- `CHARTER` = `RESEARCH_CHARTER.md`
- `CONCLUSION` = `PEDA_CONCLUSION.md` (carries an ERRATA banner; cited here only for claims not listed in that errata and consistent with the canonical `CLAIMS_VS_EVIDENCE.md`)
- `CVE` = `paper/CLAIMS_VS_EVIDENCE.md` (canonical claim cross-reference)
- `THEORY` = `paper/evidence/theory.md` (evidence bundle E01-E19; quotes underlying sources with verified line numbers)
- `REPORT` = `peda_report_v11.agent.final.md` (v1.1 architecture report, quoted via THEORY §6.1)

---

# 1. Introduction

## 1.1 Motivation and research question

Mainstream LLM agents are reactive systems: they idle until an external trigger — a prompt, an API call, a timer — supplies a goal and a task decomposition. PEDA (Predictive-Error-Driven Autonomous Agent) investigates the alternative: an agent that continuously predicts the consequences of its own actions through a learned World Model, and treats the mismatch between prediction and observation — prediction error — as the intrinsic signal that selects and drives behavior.

The research charter fixes the exact question under test:

> 当 Agent 的 World Model 不完美时，预测误差（prediction error）是否能作为一种内在驱动信号，引导 Agent 主动探索不确定性区域，从而比纯目标导向的行为更有效？
> (CHARTER:13, "当 Agent 的 World Model 不完美时，预测误差（prediction error）是否能作为一种内在驱动信号，引导 Agent 主动探索不确定性区域，从而比纯目标导向的行为更有效？")

"When an agent's World Model is imperfect, can prediction error serve as an intrinsic drive signal that guides the agent to actively explore regions of uncertainty, more effectively than purely goal-directed behavior?" The charter classifies the project as research exploration with no product, schedule, or performance-bar targets (CHARTER:3-5, "**项目定位**: 研究探索 / **核心问题**: Active Inference 的预测误差驱动在 LLM-based Agent 中是否可行？ / **非目标**: 商业产品交付、时间表约束、性能指标达标").

## 1.2 Three sub-questions

The core question decomposes into three ordered sub-questions (CHARTER:15, "这个问题分解为三个子问题："), each stated verbatim in the charter:

1. **Signal (Q1)** — (CHARTER:17, "1. **信号问题**: LLM-based World Model 能否产生可测量的、与预测质量相关的预测误差信号？（epistemic error > 0）"): can an LLM-based World Model produce measurable prediction error that is related to prediction quality, i.e., epistemic error > 0?
2. **Drive (Q2)** — (CHARTER:18, "2. **驱动问题**: 这个预测误差信号能否驱动 Action Generator 选择探索性行为？（EFE 有效）"): can that error signal drive the Action Generator to select exploratory behavior, i.e., is Expected Free Energy (EFE) effective?
3. **Effect (Q3)** — (CHARTER:19, "3. **效果问题**: 由预测误差驱动的探索是否比基线（纯 pragmatic、纯随机）更有效？（PEDA > baseline）"): is prediction-error-driven exploration more effective than baselines (purely pragmatic, purely random)?

The three sub-questions form a dependency chain: a negative answer at any level invalidates the hypothesis under those conditions, and that negative answer is itself an accepted research conclusion (CHARTER:21, "三个子问题层层递进。如果任何一个子问题的答案是"否"，整个假设在此条件下不成立——**但这本身就是一个有价值的研究结论**"). All three questions are accepted as falsifiable, and the answers reported in §5 are conditional on the specific conditions tested — they do not claim impossibility in general (CONCLUSION:44, "All three questions are accepted as falsifiable. A negative answer at any level is an accepted, valuable conclusion per the research charter. The answers below are conditional on the specific experimental conditions tested — they do not claim impossibility in general, only that under the conditions explored, the hypothesis does not hold."). The charter's governing principle is that negative results are knowledge, not failure (CHARTER:37, "**关键原则**: 负结果不是项目失败，而是知识。一个诚实记录的负结果比一个人为制造的"成功"更有科学价值。"). The project's success criterion is therefore a deepened understanding of Active Inference feasibility, not a working agent (CHARTER:79-81, "> **我们是否对"Active Inference 在 LLM-based Agent 中的可行性"有了比项目开始前更深的理解？**").

## 1.3 The cold-start problem

The bootstrap circularity is one of the negative results the charter explicitly accepts as a valid outcome (CHARTER:34, "| 冷启动无法解决 | 初始数据质量太低导致学习循环断裂 | 说明 bootstrap 策略或模型先验知识不足 |"): the World Model needs training data to make accurate predictions, while the Action Generator needs an accurate World Model to generate useful training data (MANUSCRIPT:62, "**Cold start problem**: The World Model needs training data to make accurate predictions, but the Action Generator requires an accurate World Model to generate useful training data (bootstrap circularity, identified by third-party review)"). Because exploration quality is bounded by prediction quality and prediction quality is bounded by data quality, the cold start determines whether the learning loop can ever close (CONCLUSION:141, "The Cold Start problem (no model without data, no exploration without model) is not solvable by better exploration algorithms: without a minimal set of diverse (state, action, next_state) transitions, no learned model can predict anything useful, and without useful predictions, no exploration signal can be generated. The project spent approximately 50% of total engineering effort on data collection and pipeline infrastructure"). The experiments in §5 treat cold start as a primary explanatory factor, not an engineering nuisance.

## 1.4 Scope and outcome preview

The hypothesis was tested across 19 controlled experiments (E01-E19) spanning 5 environments (Grid World, TextWorld, Busybox Linux Sandbox v1-v4, Grid Maze, Giant Maze), using 4 model architectures (Qwen2.5-0.5B with LoRA, JEPA MLP predictors, RSSM, STRIPS action models) and ~2,000 total evaluation episodes. All three charter sub-questions answer **No** under the tested conditions.

The remainder of the paper: §2 states the theoretical framework (Free Energy Principle, Expected Free Energy, epistemic/aleatoric decomposition) that motivates the design; §3 describes the seven-module architecture and its three-level prediction hierarchy; §4 documents environments, models, and metrics; §5 reports results per phase against the claim cross-reference (CVE); §6 analyzes root causes; §7 concludes.

---

# 2. Theoretical Framework

## 2.1 The Free Energy Principle (FEP)

FEP proposes that self-organizing systems maintain their organization by minimizing variational free energy, given a generative model p(o, s) over observations o and hidden states s with variational posterior q(s) (MANUSCRIPT:74, "The Free Energy Principle [Friston, 2009, 2010; Friston et al., 2006] proposes that all self-organizing systems—from single cells to complex brains—maintain their organization by minimizing variational free energy. Formally, given a generative model $p(o, s)$ over observations $o$ and hidden states $s$, with variational posterior $q(s)$, the variational free energy is:"). The variational free energy is:

$$F = \underbrace{-\ln p(o)}_{\text{surprise}} + \underbrace{D_{KL}[q(s) \| p(s|o)]}_{\text{approximation error}}$$ (MANUSCRIPT:76, verbatim)

Because the KL divergence is non-negative, F is an upper bound on surprise, F ≥ −ln p(o); minimizing F simultaneously improves the agent's model of the world and reduces unexpected sensory input (MANUSCRIPT:78, "Since the KL divergence is non-negative, $F \geq -\ln p(o)$: free energy is an upper bound on surprise. Minimizing $F$ simultaneously improves the agent's model of the world (perception) and reduces unexpected sensory input (action)."). In this framework perception and action are two sides of the same coin — perception updates beliefs to predict sensory input, action changes sensory input to match predictions — so perception, action, and learning share one objective (MANUSCRIPT:80, "perception updates internal beliefs to better predict sensory input, while action changes sensory input to better match internal predictions. This unified view eliminates the need for separate objective functions for perception, action, and learning—they all serve the same imperative of free energy minimization [Friston et al., 2017]").

## 2.2 Active Inference and Expected Free Energy

Active Inference extends FEP from perception to action selection: the agent infers which policy π (sequence of actions) will minimize expected future free energy, rather than solving action as a separate optimization problem as in reinforcement learning (MANUSCRIPT:84, "Active Inference frames action as an inference process: the agent infers which policy $\pi$ (sequence of actions) will minimize expected future free energy [Friston et al., 2017; Parr et al., 2022]"). The Expected Free Energy for policy π is:

$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[D_{KL}[q(s|o,\pi) \| q(s|\pi)]]}_{\text{Epistemic Value (exploration)}} + \underbrace{D_{KL}[q(o|\pi) \| p(o|C)]}_{\text{Pragmatic Value (exploitation)}}$$ (MANUSCRIPT:88, verbatim)

with the compact form (MANUSCRIPT:90, "A more compact formulation (from Friston et al., 2017):"):

$$G(\pi) = \underbrace{H[q(o|\pi)]}_{\text{epistemic value}} + \underbrace{D_{KL}[q(o|\pi) \,||\, C(o)]}_{\text{pragmatic value}}$$ (MANUSCRIPT:92, verbatim)

The two terms are the two behavioral drives: epistemic value H[q(o|π)] is the entropy of predicted observations under π — high entropy means the agent stands to gain information, and this term drives exploration (MANUSCRIPT:95, "**Epistemic value** $H[q(o|\pi)]$ is the entropy of the predicted observations under policy $\pi$. High entropy means the agent is uncertain about what it will observe—and thus stands to gain information. This term drives **exploration**."). Pragmatic value is the KL divergence between predicted observations and the agent's preferred observations C(o), driving goal-directed behavior (MANUSCRIPT:96, "**Pragmatic value** $D_{KL}[q(o|\pi) \| C(o)]$ is the KL divergence between predicted observations and the agent's preferred observations $C(o)$. This term drives **goal-directed behavior** (exploitation)."). The formulation resolves the exploration-exploitation trade-off inside the mathematics: no manual epsilon-greedy schedule is needed (MANUSCRIPT:98, "The EFE formulation resolves the exploration-exploitation trade-off naturally: policies that reduce the agent's uncertainty (high epistemic value) are preferred in uncertain regions; policies that achieve preferred outcomes (low pragmatic divergence) are preferred in familiar regions. **No manual epsilon-greedy schedule is needed**—the balance emerges from the mathematics."). One v1.0 formulation error was corrected before implementation: FEP does not eliminate goals — it converts the external reward function into an internal preference distribution C(o), and even a uniform C(o) is a goal (REPORT §2.2, quoted verbatim in THEORY §6.1, "v1.0曾表述为"不需要外部目标"，这是一种过度简化。FEP并非消除目标，而是**将目标的形式从外部reward函数转变为内部偏好分布C(o)**").

## 2.3 Epistemic versus aleatoric uncertainty

The distinction between reducible and irreducible uncertainty underlies EFE's claim to superiority over raw prediction error as an exploration signal (MANUSCRIPT:102, "The critical distinction between epistemic and aleatoric uncertainty underlies the EFE's superiority over raw prediction error as an exploration signal:"). **Epistemic uncertainty** is reducible by gathering more data; in ensemble terms it corresponds to variance across model checkpoints, where different training snapshots disagree because they lack evidence in that region of state space (MANUSCRIPT:104, "**Epistemic uncertainty** (reducible): Uncertainty that can be reduced by gathering more data. It reflects what the agent *does not know but could know*. In terms of ensemble methods, it corresponds to variance across model checkpoints—different training snapshots disagree on predictions because they lack sufficient evidence in that region of state space."). **Aleatoric uncertainty** is irreducible: stochastic transitions and random noise persist even with infinite data, such as the precise output of `date` or `shuf` (MANUSCRIPT:106, "**Aleatoric uncertainty** (irreducible): Uncertainty inherent to the environment—stochastic transitions, random noise, intrinsic variability. Even with infinite data, this uncertainty persists. The LLM's inability to predict the precise output of `date` or `shuf` is aleatoric.").

Purely prediction-error-driven methods such as ICM and RND conflate the two types; their canonical failure mode is the Noisy TV Problem, in which an agent becomes trapped by perpetual prediction error from unpredictable random noise (MANUSCRIPT:108, "Purely prediction-error-driven methods (ICM [Pathak et al., 2017], RND [Burda et al., 2018]) conflate these two types. The **Noisy TV Problem**—where an agent becomes trapped watching unpredictable random noise because it generates perpetual prediction error—is the canonical failure mode [Pathak et al., 2017; Burda et al., 2018]."). Active Inference avoids the trap through the information-theoretic structure of EFE: **information gain** — the magnitude of belief update — differs fundamentally from prediction error — the magnitude of observation mismatch; a Noisy TV produces high prediction error forever but zero information gain once its randomness is learned (MANUSCRIPT:110, "Active Inference avoids the Noisy TV trap through the information-theoretic structure of EFE: **information gain** (belief update magnitude) differs fundamentally from prediction error (observation mismatch). A Noisy TV produces high prediction error forever, but zero information gain once the agent has learned that the TV output is uniformly random. The epistemic value term $H[q(o|\pi)]$ selects actions that actually change the agent's beliefs, not merely actions that generate surprising observations."). PEDA's design inherits this commitment: the exploration signal is EFE's epistemic term, not raw prediction error.

## 2.4 World models: learning in imagination

The World Model paradigm provides the architectural substrate for EFE-based planning: in the Dreamer series, agents learn a latent dynamics model (RSSM — Recurrent State-Space Model) and train policies entirely within imagined trajectories generated by this model (MANUSCRIPT:114, "The World Model paradigm [Ha & Schmidhuber, 2018] provides the architectural substrate for EFE-based planning. In the Dreamer series [Hafner et al., 2020, 2021, 2023; Hafner et al., 2025], agents learn a latent dynamics model (RSSM—Recurrent State-Space Model) and then train policies entirely within "imagined" trajectories generated by this model."). The lineage's key results: World Models (2018) scored +103.8 versus +4.84 for random on CarRacing (MANUSCRIPT:118, "| World Models (2018) | CarRacing | VAE+RNN+Controller: +103.8 vs random +4.84 | [Ha & Schmidhuber, 2018] |"); DreamerV1 achieved SOTA data efficiency on 20 visual control tasks (MANUSCRIPT:119, "| DreamerV1 (2020) | 20 visual control tasks | SOTA data efficiency | [Hafner et al., 2020] |"); DreamerV2 was the first world model to reach human level on 55 Atari games (MANUSCRIPT:120, "| DreamerV2 (2021) | 55 Atari games | First world model to reach human level | [Hafner et al., 2021] |"); DreamerV3 used a single hyperparameter set across 150+ tasks (MANUSCRIPT:121, "| DreamerV3 (2023/2025) | 150+ tasks | Single hyperparameter set, Minecraft diamond | [Hafner et al., 2023] |"); DreamerV4 reached 0.7% offline Minecraft diamond crafting versus a baseline below 0.1% (MANUSCRIPT:122, "| DreamerV4 (2025) | Minecraft | Offline diamond: 0.7% vs baseline <0.1% | [Hafner et al., 2025] |").

PEDA inherits the lineage's core insight — latent state prediction is more efficient than raw observation prediction — but replaces the RSSM architecture (deterministic GRU + stochastic categorical latent) with an LLM + LoRA, trading architectural specialization for the LLM's broad world knowledge (MANUSCRIPT:124, "PEDA inherits the Dreamer lineage's core insight: **latent state prediction is more efficient than raw observation prediction**. However, PEDA replaces the RSSM architecture (deterministic GRU + stochastic categorical latent) with an LLM + LoRA, trading architectural specialization for the LLM's broad world knowledge.").

## 2.5 From theory to implementation

Three operational commitments bridge the framework to the architecture of §3. First, epistemic error must be measurable from an LLM-based World Model, which requires an ensemble uncertainty estimate rather than raw model confidence (MANUSCRIPT:63, "**Epistemic/aleatoric decomposition**: LLM confidence scores are poorly calibrated [Guo et al., 2017]; ensemble variance is only a heuristic proxy for model uncertainty."). Second, prediction must be decomposed into epistemic and aleatoric components so that the exploration signal tracks information gain, not noise (MANUSCRIPT:110, quoted in §2.3). Third, the EFE objective is modulated by a homeostatic drive system, so that the action selected is the one minimizing EFE minus the weighted value of the four drives (MANUSCRIPT:291, "$$\pi^* = \arg\min_{\pi} \left[ G(\pi) - \sum_{d \in \text{Drives}} w_d \cdot V_d(\pi) \right]$$"). The architecture implements exactly this loop.

---

# 3. Architecture

## 3.1 System overview

PEDA's cognitive architecture consists of seven interacting modules organized around the prediction-error-driven control loop (MANUSCRIPT:164, "PEDA's cognitive architecture consists of seven interacting modules organized around the prediction-error-driven control loop."). The loop (MANUSCRIPT:169-175, verbatim):

```
[Perception] → [World Model (LLM+LoRA)] → [Predictive Error Computer]
                                              ↓
[Environment] ← [Action Executor] ← [Action Generator (EFE-driven)]
                      ↑
              [Learning Module] ← 间歇性批量 LoRA 微调
                      ↑
         [Homeostatic Drive System] ← 四 Drive 动态平衡
```

The main control loop is a single function (MANUSCRIPT:180-199, verbatim "```python / def peda_step(current_state, world_model, drives): … return None / ```"):

```python
def peda_step(current_state, world_model, drives):
    predicted_state = world_model.predict(current_state, action=None)
    perceptual_error = compute_error(predicted_state, current_state)
    if perceptual_error.total > THRESHOLD:
        candidates = generate_candidates(current_state, max_candidates=3)
        for action in candidates:
            trajectory = world_model.rollout(current_state, action, horizon=2)
            efe = compute_efe(trajectory, drives)
        best_action = select_min_efe(candidates)
        result = execute(best_action)
        actual_next_state = perceive(result)
        model_error = compute_error(
            world_model.predict(current_state, best_action),
            actual_next_state
        )
        learning_buffer.store(current_state, best_action, actual_next_state, model_error)
        return best_action
    elif drives.novelty > THRESHOLD:
        return generate_exploratory_action(current_state, drives)
    return None
```

The candidate pool is capped at 3 actions (MANUSCRIPT:185, "candidates = generate_candidates(current_state, max_candidates=3)") and rollouts run for horizon=2 (MANUSCRIPT:187, "trajectory = world_model.rollout(current_state, action, horizon=2)").

Four project invariants constrain every module in the loop (THEORY §6.4, quoting AGENTS.md:56-59 verbatim): (1) "**No Prompt, only Prediction Error.** Never add features that require user input to trigger behavior."; (2) "**Drive is emergent, not hardcoded.** Never write fixed goal lists or fixed drive weights."; (3) "**World Model is the core.** Spend ~80% of effort on the World Model; any new module must directly improve its predictions."; (4) "**Learning is intermittent, not continuous.** Collect data, then batch-update. Never do per-step online SGD."

## 3.2 Perception Module

Perception converts raw environmental signals into structured `State` objects; the state representation is JSON-structured rather than free text, which forces the LLM's attention onto causal state variables (MANUSCRIPT:207, "Converts raw environmental signals (file listings, process output, sensor readings) into structured `State` objects. Key design choice: **state representation must be JSON-structured, not free text** (GLM-5.2 recommendation, `PROMPT_PHASE2_START.md`). This forces the LLM's attention onto causal state variables and reduces the semantic gap between observation and prediction."). In the Busybox sandbox the state is a JSON object with fields `cwd`, `files`, `last_command`, `last_exit_code`, `last_output` (truncated to 200 chars), and `step` (MANUSCRIPT:209, "In the Busybox sandbox, state is a JSON object with fields: `cwd`, `files`, `last_command`, `last_exit_code`, `last_output` (truncated to 200 chars), `step`.").

## 3.3 World Model (LLM + LoRA)

The World Model is the cognitive core. It predicts **key state variable changes**, not complete future states, because joint prediction over a full state is intractable: a state with 10 independent variables, each 80% predictable, has a joint accuracy ceiling of 0.8^10 ≈ 10.7% (MANUSCRIPT:213, "PEDA's World Model predicts **key state variable changes**, not complete future states. This is a critical engineering decision informed by the Phase 1 evaluation: predicting complete states in a complex environment (Linux sandbox) is infeasible—a state with 10 independent variables, each 80% predictable, has a joint accuracy ceiling of $0.8^{10} \approx 10.7\%$.").

The **Three-Level Prediction Hierarchy** (MANUSCRIPT:215, "**Three-Level Prediction Hierarchy** (from `peda_report_v11.agent.final.md` §3.3.3):") targets exactly the state variables that are both causally relevant and predictable:

| Level | Prediction Target | Target Accuracy |
|-------|-------------------|-----------------|
| L1 | Command exit code | ≥90% (MANUSCRIPT:219, "| L1 | Command exit code | ≥90% | Low—deterministic from command semantics |") |
| L2 | Filesystem delta (create/delete/modify) | ≥70% (MANUSCRIPT:220, "| L2 | Filesystem delta (create/delete/modify) | ≥70% | Medium—requires causal understanding |") |
| L3 | Output summary (first 100 chars semantic) | ≥50% (MANUSCRIPT:221, "| L3 | Output summary (first 100 chars semantic) | ≥50% | High—many outputs inherently unpredictable |") |

Variables explicitly classified as **aleatoric** — timestamps, PIDs, random number outputs, precise network latency, exact memory usage — are not predicted and are excluded from accuracy (MANUSCRIPT:223, "Variables explicitly classified as **aleatoric** (not predicted): timestamps, PIDs, random number outputs, precise network latency, exact memory usage.").

Model architecture: Qwen2.5-0.5B-Instruct with LoRA adapters (rank=16); LoRA fine-tuning is intermittent, every 1000 steps, never online; multiple checkpoint snapshots are saved during training for ensemble uncertainty estimation (MANUSCRIPT:225, "Model architecture: Qwen2.5-0.5B-Instruct with LoRA adapters (rank=16). LoRA fine-tuning occurs intermittently (every 1000 steps), not online. Multiple checkpoint snapshots are saved during training for ensemble uncertainty estimation.").

## 3.4 Predictive Error Computer

The Predictive Error Computer quantifies the gap between prediction and actual outcome and decomposes it into epistemic and aleatoric components via an ensemble-variance method (MANUSCRIPT:229, "Quantifies the gap between World Model predictions and actual outcomes, decomposing error into epistemic and aleatoric components. The decomposition method:"). Per (state, action), the ensemble computes the variance across checkpoint predictions, then heuristically decomposes (MANUSCRIPT:243-244, "epistemic = ensemble_var                  # high when models disagree / aleatoric = max(0, mean_deviation - ensemble_var)  # residual: model agrees but wrong"): epistemic error is the ensemble variance (models disagree → worth exploring), and aleatoric error is the residual mean deviation minus the variance (models agree but are wrong → environment noise). The intuition is operationalized with 5 checkpoints (MANUSCRIPT:247, "**Epistemic intuition**: 5 checkpoints predicting `[0, 1, 0, 0, 1]` for `python train.py` → high ensemble variance → models "disagree" → epistemic uncertainty high → worth exploring."). The method is explicitly a **heuristic**, not a rigorous mathematical decomposition, and its validity depends on checkpoints representing different model beliefs rather than merely training noise (MANUSCRIPT:251, "This is explicitly a **heuristic** method, not a rigorous mathematical decomposition (acknowledged in `peda_report_v11.agent.final.md` §3.4.3 and `THIRD_PARTY_REVIEW_RESPONSE.md` §2.2). Its validity depends on checkpoints representing "different model beliefs" rather than merely training noise.").

## 3.5 Action Generator (EFE Minimizer)

The Action Generator selects actions by minimizing EFE, with per-trajectory computation (MANUSCRIPT:253, "Selects actions by minimizing Expected Free Energy. Implementation constrained by inference speed:"). The implemented EFE accumulation uses predicted uncertainty — 1.0 minus L1 confidence — discounted over the trajectory, scaled by the curiosity drive weight, plus the pragmatic term (MANUSCRIPT:260-267, "def compute_efe(self, trajectory, drives): … predicted_uncertainty = 1.0 - trajectory[i].level1_confidence … epistemic += predicted_uncertainty * epistemic_ratio * (DISCOUNT ** i) … drive_adjusted_epistemic = epistemic * drives.curiosity_weight / return drive_adjusted_epistemic + pragmatic  # pragmatic = 0 in pure-exploration mode").

Inference speed constrains the rollout. With horizon=2-3 and candidates=2-3, each decision step costs 4-9 LLM calls; at ~2 seconds per call, an unmitigated 50-100 calls per step would limit 48-hour runs to ~520-1700 steps, so the system degrades gracefully to single-step greedy information-gain selection when the budget is insufficient (MANUSCRIPT:270, "**Graceful Degradation**: When inference budget is insufficient for full rollout (horizon=2-3, candidates=2-3 → 4-9 LLM calls/step), the system degrades to single-step greedy information-gain selection. This was motivated by the inference speed bottleneck identified in third-party review (`THIRD_PARTY_REVIEW_RESPONSE.md` §2.3): at ~2 seconds per LLM call, 50-100 calls per step would limit 48-hour runs to ~520-1700 steps.").

## 3.6 Learning Module

The Learning Module implements intermittent batch learning to avoid catastrophic forgetting and unstable gradients: every 1000 steps it samples from an experience buffer (prioritized by epistemic error), performs LoRA fine-tuning (3 epochs, lr=2e-4, rank=16), saves a checkpoint for the ensemble, and clears the buffer (MANUSCRIPT:274, "Implements **intermittent batch learning**—a design choice to avoid catastrophic forgetting and unstable gradients. Every 1000 steps, the module samples from an experience buffer (prioritized by epistemic error), performs LoRA fine-tuning (3 epochs, lr=2e-4, rank=16), saves a checkpoint for the ensemble, and clears the buffer."). A Saturation Detector monitors the ratio of recent to older prediction errors within a window; when the error decline rate falls below 15%, it signals saturation and raises the novelty drive, pushing the agent toward new uncertainty sources instead of cycling in mastered regions (MANUSCRIPT:276, "**Saturation Detection** (from `peda_report_v11.agent.final.md` §3.6.3): monitors the ratio of recent (last 50) to older (first 50 of window) prediction errors. If error decline rate < 15%, the detector signals saturation. This triggers the Drive System to increase the novelty drive, pushing the agent to seek new uncertainty sources rather than cycling in mastered regions.").

## 3.7 Homeostatic Drive System

Four intrinsic drives modulate the EFE calculation, preventing pure prediction-error minimization from degenerating into endless chasing of the largest uncertainty (MANUSCRIPT:280, "Four intrinsic drives modulate the EFE calculation, preventing pure prediction-error minimization from degenerating into "epistemic gluttony" (endlessly chasing the largest uncertainty):"):

| Drive | Source | Behavior Effect | Strength Function |
|-------|--------|-----------------|-------------------|
| **Curiosity** | High epistemic error regions | Increases exploration priority | `tanh(α × epistemic_error)` (MANUSCRIPT:284, verbatim) |
| **Competence** | Success history (error decline) | Seeks challenge at ability edge | `optimal_challenge_zone(success_rate)` (MANUSCRIPT:285, verbatim) |
| **Boredom** | Low behavioral entropy | Forces action diversity | `1 - normalize_entropy(recent_actions)` (MANUSCRIPT:286, verbatim) |
| **Novelty** | Time since last external input | Seeks new information sources | `exp(-λ × time_since_last_input)` (MANUSCRIPT:287, verbatim) |

Final selection incorporates the drive weights into the EFE objective (MANUSCRIPT:291, "The final action selection formula incorporates drive weights:" and "$$\pi^* = \arg\min_{\pi} \left[ G(\pi) - \sum_{d \in \text{Drives}} w_d \cdot V_d(\pi) \right]$$"). The initial weights — curiosity=0.5, competence=0.5, boredom=0.3, novelty=0.4 — are experience-based guesses, not optimized; a grid search over {0.2, 0.5, 0.8} per weight (81 combinations) was planned but not completed (MANUSCRIPT:293, "Initial drive weights (curiosity=0.5, competence=0.5, boredom=0.3, novelty=0.4) and update parameters (tanh scaling, λ decay rate) are experience-based guesses, not optimized. Grid search over {0.2, 0.5, 0.8} for each drive weight (81 combinations) was planned but not completed in Phase 1 due to the environment-model mismatch problem (WATCHDOG C4).").

## 3.8 Action Executor and Safety Layer

Commands are filtered through a rule engine (`BLOCKED_PATTERNS`: `rm -rf /`, `mkfs.*`, fork bomb patterns, etc.) before execution, and World Model predictions undergo rule-based sanity checks (e.g., if the action is `rm file.txt`, the prediction must include file deletion). The Docker sandbox enforces `--read-only` mounts, `--cap-drop=ALL`, `--network=none` (or whitelist proxy), `--pids-limit=64`, and non-root user execution (MANUSCRIPT:297, "Commands are filtered through a rule engine (`BLOCKED_PATTERNS`: `rm -rf /`, `mkfs.*`, fork bomb patterns, etc.) before execution. World Model predictions undergo rule-based sanity checks (e.g., "if action is `rm file.txt`, prediction must include file deletion"). The Docker sandbox enforces `--read-only` mounts, `--cap-drop=ALL`, `--network=none` (or whitelist proxy), `--pids-limit=64`, and non-root user execution.").

## 3.9 Data flow summary

The complete per-step cycle has eight stages (MANUSCRIPT:303-310, "1. **Perception**: Raw environment → structured `State` (JSON) / 2. **Prediction**: World Model generates `PredictedState` (L1 exit code, L2 filesystem delta, L3 output summary) / 3. **Error Computation**: Compare predicted vs actual → `ErrorVector` (epistemic/aleatoric decomposition) / 4. **Drive Update**: Homeostatic Drive System adjusts weights based on error history / 5. **Action Selection**: Generate candidates → rollout each → compute EFE → select min-EFE action / 6. **Execution**: Execute command (through safety filters) → produce outcome / 7. **Model Error**: Compare World Model's action-conditioned prediction vs actual outcome / 8. **Learning Storage**: Buffer (state, action, next_state, error) for intermittent training"): perception → prediction → error computation → drive update → action selection → execution → model error → learning storage. This is the closed loop the experiments in §5 put under test.
# 4. Experimental Setup

All numbers cite the evidence bundles (`PEDA_FINAL/paper/evidence/*.md`) plus primary source files; every claim below is REPRODUCIBLE or PARTIAL per `CLAIMS_VS_EVIDENCE.md`.

## 4.1 Model

- **Base model:** Qwen2.5-0.5B-Instruct, loaded locally from `~/models/` — (evidence/phase1.md:50, results/phase1_eval.json: `"model": "Qwen/Qwen2.5-0.5B-Instruct"`).
- **Fine-tuning:** LoRA `r=16, alpha=32, dropout=0.05, bias="none", task_type="CAUSAL_LM"`, targets 7 linear projections — (evidence/phase2.md:119, `checkpoints/phase2/sandbox_adapter_e2/adapter_config.json`: `"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none"`; evidence/phase1.md:54, `src/phase1/world_model.py:41-43, 69-76`).
- **Training scale:** 65–1,378 transitions, 1–3 LoRA epochs; later phases add JEPA MLP predictors (1–3 hidden layers, `n_ensemble=3`, Adam `lr=1e-3`) and zero-shot RSSM — (PEDA_FINAL/PEDA_CONCLUSION.md:140, "65-1378 transitions, 1-3 epochs LoRA, 500-2000 steps JEPA"; evidence/phase5.md:158-159, `src/phase5/jepa_wm.py:37-46, 53-60`).
- **Inference mode:** real-LLM (no stubs); Phase 1 eval isolated in subprocesses, `max-candidates=4` — (evidence/phase1.md:58, `results/phase1_report.md` caveats 4-5).

## 4.2 Hardware

| Resource | CPU workstation (Phases 1–2) | GPU instance (Phases 3–8) |
|---|---|---|
| CPU | Intel Core Ultra 9 185H, 22 cores | g4dn.xlarge: 4 vCPU |
| GPU | None (`CUDA: false`; Intel Arc unusable by PyTorch) | NVIDIA T4 16 GB |
| RAM | 30 GB (18 GB available during experiment) | 16 GB |
| Region / runtime | local | AWS us-east-1; torch 2.13.0 |

- CPU row — (evidence/phase3.md:122, `cpu: Intel Core Ultra 9 185H (22 cores), gpu: None (CUDA: false), ram: 30GB`; evidence/phase2.md:136, "Intel ARC not usable by PyTorch"; evidence/phase2.md:146, `torch 2.13.0, cuda_available false`).
- GPU row — (evidence/phase7.md:79, `PHASE4_EXPERIMENT_PLAN.md:267`: "Instance type: g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM), us-east-1").
- **CPU latency made PEDA infeasible:** first call ~176 s cold start, then ~3 s/call; 12–24 inference calls/step with 3-ensemble → 10–60+ min/episode — (evidence/phase3.md:120, `results/phase3_experiment/report.json`); Phase 1 CPU predict ~2.4–3.1 s/call, ~16 s/step, 100-episode eval ~22 h — (evidence/phase1.md:57). GPU runs: Grid World N=20 in 876 s (14.6 min) — (evidence/phase3.md:113); Phase 2 GPU session ~4.5 h / ~$2.40 — (evidence/phase2.md:145); Phase 4 ~14 GPU-hours — (evidence/phase4.md:42).

## 4.3 Environments

| Environment | Scale | Actions | Notes |
|---|---|---|---|
| Grid World (Ph. 1) | 5×5, max_steps=50 | UP/DOWN/LEFT/RIGHT (4) | rewards wall −0.2 / move −0.05 / goal +1.0 |
| TextRoomEnv (Ph. 1.5) | 2 rooms (study↔hallway) | 6 | custom env, 3-step optimal; real TextWorld never evaluated |
| Busybox Sandbox (Ph. 2–5, 8) | v1→v4 (see Table 2) | shell commands, 12-command whitelist | Docker-contained (see §4.4) |
| Grid Maze (Ph. 6–7) | 5×5 / 10×10 / 20×20 | 4 moves | max_steps = min(w·h·4, 500); sizes cited, not state counts |

- Grid World — (evidence/phase1.md:49, `5x5 GridWorld, max_steps=50, rewards wall -0.2 / move -0.05 / goal +1.0`; evidence/phase1.md:140, 4 actions).
- TextRoomEnv — (evidence/phase1_5.md:11, `src/phase1_5/text_env.py:1-4`: "Two rooms connected by a door"; evidence/phase1_5.md:19, "real TextWorld was never used for evaluation"; evidence/phase1_5.md:83, "3-step optimal").
- Grid Maze — (evidence/phase6.md:105, `max_steps = min(width*height*4, 500)`; CLAIMS_VS_EVIDENCE.md:80, CANONICAL: "cite maze size (10x10, 20x20), not state counts").

**Table 2. Sandbox versions v1–v4** — (evidence/phase2.md:87-90):

| Version | Dirs | Files | Unique (s,a) | Source |
|---|---|---|---|---|
| v1 | 4 incl. root (docs, tmp, data) | 3 (hello.txt, docs/note.txt, data/lines.txt) | 22 | phase2.md:87 (`Dockerfile.busybox`) |
| v2 | 7 subdirs | 14 | 65 | phase2.md:88 (AGENTS.md:120, "3.0× v1") |
| v3 | 7 subdirs | 15 [INFERENCE] | — | phase2.md:89 (`Dockerfile.busybox_v3`) |
| v4 | 18 incl. root | 29 [INFERENCE] | 270 | phase2.md:90 (`tasks.py:123` "18 dirs"); phase5.md:24 |

## 4.4 Docker containment

- Per-episode container: `docker run -d --rm --cap-drop=ALL --read-only --tmpfs /tmp --network none` — (src/phase2/sandbox_env.py:138-141, verbatim flags; evidence/phase8.md:88).
- Command safety: 12-command whitelist `{ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep, find}` + 14 blocklist regexes (`rm/mv/cp/chmod/chown/dd/mkfs/mount/sudo/su/docker/kill/shutdown/reboot`) — (src/phase2/sandbox_env.py:17-24, verbatim `WHITELIST`/`BLOCKLIST_PATTERNS`; evidence/phase8.md:88).
- Read-only rootfs ⇒ `create_file: LIMIT` — (evidence/phase2.md:81, PEDA_WORKING_LOG.md:1505). Environment reward is always 0; all signal comes from the binary goal predicate — (evidence/phase8.md:146).

## 4.5 Metrics

- **G1/G2/G3 (Grid World):** G1 next-state prediction accuracy > 0.90; G2 mean steps < 0.50 × random; G3 revisit rate < 0.20 — (evidence/phase1.md:20, `scripts/phase1_eval.py`: "G1 = {g1:.4f} (target > 0.90)"; theory.md:192).
- **L1/L2/L3 (Sandbox), thresholds 0.90/0.70/0.50:** L1 = exact exit-code match ≥ 0.90; L2 = exact predicted files-set match ≥ 0.70; L3 = token-overlap ≥ 0.5 on `last_output` ≥ 0.50 — (evidence/phase2.md:21, `scripts/phase2_measure_l1l2l3.py:126-128`; evidence/phase2.md:24, definitions).
- **FHT:** step index of the first action passing the task's goal check; −1 if never — (evidence/phase2.md:81, `scripts/phase2_collect_data.py:206-219`).
- **SCR:** |unique (cwd, files) states| / steps — (evidence/phase2.md:83, `phase2_collect_data.py:221-224`).
- **DLR:** fraction of steps i≥2 where actions[i]==actions[i−1]==actions[i−2] — (evidence/phase2.md:84, `phase2_collect_data.py:225-229`).
- **Methodology correction:** the `success` field was `SCR > 0` (constant-true tautology); all Phase 3+ hit rates use FHT≥0 — (CLAIMS_VS_EVIDENCE.md:43, `phase3_sandbox_experiment.py:132`; CLAIMS_VS_EVIDENCE.md:56).

## 4.6 Baselines and conditions

- **Pragmatic:** `pragmatic_only` agent scoring only the EFE pragmatic term (goal-distance minimization), same `pragmatic_weight=3.0` as PEDA — (evidence/phase2.md:147; evidence/phase1.md:130, "Pragmatic-only: pragmatic_only=True, SAME pragmatic_weight=3.0").
- **Random:** uniform action selection, seed 42 — (evidence/phase2.md:147, "random (seed 42)").
- **Heuristic:** random + repetition penalty (avoid >2 repeats in last 5) — (evidence/phase2.md:147).
- **Count:** count-based pair novelty `0.5·(1/√(1+state_count)) + 0.5·(1/√(1+pair_count))`, backtrack penalty ×0.5, success cache — (evidence/phase5.md:180, `src/phase5/explorer.py:28-36`; evidence/phase7.md:115).
- **Fairness controls:** identical (start, goal, seed) episode pairs across agents; known/unknown CWDs counterbalanced round-robin (7,7,6 per CWD) — (evidence/phase1.md:130; evidence/phase3.md:45).
- **Drive config:** Phase 1 weights all 0.5 — (evidence/phase1.md:59); Phases 1.5/2: curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0, pragmatic_weight=3.0 — (evidence/phase2.md:148).
# 5.1 Phase 1–2: Grid World, TextWorld, and the Busybox Sandbox

Experiments E01–E07 cover the first two environments (a 5×5 Grid World and a custom two-room TextWorld) plus the Docker busybox sandbox. The headline finding is uniform: formal gates pass on training distributions, the driving signal does not appear. Across all seven experiments the ensemble epistemic error that PEDA is built on measured ≈0, and every observed behavioral difference is attributable to the drive system or the candidate set, not to prediction error.

## E01 — Grid World full-space training: gates pass by memorization

The World Model (Qwen2.5-0.5B-Instruct + LoRA, adapter `partial_adapter_real_25_e3`) was evaluated on the same 5×5 grid it was trained on (results/phase1_eval.json:3-4, `"model": "Qwen/Qwen2.5-0.5B-Instruct"`, `"adapter": "checkpoints/phase1/partial_adapter_real_25_e3"`). All three formal gates pass:

| Gate | Value | Threshold | Source (results/phase1_eval.json) |
|---|---|---|---|
| G1 next-state accuracy | 1.0 | > 0.90 | :19 `"g1_accuracy": 1.0` |
| G2 drive/random steps ratio | 0.4337 | < 0.50 | :17 `"g2_ratio": 0.4337349397590361` |
| G3 revisit rate | 0.0 | < 0.20 | :18 `"revisit_rate": 0.0` |

N=10 episodes, 10/10 success, mean 3.6 steps vs 8.3 for the random baseline (results/phase1_eval.json:13-16, `"episodes": 10`, `"mean_steps": 3.6`, `"random_mean_steps": 8.3`). The adapter was trained on 1920 synthetic transitions at train_fraction=0.25 (checkpoints/phase1/partial_adapter_real_25_e3/training_info.json, `{"transitions":1920,"train_fraction":0.25,"epochs":3,"batch_size":8,"learning_rate":0.0003}`). These passes are in-distribution memorization, not generalization: (results/phase1_report.md:24, `**In-distribution memorization, not generalization.** The adapter was trained on this exact 5×5 Grid World; evaluation uses the same distribution.`) and (PEDA_FINAL/archive/phase1/phase1_gap_report.md:28, `**Caveat**: All G1/G2/G3 "passes" are on the **same 5×5 grid used for training**. This is memorization, not generalization or mechanism validation.`). When the World Model predicts perfectly, the pragmatic term dominates EFE and drive-weight variation has negligible effect (results/phase1_report.md:25, `**G1=1.0 trivializes the other gates.**`).

A control with the untrained base model makes the environment's triviality explicit: with G1 ≈ 0.18 (results/phase1_g1_accuracy.json:5, `"g1_accuracy": 0.18`), both PEDA and pragmatic-only still solved 10/10 episodes in 3.6 steps each (results/phase1_base_model_comparison_summary.json, `"peda": {… "success_rate": 1.0, "mean_steps": 3.6 …}`, `"pragmatic_only": {… same …}`). The work log records the implication directly (PEDA_WORKING_LOG.md:1000-1002, `1. 即使 World Model 的下一状态预测准确率只有约 0.18，5×5 Grid World 仍能被纯 pragmatic planning 完美解决。 2. Drive System 的 epistemic / curiosity / novelty 信号没有改变成功率、步数或回访率。 3. 这证明 5×5 Grid World 无法衡量 PEDA 的预测误差驱动探索机制。`).

## E02 — Partial training (25% of cells): epistemic signal collapses to zero

A held-out test attempted to force genuine uncertainty by training on only 6 of 25 cells (train_fraction=0.25). The 1-epoch adapter reached g1_test_set = 0.8684 (results/phase1_partial_eval_10eps.json:57, `"g1_test_set": 0.8684`); the 3-epoch adapter generalized the remaining cells perfectly (PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:9, `` `g1_test_set = 1.0` on held-out pairs (previous 1-epoch run had 0.8684) ``), with per-epoch checkpoints converging 0.0308 → 0.0047 → 0.0035 (phase1_epistemic_blocker_report.md:5, `3 checkpoints saved (checkpoint_epoch_1/2/3) with decreasing loss (0.0308 → 0.0047 → 0.0035)`).

The ensemble variance that is PEDA's epistemic error was measured directly: of 28 state-action probes × 4 actions across the grid, only 2/28 showed any checkpoint disagreement (phase1_epistemic_blocker_report.md:8, `28 state-action probes × 4 actions across grid: only 2/28 showed any checkpoint disagreement`), and mean_epistemic_error = 0.0 in the smoke test (phase1_epistemic_blocker_report.md:10, `mean_epistemic_error = 0.0 in smoke test (1 episode × 2 conditions, max_steps=10)`). Root cause, stated in the blocker report: (phase1_epistemic_blocker_report.md:15, `` `train_fraction=0.25` (6/25 known cells) provides enough coverage for 0.5B model with 3 epochs to perfectly generalize the 5×5 grid dynamics. The grid is too small, the transition rules too simple, and 0.5B too large for this environment. ``). The functional consequence is that PEDA reduces to pragmatism (phase1_epistemic_blocker_report.md:23, `- PEDA ≈ pragmatic_only under current setup. Any "PEDA advantage" would come from drive system modulation (curiosity/boredom/novelty) not prediction-error-driven exploration.`); with the epistemic term zero, the EFE formula collapses to drive-modulated pragmatism (PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:38, `由于 epistmic_error=0，PEDA 的 EFE 公式退化为 drive_system.apply_to_efe(pragmatic * pragmatic_weight)`).

## E03 — PEDA vs pragmatic-only, N=10: identical behavior

The canonical comparison is the GPU rerun of 2026-07-27, not the Phase 1 CPU archive runs. With N=10 per condition, both agents took mean 2.6 steps to the unknown goal (results/phase3_gpu/report.json:46,57, `"mean_steps": 2.6`), Fisher exact p=1.0 (results/phase3_gpu/report.json:67, `"p_value": 1.0`) and Mann-Whitney p=1.0 (results/phase3_gpu/report.json:73-75, `"p_value": 1.0, "peda_mean_steps": 2.6, "pragmatic_mean_steps": 2.6`), with both at 100% success (results/phase3_gpu/report.json, `"goal_unknown_success_fisher": {… "peda": "10.0/10 (100%)", "pragmatic": "10.0/10 (100%)"}`). The report verdict is CORE_HYPOTHESIS_NOT_SUPPORTED. This rerun used a confidence-based epistemic proxy with `"ensemble_checkpoints": 0` (results/phase3_gpu/report.json:6) — the ensemble-variance signal is absent by construction, so the null result holds for a PEDA whose epistemic term is a proxy at best.

The CPU pilot reported directional differences in goal_unknown mean steps, PEDA 16.6 vs pragmatic 21.1 (PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:82-83, `|**goal_unknown**|**PEDA**|**0.7**|**16.6**|**0.31**|0.73|`, `|Pragmatic-only|0.6|21.1|0.37|0.75|`), but this is directional only: no significance test was computed (phase1_partial_training_eval_report.md:40, `**样本量**：每个条件每 agent 10 episode，不足以做显著性检验。`). The held-out obstacle test reached the same conclusion as the GPU rerun: (PEDA_FINAL/archive/phase1/phase1_gap_report.md:36, `` **Not validated.** The held-out obstacle test showed PEDA and `pragmatic_only` behaved identically — the epistemic/drive component produced no measurable behavioral difference. ``).

## E04 — TextWorld (custom two-room): distinguishable but not better, no statistics

The environment was a custom TextRoomEnv — study and hallway connected by a door, 6 legal actions, 3-step optimal path (take key → go north → unlock chest) (src/phase1_5/text_env.py:114-118, `all_actions() = ["look", "inventory", "take key", "go north", "go south", "unlock chest with key"]`); real TextWorld was never evaluated. Training data was 113 unique transitions, 114 after augmentation (PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:88, `最终：**113 条唯一样本**`; phase1_5_iteration2_report.md:30-31, `去重后：**114 条**`), roughly two orders of magnitude below what a 0.5B model needs (phase1_5_deviation_report.md:18, `113 unique samples (50 walks × 20 steps, deduped); after augmentation: 114 | 2 orders of magnitude below what a 0.5B model needs`). Both agents scored 0% success in both iterations (results/phase1_5_eval_chunk_0.json:137-140, `"PEDA success_rate=0.00 vs pragmatic=0.00, diff=+0.00 (threshold +0.10)"`; phase1_5_iteration2_report.md:50-51, `**PEDA** take key → inventory → look → inventory×7 ❌ 10/10 | **Pragmatic** look×10 ❌ 10/10`). The World Model also learned wrong dynamics: all three checkpoints predicted `take key` exits with error (PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:112-116, `所有 3 个 checkpoint 对 take key 的预测都是 exit=1（不能拿钥匙）— 错误的。环境实际允许拿钥匙。`).

PEDA's behavior was distinguishable: it attempted `take key` at step 3 in iteration 1 and step 1 in iteration 2, while pragmatic issued `look` for the entire episode (phase1_5_complete_report.md:138-139, `**PEDA** ❌ 20/20 | inventory → look → **take key** (step 3!) → inventory × 17 | **Pragmatic** ❌ 20/20 | look × 20`; phase1_5_iteration2_report.md:55, `PEDA 在第 1 步就尝试 take key（比 Iteration 1 的 step 3 更快）。Pragmatic 从未尝试。`). The measured epistemic error rose from 0.0 to 0.20 after a decompose_error fix, but this signal was inventory-state confusion, not environmental complexity (PEDA_FINAL/archive/phase1_5/phase1_5_deviation_report.md:57, `The decompose_error fix raised mean_epistemic_error to 0.20, but this was driven by inventory-state confusion, not genuine environmental complexity.`). No p-values exist for Phase 1.5: all findings rest on 1–2 episodes per condition and statistical significance is explicitly unknown (phase1_5_deviation_report.md:65, `**No multi-episode statistics**: All Phase 1.5 behavioral findings are based on 1-2 episodes per condition. Statistical significance is unknown.`). The phase proved PEDA ≠ Pragmatic, not PEDA > Pragmatic (phase1_5_deviation_report.md:55, `Phase 1.5 proved 'PEDA ≠ Pragmatic' but not 'PEDA > Pragmatic in task completion'`).

## E05 — Sandbox v1: L1/L2/L3 pass in-distribution only

On the v1 sandbox (4 directories), the e2 adapter (200 curated transitions) passed all three prediction thresholds on held-out data: L1=1.000, L2=0.900, L3=0.550 (PEDA_WORKING_LOG.md:1479, `e2 held-out: L1=1.000 PASS, L2=0.900 PASS, L3=0.550 PASS`; AGENTS.md:116, `Phase 2b: L1=1.000, L2=0.900, L3=0.550 held-out [OK]`). The held-out split comes from the same training distribution (PEDA_WORKING_LOG.md:1298, `Caveat: held-out split is from the same random/heuristic training distribution; does not prove OOD generalization.`).

More data regressed: the e3 adapter trained on the full 10,040-transition dataset — 610 episodes of random+heuristic data (PEDA_WORKING_LOG.md:1334, `Merged existing + fast baseline data into results/phase2_train_merged_v2.jsonl: 610 episodes, 10,040 transitions.`) — scored L1=0.833, L2=0.333, L3=0.133 (PEDA_WORKING_LOG.md:1481, `e3 held-out: L1=0.833 FAIL, L2=0.333 FAIL, L3=0.133 FAIL`) — data quality over quantity (PEDA_WORKING_LOG.md:1490, `数据质量 > 数量：e3 (10,040 random+heuristic) 退化。`). PEDA completed 20/20 multi-task episodes (read_note, count_lines, read_hello, find_secret) with FHT=0.00, SCR=1.00, all 1-step (PEDA_WORKING_LOG.md:1484-1486, `read_note/count_lines/read_hello/find_secret: FHT=0.00, SCR=1.00, all 1-step` / `20/20 episodes 全部一次完成`), but the mechanism was action visibility, not prediction-error exploration (PEDA_WORKING_LOG.md:1492, `PEDA 可靠完成 4/5 任务：max_candidates=8 + goal_predicate 使候选集包含完成动作。但机制是动作可见性，非预测误差探索。`).

## E06 — Sandbox v2 OOD: every threshold fails

The same e2 adapter, evaluated on 35 held-out transitions from the v2 sandbox (7 directories including logs/, projects/, README.txt — none seen in training), scored L1=0.800, L2=0.686, L3=0.229, failing all thresholds:

| Level | In-distribution (v1) | OOD (v2, 35 samples) | Threshold |
|---|---|---|---|
| L1 exit code | 1.000 PASS (PEDA_WORKING_LOG.md:1479) | 0.800 FAIL (results/phase2_remaining/l1l2l3_heldout.json:3) | 0.90 |
| L2 filesystem delta | 0.900 PASS (PEDA_WORKING_LOG.md:1479) | 0.686 FAIL (l1l2l3_heldout.json:4) | 0.70 |
| L3 output summary | 0.550 PASS (PEDA_WORKING_LOG.md:1479) | 0.229 FAIL (l1l2l3_heldout.json:5) | 0.50 |

The artifact records all three `*_pass` fields false (results/phase2_remaining/l1l2l3_heldout.json:6-8, `"l1_pass": false, "l2_pass": false, "l3_pass": false`). The evaluation report states this is a genuine generalization failure (results/phase2_remaining/report.json:7-9, `Held-out test on sandbox v2 OOD directories (logs/, projects/, README.txt). e2 adapter (trained on old v1 4-dir sandbox …) shows expected degradation … This is a genuine finding — the WM does not generalize to new directory layouts.`). Consequently the Phase 2 "success" claims hold only on v1 (PEDA_WORKING_LOG.md:1523, `Phase 2 的"成功"声明（L1=1.000, 20/20 多任务完成）仅在 v1 沙箱（4 目录）上成立。v2 沙箱（7 目录）上 WM 不泛化。Phase 2 实质上是沙箱基建 + 数据管道，不是 PEDA 运行。`).

## E07 — Multi-baseline v2: pragmatic beats PEDA; read_note fails for everyone

On v2 with 5 episodes per condition, read_hello was completed by pragmatic at 100% / mean 1.0 steps versus PEDA at 80% / mean 2.8 steps (results/phase2_remaining/multi_baseline_results.json:12-20,32-40, `"peda/read_hello": {… "success_rate": 0.8, "mean_steps": 2.8}`, `"pragmatic/read_hello": {… "success_rate": 1.0, "mean_steps": 1.0}`). read_note: every baseline — PEDA, pragmatic, random — scored 0% success with mean_steps=10.0 (multi_baseline_results.json:22-30, `"peda/read_note": {… "success_rate": 0.0, "mean_steps": 10.0}`; PEDA_WORKING_LOG.md:1519, `read_note 任务：所有基线 0% 成功率`); pragmatic additionally showed an 0.8 revisit rate on read_note, a dead-loop pattern (results/phase2_remaining/report.json:18, `Pragmatic showed 80% revisit rate on read_note (dead-loop behavior). PEDA had 0% revisit but also 0% success.`).

The 300-episode fast-baseline aggregate contains no PEDA row — PEDA was infeasible on CPU, where ActionGenerator inference hung (PEDA_WORKING_LOG.md:1333, `PEDAData failed: ActionGenerator WorldModel inference hung on CPU-only hardware (0 episodes, 0 transitions)`). Random scored AvgSCR 0.180, AvgDL 0.080; heuristic AvgSCR 0.220, AvgDL 0.000 (PEDA_WORKING_LOG.md:1351-1352, `random: AvgFHT=1.0, AvgSCR=0.180, AvgDL=0.080` / `heuristic: AvgFHT=1.0, AvgSCR=0.220, AvgDL=0.000`; results/phase2_multi_baseline_aggregate.json, `"random": {… "scr_sum": 54.0 …}`, `"heuristic": {… "scr_sum": 66.0 …}`).

## Verdict

Across E01–E07 the World Model memorizes its training distribution (G1=1.0; in-distribution L1/L2/L3 all pass) and fails out of distribution (OOD L1=0.800, L2=0.686, L3=0.229). The ensemble epistemic signal that drives PEDA measures ≈0 wherever it was instrumented (2/28 probe disagreement; mean_epistemic_error = 0.0), and the only behavioral differences observed — E03's 2.6 vs 2.6 steps at p=1.0, E04's take-key timing, E07's pragmatic outperforming PEDA on read_hello — do not constitute prediction-error-driven exploration. The core hypothesis is not supported by Phases 1 and 2.
# 5.2 Phase 3–4: The N=20 Confirmatory Test and Its Collapse Under Metric Audit

> Sources: `evidence/phase3.md` (E08, E09, E03) and `evidence/phase4.md` (E10–E12).
> Every number carries `(source_file:line, verbatim_quote)`; claims follow
> `CLAIMS_VS_EVIDENCE.md` (REPRODUCIBLE as-is, PARTIAL caveated).
> The Phase 3 positive result is a **candidate-engineering artifact**, not
> prediction-error-driven exploration — this caveat accompanies every citation of it.

## 5.2.1 E08/E09: N=20 sandbox confirmatory test

**Design.** 80 episodes, 4 conditions × N=20, task `read_hello`, adapter `sandbox_adapter_v2_full`, T4 (g4dn.xlarge)
(`results/phase3_sandbox_n20/ANALYSIS_REPORT.md:14-20`, "| PEDA | unknown | /sandbox/logs, /sandbox/projects, /sandbox/tmp | 20 | PEDA agent on novel CWDs (not seen
during training) |"). CWDs known `/sandbox, /sandbox/data, /sandbox/docs`; unknown `/sandbox/logs, /sandbox/projects, /sandbox/tmp`; counterbalanced 7/7/6 per CWD,
max_steps=10 (`ANALYSIS_REPORT.md:25`, "CWDs are counterbalanced across conditions — each condition sees the same three CWDs in the same round-robin pattern (7, 7, 6
episodes per CWD)").

**Primary comparison (PEDA unknown vs Pragmatic unknown).** Mean steps 7.20 vs 10.00; medians equal at 10.0/10.0; SD 3.91 vs 0.00 (`ANALYSIS_REPORT.md:65-68`, "|
Mean steps | 7.20 | 10.00 |" / "| Median steps | 10.0 | 10.0 |"). Mann-Whitney two-sided p = 0.0043 (raw 0.004294336884693755, `phase3_n20_result.json:12`), U =
130.0, Cohen's d = −1.01 (`ANALYSIS_REPORT.md:71-74`, "- **Cohen's d:** -1.01 (large effect; negative because PEDA has fewer steps)"), rank-biserial r = 0.35 medium
(`phase3_n20_result.json:14`, "verdict: … MW p=0.0043, r=0.35, medium effect").

**Crossover interaction.** Advantage flips sign: −3.15 in known (Pragmatic better) vs +2.80 in unknown (PEDA better) (`ANALYSIS_REPORT.md:115-116`, "| Known | −3.15
(Pragmatic better) |" / "| Unknown | +2.80 (PEDA better) |"); interaction MW U = 315.5, p = 0.0001 (`ANALYSIS_REPORT.md:120`, "**Interaction Mann-Whitney U**
(advantage_unknown > advantage_known): U = 315.5, **p = 0.0001**").

**Caveat — candidate engineering, not prediction-error-driven exploration.** The benefit is entirely one CWD (`ANALYSIS_REPORT.md:96`, "the World Model maps
`/sandbox/projects` to its nearest training CWD and navigates directly"): PEDA 2.00 vs Pragmatic 10.00 steps, MW U = 0.0, p = 0.0004, all 7 episodes in 2 steps
(`ANALYSIS_REPORT.md:91,96`, "| /sandbox/projects | 2.00 | 10.00 | 0.0 | 0.0004 |" / "All 7 episodes in this CWD completed in 2 steps"). `/sandbox/logs` and
`/sandbox/tmp` show zero advantage — 10.00 vs 10.00, U = 24.5 / 18.0, "— (identical)" (`ANALYSIS_REPORT.md:90,92,97`). Adapter generalization to the nearest training
CWD is candidate/layout engineering (`evidence/phase3.md` §5.2, "adapter generalization … not prediction-error-driven exploration"), consistent with the hypothesis
being disproven.

**Metric validity — the tautology.** `success` was defined as `scr > 0` (`scripts/phase3_sandbox_experiment.py:132`, `"success": metrics["scr"] > 0,`), so all 80
episodes report `success=true` (`ANALYSIS_REPORT.md:26`). Real completion (fht ≥ 0) occurred in 14/80 episodes — 7 PEDA-unknown-`/sandbox/projects` (fht=1, 2 steps)
+ 7 Pragmatic-known-`/sandbox` (fht=0, 1 step) (`evidence/phase3.md` §5.1; raw `phase3_sandbox_n20_peda_unknown.jsonl:2`, `"cwd": "/sandbox/projects", "steps_count":
2, "success": true, "fht": 1`). In the unknown condition true completion is PEDA 7 vs Pragmatic 0 — the only genuine (and tiny) signal. 66/80 episodes hit the
10-step ceiling (`evidence/phase3.md` §5.6).

**E09 negative control.** PEDA known 10.00 steps, deterministic 20×10 (`ANALYSIS_REPORT.md:139`, "- **PEDA known steps:** [10, 10, ... (x20)]") vs Pragmatic known
6.85 (7×1 + 13×10; `ANALYSIS_REPORT.md:140`), MW U = 270.0, p = 0.0043, d = 1.01 (`ANALYSIS_REPORT.md:150-152`) — PEDA pays a significant cost in familiar
environments, recouped only via the single-CWD effect above. Confounds: Pragmatic's higher steps are driven by dead-loops (dlr 0.52 known / 0.80 unknown vs PEDA
0.00; `ANALYSIS_REPORT.md:41-42`), and PEDA is ~2× slower wall-clock (302.5/203.5s vs 129.6/159.2s; `ANALYSIS_REPORT.md:39-42`), so the step "advantage" is not an
efficiency win (`evidence/phase3.md` §5.5).

**Corroborating null.** The cleaner confidence-based grid-world test (empty ensemble, epistemic = 1 − confidence; `scripts/phase3_fast.py:15-19`) is a flat null:
PEDA 2.6 vs Pragmatic 2.6 steps, Fisher p=1.0000, MW p=1.0000, verdict `CORE_HYPOTHESIS_NOT_SUPPORTED`, 3/7 criteria (`results/phase3_gpu/report.json`,
"goal_unknown_steps_mannwhitney: p_value 1.0, peda_mean_steps 2.6, pragmatic_mean_steps 2.6" / "verdict: CORE_HYPOTHESIS_NOT_SUPPORTED … (Fisher p=1.0000, MW
p=1.0000)"). The two operationalizations of the epistemic signal disagree, and E03 is the cleaner test (`evidence/phase3.md` §5.8).

## 5.2.2 E10: closed-loop self-training (4 blocks × N=10, read_hello)

PEDA+Train success rose 20% → 60% → 80% → 60% across blocks with mean steps 16.2 → 11.0 → 6.8 → 14.6, while PEDA+Freeze stayed flat at 2/10 (20%), 16.2 steps
(`results/phase4a/PHASE4_RESULTS.md:22-25`, "| 1 | 2/10 (20%) | 16.2 | 2/10 (20%) | 16.2 |" … "| 4 | 6/10 (60%) | 14.6 | 2/10 (20%) | 16.2 |"; `:33`, "**PEDA+Train
success rate increased 4× across blocks (2/10 → 8/10)…**").

**PARTIAL — caveated.** Per-episode JSONL was lost (`PHASE4_RESULTS.md:38`, "Per-episode JSONL data lost (instance terminated before download)"); the curve survives
only in tmux scrollback; the `success` field was later proven tautological (see 5.2.1); the conclusion doc reclassifies the result as `POSITIVE — but success-cache
mechanism` (`PEDA_FINAL/PEDA_CONCLUSION.md:61`, "PEDA+Train: 20%-60%-80%-60% success. PEDA+Freeze: flat 20% | POSITIVE — but success-cache mechanism"). Block 4
regressed to 6/10, 14.6 steps (`PHASE4_RESULTS.md:40`). Runtime: ~14 GPU-hours on T4 (`PHASE4_RESULTS.md:6`).

## 5.2.3 E11: multi-task generalization (4 tasks × N=5, max_steps=20)

65 episodes / 13 cells (`results/phase4b_rerun/ANALYSIS_REPORT.md:4`, "13 files, 5 episodes each (65 total)") — and all 65 report `success=True`, again the tautology
(`ANALYSIS_REPORT.md:25`, "All 65 episodes report `success=True`. … **The real success metric is FHT >= 0.**"). Hits by FHT: `read_hello` is the only task with any —
PEDA unknown 2/5 (both `/sandbox/projects`, fht=1, 2 steps; `ANALYSIS_REPORT.md:33`, "| read_hello | all -1 | [-1, 1, -1, -1, 1] | 0/5 | **2/5** | p=0.1770 |";
`:38`, "Direction consistent with epistemic advantage but **not significant** (p=0.1770, N=5 per cell)") and Pragmatic known 2/5 instant solves
(`ANALYSIS_REPORT.md:46,51`); `count_lines`, `find_secret`, `read_note` are 0/5 in **all** conditions (`ANALYSIS_REPORT.md:34-36`). PEDA never dead-loops (dlr 0.00
everywhere) vs Pragmatic 0.54–0.90 (`ANALYSIS_REPORT.md:59-64`, "**Key finding**: … PEDA maintains dead_loop_rate=0.00 across all conditions — **PEDA never
dead-loops**"). Ground truth: `FAIL — WM cannot solve any task beyond cat hello.txt` (`PEDA_CONCLUSION.md:62`); the World Model's overconfident wrong predictions
leave no prediction to be uncertain about (`PEDA_CONCLUSION.md:83`).

## 5.2.4 E12: v4 replication with corrected metric (max_steps=10)

Re-auditing Phase 3 with fht ≥ 0 gives peda_known 0/20, peda_unknown **7/20 (35%)**, pragmatic_known 7/20 (35%), pragmatic_unknown 0/20
(`PEDA_WORKING_LOG.md:1700-1705`, "| peda_unknown read_hello | **7/20 (35%)** | **2/5 (40%)** |"). The v4 rerun (N=5) reproduces it: peda_unknown 2/5 (40%),
pragmatic_known 2/5 (40%), all other 14 cells 0 hits (`PEDA_WORKING_LOG.md:1735-1752`, "| peda_unknown_read_hello | 5 | 2 | 40% | 6.8 | 0.300 | 0.00 |"; raw
`results/phase4b_v4/peda_unknown_read_hello.jsonl:2`, `"cwd": "/sandbox/projects", "steps_count": 2, "fht": 1`; `pragmatic_unknown_read_hello.jsonl:1`, `"fht": -1,
"dead_loop_rate": 0.8`). Verdict: "**Phase 4B v4 完全复现 Phase 3，无退化，无翻车。**" (`PEDA_WORKING_LOG.md:1707`); ground truth `CONFIRMED — only read_hello, only
/sandbox/projects` (`PEDA_CONCLUSION.md:63`). This confirms the Phase 3 signal is real but **candidate-specific**: it survives only for `read_hello` from
`/sandbox/projects`, the CWD the adapter generalizes to (`PEDA_CONCLUSION.md:63`; `ANALYSIS_REPORT.md:96`).

## 5.2.5 The five Phase-4 bugs (fixed pre-Phase-5)

All five are documented in `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:57-61` ("## Bugs Found (5 fixed)"):
1. Path predicate cwd-unaware, breaking cwd-relative task checks (`README.md:57`).
2. Novelty tie-breaking alphabetical, distorting equal-score action selection (`README.md:58`).
3. `final_state.victory` always False, making success/fht unmeasurable (`README.md:59`).
4. Dead-loop rate 0.53–0.80 from repeated (state, action) pairs — fixed by a repeat-action penalty (score ×= 0.1) (`README.md:60`); post-fix PEDA dlr = 0.00 across all conditions (`phase4b_rerun/ANALYSIS_REPORT.md:59-62`).
5. JEPA predictor cross-task contamination, fixed by `reset_predictors()` between conditions (`README.md:61`).

## 5.2.6 What Phase 3–4 establish

The one robust, reproducible finding is narrow: PEDA's adapter World Model generalizes to exactly one novel CWD (`/sandbox/projects`), yielding 7 true completions vs
0 for Pragmatic in the unknown condition (E08, p=0.0043; E12, 35–40% vs 0%) — a **candidate-engineering effect, not prediction-error-driven exploration**, as
evidenced by zero effect at `/sandbox/logs` and `/sandbox/tmp`, the E03 null, and the failure on every task except `read_hello`. The `success`-field tautology and
lost per-episode data invalidate the E10 self-training headline; the reliable engineering outcomes are PEDA's dead-loop immunity and the five-metric-bug audit that
made fht the only admissible success metric.
# 5.3 Phase 5–7: JEPA and RSSM Epistemic Signals vs. Count-Based Novelty

Phases 5–7 tested whether *learned* forward-dynamics predictors — a JEPA MLP ensemble (Phases 5–6) and a MiniRSSM (Phase 7) — can supply an epistemic uncertainty signal that drives exploration better than count-based pair novelty. The verdict is negative in every regime tested:

> "Across 11 experiments spanning 4 sandboxes (v2/v3/v4 grid maze deterministic/stochastic), JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**." (PEDA_FINAL/archive/phase5_jepa_exploration/README.md:9)

Count-based pair novelty, not learned prediction error, survived as the reliable exploration mechanism:

> "| **Count-based pair novelty** | Core exploration driver | Beat all learned signals in 17 experiments at <1000 states; handles stochastic items (Phase 6) |" (PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:16)

## E13 — Phase 5: JEPA hybrid exploration (11 sub-configs)

Of the 11 sub-configs, only 7 survive as anchors (E13.01–E13.07); the other 4 (E13.08–E13.11) are unrecoverable — "4 sub-configs: raw JSONL deleted | MISSING | No on-disk source | Exclude from paper or cite as 'unrecoverable'" (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md, E13 rows). The 7 surviving anchors:

| Sub-ID | Environment | Count-based | JEPA best | Source |
|---|---|---|---|---|
| E13.01 | Sandbox v2 | 50% | 50% (hybrid) | README.md:13 |
| E13.02 | Sandbox v4 | 42% | 8% (hybrid) | README.md:14 |
| E13.03 | Maze 10x10 (deterministic) | 100% | 0% | README.md:15 |
| E13.04 | Maze 10x10 (stochastic) | 100% | 67% (hybrid); pure JEPA 0% | README.md:17 |
| E13.05 | Maze 20x20 | 0% | 0% | README.md:16 |
| E13.06 | P4 EFE (4 rounds) | 50% | 25% | README.md:18 |
| E13.07 | read_hello (v2) | 50% (novelty-only) | 17% (jepa_efe) | PEDA_CONCLUSION.md:64 |

Every row is verbatim from its cited line, e.g. `| Sandbox v2 | 65 | 50% | 50% (hybrid) | — |` (README.md:13) and `| Sandbox v4 | 270 | 42% | 8% (hybrid) | — |` (README.md:14). The maze rows carry doc-cited state counts (1,100 for 10x10, 8,400 for 20x20) that are **not reproducible**: `GridMaze.state_estimate()` returns `base * (items + 1)` (src/phase6/maze_generator.py:137-141), so we cite maze **size**, not state counts (CLAIMS_VS_EVIDENCE.md, "State counts" row).

Headline comparison, per task:

> "Novelty-only 50% > jepa_efe 17% on read_hello. JEPA loss converges, no exploration gain | FAIL — JEPA uncertainty flat across all unexplored states" (PEDA_FINAL/PEDA_CONCLUSION.md:64)

The JEPA MLP does learn the dynamics — "Loss always converges (45→15), MLP learns dynamics" (README.md:27) — but the derived signal is behaviorally inert. The often-quoted cost multiplier is **unbenchmarked**: "computed at approximately 37x the computational cost (MLP forward pass + embedding computation vs integer increment)" (PEDA_FINAL/PEDA_CONCLUSION.md:101) — no benchmark artifact survives, so we report it only as an estimate.

Root cause, stated in the archive:

> "All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower." (README.md:38)

> "For JEPA to beat counting, it needs to say 'THIS unexplored direction is more promising than THAT one.' This requires goal-conditioning or learned value in the embedding space — beyond what was tested." (README.md:40)

## E14 — Phase 5: pure epistemic explorer (jepa_only)

When the pragmatic term is removed entirely, exploration collapses:

> "| 5 | Sandbox v4 | Pure epistemic (jepa_only) explorer | Q2 | SCR ~0 across all tasks, zero room exploration | FAIL — epistemic signal too weak to drive useful behavior |" (PEDA_FINAL/PEDA_CONCLUSION.md:65)

> "- **Pure epistemic (jepa_only)**: SCR ~0, no room exploration" (README.md:33)

The only surviving raw record of this mode shows the same pattern at 5x5 maze scale: `{"mode": "jepa_only", "steps_count": 20, "success": false, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 49.932384}` (results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl:1) — 90% dead-loop rate, negligible state coverage.

## What survived: STRIPS action learning (candidate hit rate)

The one Phase 5 positive is not exploration; it is a learned action model on the **action-prediction task** (does the generated candidate set contain the correct action):

> "| **STRIPS action schemas** | Learned action model | 45.8% hit rate vs 31.3% fallback on v2 sandbox traces |" (PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:17)

> "| Hit rate | 45.8% | 31.3% | +14.5pp |" (PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:68)

The raw trace behind this figure does **not** survive (CLAIMS_VS_EVIDENCE.md, STRIPS rows: "raw trace missing"), so it is cited as a derived claim only. It must not be conflated with episode success: surviving v3 episode files show learned-mode **0/48** vs fallback **14/48 (29.2%)** episode completions (results/v3_learned_*.jsonl all `"success": false`; fallback 12/12 read_greeting plus 1/12 find_secret_note and 1/12 read_user_guide), a reversal explained by different measurement units and online-from-zero learning (evidence/phase5.md §4.1).

## E15 — Phase 6: Grid Maze 10x10

Deterministic 10x10: count reaches the goal every episode, every JEPA variant scores zero:

> "| Maze 10x10 | 1,100 | 100% | 0% | — |" (README.md:15)

> "| Maze 10x10 (deterministic) | 1,100 | 100% | 0% (any JEPA variant) | — |" (PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:51)

Stochastic 10x10 (item respawn): count 100%, hybrid 67%, pure JEPA 0%:

> "| Stochastic 10x10 | 1,100 | 100% | 67% | 0% |" (README.md:17)

The archive README is the authoritative source for the 67% figure: a later summary misattributed it to the deterministic maze and the errata corrects that (PEDA_FINAL/PEDA_CONCLUSION.md:4-8). All 10x10 numbers are **doc-claimed only**: `results/phase6_maze_count_10x10_seed42.jsonl` is **0 bytes** (empty), and no stochastic result file exists. The results also contradict the runner's stated expectation — "Expectation: 0% success — novelty expires after one visit per room, so the agent never returns to check for newly spawned items" (scripts/phase6_stochastic_count.py:6) — so without raw data the stochastic figures remain uncertain (CLAIMS_VS_EVIDENCE.md, "Stochastic" row).

Only 5x5 raw data survives (results/phase6_maze_*_5x5_seed42.jsonl):

- **Count, 6/6 episodes success**: `"steps_count": 42, "success": true, "fht": 41, "scr": 0.405, "dead_loop_rate": 0.0` (count file lines 1–6).
- **Pure novelty, 4/6 success** (2 failures hit the 100-step ceiling, scr 0.04 and 0.05); successes: scr 0.25–1.0, fht 15–67, dead_loop_rate 0.046–0.125 (pure_novelty file lines 1–6, verified directly).
- **jepa_only, 0/1 success**: scr 0.05, dead_loop_rate 0.9 (jepa_only file:1).

## E16 — Phase 6: Grid Maze 20x20

Neither agent solves 20x20:

> "| 6 | Grid Maze 20x20 | Count vs JEPA (8400 states) | Q3 | Count: 0%. JEPA: 0%. Both agents hit state-space ceiling | FAIL — neither approach scales to 8400+ states |" (PEDA_FINAL/PEDA_CONCLUSION.md:67)

> "| Maze 20x20 | 8,400 | 0% | 0% | — |" (README.md:16)

No 20x20 raw data survives; these figures are doc-claimed per the archive README (CLAIMS_VS_EVIDENCE.md, E16 row). Corroborating (but Phase 7, count-only) records show count at 20x20 achieving 0/3 success with scr 0.0 over 500 steps (results/phase7_giant_20x20.jsonl:1-3).

## E17 — Phase 7: GPU 5-track (RSSM, Goal-JEPA, Giant-JEPA, Curriculum, Count)

Only the RSSM track has persisted results; the other three learned tracks have no result files at all, and the Giant-JEPA rows are missing entirely: `results/phase7_giant_all.jsonl` is byte-identical to `results/phase7_giant_20x20.jsonl` (3 count-only rows, no `"method": "jepa"` records; CLAIMS_VS_EVIDENCE.md, E17.2–E17.4 rows).

The canonical summary's DLR claim was corrected by errata — "DLR ~0.996 → corrected to 0.8-0.9 (raw JSONL shows 0.8-0.9, not 0.996)" (PEDA_FINAL/PEDA_CONCLUSION.md:4-5). Every persisted Phase 7 JEPA record shows DLR 0.8–0.9:

- RSSM: `{"mode": "rssm", "steps_count": 20, "success": false, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 0.20126}` (results/phase7_rssm_rssm_5x5_seed42.jsonl:1)
- MLP-JEPA: `{"mode": "mlp_jepa", "steps_count": 20, "success": false, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 0.03715}` (results/phase7_rssm_mlp_jepa_5x5_seed42.jsonl:1)
- All-modes file (10-step cap): rssm scr 0.1, dlr 0.8; mlp_jepa scr 0.1, dlr 0.8 (results/phase7_rssm_all_modes_5x5_seed42.jsonl:1-2)

Count, in the same runs, reaches 4× the state coverage with zero dead loops: `{"mode": "count", "steps_count": 10, "success": false, "scr": 0.4, "dead_loop_rate": 0.0}` (results/phase7_rssm_count_5x5_seed42.jsonl:1).

Count SCR 0.4 vs RSSM/MLP-JEPA SCR 0.05 is the only RSSM-vs-count comparison the persisted data supports; the "count wins every track" claim otherwise rests on unpersisted runs (CLAIMS_VS_EVIDENCE.md, E17 row). A methodology caveat applies to the RSSM track itself: the module docstring promises "Ensemble of 3 RSSMs provides epistemic uncertainty via prediction variance" (src/phase7/rssm_wm.py:10), but the experiment builds **one** MiniRSSM and scores actions with prior-variance magnitude (scripts/phase7_rssm_experiment.py:203-210) — a deviation flagged alongside the result (CLAIMS_VS_EVIDENCE.md, E17.1 row).

## Synthesis (E13–E17)

Across 11 Phase 5 sub-configs (7 recoverable), Phase 6 mazes (10x10, 20x20, deterministic and stochastic), and the Phase 7 GPU tracks, learned forward-dynamics epistemic signals never beat count-based pair novelty. JEPA loss converges (45→15) and its uncertainty is uniform over unexplored transitions — behaviorally equivalent to counting at an unbenchmarked ~37× cost. The one statistically significant positive in the whole paper (Phase 3, p=0.0043) is "attributable to candidate set engineering and success caching, not epistemic prediction error" (PEDA_FINAL/PEDA_CONCLUSION.md:28). Under the tested conditions, count-based pair novelty is the reliable exploration mechanism.
## 5.4 Phase 8: Count-Driven Agent (E18–E19)

Phase 8 runs the final ablation: the agent keeps only count-based novelty and strips every learned component that motivated the project. There is no prediction-error mechanism, no World Model, and no Expected Free Energy term. The runner is described verbatim as "count-driven, no prediction error" (`results/phase8_gpu_run_2026-07-31.md:5`), and its class docstring states: `"No prediction-error mechanism. Novelty = count-based bonus. STRIPS action schemas are learned from experience. JEPA forward dynamics training is optional (--train-jepa flag)."` (`paper/evidence/phase8.md` §3, `src/phase8/count_driven_agent.py:76-88`).

### Agent design

- **Explorer** (`NoveltyExplorer`): intrinsic reward `0.5 * (1/sqrt(1+state_novelty)) + 0.5 * (1/sqrt(1+pair_novelty))` over (state, action) visit counts; selection order is cached-success replay, then highest novelty bonus, tie-broken by action priority (`paper/evidence/phase8.md` §4, `src/phase5/explorer.py:20,34-35,64-76`).
- **Success cache**: winning (state, action) pairs are memoized (`src/phase8/count_driven_agent.py:87`); counts and cache persist across episodes (`paper/evidence/phase8.md` §4).
- **Action generation**: STRIPS action schemas learned from experience (`ActionModelLearner`), candidates capped at 16 (`paper/evidence/phase8.md` §3).
- **JEPA (E19 only)**: trained as a side-effect on the last 20 transitions per episode; `select_action` never consults it — E19 is a training-side-effect ablation, not JEPA-driven exploration (`paper/evidence/phase8.md` §7.4, `count_driven_agent.py:147,192-197`).

### E18 — Count-only: 28/45 (62.2%) across 9 tasks

Setup: 5 episodes/task × 9 tasks = 45 episodes, max 10 steps/episode (`:14-15`: `"- Episodes per task: 5"` / `"- Max steps per episode: 10"`), 120 s timeout per task (`:17`), Docker `peda-sandbox:v2` (4 tasks) + `peda-sandbox:v4` (5 tasks) (`:8`), commit a348c1e (`:10`), g4dn.xlarge/T4 (`:6`). Every row below is `results/phase8_gpu_run_2026-07-31.md:23-32`.

| Task | Sandbox | Success | Avg steps |
|---|---|---|---|
| read_hello | v2 | 100% (5/5) | 1.2 |
| read_note | v2 | 20% (1/5) | 8.8 |
| count_lines | v2 | 0% (0/5) | 10.0 |
| find_secret | v2 | 100% (5/5) | 1.6 |
| read_welcome | v4 | 100% (5/5) | 1.4 |
| find_api_key | v4 | 20% (1/5) | 10.0 |
| count_measurements | v4 | 100% (5/5) | 1.6 |
| find_errors_v4 | v4 | 20% (1/5) | 10.0 |
| read_changelog_v4 | v4 | 100% (5/5) | 1.2 |
| **TOTAL** | | **28/45 (62.2%)** | — |

(`:32`: `"| | **TOTAL** | | **28/45 (62.2%)** | |"`). The failure pattern is categorical (`:36-38`):

- **Direct reads, 100% (5/9 tasks)** — read_hello, find_secret, read_welcome, count_measurements, read_changelog_v4: `"success cache enables 1-2 step solves after initial discovery"`.
- **Deep path reads, 20% (3/9 tasks)** — read_note, find_api_key, find_errors_v4: `"10-step ceiling exhausted before reaching target file in deep directory"`.
- **Zero (1/9 tasks)** — count_lines: `"wc -l never targets the correct filename"`.

### E19 — Count+JEPA: identical, zero delta

The same run with `--train-jepa`: `"JEPA training: ON (forward dynamics as side-effect, not exploration driver)"` (`:43`). Every per-task row is byte-identical to E18 (`:47-55`), total `28/45 (62.2%)` (`:56`: `"| | **TOTAL** | | **28/45 (62.2%)** | |"`).

| Metric | Count-Only | Count+JEPA | Delta |
|--------|:----------:|:----------:|:-----:|
| Total success | 28/45 (62.2%) | 28/45 (62.2%) | **0** |
| Per-task success | identical | identical | **0** |
| Avg steps | identical | identical | **0** |

(`:62-64`, verbatim).

### Interpretation

- **Zero delta is exact, not approximate**: `"JEPA forward dynamics training contributes zero additional value to the count-driven agent. Every task's success/failure pattern is identical with and without JEPA. This is consistent with 17 prior JEPA experiments where learned forward dynamics never improved exploration or task completion over count-based novelty"` (`:66`).
- **Scope of the falsification**: E19 shows that training JEPA as a side-effect changed nothing; the explorer never consults JEPA uncertainty for action selection (`paper/evidence/phase8.md` §7.4), so count-based novelty alone is the active mechanism in both conditions.
- **Start-cwd confound**: 3 of the 5 "direct read" tasks run from engineered start directories (`count_measurements` → `/sandbox/data/raw`, `read_changelog_v4` → `/sandbox/docs`; `count_driven_agent.py:35-48`). count_measurements (100%) vs count_lines (0%) are both `wc -l` tasks differing only in start cwd (`paper/evidence/phase8.md` §7.3).
- **Raw transcripts**: per-episode traces live in the GPU-run session as `artifact://870` / `artifact://872`, not on disk; only the aggregate table above is locally verifiable (`:70`).
# 6. Discussion

## 6.1 The hypothesis, restated and answered

PEDA asked whether prediction error from an LLM-based World Model can serve as an intrinsic drive signal that guides an agent to explore uncertain regions more effectively than goal-directed baselines. The hypothesis was tested across 19 controlled experiments (E01-E19) spanning 5 environments (Grid World, TextWorld, Busybox Linux Sandbox v1-v4, Grid Maze, Giant Maze), using 4 model architectures and ~2,000 total evaluation episodes.

All three charter sub-questions answer **No** under the tested conditions (PEDA_CONCLUSION.md:24, "All three charter sub-questions answer **No** under the tested conditions"):

- **Q1 (Signal):** LLM World Models produce epistemic error ~0 on small state spaces (<100 states) and uniform uncertainty on larger ones (PEDA_CONCLUSION.md:25, "LLM World Models produce epistemic error ~0 on small state spaces (<100 states) and uniform uncertainty on larger ones (JEPA ensemble, all DLR ~0.996)"). The model is too certain or uniformly uncertain — never differentially uncertain. (The ~0.996 DLR figure is an errata in the conclusion doc: every persisted run shows 0.8–0.9; see §6.4.)
- **Q2 (Drive):** EFE is dominated by pragmatic value; the epistemic term only changes action selection when all candidates are equally unpromising (PEDA_CONCLUSION.md:27, "EFE is dominated by pragmatic value. The epistemic term only changes action selection when all candidates are equally unpromising — at which point any action is equivalent").
- **Q3 (Effect):** PEDA never beats count-based novelty (PEDA_CONCLUSION.md:29, "PEDA never beats count-based novelty. The one statistically significant result (Phase 3, N=20, p=0.0043) is attributable to candidate set engineering and success caching, not epistemic prediction error. Phase 8 confirmed: count-driven reaches 62.2% across 9 tasks; toggling JEPA on adds zero delta").

The negative answers are conditional on the conditions tested, not a claim of impossibility in general (PEDA_CONCLUSION.md:44, "The answers below are conditional on the specific experimental conditions tested — they do not claim impossibility in general, only that under the conditions explored, the hypothesis does not hold"). Count-based pair novelty is the reliable exploration mechanism in state spaces under ~1,000 states (PEDA_CONCLUSION.md:30, "Count-based pair novelty, not epistemic prediction error from learned World Models, is the reliable exploration mechanism in state spaces under ~1,000 states").

## 6.2 Root Cause 1: The World Model is too certain

The 0.5B Qwen2.5 model with LoRA memorizes its training distribution near-perfectly (PEDA_CONCLUSION.md:93, "The 0.5B Qwen2.5 model with LoRA memorizes its training distribution near-perfectly"). Evidence:

- Grid World, 25% training (6/25 cells), 3 epochs: held-out next-state accuracy reached 0.8684 (results/phase1_partial_eval_10eps.json, `"g1_test_set": 0.8684`), and the 3-epoch adapter generalized perfectly to held-out pairs with only 2/28 state-action probes showing any checkpoint disagreement (PEDA_FINAL/archive/phase1/phase1_epistemic_blocker_report.md:8-9, "28 state-action probes × 4 actions across grid: only 2/28 showed any checkpoint disagreement"; "g1_test_set = 1.0 on held-out pairs (previous 1-epoch run had 0.8684)").
- Root cause stated at the time: (phase1_epistemic_blocker_report.md:15, "train_fraction=0.25 (6/25 known cells) provides enough coverage for 0.5B model with 3 epochs to perfectly generalize the 5×5 grid dynamics. The grid is too small, the transition rules too simple, and 0.5B too large for this environment").
- Sandbox v1, 200 curated transitions: L1=1.000, L2=0.900, L3=0.550 on the training distribution (PEDA_WORKING_LOG.md:1479, "e2 held-out: L1=1.000 PASS, L2=0.900 PASS, L3=0.550 PASS"), yet all three thresholds fail out of distribution (results/phase2_remaining/l1l2l3_heldout.json:1, `{"total": 35, "l1": 0.8, "l2": 0.6857, "l3": 0.2286, "l1_pass": false, "l2_pass": false, "l3_pass": false}`).
- Behavioral consequence: PEDA and pragmatic-only produced identical step counts, 2.6 vs 2.6, Fisher p=1.0, MW p=1.0 on goal_unknown (results/phase3_gpu/report.json, statistical_tests `goal_unknown_steps_mannwhitney: p_value 1.0, peda_mean_steps 2.6, pragmatic_mean_steps 2.6`; this GPU run used a confidence-based epistemic proxy with `"ensemble_checkpoints": 0`).

Epistemic error ~0 means EFE collapses to pragmatic value alone — Active Inference's exploration advantage evaporates (PEDA_CONCLUSION.md:93, "Epistemic error ~0 means EFE collapses to pragmatic value alone — Active Inference's exploration advantage evaporates").

## 6.3 Root Cause 2: Action-space engineering masks exploration

PEDA's apparent wins depend on the candidate set, not on intrinsic exploration (PEDA_CONCLUSION.md:95, "PEDA's apparent wins depend on the candidate set, not on intrinsic exploration"). The one statistically significant positive result — Phase 3, N=20, read_hello on unknown CWDs, 7.2 steps vs Pragmatic 10.0, MW p=0.0043, Cohen's d=-1.01 (results/phase3_sandbox_n20/ANALYSIS_REPORT.md:72-74, "p-value (two-sided): 0.0043"; "Cohen's d: -1.01 (large effect; negative because PEDA has fewer steps)") — is attributable to candidate engineering and success caching, not to epistemic prediction error. Every citation of this result carries this caveat:

- The advantage is entirely concentrated in one CWD, `/sandbox/projects` (2.00 vs 10.00 steps, p=0.0004), where all 7 episodes completed in 2 steps; in `/sandbox/logs` and `/sandbox/tmp` both agents hit the 10.0-step ceiling (results/phase3_sandbox_n20/ANALYSIS_REPORT.md:90-98, "| `/sandbox/projects` | 2.00 | 10.00 | 0.0 | 0.0004 |"; "The effect is entirely concentrated in `/sandbox/projects`, but it is perfectly reliable (zero variance) across all 7 repetitions").
- The correct action `cat hello.txt` was present in PEDA's candidate set and replayed from the success cache; the World Model did not "explore" (PEDA_CONCLUSION.md:79, "PEDA's advantage in `/sandbox/projects` (2.0 steps vs 10.0, p=0.0004) comes from the `NovellyExplorer`'s candidate set containing the correct action `cat hello.txt`, combined with the success cache replaying it after the first hit").
- Under the FHT (first-hitting-time) metric — the only discriminating metric, since `success` was a constant-true field defined as SCR > 0 (PEDA_WORKING_LOG.md:1712, "`phase3_sandbox_experiment.py:132`: `"success": metrics["scr"] > 0`") — tasks were actually completed in only 14/80 episodes, of which PEDA unknown contributed 7/20 versus Pragmatic unknown 0/20 (PEDA_WORKING_LOG.md:1715-1718, "- peda_unknown: 7/20（全来自 /sandbox/projects）").
- The effect never replicated on any other task: Phase 4B showed 0/5 hits on count_lines, find_secret, and read_note for ALL baselines including PEDA (PEDA_CONCLUSION.md:51, "read_hello peda_unknown 40% (2/5). count_lines/find_secret/read_note: **all zero** hits").
- The mechanism was identified as action visibility already in Phase 2: (PEDA_WORKING_LOG.md:1492, "PEDA 可靠完成 4/5 任务：max_candidates=8 + goal_predicate 使候选集包含完成动作。但机制是动作可见性，非预测误差探索。" — "the mechanism is action visibility, not prediction-error exploration").

## 6.4 Root Cause 3: JEPA uncertainty is flat

JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states (PEDA_CONCLUSION.md:90, "JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states"). Across 11 experiments spanning 4 sandboxes, JEPA-learned epistemic signal did not improve exploration over count-based novelty in any regime (PEDA_FINAL/archive/phase5_jepa_exploration/README.md:9, "JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**").

The pattern holds at every scale tested (PEDA_FINAL/archive/phase5_jepa_exploration/README.md:13-18):

- Sandbox v2 (65 states): count 50% vs JEPA hybrid 50% — tied (README.md:13, "| Sandbox v2 | 65 | 50% | 50% (hybrid) | — |").
- Sandbox v4 (270 states): count 42% vs JEPA hybrid 8% (README.md:14, "| Sandbox v4 | 270 | 42% | 8% (hybrid) | — |").
- Maze 10x10: count 100% vs JEPA 0% (README.md:15, "| Maze 10x10 | 1,100 | 100% | 0% | — |").
- Stochastic 10x10: count 100% vs JEPA best (hybrid) 67% vs pure JEPA 0% (README.md:17, "| Stochastic 10x10 | 1,100 | 100% | 67% | 0% |").
- Maze 20x20: count 0% vs JEPA 0% (README.md:16, "| Maze 20x20 | 8,400 | 0% | 0% | — |").
- Maze results are cited by maze size because the documented state counts (1,100/8,400) do not match `GridMaze.state_estimate()` (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md E15-E16, "CANONICAL: cite maze size (10x10, 20x20), not state counts").
- The 67% hybrid figure is attributed to the stochastic 10x10 maze, per the contemporaneous archive README (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md E15, "CANONICAL: stochastic 10x10 maze (archive README is contemporaneous source with internally consistent deterministic/stochastic rows)").

The pure-epistemic explorer (jepa_only) achieved SCR ~0 across all tasks with zero room exploration (PEDA_CONCLUSION.md:54, "| 5 | Sandbox v4 | Pure epistemic (jepa_only) explorer | Q2 | SCR ~0 across all tasks, zero room exploration | FAIL — epistemic signal too weak to drive useful behavior"). Persisted runs confirm:

- Maze 5x5, jepa_only: scr 0.05, dead_loop_rate 0.9 (results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl:1, `"scr": 0.05, "dead_loop_rate": 0.9`), while count solved 6/6 episodes (results/phase6_maze_count_5x5_seed42.jsonl:1-6, all `"success": true`).
- GPU Phase 7: RSSM and MLP-JEPA tracks both scored scr 0.05 with dead_loop_rate 0.9, versus count's scr 0.4 with dead_loop_rate 0.0 (results/phase7_rssm_rssm_5x5_seed42.jsonl:1, `"scr": 0.05, "dead_loop_rate": 0.9`; results/phase7_rssm_count_5x5_seed42.jsonl:1, `"scr": 0.4, "dead_loop_rate": 0.0`).
- Persisted DLR values are 0.8–0.9, not 0.996 (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md E17, "DOWNGRADE: 'DLR 0.8-0.9 (high determinism, low exploration)' — 0.996 NOT reproducible").
- Phase 8: toggling JEPA on added zero delta — 28/45 (62.2%) with and without JEPA (results/phase8_gpu_run_2026-07-31.md:62, "| Total success | 28/45 (62.2%) | 28/45 (62.2%) | **0** |").

JEPA's learned signal is "how uncertain am I about this (state, action) transition?" — equally high for every unseen transition, i.e., identical to count-based novelty at approximately 37x the computational cost (README.md:38, "All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower."). For JEPA to beat counting it must say "THIS unexplored direction is more promising than THAT one", which requires goal-conditioning or learned value in the representation space (README.md:40, "For JEPA to beat counting, it needs to say "THIS unexplored direction is more promising than THAT one." This requires goal-conditioning or learned value in the embedding space — beyond what was tested").

## 6.5 Root Cause 4: The World Model is uncalibrated — high confidence, low accuracy

The World Model shows high confidence in predictions even when they are wrong; the agent "confidently picks wrong actions" (PEDA_CONCLUSION.md:105, "The World Model shows high confidence in its predictions even when those predictions are wrong. In Phase 4B, PEDA in known CWDs has 0% hit rate on read_hello because the WM confidently predicts file contents that do not match the actual sandbox state. The agent "confidently picks wrong actions.""). Evidence:

- Calibration cost in familiar environments: PEDA in known CWDs completed 0/20 read_hello episodes while Pragmatic known completed 7/20 (PEDA_WORKING_LOG.md:1715-1718, "- peda_known: 0/20（所有 cwd 全零）"; "- pragmatic_known: 7/20（全来自 /sandbox，1-step cat hello.txt）").
- Earliest instance, Phase 1.5: all 3 checkpoints predicted `take key` would exit 1 when the environment actually allows it (PEDA_FINAL/archive/phase1_5/phase1_5_complete_report.md:112-116, "所有 3 个 checkpoint 对 `take key` 的预测都是 exit=1（不能拿钥匙）— 错误的。环境实际允许拿钥匙。"), and the post-fix epistemic signal of 0.20 was driven by inventory-state confusion, not genuine environmental complexity (phase1_5_deviation_report.md:57, "The decompose_error fix raised mean_epistemic_error to 0.20, but this was driven by inventory-state confusion, not genuine environmental complexity").
- The confidence penalty (inject noise when `avg_conf > 0.95`) was added as a mitigation but treats the symptom, not the cause (PEDA_CONCLUSION.md:105, "The confidence penalty (inject noise when `avg_conf > 0.95`) was added as a mitigation but treats the symptom, not the cause: the model's predictive distribution does not reflect its actual competence").

## 6.6 Root Cause 5: CPU bottleneck prevented systematic ensemble evaluation

Real-LLM evaluation with Qwen2.5-0.5B on CPU takes ~176s cold start + ~3s per inference call; with 3 checkpoint ensembles and 4-8 candidates per step, each step requires 12-24 model calls, making a single sandbox episode take 10-60+ minutes (PEDA_CONCLUSION.md:107, "Real-LLM evaluation with Qwen2.5-0.5B on CPU takes ~176s cold start + ~3s per inference call. With 3 checkpoint ensembles and 4-8 candidates per step, each agent step requires 12-24 model calls. A single sandbox episode with PEDA can take 10-60+ minutes"). The Phase 3 confirmatory experiment was estimated at 60-120 hours on CPU — infeasible (results/phase3_experiment/report.json, "Without GPU, the experiment is impractical (estimated 60-120 hours total on this CPU)"). Consequences:

- Training e3 on 10,040 transitions timed out after 30 minutes on CPU (PEDA_WORKING_LOG.md:1337-1339, "Attempted to train sandbox_adapter_e3 on the full 10,040-transition dataset (3 epochs): timed out after 30 min." / "Root cause: CPU-only PyTorch inference for Qwen2.5-0.5B + LoRA is too slow for training on this machine").
- A stale latency config (median_ms=4750 vs ~2500 real) forced horizon 1 on CPU (PEDA_FINAL/paper/evidence/phase1.md §4, "Horizon 2 rollout (falls to 1 when latency budget exceeded on CPU); latency config config/phase1_model.json median_ms=4750 stale vs ~2500 real → horizon always 1 on CPU").
- The Phase 1 CPU partial-training comparison used a single checkpoint, making epistemic error identically zero by construction (PEDA_FINAL/archive/phase1/phase1_partial_training_eval_report.md:38, "由于 epistmic_error=0，PEDA 的 EFE 公式退化为 drive_system.apply_to_efe(pragmatic * pragmatic_weight)").
- The project switched to heuristic proxies (model confidence as epistemic proxy, single-checkpoint evaluation) before the theoretically correct ensemble approach could be systematically tested (PEDA_CONCLUSION.md:108, "This bottleneck forced the project to switch to heuristic proxies (model confidence as epistemic proxy, single-checkpoint evaluation) long before the core hypothesis could be properly tested").
- Even the Phase 7 RSSM track deviated from its documented ensemble-of-3 design, scoring actions with single-model prior variance (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md E17.1, "RSSM used single-model prior variance, not ensemble-of-3 | METHODOLOGY MISMATCH").

## 6.7 What survived

Five engineering components are independently validated and reusable:

- **Count-based pair novelty.** The `NovellyExplorer` with (state, action) pair counting achieves optimal exploration in environments with <1000 states (PEDA_CONCLUSION.md:117, "The `NovellyExplorer` with (state, action) pair counting achieves optimal exploration in environments with <1000 states"). The count-driven Phase 8 agent reached 62.2% (28/45) success across 9 tasks on the v2+v4 sandboxes (results/phase8_gpu_run_2026-07-31.md:23-32, per-task rows "| 1 | read_hello | v2 | 5/5 (100%) | 1.2 |" ... "| | **TOTAL** | | **28/45 (62.2%)** | |" — 4 tasks on `peda-sandbox:v2`, 5 on `peda-sandbox:v4`).
  - Where counting succeeds and fails is task-structured: direct reads 100% (read_hello, find_secret, read_welcome, count_measurements, read_changelog_v4), deep-path reads 20% (read_note, find_api_key, find_errors_v4), and count_lines 0% (results/phase8_gpu_run_2026-07-31.md:36-38, "**Direct reads (100%)**: ... success cache enables 1-2 step solves after initial discovery"; "**Deep path reads (20%)**: ... 10-step ceiling exhausted before reaching target file in deep directory"; "**Zero (0%)**: count_lines — wc -l never targets the correct filename").
- **STRIPS action learning.** Learned action schemas reach 45.8% learned vs 31.3% fallback as a candidate hit rate on the action prediction task (PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:17, "| **STRIPS action schemas** | Learned action model | 45.8% hit rate vs 31.3% fallback on v2 sandbox traces |"; raw per-experiment trace not preserved — cited as candidate hit rate, not episode success).
- **Data-driven candidate generation.** The sandbox candidate generator evolved from hardcoded heuristics (v1, 4 candidates) to data-driven enumeration (v2, 65 pairs; v3/v4, 270+ pairs) with zero crashes during migration (PEDA_CONCLUSION.md:121, "The sandbox candidate generator evolved from hardcoded heuristics (v1, 4 candidates) to data-driven enumeration (v2, 65 pairs; v3/v4, 270+ pairs) with zero crashes during migration").
- **Success cache.** One-step solves for seen state-action pairs; this cache provided the mechanism behind Phase 3's positive result (PEDA_CONCLUSION.md:123, "This cache provided the mechanism behind Phase 3's positive result: once PEDA discovers `cat hello.txt` in `/sandbox/projects`, the cache replays it instantly on subsequent episodes"), with 20/20 1-step completions across 4 tasks on repeated (state, action) pairs (PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:19, "| **Success cache (memoization)** | 1-step solver | 20/20 1-step completions across 4 tasks on repeated (state, action) pairs |").
- **Dead-loop immunity.** PEDA maintains dead_loop_rate=0.00 across all conditions while Pragmatic exhibits 0.54-0.90 (results/phase4b_rerun/ANALYSIS_REPORT.md:59-64, table rows "| read_hello | 0.00 | 0.00 | **0.54** |" ... "| read_note | 0.00 | 0.00 | **0.90** |"; "PEDA maintains dead_loop_rate=0.00 across all conditions — **PEDA never dead-loops**, even when it fails to find the target"). The drive system's boredom term and candidate diversity provide reliable dead-loop avoidance even when no epistemic signal exists (PEDA_CONCLUSION.md:125, "The drive system's boredom term and candidate diversity provide reliable dead-loop avoidance, even when no epistemic signal exists").

## 6.8 Lessons

1. **Epistemic signal requires state spaces large enough that the model CAN be uncertain about parts of them.** A 0.5B model with LoRA on 25-200 training examples generalizes perfectly on 5x5 grids and ~65-state sandboxes; the model is too capable for the environment (PEDA_CONCLUSION.md:133, "A 0.5B parameter model with LoRA on 25-200 training examples generalizes perfectly on 5x5 grids and ~65-state sandboxes. The model is too capable for the environment").
2. **JEPA-style forward dynamics train but do not differentiate actions.** The learned uncertainty is a scalar per transition, not a comparative signal across actions; all unexplored transitions are equally uncertain (PEDA_CONCLUSION.md:135, "The JEPA MLP predictor learns to predict next-state embeddings from (state, action) pairs, as shown by decreasing loss curves across all experiments (loss 45 to 15). But the learned uncertainty is a scalar per transition, not a comparative signal across actions").
3. **EFE is dominated by pragmatic value at any practically testable horizon.** At horizon 1-3, the pragmatic term dominates by 3-10x; the epistemic term only changes selection when all candidates are equally unpromising. This is not a fixable hyperparameter issue but a structural property of EFE in goal-directed tasks with small lookahead horizons (PEDA_CONCLUSION.md:137, "the pragmatic term dominates by 3-10x"; "This is not a fixable hyperparameter issue; it is a structural property of EFE in goal-directed tasks with small lookahead horizons").
4. **Counting is surprisingly robust even with stochastic environment elements.** Pair-counting tolerates changing file listings by treating each (cwd, command) pair as an independent counter; it is computationally free (integer increment) and empirically matches or exceeds every learned exploration signal tested (PEDA_CONCLUSION.md:139, "This is simple, computationally free (integer increment), and empirically matches or exceeds every learned exploration signal we tested").
5. **Bootstrap data quality is the critical bottleneck for any learned exploration mechanism.** The Cold Start problem — no model without data, no exploration without model — is not solvable by better exploration algorithms (PEDA_CONCLUSION.md:141, "The Cold Start problem (no model without data, no exploration without model) is not solvable by better exploration algorithms"), and the project spent approximately 50% of total engineering effort on data collection and pipeline infrastructure (PEDA_CONCLUSION.md:141, "The project spent approximately 50% of total engineering effort on data collection and pipeline infrastructure"). Data quality dominates quantity: the 200-transition curated e2 adapter (L1=1.000, L2=0.900, L3=0.550) outperformed the 10,040-transition random+heuristic e3 adapter, which regressed to L1=0.833, L2=0.333, L3=0.133 (PEDA_WORKING_LOG.md:1479, "e2 held-out: L1=1.000 PASS, L2=0.900 PASS, L3=0.550 PASS"; PEDA_WORKING_LOG.md:1481, "e3 held-out: L1=0.833 FAIL, L2=0.333 FAIL, L3=0.133 FAIL"; PEDA_WORKING_LOG.md:1490, "数据质量 > 数量：e3 (10,040 random+heuristic) 退化" — "data quality > quantity: e3 regressed; random data dilutes the completion signal").

## 6.9 Conclusion: a valid negative result

The PEDA hypothesis — that prediction error from an LLM-based World Model can drive autonomous exploration more effectively than baselines in LLM-based agents — is **DISPROVEN** under the conditions tested (PEDA_CONCLUSION.md:147, "The PEDA hypothesis — that prediction error from an LLM-based World Model can drive autonomous exploration more effectively than baselines in LLM-based agents — is **DISPROVEN** under the conditions tested"). Every sub-question failed at its own level: no differential epistemic signal (Q1), no behavioral effect of EFE's epistemic term (Q2), no advantage over count-based novelty (Q3). The one statistically significant result (Phase 3, p=0.0043) is explained without invoking epistemic prediction error — candidate set engineering plus success caching — and did not replicate on any task other than read_hello (see §6.3).

The negative result is a valid scientific conclusion per the research charter:

- (RESEARCH_CHARTER.md:21, "如果任何一个子问题的答案是"否"，整个假设在此条件下不成立——**但这本身就是一个有价值的研究结论**" — "if any sub-question is answered No, the hypothesis does not hold under these conditions — and this itself is a valuable research conclusion").
- (RESEARCH_CHARTER.md:37, "负结果不是项目失败，而是知识。一个诚实记录的负结果比一个人为制造的"成功"更有科学价值" — "a negative result is not project failure but knowledge; an honestly recorded negative result has more scientific value than a manufactured 'success'").
- The charter's success criterion is deeper understanding, not agent autonomy: (RESEARCH_CHARTER.md:79-81, "我们是否对"Active Inference 在 LLM-based Agent 中的可行性"有了比项目开始前更深的理解？" — "have we gained a deeper understanding of Active Inference feasibility in LLM-based agents than before the project started?").
- (PEDA_CONCLUSION.md:158, "**This is a valid scientific result.** It closes one specific research path (LLM-based Active Inference with ensemble/JEPA uncertainty as exploration drive) and constrains the search space for future work").

The empirical answer under the tested conditions is specific and actionable: in state spaces under ~1,000 states, with 0.5B models and LoRA fine-tuning, count-based novelty is the reliable exploration mechanism, and learned epistemic signals add nothing beyond it (PEDA_CONCLUSION.md:145, "under practical conditions (small state spaces <1000 states, 0.5B models, LoRA fine-tuning, CPU-limited inference), count-based novelty is the reliable exploration mechanism. Epistemic prediction error from learned World Models does not produce behaviorally distinguishable exploration — it either equals zero (WM too certain), matches counting at extreme computational cost (JEPA), or is dominated by pragmatic value (EFE collapse at horizon 1-3)").

## 6.10 Limits of the evidence base

The DISPROVEN verdict rests on the reproducible core of the record: E01-E09 (Grid World, sandbox v1/v2, Phase 3 N=20), E11-E12 (Phase 4B FHT-based hits), E14 (pure-epistemic SCR ~0), E17.1 (persisted RSSM/MLP-JEPA 5x5 runs), and E18-E19 (Phase 8 count-driven, 45 episodes per condition). Several secondary configurations could not be independently re-verified and are therefore excluded from the quantitative claims above, per the evidence audit (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md):

- Four JEPA sub-configurations (E13.08-E13.11) have no on-disk source: (CLAIMS_VS_EVIDENCE.md E13, "4 sub-configs: raw JSONL deleted | MISSING | No on-disk source | Exclude from paper").
- Three of five Phase 7 GPU tracks produced no persisted result files: (CLAIMS_VS_EVIDENCE.md E17.2-E17.4, "Goal-JEPA, Curriculum, Random-Maze: NO result files | MISSING | Exclude from quantitative claims; cite as 'code complete, results not preserved'").
- The 10x10 and 20x20 maze results survive only as doc claims — the 10x10 count JSONL is 0 bytes and no 20x20 raw file exists (CLAIMS_VS_EVIDENCE.md E15, "10x10 count JSONL = 0 bytes; numbers survive only in docs"; E16, "No 20x20 raw data"). They are cited above with that caveat and carry no weight beyond the reproducible 5x5 runs.
- Phase 4A self-training block aggregates were recovered from tmux scrollback after the per-episode JSONL was lost before instance termination (PEDA_FINAL/paper/evidence/phase4.md §E10, "Per-episode JSONL LOST"), and its success curve used the subsequently invalidated `success` field; it is therefore reported as "POSITIVE — but success-cache mechanism" (PEDA_CONCLUSION.md:50) and does not contribute to any claim in this Discussion.
- The STRIPS 45.8% vs 31.3% figure is cited strictly as a candidate hit rate on the action prediction task because its raw trace was not preserved (PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md STRIPS, "45.8% learned vs 31.3% fallback | PARTIAL | CHARTER.md:17,35; raw trace missing | Cite as 'candidate hit rate on action prediction task'").

These boundaries describe what the evidence does not contain; they do not reopen the hypothesis. Every claim that carries the negative result — the three No answers, the five root causes, and the superiority of count-based novelty under ~1,000 states — is supported by the reproducible record cited in §6.1-§6.9.

---

## References

[Friston, 2009] K. Friston. The free-energy principle: a rough guide to the brain? Trends in Cognitive Sciences, 13(7):293-301, 2009.
[Friston et al., 2017] K. Friston, T. FitzGerald, F. Rigoli, P. Schwartenbeck, G. Pezzulo. Active inference: a process theory. Neural Computation, 29(1):1-49, 2017.
[Ha & Schmidhuber, 2018] D. Ha and J. Schmidhuber. World models. NeurIPS, 2018.
[Hafner et al., 2020] D. Hafner, T. Lillicrap, J. Ba, M. Norouzi. Dream to control: learning behaviors by latent imagination. ICLR, 2020.
[Hafner et al., 2021] D. Hafner, T. Lillicrap, M. Norouzi, J. Ba. Mastering Atari with discrete world models. ICLR, 2021.
[Hafner et al., 2023] D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. Mastering diverse domains through world models. arXiv:2301.04104, 2023.
[LeCun, 2022] Y. LeCun. A path towards autonomous machine intelligence. OpenReview, 2022.
[Pathak et al., 2017] D. Pathak, P. Agrawal, A. A. Efros, T. Darrell. Curiosity-driven exploration by self-supervised prediction. ICML, 2017.
[Burda et al., 2018] Y. Burda, H. Edwards, A. Storkey, O. Klimov. Exploration by random network distillation. ICLR, 2019.
[Guo et al., 2022] Z. Guo et al. BYOL-Explore: Exploration by bootstrapped prediction. NeurIPS, 2022.
[Wang et al., 2023] G. Wang et al. Voyager: an open-ended embodied agent with large language models. NeurIPS, 2023.
[Yao et al., 2023] S. Yao et al. ReAct: synergizing reasoning and acting in language models. ICLR, 2023.
[Shinn et al., 2023] N. Shinn et al. Reflexion: language agents with verbal reinforcement learning. NeurIPS, 2023.
[Guo et al., 2017] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. ICML, 2017.
[Schmidhuber, 1991] J. Schmidhuber. A possibility for implementing curiosity and boredom in model-building neural controllers. From Animals to Animats, 1991.
