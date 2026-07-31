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

The hypothesis was tested across 17+ controlled experiments spanning 5 environments, using 4 model architectures and ~2,000 total evaluation episodes (CONCLUSION:22, "the hypothesis — that Expected Free Energy (EFE) minimization with epistemic uncertainty drives more effective exploration than pragmatic heuristics — was tested across 17+ controlled experiments spanning 5 environments (Grid World, Sandbox v1/v2/v3/v4, Grid Maze, Giant Maze, TextWorld), using 4 model architectures (Qwen2.5-0.5B with LoRA, JEPA MLP predictors, RSSM, STRIPS action models), with ~2,000 total evaluation episodes"). All three charter sub-questions answer **No** under the tested conditions (CONCLUSION:24, "All three charter sub-questions answer **No** under the tested conditions:"). The reliable exploration mechanism turned out to be count-based pair novelty, not epistemic prediction error from learned World Models, in state spaces under ~1,000 states (CONCLUSION:30, "Count-based pair novelty, not epistemic prediction error from learned World Models, is the reliable exploration mechanism in state spaces under ~1,000 states. This negative result is a valid scientific conclusion per the research charter, and constitutes genuine knowledge about the feasibility of Active Inference with LLM-based agents"). The PEDA hypothesis is therefore **DISPROVEN** under the conditions tested (CONCLUSION:147, "The PEDA hypothesis — that prediction error from an LLM-based World Model can drive autonomous exploration more effectively than baselines in LLM-based agents — is **DISPROVEN** under the conditions tested:").

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
