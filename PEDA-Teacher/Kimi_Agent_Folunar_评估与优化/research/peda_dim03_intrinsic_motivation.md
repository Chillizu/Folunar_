## Dimension: Intrinsic Motivation and Curiosity-Driven Learning

---

## 1. ICM & RND Analysis

### 1.1 Intrinsic Curiosity Module (ICM) - Pathak et al. (2017)

**Core Mechanism:**
ICM formulates curiosity as the error in an agent's ability to predict the consequences of its own actions in a learned feature space [^33^][^607^]. It consists of three submodules:
- **Feature Extractor**: Encodes high-dimensional states s_t and s_{t+1} into low-dimensional feature vectors phi(s_t) and phi(s_{t+1})
- **Inverse Dynamics Model**: Predicts the action a_t given phi(s_t) and phi(s_{t+1}) - trained to model only features controllable by the agent
- **Forward Dynamics Model**: Predicts phi(s_{t+1}) given phi(s_t) and a_t

The intrinsic reward is computed as the squared L2 prediction error of the forward model:
```
r^i_t = ||phi(s_{t+1}) - phi_hat(s_{t+1})||_2^2
```

**Key Innovation**: The inverse model ensures the learned feature space encodes only aspects of the environment that are controllable by the agent, filtering out unpredictable elements [^607^][^33^].

**Limitations**:
- **Noisy TV Problem**: ICM can get stuck at unlearnable, stochastic transitions (e.g., a TV displaying random noise) because the prediction error remains permanently high, providing perpetual intrinsic reward [^53^][^610^]
- Per-step prediction error structure is blind to cumulative model improvement across the full history [^52^]
- Does not cleanly separate aleatoric (irreducible) from epistemic (reducible) prediction error [^53^]

### 1.2 Random Network Distillation (RND) - Burda et al. (2018)

**Core Mechanism:**
RND uses a simpler approach with two networks: a fixed randomly initialized target network T(s) and a trainable predictor network P(s) [^612^][^613^]. The predictor is trained to match the random outputs of the target network. The intrinsic reward is the mean squared error:

```
b_RND(s_t) = ||P(s_t) - T(s_t)||_2^2
```

The intuition is that for states similar to previously encountered ones, the error will be low; for novel states, the error will be high [^72^]. Since the target is deterministic and state-dependent, the reward decays to zero as a state is visited repeatedly - effectively a "soft count-based bonus" immune to irreducible noise [^52^].

**Large-Scale Study (Burda et al., 2018)**:
RND was evaluated across 54 standard benchmark environments in the first large-scale study of purely curiosity-driven learning (without extrinsic rewards) [^171^][^690^]. Key findings:
- RND achieved state-of-the-art performance on Montezuma's Revenge (finding all 24 rooms) without demonstrations [^170^]
- Random features are sufficient for many game benchmarks, but learned features generalize better (e.g., to novel game levels) [^702^]
- Combined with PPO, RND achieved new SotA for Gravitar and Montezuma's Revenge [^170^]

**Limitations**:
- Lacks a robust mathematical foundation compared to count-based techniques [^27^]
- Bonus inconsistency during initial training stages [^27^]
- Struggles to precisely represent data distribution as training progresses [^27^]
- Can be attracted to dangerous states simply because they are difficult to predict ("dancing with skulls" phenomenon in Montezuma's Revenge) [^170^]
- In high-dimensional robotic observations, RND struggles to provide meaningful exploration signals [^76^]

---

## 2. Noisy TV Problem

### 2.1 Detailed Explanation

The Noisy TV problem is the canonical failure mode of prediction-error-based curiosity methods [^603^][^610^]. When an agent encounters an inherently unpredictable element in the environment (e.g., a TV displaying random static, coin flips, or stochastic weather patterns), the forward model can never learn to accurately predict the outcome. Since the prediction error remains permanently high, the agent receives a perpetual intrinsic reward and becomes trapped, unable to explore other parts of the environment.

**Why it happens:**
- Prediction-error curiosity conflates two types of uncertainty [^53^]:
  - **Epistemic uncertainty**: Lack of knowledge that can be reduced by learning
  - **Aleatoric uncertainty**: Inherent randomness that cannot be eliminated
- Standard curiosity methods (ICM, V1) reward both equally, causing agents to seek out irreducible noise [^53^]

**Manifestations:**
- Agent becomes a "couch potato" with a TV and remote control [^715^]
- Attracted forever to the most noisy states with unpredictable outcomes
- Agent may interact with potentially dangerous objects simply because they are difficult to predict [^170^]

### 2.2 Why FEP Can Avoid This Problem

The Free Energy Principle (FEP) and Active Inference framework provide a principled solution to the Noisy TV problem through the concept of **Expected Free Energy (EFE)** [^667^][^698^]:

**Key distinctions:**
1. **Epistemic vs. Aleatoric separation**: EFE naturally decomposes into epistemic value (information gain about hidden states) and pragmatic value (goal-directed utility) [^667^][^698^]. Epistemic value is about reducing *reducible* uncertainty - it does not reward encountering irreducible noise.

2. **Information gain vs. Prediction error**: Prediction error is simply the mismatch between prediction and observation. Information gain (Bayesian surprise) measures how much the agent's *beliefs* are updated by new observations [^706^]. A Noisy TV provides high prediction error but zero information gain because no learning occurs.

3. **Natural resolution of exploration-exploitation**: Under FEP, "epistemic value is maximized until there is no further information gain, after which exploitation is assured through maximization of extrinsic value" [^667^]. The agent naturally loses interest in unpredictable stimuli when they no longer provide model updates.

4. **Learning progress as intrinsic motivation**: Schmidhuber's artificial curiosity (1991) rewards learning *progress* (improvement in prediction accuracy), not raw prediction error [^53^][^713^]. An inherently unpredictable TV provides no learning progress, so no reward is generated.

---

## 3. Advanced Methods

### 3.1 BYOL-Explore (Guo et al., 2022, DeepMind)

**Core Mechanism:**
BYOL-Explore (Bootstrap Your Own Latent - Explore) learns a world representation, dynamics model, and exploration policy all together by optimizing a **single prediction loss** in latent space with no additional auxiliary objective [^627^][^62^]. It builds on BYOL's self-supervised learning approach:
- Uses an online network and a target network (moving average of the online network)
- Predicts the latent representation of future states
- Intrinsic reward comes from the prediction error in this learned latent space

**Key Results:**
- Solved majority of tasks in DM-HARD-8 (challenging partially-observable continuous-action 3D environments) purely through intrinsic reward augmentation, where prior work required human demonstrations [^627^]
- Achieved superhuman performance on the ten hardest exploration games in Atari [^627^]
- Much simpler design than other competitive agents (single prediction loss vs. multiple auxiliary objectives)

**Advantage over ICM/RND**: The bootstrapped prediction approach avoids some pitfalls of fixed random targets and learns more semantically meaningful representations.

### 3.2 NovelD - Novelty Difference (Zhang et al., 2021)

**Core Mechanism:**
NovelD encourages exploration at the boundary between explored and unexplored regions using the difference in novelty between consecutive states [^72^][^630^]:

```
b_NovelD(s_t, a, s_{t+1}) = [b_RND(s_{t+1}) - c * b_RND(s_t)]_+ * I[N_e(s_{t+1}) = 1]
```

Where:
- First term: global novelty bonus (difference in RND bonuses between consecutive steps)
- Second term: episodic novelty bonus (only rewards first visit to a state within an episode)
- This encourages the agent to explore in a breadth-first search manner [^629^]

**Key Innovation:** The novelty difference drives the agent toward the boundary of explored regions, maximizing the chance of crossing into unknown space [^635^]. Previously claimed state-of-the-art performance on some NetHack environments [^75^].

### 3.3 E3B - Exploration via Elliptical Episodic Bonuses (Henaff et al., 2022)

**Core Mechanism:**
E3B extends count-based episodic bonuses to continuous state spaces [^69^][^618^]. It encourages an agent to explore states that are diverse under a learned embedding within each episode:

```
b_E3B(s_t) = phi(s_t)^T * [sum_{i=t_0}^{t-1} phi(s_i) phi(s_i)^T + lambda*I]^{-1} * phi(s_t)
```

Where phi is a feature extractor learned using an inverse dynamics model (similar to ICM), and the embedding captures controllable aspects of the environment [^69^].

**Key Properties:**
- Reduces to inverse episodic counts if phi is a one-hot encoding [^72^]
- Models the complete set of state-visitation frequencies over the state space, not just the most recent state [^622^]
- Episodic bonuses are reset at the beginning of each episode, preventing reward vanishing [^610^]

**Results:**
- Set new state-of-the-art across 16 challenging tasks from MiniHack suite without task-specific inductive biases [^69^][^77^]
- Matches existing methods on sparse reward, pixel-based VizDoom environments
- Outperforms existing methods in reward-free exploration on Habitat [^77^]
- ICM and RND fail to provide good exploration signal in procedurally generated environments, while E3B succeeds [^620^]

**Limitation**: E3B's episodic nature means it doesn't scale well when each state is rarely seen more than once in complex environments [^72^].

### 3.4 MaxInfoRL - Information Gain Maximization (Sferrazza et al., 2024)

**Core Mechanism:**
MaxInfoRL steers exploration toward informative transitions by maximizing information gain about the underlying task [^189^][^184^]. It modifies Boltzmann exploration with two critics:
- Q_extrinsic: for the external reward
- Q_intrinsic: for the information gain bonus

```
r_intrinsic = I(s'; f* | s, a)  # information gain about the MDP
```

The policy trades off value maximization with entropy over states, rewards, and actions [^184^]. It uses an ensemble of forward dynamics models to estimate information gain and features auto-tuning of the temperature parameter for the information gain bonus.

**Key Results:**
- Achieves sublinear regret in the multi-armed bandit setting [^189^]
- Consistently outperforms baselines across state-based and visual control benchmarks
- Combines information gain with any off-policy RL algorithm (SAC, REDQ, DrQ, DrQv2) [^184^]
- Obtains highest performance in challenging visual control tasks

**Fundamental Difference from Prediction Error:**
Information gain measures how much an agent's *beliefs* change after an observation. Unlike prediction error, it naturally separates epistemic from aleatoric uncertainty - an observation of an inherently stochastic process provides high prediction error but zero information gain once the stochasticity is known.

### 3.5 NGU - Never Give Up (Badia et al., 2020)

**Core Mechanism:**
NGU combines episodic and lifelong novelty to generate intrinsic rewards that don't vanish over time [^704^][^705^]:
- **Epodic novelty**: Uses episodic memory with pseudo-counts to encourage diverse state visitation within each episode
- **Lifelong novelty**: Computed via RND to promote exploration across episodes
- Uses inverse dynamics model embeddings for k-nearest neighbor lookups, biasing novelty toward controllable aspects

**Key Innovation**: Learns a family of policies parameterized by beta (exploration strength) that make different trade-offs between exploration and exploitation using the UVFA framework [^705^]. First algorithm to achieve non-zero rewards in Pitfall! without demonstrations or hand-crafted features [^608^].

### 3.6 RIDE - Rewarding Impact-Driven Exploration (Raileanu & Rocktaschel, 2020)

**Core Mechanism:**
RIDE uses an episodic novelty bonus that is the product of two terms [^96^][^664^]:
- Count-based reward: 1/sqrt(N_e(s_t)) (discourages revisiting states)
- State embedding difference: ||phi(s_{t+1}) - phi(s_t)||_2 (rewards actions causing significant changes)

**Key Property**: Unlike ICM and RND, RIDE's intrinsic reward does not diminish during training and rewards interacting with controllable objects substantially more [^664^].

---

## 4. Prediction Error vs Information Gain: Fundamental Distinction

### 4.1 Core Difference

| Aspect | Prediction Error (ICM/RND) | Information Gain (MaxInfoRL/VIME) |
|--------|---------------------------|-----------------------------------|
| **What it measures** | Mismatch between predicted and actual observation | Reduction in uncertainty about hidden states/model parameters |
| **Aleatoric noise** | Rewards both learnable and irreducible uncertainty | Only rewards reducible (epistemic) uncertainty |
| **Noisy TV robustness** | Fails - gets attracted to permanent noise | Robust - no information gain from unlearnable patterns |
| **Computational cost** | Low - single forward pass | High - requires Bayesian inference or ensemble methods |
| **Theoretical grounding** | Heuristic | Principled (FEP, Bayesian surprise) |

### 4.2 Why Information Gain is Better

1. **Separates learnable from unlearnable**: Information gain only rewards situations where the agent's model actually improves [^625^][^670^]. A Noisy TV has high prediction error but zero information gain.

2. **Expected information gain drives curiosity**: Curiosity reflects *expected information gain* - prediction errors from unexpected events provide an estimate of how much new information is expected to minimize future prediction errors [^670^].

3. **Connects to principled frameworks**: Information gain naturally emerges from the Free Energy Principle's Expected Free Energy decomposition [^698^][^667^].

4. **Better exploration efficiency**: Methods like MaxInfoRL and VIME (Variational Information Maximizing Exploration) that directly optimize information gain achieve better exploration efficiency than raw prediction error methods [^189^].

### 4.3 Learning Progress (Schmidhuber's Approach)

Schmidhuber (1991) proposed rewarding the *learning progress* (improvement in prediction accuracy) rather than raw prediction error [^53^][^713^]:
- **Curiosity V1**: Reward proportional to prediction error r_t = ||theta_t(s_t, a_t) - s_{t+1}|| [^52^]
- **Curiosity V2**: Reward proportional to one-step improvement in prediction error [^52^]
- **Key insight**: An optimal curious agent's interest lies "in the narrow corridor between what is simply too compressible and therefore uninteresting and boring, and what is not compressible at all because of a lack of regularity" [^716^]

---

## 5. FEP Framework: Unification of Intrinsic Motivation

### 5.1 Expected Free Energy (EFE) Decomposition

The Free Energy Principle provides a principled unification of exploration and exploitation through Expected Free Energy minimization [^667^][^698^]:

```
EFE = Epistemic Value + Pragmatic Value
     = Information Gain + Expected Utility
     = Exploration Drive + Exploitation Drive
```

**Epistemic Value (Intrinsic Motivation):**
- Information gain about hidden states given observations [^698^]
- Drives active sensing and perceptual inference [^625^]
- Corresponds to *curiosity* - seeking to improve the predictive model [^698^]
- Naturally resolves uncertainty about environmental contingencies

**Pragmatic Value (Extrinsic Motivation):**
- Expected utility of outcomes given prior preferences/goals [^667^]
- Drives goal-directed behavior
- Corresponds to *exploitation* - leveraging reliable expectations

### 5.2 How FEP Unifies Intrinsic Motivation

Friston's framework mathematically formalizes various forms of uncertainty that lead to different types of exploration [^625^]:

1. **Uncertainty about hidden states given a policy** -> Active sensing/perceptual inference (improving world state estimation)
2. **Uncertainty about policies in terms of expected future states** -> Epistemic exploration/learning (improving predictive model)
3. **Uncertainty about model structure itself** -> Structure learning and insight (finding new abstractions)
4. **Uncertainty about self-evaluation of goal competences** -> Goal-driven curiosity [^625^]

### 5.3 Active Inference vs. Standard RL

| Aspect | Standard RL | Active Inference |
|--------|------------|-----------------|
| Exploration | Separate heuristic (epsilon-greedy, noise) | Emerges naturally from epistemic value |
| Reward maximization | Explicit objective | Part of pragmatic value |
| Intrinsic motivation | Added bonus | Built into EFE decomposition |
| Exploration-exploitation trade-off | Ad hoc parameter tuning | Natural balance via precision |

**Critical insight**: "Minimizing expected free energy is therefore equivalent to maximizing extrinsic value or expected utility, while maximizing information gain or intrinsic value" [^667^]. The exploration-exploitation dilemma is resolved: "Epistemic value is maximized until there is no further information gain, after which exploitation is assured" [^667^].

### 5.4 Pragmatic Curiosity

Recent work (Li et al., 2026) introduced "pragmatic curiosity" that implements AIF as an acquisition rule [^628^]:

```
alpha(x | D_t) = beta_t * I(s; (x,y) | D_t) - E[h(y | D_t)]
```

Where beta_t controls the exchange rate between information gain and pragmatic cost. This makes the design tension explicit: too little curiosity risks myopic exploitation; too much risks over-exploration [^628^].

---

## 6. Real-World and Structured Environment Applications

### 6.1 Procedurally Generated Environments

**MiniHack/NetHack**: MiniHack is a sandbox framework for designing RL environments based on NetHack - one of the hardest roguelikes [^734^][^737^]. It provides:
- Text-based observations with rich entities and dynamics
- Procedurally generated dungeons
- Large action space (75+ actions for skill tasks)
- Sparse rewards requiring deep exploration

E3B set state-of-the-art across 16 MiniHack tasks [^77^]. NGU and RIDE also show strong performance [^608^]. The Language Wrapper translates observations into text representations [^734^].

### 6.2 Symbolic/Structured Environments

**Symbolic Equation Solving**: Curiosity-based methods have been applied to solving algebraic equations [^693^][^696^]:
- PPO with RND achieved 0.80 success rate on complex equations c+d/(ax+b)
- PPO with NGU and ICM achieved 0.50 success rate
- Curiosity-based exploration was required to solve non-elementary equations
- A* algorithm failed on complex equations [^693^]

**Key insight**: For structured, discrete environments (like expression trees or command-line interfaces), count-based methods (RND, NGU) often outperform prediction-error methods because:
1. State space is well-defined
2. Episodic memory is effective
3. Information gain can be computed more reliably

### 6.3 Real-World Robotics

**Challenges** [^76^][^73^]:
- **Prediction model overfitting**: Curiosity rewards overfit to familiar states
- **Scalability to long horizons**: Intrinsic rewards decay over extended episodes
- **Reward interference**: Intrinsic signals interfere with sparse extrinsic rewards
- **Sim-to-real gap**: Benchmarks show poor transfer to real robots

**Successful applications**:
- Visual navigation with curiosity-driven exploration [^76^]
- Robotic manipulation with CMPO (Curiosity Model Policy Optimization) [^695^]
- Autonomous goal setting for developmental robotics [^625^]

### 6.4 Applicability to Linux/Command-Line Environments

For structured environments like Linux command lines, the analysis suggests:

**Advantages for curiosity-driven methods:**
- Discrete action space (commands are well-defined)
- Structured state space (text output, directory structures)
- Deterministic transitions (mostly)
- Sparse rewards (task completion signals only)

**Recommended approaches:**
1. **E3B/NovelD**: Best for episodic exploration with diverse state visitation
2. **MaxInfoRL**: Best for principled information gain estimation
3. **NGU**: Best for maintaining persistent exploration without reward vanishing
4. **RND**: Simplest to implement, good baseline

**Key considerations:**
- Text observations require appropriate encoding (language models, token embeddings)
- Episodic bonuses work well because command-line sessions have natural episode boundaries
- The structured nature of command-line environments means information gain can be computed over program execution traces

---

## 7. Summary and Key Takeaways

### 7.1 Evolution of Curiosity Methods

```
Schmidhuber (1991) -> ICM (2017) -> RND (2018) -> NGU/RIDE (2020) -> E3B (2022) -> BYOL-Explore (2022) -> MaxInfoRL (2024)
Prediction Error    -> Feature Space -> Soft Count  -> Episodic+Global -> Elliptical -> Bootstrap Latent -> Information Gain
```

### 7.2 Key Insights

1. **Prediction error is a proxy, not the goal**: True curiosity should reward *learning progress* (model improvement), not raw prediction error [^53^][^713^]

2. **Episodic bonuses are essential**: For complex environments, episodic novelty bonuses (E3B, NGU, NovelD) outperform global bonuses alone [^72^][^610^]

3. **FEP provides a unifying framework**: Active Inference naturally unifies exploration (epistemic value) and exploitation (pragmatic value) within a single objective [^667^][^698^]

4. **Information gain > Prediction error**: Information gain-based methods (MaxInfoRL, VIME) are theoretically principled and empirically superior to raw prediction error methods [^189^][^184^]

5. **Structured environments favor count-based methods**: In text/symbolic environments (MiniHack, equation solving), count-based and episodic methods (RND, NGU, E3B) tend to outperform pure prediction-error methods [^693^][^620^]

6. **The Noisy TV problem reveals a deep limitation**: Only frameworks that explicitly separate epistemic from aleatoric uncertainty (FEP, information gain) can fundamentally solve this problem

---

## Sources

[^33^] https://arxiv.org/pdf/2109.11052v1.pdf - ICM detailed analysis
[^52^] https://arxiv.org/html/2604.18701v1 - Curiosity-Critic: Cumulative Prediction Error Improvement
[^53^] https://arxiv.org/html/2604.18701v3 - Schmidhuber curiosity formulations and limitations
[^62^] https://hal.science/hal-05413284/document - BYOL-Explore paper
[^67^] https://arxiv.org/html/2402.16801v2 - Craftax: BYOL-Explore and E3B baselines
[^69^] https://arxiv.org/pdf/2210.05805.pdf - E3B: Exploration via Elliptical Episodic Bonuses
[^72^] https://ar5iv.labs.arxiv.org/html/2306.03236 - Global and Episodic Bonuses for Exploration
[^73^] https://milvus.io/ai-quick-reference/what-are-curiositydriven-exploration-methods - Curiosity-driven exploration overview
[^75^] https://arxiv.org/html/2310.00166 - Intrinsic Motivation from AI Feedback (NovelD comparison)
[^76^] https://papersflow.ai/research/topics/reinforcement-learning-in-robotics/curiosity-driven-exploration-in-rl - Curiosity in robotics
[^77^] https://arxiv.org/abs/2210.05805 - E3B paper
[^96^] https://arxiv.org/pdf/2306.03236.pdf - NovelD, RIDE, E3B formulas
[^164^] https://milvus.io/ai-quick-reference/what-is-intrinsic-motivation-in-reinforcement-learning - Intrinsic motivation overview
[^170^] https://v1.endtoend.ai/slowpapers/rnd/ - RND detailed analysis
[^171^] https://arxiv.org/abs/1808.04355 - Large-Scale Study of Curiosity-Driven Learning
[^184^] https://arxiv.org/html/2412.12098v1 - MaxInfoRL: Information Gain Maximization
[^189^] http://arxiv.org/abs/2412.12098v1 - MaxInfoRL abstract
[^603^] https://arxiv.org/html/2412.04775v1 - TeCLE: Noisy TV robustness
[^607^] https://www.mdpi.com/2077-1312/14/1/70 - ICM for Unmanned Surface Vehicle Control
[^608^] https://arxiv.org/html/2501.12627v1 - Hybrid Intrinsic Reward Model (NGU, E3B)
[^609^] https://arxiv.org/html/2501.11463v2 - Curiosity-Driven RL from Human Feedback
[^610^] https://arxiv.org/html/2405.19548v2 - RLeXplore: Intrinsically-Motivated RL
[^612^] https://arxiv.org/html/2502.07279v1 - Exploratory Diffusion Policy (ICM, RND, LBS)
[^613^] https://arxiv.org/pdf/2604.04648 - RND and ICM for OOD detection
[^618^] https://arxiv.org/pdf/2402.03972.pdf - E3B in Multi-Agent RL
[^620^] https://arxiv.org/html/2310.18144v4 - Improving Intrinsic Exploration with SOFE
[^622^] https://arxiv.org/html/2503.18980v1 - CAE: Critic as Explorer
[^625^] https://arxiv.org/pdf/1802.10546.pdf - Computational Theories of Curiosity-Driven Learning
[^627^] https://arxiv.org/pdf/2206.08332 - BYOL-Explore: Exploration by Bootstrapped Prediction
[^628^] https://arxiv.org/pdf/2602.06029 - Active Inference Bridges Learning and Optimization
[^629^] https://ar5iv.labs.arxiv.org/html/2304.10770 - DEIR: Discriminative-Model-Based Episodic Intrinsic Rewards
[^630^] https://ar5iv.labs.arxiv.org/html/2301.02083 - Self-Motivated Multi-Agent Exploration
[^635^] https://arxiv.org/html/2503.18234v1 - KEA: Keeping Exploration Alive
[^637^] https://proceedings.neurips.cc/paper_files/paper/2022/file/ced0d3b92bb83b15c43ee32c7f57d867-Paper-Conference.pdf - BYOL-Explore NeurIPS 2022
[^664^] https://arxiv.org/abs/2002.12292 - RIDE: Rewarding Impact-Driven Exploration
[^667^] https://www.fil.ion.ucl.ac.uk/~karl/Active%20inference%20and%20epistemic%20value.pdf - Active Inference and Epistemic Value (Friston)
[^670^] https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/prediction-error-minimization-as-a-common-computational-principle-for-curiosity-and-creativity/50C2809278B520BD06E75EAD4658F7D0 - Prediction error minimization for curiosity and creativity
[^691^] https://arxiv.org/pdf/1906.10538 - Expected Free Energy and cultural psychology
[^693^] https://www.arxiv.org/pdf/2510.17022 - Curiosity-driven RL for symbolic equation solving
[^695^] https://www.frontiersin.org/articles/10.3389/fnbot.2024.1376215/full - Curiosity model policy optimization for robotic manipulator
[^698^] https://publish.obsidian.md/active-inference/knowledge_base/citations/friston_2017_curiosity - Active Inference, Curiosity and Insight (Friston et al., 2017)
[^700^] https://zilliz.com/ai-faq/what-is-intrinsic-motivation-in-reinforcement-learning - Intrinsic motivation in RL FAQ
[^701^] https://www.emergentmind.com/topics/expected-free-energy-efe-minimization - Expected Free Energy Minimization
[^702^] https://github.com/opendilab/awesome-exploration-rl - Awesome Exploration Methods in RL
[^703^] https://arxiv.org/html/2606.20658v1 - Expected Free Energy-based Planning as Variational Inference
[^706^] https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00005/full - Information Theoretic Characterization of Uncertainty
[^713^] https://people.idsia.ch/~juergen/artificial-curiosity-since-1990.html - Schmidhuber: Artificial Curiosity & Creativity Since 1990-91
[^715^] https://www.andrew.cmu.edu/course/10-703/slides/Lecture_exploration.pdf - CMU: Exploration in RL (Noisy TV explanation)
[^716^] https://direct.mit.edu/isal/proceedings-pdf/ecal2013/25/997/1901729/978-0-262-31709-2-ch148.pdf - Intrinsic motivation computational models
[^734^] https://github.com/facebookresearch/minihack - MiniHack framework
[^734^] https://minihack.readthedocs.io/en/latest/envs/index.html - MiniHack Environment Zoo
