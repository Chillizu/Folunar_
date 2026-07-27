# WATCHDOG.md — PEDA Project Guardian Rules

This file provides project-specific guidance for the oh-my-pi advisor.
The advisor reviews the primary agent's work each turn and may inject
concise advice at severity levels: nit, concern, or blocker.

See: https://github.com/can1357/oh-my-pi/blob/main/docs/advisor-watchdog.md

---

## Project Context

PEDA (Predictive-Error-Driven Autonomous Agent) is an experimental project
trying to replace user prompts with internal prediction error as the driving
signal for an AI agent. It learns from the failures of its predecessor
(Folunar_ / Trahexa). This WATCHDOG.md encodes those lessons so the advisor
can help prevent recurrence.

**If any rule conflicts with "making progress quickly", the rule wins.**
PEDA's biggest risk is not moving too slowly — it is repeating Folunar_'s
mistake of using planning/documentation/code-volume as a proxy for actual
progress.

---

## Blocker Rules (STOP — advisor must emit blocker severity)

### B1: Phase advancement without hypothesis validation

**Trigger**: The primary agent declares a Phase (1, 1.5, 2, 3) "complete",
"implemented", "done", or marks it complete in AGENTS.md — but the
corresponding go/no-go criteria have not been experimentally verified.

**Why**: Folunar_ declared 20 phases "implemented" across 142 commits
without ever validating whether the core idea worked. PEDA must not repeat
this. Phase advancement is a scientific decision, not a documentation update.

**Correct behavior**: If go/no-go criteria are not met, the agent must stay
in the current Phase and run the missing validation experiments. Document the
results (pass or fail), not the completion status.

**Reference**: `peda_reflection_v11.md` — Problem 5 "过早commit/push"

---

### B2: Fabricating uncertainty / injecting fake data into stubs

**Trigger**: The primary agent modifies a stub, mock, or test fixture to
inject artificial randomness, uncertainty, or "realistic noise" so that
tests pass or metrics look better.

**Why**: This is equivalent to data falsification. A stub's purpose is to
validate code path correctness, not to simulate realistic behavior. If the
agent needs realistic model behavior to validate the core hypothesis, it must
use a real (even if small/cheap) model — not fake the uncertainty.

**Correct behavior**: Stubs stay deterministic. If deterministic stubs cause
certain metrics to fail predictably, document those as "expected with stubs"
and create a follow-up task to validate with a real model. Never modify a
stub to make a metric pass.

**Reference**: `peda_reflection_v11.md` — Problem 2 "Stub模式陷阱"

---

### B3: Adding modules without passing the module review gate

**Trigger**: A new Python module or file is added without satisfying ALL
three of the following:

1. It directly helps the World Model predict more accurately, OR
2. There is a published paper or established open-source project proving the
   technique works for similar problems, OR
3. The agent has attempted to implement the functionality using existing
   modules and demonstrated why it is infeasible.

**Why**: Folunar_ grew to 40+ modules, many <150 lines, many just for
"completeness". This created maintenance burden and obscured what actually
mattered. PEDA's core is 5 modules; everything else must earn its place.

**Correct behavior**: Before creating a new file, the agent must write a
brief justification (can be a comment in the commit message) citing which of
the 3 gates it passes. If none, do not create the file.

**Reference**: `peda_reflection_v11.md` — Problem 3 "模块膨胀"

---

### B4: Creating new PLAN/ARCH documents instead of updating existing ones [DEMOTED: Blocker → Concern]

> **Update 2026-07-07 (GLM-5.2 follow-up)**: Demoted from Blocker to Concern.
> In exploration phases, direction pivots are normal. A new focused document
> is often clearer than appending to an outdated monolith. This rule still
> applies during validation / write-up phases.

**Trigger**: A new file matching `PLAN_*.md`, `ARCH_*.md`, or similar is
created, OR an existing document grows by >50% without a corresponding code
change of equal magnitude.

**Why**: Folunar_ produced ~30,000 words of planning documents vs ~3,700
lines of active code. This created the illusion of progress. PEDA must
measure progress by validated hypotheses, not document word count.

**Correct behavior**: If a new plan is needed, append it to the existing
`PLAN.md` or `ARCHITECTURE.md`. Do not create new files. If the existing
doc is too large, that is a signal to delete, not to split.

**Reference**: `peda_reflection_v11.md` — Problem 4 "计划文档通货膨胀"

---

### B5: Insufficient sample size claimed as hypothesis validation

**Trigger**: The primary agent runs an experiment with <5
episodes/condition (or <30 total observations) and uses the results to
declare a hypothesis "validated", "confirmed", or "proven" — or to make a
go/no-go decision about Phase advancement.

**Why**: Phase 1 partial-training pilot showed a strong signal (PEDA 2
steps vs pragmatic_only 20 steps failure) with only 1 episode/condition.
The agent was tempted to treat this as confirmation. But 1 episode cannot
distinguish signal from luck. Statistical inference requires enough data
to rule out randomness. PEDA is a scientific project, not a demo.

**Correct behavior**: 
- Pilot (exploratory): 1-3 episodes, used to discover if an effect might
  exist. Results are directional only, never decisive.
- Confirmatory (validation): ≥10 episodes/condition, used to test a
  pre-registered hypothesis with a pre-defined success threshold.
- Never mix the two: a pilot result, no matter how strong, does NOT
  replace a confirmatory experiment.
- If hardware limits sample size, state the statistical uncertainty
  explicitly (e.g., "N=3, directional signal only, p-value not computed").

**Reference**: `PHASE1_PARTIAL_EVALUATION.md` — "统计显著性分析" section

---

### B6: Cherry-picking experimental conditions to fit desired outcomes

**Trigger**: The primary agent changes an experimental condition
(e.g., train_fraction, pragmatic_weight, grid size, model size) AFTER
seeing initial results, and the change direction makes the results look
better rather than testing a pre-registered hypothesis. OR: the agent
reports only a subset of conditions where results were favorable while
omitting unfavorable conditions.

**Why**: Lowering train_fraction from 0.5 to 0.25 because g1_test_set
was too high (>0.90) is a valid scientific decision — it increases
experimental difficulty to create a meaningful test. But doing so without
a pre-registered protocol, or repeatedly adjusting until results "look
good", is p-hacking. The line between legitimate protocol refinement and
cherry-picking is thin and must be explicitly defended.

**Correct behavior**: 
- Before running experiments, pre-register the experimental conditions
  (train_fraction, grid size, episode count, success thresholds) in the
  evaluation script or a brief protocol note.
- If a condition must be changed mid-experiment, document the reason
  ("g1_test_set=0.95, need more uncertainty") and re-run ALL conditions
  with the new parameter — do not selectively re-run only the ones that
  were previously unfavorable.
- Report ALL results, including null and negative results. A negative
  result that is well-controlled is more valuable than a positive result
  that is cherry-picked.

**Reference**: `PHASE1_PARTIAL_EVALUATION.md` — "pilot vs confirmatory" distinction

---

### B7: Environment-model mismatch causing zero epistemic signal [DEMOTED: Blocker → Concern]

> **Update 2026-07-07 (GLM-5.2 follow-up)**: Demoted from Blocker to Concern.
> Hard-blocking judgment calls stifles boundary probing. The spirit of this
> rule ("recognize when to pivot") remains, but the advisor should counsel
> rather than block. Use the "3 Questions" framework (see end of doc) to
> evaluate whether continued attempts are justified.

**Trigger**: The primary agent trains a World Model on an environment and
achieves near-perfect out-of-distribution accuracy (g1 > 0.90) with minimal
training data (<25% of state-action space). Despite this, the agent
continues attempting to make the environment work by further lowering
train_fraction, adding epochs, or tuning hyperparameters — rather than
recognizing that the environment is too simple for the model.

**Why**: Phase 1 attempted Grid World validation with 0.5B Qwen2.5.
Results: 25% train → g1=0.87; 10% train + 3 epochs → g1=1.0. The
agent spent multiple rounds trying 25% → 10% → 3 epochs before finally
accepting that 5×5 grid is too simple for 0.5B. This is not a failure
of the hypothesis — it is a failure of the experimental design.

**Correct behavior**: 
- If g1 > 0.90 with <50% training data after 1-2 attempts, the
  environment is too simple for the model. Do not try a third time.
- Accept the finding: "Environment X does not create enough uncertainty
  for Model Y under current conditions."
- Pivot immediately to a more complex environment (TextWorld, busybox
  sandbox, larger grid with obstacles) rather than further reducing
  train_fraction.
- Do not view this as "giving up" — it is correcting an experimental
  design flaw. The engineering infrastructure validated by the simple
  environment still has value.

**Reference**: `PHASE1_PARTIAL_EVALUATION.md` — "实际路径 vs PEDA v1.1 计划" section

---

### B8: "Just one more try" death spiral

**Trigger**: The primary agent has attempted the same experimental
approach ≥3 times with progressively more extreme parameters (e.g.,
train_fraction: 0.5 → 0.25 → 0.10 → 0.05; pragmatic_weight: 3.0 →
1.0 → 0.5) without achieving the desired result. The agent continues to
propose further parameter adjustments rather than acknowledging the
approach may be fundamentally unsuitable.

**Why**: Phase 1's Grid World saga: 25% train fraction → 10% → 3 epochs
→ each attempt took hours → none produced epistemic signal. This pattern
is rationalized as "scientific persistence" but is actually sunk-cost
fallacy. Each failed attempt makes the next attempt psychologically harder
to abandon, even though the evidence points to environment-model mismatch.

**Correct behavior**: 
- Maximum 2 attempts per experimental condition. After the 2nd failure,
  document the negative result and pivot.
- The 3rd attempt requires written justification citing WHY this attempt
  will succeed where the previous two failed. If the justification is
  "maybe this parameter will work", the attempt is not authorized.
- Negative results are valuable. "Grid World cannot produce epistemic
  signal with 0.5B" is a valid scientific conclusion, not a failure.

**Reference**: `PHASE1_PARTIAL_EVALUATION.md` — "实际路径 vs PEDA v1.1 计划" section

---

### B9: Anomalous results unexplained after 1 hour

> **Added 2026-07-07 (GLM-5.2 follow-up)**. New Blocker rule.

**Trigger**: An experiment produces an anomalous result (e.g., score >0.99
on a hard task, 100% accuracy with minimal data, a behavior loop that
cannot be explained by the current model of how EFE/Drive System works),
and the agent has not provided a falsifiable explanation grounded in
the underlying math/logic within 1 hour of observing the anomaly.

**Why**: Phase 1's Grid Search showed score=0.996, steps=1.0 — likely a
bug in goal placement or step counting — but was accepted without
investigation. Phase 1.5 showed inventory confidence=0.999 causing a
17-step dead loop — this was eventually explained (overconfident WM +
weak boredom drive), but only after multiple eval cycles. Anomalies are
often the only signal of hidden bugs or fundamental limitations. Ignoring
them is not "moving fast" — it is building on sand.

**Correct behavior**:
- When an anomaly is observed, start a timer.
- Within 1 hour: (1) reproduce the result, (2) inspect intermediate
  variables, (3) either locate a bug or provide a falsifiable explanation,
  (4) document as known-issue-with-explanation if unresolvable.
- If the anomaly cannot be explained within 1 hour: **STOP writing code**.
  Switch to theory-analysis mode: draw the computational graph, trace the
  signal flow, identify where the math breaks down.
- Do not "work around" an unexplained anomaly by adding compensating
  code (e.g., "if score > 0.99: score = 0.5"). This is B2-level
  data fabrication.
- After explanation: if the anomaly reveals a bug → fix it. If it
  reveals a fundamental limitation → document it as a project finding
  and adjust the hypothesis.

**Reference**: `CODING_AGENT_EVALUATION.md` — Problem 3 "Grid Search结果可疑未调查"

---

## Concern Rules (WARN — advisor emits concern, held and re-confirmed)

### C1: Lint/docs/git consuming >2 consecutive turns [DEMOTED: Concern → Nit]

> **Update 2026-07-07 (GLM-5.2 follow-up)**: Demoted from Concern to Nit.
> Lint and docs are the lowest priority during exploration. The advisor
> should note them but not hold progress. This rule returns to Concern level
> only during pre-publication / write-up phases.

**Trigger**: Two or more consecutive turns are spent on ruff/pyright/mypy
fixes, AGENTS.md/README updates, git operations (commit/push/merge), or IRC
communication, while the current Phase's core hypothesis validation is
incomplete.

**Why**: This is the #1 symptom of priority inversion observed in the Folunar_
→ PEDA transition. Lint can wait. Hypothesis validation cannot.

**Correct behavior**: Use `# noqa` or `type: ignore` to bypass lint issues
temporarily. Freeze AGENTS.md updates until Phase validation completes.
Commit only after validation passes. Batch IRC communication.

**Reference**: `CODING_AGENT_EVALUATION.md` — Problem 4 "过度纠结lint"

---

### C2: Reporting process metrics as progress

**Trigger**: The agent reports progress using process metrics instead of
go/no-go criteria. Process metrics include: "tests pass", "lint clean",
"X lines of code written", "Y modules added", "Z commits made". Go/no-go
criteria include: "G1: PEDA exploration efficiency > random baseline",
"G2: World Model prediction accuracy improves with experience",
"G3: behavioral diversity metrics (coverage > 0.8, entropy > 0.5)".

**Why**: Folunar_'s 92.2% "success rate" was command execution success, not
task completion success — a classic process-metric-as-outcome substitution.
PEDA must be honest about what it is measuring.

**Correct behavior**: Every progress report must include at least one
go/no-go criterion status. If no criterion has moved, report "no measurable
progress this turn" rather than inflating with process metrics.

**Reference**: `peda_reflection_v11.md` — Problem 6 "Eval G2/G3失败被合理化"

---

### C3: Anomalous experiment results left uninvestigated

**Trigger**: An experiment produces a result that is surprising or
statistically anomalous (e.g., score=0.996, steps=1.0, 100% accuracy on a
hard task), and the agent dismisses it with "expected outcome",
"known limitation", or moves on without root cause analysis.

**Why**: Anomalies are often the only signal of a hidden bug. The Grid
World grid search in the coding history showed score=0.996, steps=1.0 —
likely a bug in goal placement or step counting — but was not investigated.

**Correct behavior**: Any result >2 standard deviations from expectation must
be investigated. The agent must: (1) reproduce, (2) inspect intermediate
variables, (3) either locate a bug or provide a falsifiable explanation,
(4) document as known-issue-with-explanation if unresolvable.

**Reference**: `CODING_AGENT_EVALUATION.md` — Problem 3 "Grid Search结果可疑未调查"

---

### C4: Drive System hyperparameters set without justification

**Trigger**: The `curiosity`, `competence`, `boredom`, or `novelty` drive
weights (or any new tunable parameter) are assigned a value without one of:
a grid search result, a citation from a paper, or an explicit ablation
showing why other values were rejected.

**Why**: PEDA's behavior is highly sensitive to these weights. Folunar_
never systematically tuned its hyperparameters, leading to brittle behavior.

**Correct behavior**: Initial weights can be rough estimates, but a grid
search over `[0.1, 0.3, 0.5, 0.7, 1.0]` must be conducted during Phase 1
and the winning configuration must be documented with the evaluation metric
that selected it.

**Reference**: `PEDA架构设计与开发计划书_v1.1.docx` — 5.5节 "Drive System 超参数敏感性"

---

### C5: Safety boundaries missing or bypassed

**Trigger**: The Docker sandbox configuration allows: destructive commands
(rm -rf, mkfs, dd), unbounded network access, unbounded resource usage, or
World Model predictions of dangerous commands without a rule-based sanity
check.

**Why**: PEDA runs an LLM that generates bash commands. Without guardrails,
it could damage the host system or exfiltrate data. This is not theoretical
— Replit's AI Agent deleted a production database.

**Correct behavior**: Maintain a `COMMAND_BLACKLIST` (rm -rf, mkfs, dd,
chmod 777 /, etc.), enforce Docker `--read-only` mounts with `--tmpfs` for
scratch space, limit network to a whitelist if enabled, and add a rule
engine that validates World Model predictions against known-safe command
patterns.

**Reference**: `PEDA架构设计与开发计划书_v1.1.docx` — 3.3节 "安全边界设计"

---

### C6: Inference speed bottleneck unaddressed

**Trigger**: The Action Generator uses >5 candidate actions or a rollout
horizon >5 steps without evidence that the target hardware can complete a
single decision within 10 seconds. If the agent has not measured and
documented the per-step LLM latency, this rule triggers.

**Why**: PEDA's EFE-based action selection requires multiple LLM calls per
decision. If each decision takes 30+ seconds, the agent cannot run
meaningfully. The plan acknowledges this but does not require measurement.

**Correct behavior**: In Phase 1, measure the actual per-step latency on the
target hardware. If latency > 10s per decision, reduce candidates to 2-3
and horizon to 2-3, or fall back to single-step greedy selection. Document
the measured latency and the selected configuration.

**Reference**: `PEDA架构设计与开发计划书_v1.1.docx` — 3.5节 "推理速度讨论"

---

### C7: Pilot success used to skip confirmatory experiment

**Trigger**: A pilot experiment (1-3 episodes) produces a favorable
result, and the agent uses it to justify moving forward — skipping the
planned confirmatory experiment (≥10 episodes) — OR starts drafting
Phase advancement documentation before the confirmatory run completes.

**Why**: Strong pilot results create psychological pressure to "just move
on". The Phase 1 partial-training pilot showed PEDA 2 steps vs
pragmatic_only 20 steps — a dramatic difference. It is very tempting to
treat this as "good enough" and proceed to Phase 1.5. But pilot results
are, by definition, unreliable. The confirmatory experiment (10 episodes)
is the only way to know if the effect is real.

**Correct behavior**: 
- A strong pilot result should INCREASE motivation to run the
  confirmatory experiment quickly, not decrease it.
- The confirmatory experiment must use the EXACT same protocol as the
  pilot (same train_fraction, same drive weights, same pragmatic_weight)
  unless a pre-registered protocol change is documented.
- If the agent says "the pilot was so strong that we don't need more
  data", treat this as a red flag requiring explicit override.

**Reference**: `PHASE1_PARTIAL_EVALUATION.md` — "pilot vs confirmatory" section

---

### C8: Epistemic signal dominated by pragmatic term

**Trigger**: The EFE calculation shows that the pragmatic term
(`pragmatic * weight`, typically weight=3.0) contributes >80% of the
EFE value while the epistemic term contributes <20%, across most
states/actions. OR: the agent observes that behavior does not change
meaningfully when epistemic is set to zero (pragmatic_only mode produces
similar trajectories to full PEDA).

**Why**: PEDA's core claim is that "prediction error drives exploration".
If pragmatic distance dominates EFE, the system is not "prediction error
driven" — it is "greedy distance minimization with a small curiosity
bonus". Phase 1 first-round validation had this exact problem:
pragmatic_weight=3.0 with epistemic≈0 created a pure greedy navigator
that happened to perform well, but did not validate the core hypothesis.

**Correct behavior**: 
- Monitor the epistemic/pragmatic ratio during experiments (log both
  terms per step).
- If pragmatic dominates consistently, consider: (a) reducing
  pragmatic_weight to 1.0 or 0.5, (b) increasing the environment's
  uncertainty (lower train_fraction, larger grid), (c) adding a minimum
  epistemic floor.
- Report the ratio explicitly in validation reports. A PEDA validation
  without epistemic contribution data is incomplete.

**Reference**: `PHASE1_EVALUATION.md` — "核心问题：这不是预测误差驱动探索" section

---

### C9: Training and evaluation on same distribution

**Trigger**: The agent trains a World Model (or any predictive model) on
a dataset and then evaluates it on a draw from the SAME distribution
without an explicit train/test split, hold-out set, or out-of-distribution
test. OR: the evaluation environment has the same structure, rules, and
state space as the training environment with no meaningful variation.

**Why**: Phase 1 first-round validation had this exact problem — G1=1.0
(World Model accuracy) because the eval data came from the same
5×5 grid with the same dynamics as the training data. The WM was not
"learning to predict"; it was "memorizing the training set". This is a
fundamental experimental design flaw that invalidates any claim about
generalization or learning.

**Correct behavior**: 
- Always maintain an explicit test set that the model has never seen
  during training. In grid world: hold out a region of cells; in text
  environments: hold out a subset of commands/tasks.
- Report both in-distribution (train-set-like) and out-of-distribution
  (novel) accuracy. In-distribution accuracy alone is meaningless.
- If OOD accuracy is not available, explicitly state this limitation
  and treat all claims as "memorization, not generalization".

**Reference**: `PHASE1_EVALUATION.md` — "训练-评估同分布问题" section

---

### C10: Plan deviation not reported to upstream

**Trigger**: The actual execution path deviates significantly from the
pre-agreed plan (e.g., skipping a Phase, spending 3x longer than
allocated on a task, changing the experimental approach), and the agent
does not produce a concise deviation report explaining: (1) what changed,
(2) why it changed, (3) what the new path looks like, and (4) the
estimated impact on overall timeline.

**Why**: PEDA v1.1 planned 29-40 weeks with Phase 1 (Grid World) as the
core hypothesis validation. Actual execution: Grid World abandoned after
multiple failed attempts, Phase 1's hypothesis validation role moved to
Phase 1.5 (text environment), and the overall timeline structure changed
significantly. While the team's decision-making was sound, the deviation
was only discussed in ad-hoc messages rather than in a structured report.
Without explicit deviation tracking, future phases may unknowingly repeat
the same misalignment.

**Correct behavior**: 
- When a deviation >25% from the planned approach occurs, produce a
  brief "Deviation Report" (can be a single paragraph) covering the 4
  points above.
- This is not about blame — it is about maintaining shared mental model
  between the agent and upstream reviewers.
- Example: "Grid World abandoned after 3 attempts (25%→10%→3epochs).
  Root cause: environment too simple for 0.5B model. New path: Phase 1.5
  (text environment, 2 rooms) takes over hypothesis validation. Timeline
  impact: Phase 1 compressed to infrastructure validation only, Phase 1.5
  extended by 1-2 weeks. Overall project timeline: 20-30 weeks (vs
  planned 29-40)."
- Attach this report to the current conversation or commit it as
  `docs/deviation_YYYYMMDD.md`.

**Reference**: `PHASE1_PARTIAL_EVALUATION.md` — "实际路径 vs PEDA v1.1 计划" section

---

### C11: Measurement method inconsistent with independent ground truth [RELAXED]

> **Update 2026-07-07 (GLM-5.2 follow-up)**: Relaxed for early exploration.
> During initial exploration, perfect measurement alignment is unrealistic.
> The rule fully applies during validation / confirmatory phases. During
> exploration: note discrepancies, prioritize fixing, but do not block
> progress if investigation would take >1 hour.

**Trigger**: Two or more measurement methods that purport to measure the
same construct produce discrepant results (>10x difference or opposite
conclusions), and the agent does not investigate the discrepancy OR
continues using the method that produces the "more favorable" result.

**Why**: Phase 1.5 had this exact problem. `decompose_error()` reported
`mean_epistemic_error=0.0` (no uncertainty detected), while the semantic
probe showed 50% disagreement on full tuples and 40% on has-key
predictions. The discrepancy existed because `decompose_error()` only
checked `(room, exit_code)` — ignoring the `inventory`/`has-key`
dimension entirely. The agent continued reporting `epistemic=0` even
though a more careful measurement showed substantial epistemic
uncertainty. Using a broken measurement method led to false conclusions
about the core hypothesis.

**Correct behavior**:
- When two measurement methods disagree by >10x, STOP and investigate.
- The investigation must: (1) examine what each method actually measures,
  (2) identify which dimensions/metrics each includes/excludes, (3)
  determine which method is more complete, (4) fix the incomplete one.
- Never "average" discrepant measurements or use the one that fits your
  hypothesis. The more complete method wins.
- After fixing, validate that the two methods now agree within 2x.
- **Exploration phase relaxation**: If fixing would take >1 hour, document
  the discrepancy with a `# TODO(measurement)` and continue. Return to
  fix before confirmatory experiments.

**Reference**: `PHASE1_5_COMPLETE_EVALUATION.md` — "验证 3：decompose_error Bug" section

---

### C12: Agent trapped in local optimum due to overconfident predictions

**Trigger**: The PEDA agent repeatedly selects the same action for ≥3
consecutive steps despite no progress toward the goal, and investigation
shows that the World Model assigns >0.99 confidence to the
state-prediction for that action — creating a self-reinforcing loop where
high confidence → low EFE → same action → same state → high confidence.

**Why**: Phase 1.5 full eval showed PEDA stuck in an `inventory` loop
for 17 consecutive steps after successfully taking the key. The model
assigned 0.999 confidence to `inventory` predictions, making its EFE
the lowest of all candidates. The agent had no mechanism to escape:
boredom drive (0.1) was too weak, and epistemic bonus was zero because
all checkpoints agreed. This is not a bug in the code — it is a
fundamental limitation of the EFE formulation when the World Model is
overconfident about uninformative actions.

**Correct behavior**:
- Log the confidence values per action per step. If any action has
  confidence >0.95 for ≥3 consecutive steps, flag as "potential loop".
- If detected: (a) temporarily boost exploration bonus for actions not
  taken in the last 3 steps, (b) consider a "minimum epistemic floor"
  that prevents any action's EFE from being zero, (c) document the loop
  pattern as a known limitation.
- Distinguish "benevolent loops" (repeated action makes progress) from
  "trapped loops" (repeated action with no state change). Only the
  latter require intervention.
- This pattern is expected in early phases. Do not over-engineer
  solutions — document and move on.

**Reference**: `PHASE1_5_COMPLETE_EVALUATION.md` — "行为分析" section

---

### C13: Token-space prediction used when latent-space is available

**Trigger**: The project has been advised (by external expert review) that
predicting next_state in token space is statistically inefficient and
noisy compared to latent-space prediction (JEPA route), yet the agent
continues to refine token-space approaches without evaluating the
alternative. OR: Phase 2 core hypothesis fails and the agent proposes
token-space improvements (more data, larger model) without considering
latent-space prediction.

**Why**: GLM-5.2 identified JEPA (I-JEPA, V-JEPA) as a major omission.
Token-space prediction forces LLM to model grammar, whitespace, and
formatting — all irrelevant to environment dynamics. Latent-space
prediction discards these "difficult-to-predict bottom-level details"
and focuses on state representation changes. If Phase 2 fails to
produce effective epistemic signals, continuing to optimize token-space
prediction is a form of B8 ("just one more try" death spiral) applied
to architecture rather than hyperparameters.

**Correct behavior**:
- Phase 2 uses token-space prediction with JSON-structured states
  (immediate implementation cost ~30 min, significant noise reduction).
- If Phase 2 hypothesis validation fails: BEFORE proposing token-space
  improvements, evaluate JEPA/latent-space prediction feasibility.
  Estimated cost: 3-5 days. This is a "Phase-level" decision, not a
  hyperparameter tuning decision.
- Mark `GLM5_2_RESPONSE_ANALYSIS.md` — "Action: Phase 2 failure →
  evaluate JEPA" as a binding constraint.

**Reference**: `GLM5_2_RESPONSE_ANALYSIS.md` — Q5 "JEPA 遗漏" section

---

### C14: Drive System claimed as theoretically valuable without controlled verification

**Trigger**: The agent or project documentation describes Drive System
(curiosity/competence/boredom/novelty) as having "independent value",
"theoretically interesting", or "FEP-validated" without having run
the controlled experiment: PEDA (full EFE + Drive) vs heuristic
baseline (random priority queue + boredom penalty only). OR: the agent
uses PEDA's behavior difference from pragmatic_only as evidence that
Drive System works, without testing if a simpler mechanism produces
the same difference.

**Why**: GLM-5.2 identified Drive System's exploration value as a
potential artifact — "essentially epsilon-greedy with short-term
memory penalty." PEDA's Phase 1.5 behavior (step 1 take key vs
pragmatic's look x20) could be explained by boredom accumulation
alone, not by the full Drive System. Claiming theoretical value for
an unverified mechanism is a form of C2 (process metrics as progress)
applied to architecture claims.

**Correct behavior**:
- Drive System is an engineering safeguard ("fool-proof design"), not
  a validated theoretical contribution until proven otherwise.
- The controlled experiment must be run in Phase 2: implement a
  "Heuristic Baseline" (random action selection + boredom penalty only)
  and compare to full PEDA. If both behave similarly → Drive System
  is confirmed as artifact. If PEDA significantly outperforms →
  Drive System has independent value.
- Until this experiment is run, all claims about Drive System must be
  prefixed with "unverified" or "engineering mechanism, not theoretical
  contribution."

**Reference**: `GLM5_2_RESPONSE_ANALYSIS.md` — Q6 "Drive System 价值评估" section

---

## Nit Rules (Minor — advisor emits nit, delivered immediately)

### N1: Work that could be deferred

**Trigger**: The agent is spending a turn on Type Hints, docstrings,
README prose, import sorting, or cosmetic refactoring while the current
Phase's go/no-go criteria are incomplete.

**Correct behavior**: Mark with `# TODO(Phase-N)` and move on. Come back
only after the Phase is validated.

---

### N2: Not reusing existing solutions

**Trigger**: The agent is implementing from scratch a capability that exists
in an established open-source library (e.g., vector DB → use ChromaDB,
config management → use pydantic-settings, caching → use diskcache).

**Correct behavior**: Check PyPI or GitHub for existing solutions before
writing >50 lines of infrastructure code. PEDA's principle is "minimum code
to validate the hypothesis", not "build everything from scratch".

---

### N3: Ignoring Trahexa/Folunar_ lessons

**Trigger**: The agent makes a design decision that directly contradicts a
lesson from the Folunar_ post-mortem without explaining why this project is
different. This includes: adding modules for "completeness", creating new
PLAN documents, online training without replay buffer, claiming "emergence"
from <1B parameters.

**Correct behavior**: Reference the relevant lesson and explain why it does
not apply, OR acknowledge the risk and add a mitigation.

---

## The "3 Questions" Framework (Added 2026-07-07)

> **Source**: GLM-5.2 follow-up, Q2. Replaces the static 23-rule checklist
> for day-to-day experimental decisions. The WATCHDOG rules (B1-B9, C1-C14,
> N1-N3) remain as guardrails, but the primary agent should answer these 3
> questions **before every non-trivial experiment**.

**Question 1**: What specific hypothesis does this experiment falsify?
- If the experiment cannot falsify any specific hypothesis, do not run it.
- "Let's see what happens" is not a hypothesis.
- Good answer: "This experiment tests whether hidden-state epistemic (JEPA
  light) produces a stronger Spearman correlation with action selection
  than token-space epistemic."

**Question 2**: If this experiment fails, what is the most likely cause,
  and can I confirm that cause within 2 hours?
- If you cannot name the most likely failure mode → you don't understand
  the experiment well enough to run it.
- If confirming the cause would take >2 hours → the experiment is too
  coarse; design a smaller pilot first.
- Good answer: "If epistemic doesn't improve, most likely cause is hidden
  states are too similar across checkpoints. I can confirm this by
  computing pairwise cosine distances in 30 minutes."

**Question 3**: Is this the first time I'm trying this approach?
- If yes: proceed.
- If no (N≥2): **require a written justification** (can be 2-3 sentences
  in the commit message or a quick note) explaining why this Nth attempt
  will succeed where previous attempts failed.
- If the justification is "maybe this parameter will work" → the attempt
  is not authorized (B8).
- Good answer: "This is the 2nd attempt at hidden-state epistemic. The
  1st failed because I used the wrong layer (layer 0 instead of last
  layer). This attempt uses last-layer pooled hidden states, which should
  capture semantic uncertainty."

**Relationship to WATCHDOG rules**:
- Q1 prevents B1 (no-hypothesis phase advancement) and C2 (process
  metrics as progress).
- Q2 prevents B8 (death spiral) by forcing early failure-mode analysis.
- Q3 enforces B8's "written justification for 3rd attempt" rule.
- All 3 questions together replace the need to memorize 23 rules for
  daily decisions. The rules remain as reference for the advisor to
  check against.

---

## Rule Changelog

### 2026-07-07 — GLM-5.2 Follow-up Round 2

| Rule | Change | Reason |
|------|--------|--------|
| B4 | Blocker → Concern | Exploration-phase pivots are normal; new focused docs are clearer |
| B7 | Blocker → Concern | Hard-blocking judgment calls stifles boundary probing |
| C1 | Concern → Nit | Lint is lowest priority during exploration |
| C11 | Relaxed (exploration) | Perfect measurement alignment unrealistic in early exploration |
| B9 | **New Blocker** | Anomalous results must be explained within 1 hour |
| "3 Questions" | **New framework** | Dynamic decision framework replacing static rules for daily use |
| C13 | Added | Token-space vs latent-space prediction evaluation |
| C14 | Added | Drive System artifact verification requirement |

---

## Self-Check Reminder (for the advisor itself)

The advisor must NOT:
- Spam repeated advice. Emit at most one advisory per concrete issue per turn.
- Inject "Stop." / "Done." without tying to a specific rule violation.
- Keep advising after the task is complete. Park when no issues are present.
- Use process metrics ("you made good progress") as advice content.

If the advisor detects it is repeating the same note, it must self-silence
until a genuinely new issue appears.

---

## Reference Files

- `PEDA架构设计与开发计划书_v1.1.docx` — full architecture and plan
- `peda_reflection_v11.md` — what went wrong in v1.0 and how v1.1 fixed it
- `CODING_AGENT_EVALUATION.md` — evaluation of the coding agent's Phase 1 execution
- `folunar_review.agent.final.md` — Folunar_ post-mortem (historical lessons)
- `PHASE1_EVALUATION.md` — evaluation of the first-round validation (G1/G2/G3)
- `PHASE1_PARTIAL_EVALUATION.md` — evaluation of the partial-training redesign
- `PROMPT_RUN_10EPS.md` — task prompt for the 10-episode confirmatory experiment
- `PROMPT_DECISION.md` — 30-min decision task (0.10 train-fraction or Phase 1.5)
- `PROMPT_PHASE1_5_TRAIN.md` — Phase 1.5 text World Model training task
- `PHASE1_5_SETUP_REPORT.md` — Phase 1.5 environment setup report
- `PHASE1_5_COMPLETE_EVALUATION.md` — evaluation of the Phase 1.5 complete report
- `PHASE1_5_ITERATION2_EVALUATION.md` — evaluation of Iteration 2 (decompose_error fix + data augmentation)
- `PROMPT_PHASE1_5_NEXT.md` — task prompt for the decompose_error fix + data augmentation iteration
- `PROMPT_PHASE2_START.md` — task prompt for Phase 2 infrastructure (busybox sandbox)
- `GLM5_2_BRIEF.md` — technical brief for external expert consultation (GLM-5.2)
- `GLM5_2_PROMPT.md` — complete prompt ready to paste to GLM-5.2 (round 1)
- `GLM5_2_RESPONSE.md` — GLM-5.2's round 1 response (7 questions answered)
- `GLM5_2_RESPONSE_ANALYSIS.md` — upstream analysis of GLM-5.2 round 1 with action items
- `GLM5_2_FOLLOWUP_PROMPT.md` — follow-up prompt (round 2, 4 questions)
- `GLM5_2_FOLLOWUP_RESPONSE.md` — GLM-5.2's round 2 response (4 questions answered)
- `GLM5_2_FOLLOWUP_ANALYSIS.md` — upstream analysis of GLM-5.2 round 2 with action items
