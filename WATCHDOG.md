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

### B4: Creating new PLAN/ARCH documents instead of updating existing ones

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

## Concern Rules (WARN — advisor emits concern, held and re-confirmed)

### C1: Lint/docs/git consuming >2 consecutive turns

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
- `CODING_AGENT_EVALUATION.md` — evaluation of the coding agent's execution
- `folunar_review.agent.final.md` — Folunar_ post-mortem (historical lessons)
