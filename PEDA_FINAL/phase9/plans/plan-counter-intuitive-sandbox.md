# Plan: Counter-Intuitive Sandbox (PEDA Post-Mortem follow-up)

Charter: `local://contract-counter-intuitive-sandbox.md`
Status: **implemented + verified for Level 1** (image built, probe suite passing).
Author: subagent `CounterIntuitiveSandboxDesign`.

---

## 0. Design principle (the "sweet spot")

Normal sandbox DLR 0.8–0.9 because `cat file.txt` output is guessable from LLM
semantic priors. The counter-intuitive sandbox makes **every reversed command's
effect a deterministic function the LLM cannot guess but a learner can fit**:

- **Deterministic & context-free** — same command, same effect, everywhere, every time.
- **Observable in `SandboxState`** — every reversal must change `files`,
  `last_output`, or `last_exit_code` (visible to perception and to the world model).
- **Self-bounding** — no unbounded state growth (`ls` twin rule converges).
- **Exit codes mirror busybox semantics** (0 ok / 1 no-match-or-missing / 2 usage)
  so L1 prediction is learnable, not random.

Epistemology: with zero interaction data, an LLM CANNOT infer the rules (priors say
`cat` reads). With ~5–15 interactions per command, any statistical learner CAN
(STRIPS learns preconditions/effects from traces; a LoRA world model fits 200
transitions). That asymmetry is the epistemic signal the PEDA hypothesis needs.

---

## 1. Rule table (command → exact reversed behavior)

### Meta-rule (internal consistency)

Each reversed command performs the **semantic opposite of its normal role**,
as a permutation on information flow:

| Normal role | Command | Reversed role |
|---|---|---|
| extract content | `cat` | destroy content (delete file) |
| emit/write string | `echo` | extract content (read file) |
| enumerate names | `ls` | materialize names (create empty `.ls` twins) |
| filter (keep matches) | `grep` | anti-filter (drop matches) |
| prefix (first N) | `head` | suffix (last N) |
| suffix (last N) | `tail` | prefix (first N) |

The mapping is a bijection: {read, write, list, keep, first, last} → {destroy,
read, create, drop, last, first}. `head`↔`tail` is a pure swap; `cat`↔`echo` is a
role swap (echo becomes what cat was, cat becomes its opposite).

### Level 1 rules (3 reversed; whitelist `{ls, cd, cat, echo, pwd, wc}`)

| Command | Exact behavior (mode A) | Output | Exit |
|---|---|---|---|
| `cat FILE...` | **Deletes** each FILE (unlink). Flags (`-*`) ignored. Nonexistent FILE → stderr `cat: F: No such file or directory`. No args → no-op. Directories → treated as nonexistent (error). | nothing | 0 if all args were existing files, else 1 |
| `echo ARG...` | **Reads** each ARG: prints contents of every FILE argument, in order. No args → no-op. Any arg that is not a regular file → that arg yields nothing. | concatenated file contents (no extra newline) | 0 if ≥1 arg and all args are files, else 1 |
| `ls [DIR]` | **Creates** one empty file `<entry>.ls` per non-`.ls` entry in DIR (default `.`), then prints the twin basenames. **Never twins the twins** → bounded growth. Flags ignored. DIR missing → stderr error. | twin names, one per line | 0 (or 1 if DIR missing) |

Level-1 normal commands (honest tools): `cd`, `pwd`, `wc` (works normally; `wc` is
the one analyzer the agent can trust — a distractor that must be *discovered* as
trustworthy, see Level 2).

### Level 2 rules (6 reversed + 2 normal confusers; whitelist = full set)

Adds, on top of Level 1:

| Command | Exact behavior (mode A) | Output | Exit |
|---|---|---|---|
| `grep PAT FILE...` | **Inverted matching**: outputs lines NOT containing PAT (≡ real `grep -v`). Flags ignored. PAT missing → usage error. No FILE → reads stdin inverted. | non-matching lines | 0 if any line output, 1 if none, 2 usage |
| `head [-n K] FILE` | **Last** K lines (default K=10). `-n K` parsed; other flags ignored. | last K lines | busybox tail's exit code |
| `tail [-n K] FILE` | **First** K lines (default K=10). | first K lines | busybox head's exit code |

**Normal confusers (Level 2 — deliberately NOT reversed):** `wc`, `find`.
They sit in the same "content analysis" family as the reversed commands; the agent
must learn *which* commands are inverted rather than assuming all-or-nothing.

Normal in all levels: `cd`, `pwd`, `mkdir`, `touch`.

### Level 3 (non-stationary — design, not yet built)

Two rule configs, switched by a mode file `/tmp/ci/mode`:

| Config | cat | echo | ls | grep | head | tail |
|---|---|---|---|---|---|---|
| A (default, = L2) | delete | read | create twins | invert | last | first |
| C (flip) | normal | normal | normal | invert | last | first |

Mechanism: every wrapper reads the mode file at invocation and dispatches
(`mode A` → reversed logic; `mode C` → `exec /bin/busybox <applet> "$@"` for
cat/echo/ls). Mode file absent → A. The harness writes the mode file at episode
reset (seed → config) and optionally mid-episode (every K steps) to make the
environment non-stationary. **Prerequisite:** for L3, `_validate_command` must
reject shell metacharacters (`; & | > < \``) so the agent cannot self-defeat the
reversal via `export`/file-deletion tricks; L1/L2 keep the current lenient
validation.

---

## 2. Micro-tasks (3, exploiting reversed rules)

All start at `/sandbox`, fresh container per episode, max_steps 20.
Goal predicates are content-based (no command check) — the agent must discover
WHICH command reveals content. Predicates are written for `src/phase2/tasks.py`
conventions (`check(state, action, next_state)`, evaluated per step):

```python
# T1 — the ONLY path to success is the reversed reader (echo).
# cat docs/note.txt deletes the secret and makes the task impossible this episode.
def _goal_predicate_read_secret_ci(state, action, next_state) -> bool:
    return "9471" in getattr(next_state, "last_output", "")

# T2 — same reader discovery on data; wc gives a count but not the content string.
def _goal_predicate_read_data_ci(state, action, next_state) -> bool:
    return "line3" in getattr(next_state, "last_output", "")

# T3 (Level 2) — requires INVERTED grep: output must contain the WARN line and
# NO "ERROR:" line. echo prints everything (incl. ERROR lines) → fails;
# normal grep prints the ERROR lines → fails. Only inverted grep passes.
def _goal_predicate_find_warn_ci(state, action, next_state) -> bool:
    out = getattr(next_state, "last_output", "")
    return "WARN" in out and "retry" in out and "ERROR" not in out
```

| ID | Task | Fixture | Wrong-prior path | Right path |
|---|---|---|---|---|
| `read_secret_ci` (L1) | Read secret key | `docs/note.txt` = `secret key: 9471` | `cat docs/note.txt` → deletes it, output empty | `echo docs/note.txt` → `secret key: 9471` |
| `read_data_ci` (L1) | Reveal content of lines file | `data/lines.txt` = line1..3 | `cat data/lines.txt` → deletes it | `echo data/lines.txt` → line1\nline2\nline3 |
| `find_warn_ci` (L2) | Get the non-ERROR line from the log | `logs/error.log` = 2×ERROR + 1×WARN | `grep ERROR logs/error.log` → outputs ERROR lines | `grep ERROR logs/error.log` → **inverted**: outputs WARN line only |

Control fixture: `docs/readme.txt` = `hello world (disposable control file)` —
a known-plaintext file the agent will likely `cat` first; the empty output +
subsequent file absence give the world model its first strong error signal.

`ls` exploitation note: `ls` twins are empty files and inert for candidate
generation (ext `.ls` is not a text extension) — `ls`'s epistemic role is the
surprising files-delta, not task completion. An optional L2 task
(`ghost_cleanup_ci`: run `ls`, then verify the twin set is stable on a second
`ls`) tests the bounded-twin rule; not required for the headline result.

---

## 3. Dockerfile implementation approach

**Approach: wrapper shell scripts shadowing busybox applets via PATH.**
(Rejected: symlink aliases — busybox dispatches on argv[0], so
`ln -s /bin/busybox /usr/local/bin/cat` just re-creates normal `cat`; shell
`alias` doesn't work under non-interactive `sh -c`. Wrappers are the only way to
get *conditional/stateful* reversed logic.)

- `FROM busybox:latest`; `WORKDIR /sandbox`; fixture files laid out at build time.
- `COPY ci/wrappers/ /usr/local/bin/` + `chmod 755`.
- `ENV PATH=/usr/local/bin:...` — /usr/local/bin first, so `sh -c "cat f"` hits
  the wrapper. Wrappers internally ALWAYS call `/bin/busybox <applet>` (never
  bare names) to avoid recursive reversal.
- **Builtin trap (discovered during implementation):** `echo` is an ash
  BUILTIN (`command -v echo` → `echo`), so a PATH wrapper is never reached.
  Fix: `/usr/local/bin/sh` is a prelude script
  `#!/bin/busybox sh` + `echo() { /usr/local/bin/echo "$@"; }` + `eval "$2"`.
  The harness execs `docker exec <cid> sh -c <action>`, and `sh` resolves via
  PATH → the prelude; the `echo()` FUNCTION shadows the builtin (POSIX:
  functions shadow builtins), then `eval "$2"` runs the agent command.
  cat/ls/grep/head/tail resolve via PATH to wrappers; cd/pwd/wc stay normal.
  Consequence: any harness-internal `sh -c 'echo ...'` (L3 mode-file writes)
  MUST use `/bin/busybox echo` explicitly.
- Every wrapper reads `/tmp/ci/mode` (absent → mode A) for L3 dispatch; Level 1
  image simply never creates the file.
- **Rootfs is writable** (cat deletes, ls creates). Security posture kept:
  `--cap-drop=ALL --network none`; fresh container per episode ⇒ no cross-episode
  contamination. This is a deliberate, documented deviation from the v2/v4
  sandbox's `--read-only` (which made reversals impossible).

**Harness contract (critical):** the harness's own perception MUST bypass the
wrappers, or every state read triggers a reversal:
- `_list_files()` → `/bin/busybox ls -1 <cwd>` (bare `ls` would create twins on
  every perception read).
- `reset()`'s `mkdir -p` → `/bin/busybox mkdir -p`.
- Container start for this image: drop `--read-only` (keep `--cap-drop=ALL`,
  `--network none`, `--tmpfs /tmp`).

Files (implemented):
- `Dockerfile.counterintuitive` (project root, next to the v2/v4 Dockerfiles)
- `ci/wrappers/{cat,echo,ls,grep,head,tail}` (6 wrapper scripts)
- `scripts/ci_probe_sandbox.py` (M0 build-gate probe, passing)
- `.dockerignore` (repo is 9.1 GB; excludes .venv/results/checkpoints/.git from
  build context — no existing Dockerfile uses COPY, so zero behavior change)

---

## 4. Validation plan: count-driven baseline vs prediction-error agent

**Setup (both agents, identical affordances):**
- Same image `peda-sandbox:counterintuitive-v1`, same tasks, same whitelist,
  20 episodes × max_steps 20, fresh container per episode.
- Same candidate generator — CI variant adds `echo <file>` for text extensions
  (the reader must be reachable) and keeps `cat <file>` (the wrong choice must
  remain possible). `generate_sandbox_candidates` already caps at 16 and
  filters through `_validate_command`.
- Same explorer priors — `_ACTION_PRIORITY` moves `echo` 3 → 0 (reader tier),
  identical for both agents (otherwise the count baseline is unfairly blind).
- **Pre-registered asymmetry:** count agent = novelty + success-cache replay
  (its Phase-8 mechanism, gets task feedback); prediction-error agent = pure
  error-driven exploration via `run_peda_episode` (step reward is always 0 in
  this harness — the agent gets NO task feedback). If error-driven exploration
  matches counts WITHOUT reward, that is the strong result.

**Protocol:**
1. **M0 build gate** — `scripts/ci_probe_sandbox.py` verifies every reversed
   rule (echo reads, cat deletes, ls twins, ls bounded, grep inverts, head/tail
   swap, busybox-ls perception bypass). 100% pass required before any experiment.
2. **M1 prior-breakage** — deterministic probe set P: 30 (state, action) pairs
   (10 per reversed verb). Run the *untrained* LLM world model (vanilla
   Qwen-0.5B or the Phase-2 normal-sandbox adapter) on P. DLR = fraction of
   {exit_code, files_delta, output} components predicted correctly.
   Require **DLR ≤ 0.35** (vs ≥ 0.8 in the normal sandbox).
3. **M2 learnability** — collect 200 reversed transitions (count explorer on CI
   tasks, `phase2_collect_data.py` flow against the CI image); LoRA fine-tune
   (`phase2_synthetic_train.py`); held-out DLR on disjoint 60-pair set:
   require **DLR ≥ 0.70**.
4. **M3 agent comparison** — run both agents on the 3 CI tasks; report per-task
   and pooled: completion rate, steps-to-first-success, learning curve
   (episodes 11–20 vs 1–10). Require prediction-error completion ≥ count − 10pp;
   both agents improve ≥ 2× in the second half; discovery steps ≤ 1.5× count.
5. **M5 fail-fast cross-check** — same tasks against the NORMAL sandbox
   (`peda-sandbox:v4`) and the normal tasks against the CI image, to prove the
   reversal is bidirectional (see §6).

Reference points: count baseline on normal sandbox = 62.2% (9 tasks, zero task
knowledge); the CI env must be HARDER for prior-driven agents but solvable by
learning agents (M5c).

---

## 5. Pre-defined metric: what counts as "useful epistemic signal"

**Definition (pre-registered):** the environment provides a *useful epistemic
signal* iff all four hold:

| # | Metric | Measurable definition | Threshold |
|---|---|---|---|
| M1 | Prior breakage | DLR of the untrained LLM world model on probe set P (30 pairs) | **≤ 0.35** (normal sandbox: ≥ 0.8) |
| M2 | Learnability | held-out DLR after LoRA on ≤ 200 CI transitions | **≥ 0.70** |
| M3 | Behavioral productivity | PE-agent task completion vs count baseline, 20 episodes × 3 tasks | **≥ count − 10pp**; both improve ≥ 2× ep11–20 vs ep1–10; discovery ≤ 1.5× count |
| M4 | Error trajectory (PE agent only) | per-step mean prediction error E(t) (1 − DLR over L1/L2/L3 components) | E(1..10) **≥ 0.5**; E(31..40) **≤ 0.5·E(1..10)** |

Interpretation: M1 fails → rules are guessable from priors (reversal too shallow;
strengthen). M2 fails → rules are chaotic noise (reversal too deep; loosen).
M3 fails → prediction error does not drive exploration better than counts (record
as negative result per research charter — the hypothesis is about the signal's
utility, and this env is the maximal-fairness test). M4 pins the mechanism: the
model must demonstrably *learn* the rules (error shrinks) before/between successes.

---

## 6. Fail-fast conditions (testable)

1. **Build gate (M0):** probe suite fails → wrappers/Dockerfile broken → stop.
   Already implemented and passing for Level 1 (see §8).
2. **Behavioral divergence (M5b):** run the identical scripted sequence
   `ls; echo docs/note.txt; cat docs/readme.txt; grep ERROR logs/error.log`
   in the normal (v4) vs CI image; assert outputs/exit codes diverge on
   **≥ 3 of 4** commands. Fails → wrappers are not actually reversing.
3. **Prior agent cannot solve (M5a):** an LLM policy with normal-sandbox priors
   (prompted "use cat to read files") must score **0/20** on `read_secret_ci`.
   If it scores > 0, the reversal is not deep enough.
4. **M1 > 0.35** → the LLM can guess the rules → deepen (more verbs, invert
   exit codes, invert `wc`).
5. **M2 < 0.70** → rules not learnable → loosen (fewer verbs, longer episodes).
6. **M5c: count baseline < 40%** on CI tasks → environment is adversarially too
   hard for ANY learner → reduce reversal count / add recovery mechanics.
7. **M3: PE < count − 10pp** → prediction error is still not a useful drive in
   the maximal-fairness env → formal negative result (charter-accepted).

Every condition is an observable, pre-registered test — no post-hoc
interpretation.

---

## 7. Harness integration changes (patch list, for the validation phase)

- `src/phase2/sandbox_env.py`:
  - `_list_files`: `ls` → `/bin/busybox ls` (perception bypass).
  - `reset()`: `mkdir` → `/bin/busybox mkdir`.
  - `_ensure_container(read_only=True)` param; CI image passes `read_only=False`.
  - `step()` file_cache: add `echo ` prefix to the cache patterns when running
    the CI image (echo is the reader there; keeps cache consistent).
  - Add `CounterIntuitiveSandbox(BusyboxSandbox)` (image
    `peda-sandbox:counterintuitive-v1`, read_only=False).
  - `generate_sandbox_candidates`: add `echo <f>` for text-file extensions.
- `src/phase2/tasks.py`: add 3 goal predicates + 3 `MICRO_TASKS` entries.
- `src/phase5/explorer.py`: `_ACTION_PRIORITY` echo 3 → 0 (both baselines).
- `src/phase8/count_driven_agent.py`: `_get_task` — CI tasks start at /sandbox
  (default); image/task overrides on the CLI.
- New: `scripts/ci_collect_probe_set.py` (M1/M2 data), agent comparison driver
  (M3) reusing `count_driven_agent.run` and `run_peda_episode`.

---

## 8. Implementation status (this deliverable)

**Done and verified:**
- `Dockerfile.counterintuitive` + 6 wrappers + `.dockerignore` + probe script.
- Image `peda-sandbox:counterintuitive-v1` built (commit 689c18e9b802).
- `scripts/ci_probe_sandbox.py` passes **12/12 checks**: echo reads (secret
  revealed; missing-file → exit 1), cat deletes (missing → exit 1 + stderr),
  ls creates bounded twins (root + subdir, no twin-of-twin growth, twin names
  printed), grep inverts (WARN only, no ERROR line; exit codes match
  real `grep -v`), head→last / tail→first, perception bypass
  (`/bin/busybox ls` side-effect free).
- End-to-end harness-path checks pass: `cd docs && echo note.txt` → secret
  (rc 0); `wc -l` / `pwd` normal; `cat` then `echo` on the same file → rc 1
  empty (destruction observable); writable rootfs without `--read-only` works.

**Not yet done (validation phase, follow-up work):** M1–M5 experiments, harness
patches from §7, Level 2/3 wrapper modes (grep/head/tail wrappers already ship
with mode-A logic — only the mode-C dispatch and whitelist extension remain),
`PEDA_WORKING_LOG.md` entry.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `cat` deletion makes tasks fragile (one wrong step = episode dead) | fresh container per episode; success-cache replay; the fragility IS the lesson |
| `ls` twin growth explodes state | twin-of-twin exclusion → converges after one `ls` per dir (10 base + 10 twins) |
| LLM WM trained on normal data has toxic priors | exactly what M1 measures; use vanilla adapter as the "prior" model |
| busybox ash quirks ($'\n', arrays, `local`) | wrappers avoid them (probe passes on real busybox) |
| agent defeats reversal via `;`-chaining (L3 mode tamper) | L3 prerequisite: metacharacter ban in `_validate_command`; L1/L2 lenient as today |
| filenames with spaces break `ls` wrapper splitting | project convention: no spaces in fixture names; documented limitation |
| 9.1 GB build context | `.dockerignore` added (no existing Dockerfile COPYs, zero behavior change) |
