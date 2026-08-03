# Phase 9 FF-HG-5 — Agent-Level Gate: Discriminator-Driven vs Count Baseline

- commit: `bf146c9e13b3233675a86672619dc494cc11966e`
- timestamp: 2026-08-02T02:00:09+00:00
- host: mioarch | cpu
- sandbox_image: peda-sandbox:v4
- episodes per run: 20 | max_steps: 10

## Gate definition (FF-HG-5, operationalized)

PHASE9_PLAN.md registers FF-HG-5 only as `Agent-level <= count baseline (post-MVP) -> DEAD` — no threshold/episodes/tasks. Per the task brief, the definition is completed **verbatim from Direction-1 M3 / FF-CI-6 spec** (20 eps x 3 tasks, PE >= count - 10pp), i.e. a protocol fill-in, not a post-hoc threshold adjustment:

| Criterion | Spec |
|---|---|
| Episodes | 20 per (agent, task) |
| Tasks | read_changelog_v4, count_measurements, find_errors_v4 (v4 sandbox) |
| Primary | PE(alpha=0.5) completion% >= count completion% - 10pp -> PASS |
| Secondary | both arms ep11-20 >= 2x ep1-10; discovery steps PE <= 1.5x count |
| Exploratory | PE(alpha=1.0) = discriminator REPLACES count (not gated) |

## Per-run summary

| agent | alpha | task | branch (MVP split) | eps | completion% | avg steps | ep1-10% | ep11-20% |
|---|---|---|---|---|---|---|---|---|
| count | 0.5 | count_measurements | train (data) | 20 | 100.0 | 1.3 | 100.0 | 100.0 |
| count | 0.5 | find_errors_v4 | held-out (logs) | 20 | 20.0 | 9.5 | 0.0 | 40.0 |
| count | 0.5 | read_changelog_v4 | train (docs) | 20 | 100.0 | 1.1 | 100.0 | 100.0 |
| pe | 0.5 | count_measurements | train (data) | 20 | 100.0 | 1.1 | 100.0 | 100.0 |
| pe | 0.5 | find_errors_v4 | held-out (logs) | 20 | 0.0 | 10 | 0.0 | 0.0 |
| pe | 0.5 | read_changelog_v4 | train (docs) | 20 | 100.0 | 1.05 | 100.0 | 100.0 |
| pe | 1.0 | count_measurements | train (data) | 20 | 100.0 | 1.1 | 100.0 | 100.0 |
| pe | 1.0 | find_errors_v4 | held-out (logs) | 20 | 0.0 | 10 | 0.0 | 0.0 |
| pe | 1.0 | read_changelog_v4 | train (docs) | 20 | 100.0 | 1.05 | 100.0 | 100.0 |

## Gate verdict (per task and overall)

| task | branch | count% | pe a0.5% | delta pp | verdict | pe a1.0% | delta pp |
|---|---|---|---|---|---|---|---|
| read_changelog_v4 | train (docs) | 100.0 | 100.0 | 0.0 | PASS | 100.0 | 0.0 |
| count_measurements | train (data) | 100.0 | 100.0 | 0.0 | PASS | 100.0 | 0.0 |
| find_errors_v4 | held-out (logs) | 20.0 | 0.0 | -20.0 | FAIL | 0.0 | -20.0 |

- **PE(alpha=0.5) aggregate** (60 eps, M3/FF-CI-6 wording): count 73.3% vs PE 66.7%, delta -6.7 pp -> PASS (band -10pp); per-task FAILs: ['find_errors_v4']
- **PE(alpha=1.0) aggregate** (60 eps, M3/FF-CI-6 wording): count 73.3% vs PE 66.7%, delta -6.7 pp -> PASS (band -10pp); per-task FAILs: ['find_errors_v4']

## Verb-distribution contrast on the held-out task (find_errors_v4)

| arm | top verbs by executed count |
|---|---|
| count | {'grep': 40, 'cd': 32, 'find': 31, 'cat': 18, 'echo': 18, 'head': 15, 'wc': 15, 'ls': 11} |
| pe_a0.5 | {'cat': 48, 'head': 47, 'wc': 45, 'cd': 32, 'echo': 9, 'grep': 6, 'ls': 6, 'pwd': 6} |
| count | {'grep': 40, 'cd': 32, 'find': 31, 'cat': 18, 'echo': 18, 'head': 15, 'wc': 15, 'ls': 11} |
| pe_a1.0 | {'head': 53, 'cat': 47, 'cd': 40, 'wc': 17, 'ls': 12, 'grep': 10, 'pwd': 10, 'echo': 10} |

## Learning curve (ep11-20 vs ep1-10) and discovery steps

| agent | alpha | task | ep1-10% | ep11-20% | ratio |
|---|---|---|---|---|---|
| count | 0.5 | count_measurements | 100.0 | 100.0 | 1.00 |
| count | 0.5 | find_errors_v4 | 0.0 | 40.0 | inf |
| count | 0.5 | read_changelog_v4 | 100.0 | 100.0 | 1.00 |
| pe | 0.5 | count_measurements | 100.0 | 100.0 | 1.00 |
| pe | 0.5 | find_errors_v4 | 0.0 | 0.0 | inf |
| pe | 0.5 | read_changelog_v4 | 100.0 | 100.0 | 1.00 |
| pe | 1.0 | count_measurements | 100.0 | 100.0 | 1.00 |
| pe | 1.0 | find_errors_v4 | 0.0 | 0.0 | inf |
| pe | 1.0 | read_changelog_v4 | 100.0 | 100.0 | 1.00 |

M3-style check (both arms >= 2x improvement): see ratios above — a task already at ceiling (100% in ep1-10) trivially cannot double; the check is reported, not gated, for FF-HG-5.

## Discriminator signal: where it helps / hurts

| task | branch | success-ep mean error | failure-ep mean error | top high-error actions |
|---|---|---|---|---|
| read_changelog_v4 | train (docs) | 0.019 | None | {'cat': 1} |
| count_measurements | train (data) | 0.0545 | None | {'cat': 1, 'wc': 1} |
| find_errors_v4 | held-out (logs) | None | 0.081 | {'grep': 1, 'find': 1, 'cd': 1, 'ls': 1} |

**Train-branch tasks (docs/data): neutral (0 pp).** Both arms hit 100% by episode 1-2; once the
success cache latches the winning action, both agents replay it and the discriminator's
selection barely matters (mean per-step error 0.019-0.055 — the discriminator predicts these
familiar transitions almost perfectly). No help, no harm: the gate here tests nothing beyond the
phase-8 success-cache mechanism.

**Held-out-branch task (find_errors_v4, logs): PE 0% vs count 20% (-20 pp).** The discriminator
actively *harmed* exploration. Mechanism (verified from step records):

1. STRIPSDiscriminator's unmatched-action fallback is `confidence = 1 - success_rate(verb)`
   (plan §3.3, implemented verbatim). `cd` is a high-success verb, so any `cd` whose
   (verb, target_type, flag) schema is unseen gets `confidence ≈ 0` → **uncertainty = 1.0**.
2. In `/sandbox/logs` (a held-out branch), subdirs app/audit/system/cache are never matched by a
   trained schema, so `cd` into them scores max uncertainty; with alpha=0.5 the blend gives
   `score = 0.5*1.0 + 0.5*count_novelty` which outranks every grep candidate (mean score
   cd=0.679 vs grep=0.434; alpha=1.0: cd=0.800 vs grep=0.202).
3. Result: PE arms burn 32-40 steps/episode on `cd app/audit/system` navigation (81%/75% of cd
   steps had uncertainty=1.0) instead of grep: verb counts on find_errors_v4 are grep 6 (pe
   a0.5) and 10 (pe a1.0) vs 40 for count; first grep is often never reached within 10 steps.

The count arm does the same `cd` exploration but its tie-break priority (grep < cd) keeps the
grep candidates in contention, so it stumbles onto the 4 successful episodes.

## Verdict

| Criterion | Result |
|---|---|
| FF-HG-5 aggregate (M3/FF-CI-6 wording, 60 eps) | **PASS (marginal)**: count 73.3% vs PE 66.7%, delta -6.7 pp (within -10 pp band) |
| FF-HG-5 per-task (strict) | **FAIL**: find_errors_v4 delta -20 pp (held-out branch) |
| FF-HG-5 literal reading (`Agent-level <= count baseline -> DEAD`) | **DEAD-worthy**: PE never exceeded count on any of the 3 tasks (0/3 tasks positive delta) |

**Bottom line:** the completed FF-HG-5 definition passes only in the marginal aggregate sense
(no positive contribution, one task 20 pp below baseline). The discriminator error signal that
showed clean offline structure (MVP F1-F3 PASS, AUC 0.808) does **not** translate into a closed-
loop exploration advantage; on the held-out branch it is actively harmful because the verb-level
prior `1 - success_rate` inverts confidence for high-success unmatched actions (`cd`), turning
navigation into a novelty magnet. Count-based novelty remains the only reliable agent-level
drive signal in Phase 8/9 (consistent with the Phase 9 review conclusion).

Per the pre-registration discipline (no post-hoc threshold adjustment), the aggregate reading is
the official PASS/FAIL of this run; the per-task and mechanism evidence above is reported so the
charter-level verdict can weight the -20 pp held-out regression.

## Consistency with MVP signal validation

MVP (results/phase9_signal_validation_20260731_074345.md): AUC_disc=0.808, KL(emp||uniform)=0.600,
KL(heldout||train)=0.959, Cohen's d(E,D)=1.457, Spearman rho=0.754 — the discriminator error field
has structure and is count-orthogonal in the offline probe study. FF-HG-5 asks whether that
structured error survives the closed loop. Answer: **no**.

- Offline V1/V2/V3 measure the *post-hoc error field* on fixed probe sets. They confirm the error
  has signal, but never test whether using pre-execution *uncertainty* as a selection score
  produces better exploration — FF-HG-5 does, and it fails to.
- The MVP probe protocol never exercises the failure mode found here: its zero-visit E probes are
  `cat projects/frontend/app.js`-style novel (verb x ext) combos, not `cd` into unseen subdirs,
  so the `1 - success_rate` inversion for high-success verbs went undetected. This is a genuine
  new closed-loop failure mode, not a threshold artifact.
- The neutral train-branch result is explained by success-cache replay dominance (both arms at
  100%), not by discriminator quality; the gate's power there is low.

## Fix pointer (not part of this gate verdict)

Changing the unmatched-verb fallback to `confidence = success_rate(verb)` (or a matched-schema
prior) would remove the navigation magnet; a re-run with that fix is a follow-up experiment, not
an adjustment of this pre-registered gate.

## Artifacts

- per-episode JSONL: `results/phase9_hg_f5/phase9_hg_f5_*.jsonl` (9 runs, meta header + 20 episode records each, per-step verdict/novelty/error)
- summary CSV: `results/phase9_hg_f5/phase9_hg_f5_summary.csv`
- experiment script: `scripts/phase9_hg_f5.py`; matrix driver: `scripts/phase9_hg_f5_run_matrix.sh`; analysis: `scripts/phase9_hg_f5_analyze.py`; agent: `src/phase9/agent.py`
- run logs: `logs/phase9_hg_f5/*.log`
