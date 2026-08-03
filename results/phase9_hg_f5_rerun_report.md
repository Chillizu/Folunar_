# Phase 9 FF-HG-5 Rerun — after unmatched-verb fallback fix (E2)

- fix commit: `cd1478a` (`phase9 HG: fix STRIPS unmatched-verb fallback confidence — use verb success rate (0.5 neutral) instead of 1 - success_rate inversion`)
- rerun code state: `cd1478a` (run-time HEAD at rerun, matches per-episode JSONL meta); rerun artifacts + this report committed as the `phase9 HG: FF-HG-5 rerun` commit (final hash reported to Main)
- timestamp: 2026-08-03
- host: local (cpu) | sandbox_image: `peda-sandbox:v4`
- episodes per run: 20 | max_steps: 10 | seeds: 0..19 (identical to original F5)

## 1. Fix diff (root-cause repair, separate commit cd1478a)

`src/phase9/discriminator.py`, `STRIPSDiscriminator.confidence()` unmatched-action branch:

```diff
-        # unmatched action -> verb-level prior
-        verb, _, _ = self._parse_action(action)
-        return 1.0 - self.verb_success_rate(verb)
+        # unmatched action -> verb-level prior: use the verb's observed success
+        # rate directly (0.5 neutral when unobserved). The previous
+        # `1 - success_rate` inversion turned high-success verbs (cd is
+        # required in every L1 task, ~100% success) into ~0-confidence
+        # navigation magnets, starving grep on the held-out branch
+        # (FF-HG-5 find_errors_v4 regression root cause).
+        verb, _, _ = self._parse_action(action)
+        return self.verb_success_rate(verb)
```

Class docstring updated to match (`unmatched action => verb-level prior success_rate(verb) (0.5 neutral when unobserved)`).
Total diff: **9 insertions / 3 deletions (12 lines)** — within the ~20-line budget.

**Motivation (comment in code):** the F5 report (§mechanism) showed the old fallback
`confidence = 1 − success_rate(verb)` gave unseen-but-required actions of high-success
verbs (cd ~100% success) `confidence ≈ 0` → `uncertainty ≈ 1.0`, so the discriminator
blend (alpha=0.5 and alpha=1.0) ranked `cd` into held-out subdirs above every grep
candidate and burned 32–40 steps/episode on navigation. The fix makes the fallback
prior the verb's *empirical* success rate: `cd` reads as high-confidence (low
uncertainty) and stops being a novelty magnet; verbs with no observations keep the
neutral 0.5 prior.

## 2. Rerun protocol — item-by-item parity with original F5

| Parameter | Original F5 (commit ceed3ee) | Rerun (commit cd1478a) | Match |
|---|---|---|---|
| tasks | read_changelog_v4, count_measurements, find_errors_v4 | identical 3 tasks | ✅ |
| episodes per (agent, task) | 20 | 20 | ✅ |
| max_steps | 10 | 10 | ✅ |
| seeds | 0..19 (reset(seed=episode)) | 0..19 (same reset call) | ✅ |
| image | peda-sandbox:v4 | peda-sandbox:v4 | ✅ |
| agent arms | count a0.5, pe a0.5, pe a1.0 | pe a0.5, pe a1.0 (count baseline **referenced**, not rerun) | ✅ (contract: count 基线引用原值不重跑) |
| runner | scripts/phase9_hg_f5.py | same script, unchanged | ✅ |
| driver | phase9_hg_f5_run_matrix.sh | phase9_hg_f5_rerun_matrix.sh (new, same args) | ✅ |
| count baseline numbers | recomputed from results/phase9_hg_f5/*.jsonl | cited from the same original JSONLs (below) | ✅ |
| output | results/phase9_hg_f5/…jsonl | results/phase9_hg_f5_rerun/phase9_hg_f5_pe_*.jsonl (6 files, meta + 20 eps each) | ✅ WATCHDOG D4 |

Count baseline (recomputed from original F5 JSONLs, not rerun):
read_changelog_v4 **100.0%** · count_measurements **100.0%** · find_errors_v4 **20.0%** · aggregate **73.3% (44/60)**.

## 3. Per-run summary (rerun PE vs original count vs original PE)

| agent | alpha | task | branch | eps | count% | pe-old% | **pe-new%** | avg steps (new) |
|---|---|---|---|---|---|---|---|---|
| pe | 0.5 | read_changelog_v4 | train (docs) | 20 | 100.0 | 100.0 | **80.0** | 2.85 |
| pe | 0.5 | count_measurements | train (data) | 20 | 100.0 | 100.0 | **100.0** | 1.05 |
| pe | 0.5 | find_errors_v4 | held-out (logs) | 20 | 20.0 | 0.0 | **30.0** | 8.65 |
| pe | 1.0 | read_changelog_v4 | train (docs) | 20 | 100.0 | 100.0 | **0.0** | 10.00 |
| pe | 1.0 | count_measurements | train (data) | 20 | 100.0 | 100.0 | **100.0** | 1.05 |
| pe | 1.0 | find_errors_v4 | held-out (logs) | 20 | 20.0 | 0.0 | **65.0** | 7.20 |

Per-episode success patterns (1=OK, 0=FAIL; 20 episodes left→right):

| arm | task | pattern |
|---|---|---|
| pe a0.5 | read_changelog_v4 | `00001111111111111111` |
| pe a1.0 | read_changelog_v4 | `00000000000000000000` |
| pe a0.5 | count_measurements | `11111111111111111111` |
| pe a1.0 | count_measurements | `11111111111111111111` |
| pe a0.5 | find_errors_v4 | `11100000010010000100` |
| pe a1.0 | find_errors_v4 | `11111011010101010101` |

## 4. Gate verdict (pre-registered, original FF-HG-5 thresholds — unchanged)

Contract pre-registration: **PE aggregate < count → DEAD (fix ineffective, root-cause
hypothesis rejected)**; **PE aggregate ≥ count AND held-out not degraded →
ALIVE-auxiliary**.

Aggregates (60 eps, M3/FF-CI-6 wording — same as original F5 report):

| arm | count% | PE% | delta pp |
|---|---|---|---|
| pe a0.5 | 73.3 | **70.0 (42/60)** | **−3.3** |
| pe a1.0 | 73.3 | **55.0 (33/60)** | **−18.3** |

Per-task gate (PE a0.5 vs count, −10pp band from FF-HG-5 operationalization):

| task | count% | pe-new a0.5% | delta pp | verdict |
|---|---|---|---|---|
| read_changelog_v4 | 100.0 | 80.0 | −20.0 | FAIL |
| count_measurements | 100.0 | 100.0 | 0.0 | PASS |
| find_errors_v4 | 20.0 | 30.0 | +10.0 | PASS (was FAIL at −20.0) |

**Verdict: DEAD maintained.** PE aggregate (a0.5 = 70.0%) is still below count
(73.3%), so the ALIVE-auxiliary condition (`PE aggregate ≥ count AND held-out 不劣化`)
is **not** met — the held-out branch *is* no longer degraded (0%→30% a0.5, 0%→65%
a1.0, both ≥ count 20%), but the aggregate fails on the train-branch regression
(read_changelog_v4). The root-cause hypothesis (cd prior inversion) is **partially
confirmed**: the navigation magnet is gone, and PE now *beats* count on the held-out
task at both alphas. But the minimal fix is insufficient to overturn the gate.

### Why the aggregate stays below count — symmetric failure mode

The `success_rate(verb)` fallback fixed the cd magnet but suppressed exploration of
**unseen targets of high-success verbs** — which is precisely the shape of the
train-branch goal actions:

- read_changelog_v4 goal = `cat changelog.txt`. `cat` is a high-success verb, and
  `cat_<changelog.txt target>` is unmatched on first contact. Old code: confidence =
  1 − 1.0 = 0.0 → uncertainty 1.0 → explored at t1 of ep0 → success cache latches →
  100%. New code: confidence = success_rate(cat) = 1.0 → uncertainty 0.0 → never
  selected (a1.0: 0/20, cache never latches) or delayed until count novelty +
  `find`-generated candidates rediscover it (a0.5: cache latches at ep4 → 80%).
- count_measurements goal (`wc -l …measurements_01.csv`) is matched by an existing
  `wc` schema from ep0 and stays at 100% — confirming the effect is specific to
  *unmatched* goal actions, not task difficulty.
- Held-out find_errors_v4: cd into unseen subdirs is no longer a magnet — cd steps
  fell 32→26 (a0.5) and 40→14 (a1.0, all remaining are `cd ..`), grep rose 6→62
  (a0.5) and 10→87 (a1.0) → success 30%/65%.

So the verb-level prior conflates "verb is reliable" with "this exact action is
known": a single scalar per verb cannot simultaneously (a) stop over-exploring
reliable-verb navigation and (b) keep exploring novel targets of reliable verbs.
A matched-schema prior (only apply success-rate confidence to *known*
(verb, target_type, flag) schemas, keep a moderate exploration prior for genuinely
novel combos) would be the next candidate — but that is a second fix, outside this
pre-registered gate.

## 5. Updated interpretation of "PE per-task variance is high"

Original F5 reading: variance was branch-shaped — train branches at ceiling
(100/100, success-cache replay), held-out at 0 (navigation magnet). After the fix
the variance **persists but flips polarity**:

- a0.5: 80 / 100 / 30 — docs task now the weak branch (goal action starved for 4
  episodes), held-out the strong one.
- a1.0: 0 / 100 / 65 — pure-discriminator arm collapses on the docs task (goal
  action never explored) while dominating the held-out task.

Interpretation update: the per-task variance is **not branch-determined and not task
difficulty**; it tracks whether the goal action's (verb × target) combo is matched by
a learned schema. Tasks whose goal is schema-matched (count_measurements) are robust
at any alpha; tasks whose goal is *unmatched* (read_changelog_v4 = `cat` novel
target; before the fix, find_errors_v4 = `cd` novel target) live or die by which
direction the verb prior pushes uncertainty. This makes the discriminator's
agent-level value sensitive to candidate-coverability rather than to the held-out
split itself.

## 6. Independent recompute verification

`scripts/phase9_hg_f5_rerun_recompute.py` recomputes every number above from the raw
per-episode JSONLs (no shared state with the runner). Key outputs:

```
read_changelog_v4      a0.5: count 100.0% | pe-old 100.0% | pe-new  80.0%
read_changelog_v4      a1.0: count 100.0% | pe-old 100.0% | pe-new   0.0%
count_measurements     a0.5: count 100.0% | pe-old 100.0% | pe-new 100.0%
count_measurements     a1.0: count 100.0% | pe-old 100.0% | pe-new 100.0%
find_errors_v4         a0.5: count  20.0% | pe-old   0.0% | pe-new  30.0%
find_errors_v4         a1.0: count  20.0% | pe-old   0.0% | pe-new  65.0%
AGGREGATE: count 73.3% (44/60); pe a0.5 70.0% (42/60) Δ-3.3pp; pe a1.0 55.0% (33/60) Δ-18.3pp
```

(Full output in the command transcript; numbers cross-check against §3/§4 tables.)

## 7. Artifacts

- per-episode JSONL (WATCHDOG D4, meta header incl. commit cd1478a + 20 records each):
  `results/phase9_hg_f5_rerun/phase9_hg_f5_pe_a0.5_{read_changelog_v4,count_measurements,find_errors_v4}.jsonl`,
  `…pe_a1.0_…jsonl` (6 files)
- this report: `results/phase9_hg_f5_rerun_report.md`
- recompute script: `scripts/phase9_hg_f5_rerun_recompute.py`; driver: `scripts/phase9_hg_f5_rerun_matrix.sh`
- run logs: `logs/phase9_hg_f5_rerun/*.log`

## 8. Bottom line

| Criterion | Result |
|---|---|
| Root-cause hypothesis (cd prior inversion) | **Partially confirmed** — held-out magnet removed; PE beats count on find_errors_v4 at both alphas (30%/65% vs 20%) |
| Pre-registered gate (PE aggregate ≥ count AND held-out OK) | **NOT met** — a0.5 aggregate 70.0% < count 73.3% → **DEAD maintained** |
| Fix side effect | Symmetric failure: unseen targets of high-success verbs are no longer explored → read_changelog_v4 100%→80% (a0.5) / 100%→0% (a1.0) |
| Direction status | unchanged: not ALIVE-auxiliary; count-driven novelty remains the only reliable agent-level drive signal |

Honest reading: the fallback *direction* was correct (using success-rate instead of
its complement) and materially fixed the held-out branch, but the minimal one-line
prior is too coarse to also preserve novel-target exploration, so the agent-level
gate remains DEAD. Recorded as **negative/mixed** per pre-registration discipline —
no threshold adjustment, no reinterpretation.
