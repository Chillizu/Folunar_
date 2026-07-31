# Phase 3 Evidence Bundle — Scout S5Phase3

**Date:** 2026-07-31
**Scope:** Phase 3 N=20 confirmatory (E08, E09) + Grid World GPU run (E03) + aborted CPU attempt
**experiment_ids:** [E08, E09, E03]

---

## 1. Files Read

| Path | Description |
|---|---|
| `results/phase3_sandbox_n20/ANALYSIS_REPORT.md` | E08/E09 canonical N=20 statistical report (217 lines) |
| `results/phase3_sandbox_n20/phase3_n20_result.json` | Machine verdict (p=0.004294336884693755, r=0.35) |
| `results/phase3_sandbox_n20/phase3_sandbox_n20_peda_known.jsonl` | E09 raw data (20 eps) |
| `results/phase3_sandbox_n20/phase3_sandbox_n20_peda_unknown.jsonl` | E08 raw data (20 eps) |
| `results/phase3_sandbox_n20/phase3_sandbox_n20_pragmatic_known.jsonl` | E09 raw data (20 eps) |
| `results/phase3_sandbox_n20/phase3_sandbox_n20_pragmatic_unknown.jsonl` | E08 raw data (20 eps) |
| `results/phase3_gpu/report.json` | E03 verdict JSON |
| `results/phase3_gpu/run.log` | E03 full per-episode log |
| `results/phase3_experiment/report.json` | Aborted CPU attempt (INCOMPLETE) + N=1 pilot reference |
| `results/phase3_experiment/fast_experiment.log` | Partial CPU grid-world log |
| `results/phase3_experiment/experiment.log` | Ensemble-3 CPU attempt log |
| `results/phase3_experiment/goal_known_peda.jsonl` | 2 CPU episodes |
| `results/phase3_sandbox/phase3_sandbox_n20_peda_unknown.jsonl` | Truncated duplicate (N=14) — do not double-count |
| `results/phase3_sandbox/phase3_sandbox_peda_unknown.jsonl` | N=5 pilot duplicate |
| `scripts/phase3_n20_runner.py` | tmux sequential runner |
| `scripts/phase3_n20.sh` | Bash sequential runner |
| `scripts/phase3_sandbox_experiment.py` | E08/E09 experiment driver |
| `scripts/phase3_n20_analysis.py` | E08 stats (MW, Fisher, rank-biserial r) |
| `scripts/phase3_analysis_sandbox.py` | Earlier sandbox analysis (superseded) |
| `scripts/phase3_fast.py` | E03 GPU script (confidence-based epistemic) |
| `scripts/phase3_analysis.py` | E03 stats helpers (fisher, MW, Cohen's h/d) |
| `scripts/phase3_run_all.py` | Earlier sandbox v2 variant (superseded) |
| `scripts/phase2_collect_data.py` | `compute_metrics` (fht/scr/dead_loop_rate) lines 206-231 |
| `scripts/phase3_run_tmux.py`, `phase3_run_all.sh`, `phase3_experiment.py` | Not read in full; superseded/orchestration only |

---

## 2. Evidence Bundles (E-ID: number | source:line | verbatim quote)

### E08: N=20 DESIGN (sandbox confirmatory, 2026-07-28)

- (80 episodes, 4 conditions x N=20, task `read_hello`, adapter `sandbox_adapter_v2_full`, NVIDIA T4 g4dn.xlarge us-east-1) | `results/phase3_sandbox_n20/ANALYSIS_REPORT.md:14-20` | conditions table: `| PEDA | known | /sandbox, /sandbox/data, /sandbox/docs | 20 | PEDA agent on familiar CWDs (seen during training) |`; `| PEDA | unknown | /sandbox/logs, /sandbox/projects, /sandbox/tmp | 20 | PEDA agent on novel CWDs (not seen during training) |`; Pragmatic rows identical (lines 19-20)
- (7,7,6 round-robin per CWD) | `ANALYSIS_REPORT.md:25` | `CWDs are counterbalanced across conditions — each condition sees the same three CWDs in the same round-robin pattern (7, 7, 6 episodes per CWD)`
- (seed = ep_idx, cwd = cwds[ep % 3], max_steps=10) | `scripts/phase3_sandbox_experiment.py:102-104` | `cwd = cwds[ep_idx % len(cwds)]` / `seed = args.seed_offset + ep_idx`

### E08: RESULT TABLE

- (100.0%, 10.00 steps, SD 0.00, dlr 0.00, 302.5s) PEDA known | `ANALYSIS_REPORT.md:39` | `| PEDA known | 100.0% | 10.00 | 0.00 | 0.00 | 302.5 |`
- (100.0%, 7.20, 3.91, 0.00, 203.5s) PEDA unknown | `ANALYSIS_REPORT.md:40` | `| PEDA unknown | 100.0% | 7.20 | 3.91 | 0.00 | 203.5 |`
- (100.0%, 6.85, 4.40, 0.52, 129.6s) Pragmatic known | `ANALYSIS_REPORT.md:41` | `| Pragmatic known | 100.0% | 6.85 | 4.40 | 0.52 | 129.6 |`
- (100.0%, 10.00, 0.00, 0.80, 159.2s) Pragmatic unknown | `ANALYSIS_REPORT.md:42` | `| Pragmatic unknown | 100.0% | 10.00 | 0.00 | 0.80 | 159.2 |`
- PEDA unknown raw steps [10,2,10,10,2,...] (13x10 + 7x2, mean 7.2) | `ANALYSIS_REPORT.md:60` | `- **PEDA unknown steps:** [10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2, 10, 10, 2]` — byte-verified against `phase3_sandbox_n20_peda_unknown.jsonl:1-20`
- Pragmatic unknown raw 20x10 | `ANALYSIS_REPORT.md:61` | `- **Pragmatic unknown steps:** [10, 10, ... (x20)]`
- (10.0 / 10.0) medians — equal on both sides | `ANALYSIS_REPORT.md:68` | `| Median steps | 10.0 | 10.0 |`

### E08: ALL P-VALUES

- (0.0043) MW two-sided PEDA-unknown vs Pragmatic-unknown | `ANALYSIS_REPORT.md:72` | `- **p-value (two-sided):** 0.0043`; raw machine value 0.004294336884693755 at `phase3_n20_result.json:12`
- (0.0021) MW one-sided (PEDA < Pragmatic) | `ANALYSIS_REPORT.md:73` | `- **p-value (one-sided, PEDA < Pragmatic):** 0.0021`
- (130.0) MW U statistic | `ANALYSIS_REPORT.md:71` | `- **Mann-Whitney U:** 130.0`
- (0.0001) crossover interaction MW U=315.5 | `ANALYSIS_REPORT.md:120` | `**Interaction Mann-Whitney U** (advantage_unknown > advantage_known): U = 315.5, **p = 0.0001**`
- (0.0004) per-CWD `/sandbox/projects` MW U=0.0 | `ANALYSIS_REPORT.md:91` | `| /sandbox/projects | 2.00 | 10.00 | 0.0 | 0.0004 |`
- (24.5) per-CWD `/sandbox/logs` U (identical) | `ANALYSIS_REPORT.md:90` | `| /sandbox/logs | 10.00 | 10.00 | 24.5 | — (identical) |`
- (18.0) per-CWD `/sandbox/tmp` U (identical) | `ANALYSIS_REPORT.md:92` | `| /sandbox/tmp | 10.00 | 10.00 | 18.0 | — (identical) |`
- (r=0.35, medium) rank-biserial | `phase3_n20_result.json:14` | `verdict: PEDA outperforms Pragmatic on goal-unknown: equal 100% success rate, but PEDA achieves it in significantly fewer steps (MW p=0.0043, r=0.35, medium effect)`

### E08: COHEN'S d

- (-1.01) d, PEDA unknown vs Pragmatic unknown | `ANALYSIS_REPORT.md:74` | `- **Cohen's d:** -1.01 (large effect; negative because PEDA has fewer steps)` — recomputed from raw data: pooled SD = sqrt(19*3.91^2/38) = 2.765; d = (7.2-10.0)/2.765 = -1.013 [VERIFIED]

### E08: PER-CWD BREAKDOWN

- `/sandbox/projects` 2.00 vs 10.00, all 7 episodes in 2 steps | `ANALYSIS_REPORT.md:96` | `PEDA dramatically outperforms Pragmatic (2 vs 10 steps, p = 0.0004). All 7 episodes in this CWD completed in 2 steps. This is the key CWD where PEDA's generalization pays off — the World Model maps /sandbox/projects to its nearest training CWD and navigates directly.`
- `/sandbox/logs` + `/sandbox/tmp` ceiling 10 both | `ANALYSIS_REPORT.md:97` | `Both agents take the ceiling 10 steps. PEDA's World Model does not provide useful generalization for these CWDs, so it falls back to exhaustive search identical to the baseline.`
- Effect concentrated in one CWD | `ANALYSIS_REPORT.md:98` | `The effect is entirely concentrated in /sandbox/projects, but it is perfectly reliable (zero variance) across all 7 repetitions.`

### E08: FHT (fishhook table / first hitting time) METRIC

- fht definition | `scripts/phase2_collect_data.py:206-219` | `def compute_metrics(steps, task_id): ... if task["check"](fake_ns, rec["action"], fake_ns): fht = rec["step"]; break` — fht = step index of first action passing task check; -1 if never. Docstring line 207: `Compute FHT, SCR, Dead-loop Rate from step records.`
- fht distribution in N=20 data: 7 x fht=1 (PEDA unknown /sandbox/projects), 7 x fht=0 (Pragmatic known /sandbox), 66 x fht=-1 | `phase3_sandbox_n20_peda_unknown.jsonl:2` | `{"baseline": "peda", "condition": "unknown", "cwd": "/sandbox/projects", ..., "steps_count": 2, "success": true, "fht": 1, "scr": 0.5, ...}` — task ACTUALLY completed in only 14/80 episodes
- scr definition | `phase2_collect_data.py:221-224` | `visited.add(f"{rec['cwd']}|{tuple(rec['files'])}")` / `scr = len(visited) / max(len(steps), 1)` — scr>0 for ANY episode with >=1 step
- dead_loop_rate definition | `phase2_collect_data.py:225-229` | `for i in range(2, len(steps)): if steps[i]["action"] == steps[i-1]["action"] == steps[i-2]["action"]: loops += 1`

### E09: NEGATIVE CONTROL (PEDA known vs Pragmatic known)

- (270.0, 0.0043, 1.01) MW U / p / d | `ANALYSIS_REPORT.md:150-152` | `- **Mann-Whitney U:** 270.0` / `- **p-value (two-sided):** 0.0043` / `- **Cohen's d:** 1.01 (large effect; PEDA takes more steps)`
- PEDA known raw 20x10 deterministic | `ANALYSIS_REPORT.md:139` | `- **PEDA known steps:** [10, 10, ... (x20)]`
- Pragmatic known raw 7x1 + 13x10, mean 6.85 | `ANALYSIS_REPORT.md:140` | `- **Pragmatic known steps:** [1, 10, 10, 1, 10, 10, ...]` — mean verified: (7*1+13*10)/20 = 6.85 [VERIFIED]
- Interpretation | `ANALYSIS_REPORT.md:160` | `This result is **not a bug**: it confirms that PEDA pays a cost for epistemic modeling in familiar settings, a cost that is recouped in novel settings (see Section 5: crossover interaction p = 0.0001).`
- Crossover cell means | `ANALYSIS_REPORT.md:108-109` | `| **PEDA** | 10.00 | 7.20 |` / `| **Pragmatic** | 6.85 | 10.00 |`
- Advantage flip (-3.15 known / +2.80 unknown) | `ANALYSIS_REPORT.md:115-116` | `| Known | −3.15 (Pragmatic better) |` / `| Unknown | +2.80 (PEDA better) |`
- Crossover interpretation | `ANALYSIS_REPORT.md:129` | `This crossover pattern is the central finding of the experiment: PEDA trades a small cost in familiar environments for a substantial benefit in novel ones, exactly as the epistemic grounding hypothesis predicts.`

### E08/E09: VERBATIM VERDICT + CAVEATS

- Verdict | `ANALYSIS_REPORT.md:199` | `**Yes.** PEDA in unknown CWDs requires significantly fewer steps than the Pragmatic baseline (p = 0.0043, d = -1.01). The effect is large and reliable.`
- Single-task caveat | `ANALYSIS_REPORT.md:210` | `1. **Single task.** All episodes used read_hello. Generalization to other tasks is not yet demonstrated.`
- Single-novel-CWD caveat | `ANALYSIS_REPORT.md:211` | `2. **Single novel CWD.** The effect is driven entirely by /sandbox/projects. The other two novel CWDs (/sandbox/logs, /sandbox/tmp) show no PEDA advantage. The World Model's generalization is selective, not uniform.`
- Ceiling-effect caveat | `ANALYSIS_REPORT.md:212` | `3. **Ceiling effects.** Pragmatic in unknown and PEDA in known both hit the 10-step ceiling. The true difference in unknown may be larger if Pragmatic were measured without the 10-step cap...`
- Dead-loop confound caveat | `ANALYSIS_REPORT.md:213` | `4. **Dead-loop rate.** Pragmatic's high dead-loop rate (0.52–0.80) is the mechanism behind its higher step count in unknown environments. PEDA's zero dead-loop rate is a correlated benefit of the World Model's epistemic grounding. The causal link requires further decomposition.`
- Summary quote | `ANALYSIS_REPORT.md:217` | `> Phase 3 provides strong confirmatory evidence for the epistemic validation hypothesis. PEDA's World Model generalizes to novel CWDs, yielding a statistically significant and practically large reduction in steps (d = −1.01, p = 0.0043). The crossover interaction (p = 0.0001) confirms that this benefit is specific to unfamiliar environments. The negative control behaves as expected, ruling out a trivial advantage. Further work should expand the task set and the range of novel CWDs.`

### E03: GRID WORLD GPU RUN (phase3_gpu, 2026-07-27, N=10/condition)

- (2.6 vs 2.6 steps, Fisher p=1.0000, MW p=1.0000) goal_unknown | `results/phase3_gpu/report.json` statistical_tests | `goal_unknown_success_fisher: p_value 1.0, peda 10.0/10 (100%), pragmatic 10.0/10 (100%)`; `goal_unknown_steps_mannwhitney: p_value 1.0, peda_mean_steps 2.6, pragmatic_mean_steps 2.6`
- (3.3 vs 3.3, std 1.35, median 4, min 1, max 6) goal_known identical; fairness p=1.0000 | `report.json` | `goal_known: peda and pragmatic_only both mean_steps 3.3, std_steps 1.35, median_steps 4, min_steps 1, max_steps 6`; `goal_known_fairness: p_value 1.0, fairness_pass true`
- (2.6, std 1.85, median 2, min 1, max 6, revisit 0.0) goal_unknown stats | `report.json` | `goal_unknown: peda n 10, success_rate 1.0, mean_steps 2.6, std_steps 1.85, median_steps 2, min_steps 1, max_steps 6, mean_revisit_rate 0.0` (pragmatic_only identical)
- PEDA and Pragmatic steps IDENTICAL in all 20 episodes | `results/phase3_gpu/run.log:11-31` | `ep0: condition=goal_known goal=(4, 4) start=(0, 4) PEDA steps=4 success=True | Prag steps=4 success=True` ... `ep9: condition=goal_unknown goal=(3, 4) start=(3, 0) PEDA steps=4 success=True | Prag steps=4 success=True`
- (3/7) success criteria passed | `report.json` | `passed_criteria: 3/7` — passed: peda_goal_unknown_success_gt_60pct, peda_goal_unknown_steps_lt_10, goal_known_fairness_pass; failed: prag_goal_unknown_success_lt_40pct, prag_goal_unknown_steps_gt_15, fisher_p_lt_005, mannwhitney_p_lt_005
- Verdict | `report.json` | `verdict: CORE_HYPOTHESIS_NOT_SUPPORTED, verdict_confidence: N/A, verdict_reason: PEDA 10.0/10 (100%) vs Pragmatic 10.0/10 (100%) success in goal_unknown (Fisher p=1.0000, MW p=1.0000). Goal_known fairness: PEDA 10.0/10 vs Pragmatic 10.0/10 (p=1.0000)`
- (876s / 14.6min) T4 runtime | `run.log:36` | `All episodes complete in 876s (14.6min)`
- Methodology: empty ensemble, epistemic = 1 - model confidence | `scripts/phase3_fast.py:15-19` | `Epistemic signal comes from model's own confidence (epistemic_ratio = 1.0 - confidence), NOT from ensemble variance`; `ec.checkpoints = []`
- Model/conditions | `report.json` + `phase3_fast.py` | `model: Qwen/Qwen2.5-0.5B-Instruct, adapter: checkpoints/phase1/partial_adapter_real_25_e3, train_fraction: 0.25, ensemble_checkpoints: 0, drive_weights: curiosity 0.5, competence 0.5, boredom 0.5, novelty 0.5, pragmatic_weight: 3.0, episodes_per_condition: 10, max_candidates 4, horizon 1, MAX_STEPS 50`
- (6/25) known cells | `run.log:3` | `[phase3] Known cells: 6 / 25`

### ABORTED CPU ATTEMPT (context for E03)

- (176s) cold start, (~3s) warm, (12-24) inference calls/step, (10-60+ min)/episode, (60-120 h) projected total | `results/phase3_experiment/report.json` | `First LLM inference call takes approximately 176 seconds (cold start + PyTorch JIT compilation). Subsequent calls stabilize at ~3s each on CPU. However, with 4-8 candidates per step and 3 ensemble checkpoints, each step requires 12-24 model inference calls, making a single episode take 10-60+ minutes` / `Without GPU, the experiment is impractical (estimated 60-120 hours total on this CPU)`
- N=1 pilot (directional only): goal_known 3.0/100% both; goal_unknown PEDA 2.0/100% vs Pragmatic 20.0/0%; g1 test set 0.8684 | `report.json` | `goal_unknown: {peda: 2.0 steps/100%, pragmatic: 20.0 steps/0%}` / `Pilot N=1 shows directional signal but insufficient for statistical inference. Required N>=10 per condition.`
- CPU hardware | `report.json` | `cpu: Intel Core Ultra 9 185H (22 cores), gpu: None (CUDA: false), ram: 30GB (18GB available during experiment)`

---

## 3. Experimental Conditions (summary table)

| Experiment | Env | Task | Model | Adapter | N | Hardware | Max steps | Epistemic source |
|---|---|---|---|---|---|---|---|---|
| E08 (N=20 confirm) | Docker sandbox v2 | read_hello | Qwen2.5-0.5B-Instruct | sandbox_adapter_v2_full | 20/cond (80 total) | T4 (g4dn.xlarge) | 10 | WM full pipeline (adapter) |
| E09 (N=20 neg control) | Docker sandbox v2 | read_hello | Qwen2.5-0.5B-Instruct | sandbox_adapter_v2_full | 20/cond | T4 | 10 | same as E08 |
| E03 (GPU grid world) | GridWorld 5x5 | reach goal cell | Qwen2.5-0.5B-Instruct | partial_adapter_real_25_e3 (6/25 cells) | 10/cond | T4 | 50 | model confidence (empty ensemble) |
| CPU attempt (aborted) | Sandbox v2 / GridWorld | read_note / goal | Qwen2.5-0.5B-Instruct | sandbox_adapter_v2_partial / partial_adapter_real_25_e3 | 0 (incomplete) | CPU Ultra 9 185H | 50 | n/a |

---

## 4. Methodology (implementation details)

1. **E08/E09 driver** (`scripts/phase3_sandbox_experiment.py`): runs `run_peda`/`run_pragmatic` from `phase2_collect_data` with `--baseline {peda,pragmatic} --condition {known,unknown} --num-episodes 20 --max-steps 10`. CWDs: KNOWN = [/sandbox, /sandbox/data, /sandbox/docs], UNKNOWN = [/sandbox/logs, /sandbox/projects, /sandbox/tmp]. Episode cwd cycles round-robin; seed = ep_idx (0-19). Writes JSONL incrementally.
2. **Metrics** (`phase2_collect_data.py:206-231`): fht = step index of first action passing task check (or -1); scr = |unique (cwd, files) states| / steps; dead_loop_rate = fraction of steps i>=2 where actions[i]==actions[i-1]==actions[i-2]; success := scr > 0 (weak proxy — trivially true).
3. **Statistics** (`scripts/phase3_n20_analysis.py`): scipy `mannwhitneyu` two-sided on steps; Fisher one-sided on success; rank-biserial r = 1 - 2U/(n1*n2). Writes `phase3_n20_result.json`.
4. **E08 report extras** (ANALYSIS_REPORT.md sections 4-6: per-CWD MW U, crossover interaction MW on advantage scores, Cohen's d with pooled SD): computed ad hoc; NO script in repo implements them (grep verified).
5. **E03 runner** (`scripts/phase3_fast.py`): loads WM once, empty ensemble (epistemic = 1 - confidence), DriveWeights all 0.5, pragmatic_weight 3.0, max_candidates 4, horizon 1, MAX_STEPS 50; per episode samples goal (known cell for goal_known, held-out cell for goal_unknown) + untrained start cell; runs PEDA then Pragmatic-only on the SAME episode seed; 7 success criteria; verdict rules in code.
6. **Run orchestration** (`scripts/phase3_n20_runner.py` / `phase3_n20.sh`): sequential 4-condition runs in tmux, python `/opt/pytorch/bin/python` on `/home/ec2-user/Folunar_`.

---

## 5. Contradictions / Gaps

1. **[CRITICAL] 'success' is vacuous in E08/E09.** `phase3_sandbox_experiment.py:133` sets `success = metrics["scr"] > 0`; scr >= 1/10 > 0 for any episode with >=1 step. So `All 80 episodes completed with success: true` (ANALYSIS_REPORT.md:26) does not mean tasks were completed. Real completion (fht >= 0) happened in 14/80: 7 PEDA-unknown-/sandbox/projects (fht=1, 2 steps) + 7 Pragmatic-known-/sandbox (fht=0, 1 step). In the unknown condition the completion counts are PEDA 7 vs Pragmatic 0 — the only genuine (and tiny) signal.
2. **[CRITICAL] E08 'supported' conflicts with E03 and the final verdict.** E03 (cleaner confidence-based epistemic test, matched trajectories) is a null: p=1.0000, verdict CORE_HYPOTHESIS_NOT_SUPPORTED, 3/7 criteria. The E08 effect is one-CWD adapter generalization (`World Model maps /sandbox/projects to its nearest training CWD`, line 96) — candidate/layout engineering, not prediction-error-driven exploration. Report caveat 2 (line 211) concedes selectivity. Consistent with hypothesis DISPROVEN (3 charter Q No).
3. Per-CWD p=0.0004 (/sandbox/projects; complete separation 7x2 vs 7x10): exact two-sided MW p = 2/C(14,7) = 0.00058 (scipy), one-sided = 0.00029; 0.0004 matches neither [INFERENCE — not reproducible from repo scripts].
4. Crossover (U=315.5, p=0.0001) and both Cohen's d values lack a repo script; d=-1.01 and d=1.01 independently verified from raw data; crossover U plausible for advantage vectors [7x(-9),13x0] vs [7x8,13x0] [INFERENCE].
5. Dead-loop confound acknowledged (line 213): Pragmatic step inflation is 3-consecutive-same-action loops (dlr 0.52-0.80); PEDA never dead-loops. Wall-clock favors Pragmatic (129.6/159.2s vs 203.5/302.5s — PEDA ~2x slower), so the step-count 'advantage' is not an efficiency win.
6. Ceiling effects: 66/80 episodes hit the 10-step cap; medians equal at 10.0 in the primary comparison — only the mean differs (driven by 7 two-step episodes).
7. Duplicate data: `results/phase3_sandbox/phase3_sandbox_n20_peda_unknown.jsonl` is a truncated copy (N=14, byte-identical prefix); `phase3_sandbox_peda_unknown.jsonl` is the N=5 pilot. Paper must cite only `results/phase3_sandbox_n20/*`.
8. E08 (adapter WM pipeline) and E03 (confidence-based) test different epistemic operationalizations; E03 is the cleaner null.
9. Machine verdict (`phase3_n20_result.json`) omits d, per-CWD, and crossover stats that appear only in the hand-written ANALYSIS_REPORT.md.
