# Phase 8 Evidence Report — Count-Driven Agent

**Scout**: S10Phase8
**experiment_ids**: [E18, E19]
**Date**: 2026-07-31
**Canonical source**: `results/phase8_gpu_run_2026-07-31.md`

---

## 1. Files Read

| Path | Role |
|---|---|
| `/home/chillizu/Projects/Folunar_/results/phase8_gpu_run_2026-07-31.md` | CANONICAL Phase 8 results (70 lines) |
| `/home/chillizu/Projects/Folunar_/src/phase8/count_driven_agent.py` | Phase8Runner (200 lines) |
| `/home/chillizu/Projects/Folunar_/src/phase8/__init__.py` | empty package marker |
| `/home/chillizu/Projects/Folunar_/scripts/phase8_closed_loop.py` | CLI entrypoint (84 lines) |
| `/home/chillizu/Projects/Folunar_/src/phase5/explorer.py` | NoveltyExplorer — count novelty + success cache |
| `/home/chillizu/Projects/Folunar_/src/phase5/action_model.py` | ActionModelLearner — STRIPS schemas (409 lines) |
| `/home/chillizu/Projects/Folunar_/src/phase5/jepa_wm.py` | JEPAEnsemble — frozen Qwen 0.5B + 3 MLP predictors |
| `/home/chillizu/Projects/Folunar_/src/phase2/sandbox_env.py` | BusyboxSandbox, SandboxState.state_hash, candidate generator |
| `/home/chillizu/Projects/Folunar_/src/phase2/tasks.py` | MICRO_TASKS + goal predicates for all 9 tasks |
| `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md` | cross-reference, Phase 8 rows :17, :58, :106, declaration |
| `/home/chillizu/Projects/Folunar_/PEDA_FINAL/COUNT_DRIVEN_CHARTER.md` | inherited components, discarded JEPA |
| `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md` | SUPERSEDED for Phase 5–8 (header :2–5 confirms) |

---

## 2. Quantitative Results — (number, source_file:line, verbatim_quote)

### E18 — Count-only, all 9 tasks (success % + avg steps)

| Task | Image | Success | Avg steps | Source |
|---|---|---|---|---|
| read_hello | v2 | 100% (5/5) | 1.2 | results/phase8_gpu_run_2026-07-31.md:23 |
| read_note | v2 | 20% (1/5) | 8.8 | :24 |
| count_lines | v2 | 0% (0/5) | 10.0 | :25 |
| find_secret | v2 | 100% (5/5) | 1.6 | :26 |
| read_welcome | v4 | 100% (5/5) | 1.4 | :27 |
| find_api_key | v4 | 20% (1/5) | 10.0 | :28 |
| count_measurements | v4 | 100% (5/5) | 1.6 | :29 |
| find_errors_v4 | v4 | 20% (1/5) | 10.0 | :30 |
| read_changelog_v4 | v4 | 100% (5/5) | 1.2 | :31 |
| **TOTAL** | | **62.2% (28/45)** | (not recorded) | :32 |

- E18: (62.2%, 28/45) `results/phase8_gpu_run_2026-07-31.md:32` — `"| | **TOTAL** | | **28/45 (62.2%)** | |"`

### E19 — Count+JEPA, all 9 tasks (identical)

- E19: (62.2%, 28/45) `results/phase8_gpu_run_2026-07-31.md:56` — `"| | **TOTAL** | | **28/45 (62.2%)** | |"`; table rows :47–55 byte-identical to count-only (:23–31).

### JEPA Delta (per metric)

- E19: (0) `results/phase8_gpu_run_2026-07-31.md:62` — `"| Total success | 28/45 (62.2%) | 28/45 (62.2%) | **0** |"`
- E19: (0) `:63` — `"| Per-task success | identical | identical | **0** |"`
- E19: (0) `:64` — `"| Avg steps | identical | identical | **0** |"`

### Run parameters

- (5 episodes/task, 10 max steps) `:14-15` — `"- Episodes per task: 5"` / `"- Max steps per episode: 10"`
- (120 s timeout per task) `:17` — `` "- Command: `PYTHONPATH=src timeout 120 python3 scripts/phase8_closed_loop.py --task $task --docker-image $img --num-episodes 5 --max-steps 10`" ``
- (JEPA OFF for count-only) `:16` — `"- JEPA training: OFF (count-only)"`; `:42-43` — `"- JEPA training: ON (forward dynamics as side-effect, not exploration driver)"`

### Failure analysis per task category (`:36-38`)

- **Direct reads (100%)**: `"read_hello, find_secret, read_welcome, count_measurements, read_changelog_v4 — success cache enables 1-2 step solves after initial discovery"`
- **Deep path reads (20%)**: `"read_note, find_api_key, find_errors_v4 — 10-step ceiling exhausted before reaching target file in deep directory"`
- **Zero (0%)**: `"count_lines — wc -l never targets the correct filename"`

### Cross-doc quantitative claims

- (62.2%, zero delta) `PEDA_FINAL/PEDA_CONCLUSION.md:17` — `"Phase 8 confirmed: count-driven reaches 62.2% across 9 tasks; toggling JEPA on adds zero delta."`
- (62.2%) `PEDA_FINAL/PEDA_CONCLUSION.md:58` — `"Count-driven: **62.2% avg success rate** across 9 tasks. JEPA toggle adds **zero delta** | FAIL — JEPA contributes nothing beyond what count-based novelty already provides"`
- (62.2%) `PEDA_FINAL/PEDA_CONCLUSION.md:106` — `"The count-driven Phase 8 agent reached 62.2% success across 9 tasks on the v2 sandbox, confirming that simple counting is the correct tool for this problem class."`
- (17 experiments, 37x slower) `PEDA_FINAL/COUNT_DRIVEN_CHARTER.md` §1 — `"JEPA (never beat counting, 37x slower)"`; §3.1 — `"Sandbox v4 | 270 | 42% | 8% (JEPA hybrid) | 5.3x"`; §3.2 — `"This is equivalent to counting, but 37x slower."`
- (20/20 cache solves) `PEDA_FINAL/COUNT_DRIVEN_CHARTER.md` §1 — `"Success cache (memoization) | 1-step solver | 20/20 1-step completions across 4 tasks on repeated (state, action) pairs"`

---

## 3. Phase8Runner Architecture

Control flow (`src/phase8/count_driven_agent.py:1-6`): `"Perception → generate_sandbox_candidates() / Action Gen → NoveltyExplorer.select_action() / Action Exec → BusyboxSandbox.step() / Learning → ActionModelLearner + JEPAEnsemble (optional)"`.

Class docstring (`:76-88`): `"No prediction-error mechanism. Novelty = count-based bonus. STRIPS action schemas are learned from experience. JEPA forward dynamics training is optional (--train-jepa flag)."` And `:87` — `"Winning (cwd,action) pairs are memoized for fast reuse."`

Components (imported, not rewritten — `:16-24` `"Proven components — imported, not rewritten"`):

1. **BusyboxSandbox** (`src/phase2/sandbox_env.py`) — Docker container per episode: `--cap-drop=ALL --read-only --tmpfs /tmp --network none`; WHITELIST = {ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep, find}; 11 blocklist regexes (rm/mv/cp/chmod/chown/dd/mkfs/mount/sudo/su/docker/kill/shutdown/reboot). `step()` returns (next_state, reward=0, done); caches successful file reads into `file_cache`.
2. **NoveltyExplorer** (`src/phase5/explorer.py`) — exploration driver (see §4).
3. **ActionModelLearner** (`src/phase5/action_model.py`) — STRIPS schemas keyed `(verb, target_type, flag)`; VERB_WHITELIST; per-task TASK_KEYWORDS; `generate_candidates` caps at 16 (`"return deduped[:16]"`); `plan_to_target` builds multi-step `cd`+`cat`/`wc` plans.
4. **JEPAEnsemble** (`src/phase5/jepa_wm.py`) — optional; constructed only when `train_jepa AND model_path` (`count_driven_agent.py:111-114`). Frozen Qwen 0.5B encoder + 3 MLP predictors (seeds 42/43/44), Adam lr=1e-3, MSE loss, epistemic = ensemble variance.

Episode loop (`run_episode`, `:131-200`): reset container (start_cwd from `_task_start_cwd`) → `explorer.reset_episode()` → for t in range(max_steps): candidates = `generate_sandbox_candidates(state)` (fallback `["ls","pwd"]`) → action = `explorer.select_action(state, candidates, result.actions)` → `(next_state, reward, done) = sandbox.step(state, action)` → task `check_fn(state, action, next_state)` sets success → `explorer.observe(state, action, success)` → buffer.append((state, action, next_state, success)) → `action_model.learn_from_step(...)` → break on success/done. Post-episode: if `self.jepa is not None and len(self.buffer) >= 5`: `jepa.train_step(last 20 transitions)` (`:192-197`).

Task start-cwd (`_task_start_cwd`, `:35-48`): `deep_tasks = {"count_measurements": "/sandbox/data/raw", "find_errors_v4": "/sandbox/logs", "read_changelog_v4": "/sandbox/docs", "find_errors": "/sandbox/logs", "count_logs": "/sandbox/logs"}`; default `/sandbox`.

CLI (`scripts/phase8_closed_loop.py:22-31`): 9 task choices; `--docker-image`, `--num-episodes`, `--max-steps` (default 10), `--train-jepa` flag, `--model-path`. Summary JSON (`:61-70`): success, `success_rate = f"{success}/{total} ({success/total*100:.0f}%)"`, `avg_steps = round(..., 1)`; exit code 0 if any success else 1.

---

## 4. Success Cache Mechanism (`src/phase5/explorer.py`)

- Two count tables + cache (`:23-26`): `"self.state_counts = defaultdict(int)  # state_hash → visit count; self.state_action_counts = defaultdict(int)  # (state_hash, action) → try count; self.success_cache = {}  # state_hash → action that succeeded"`
- State key (`src/phase2/sandbox_env.py:75-82`): `"Keys on cwd + sorted file list only — no command history or output"` → hash = `f"{self.cwd}|{','.join(sorted(self.files))}"`
- Intrinsic reward (`explorer.py:20`): `"Intrinsic reward = 0.5 * (1/sqrt(1+state_novelty)) + 0.5 * (1/sqrt(1+pair_novelty))"` where `state_novelty = 1/sqrt(1+state_counts[sh])`, `pair_novelty = 1/sqrt(1+state_action_counts[(sh, action)])` (`:34-35`).
- Selection order (`:64-76`): (1) `"Cached success replay"` — `"if sh in self.success_cache: cached = self.success_cache[sh]; if cached in candidates: return cached"`; (2) `"Highest novelty bonus, tie-broken by action priority (file readers > analysis > navigation > passive)"` via `min(candidates, key=lambda a: (-self.novelty_bonus(state, a), self._action_priority(a)))`.
- Cache write (`observe`, `:80-84`): `state_counts[sh] += 1`; `state_action_counts[(sh, action)] += 1`; `"if success: self.success_cache[sh] = action"`.
- Action-type priority (`:38-46`): cat/head/tail=0, grep/find/wc=1, cd=2, ls/pwd/echo=3, unknown=4.
- Counts PERSIST across episodes (`reset_episode` is a no-op, `:86-88` `"Currently no episode-local state — counts persist across episodes so the explorer gets better over time"`). Persistence + cache explains the 100% 1–2-step direct-read solves "after initial discovery" (results `:36`).

---

## 5. Experimental Conditions

- **Hardware**: g4dn.xlarge (T4 16GB), instance i-06b0ba3dbdc214761, IP 13.220.38.201 (`results :6`)
- **Model**: Qwen2.5-0.5B-Instruct — `"CPU-only, not used for count-only run"` (`:7`); used only for JEPA embeddings when `--train-jepa`
- **Environments**: Docker images `peda-sandbox:v2` (tasks 1–4) and `peda-sandbox:v4` (tasks 5–9), rebuilt post-reboot (`:8`)
- **Code + commit**: `src/phase8/count_driven_agent.py`, `scripts/phase8_closed_loop.py` @ a348c1e (dev branch) (`:9-10`)
- **Episodes**: 5 per task × 9 tasks = 45 episodes per condition; max 10 steps; 120 s timeout per task (`:14-17`)
- **Task check functions** (`src/phase2/tasks.py`): goal predicates require exact action verb + target path + exit_code 0. Examples: `_goal_predicate_read_note` (file reader + `docs/note.txt` + exit 0); count tasks require `action.startswith("wc -l")` + target + exit 0; `_goal_predicate_find_errors_v4` (grep containing ERROR/error + exit 0); `_goal_predicate_find_api_key` (reader of `docs/api_reference.md` + output contains `"API_KEY"`); `_goal_predicate_read_changelog_v4` (reader of `docs/changelog.txt` + output contains `"v4"`).
- **JEPA training config**: frozen Qwen 0.5B encoder, 3 MLP predictors, Adam lr=1e-3, MSE loss, batch = last 20 transitions when buffer ≥ 5 (`count_driven_agent.py:192-197`)
- **Result artifacts**: `artifact://870` (count-only), `artifact://872` (count+JEPA) — live in the GPU-run session, NOT files on disk in this repo (`results :70`)

---

## 6. Verbatim Conclusions from Source Docs

- `results/phase8_gpu_run_2026-07-31.md:66` — `"**Conclusion**: JEPA forward dynamics training contributes zero additional value to the count-driven agent. Every task's success/failure pattern is identical with and without JEPA. This is consistent with 17 prior JEPA experiments where learned forward dynamics never improved exploration or task completion over count-based novelty."`
- `PEDA_FINAL/PEDA_CONCLUSION.md:17` (Q3) — `"PEDA never beats count-based novelty. The one statistically significant result (Phase 3, N=20, p=0.0043) is attributable to candidate set engineering and success caching, not epistemic prediction error. Phase 8 confirmed: count-driven reaches 62.2% across 9 tasks; toggling JEPA on adds zero delta."`
- `PEDA_FINAL/PEDA_CONCLUSION.md:58` — `"FAIL — JEPA contributes nothing beyond what count-based novelty already provides"`
- `PEDA_FINAL/PEDA_CONCLUSION.md:106` — `"The count-driven Phase 8 agent reached 62.2% success across 9 tasks on the v2 sandbox, confirming that simple counting is the correct tool for this problem class."`
- `PEDA_FINAL/PEDA_CONCLUSION.md` declaration (~:148-150) — task list: `"read_hello, count_lines, find_secret, read_note, read_welcome, find_api_key, count_measurements, find_errors_v4, read_changelog_v4"`
- `PEDA_FINAL/COUNT_DRIVEN_CHARTER.md` §8 — `"PEDA is concluded. This charter inherits: surviving code (src/phase5/action_model.py, explorer.py, candidate generator), experimental methodology (controlled baselines, ablation-first), archival discipline, and the 17 negative-result experiments."`

---

## 7. Contradictions / Gaps

1. **"v2 sandbox" mislabel**: `PEDA_CONCLUSION.md:106` says Phase 8 ran "on the v2 sandbox", but the canonical file shows 5 tasks on `peda-sandbox:v2` AND 4 on `peda-sandbox:v4` (`:23-31`). Paper should say "v2+v4 sandboxes".
2. **Aggregate avg-steps missing**: TOTAL row (`:32`/`:56`) has no avg-steps value; only per-task. "Avg steps identical" (`:64`) is asserted without aggregate numbers.
3. **Start-cwd confound**: the 100% "direct reads" category includes 3 tasks with engineered start cwds (count_measurements → `/sandbox/data/raw`, read_changelog_v4 → `/sandbox/docs`) while all 20% "deep path" tasks start at `/sandbox` root. count_measurements (100%) vs count_lines (0%) are BOTH `wc -l` tasks — the only difference is start cwd.
4. **JEPA condition is NOT a JEPA-driven-exploration ablation**: with `--train-jepa`, JEPA trains as a side-effect but `select_action` never consults `epistemic_uncertainty` (`count_driven_agent.py:147` vs `explorer.py:64-76`). E19's "zero delta" therefore falsifies training side-effects only — frame as "JEPA trained as a side-effect changed nothing".
5. **Marginal successes at the ceiling**: find_api_key/find_errors_v4 at 20% with avg 10.0 steps implies the single success occurred exactly AT the step ceiling (all 5 episodes at 10 steps); read_note 8.8 avg with 1/5 implies one success at ~4 steps: (10×4+4)/5 = 8.8.
6. **Raw transcripts not in repo**: artifact://870/872 live in the GPU-run session (`results :70`), not readable from this filesystem — per-episode action traces cannot be locally re-verified.
7. **No per-task timing data** for the count+JEPA condition (Qwen 0.5B CPU load per task), so the "37x slower" claim cannot be quantified for Phase 8 specifically.
8. **No shaped reward**: env reward is always 0; all learning signal comes from the binary goal check + counts. Success uses exact goal predicates (`tasks.py`).
