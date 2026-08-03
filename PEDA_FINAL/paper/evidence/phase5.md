# Phase 5 Evidence Report — JEPA + Action Model (scout S7Phase5)

**experiment_ids: [E13, E14]**

Scope: Phase 5 JEPA exploration (E13, 11 sub-configs E13.01–E13.11), pure-epistemic explorer (E14), plus Phase 5 supporting evidence feeding the paper's "What Survived" section (STRIPS action learning, v2→v3 migration, 5 bugs, JEPA MLP architecture, ActionModelLearner algorithm).

**Verdict anchor (ground truth):** `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:9`
> "Across 11 experiments spanning 4 sandboxes (v2/v3/v4 grid maze deterministic/stochastic), JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**."

**Decision anchor:** `PEDA_FINAL/archive/phase5_jepa_exploration/README.md:3-4`
> "**Date:** 2026-07-30  **Decision:** ARCHIVED — negative result.  **Core hypothesis:** Learned forward dynamics (JEPA) → better exploration decisions"

---

## 1. E13 — Phase 5 JEPA hybrid (11 sub-configs)

### 1.1 Sub-config anchor table (E13.01–E13.11)

Sub-IDs assigned to every survivable anchor; the 11-config composition is NOT fully recoverable from surviving artifacts (raw `results/phase5_jepa_*.jsonl` outputs are absent from the repo — see Gaps §7.4).

| Sub-ID | Config | States | Count-based | JEPA best | Source |
|---|---|---|---|---|---|
| E13.01 | Sandbox v2 | 65 | 50% | 50% (hybrid) | README.md:13 |
| E13.02 | Sandbox v4 | 270 | 42% | 8% (hybrid) | README.md:14 |
| E13.03 | Maze 10x10 deterministic | 1,100 | 100% | 0% | README.md:15 |
| E13.04 | Maze 10x10 stochastic | 1,100 | 100% | 67% (hybrid); pure JEPA 0% | README.md:17 |
| E13.05 | Maze 20x20 | 8,400 | 0% | 0% | README.md:16 |
| E13.06 | P4 EFE (4 rounds) | 65-270 | 50% | 25% | README.md:18 |
| E13.07 | read_hello headline pair | 65 | novelty-only **50%** | **jepa_efe 17%** | PEDA_CONCLUSION.md:53 |
| E13.08–E13.11 | remaining sub-configs | — | — | — | NOT RECOVERABLE (raw JSONL deleted; see §7.4) |

Second anchor for the same table: `PEDA_FINAL/COUNT_DRIVEN_CHARTER.md:31-55` (§3.1 "Count-based exploration works well at small scale"), which adds the count-vs-learned ratio column:
- E13.01: "| Sandbox v2 | 65 | 50% | 50% (hybrid tied) | 1.0x |" (CHARTER.md:36)
- E13.02: "| Sandbox v4 | 270 | 42% | 8% (JEPA hybrid) | 5.3x |" (CHARTER.md:37)
- E13.03: "| Maze 10x10 (deterministic) | 1,100 | 100% | 0% (any JEPA variant) | — |" (CHARTER.md:38)
- E13.04: "| Maze 10x10 (stochastic) | 1,100 | 100% | 67% (JEPA hybrid) | 1.5x |" (CHARTER.md:39)
- E13.05: "| Maze 20x20 | 8,400 | 0% | 0% | — |" (CHARTER.md:40)
- Source line: "Source: Phase 5 JEPA exploration, 11 experiments across 4 sandboxes." (CHARTER.md:55)

### 1.2 E13 headline quantitative claims

- **E13: novelty-only 50% > jepa_efe 17% (read_hello).** `PEDA_FINAL/PEDA_CONCLUSION.md:53`
  > "| 5 | Sandbox v2/v3/v4 | JEPA forward dynamics + hybrid, 11 exps | Q1, Q2, Q3 | Novelty-only 50% > jepa_efe 17% on read_hello. JEPA loss converges, no exploration gain | FAIL — JEPA uncertainty flat across all unexplored states |"
- **E13.07 condition definitions (jepa_efe):** `scripts/phase4_jepa_experiment.py:6`
  > "jepa_efe: JEPA epistemic + pragmatic (EFE balanced, alpha=0.5)"
  and `scripts/phase4_jepa_experiment.py:63-66`: `"jepa_efe": {"alpha": 0.5, "description": "JEPA epistemic + pragmatic (EFE balanced)"}`.
  EFE formula: `src/phase4/jepa_peda.py:11-16` docstring:
  > "efe = epistemic * alpha + pragmatic * (1 - alpha) / pick action with max EFE"
- **E13.02 pure-hybrid comparison (v4, 270 states):** count 42% vs JEPA hybrid 8% (README.md:14 / CHARTER.md:37, ratio 5.3x).
- **E13.06 P4 EFE 4 rounds:** "| P4 EFE (4 rounds) | 65-270 | 50% | 25% | — |" (README.md:18).

### 1.3 E13 methodology (harness)

- `scripts/phase5_jepa_experiment.py` (Phase 5 hybrid harness): 3 modes — `pure_novelty`, `jepa_only`, `hybrid` where hybrid = `0.5 * novelty + 0.5 * epistemic` (script docstring "Compares 3 exploration modes"; `JEPAExplorer.__init__` novelty_weight=0.5, epistemic_weight=0.5; score(): "hybrid: nov = self.novelty_bonus(...); epi = ...; return self.novelty_weight * nov + self.epistemic_weight * epi").
- Experiment conditions: tasks from `MICRO_TASKS` (read_hello, count_lines, find_secret, read_note), default `--num-episodes 6`, `--max-steps 10`, `--mode all`; CWDs = KNOWN_CWDS (5) + UNKNOWN_CWDS (8) = 13 start dirs (script: "KNOWN_CWDS = [...] UNKNOWN_CWDS = [...]"); model default `~/models/Qwen2.5-0.5B-Instruct`; JEPA ensemble `n_ensemble=3`, device cuda-or-cpu; post-episode batch training of the ensemble on collected transitions for jepa_only/hybrid modes.
- Metrics: FHT (first-hit step), SCR (unique visited states / steps), dead_loop_rate (3 consecutive identical actions), per-episode `train_loss` recorded in output JSONL (script `compute_metrics`).
- `scripts/phase4_jepa_experiment.py` (P4 EFE, E13.06): 4 conditions jepa_efe / jepa_only / pragmatic_only / novelty_only (lines 6-9, 249-272); docker image `peda-sandbox:v4` (line 45); 4 tasks x 4 conditions x 12 episodes recommended (lines 21-26).
- **What Failed list:** `README.md:31-34` — "EFE with binary pragmatic: dominated by 0/1 term, α irrelevant"; "Hybrid (novelty + epistemic): never beat pure novelty"; "Pure epistemic (jepa_only): SCR ~0, no room exploration"; "Scaling: 8400 states still too small for epistemic advantage".
- **Root cause (verbatim):** `README.md:38` — "JEPA predicts "(state, action) → next embedding". Its uncertainty is "how uncertain am I about this transition?" All unexplored transitions are equally uncertain — equivalent to counting, but 37× slower." And `README.md:40`: "For JEPA to beat counting, it needs to say "THIS unexplored direction is more promising than THAT one." This requires goal-conditioning or learned value in the embedding space — beyond what was tested."

### 1.4 JEPA loss curve (start → end)

- **45 → 15.** `README.md:27`: "| JEPA MLP predictor | UNCERTAIN | Loss always converges (45→15), MLP learns dynamics |"
- `PEDA_CONCLUSION.md:90`: "JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states... computed at approximately 37x the computational cost (MLP forward pass + embedding computation vs integer increment)."
- `PEDA_CONCLUSION.md:124`: "The JEPA MLP predictor learns to predict next-state embeddings from (state, action) pairs, as shown by decreasing loss curves across all experiments (loss 45 to 15)."
- GAP: no per-episode loss curve data survives (would have been in the deleted phase5_jepa_*.jsonl `train_loss` fields).

### 1.5 Q1/Q3 cross-phase verdicts citing Phase 5

- `PEDA_CONCLUSION.md:15` (Q1): "LLM World Models produce epistemic error ~0 on small state spaces (<100 states) and uniform uncertainty on larger ones (JEPA ensemble, all DLR ~0.996). The model is too certain or uniformly uncertain — never differentially uncertain."
- `PEDA_CONCLUSION.md:17` (Q3): "PEDA never beats count-based novelty." + Phase 8 "toggling JEPA on adds zero delta."
- `PEDA_CONCLUSION.md:145`: "under practical conditions (small state spaces <1000 states, 0.5B models, LoRA fine-tuning, CPU-limited inference), count-based novelty is the reliable exploration mechanism."

---

## 2. E14 — Phase 5 JEPA pure epistemic (jepa_only)

- `PEDA_CONCLUSION.md:54`
  > "| 5 | Sandbox v4 | Pure epistemic (jepa_only) explorer | Q2 | SCR ~0 across all tasks, zero room exploration | FAIL — epistemic signal too weak to drive useful behavior |"
- `README.md:33`: "- **Pure epistemic (jepa_only)**: SCR ~0, no room exploration"
- Implementation: `scripts/phase5_jepa_experiment.py` score() for jepa_only returns only `self.jepa.epistemic_uncertainty(state_text, action)`; the ensemble variance signal (jepa_wm.py:149-158) — "Mean squared deviation across ensemble and all dims", range "~[0, 2+]".
- Supporting: `README.md:27` "Loss always converges (45→15), MLP learns dynamics" — the model learns transitions but the derived signal is behaviorally inert.

---

## 3. Phase 5 supporting evidence — STRIPS action learning (45.8% vs 31.3%)

NOTE: this feeds the paper's "What Survived" section, not the E13/E14 verdicts. The claim is specifically a **candidate hit rate on the action prediction task** (does the generated candidate set contain the correct action), NOT episode success.

- `COUNT_DRIVEN_CHARTER.md:17`: "| **STRIPS action schemas** | Learned action model | 45.8% hit rate vs 31.3% fallback on v2 sandbox traces |"
- `COUNT_DRIVEN_CHARTER.md:35`: "**Composition question**: Can STRIPS schemas learned from single-step traces chain into multi-step plans? (Current baseline: 45.8% learned hits vs 31.3% fallback — improvement exists, gap remains.)"
- `COUNT_DRIVEN_CHARTER.md:67-71`:
  > "| Hit rate | 45.8% | 31.3% | +14.5pp |" ... "Learned schemas are cwd-independent path predicates (post-bugfix). The 45.8% ceiling may be a data coverage issue, not a fundamental ceiling."
- `PEDA_CONCLUSION.md:108`: "**STRIPS action learning.** Learned action schemas (preconditions + effects) from execution traces reach 45.8% learned vs 31.3% fallback on the action prediction task. The `ActionModelLearner` correctly extracts cwd-change preconditions and filesystem effects from observed transitions."
- `README.md:24`: "| STRIPS action learning | KEEP | 45.8% learned vs 31.3% fallback |"
- Raw per-experiment data for this figure does NOT survive in the repo (see §7.3).

---

## 4. Phase 5 supporting evidence — v2→v3 migration results

- `README.md:26`: "| Data-driven candidate generation | KEEP | v2→v3 migration 0 crashes |"
- `COUNT_DRIVEN_CHARTER.md:18`: "| **Data-driven candidate generation** | Infrastructure | v2→v3→v4 sandbox migrations: 0 crashes, zero regressions |"
- `PEDA_CONCLUSION.md:110`: "The sandbox candidate generator evolved from hardcoded heuristics (v1, 4 candidates) to data-driven enumeration (v2, 65 pairs; v3/v4, 270+ pairs) with zero crashes during migration."
- Work log data-collection context: `PEDA_WORKING_LOG.md:1805-1806`: "数据采集：GPU 上跑 8 轮 random + heuristic × 4 tasks，收集 **1378 条新转移**（`results/phase5_train_data/`）" and "合并：1378 新 + 旧 65 = **114 episodes** → `results/phase5_merged_train.jsonl`". (8 files present in `results/phase5_train_data/`: golden_paths + random/heuristic × 4 tasks.)

### 4.1 v3 per-task episode results (computed from surviving JSONL — Phase 5 no-WM harness, v3 sandbox tasks)

Harness: `scripts/phase5_no_wm_experiment.py` — NoveltyExplorer + ActionModelLearner, `--candidates learned|fallback`, 12 episodes, 10 max steps, CWD rotation over /sandbox, /sandbox/data, /sandbox/docs, /sandbox/logs, /sandbox/projects, /sandbox/tmp.

Learned mode (ActionModelLearner.generate_candidates), 48 episodes:
- read_greeting: 0/12 (`results/v3_learned_read_greeting.jsonl`, all lines: success:false)
- find_secret_note: 0/12 (`results/v3_learned_find_secret_note.jsonl`)
- count_entries: 0/12 (`results/v3_learned_count_entries.jsonl`)
- read_user_guide: 0/12 (`results/v3_learned_read_user_guide.jsonl`)
- **Aggregate: 0/48 (0%)**

Fallback mode (generate_sandbox_candidates), 48 episodes:
- read_greeting: **12/12** (`results/v3_fallback_read_greeting.jsonl`; e.g. episode 0: `"success": true, "fht": 1`; episodes 7-11 FHT=9)
- find_secret_note: **1/12** (`results/v3_fallback_find_secret_note.jsonl` episode 4, /sandbox/projects: `"success": true, "fht": 8`)
- count_entries: 0/12 (`results/v3_fallback_count_entries.jsonl`)
- read_user_guide: **1/12** (`results/v3_fallback_read_user_guide.jsonl` episode 4, /sandbox/projects: `"success": true, "fht": 9`)
- **Aggregate: 14/48 (29.2%)**

CAUTION (contradiction): these v3 episode files show learned-mode episode success 0% vs fallback 29%, opposite to the 45.8% > 31.3% claim. The two are different quantities (candidate hit rate on v2 traces vs episode success on v3) and the learned model starts from zero each run (online learning within the run; first episodes have no schemas). The paper MUST cite 45.8/31.3 exactly as "action prediction task / candidate hit rate" (CHARTER.md:17,68; CONCLUSION.md:108) and must NOT use these v3 files to claim learned > fallback.

---

## 5. Phase 5 supporting evidence — the 5 bugs (before/after)

Source list: `README.md:55-59` ("## Bugs Found (5 fixed)").

1. **Path predicate cwd-unaware → `_action_hits_target`** (`README.md:57`)
   - AFTER (current): `src/phase2/tasks.py:4-17` — `_action_hits_target` handles cwd-relative paths:
     > "Both `cat docs/note.txt` (absolute) and `cat note.txt` from /sandbox/docs (relative) correctly hit "docs/note.txt"." — logic: `target_rel_path in action` OR (`filename in action AND cwd == expected_dir`).
   - BEFORE: simple substring match with no cwd check → `cat note.txt` from /sandbox could be counted as hitting `docs/note.txt` (false positives in task success predicates).
2. **Novelty tie-breaking alphabetical → action priority ordering** (`README.md:58`)
   - AFTER: `src/phase5/explorer.py:38-49` `_ACTION_PRIORITY = {"cat":0,"head":0,"tail":0,"grep":1,"find":1,"wc":1,"cd":2,"ls":3,"pwd":3,"echo":3}` with comment "Lower = prefer: file readers > content analysis > navigation > passive"; used as tie-break in `select_action` (`explorer.py:74-75`: `min(candidates, key=lambda a: (-self.novelty_bonus(state, a), self._action_priority(a)))`). Same priority table + `max(candidates, key=lambda a: (score(a), -self._action_priority(a)))` in `scripts/phase5_jepa_experiment.py` JEPAExplorer.
   - BEFORE: ties resolved alphabetically by action string → arbitrary command ordering dominated selection.
3. **`final_state.victory` always False → return next_state** (`README.md:59`)
   - AFTER: `scripts/phase5_jepa_experiment.py` run_jepa_episode: `if success: next_state.victory = True; final_state = next_state; break` ... `final_state = final_state if "final_state" in dir() else state`. Identical pattern in `scripts/phase4_jepa_experiment.py:163-173` and `scripts/phase5_no_wm_experiment.py`.
   - BEFORE: `final_state` bound to the pre-action `state` while the victory flag was set on `next_state`, so success was never observed → 0% success reported even on completions.
4. **DLR 0.53-0.80 → repeat action penalty** (`README.md:59`, listed as item 4)
   - AFTER: `src/phase4/jepa_peda.py` select_action: dead-loop penalty using stable filesystem signature `state_sig = (getattr(state, 'cwd', ''), frozenset(getattr(state, 'files', [])))`; `repeat_count = self._recent_history.count((state_sig, action)); if repeat_count >= 2: score *= 0.1`; `_recent_history` capped at 5.
   - BEFORE: repeated same (state, action) pairs at DLR 0.53–0.80 (README bug list; corroborated in work log `PEDA_WORKING_LOG.md:1761`: "Pragmatic 0.48-0.80（频繁 ls ↔ ls data 振荡）").
5. **JEPA predictors cross-task contamination → `reset_predictors()`** (README.md:59, item 5)
   - AFTER: `src/phase5/jepa_wm.py:100-108` — `reset_predictors()` reinitializes all MLP weights (`xavier_uniform_` + zero bias) and fresh Adam optimizers (`lr=1e-3`); invoked by `src/phase4/jepa_peda.py` `reset()`: "Reinitializes MLP weights (clears cross-task contamination) and clears dead-loop history and novelty counts."
   - BEFORE: predictors carried trained weights across tasks, so per-task novelty scores were contaminated by prior task dynamics.

---

## 6. Implementation details (for the paper's methods)

### 6.1 JEPA MLP architecture (`src/phase5/jepa_wm.py`)

- `jepa_wm.py:4`: "Uses frozen Qwen 0.5B encoder + trainable MLP ensemble predictors."
- `jepa_wm.py:7`: "(state_text, action) → [frozen Qwen 0.5B] → state_emb (768-dim)" — hidden_size = Qwen config hidden_size (768 for Qwen2.5-0.5B).
- `jepa_wm.py:37-46` `MLPPredictor(nn.Module)`: `nn.Sequential(nn.Linear(hidden_size * 2, 256), nn.ReLU(), nn.Linear(256, hidden_size))` — concat(state_emb, action_emb) → 1 hidden layer (256) → predicted next embedding. (Paper-level: 1 hidden layer in shipped code; PEDA_CONCLUSION.md:138 says "JEPA MLP predictors (1-3 hidden layers)" across experiments.)
- `jepa_wm.py:53-60` `JEPAEnsemble`: n_ensemble=3, seeds `torch.manual_seed(42 + i)` (line 83), per-member Adam `lr=1e-3` (line 89).
- Encoding: `encode_state` = mean-pooled last hidden state (max_length=256, cached per state_text); `encode_action` = mean-pooled embedding layer (max_length=32).
- Epistemic signal: `jepa_wm.py:149-158` `epistemic_uncertainty` = mean squared deviation of ensemble predictions from the ensemble mean across all dims; "Range ~[0, 2+] depending on embedding scale."
- Training: `jepa_wm.py:166` `train_step(transitions)` — per-member MSE against frozen-encoded next-state embedding, per-member optimizer step, returns ensemble-average loss; batch training per episode (intermittent, not per-step SGD).
- Declaration-level conditions: `PEDA_CONCLUSION.md:138-140`: "Model: Qwen2.5-0.5B-Instruct with LoRA (rank=16), JEPA MLP predictors (1-3 hidden layers), zero-shot RSSM"; "Training: 65-1378 transitions, 1-3 epochs LoRA, 500-2000 steps JEPA, CPU (Intel Core Ultra 9) or GPU (NVIDIA T4 16GB)".

### 6.2 ActionModelLearner — STRIPS extraction algorithm (`src/phase5/action_model.py`)

- `action_model.py:17-47` `ActionSchema` dataclass: verb, target_type, flag, preconditions/effects as `(predicate, value)` tuples, success_count, attempt_count; `success_rate` property; `as_action_pattern()` renders e.g. `cat {txt} > exit=0`.
- `action_model.py:49-78` `ActionModelLearner`: schemas keyed `(verb, target_type, flag)`; `dir_contents` dir→files map; `VERB_WHITELIST = {ls, pwd, cd, cat, echo, mkdir, touch, wc, head, tail, grep, find}`; `TASK_KEYWORDS` per task (read_hello→{hello.txt}, count_lines→{lines.txt}, read_note→{note.txt}, find_secret→{secret}, etc.).
- Generalization mechanism (docstring lines 6-9): target typing — "cat hello.txt" generalizes to "cat .txt files in current directory".
- `_classify_target` (action_model.py:79-104): returns "parent" (for ..), "dir", file extension ("txt","csv","py","log","ini","md"), or "any".
- `_infer_preconditions` (107-137): cat/head/tail/wc X → `("file_in_cwd", target)`; cd D → `("dir_in_cwd", target)`; cd .. → `("is_not_root", "")`; grep → `("target_exists", target)`.
- `_infer_effects` (140-164): always `("exit_code", str(ec))`; cd → `("cwd_changed_to", new_cwd | "same")`; updates dir_contents.
- `learn_from_step` (167-213): parse verb/target/flag → classify → create schema (with inferred preconditions/effects) or refine effects → increment attempt_count (+ success_count on success) → update directory map.
- `generate_candidates` (216-312): always `ls`, `pwd`, `cd ..` (in subdirs); schemas with `success_rate >= 0.5` (or attempt_count < 3, line 245) instantiated against current files matching target_type; task-specific keywords; navigation via `_find_file_location` (314+) + `_rel_dir`; appends `grep -r secret .`, `find . -name '*.txt'`; dedup, cap `deduped[:16]` (line 312).
- `plan_to_target` (352-409): returns multi-step plans like `["cd data", "wc -l lines.txt"]` using the dir_contents map (direct action if target accessible; else navigation steps + final read/count).
- `target_type_matches` (end of file): "txt"↔"txt", anything matches "any", "dir"↔"dir".

### 6.3 Count-based explorer (the winning baseline, `src/phase5/explorer.py`)

- `explorer.py:28-36` `novelty_bonus`: `state_novelty = 1.0 / sqrt(1 + state_counts[sh])`, `pair_novelty = 1.0 / sqrt(1 + state_action_counts[(sh, action)])`, bonus = `0.5 * state_novelty + 0.5 * pair_novelty`.
- `explorer.py:51-76` `select_action`: 1) cached success replay; 2) highest novelty, tie-broken by action priority; empty candidates → "ls".
- `explorer.py:78-84` `observe`: increments both count tables; caches success per state hash.
- Charter verdict `CHARTER.md:16`: "Count-based pair novelty | Core exploration driver | Beat all learned signals in 17 experiments at <1000 states; handles stochastic items (Phase 6)".

### 6.4 Other Phase 5 scripts

- `scripts/phase5_combinator.py`: DynamicCandidateGenerator (verb × target_type success stats, exploration bonus 0.05/attempts, cap 12 candidates) + MinimalExplorer (unseen-pair preference, random fallback); 100% tabular.
- `scripts/phase5_minimal_explorer.py`: MinimalExplorer (seen triples, success cache), 5-12 episodes.
- `scripts/phase5_rlvr_train.py`: Error-Weighted Fine-Tuning (weight = 1 − accuracy; repeat weight×5; Qwen2.5-1.5B default) — Phase 5 WM-training adjunct, not part of the JEPA experiments.

---

## 7. Contradictions and gaps

1. **"11 sub-configs" not individually enumerable.** Only aggregate anchors survive (README.md:9,13-18; CONCLUSION.md:53; CHARTER.md:55). 7 anchors map to E13.01–E13.07; E13.08–E13.11 (4 sub-configs) are unrecoverable — raw `results/phase5_jepa_*.jsonl` do not exist in the repo (glob `**/*phase5_jepa*` returns nothing outside the archive README).
2. **v2 per-task count+STRIPS results missing.** `scripts/phase5_no_wm_experiment.py` writes `results/phase5_no_wm_{task}.jsonl` / `phase5_fallback_{task}.jsonl` (v2 tasks read_hello, find_secret, count_lines, read_note), but none of those files survive; only the v3-sandbox task files (read_greeting, find_secret_note, count_entries, read_user_guide) remain.
3. **45.8% vs 31.3% has no surviving raw trace** — only the derived claim in CHARTER.md:17,35,67-68, CONCLUSION.md:108, README.md:24. Must be cited as "candidate hit rate on the action prediction task."
4. **v3 episode files contradict a naive reading:** learned 0/48 vs fallback 14/48 (29.2%) episode success (§4.1). Different metric + online-from-zero learning; do not use to support learned > fallback.
5. **"hybrid tied 50%" vs "jepa_efe 17%":** README.md:13 (v2 hybrid 50% = tied with count) vs CONCLUSION.md:53 (novelty 50% > jepa_efe 17% on read_hello). Consistent only if "hybrid" (novelty+epistemic, phase5_jepa_experiment) is distinguished from "jepa_efe" (EFE-balanced, phase4_jepa_experiment α=0.5). Also CHARTER.md:37 v4 hybrid = 8% (not 17%) — different condition/round.
6. **Loss curve endpoints only (45→15):** no per-episode train_loss series survives (would be in deleted JSONL).
7. **"1-3 hidden layers" (CONCLUSION.md:138) vs shipped MLPPredictor with exactly 1 hidden layer (256)** (jepa_wm.py:42-46) — the range covers variants across experiments not present in the repo.
8. **JEPA "37x slower"** is a stated estimate (CONCLUSION.md:90 "approximately 37x"; README.md:38 "37× slower") with no benchmark artifact in the repo.
9. Work log (`PEDA_WORKING_LOG.md`) contains NO section detailing the 11 Phase 5 JEPA experiments or the 5-bug fixes; the only Phase 5 log entries cover data collection (lines 1795-1820) and 45→15/50-vs-17 claims live solely in the canonical docs. Bug before/after detail above is reconstructed from current code + README bug list.
