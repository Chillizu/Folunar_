# Phase 6 Evidence — Grid Maze (S8Phase6)

**Scout:** S8Phase6
**Date:** 2026-07-31
**experiment_ids:** [E15, E16]
**Canonical result summary:**
- E15: Grid Maze 10x10 (1100 states) — count 100%, JEPA 0%, hybrid 67%
- E16: Grid Maze 20x20 (8400 states) — count 0%, JEPA 0%
- Stochastic 10x10 (archive README, tagged E15 family): count 100%, JEPA best 67%, pure JEPA 0%

---

## 1. Files Read

| File | Role |
|---|---|
| `results/phase6_maze_count_5x5_seed42.jsonl` | Raw count results, 5x5 (6 episodes, all success) |
| `results/phase6_maze_count_10x10_seed42.jsonl` | Raw count results, 10x10 — **0 bytes (EMPTY)** |
| `results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl` | Raw jepa_only, 5x5 (1 line, failure) |
| `results/phase6_maze_jepa_all_modes_5x5_seed42.jsonl` | Raw all-modes 5x5 (6 lines, pure_novelty only; 2 fail + 4 success) |
| `results/phase6_maze_jepa_pure_novelty_5x5_seed42.jsonl` | Raw pure_novelty, 5x5 (6 lines, all success) |
| `src/phase6/maze_generator.py` | DFS recursive-backtracker maze generation |
| `src/phase6/grid_env.py` | GridMazeEnv, GridState, text observations |
| `src/phase6/stochastic_maze.py` | StochasticMazeEnv (item respawn, emerald task) |
| `src/phase6/__init__.py` | Package docstring only |
| `scripts/phase6_maze_count.py` | Count-based experiment runner |
| `scripts/phase6_maze_jepa.py` | JEPA experiment runner (3 modes) |
| `scripts/phase6_stochastic_count.py` | Stochastic count runner |
| `scripts/phase6_stochastic_jepa.py` | Stochastic JEPA runner |
| `src/phase5/jepa_wm.py` | JEPAEnsemble (methodology dependency) |
| `PEDA_FINAL/PEDA_CONCLUSION.md` | Canonical E15/E16 rows (ground truth) |
| `PEDA_FINAL/archive/phase5_jepa_exploration/README.md` | Phase 6 summary table (per-size numbers) |
| `PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md` | Phase 6 only "Planned" — superseded, not evidence |
| `PEDA_WORKING_LOG.md` | No phase6 entries found |

---

## 2. Evidence Bundles (number, source_file:line, verbatim)

### E15 — Grid Maze 10x10 (count 100%, JEPA 0%, hybrid 67%)

- (100%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:55`, `| 6 | Grid Maze 10x10 | Count vs JEPA vs hybrid (1100 states) | Q3 | Count: 100% goal-reaching. JEPA: 0%. Hybrid: 67% (count-driven carries JEPA) | FAIL — at 1100 states, count is already optimal |`)
- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:55`, `JEPA: 0%` — same row)
- (67%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:55`, `Hybrid: 67% (count-driven carries JEPA)` — same row)
- (1100, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:55`, `Grid Maze 10x10 | Count vs JEPA vs hybrid (1100 states)`)
- (100%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:15`, `| Maze 10x10 | 1,100 | 100% | 0% | — |`)
- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:15`, `| Maze 10x10 | 1,100 | 100% | 0% | — |` — JEPA best column)

### E16 — Grid Maze 20x20 (both 0%)

- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:56`, `| 6 | Grid Maze 20x20 | Count vs JEPA (8400 states) | Q3 | Count: 0%. JEPA: 0%. Both agents hit state-space ceiling | FAIL — neither approach scales to 8400+ states |`)
- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:56`, `JEPA: 0%` — same row)
- (8400, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:56`, `Grid Maze 20x20 | Count vs JEPA (8400 states)`)
- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:16`, `| Maze 20x20 | 8,400 | 0% | 0% | — |`)
- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:16`, `JEPA best` column, same row)
- (8400, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:34`, `**Scaling**: 8400 states still too small for epistemic advantage`)

### E15-family — Stochastic Maze 10x10 (count vs JEPA)

- (100%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:17`, `| Stochastic 10x10 | 1,100 | 100% | 67% | 0% |` — count column)
- (67%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:17`, `| Stochastic 10x10 | 1,100 | 100% | 67% | 0% |` — JEPA best (hybrid) column)
- (0%, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:17`, `| Stochastic 10x10 | 1,100 | 100% | 67% | 0% |` — pure JEPA column)
- (0.1, `/home/chillizu/Projects/Folunar_/src/phase6/stochastic_maze.py:54`, `respawn_p: float = 0.1,`)
- (0.02, `/home/chillizu/Projects/Folunar_/src/phase6/stochastic_maze.py:55`, `rare_p: float = 0.02,`)
- (0.3, `/home/chillizu/Projects/Folunar_/src/phase6/stochastic_maze.py:73`, `if items_at and random.random() < 0.3:` — 30% per-room item removal)
- (0.02, `/home/chillizu/Projects/Folunar_/scripts/phase6_stochastic_count.py:9`, `Task: "find emerald" — spawns with p=0.02 in rooms where (x+y)%3==0.`)
- Expectation (count): `/home/chillizu/Projects/Folunar_/scripts/phase6_stochastic_count.py:6`, `Expectation: 0% success — novelty expires after one visit per room, so the agent never returns to check for newly spawned items.`
- Expectation (JEPA): `/home/chillizu/Projects/Folunar_/scripts/phase6_stochastic_jepa.py:6`, `Expectation: >0% success — epistemic uncertainty about item presence drives revisitation of rooms to check for newly spawned items.`
- **Surprise vs expectation:** count reached 100% on the stochastic maze (expected 0%); JEPA hybrid 67% < count. Docstring expectation in `stochastic_maze.py:41-43`: `Count-based explorers should fail because novelty expires after one visit per room, so they never return to check for newly spawned items.` — contradicted by the 100% count result.

### Raw 5x5 data (pre-canonical; supports E15/E16 scaling story)

- (0.405, `/home/chillizu/Projects/Folunar_/results/phase6_maze_count_5x5_seed42.jsonl:1`, `{"width": 5, "height": 5, "seed": 42, "episode": 0, "steps_count": 42, "success": true, "fht": 41, "scr": 0.405, "dead_loop_rate": 0.0, "goal_x": 4, "goal_y": 4, ...}`)
- (0.0, `/home/chillizu/Projects/Folunar_/results/phase6_maze_count_5x5_seed42.jsonl:1`, `"dead_loop_rate": 0.0` — DLR 0 on all 6 episodes)
- (6/6, `/home/chillizu/Projects/Folunar_/results/phase6_maze_count_5x5_seed42.jsonl:1-6`, all 6 episodes `"success": true`, identical `steps_count: 42, fht: 41, scr: 0.405` — deterministic explorer + fixed maze seed)
- (0.05, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl:1`, `{"width": 5, "height": 5, "mode": "jepa_only", "seed": 42, "episode": 0, "steps_count": 20, "success": false, "fht": null, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 49.932384, "elapsed": 43.1}`)
- (0.9, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl:1`, `"dead_loop_rate": 0.9` — jepa_only dead-loops 90% of steps on 5x5)
- (49.932384, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl:1`, `"train_loss": 49.932384` — first JEPA train_step loss)
- (1.0, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_pure_novelty_5x5_seed42.jsonl:2`, `{"width": 5, "height": 5, "mode": "pure_novelty", "seed": 43, "episode": 1, "steps_count": 16, "success": true, "fht": 15, "scr": 1.0, "dead_loop_rate": 0.125, ...}`)
- (0.046-0.125, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_pure_novelty_5x5_seed42.jsonl:1-6`, DLR range across 6 successful pure_novelty episodes; SCR range 0.25-1.0; FHT 15-67)
- (0.04, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_all_modes_5x5_seed42.jsonl:2`, `"mode": "pure_novelty", "seed": 43, "episode": 1, "steps_count": 100, "success": false, "fht": null, "scr": 0.04, "dead_loop_rate": 0.0` — failed run in all_modes file, hit 100-step ceiling)
- (0.05, `/home/chillizu/Projects/Folunar_/results/phase6_maze_jepa_all_modes_5x5_seed42.jsonl:1`, `"steps_count": 100, "success": false, ... "scr": 0.05, "dead_loop_rate": 0.01` — second failed run in all_modes file)

### JEPA flatness / 37x cost (root-cause context for E15/E16)

- (37, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:90`, `JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states. The learned signal is "how uncertain am I about this (state, action) transition?" — which is equally high for every transition the agent has never seen. This is identical to count-based novelty (unvisited = uncertain) but computed at approximately 37x the computational cost (MLP forward pass + embedding computation vs integer increment).`)
- (45→15, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:27`, `JEPA MLP predictor | UNCERTAIN | Loss always converges (45→15), MLP learns dynamics`)
- (0.996, `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:57`, `All JEPA tracks: DLR ~0.996 (near-perfect determinism, zero epistemic signal)` — E17 context, NOT phase 6; do not attribute to phase 6)

### Verbatim conclusions

- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:55`: `FAIL — at 1100 states, count is already optimal`
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/PEDA_CONCLUSION.md:56`: `FAIL — neither approach scales to 8400+ states`
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:9`: `Across 11 experiments spanning 4 sandboxes (v2/v3/v4 grid maze deterministic/stochastic), JEPA-learned epistemic signal **did not improve exploration over count-based novelty in any regime**.`
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:33`: `**Pure epistemic (jepa_only)**: SCR ~0, no room exploration`
- `/home/chillizu/Projects/Folunar_/PEDA_FINAL/archive/phase5_jepa_exploration/README.md:25`: `Count-based novelty (pair) | KEEP | Optimal at <1000 states, handles stochastic items`

---

## 3. Experimental Conditions

- **Environments:** `GridMazeEnv` (deterministic text-grid maze) and `StochasticMazeEnv` (item respawn). Maze sizes planned: 5x5, 10x10, 20x20, 30x30 (`scripts/phase6_maze_count.py:15`, `for s in 5 10 20 30`).
- **State-space sizes (as cited in docs):** 10x10 = 1,100 states (E15); 20x20 = 8,400 states (E16). grid_env.py:8 describes cell counts: `State space scales with maze size (small: ~25 states, xl: ~900)`.
- **Task:** reach bottom-right room (`--goal far` default → `goal_room = (width-1, height-1)`); stochastic task = pick up "emerald".
- **Step limits:** `max_steps = min(width*height*4, 500)` → 100 for 5x5, 400 for 10x10, capped 500 (`scripts/phase6_maze_count.py:253`); stochastic default 500 (`stochastic_maze.py:60`).
- **Model:** Qwen2.5-0.5B-Instruct (frozen encoder, `~/models/Qwen2.5-0.5B-Instruct`), hidden_size 768, JEPA ensemble of 3 MLP predictors `Linear(2H→256)→ReLU→Linear(256→H)`, Adam lr=1e-3, one `train_step` per episode on collected transitions (`src/phase5/jepa_wm.py:40-46,60-86,166+`). Epistemic = mean squared deviation across ensemble predictions (`jepa_wm.py:149-163`).
- **Exploration modes:** pure_novelty (count only), jepa_only (ensemble variance only), hybrid (0.5*novelty + 0.5*epistemic) (`scripts/phase6_maze_jepa.py` MazeJEPAExplorer docstring). Count bonus = `0.5/sqrt(1+state_count) + 0.5/sqrt(1+(state,action)_count)`, backtrack penalty 0.5, success cache, action-priority tie-break (`scripts/phase6_maze_count.py` MazeNoveltyExplorer).
- **Stochastic respawn mechanics:** every step, per room: 30% chance all items vanish; `respawn_p=0.1` common item spawn; `rare_p=0.02` emerald spawn in rooms with `(x+y)%3==0`; items change BEFORE observation (`stochastic_maze.py:62-93`). State hash excludes transient items — only (x, y, inventory) (`stochastic_maze.py:26-32`).
- **Metrics:** FHT = first step reaching goal; SCR = unique visited states / steps; DLR = fraction of steps where 3 consecutive actions identical (`scripts/phase6_maze_count.py` compute_metrics).
- **Data sizes:** 6 episodes per 5x5 run (count), 6 episodes pure_novelty, 1 episode jepa_only 5x5. `--num-episodes` defaults: count 12, JEPA 6.
- **Hardware:** phase 6 scripts auto-detect `cuda` else `cpu`; workstation is Intel Core Ultra 9 185H (Intel Arc, no CUDA) → phase 6 runs were CPU. Conclusion declaration: `CPU (Intel Core Ultra 9) or GPU (NVIDIA T4 16GB)` (`PEDA_CONCLUSION.md:140`). Elapsed 43.1s for one jepa_only 5x5 episode incl. training.
- **Maze generation:** DFS recursive backtracker producing spanning tree (every cell reachable) (`src/phase6/maze_generator.py:74-112`); room templates at every cell; items scattered in ~1/3 of rooms (`item_count = max(width*height//3, 5)`, `maze_generator.py:127`); locked-door support exists but unused (env.setup() clears `_locked_doors`, `grid_env.py`).

---

## 4. Gaps and Contradictions (report to paper authors)

1. **E15/E16 raw data lost.** `results/phase6_maze_count_10x10_seed42.jsonl` is **0 bytes** (empty). No 20x20 or 30x30 result files exist anywhere (`results/**/*phase6*` = 5 files, all 5x5 or empty-10x10). E15 (10x10: 100/0/67) and E16 (20x20: 0/0) numbers exist ONLY in `PEDA_CONCLUSION.md:55-56` and `archive/phase5_jepa_exploration/README.md:15-16`. No raw DLR/SCR per size available for 10x10/20x20 — only aggregate success rates.
2. **30x30: no data at all.** Planned in script docstrings (`for s in 5 10 20 30`) but no result file, no doc row, no mention in conclusion. Do not cite 30x30 numbers.
3. **Stochastic maze results absent from disk.** No `phase6_stochastic_*.jsonl` files exist. The 100%/67%/0% stochastic numbers come only from `archive/phase5_jepa_exploration/README.md:17`. Also contradict the count-script expectation of 0%: count actually got 100% on the stochastic maze.
4. **"Hybrid 67%" attribution conflict.** `PEDA_CONCLUSION.md:55` attributes `Hybrid: 67%` to the deterministic Grid Maze 10x10 row, but `archive/.../README.md:15` shows deterministic Maze 10x10 "JEPA best: 0%" and `:17` shows the 67% figure under **Stochastic** 10x10. The two canonical docs disagree on whether 67% came from the deterministic or stochastic maze.
5. **State-count derivation undocumented.** Docs cite 1,100 (10x10) and 8,400 (20x20) states, but `GridMaze.state_estimate()` (`maze_generator.py:137-141`: `base * (items+1)`) yields 25×9=225 (5x5), 100×34=3,400 (10x10), 400×134=53,600 (20x20) — none match 1,100/8,400. The 1,100/8,400 derivation must be reconstructed or recomputed for the paper.
6. **5x5 count runs are degenerate.** All 6 count episodes are byte-identical (42 steps, fht 41, scr 0.405, goal (4,4)) because the maze uses fixed `seed=42` and the count explorer is deterministic with per-episode fresh counts — "seed" field is not a maze-generation seed. Do not present 5x5 count as 6 independent maze samples.
7. **jepa_only 5x5 data incomplete.** File has exactly 1 line (seed 42 only, 20 steps — max_steps=20, not the auto-scaled 100), suggesting an interrupted run or explicit `--max-steps 20`; no episodes for seeds 43+.
8. **all_modes 5x5 file is a concatenation of two runs** (2 failed 100-step pure_novelty lines + 4 successful lines) with no jepa_only/hybrid lines despite the filename; mode field = "pure_novelty" on every line.
9. **Work log silence.** `PEDA_WORKING_LOG.md` contains no phase6 entries — no contemporaneous record of when/how 10x10/20x20/stochastic runs were executed.
10. **Old manuscript superseded.** `PEDA_RESEARCH_MANUSCRIPT.md:570` lists Phase 6 only as `Scaling & Optimization 🔲 Planned` — do not cite manuscript for phase 6 results; use only its Theory (S2) and Architecture (S3) sections.
