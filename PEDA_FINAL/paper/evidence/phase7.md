# Phase 7 Evidence Bundle — GPU 5-Track (E17)

- **experiment_ids:** [E17]
- **Sub-IDs:** E17.1 (RSSM), E17.2 (Goal-JEPA), E17.3 (Giant-JEPA), E17.4 (Curriculum), E17.5 (Count)
- **Scout:** S9Phase7
- **Canonical source:** `PEDA_FINAL/PEDA_CONCLUSION.md` (ground truth, supersedes manuscript)
- **Verdict:** FAIL — "JEPA produces no differentiable epistemic signal at any scale tested"

---

## 1. Track Inventory (E17.1–E17.5)

From `PEDA_FINAL/PEDA_CONCLUSION.md:57`:

> `| 7 | GPU 5-track | 5 independent tracks (RSSM, Goal-JEPA, Giant-JEPA, Curriculum, Count) | Q1, Q3 | All JEPA tracks: DLR ~0.996 (near-perfect determinism, zero epistemic signal). Count wins every track | FAIL — JEPA produces no differentiable epistemic signal at any scale tested |`

| Sub-ID | Track | What it tested | Script | Persisted results? |
|---|---|---|---|---|
| E17.1 | RSSM | MiniRSSM (GRU + stochastic latent, prior-variance signal) vs SimpleJEPA (MLP ensemble) vs Count on 5x5 GridMaze | `scripts/phase7_rssm_experiment.py` | YES — 4 JSONL files |
| E17.2 | Goal-JEPA | Goal-conditioned JEPA (variance x goal-progress scaling) vs vanilla JEPA vs count on 10x10 | `scripts/phase7_goal_jepa.py` | **NO result file on disk** |
| E17.3 | Giant-JEPA | Lightweight 2-member JEPA vs count at 20x20/50x50/100x100 | `scripts/phase7_giant_maze.py` | **PARTIAL — count rows only; jepa rows missing** |
| E17.4 | Curriculum | 5 modes (pure_count, pure_jepa, hybrid, curriculum, ucb) at 10x10/20x20 | `scripts/phase7_curriculum_experiment.py` | **NO result file on disk** |
| E17.5 | Count | Count-based pair novelty (1/sqrt(count), 0.5x backtrack penalty); also stochastic-room control | `scripts/phase7_baseline_count.py`, embedded in rssm/giant/curriculum scripts | YES — 5x5 and 20x20 count rows |

---

## 2. Canonical Quantitative Claims (E17)

| Number | Source | Quote |
|---|---|---|
| DLR ~0.996 (all JEPA tracks) | `PEDA_FINAL/PEDA_CONCLUSION.md:15` | `- **Q1 (Signal):** LLM World Models produce epistemic error ~0 on small state spaces (<100 states) and uniform uncertainty on larger ones (JEPA ensemble, all DLR ~0.996). The model is too certain or uniformly uncertain — never differentially uncertain.` |
| DLR ~0.996; count wins every track | `PEDA_FINAL/PEDA_CONCLUSION.md:57` | `All JEPA tracks: DLR ~0.996 (near-perfect determinism, zero epistemic signal). Count wins every track` |
| JEPA loss 45 → 15; 37x cost | `PEDA_FINAL/PEDA_CONCLUSION.md:90` | `JEPA-based forward dynamics learn transition predictions (loss converges from 45 to 15 across training), but their uncertainty is uniform across all unexplored states. ... This is identical to count-based novelty (unvisited = uncertain) but computed at approximately 37x the computational cost (MLP forward pass + embedding computation vs integer increment).` |
| JEPA does not differentiate actions | `PEDA_FINAL/PEDA_CONCLUSION.md:124` | `**2. JEPA-style forward dynamics train but do not differentiate actions.** The JEPA MLP predictor learns to predict next-state embeddings from (state, action) pairs, as shown by decreasing loss curves across all experiments (loss 45 to 15). But the learned uncertainty is a scalar per transition, not a comparative signal across actions. All unexplored transitions are equally uncertain — equivalent to counting, at 37x the computational cost.` |
| Hardware: GPU NVIDIA T4 16GB | `PEDA_FINAL/PEDA_CONCLUSION.md:140` | `- **Training:** 65-1378 transitions, 1-3 epochs LoRA, 500-2000 steps JEPA, CPU (Intel Core Ultra 9) or GPU (NVIDIA T4 16GB)` |

---

## 3. Raw Persisted Numbers (E17.1, E17.5)

### E17.1 RSSM 5x5 experiment (`results/phase7_rssm_*_5x5_seed42.jsonl`)

Per-mode files (max-steps run at 20 for learned modes, 10 for count):

| Number | Source | Quote (JSON record) |
|---|---|---|
| scr 0.05, dlr 0.9, steps 20, loss 0.20126 | `results/phase7_rssm_rssm_5x5_seed42.jsonl:1` | `{"width": 5, "height": 5, "mode": "rssm", "seed": 42, "episode": 0, "steps_count": 20, "success": false, "fht": null, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 0.20126, "elapsed": 1.5}` |
| scr 0.05, dlr 0.9, steps 20, loss 0.03715 | `results/phase7_rssm_mlp_jepa_5x5_seed42.jsonl:1` | `{"width": 5, "height": 5, "mode": "mlp_jepa", "seed": 42, "episode": 0, "steps_count": 20, "success": false, "fht": null, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 0.03715, "elapsed": 1.5}` |
| scr 0.4, dlr 0.0, steps 10 | `results/phase7_rssm_count_5x5_seed42.jsonl:1` | `{"width": 5, "height": 5, "mode": "count", "seed": 42, "episode": 0, "steps_count": 10, "success": false, "fht": null, "scr": 0.4, "dead_loop_rate": 0.0, "train_loss": null, "elapsed": 0.0}` |

Interrupted `all_modes` file (10-step cap, all 3 modes):

| Number | Source | Quote |
|---|---|---|
| rssm: scr 0.1, dlr 0.8, loss 0.166887 | `results/phase7_rssm_all_modes_5x5_seed42.jsonl:1` | `{"width": 5, "height": 5, "mode": "rssm", "seed": 42, "episode": 0, "steps_count": 10, "success": false, "fht": null, "scr": 0.1, "dead_loop_rate": 0.8, "train_loss": 0.166887, "elapsed": 1.2}` |
| mlp_jepa: scr 0.1, dlr 0.8, loss 0.03715 | `results/phase7_rssm_all_modes_5x5_seed42.jsonl:2` | `{"width": 5, "height": 5, "mode": "mlp_jepa", "seed": 42, "episode": 0, "steps_count": 10, "success": false, "fht": null, "scr": 0.1, "dead_loop_rate": 0.8, "train_loss": 0.03715, "elapsed": 0.0}` |
| count: scr 0.4, dlr 0.0 | `results/phase7_rssm_all_modes_5x5_seed42.jsonl:3` | `{"width": 5, "height": 5, "mode": "count", "seed": 42, "episode": 0, "steps_count": 10, "success": false, "fht": null, "scr": 0.4, "dead_loop_rate": 0.0, "train_loss": null, "elapsed": 0.0}` |

### E17.3 Giant maze (count rows only)

| Number | Source | Quote |
|---|---|---|
| count 20x20: 0/3 success, scr 0.0, max_dist 20, steps 500 | `results/phase7_giant_20x20.jsonl:1-3` (identical rows seeds 42/43/44) | `{"size": "20x20", "width": 20, "height": 20, "method": "count", "seed": 42, "episode": 0, "steps_count": 500, "success": false, "fht": -1, "scr": 0.0, "max_dist": 20, "train_loss": null, "goal_x": 19, "goal_y": 19, "elapsed": 0.0}` |
| combined file identical (count-only) | `results/phase7_giant_all.jsonl:1-3` | same 3 count records; **no `method: "jepa"` rows exist in either file** |

### Adjacent evidence (Phase 6, E14-style JEPA-only, for DLR comparison)

| Number | Source | Quote |
|---|---|---|
| jepa_only: scr 0.05, dlr 0.9, loss 49.93 | `results/phase6_maze_jepa_jepa_only_5x5_seed42.jsonl:1` | `{"width": 5, "height": 5, "mode": "jepa_only", "seed": 42, "episode": 0, "steps_count": 20, "success": false, "fht": null, "scr": 0.05, "dead_loop_rate": 0.9, "train_loss": 49.932384, "elapsed": 43.1}` |

---

## 4. Experimental Conditions (E17)

- **Models:** Qwen2.5-0.5B-Instruct frozen encoder (mean-pooled last hidden, 768-dim) for Goal-JEPA and Giant-JEPA; CPU-friendly learned encoders (35-dim one-hot features) for MiniRSSM/SimpleJEPA; no LoRA in Phase 7.
- **Environments:** GridMazeEnv 5x5 (25 rooms) for E17.1/E17.5; 10x10 for E17.2/E17.4 (planned); GiantMaze 20x20-100x100 (up to ~100M states) for E17.3; RandomMazeEnv (stochastic room descriptions) for the JEPA-vs-count stochastic-text control (`scripts/phase7_baseline_jepa.py`, `scripts/phase7_baseline_count.py`).
- **Data sizes:** 5x5 = 25 states; 20x20 = 400 cells (~8400 states with inventory per PEDA_CONCLUSION.md:56); 100x100 = 10K cells, `state_estimate()` = base * (items+1) (`src/phase7/giant_maze.py`, sparse items ~1 per 20 cells).
- **Hardware:** GPU NVIDIA T4 16GB (`PEDA_FINAL/PEDA_CONCLUSION.md:140`); instance g4dn.xlarge documented for the GPU run in `results/phase8_gpu_run_2026-07-31.md:5` (`- **GPU instance**: i-06b0ba3dbdc214761 (g4dn.xlarge, T4 16GB), IP 13.220.38.201`) and planning doc `PEDA_FINAL/PHASE4_EXPERIMENT_PLAN.md:267` (`**Instance type:** g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM), us-east-1.`); torch pinned at 2.13.0 in `uv.lock:2257` (`name = "torch"` / `version = "2.13.0"`).
- **Episodes/steps:** RSSM 5x5 run: 1 episode/mode (persisted), 10-20 steps; giant 20x20: 3 episodes x 500 steps; curriculum default 6 episodes x 500 steps; goal_jepa default 3 episodes x 500 steps; baseline_jepa/count default 3 episodes x 500 steps.

---

## 5. Methodology & Implementation Details (E17)

### E17.1 RSSM architecture (`src/phase7/rssm_wm.py`)

- MiniRSSM: `self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)` — `rssm_wm.py:138`; defaults `state_dim=32, action_dim=16, hidden_dim=128, latent_dim=16` — `rssm_wm.py:121-124`; instantiated identically in `scripts/phase7_rssm_experiment.py:580-583`.
- Posterior `nn.Linear(hidden_dim + state_dim, 2 * latent_dim)` = Linear(160, 32); prior `nn.Linear(hidden_dim, 2 * latent_dim)` = Linear(128, 32) — `rssm_wm.py:140-143`.
- Decoder: Linear(16,128) → ReLU → Linear(128,32). State features: x_oh(10)+y_oh(10)+inv(14)+goal(1) = 35 dims (`rssm_wm.py:18-20`). Action embedding table size 64 (hash-based).
- Training: `rssm_training_step` — Adam lr 1e-3, KL weight 0.1, per-step hidden/latent detach (no BPTT) — `rssm_wm.py:377-411`.
- **Caveat:** docstring promises "Ensemble of 3 RSSMs provides epistemic uncertainty via prediction variance" (`rssm_wm.py:10`), but the experiment builds ONE MiniRSSM and scores actions with prior-variance magnitude: `# Use prior variance as epistemic signal` / `# Prior variance magnitude = epistemic uncertainty about latent` — `scripts/phase7_rssm_experiment.py:203-210`. The RSSM track never actually ran ensemble variance.
- TemporalJEPA (fallback): GRUCell(state 32 + action 16, hidden 64), 3 MLP predictors (`rssm_wm.py:325-328`). SimpleJEPA: 3 MLPs, no recurrence. Only SimpleJEPA (`mlp_jepa` mode) was used in the persisted runs.

### E17.2 Goal-JEPA (`src/phase7/goal_jepa.py`)

- Frozen Qwen encoder, 3 GoalMLP predictors hidden_dim=256 (`goal_jepa.py:100`), input = 768*2 + 64 action emb (`goal_jepa.py:92-94`); trainable action projection 768→64 (`goal_jepa.py:86-90`).
- Scoring: `score = epistemic * (1.0 + max(0.0, progress))` where progress = current_dist_to_goal − predicted_dist_to_goal in embedding space (`goal_jepa.py:169-174`).
- Goal text: "reach the Treasury at position (9,9)" on 10x10 (`scripts/phase7_goal_jepa.py`). Hypothesis: goal-progress scaling gives epistemic uncertainty "genuine discriminatory power" (>0% success where vanilla JEPA achieves 0%).

### E17.3 Giant-JEPA (`src/phase7/giant_jepa.py`)

- 2 ensemble members, hidden 128 ("vs phase5's 896 → 256 → 896 × 3"); MLP 896→128→25→896 (`giant_jepa.py:28-36`); down-sampled training `max_samples=5` random transitions per episode (`giant_jepa.py:114-116`).
- Hypothesis (docstring): "At 100x100 scale (~100M states), JEPA's learned embedding abstraction might compress the state space and provide useful uncertainty where counting gives uniformly high novelty."
- SIZE_CONFIGS: 20x20 (500 steps), 50x50 (1000), 100x100 (2000), 3 episodes each (`scripts/phase7_giant_maze.py`). "DO NOT run experiments from this script alone — GPU Manager orchestrates."

### E17.4 Curriculum (`src/phase7/curriculum_explorer.py`)

- warmup_episodes=2 (pure count, alpha=0), phase_in=5 episodes linear ramp, epistemic_cap=0.7 (`curriculum_explorer.py:14-17,62-66`).
- Score: `nov * (1 + alpha * ep)` if nov > 0 else `ep * 0.1` (`curriculum_explorer.py:69-70`).
- 5 modes in `scripts/phase7_curriculum_experiment.py`: pure_count, pure_jepa (variance only), hybrid (0.5/0.5, "known to dilute"), curriculum, ucb (0.7*nov + 0.3*ep*sqrt(log(N)/(1+n_a))). Hypothesis: "curriculum > hybrid, possibly > count at 20x20."

### E17.5 Count (all scripts)

- `novelty_bonus = 0.5 * (1/sqrt(1+state_count)) + 0.5 * (1/sqrt(1+pair_count))`, backtrack (reverse-move) penalty ×0.5; success cache replays winning action per state hash.
- Random maze control: `RandomGridState.hash_key()` excludes stochastic description — "count-based novelty expires after one visit per room" while JEPA text includes it (`src/phase7/random_maze.py:78-83,107-110`). Key metric: avg_desc_per_room.

---

## 6. Verbatim Conclusions (E17)

- `PEDA_FINAL/PEDA_CONCLUSION.md:15` — "The model is too certain or uniformly uncertain — never differentially uncertain."
- `PEDA_FINAL/PEDA_CONCLUSION.md:57` — "All JEPA tracks: DLR ~0.996 (near-perfect determinism, zero epistemic signal). Count wins every track | FAIL — JEPA produces no differentiable epistemic signal at any scale tested"
- `PEDA_FINAL/PEDA_CONCLUSION.md:90` — "The learned signal is 'how uncertain am I about this (state, action) transition?' — which is equally high for every transition the agent has never seen. This is identical to count-based novelty (unvisited = uncertain) but computed at approximately 37x the computational cost"
- `PEDA_FINAL/PEDA_CONCLUSION.md:124` — "All unexplored transitions are equally uncertain — equivalent to counting, at 37x the computational cost. Breaking this uniformity requires goal-conditioned embeddings or learned value in the representation space"
- `PEDA_FINAL/PEDA_CONCLUSION.md:137` (Declaration) — "**Model:** Qwen2.5-0.5B-Instruct with LoRA (rank=16), JEPA MLP predictors (1-3 hidden layers), zero-shot RSSM"
- `PEDA_FINAL/PEDA_CONCLUSION.md` (What Survived) — "PEDA's dead-loop immunity. Across all phases (1-8), PEDA consistently showed zero dead-loop rate, versus Pragmatic's 48-80% and Random's variable rates."

---

## 7. Contradictions & Gaps (E17) — MUST be flagged in the paper

1. **DLR ~0.996 is NOT reproducible from persisted data.** The canonical claim (PEDA_CONCLUSION.md:15,57) cites DLR ~0.996 for "all JEPA tracks," but every persisted Phase 7 JEPA record shows DLR 0.8–0.9 (phase7_rssm_rssm_5x5:1, phase7_rssm_mlp_jepa_5x5:1, phase7_rssm_all_modes:1-2, phase6 jepa_only:1). No raw record contains 0.996; the figure presumably came from GPU-run logs that were never persisted. Paper must either cite the conclusion doc as the source or note the discrepancy.
2. **3 of 5 tracks have NO persisted results.** Goal-JEPA (E17.2), Curriculum (E17.4), and the Random-Maze JEPA/count controls produced zero result files in `results/`. `results/` contains only 6 phase7 JSONLs (4 rssm 5x5 + 2 giant).
3. **Giant-JEPA (E17.3) JEPA rows missing entirely.** `phase7_giant_all.jsonl` is byte-identical to `phase7_giant_20x20.jsonl` (3 count-only rows). No 50x50/100x100 rows and no `method:"jepa"` rows exist — the "any scale tested" claim rests on unpersisted runs.
4. **RSSM track did not use the documented ensemble.** Docstring promises ensemble-of-3 variance (rssm_wm.py:10); implementation uses single-model prior variance (phase7_rssm_experiment.py:203-210). The paper's "zero-shot RSSM" framing should reflect prior-variance, not ensemble-variance.
5. **RSSM/MLP-JEPA run-length inconsistency:** per-mode files show 20 steps (dlr 0.9) while all_modes shows 10 steps (dlr 0.8) for the same modes — evidence the runs were truncated/interrupted; train_loss for rssm differs (0.20126 vs 0.166887) between the two files.
6. **No Phase 7-specific hardware log.** g4dn.xlarge/T4 is documented for the Phase 8 GPU run (phase8_gpu_run_2026-07-31.md:5) and the Phase 4 plan (PHASE4_EXPERIMENT_PLAN.md:267); torch 2.13.0 only in uv.lock:2257. The Phase 7 GPU run itself has no persisted run-config doc.
