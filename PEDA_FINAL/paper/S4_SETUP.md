# 4. Experimental Setup

All numbers cite the evidence bundles (`PEDA_FINAL/paper/evidence/*.md`) plus primary source files; every claim below is REPRODUCIBLE or PARTIAL per `CLAIMS_VS_EVIDENCE.md`.

## 4.1 Model

- **Base model:** Qwen2.5-0.5B-Instruct, loaded locally from `~/models/` — (evidence/phase1.md:50, results/phase1_eval.json: `"model": "Qwen/Qwen2.5-0.5B-Instruct"`).
- **Fine-tuning:** LoRA `r=16, alpha=32, dropout=0.05, bias="none", task_type="CAUSAL_LM"`, targets 7 linear projections — (evidence/phase2.md:119, `checkpoints/phase2/sandbox_adapter_e2/adapter_config.json`: `"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none"`; evidence/phase1.md:54, `src/phase1/world_model.py:41-43, 69-76`).
- **Training scale:** 65–1,378 transitions, 1–3 LoRA epochs; later phases add JEPA MLP predictors (1–3 hidden layers, `n_ensemble=3`, Adam `lr=1e-3`) and zero-shot RSSM — (PEDA_FINAL/PEDA_CONCLUSION.md:140, "65-1378 transitions, 1-3 epochs LoRA, 500-2000 steps JEPA"; evidence/phase5.md:158-159, `src/phase5/jepa_wm.py:37-46, 53-60`).
- **Inference mode:** real-LLM (no stubs); Phase 1 eval isolated in subprocesses, `max-candidates=4` — (evidence/phase1.md:58, `results/phase1_report.md` caveats 4-5).

## 4.2 Hardware

| Resource | CPU workstation (Phases 1–2) | GPU instance (Phases 3–8) |
|---|---|---|
| CPU | Intel Core Ultra 9 185H, 22 cores | g4dn.xlarge: 4 vCPU |
| GPU | None (`CUDA: false`; Intel Arc unusable by PyTorch) | NVIDIA T4 16 GB |
| RAM | 30 GB (18 GB available during experiment) | 16 GB |
| Region / runtime | local | AWS us-east-1; torch 2.13.0 |

- CPU row — (evidence/phase3.md:122, `cpu: Intel Core Ultra 9 185H (22 cores), gpu: None (CUDA: false), ram: 30GB`; evidence/phase2.md:136, "Intel ARC not usable by PyTorch"; evidence/phase2.md:146, `torch 2.13.0, cuda_available false`).
- GPU row — (evidence/phase7.md:79, `PHASE4_EXPERIMENT_PLAN.md:267`: "Instance type: g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM), us-east-1").
- **CPU latency made PEDA infeasible:** first call ~176 s cold start, then ~3 s/call; 12–24 inference calls/step with 3-ensemble → 10–60+ min/episode — (evidence/phase3.md:120, `results/phase3_experiment/report.json`); Phase 1 CPU predict ~2.4–3.1 s/call, ~16 s/step, 100-episode eval ~22 h — (evidence/phase1.md:57). GPU runs: Grid World N=20 in 876 s (14.6 min) — (evidence/phase3.md:113); Phase 2 GPU session ~4.5 h / ~$2.40 — (evidence/phase2.md:145); Phase 4 ~14 GPU-hours — (evidence/phase4.md:42).

## 4.3 Environments

| Environment | Scale | Actions | Notes |
|---|---|---|---|
| Grid World (Ph. 1) | 5×5, max_steps=50 | UP/DOWN/LEFT/RIGHT (4) | rewards wall −0.2 / move −0.05 / goal +1.0 |
| TextRoomEnv (Ph. 1.5) | 2 rooms (study↔hallway) | 6 | custom env, 3-step optimal; real TextWorld never evaluated |
| Busybox Sandbox (Ph. 2–5, 8) | v1→v4 (see Table 2) | shell commands, 12-command whitelist | Docker-contained (see §4.4) |
| Grid Maze (Ph. 6–7) | 5×5 / 10×10 / 20×20 | 4 moves | max_steps = min(w·h·4, 500); sizes cited, not state counts |

- Grid World — (evidence/phase1.md:49, `5x5 GridWorld, max_steps=50, rewards wall -0.2 / move -0.05 / goal +1.0`; evidence/phase1.md:140, 4 actions).
- TextRoomEnv — (evidence/phase1_5.md:11, `src/phase1_5/text_env.py:1-4`: "Two rooms connected by a door"; evidence/phase1_5.md:19, "real TextWorld was never used for evaluation"; evidence/phase1_5.md:83, "3-step optimal").
- Grid Maze — (evidence/phase6.md:105, `max_steps = min(width*height*4, 500)`; CLAIMS_VS_EVIDENCE.md:80, CANONICAL: "cite maze size (10x10, 20x20), not state counts").

**Table 2. Sandbox versions v1–v4** — (evidence/phase2.md:87-90):

| Version | Dirs | Files | Unique (s,a) | Source |
|---|---|---|---|---|
| v1 | 4 incl. root (docs, tmp, data) | 3 (hello.txt, docs/note.txt, data/lines.txt) | 22 | phase2.md:87 (`Dockerfile.busybox`) |
| v2 | 7 subdirs | 14 | 65 | phase2.md:88 (AGENTS.md:120, "3.0× v1") |
| v3 | 7 subdirs | 15 [INFERENCE] | — | phase2.md:89 (`Dockerfile.busybox_v3`) |
| v4 | 18 incl. root | 29 [INFERENCE] | 270 | phase2.md:90 (`tasks.py:123` "18 dirs"); phase5.md:24 |

## 4.4 Docker containment

- Per-episode container: `docker run -d --rm --cap-drop=ALL --read-only --tmpfs /tmp --network none` — (src/phase2/sandbox_env.py:138-141, verbatim flags; evidence/phase8.md:88).
- Command safety: 12-command whitelist `{ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep, find}` + 14 blocklist regexes (`rm/mv/cp/chmod/chown/dd/mkfs/mount/sudo/su/docker/kill/shutdown/reboot`) — (src/phase2/sandbox_env.py:17-24, verbatim `WHITELIST`/`BLOCKLIST_PATTERNS`; evidence/phase8.md:88).
- Read-only rootfs ⇒ `create_file: LIMIT` — (evidence/phase2.md:81, PEDA_WORKING_LOG.md:1505). Environment reward is always 0; all signal comes from the binary goal predicate — (evidence/phase8.md:146).

## 4.5 Metrics

- **G1/G2/G3 (Grid World):** G1 next-state prediction accuracy > 0.90; G2 mean steps < 0.50 × random; G3 revisit rate < 0.20 — (evidence/phase1.md:20, `scripts/phase1_eval.py`: "G1 = {g1:.4f} (target > 0.90)"; theory.md:192).
- **L1/L2/L3 (Sandbox), thresholds 0.90/0.70/0.50:** L1 = exact exit-code match ≥ 0.90; L2 = exact predicted files-set match ≥ 0.70; L3 = token-overlap ≥ 0.5 on `last_output` ≥ 0.50 — (evidence/phase2.md:21, `scripts/phase2_measure_l1l2l3.py:126-128`; evidence/phase2.md:24, definitions).
- **FHT:** step index of the first action passing the task's goal check; −1 if never — (evidence/phase2.md:81, `scripts/phase2_collect_data.py:206-219`).
- **SCR:** |unique (cwd, files) states| / steps — (evidence/phase2.md:83, `phase2_collect_data.py:221-224`).
- **DLR:** fraction of steps i≥2 where actions[i]==actions[i−1]==actions[i−2] — (evidence/phase2.md:84, `phase2_collect_data.py:225-229`).
- **Methodology correction:** the `success` field was `SCR > 0` (constant-true tautology); all Phase 3+ hit rates use FHT≥0 — (CLAIMS_VS_EVIDENCE.md:43, `phase3_sandbox_experiment.py:132`; CLAIMS_VS_EVIDENCE.md:56).

## 4.6 Baselines and conditions

- **Pragmatic:** `pragmatic_only` agent scoring only the EFE pragmatic term (goal-distance minimization), same `pragmatic_weight=3.0` as PEDA — (evidence/phase2.md:147; evidence/phase1.md:130, "Pragmatic-only: pragmatic_only=True, SAME pragmatic_weight=3.0").
- **Random:** uniform action selection, seed 42 — (evidence/phase2.md:147, "random (seed 42)").
- **Heuristic:** random + repetition penalty (avoid >2 repeats in last 5) — (evidence/phase2.md:147).
- **Count:** count-based pair novelty `0.5·(1/√(1+state_count)) + 0.5·(1/√(1+pair_count))`, backtrack penalty ×0.5, success cache — (evidence/phase5.md:180, `src/phase5/explorer.py:28-36`; evidence/phase7.md:115).
- **Fairness controls:** identical (start, goal, seed) episode pairs across agents; known/unknown CWDs counterbalanced round-robin (7,7,6 per CWD) — (evidence/phase1.md:130; evidence/phase3.md:45).
- **Drive config:** Phase 1 weights all 0.5 — (evidence/phase1.md:59); Phases 1.5/2: curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0, pragmatic_weight=3.0 — (evidence/phase2.md:148).
