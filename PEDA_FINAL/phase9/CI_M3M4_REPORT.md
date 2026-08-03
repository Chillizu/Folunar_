# Phase 9 Direction 1 — M3/M4 Completion Report (CI Sandbox)

**Date:** 2026-08-02 (updated after GPU clean rerun)
**Author:** L1-CI / L1-CI2 (GPU rerun + final adjudication)
**Scope:** M3 (behavioral productivity, PE vs count) + M4 (error trajectory), FF-CI-6 verdict.
**Status:** M3 COMPLETE (GPU clean rerun: PE 12/30 = 0.400 vs count 11/30 = 0.367); M4 COMPLETE (FAIL); **FF-CI-6 PASS (not triggered)**; OOM root cause fixed (§6 RESOLVED)

---

## 0. Environment & Protocol Deviations (MUST read first)

| Item | Pre-registered | Actual | Impact |
|---|---|---|---|
| Compute | GPU (g4dn.xlarge T4) | Mixed: M4 + count on local CPU (mioarch; AWS session expired at the time); **M3 PEDA final rerun on AWS GPU (g4dn.xlarge T4, ip-172-31-85-234) after credentials restored** | GPU run ~10-30x faster; episode counts halved (see next row) |
| M3 episodes | 20 eps × 3 tasks × max_steps 20 | **10 eps × 3 tasks × max_steps 20** (CPU-sanctioned halving; retained on GPU per Main) | Early/late split uses ep1-5 vs ep6-10 |
| M3 count | 20 eps | 10 eps canonical (matched) + **supplementary 20 eps** (`phase9_ci_m3_count_20ep.jsonl`) | FF-CI-5 discussed from 20ep file |
| M3 PEDA | 20 eps × 3 tasks | **10 eps × 3 tasks on AWS GPU (g4dn.xlarge T4) — clean full rerun, COMPLETE** (`phase9_ci_m3_peda_gpu.jsonl`, 30/30 episodes). CPU partial (OOM-truncated) kept as reference (`phase9_ci_m3_peda_cpu_partial_readsecret10ep.jsonl` + `phase9_ci_m3_peda_partial_killed.jsonl`). GPU instance code ≡ local dev 9570016 (verified by sha256 of 6 key files; instance git hash 6a86ee6 recorded in meta) | bs=1 finetune deviation (see §0 Script fixes) |
| M4 | 3 trials × 40 steps | 3 × 40 (interrupted once by sibling `tmux kill-server` at trial 3; resumed as detached hub process, merged) | All 120 rows present |
| Model | Qwen2.5-0.5B-Instruct fp32 | same, fp32 (CPU for M4/count; GPU T4 for M3 PEDA rerun) | — |

**Script fixes (in scope):**
- `phase9_ci_m3.py`: incremental per-episode JSONL appends; resume support (skips episodes already on disk); summary split adaptive to episode count; loads counterpart side from disk for verdict.
- `phase9_ci_m4.py`: incremental per-trial appends; sandbox `max_steps` now set to 40 (was 20 → would truncate trials at step 20, making E(31..40) unmeasurable); `_predict_model` made transformers-4.50-compatible (apply_chat_template returns tensor, not BatchEncoding).
- `src/phase2/run.py` (commit 9570016): **OOM root cause fixed at the source** — `SandboxLearningModule.update()` now calls `lora_finetune(..., batch_size=1)` (was 4) + `malloc_trim(0)` after the update. See §6 for evidence.

**OOM root cause (RESOLVED — commit 9570016):** the earlier diagnosis (glibc malloc arena bloat) was INCOMPLETE. With `MALLOC_ARENA_MAX=2` the steady-state RSS is ~2.5-3.3GB (verified), but the episode-end `lora_finetune` forward/backward in fp32 materializes the full autograd activation graph for the 0.5B model — **~13-16GB at batch_size=4** (measured: both L1-CI's probe and my first clean rerun were OOM-killed at 13.7/15.7GB anon-rss exactly during the first update, ~30-45 min in; the update allocates fast, so `MALLOC_ARENA_MAX` — which caps arena COUNT, not single large allocations — cannot prevent it). **Fix: batch_size 4→1** (peak ~5.8GB incl. model, verified by probe: healthy loss, no NaN) + `malloc_trim(0)` after each update (verified: VmRSS returns to ~2.5GB after the update, no ratchet across 2 sequential updates). GPU rerun ran 30/30 episodes to completion with RSS stable and no OOM. **Rerun command (GPU, used):**
`MALLOC_ARENA_MAX=2 python3 scripts/phase9_ci_m3.py --agent peda --num-episodes 10 --max-steps 20` (device=cuda)

**bs=1 deviation note:** batch_size=1 vs the original 4 changes the online training dynamics slightly (more, smaller gradient steps per update — 20 vs 5 at first update). This is an internal training hyperparameter, NOT pre-registered; the behavioral comparison (PE vs count) is the pre-registered gate. The loss at bs=1 is healthy (0.63 → 0.09 across a 20-sample update), whereas the bs=4 path showed NaN loss in the pre-fix probe.

**Coordination notes:** L1-QW's `bdc1f68` (10:11) broke the CI count baseline (`cd_child` AttributeError); fixed by L1-QW in `91a0c3c`. M4 trial-3 interruption: sibling `tmux kill-server`; rerun via detached hub process, merged; meta `interrupted_and_resumed: true`.

---

## 1. M3 — Behavioral Productivity (PE vs count)

### 1.1 Count baseline (COMPLETE, matched 10ep + supplementary 20ep)

| task | ep1-10 | ep11-20 (20ep run) | total (20ep) | 10ep |
|---|---|---|---|---|
| read_secret_ci | 4/10 | 9/10 | 13/30 (0.433) | 4/10 |
| read_data_ci | 7/10 | 10/10 | 17/30 (0.567) | 7/10 |
| find_warn_ci | 0/10 | 0/10 | 0/30 (0.000) | 0/10 |
| **POOLED** | 11/30 | 19/30 | **30/90 (0.333)** | 11/30 (0.367) |

Key: count **learns** read_secret/read_data strongly in the second half (success-cache replay), but **never** discovers inverted grep (find_warn_ci 0/30).

**FF-CI-5 note:** count pooled = 0.333 < 0.40 threshold → the gate formally triggers, but it is entirely driven by find_warn_ci (0/30); the two L1 tasks are 0.433/0.567 and improving. The env is NOT uniformly too hard — one task is beyond the count agent's candidate/novelty mechanism. Flagged for top-level adjudication.

### 1.2 PEDA side (COMPLETE — GPU clean rerun, 30/30 episodes)

From `phase9_ci_m3_peda_gpu.jsonl` (T4, fp32, bs=1 finetune, max_steps 20):

| task | ep1-5 | ep6-10 | total | steps (ep0→ep9) |
|---|---|---|---|---|
| read_secret_ci | 0/5 | 4/5 | **4/10** | 20,20,20,20,20,20,2,2,2,2 |
| read_data_ci | 3/5 | 5/5 | **8/10** | 20,20,2,2,2,2,2,2,2,2 |
| find_warn_ci | 0/5 | 0/5 | **0/10** | 20,20,20,20,20,20,20,20,20,20 |
| **POOLED** | 3/15 | 9/15 | **12/30 (0.400)** | — |

Learning curve: on both L1 tasks the agent discovers the echo-as-reader reversal mid-run, then locks in **2-step wins** (victory at step 1) for the remainder (read_secret: ep6-9 four consecutive; read_data: ep2-9 eight consecutive). Epistemic error drives the EFE selection to `echo` (epi_err reaches 1.000 on read_data successes — the model has maximal uncertainty about the reversal). **find_warn_ci is never solved (0/10)** — the inverted-grep L2 task defeats the PE mechanism just as it defeats count (0/30).

**CPU partial reference** (`phase9_ci_m3_peda_cpu_partial_readsecret10ep.jsonl`, OOM-truncated pre-fix run): read_secret_ci 3/10 (ep1@17, ep5@4, ep6@20 — noisy rediscovery, never consolidated). The GPU run is the protocol-faithful clean rerun and supersedes it; the CPU partial confirms the same mechanism (in-episode LoRA consolidation → 2-step wins) with different timing.

### 1.3 FF-CI-6 verdict

**PASS — NOT TRIGGERED.** Pre-registered fail-fast: PE < count − 10pp → formal negative result.

| metric | PE (GPU) | count | delta | criterion | verdict |
|---|---|---|---|---|---|
| Pooled vs count 10ep (matched) | 12/30 = 0.400 | 11/30 = 0.367 | **+3.3pp** | PE ≥ 0.267 | **PASS** |
| Pooled vs count 20ep (supplementary) | 12/30 = 0.400 | 30/90 = 0.333 | **+6.7pp** | PE ≥ 0.233 | **PASS** |
| read_secret_ci | 4/10 | 4/10 (13/30 on 20ep) | 0 | — | tie |
| read_data_ci | 8/10 | 7/10 (17/30 on 20ep) | +1 | — | PE better |
| find_warn_ci | 0/10 | 0/10 (0/30 on 20ep) | 0 | — | tie |

Sub-metrics (10ep split, ep1-5 vs ep6-10): PE pooled 0.200 → 0.600 (**3×, meets ≥2×**); count 0.267 → 0.467 (**1.75×, does NOT meet 2×** in either 10ep or 20ep split). Discovery steps: PE min victory_step = 1 vs count = 2 → **PE ≤ 1.5× count satisfied**.

**Adjudication (see §7):** prediction error is NOT worse than counting by 10pp — the failure hypothesis (FF-CI-6) is rejected. PE is numerically ≥ count pooled on both matched and supplementary count baselines, driven by read_data (+1 task); read_secret and find_warn are exact ties. The margin is thin (+3.3pp pooled) and task-dependent — this is a weak positive, not a PEDA vindication: both agents fully fail find_warn_ci, and on read_secret both land at 4/10. **The result does NOT support the counter-intuitive-sandbox narrative that prediction error is a superior drive signal; it supports "at least as productive as counting" and decisively fails to trigger the pre-registered kill condition.**

---

## 2. M4 — Error Trajectory E(t) (COMPLETE)

**Pre-registered:** E(1..10) ≥ 0.5; E(31..40) ≤ 0.5·E(1..10); E(t) = 1 − DLR(t); DLR = mean(L1 exit_code, L2 files_delta, L3 output_nonempty).

### 2.1 Numbers

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| E(1..10) | **0.6444** | ≥ 0.5 | **PASS** |
| E(31..40) | **0.4333** | ≤ 0.5·E(1..10) = 0.3222 | **FAIL** |
| M4 overall | — | — | **FAIL** |

Per-trial: trial 0: 0.633 → 0.333 (−47%); trial 1: 0.667 → 0.633 (no drop); trial 2: 0.633 → 0.333 (−47%).

### 2.2 Per-step E(t) (mean over 3 trials; full table in `results/phase9_ci_m4_summary.csv`)

| step | mean_dlr | mean_e | step | mean_dlr | mean_e |
|---|---|---|---|---|---|
| 1 | 0.333 | 0.667 | 21 | 0.333 | 0.667 |
| 5 | 0.111 | 0.889 | 25 | 0.667 | 0.333 |
| 10 | 0.556 | 0.444 | 30 | 0.556 | 0.444 |
| 15 | 0.111 | 0.889 | 35 | 0.556 | 0.444 |
| 20 | 0.556 | 0.444 | 40 | 0.556 | 0.444 |

### 2.3 Interpretation — the two explanations, separated

**Explanation A — "the signal does not exist (in this online setup)":** partially supported by trial 1 (E(31..40)=0.633 ≈ E(1..10)=0.667 — LoRA updates did nothing that trial) and by the pooled miss (0.433 > 0.322).

**Explanation B — "batch learning ≠ online setting":** strongly supported by mechanism data:
1. M2 (batch) proved the rules ARE learnable: LoRA on 200 CI transitions → held-out DLR 0.713 (E ≈ 0.287). The signal exists in batch.
2. Online, the agent's action distribution **collapses**: after the step-20 LoRA update, the EFE policy converges to `pwd`-only (observed in trials 0/2). The world model then only ever sees the agent's self-selected actions — once the agent stops executing reversed commands (`echo`/`cat`/`grep`), the model never collects their transitions, so E(t) can never drop below the `pwd` plateau (~0.33). Per-step DLR for `pwd` = 0.667 → E = 0.333 floor.
3. Trials 0/2 DO show in-episode learning (E −47%), exactly the M4 mechanism — but it stalls at the plateau instead of reaching ≤0.25.
4. Update cadence: only 2 LoRA updates per 40-step trial, each on a tiny (≤20) self-correlated buffer sample.

**Verdict:** M4 FAIL per pre-registration. Primary cause is **B (online/batch mismatch: exploration collapse + tiny updates cap E(t) at the pwd plateau)**; the "signal absent" reading (A) holds only for the degenerate online policy, NOT the environment (contradicted by M2 DLR 0.713).

---

## 3. File inventory

| File | Contents | Rows |
|---|---|---|
| `results/phase9_ci_m3_count.jsonl` | M3 count baseline 10ep × 3 tasks (canonical, matched) | 31 |
| `results/phase9_ci_m3_count_20ep.jsonl` | M3 count supplementary 20ep × 3 tasks (FF-CI-5 context) | 91 |
| `results/phase9_ci_m3_peda_gpu.jsonl` | **M3 PE clean rerun, T4 GPU, 10ep × 3 tasks (primary for FF-CI-6)** | 31 |
| `results/phase9_ci_m3_peda_cpu_partial_readsecret10ep.jsonl` | M3 PE CPU partial snapshot (post-OOM-fix run, stopped by Main for GPU migration): read_secret 3/10 | 11 |
| `results/phase9_ci_m3_peda_partial_killed.jsonl` | M3 PE OOM-truncated evidence: 10/10 read_secret_ci + 1/10 read_data_ci (pre-fix) | 12 |
| `results/phase9_ci_m3_peda_residue_2ep.jsonl` | M3 PE early-failure residue (2ep, pre-fix verification attempt) | 3 |
| `results/phase9_ci_m3_summary.csv` | M3 summary (count side) | — |
| `results/phase9_ci_m4.jsonl` | M4 per-step 3 trials × 40 steps (merged, interrupted_and_resumed) | 121 |
| `results/phase9_ci_m4_summary.csv` | M4 per-step summary + verdict | — |
| `scripts/phase9_ci_m3.py` | M3 driver (fixed: incremental, resume, adaptive split) | — |
| `scripts/phase9_ci_m4.py` | M4 driver (fixed: incremental, max_steps=40, tf-4.50 compat) | — |
| `scripts/phase9_ci_m3m4_analyze.py` | analysis helper (count/M4 paths) | — |
| `scripts/phase9_ci_m3m4_gpu_analyze.py` | analysis helper for the GPU PE run (FF-CI-6 tables/curves) | — |
| `PEDA_FINAL/phase9/CI_M3M4_REPORT.md` | this report | — |

D4 compliance: all experiment files carry the meta header (phase, direction, commit, timestamp, host, cpu_or_gpu, sandbox_images, model, per_episode_data_present). GPU meta: commit 6a86ee6 (instance git; content ≡ local 9570016 by sha256), host ip-172-31-85-234, device cuda, ts 2026-08-02T12:51:30Z.

---

## 4. What was NOT done / caveats

- **Compute:** final M3 PEDA run on AWS GPU (g4dn.xlarge T4) per Main after credentials restored; earlier CPU attempts OOM-blocked (now fixed). M3 episodes remain 10 (CPU-sanctioned halving, retained); the "≥2× improvement" window is ep1-5 vs ep6-10.
- **bs=1 deviation** (commit 9570016): online finetune batch_size 4→1 to fit the fp32 autograd activation graph in memory; not pre-registered, does not change the behavioral gate. See §0.
- **FF-CI-5 formally triggers** at pooled count 0.333 < 0.40 (20ep), but driven entirely by find_warn_ci (0/30); flagged, not adjudicated (top-level).
- **find_warn_ci fails for BOTH agents** (PE 0/10 GPU, count 0/30) — the L2 inverted-grep task is beyond both mechanisms; FF-CI-6 verdict rests on the two L1 tasks (ties/win).
- **count does not meet the 2× improvement sub-metric** (10ep: 0.267→0.467; 20ep: 0.367→0.633, both <2×); PE meets it (0.200→0.600). The pre-registered M3 phrasing is "both improve ≥ 2×" — count misses; recorded, not adjudicated here.
- M4 trial-1 no-learning anomaly reported as variance (1/3 trials).
- No changes to `PHASE9_PLAN.md` (owned by top-level acceptance).

---

## 5. Verdict summary

| Gate | Result |
|---|---|
| M3 | **COMPLETE** — PE (GPU clean rerun) pooled 12/30 = 0.400 vs count 10ep 11/30 = 0.367 (+3.3pp) / count 20ep 0.333 (+6.7pp); PE meets ≥2× improvement + discovery ≤1.5×; count misses the 2× sub-metric |
| M4 | **FAIL** (E(1..10)=0.644 PASS; E(31..40)=0.433 > 0.322 FAIL) |
| FF-CI-6 | **PASS — NOT TRIGGERED** (PE 0.400 ≥ count 0.367 − 0.10; failure hypothesis rejected; formal negative result NOT declared) |

---

## 6. OOM root cause & fix — RESOLVED (commit 9570016)

**Status: RESOLVED.** Two clean CPU rerun attempts (L1-CI's `m3-peda` and L1-CI2's `m3peda`) plus a probe all died at the same point — the episode-end LoRA update — with anon-rss 13.7/15.7GB, despite `MALLOC_ARENA_MAX=2` (steady-state ~2.5-3.3GB was fine). Evidence chain: L1-CI's RSS probe (trace `/tmp/rss_trace.txt`) shows 2.5GB stable through 20+ min of inference, then `[lora_finetune] batch 0` → killed at 13.7GB. Root cause: `lora_finetune` at batch_size=4, fp32, 0.5B model materializes the full autograd activation graph — a single large allocation class that arena caps cannot bound. Fix (commit 9570016, `src/phase2/run.py`): batch_size=1 (probe-verified peak 5.78GB incl. model, healthy loss 0.63→0.09, 20-sample update in ~53s CPU) + `malloc_trim(0)` after each update (probe-verified VmRSS returns to ~2.5GB, no ratchet over 2 sequential updates). GPU clean rerun then ran 30/30 episodes with stable memory.

---

## 7. FF-CI-6 adjudication (2026-08-02, post-GPU-rerun)

**Pre-registered criterion:** PE ≥ count − 10pp (pooled over the 3 CI tasks); PE < count − 10pp → formal negative result, charter-accepted.

**Numbers (PE = GPU clean rerun, 10ep × 3 tasks):**
- PE pooled **12/30 = 0.400**; count pooled (matched 10ep) **11/30 = 0.367**; delta **+3.3pp**. vs count 20ep (30/90 = 0.333): delta **+6.7pp**.
- Per-task (PE vs count10): read_secret_ci 4/10 = 4/10 (tie); read_data_ci 8/10 vs 7/10 (+1); find_warn_ci 0/10 = 0/10 (tie).
- Learning curves: read_secret ep6-9 four consecutive 2-step wins (victory_step 1); read_data ep2-9 eight consecutive 2-step wins; find_warn 0/10 (all 20-step fails). Pooled early/late: 0.200 → 0.600 (3×).
- Discovery steps: PE 1 ≤ 1.5 × count 2.

**Decision: FF-CI-6 PASS — the failure hypothesis ("prediction error still not a useful drive signal") is REJECTED.** PE is numerically ≥ count on both count baselines and never falls below count − 10pp on any task (min per-task delta 0).

**Honest reading (do not over-claim):**
1. The margin is thin (+3.3pp pooled, one task win) — PE does not beat count convincingly, it ties-or-wins slightly.
2. Both agents fail find_warn_ci entirely; the CI sandbox's L2 task is beyond both mechanisms, so the verdict rests on the two L1 reversals.
3. The M3 PASS coexists with M4 FAIL coherently: M3 measures whether prediction error DRIVES exploration (EFE selects high-epistemic-error actions — it does, epi_err=1.000 on read_data successes); M4 measures whether the model's error SHRINKS in-episode (it doesn't reach the threshold — exploration collapse + tiny online updates cap E(t)). Prediction error can drive behavior without the world model converging.
4. Not a PEDA vindication: the counter-intuitive narrative (structured epistemic signal beats counting) is only weakly supported; the strongest honest claim is "PE is at least as productive as count on CI tasks, within the pre-registered non-inferiority bound."

**Coordination note:** the CPU partial evidence (read_secret 3/10, noisy rediscovery) was superseded by the GPU clean rerun per the protocol-faithful decision; preserved in `phase9_ci_m3_peda_cpu_partial_readsecret10ep.jsonl`. Instance not terminated until data verified locally (done: 31 rows, meta/30 episodes checked).
