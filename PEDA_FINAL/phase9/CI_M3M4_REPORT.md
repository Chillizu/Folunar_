# Phase 9 Direction 1 — M3/M4 Completion Report (CI Sandbox)

**Date:** 2026-08-02
**Author:** L1-CI
**Scope:** M3 (behavioral productivity, PE vs count) + M4 (error trajectory), FF-CI-6 verdict.
**Status:** M4 COMPLETE (FAIL); M3 count COMPLETE; M3 PEDA **BLOCKED by OOM** (partial data + ready scripts; see §6)

---

## 0. Environment & Protocol Deviations (MUST read first)

| Item | Pre-registered | Actual | Impact |
|---|---|---|---|
| Compute | GPU (g4dn.xlarge T4) | **CPU only** (local mioarch; AWS session expired, Main: no new instances) | Runs 10-30x slower; episode counts adjusted |
| M3 episodes | 20 eps × 3 tasks × max_steps 20 | **10 eps × 3 tasks × max_steps 20** (CPU-sanctioned halving) | Early/late split uses ep1-5 vs ep6-10 |
| M3 count | 20 eps | 10 eps canonical (matched) + **supplementary 20 eps** (`phase9_ci_m3_count_20ep.jsonl`) | FF-CI-5 discussed from 20ep file |
| M3 PEDA | 20 eps × 3 tasks | **BLOCKED**: 4 OOM kills (exit 137) — completed 10/10 read_secret_ci + 1/10 read_data_ci before each kill (same point, ~45 min in) | partial data in `phase9_ci_m3_peda_partial_killed.jsonl` |
| M4 | 3 trials × 40 steps | 3 × 40 (interrupted once by sibling `tmux kill-server` at trial 3; resumed as detached hub process, merged) | All 120 rows present |
| Model | Qwen2.5-0.5B-Instruct fp32 | same, CPU | — |

**Script fixes (in scope):**
- `phase9_ci_m3.py`: incremental per-episode JSONL appends; resume support (skips episodes already on disk); summary split adaptive to episode count; loads counterpart side from disk for verdict.
- `phase9_ci_m4.py`: incremental per-trial appends; sandbox `max_steps` now set to 40 (was 20 → would truncate trials at step 20, making E(31..40) unmeasurable); `_predict_model` made transformers-4.50-compatible (apply_chat_template returns tensor, not BatchEncoding).

**OOM root cause (critical for any rerun):** the M3 PEDA process RSS plateaued at 11.7GB for a 0.5B model → glibc malloc arena bloat (22 cores → up to 8×22 arenas × 64MB). Verified fix: `MALLOC_ARENA_MAX=2` caps RSS at ~3.3GB (probe: 3279MB after load + churn). All 4 kills were global OOM with the process as victim (oom_score_adj=200, anon-rss ~15GB) triggered by `mihomo-smart` invoking the oom-killer under combined sibling memory pressure. **Rerun command (ready):**
`MALLOC_ARENA_MAX=2 python3 scripts/phase9_ci_m3.py --agent peda --num-episodes 10 --max-steps 20`
resume-safe: skips the 11 episodes already on disk (note: fresh process = fresh wm, so read_secret's LoRA learning does NOT transfer; a full clean rerun is the protocol-faithful choice).

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

### 1.2 PEDA side (BLOCKED — partial evidence)

From `phase9_ci_m3_peda_partial_killed.jsonl` (10/10 read_secret_ci episodes completed before OOM):

| episode | success | steps |
|---|---|---|
| 0 | fail | 20 |
| 1 | fail | 20 |
| 2 | success | 2 |
| 3 | success | 2 |
| 4 | success | 2 |
| 5 | success | 2 |
| 6 | success | 2 |
| 7 | success | 2 |
| 8 | success | 2 |
| 9 | success | 2 |

**read_secret_ci: PE 8/10 in 2 steps each after 2 discovery episodes — vs count 4/10.** The PE agent's in-episode LoRA learning (1 update per 20-step episode at step 19) clearly discovers the echo-as-reader reversal. On the one task completed, PE (0.80) beats count (0.40) by +40pp — opposite direction from the FF-CI-6 failure hypothesis. **This is partial evidence only** (1 of 3 tasks; read_data/find_warn not completed due to OOM).

### 1.3 FF-CI-6 verdict

**NOT ADJUDICABLE** — PEDA side incomplete (10/30 episodes across 1 task). Pre-registered criterion (PE ≥ count − 10pp pooled over 3 tasks) cannot be evaluated. Partial evidence on read_secret_ci suggests PE ≥ count (0.80 vs 0.40), but this is 1 task and must not be extrapolated. **Rerun required** (script ready, OOM fix verified, resume-safe).

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
| `results/phase9_ci_m3_peda_partial_killed.jsonl` | M3 PE partial: 10/10 read_secret_ci + 1/10 read_data_ci (OOM evidence) | 12 |
| `results/phase9_ci_m3_summary.csv` | M3 summary (count side; partial verdict) | — |
| `results/phase9_ci_m4.jsonl` | M4 per-step 3 trials × 40 steps (merged, interrupted_and_resumed) | 121 |
| `results/phase9_ci_m4_summary.csv` | M4 per-step summary + verdict | — |
| `scripts/phase9_ci_m3.py` | M3 driver (fixed: incremental, resume, adaptive split) | — |
| `scripts/phase9_ci_m4.py` | M4 driver (fixed: incremental, max_steps=40, tf-4.50 compat) | — |
| `scripts/phase9_ci_m3m4_analyze.py` | analysis helper | — |
| `PEDA_FINAL/phase9/CI_M3M4_REPORT.md` | this report | — |

D4 compliance: all experiment files carry the meta header (phase, direction, commit, timestamp, host, cpu_or_gpu, sandbox_images, model, per_episode_data_present).

---

## 4. What was NOT done / caveats

- **GPU path skipped** per Main (AWS session expiry + no new instances). All numbers CPU-mode. M3 episodes halved (10 vs 20); the "≥2× improvement" window is ep1-5 vs ep6-10.
- **M3 PEDA incomplete** — 4 OOM kills at the same point (read_data_ci ep1, ~45 min in). Root cause fixed (MALLOC_ARENA_MAX=2 verified); scripts resume-safe; a clean rerun is the protocol-faithful path. **This is a BLOCKING gap for FF-CI-6.**
- **FF-CI-5 formally triggers** at pooled count 0.333 < 0.40, but driven entirely by find_warn_ci (0/30); flagged, not adjudicated (top-level).
- M4 trial-1 no-learning anomaly reported as variance (1/3 trials).
- No changes to `PHASE9_PLAN.md` (owned by top-level acceptance).

---

## 5. Verdict summary

| Gate | Result |
|---|---|
| M3 | **INCOMPLETE** (count done; PEDA blocked by OOM — partial: PE 8/10 vs count 4/10 on read_secret_ci) |
| M4 | **FAIL** (E(1..10)=0.644 PASS; E(31..40)=0.433 > 0.322 FAIL) |
| FF-CI-6 | **NOT ADJUDICABLE** — needs complete PEDA side (scripts ready; rerun command in §0) |
