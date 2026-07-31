# Phase 9 Hypothesis-Generator — MVP Signal Validation

- commit: `81fe8b293a6b7678f10e52916d5f13e0aa4d0b21`
- timestamp: 2026-07-31T07:43:45+00:00
- host: mioarch | cpu
- sandbox_image: peda-sandbox:v4
- model: STRIPSDiscriminator (primary) + MLPDiscriminator (arm); LLM baseline skipped
- seeds: [1, 2, 3]
- LLM direct baseline: SKIPPED (MVP; later phase)

| Metric | Threshold | per-seed | mean | SD |
|---|---|---|---|---|
| V1 AUC_disc | >= 0.7 | [0.7593, 0.8816, 0.7817] | 0.8075 | 0.0532 |
| F1 KL(emp || uniform) [held-out] | >= 0.35 nats | [0.572, 0.6563, 0.572] | 0.6001 | 0.0397 |
| F2 KL(P_heldout || P_train) | >= 0.3 nats | [0.7904, 1.2312, 0.8563] | 0.9593 | 0.1941 |
| V2 Cohen's d (E vs D) | >= 1.0 | [1.1822, 1.9087, 1.2792] | 1.4567 | 0.3221 |
| V3 Spearman rho | > 0.4 | [0.6983, 0.839, 0.7243] | 0.7539 | 0.0611 |

## Fail-fast gates

| Gate | Condition | Dead when |
|---|---|---|
| FF-HG-1 (F1) | KL(emp || uniform) >= 0.35 | flat error field |
| FF-HG-2 (F2) | AUC >= 0.7 AND KL >= 0.3 | no novel/known separation |
| FF-HG-3 (F3) | Cohen's d >= 1.0 | error = visit-count only |
| FF-HG-4 (F4) | proposer diversity | N/A in MVP (no LLM) |

### seed 1: F1_flatness=PASS, F2_separation=PASS, F3_count_orthogonality=PASS, V3_gradient_optional=PASS

- n transitions: train=97, held-out=143
- probes: D=27, E=30
- AUC_disc=0.7593 (secondary over transitions: 0.5014)
- KL(emp||uniform)=0.572 nats; KL(heldout||train)=0.7904 nats
- Cohen's d(E,D)=1.1822; Spearman rho=0.6983
- error D: mean=0.2074 sd=0.1796; error E: mean=0.4933 sd=0.2864

### seed 2: F1_flatness=PASS, F2_separation=PASS, F3_count_orthogonality=PASS, V3_gradient_optional=PASS

- n transitions: train=85, held-out=155
- probes: D=29, E=30
- AUC_disc=0.8816 (secondary over transitions: 0.4805)
- KL(emp||uniform)=0.6563 nats; KL(heldout||train)=1.2312 nats
- Cohen's d(E,D)=1.9087; Spearman rho=0.839
- error D: mean=0.1517 sd=0.1825; error E: mean=0.5933 sd=0.2703

### seed 3: F1_flatness=PASS, F2_separation=PASS, F3_count_orthogonality=PASS, V3_gradient_optional=PASS

- n transitions: train=88, held-out=152
- probes: D=30, E=30
- AUC_disc=0.7817 (secondary over transitions: 0.4932)
- KL(emp||uniform)=0.572 nats; KL(heldout||train)=0.8563 nats
- Cohen's d(E,D)=1.2792; Spearman rho=0.7243
- error D: mean=0.1867 sd=0.1814; error E: mean=0.4933 sd=0.2864

## Verdict (per PHASE9_PLAN.md)

FF-HG-1/2/3 verdicts: see gate columns above. Any FAIL with its measured value is a pre-registered dead-end condition — no post-hoc threshold adjustment permitted.
