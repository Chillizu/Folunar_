# Phase 2 e4 Evaluation Contract (Updated)

## Goal
Evaluate sandbox_adapter_e4 against e2 baseline.

## Background
- No `scripts/phase2_measure_l1l2l3.py` exists locally (created ad-hoc on GPU instance).
- Fallback: use `scripts/phase2_collect_data.py` PEDA smoke test + S3 results for L1/L2/L3 comparison.

## Test A: PEDA Multi-Episode Smoke Test
```bash
source venv/bin/activate
for task in read_note count_lines read_hello find_secret; do
  python scripts/phase2_collect_data.py \
    --baseline peda --task "$task" --max-steps 10 --num-episodes 5 \
    --adapter-path checkpoints/phase2/sandbox_adapter_e4 \
    --output results/e4_peda_"$task".jsonl
done
```
- Expect: FHT=0, SCR=1.0 on all tasks (same as e2).

## Test B: Build quick L1/L2/L3 measurement
If the agent is capable, create a minimal L1/L2/L3 script that:
1. Loads the adapter
2. For each sample in OOD test set (results/phase2_ood_test.jsonl, limit 20):
   - Predict next state given (state, action)
   - L1: match predicted exit_code vs actual — accuracy
   - L2: match predicted files list (Jaccard) vs actual — accuracy ≥ 0.5
   - L3: match predicted output prefix vs actual — accuracy ≥ 0.5
3. Report averages

Compare against e2 baseline from S3:
- `s3://chillizu-peda-checkpoints/phase2/results/e2_heldout_l1l2l3.json`
- `s3://chillizu-peda-checkpoints/phase2/results/e2_ood_l1l2l3.json`

## Success Criteria
- [ ] All 4 tasks complete (FHT=0, SCR=1.0) per PEDA smoke test
- [ ] L1/L2/L3 held-out ≥ e2 baseline
- [ ] L1/L2/L3 OOD ≥ e2 baseline
- [ ] If L1/L2/L3 measurement infeasible, at minimum smoke test passes on all tasks

## Reference
- Best e2 results: L1=1.000, L2=0.900, L3=0.550 (held-out); L1=1.000, L2=0.900, L3=0.400 (OOD)
