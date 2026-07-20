# PEDA Phase 1 Archive Summary

**Date**: 2026-07-20  
**Status**: Formally complete on the 5×5 Grid World; core mechanism not validated in this environment.

## Objective
Implement the PEDA Phase 1 (Grid World) mid-term objective: use an LLM as the World Model backbone in a 5×5 grid, verify prediction-error-driven autonomous exploration, and produce an initial Pareto-frontier drive-weight recommendation.

## Formal Gate Results

| Gate | Metric | Value | Threshold | Status |
|---|---|---|---|---|
| G1 | Next-state accuracy | 1.0000 | > 0.90 | PASS |
| G2 | Drive steps / random steps | 0.4337 | < 0.50 | PASS |
| G3 | Revisit rate | 0.0000 | < 0.20 | PASS |

- Evaluation: 10 episodes with real LLM + LoRA adapter, subprocess isolation, `max-candidates=4`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Adapter: `checkpoints/phase1/partial_adapter_real_25_e3`

## Drive-Weight Recommendation

```json
{"curiosity": 0.5, "competence": 0.5, "boredom": 0.5, "novelty": 0.5}
```

Stored in `config/phase1_default_drives.json`.

## What Was Verified

1. The LoRA adapter raises next-state accuracy from G1≈0.18 (base model) to G1=1.0 on the 5×5 training distribution.
2. Real-LLM multi-episode evaluation requires subprocess isolation to avoid intermittent hangs.
3. `max-candidates=4` is required; `max-candidates=2` limits the action space to UP/DOWN and causes failure.
4. The adapter generalizes to held-out obstacle layouts (100% success on completed episodes), with near-zero epistemic error.

## What Was NOT Verified

- The **prediction-error-driven exploration mechanism** could not be demonstrated in the 5×5 Grid World.
- Both clean-grid and obstacle-grid comparisons show PEDA and pragmatic-only planning behave identically when the World Model is certain.
- Even the base 0.5B model (G1≈0.18) solves the clean grid perfectly, confirming the environment is too simple.

## Conclusion

Phase 1 form deliverables are complete and archived. The Grid World successfully validates the PEDA loop mechanics and the LoRA adapter's predictive power, but it cannot validate the core hypothesis that the Drive System enables exploration. Mechanism validation must happen in a richer environment, such as Phase 2 sandbox or TextWorld.

## Artifacts

- `results/phase1_eval.json`
- `results/phase1_report.md`
- `results/phase1_grid_search.json`
- `results/phase1_shell_eval.jsonl`
- `results/phase1_shell_grid_search.jsonl`
- `results/phase1_base_model_comparison_summary.json`
- `results/phase1_heldout_summary.json`
- `results/phase1_heldout_*.jsonl`
- `config/phase1_default_drives.json`
- `scripts/phase1_shell_eval.sh`
- `scripts/phase1_shell_grid_search.sh`
- `scripts/phase1_base_model_comparison.sh`
- `scripts/phase1_heldout_test.sh`
- `scripts/phase1_heldout_episode.py`
- `AGENTS.md`
- `PEDA_WORKING_LOG.md`

## Next Phase

Transition to Phase 2 (sandbox environment) to validate the prediction-error-driven exploration mechanism in a setting where the World Model is genuinely uncertain and the Drive System can provide measurable value.
