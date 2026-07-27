# Phase 2 e4 Training Contract

## Goal
Train a new LoRA adapter `sandbox_adapter_e4` on the merged v3 dataset.

## Input
- Data: `results/phase2_train_merged_v3.jsonl` (produced by merge step)
- Base model: `~/models/Qwen2.5-0.5B-Instruct`
- Script: `scripts/phase2_synthetic_train.py`

## Command
```bash
python scripts/phase2_synthetic_train.py \
  --data results/phase2_train_merged_v3.jsonl \
  --output-dir checkpoints/phase2/sandbox_adapter_e4 \
  --epochs 3 \
  --batch-size 4
```

## Success Criteria
- [ ] Training completes without error
- [ ] `checkpoints/phase2/sandbox_adapter_e4/adapter_model.safetensors` exists
- [ ] At least 1 epoch checkpoint saved (`checkpoint_epoch_1/`)
- [ ] Final epoch loss printed
- [ ] `training_info.json` or equivalent manifest written

## Notes
- Uses venv: `source venv/bin/activate`
- CPU-only training. Expect ~10-30 min depending on dataset size.
- If OOM or CUDA errors on CPU, try `--batch-size 1`.
- Print loss per epoch.
