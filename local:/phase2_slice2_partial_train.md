# Slice 2: Partial Adapter Training

## Goal
Train `sandbox_adapter_v2_partial` on KNOWN sandbox areas only.

## Input
- `results/phase2_v2_known_train.jsonl` (from Slice 1)
- Base model: `~/models/Qwen2.5-0.5B-Instruct`
- Script: `scripts/phase2_synthetic_train.py`

## Steps
1. Wrap flat records into episode format: `{"task":"explore", "baseline":"systematic", "records":[flat_record]}` → write to `results/phase2_v2_known_wrapped.jsonl`
2. Train:
   ```bash
   source venv/bin/activate
   python scripts/phase2_synthetic_train.py \
     --data results/phase2_v2_known_wrapped.jsonl \
     --output-dir checkpoints/phase2/sandbox_adapter_v2_partial \
     --epochs 3 --batch-size 4
   ```

## Success
- TRAINING_FINISHED
- `checkpoints/phase2/sandbox_adapter_v2_partial/adapter_model.safetensors` exists
- Loss decreases: epoch 3 < epoch 1
