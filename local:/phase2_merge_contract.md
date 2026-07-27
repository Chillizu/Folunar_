# Phase 2 Data Merge Contract

## Goal
Merge 3 data sources into a single deduplicated JSONL training set for sandbox_adapter_e4.

## Input Sources

### 1. PEDA self-trajectories (Strategy 1)
- Pattern: `results/phase2_peda_self_{task}.jsonl`
- Format: Each line = episode wrapper with `records` list inside
  ```json
  {"baseline": "peda", "task": "...", "records": [{"agent_type": "peda", "task_id": "...", "step": 0, "cwd": "..", "files": [...], "action": "..", "next_cwd": "..", "next_files": [...], "exit_code": N, "output": "..", "step_count": N}]}
  ```
- Extraction: for each line, extract all objects from `records[]` and flatten to individual JSONL lines.
- **KNOWN ISSUE**: PEDA is deterministic — all 25 episodes per task produce identical records. MUST deduplicate by `(cwd, files, action)` key. Only keep ONE copy of each unique transition per task.
- **KNOWN ISSUE**: Some episodes corrupted by Docker container collision. Filter: drop any record where `output` contains `"Error response from daemon"` or `next_files` is empty `[]` when `exit_code == 1`.

### 2. Expert demos (Strategy 2)
- File: `results/phase2_expert_demos.jsonl`
- Format: Already flat records (one per line), NO episode wrapper
  ```json
  {"agent_type": "expert_demo", "task_id": "...", "step": N, "cwd": "..", "files": [...], "action": "..", "next_cwd": "..", "next_files": [...], "exit_code": N, "output": "..", "step_count": N}
  ```
- These are hand-verified with real sandbox output — trust them as-is.
- May deduplicate by `(cwd, files, action)` but expert demos have intentional multi-step sequences so don't over-dedup — only dedup across different demo paths, not within the same path.

### 3. Original e2 training data
- File: `results/phase2_train_merged.jsonl` (the original 200 transitions)
- Format: same as PEDA self (episode wrapper with records[]), same extraction logic.

## Output
- File: `results/phase2_train_merged_v3.jsonl`
- Format: Flat JSONL, one record per line. Use the record schema:
  ```json
  {"cwd": "..", "files": [...], "action": "..", "next_cwd": "..", "next_files": [...], "exit_code": N, "output": ".."}
  ```
- No wrapper, no agent_type, no task_id, no step, no step_count — strip these.
- This is the exact format `phase2_synthetic_train.py` expects.

## Rules
1. Deduplicate by `(cwd, tuple(sorted(files)), action)` — same state + same action = same transition, keep one.
2. Filter corrupted records (empty next_files with exit_code=1 + daemon error).
3. Preserve all unique transitions from expert demos and original e2 data.
4. Output MUST be valid JSONL (one JSON object per line, no trailing comma).
5. Print summary: total records, unique records, per-task breakdown.
