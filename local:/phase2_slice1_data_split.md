# Slice 1: Partial Training Data Split

## Goal
Split v2 systematic data into known (train) and unknown (test) sets by sandbox directory.

## Input
- `results/phase2_v2_all.jsonl` — 65 flat transitions, one per line
  ```json
  {"cwd": "/sandbox/docs", "files": ["note.txt",...], "action": "cat note.txt", ...}
  ```

## Rules
- KNOWN cwds (train): `/sandbox`, `/sandbox/docs`, `/sandbox/data`
- UNKNOWN cwds (test): `/sandbox/logs`, `/sandbox/projects`, `/sandbox/projects/app`, `/sandbox/projects/lib`
- Transitions with `cwd` in KNOWN → training set
- Transitions with `cwd` in UNKNOWN → test set

## Output
1. `results/phase2_v2_known_train.jsonl` — flat JSONL, same schema as input
2. `results/phase2_v2_unknown_test.jsonl` — flat JSONL, same schema

## Verification
- Print counts: known vs unknown transitions
- Print unique cwds in each set
- Record: agent_type="systematic" for all entries
