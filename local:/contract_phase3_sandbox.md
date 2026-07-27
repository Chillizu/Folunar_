# Phase 3 Sandbox: Epistemic vs Pragmatic on GPU

## Role
Mini-Orchestrator. Delegate ALL execution.

## GPU Instance
- ID: i-0cbdb085a1e726bef, IP: 44.211.123.115, user: ec2-user
- Python: /opt/pytorch/bin/python (CUDA T4 15GB)
- SSH: push key then `ssh -o StrictHostKeyChecking=no -i /tmp/peda-temp-key ec2-user@44.211.123.115`
- Key push: `aws ec2-instance-connect send-ssh-public-key --instance-id i-0cbdb085a1e726bef --instance-os-user ec2-user --ssh-public-key file:///tmp/peda-temp-key.pub --region us-east-1`

## Why This Is The Real Test
Grid World Phase 3 failed because 0.5B generalizes perfectly on 5×5 grid (ceiling effect).
Sandbox v2 is different: v2_full adapter held-out L1=0.70-0.80 — WM is GENUINELY uncertain.
This is where epistemic signal should matter.

## Experiment Design

### Environment
- Sandbox v2 (Docker peda-sandbox:v2)
- Adapter: checkpoints/phase2/sandbox_adapter_v2_full/ (65 transitions, L1~0.70-0.80 held-out)
- Task: read_hello (cat hello.txt → "Hello World") — simpler than read_note

### Conditions (4 total)
1. goal_known + PEDA: start in known cwd, full EFE
2. goal_known + pragmatic: start in known cwd, epistemic_weight=0
3. goal_unknown + PEDA: start in unknown cwd, full EFE
4. goal_unknown + pragmatic: start in unknown cwd, epistemic_weight=0

### Known vs Unknown
- Known cwds: /sandbox, /sandbox/data, /sandbox/docs (from v2_full training data)
- Unknown cwds: /sandbox/logs, /sandbox/projects, /sandbox/tmp (NOT in training)

### Sample Size
- N>=5 per condition (sandbox is slower than Grid World)
- Total: 20 episodes minimum

### Metrics
- Success rate (primary)
- Mean steps to completion
- Revisit rate
- Mean epistemic error (PEDA only)
- First action choice

## Execution

### Slice 1: Sync to GPU
- rsync Folunar_ to /home/ec2-user/Folunar_/ (include adapter + data)
- Verify: `/opt/pytorch/bin/python -c "from peft import PeftModel; ..."`

### Slice 2: Run Experiment
- Use scripts/phase2_collect_data.py (it already supports --baseline peda/pragmatic)
- OR create a simple runner script that loops over conditions
- Use tmux (2 sessions: goal_known + goal_unknown in parallel)
- Each episode: max_steps=10 (or 5 for speed)
- Save per-episode JSONL to results/phase3_sandbox/

### Slice 3: Statistical Analysis
- Fisher exact: goal_unknown success rate PEDA vs Pragmatic
- Mann-Whitney U: steps
- Effect size
- Verdict: PASS (p<0.05) or FAIL

### Slice 4: Download Results
- rsync results/ back to local

## Output
```json
{
  "goal_known": {"peda_success": float, "prag_success": float, "peda_steps": float, "prag_steps": float},
  "goal_unknown": {"peda_success": float, "prag_success": float, "peda_steps": float, "prag_steps": float},
  "statistics": {"fisher_p": float, "mann_whitney_p": float, "effect_size": float},
  "verdict": "PASS|FAIL|INCONCLUSIVE",
  "results_local": "results/phase3_sandbox/"
}
```

## Rules
- Sub-subagents for independent slices
- Use tmux on GPU
- If read_hello works, can optionally run read_note too
- Target: 30-60 min GPU
- DO NOT run sandbox experiments on CPU
