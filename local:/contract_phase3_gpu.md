# Phase 3: Epistemic Validation on GPU

## Role
Mini-Orchestrator. Delegate execution, NEVER run code yourself.

## GPU Instance
- Instance ID: i-0cbdb085a1e726bef
- IP: 44.211.123.115
- User: ec2-user
- SSH key: /tmp/peda-temp-key (push before each SSH: `aws ec2-instance-connect send-ssh-public-key --instance-id i-0cbdb085a1e726bef --instance-os-user ec2-user --ssh-public-key file:///tmp/peda-temp-key.pub --region us-east-1`)
- Python: /opt/pytorch/bin/python (PyTorch 2.11 + CUDA 13.2 + T4 15GB)
- Note: SSH banner warns about PQ key exchange — ignore, it's a known harmless warning

## Pre-Read on Local Machine
1. `scripts/phase3_experiment.py` and `scripts/phase3_analysis.py` — experiment scripts
2. `checkpoints/phase1/partial_adapter_real_25_e3/` — Grid World partial adapter
3. `src/phase1/grid_env.py` and `src/phase1/run.py` — Grid World environment

## Task: Run Epistemic vs Pragmatic Controlled Experiment

### Slice C1: Sync Code to GPU
- Copy Folunar_ project to GPU instance (rsync or git clone)
- Target: /home/ec2-user/Folunar_/
- Must include: src/, scripts/, checkpoints/phase1/partial_adapter_real_25_e3/
- Verify adapter loads on GPU: `/opt/pytorch/bin/python -c "from peft import PeftModel; ..."`

### Slice C2: Run goal_known Condition
- 10 episodes PEDA, 10 episodes pragmatic-only
- Grid World 5×5, partial adapter
- Goal in known region (trained state-action space)
- Record: success, steps, revisit_rate, epistemic_error
- Save to results/phase3_gpu/goal_known.jsonl
- Use tmux to survive SSH disconnect

### Slice C3: Run goal_unknown Condition
- 10 episodes PEDA, 10 episodes pragmatic-only
- Start in unknown region (NOT in training data)
- Record: success, steps, revisit_rate, epistemic_error
- Save to results/phase3_gpu/goal_unknown.jsonl
- THE CRITICAL SLICE

### Slice C4: Statistical Analysis
- Fisher exact test: goal_unknown success rate PEDA vs Pragmatic
- Mann-Whitney U: goal_unknown steps PEDA vs Pragmatic
- Effect size (Cohen's h, Cliff's delta)
- Verdict: PASS (p<0.05, PEDA better) or FAIL

### Slice C5: Download Results
- rsync results back to local machine: results/phase3_gpu/
- rsync any new adapter checkpoints

## Output
```json
{
  "goal_known": { "peda_success": float, "prag_success": float, "peda_steps": float, "prag_steps": float },
  "goal_unknown": { "peda_success": float, "prag_success": float, "peda_steps": float, "prag_steps": float },
  "statistics": { "fisher_p": float, "mann_whitney_p": float, "cohens_h": float },
  "verdict": "PASS|FAIL|INCONCLUSIVE",
  "results_local": "results/phase3_gpu/"
}
```

## Rules
- Sub-subagents: one per slice. C2 and C3 can parallelize on GPU (two tmux sessions).
- Use tmux for long-running training/eval on GPU.
- Epic uncertainty = variance across ensemble checkpoints (if multiple exist).
- Target runtime: 10-30 min total.
- Verify ALL results before reporting.
