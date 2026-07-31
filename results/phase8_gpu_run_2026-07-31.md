# Phase 8 GPU Run — 2026-07-31

## Run Configuration

- **Agent**: Phase8Runner (count-driven, no prediction error)
- **GPU instance**: i-06b0ba3dbdc214761 (g4dn.xlarge, T4 16GB), IP 13.220.38.201
- **Model**: Qwen2.5-0.5B-Instruct (CPU-only, not used for count-only run)
- **Docker images**: peda-sandbox:v2, peda-sandbox:v4 (rebuilt post-reboot)
- **Code**: src/phase8/count_driven_agent.py, scripts/phase8_closed_loop.py
- **Commit**: a348c1e (dev branch)

## Run Parameters

- Episodes per task: 5
- Max steps per episode: 10
- JEPA training: OFF (count-only)
- Command: `PYTHONPATH=src timeout 120 python3 scripts/phase8_closed_loop.py --task $task --docker-image $img --num-episodes 5 --max-steps 10`

## Count-Only Results

| # | Task | Image | Success | Avg Steps |
|:--:|------|:-----:|:-------:|:---------:|
| 1 | read_hello | v2 | 5/5 (100%) | 1.2 |
| 2 | read_note | v2 | 1/5 (20%) | 8.8 |
| 3 | count_lines | v2 | 0/5 (0%) | 10.0 |
| 4 | find_secret | v2 | 5/5 (100%) | 1.6 |
| 5 | read_welcome | v4 | 5/5 (100%) | 1.4 |
| 6 | find_api_key | v4 | 1/5 (20%) | 10.0 |
| 7 | count_measurements | v4 | 5/5 (100%) | 1.6 |
| 8 | find_errors_v4 | v4 | 1/5 (20%) | 10.0 |
| 9 | read_changelog_v4 | v4 | 5/5 (100%) | 1.2 |
| | **TOTAL** | | **28/45 (62.2%)** | |

### Task Categories

- **Direct reads (100%)**: read_hello, find_secret, read_welcome, count_measurements, read_changelog_v4 — success cache enables 1-2 step solves after initial discovery
- **Deep path reads (20%)**: read_note, find_api_key, find_errors_v4 — 10-step ceiling exhausted before reaching target file in deep directory
- **Zero (0%)**: count_lines — wc -l never targets the correct filename

## Count+JEPA Results

- Command: same as above with `--train-jepa` flag
- JEPA training: ON (forward dynamics as side-effect, not exploration driver)

| # | Task | Image | Success | Avg Steps |
|:--:|------|:-----:|:-------:|:---------:|
| 1 | read_hello | v2 | 5/5 (100%) | 1.2 |
| 2 | read_note | v2 | 1/5 (20%) | 8.8 |
| 3 | count_lines | v2 | 0/5 (0%) | 10.0 |
| 4 | find_secret | v2 | 5/5 (100%) | 1.6 |
| 5 | read_welcome | v4 | 5/5 (100%) | 1.4 |
| 6 | find_api_key | v4 | 1/5 (20%) | 10.0 |
| 7 | count_measurements | v4 | 5/5 (100%) | 1.6 |
| 8 | find_errors_v4 | v4 | 1/5 (20%) | 10.0 |
| 9 | read_changelog_v4 | v4 | 5/5 (100%) | 1.2 |
| | **TOTAL** | | **28/45 (62.2%)** | |

## JEPA Delta

| Metric | Count-Only | Count+JEPA | Delta |
|--------|:----------:|:----------:|:-----:|
| Total success | 28/45 (62.2%) | 28/45 (62.2%) | **0** |
| Per-task success | identical | identical | **0** |
| Avg steps | identical | identical | **0** |

**Conclusion**: JEPA forward dynamics training contributes zero additional value to the count-driven agent. Every task's success/failure pattern is identical with and without JEPA. This is consistent with 17 prior JEPA experiments where learned forward dynamics never improved exploration or task completion over count-based novelty.

## Raw Output

Full transcript artifact: this session, artifact://870 (count-only), artifact://872 (count+JEPA).
