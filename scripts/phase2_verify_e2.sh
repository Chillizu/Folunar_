#!/usr/bin/env bash
set -euo pipefail

# phase2_verify_e2.sh — Verify sandbox_adapter_e2 with ensemble mode
# Run this AFTER sandbox_adapter_e2 training finishes.
# Usage: bash scripts/phase2_verify_e2.sh [--max-steps N] [--task read_note]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MAX_STEPS="${1:-10}"
TASK="${2:-read_note}"
ADAPTER_PATH="checkpoints/phase2/sandbox_adapter_e2"
OUTPUT_JSONL="results/phase2_verify_e2_ensemble.jsonl"
OUTPUT_LOG="results/phase2_verify_e2_ensemble.log"

echo "======================================================================"
echo " Phase 2 — e2 Ensemble Verification"
echo "======================================================================"
echo "Adapter:  $ADAPTER_PATH"
echo "Baseline: peda"
echo "Task:     $TASK"
echo "Steps:    $MAX_STEPS"
echo "Output:   $OUTPUT_JSONL"
echo "======================================================================"
echo ""

python scripts/phase2_collect_data.py \
  --baseline peda \
  --task "$TASK" \
  --max-steps "$MAX_STEPS" \
  --adapter-path "$ADAPTER_PATH" \
  --output "$OUTPUT_JSONL" \
  2>&1 | tee "$OUTPUT_LOG"

EXIT_CODE="${PIPESTATUS[0]}"

echo ""
echo "======================================================================"
echo " Verification Complete (exit=$EXIT_CODE)"
echo "======================================================================"
echo ""
echo "--- Action Sequence ---"
grep -E '^\s+\[step' "$OUTPUT_LOG" || true
echo ""
echo "--- Metrics ---"
grep -E '^\s+-> steps=' "$OUTPUT_LOG" || true
echo ""
echo "--- Full result ---"
cat "$OUTPUT_JSONL" 2>/dev/null | python3 -m json.tool 2>/dev/null || true

echo ""
echo "======================================================================"
echo " Interpretation Guide"
echo "======================================================================"
echo ""
echo "SUCCESS indicators:"
echo "  - Action diverges from the ls/ls-data oscillation seen in e1"
echo "  - FHT is not null (task completed within max_steps)"
echo "  - SCR > 0.2 (above the e1 sanity threshold)"
echo "  - At least one action is ls->cat, cd, or any non-ls command"
echo ""
echo "FAILURE indicators (same as e1):"
echo "  - Action pattern: ls / ls data / ls / ls data ... (oscillation)"
echo "  - FHT = null, SCR <= 0.2"
echo "  - No progression toward cat docs/note.txt"
echo ""
echo "Rationale:"
echo "  e1 was trained with exit_code=2 task-completion labels but still"
echo "  oscillated between ls and ls data. e2 retrains the same data with"
echo "  the same labels — if behavior is identical, the problem is not"
echo "  insufficient training but insufficient pragmatic reward signal"
echo "  in EFE (WATCHDOG C12 variant)."
echo "======================================================================"

exit $EXIT_CODE
