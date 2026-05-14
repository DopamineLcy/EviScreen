#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/constants.sh"

PYTHON_BIN="${PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
DATA_ROOT="${DATA_ROOT:-${EVISCREEN_STAGE0_OUTPUT_ROOT}}"
EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/reproduce_directly/eval}"
OUTPUT_JSON="${OUTPUT_JSON:-${EVAL_DIR}/category_mean_metrics.json}"

if [[ "$#" -gt 0 ]]; then
  "${PYTHON_BIN}" -B "${SCRIPT_DIR}/evaluate_category_metrics.py" "$@"
else
  "${PYTHON_BIN}" -B "${SCRIPT_DIR}/evaluate_category_metrics.py" \
    --data-root "${DATA_ROOT}" \
    --eval-dir "${EVAL_DIR}" \
    --output-json "${OUTPUT_JSON}"
fi
