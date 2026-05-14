#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

PYTHON_BIN="${PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
RAW_ROOT="${RAW_ROOT:-${EVISCREEN_RAW_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVISCREEN_STAGE0_OUTPUT_ROOT}}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/prepare_fundus_data.py" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --skip-preprocess \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}"

echo "Fundus CSV files generated under: ${OUTPUT_ROOT}/fundus"
echo "Use this path as stage 1 --data-root: ${OUTPUT_ROOT}"
