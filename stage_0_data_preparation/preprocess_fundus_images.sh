#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

PYTHON_BIN="${PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
RAW_ROOT="${RAW_ROOT:-${EVISCREEN_RAW_ROOT}}"
NUM_WORKERS="${NUM_WORKERS:-8}"

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/preprocess_fundus_images.py" \
  --raw-root "${RAW_ROOT}" \
  --num-workers "${NUM_WORKERS}"

echo "Fundus image preprocessing completed under: ${RAW_ROOT}"
