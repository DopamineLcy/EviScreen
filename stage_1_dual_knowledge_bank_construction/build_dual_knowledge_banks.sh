#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

PYTHON_BIN="${PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
DATA_ROOT="${DATA_ROOT:-${EVISCREEN_STAGE0_OUTPUT_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${EVISCREEN_RAW_ROOT}}"
CHECKPOINT="${CHECKPOINT:-${EVISCREEN_CHECKPOINT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVISCREEN_STAGE1_OUTPUT_DIR}}"
FAISS_ON_GPU="${FAISS_ON_GPU:-1}"

[[ -d "${DATA_ROOT}/fundus" ]] || { echo "Missing fundus CSV root: ${DATA_ROOT}/fundus" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }

ARGS=(
  --data-root "${DATA_ROOT}"
  --raw-root "${RAW_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --gpu "${GPU:-0}"
  --seed "${SEED:-0}"
  --batch-size "${BATCH_SIZE:-4}"
  --num-workers "${NUM_WORKERS:-8}"
  --chunk-size "${CHUNK_SIZE:-5000}"
  --train-scale "${TRAIN_SCALE:-5000}"
  --coreset-percentage "${CORESET_PERCENTAGE:-0.1}"
)

if [[ "${FAISS_ON_GPU}" != "0" ]]; then
  ARGS+=(--faiss-on-gpu)
fi

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/train_dual_knowledge_banks.py" "${ARGS[@]}"

echo "Dual knowledge banks: ${OUTPUT_DIR}"
