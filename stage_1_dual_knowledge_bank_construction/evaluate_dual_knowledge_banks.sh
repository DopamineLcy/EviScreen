#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

PYTHON_BIN="${PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
DATA_ROOT="${DATA_ROOT:-${EVISCREEN_STAGE0_OUTPUT_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${EVISCREEN_RAW_ROOT}}"
CHECKPOINT="${CHECKPOINT:-${EVISCREEN_CHECKPOINT}}"
BANK_ROOT="${BANK_ROOT:-${EVISCREEN_STAGE1_OUTPUT_DIR}}"
NORMAL_BANK="${NORMAL_BANK:-${BANK_ROOT}/normal}"
PATHOLOGICAL_BANK="${PATHOLOGICAL_BANK:-${BANK_ROOT}/pathological}"
OUTPUT_DIR="${OUTPUT_DIR:-${BANK_ROOT}/eval}"
FAISS_ON_GPU="${FAISS_ON_GPU:-1}"

[[ -d "${DATA_ROOT}/fundus" ]] || { echo "Missing fundus CSV root: ${DATA_ROOT}/fundus" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${NORMAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing normal bank: ${NORMAL_BANK}" >&2; exit 1; }
[[ -f "${PATHOLOGICAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing pathological bank: ${PATHOLOGICAL_BANK}" >&2; exit 1; }

ARGS=(
  --data-root "${DATA_ROOT}"
  --raw-root "${RAW_ROOT}"
  --normal-bank "${NORMAL_BANK}"
  --pathological-bank "${PATHOLOGICAL_BANK}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --test-sets JSIEC_original RIADD_original
  --nn "${NN:-16}"
  --gpu "${GPU:-0}"
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-8}"
)

if [[ "${FAISS_ON_GPU}" != "0" ]]; then
  ARGS+=(--faiss-on-gpu)
fi
if [[ -n "${MAX_TEST_SAMPLES:-}" ]]; then
  ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
fi

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/evaluate_dual_knowledge_banks.py" "${ARGS[@]}"

EVAL_DIR="${OUTPUT_DIR}" OUTPUT_JSON="${OUTPUT_DIR}/category_mean_metrics.json" \
bash "${SCRIPT_DIR}/evaluate_category_metrics.sh"

echo "Evaluation outputs: ${OUTPUT_DIR}"
