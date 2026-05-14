#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

INFER_PYTHON_BIN="${INFER_PYTHON_BIN:-${EVISCREEN_REASONING_PYTHON_BIN}}"
METRICS_PYTHON_BIN="${METRICS_PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
DATA_ROOT="${DATA_ROOT:-${EVISCREEN_STAGE0_OUTPUT_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${EVISCREEN_RAW_ROOT}}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_DIR}/model_for_evaluation}"
NORMAL_BANK="${NORMAL_BANK:-${MODEL_ROOT}/normal}"
PATHOLOGICAL_BANK="${PATHOLOGICAL_BANK:-${MODEL_ROOT}/pathological}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-${MODEL_ROOT}/model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval_stage2_direct}"
RETRIEVED_DATA_DIR="${RETRIEVED_DATA_DIR:-${OUTPUT_DIR}/retrieved_data}"
TEST_SETS="${TEST_SETS:-JSIEC_original RIADD_original}"

[[ -d "${DATA_ROOT}/fundus" ]] || { echo "Missing fundus CSV root: ${DATA_ROOT}/fundus" >&2; exit 1; }
[[ -d "${RETRIEVED_DATA_DIR}" ]] || { echo "Missing retrieved data: ${RETRIEVED_DATA_DIR}" >&2; exit 1; }
[[ -f "${HEAD_CHECKPOINT}" ]] || { echo "Missing Stage 2 head checkpoint: ${HEAD_CHECKPOINT}" >&2; exit 1; }
[[ -f "${NORMAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing normal bank: ${NORMAL_BANK}" >&2; exit 1; }
[[ -f "${PATHOLOGICAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing pathological bank: ${PATHOLOGICAL_BANK}" >&2; exit 1; }

read -r -a TEST_SET_ARGS <<< "${TEST_SETS}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --raw-root "${RAW_ROOT}"
  --normal-bank "${NORMAL_BANK}"
  --pathological-bank "${PATHOLOGICAL_BANK}"
  --head-checkpoint "${HEAD_CHECKPOINT}"
  --retrieved-data-root "${RETRIEVED_DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --test-sets "${TEST_SET_ARGS[@]}"
  --gpu "${GPU:-0}"
  --batch-size "${BATCH_SIZE:-32}"
  --num-workers "${NUM_WORKERS:-8}"
)

if [[ -n "${MAX_TEST_SAMPLES:-}" ]]; then
  ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
fi

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "${INFER_PYTHON_BIN}" -B "${SCRIPT_DIR}/evaluate_stage2_direct_retrieval.py" "${ARGS[@]}"

"${METRICS_PYTHON_BIN}" -B "${REPO_DIR}/evaluate_category_metrics.py" \
  --data-root "${DATA_ROOT}" \
  --eval-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/category_mean_metrics.json"

echo "Stage 2 outputs: ${OUTPUT_DIR}"
