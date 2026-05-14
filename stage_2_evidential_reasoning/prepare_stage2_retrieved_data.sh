#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

PYTHON_BIN="${PREPARE_PYTHON_BIN:-${EVISCREEN_BANK_PYTHON_BIN}}"
DATA_ROOT="${DATA_ROOT:-${EVISCREEN_STAGE0_OUTPUT_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${EVISCREEN_RAW_ROOT}}"
CHECKPOINT="${BACKBONE_CHECKPOINT:-${CHECKPOINT:-${EVISCREEN_CHECKPOINT}}}"
MODEL_ROOT="${MODEL_ROOT:-${EVISCREEN_STAGE1_OUTPUT_DIR}}"
NORMAL_BANK="${NORMAL_BANK:-${MODEL_ROOT}/normal}"
PATHOLOGICAL_BANK="${PATHOLOGICAL_BANK:-${MODEL_ROOT}/pathological}"
RETRIEVED_DATA_DIR="${RETRIEVED_DATA_DIR:-${EVISCREEN_STAGE2_RETRIEVED_DATA_DIR}}"
SPLITS="${SPLITS:-fundus_remain_5000 fundus_val JSIEC_original RIADD_original}"
FAISS_ON_GPU="${FAISS_ON_GPU:-1}"

[[ -d "${DATA_ROOT}/fundus" ]] || { echo "Missing fundus CSV root: ${DATA_ROOT}/fundus" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing backbone checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${NORMAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing normal bank: ${NORMAL_BANK}" >&2; exit 1; }
[[ -f "${PATHOLOGICAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing pathological bank: ${PATHOLOGICAL_BANK}" >&2; exit 1; }

read -r -a SPLIT_ARGS <<< "${SPLITS}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --raw-root "${RAW_ROOT}"
  --normal-bank "${NORMAL_BANK}"
  --pathological-bank "${PATHOLOGICAL_BANK}"
  --backbone-checkpoint "${CHECKPOINT}"
  --output-dir "${RETRIEVED_DATA_DIR}"
  --splits "${SPLIT_ARGS[@]}"
  --nn "${NN:-16}"
  --gpu "${GPU:-0}"
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-8}"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${FAISS_ON_GPU}" != "0" ]]; then
  ARGS+=(--faiss-on-gpu)
fi

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/prepare_stage2_retrieved_data.py" "${ARGS[@]}"

echo "Retrieved data: ${RETRIEVED_DATA_DIR}"
