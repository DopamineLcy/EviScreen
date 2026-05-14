#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

PYTHON_BIN="${TRAIN_PYTHON_BIN:-${EVISCREEN_REASONING_PYTHON_BIN}}"
MODEL_ROOT="${MODEL_ROOT:-${EVISCREEN_STAGE1_OUTPUT_DIR}}"
NORMAL_BANK="${NORMAL_BANK:-${MODEL_ROOT}/normal}"
PATHOLOGICAL_BANK="${PATHOLOGICAL_BANK:-${MODEL_ROOT}/pathological}"
RETRIEVED_DATA_DIR="${RETRIEVED_DATA_DIR:-${EVISCREEN_STAGE2_RETRIEVED_DATA_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVISCREEN_STAGE2_HEAD_OUTPUT_DIR}}"
TEST_SETS="${TEST_SETS:-JSIEC_original RIADD_original}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"

[[ -d "${RETRIEVED_DATA_DIR}" ]] || { echo "Missing retrieved data: ${RETRIEVED_DATA_DIR}" >&2; exit 1; }
[[ -f "${NORMAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing normal bank: ${NORMAL_BANK}" >&2; exit 1; }
[[ -f "${PATHOLOGICAL_BANK}/nnscorer_search_index.faiss" ]] || { echo "Missing pathological bank: ${PATHOLOGICAL_BANK}" >&2; exit 1; }

read -r -a TEST_SET_ARGS <<< "${TEST_SETS}"

ARGS=(
  --retrieved-data-root "${RETRIEVED_DATA_DIR}"
  --normal-bank "${NORMAL_BANK}"
  --pathological-bank "${PATHOLOGICAL_BANK}"
  --output-dir "${OUTPUT_DIR}"
  --model RetrievingDistanceCatHead
  --modality fundus
  --num-patches 256
  --epochs "${EPOCHS:-50}"
  --warmup-epochs "${WARMUP_EPOCHS:-10}"
  --blr "${BLR:-8e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-1}"
  --accum-iter "${ACCUM_ITER:-1}"
  --save-freq "${SAVE_FREQ:-1}"
  --note "${NOTE:-original_range_disdim1024}"
  --script "${SCRIPT_DIR}/train_stage2_evidential_reasoning.sh"
  --gpu "${LOCAL_GPU:-0}"
  --test-sets "${TEST_SET_ARGS[@]}"
)

if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi
if [[ "${EVAL:-0}" != "0" ]]; then
  ARGS+=(--eval)
fi
if [[ "${FROM_BEGIN:-0}" != "0" ]]; then
  ARGS+=(--from-begin)
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
  ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi
if [[ -n "${MAX_TEST_SAMPLES:-}" ]]; then
  ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
fi

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
"${PYTHON_BIN}" -B "${SCRIPT_DIR}/train_stage2_evidential_reasoning.py" "${ARGS[@]}"

echo "Stage 2 training outputs: ${OUTPUT_DIR}"
