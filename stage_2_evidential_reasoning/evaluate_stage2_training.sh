#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

MODEL_ROOT="${MODEL_ROOT:-${EVISCREEN_STAGE1_OUTPUT_DIR}}"
RETRIEVED_DATA_DIR="${RETRIEVED_DATA_DIR:-${EVISCREEN_STAGE2_RETRIEVED_DATA_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVISCREEN_STAGE2_OUTPUT_DIR}/eval_stage2}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-${EVISCREEN_STAGE2_HEAD_OUTPUT_DIR}/pretrain_MODELRetrievingDistanceCatHead_BACKBONEresnet50_TEMP0.07_EP50_WM10_LR3.125e-06_BS1_MODALITYfundus_original_range_disdim1024/checkpoint-best_auroc.pth}"

[[ -f "${HEAD_CHECKPOINT}" ]] || { echo "Missing Stage 2 checkpoint: ${HEAD_CHECKPOINT}" >&2; exit 1; }

MODEL_ROOT="${MODEL_ROOT}" \
HEAD_CHECKPOINT="${HEAD_CHECKPOINT}" \
RETRIEVED_DATA_DIR="${RETRIEVED_DATA_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${REPO_DIR}/reproduce_directly/infer_stage2_direct_retrieval.sh"
