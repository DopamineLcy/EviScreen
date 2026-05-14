#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

MODEL_ROOT="${MODEL_ROOT:-${REPO_DIR}/model_for_evaluation}"

BANK_ROOT="${MODEL_ROOT}" \
NORMAL_BANK="${NORMAL_BANK:-${MODEL_ROOT}/normal}" \
PATHOLOGICAL_BANK="${PATHOLOGICAL_BANK:-${MODEL_ROOT}/pathological}" \
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval}" \
bash "${REPO_DIR}/stage_1_dual_knowledge_bank_construction/evaluate_dual_knowledge_banks.sh"
