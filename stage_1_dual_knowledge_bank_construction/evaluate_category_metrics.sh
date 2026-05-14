#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/constants.sh"

export EVAL_DIR="${EVAL_DIR:-${EVISCREEN_STAGE1_OUTPUT_DIR}/eval}"
bash "${REPO_DIR}/evaluate_category_metrics.sh" "$@"
