#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/eval}"
bash "${REPO_DIR}/evaluate_category_metrics.sh" "$@"
