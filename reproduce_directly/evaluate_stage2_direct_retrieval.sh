#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/prepare_stage2_retrieved_data.sh"
bash "${SCRIPT_DIR}/infer_stage2_direct_retrieval.sh"

echo "Direct Stage 2 evidential reasoning completed."
