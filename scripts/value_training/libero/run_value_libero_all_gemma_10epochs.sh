#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_SUITE=all
export BIG_NEGATIVE=${BIG_NEGATIVE:-250}
exec bash "${SCRIPT_DIR}/run_value_libero_gemma_10epochs.sh"
