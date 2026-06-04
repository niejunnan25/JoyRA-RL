#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_SUITE=spatial
export BIG_NEGATIVE=${BIG_NEGATIVE:-110}
exec bash "${SCRIPT_DIR}/run_value_libero_gemma_10epochs.sh"
