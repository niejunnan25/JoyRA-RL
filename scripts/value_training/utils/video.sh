#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_env.sh"
setup_value_training_project "${BASH_SOURCE[0]}"

python scripts/scan_lerobot_videos_decord.py \
  --data_root_dir "${DATA_ROOT_DIR:-/mnt/workspace/datasets}" \
  --data_mix "${DATA_MIX:-robotwin_orig_plus_offline_v2}" \
  --output_json "${OUTPUT_JSON:-/tmp/bad_videos.json}"
