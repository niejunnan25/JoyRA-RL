#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# 25k pi0.5 demo_clean qpos rollouts converted to LeRobot.
# 8 GPUs x batch_size 128 = global batch size 1024.
# 10 traditional epochs over train_split=0.9:
#   9,650,276 * 0.9 * 10 / 1024 ~= 84,817 optimizer steps.
export DATA_ROOT_DIR=${DATA_ROOT_DIR:-/mnt/workspace/users/niejunnan/datasets}
export DATA_MIX=${DATA_MIX:-robotwin_pi05_demo_clean_qpos_rollout}
export BIG_NEGATIVE=${BIG_NEGATIVE:-600}
export BATCH_SIZE=${BATCH_SIZE:-128}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-84817}
export FRAME_STRIDE=${FRAME_STRIDE:-1}
export MASTER_PORT=${MASTER_PORT:-29540}
export RUN_SUFFIX=${RUN_SUFFIX:-"gemma270_siglip2_robotwin_pi05_qpos_10epochs_bs1024_neg${BIG_NEGATIVE}_stride${FRAME_STRIDE}_$(date +%H%M%S)"}

exec bash "${PROJECT_ROOT}/scripts/value_training/robotwin/run_value_robotwin_gemma_with_RL_T_bs2x.sh"
