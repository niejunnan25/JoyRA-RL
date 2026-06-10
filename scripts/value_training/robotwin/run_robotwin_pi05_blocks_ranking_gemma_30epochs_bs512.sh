#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Two PI0.5 qpos block-ranking tasks converted to standalone LeRobot datasets:
#   blocks_ranking_rgb:  481,723 frames, 184 success / 316 failure
#   blocks_ranking_size: 559,157 frames,  60 success / 440 failure
#
# 8 GPUs x batch_size 64 = global batch size 512.
# 30 traditional epochs over train_split=0.9:
#   (481,723 + 559,157) * 0.9 * 30 / 512 ~= 54,891 optimizer steps.
export DATA_ROOT_DIR=${DATA_ROOT_DIR:-/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot}
export DATA_MIX=${DATA_MIX:-robotwin_pi05_blocks_ranking_all}
export BIG_NEGATIVE=${BIG_NEGATIVE:-600}
export BATCH_SIZE=${BATCH_SIZE:-64}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-54891}
export FRAME_STRIDE=${FRAME_STRIDE:-1}
export MASTER_PORT=${MASTER_PORT:-29541}
export SAVE_STEPS=${SAVE_STEPS:-2500}
export EVAL_STEPS=${EVAL_STEPS:-2500}
export SAVE_BEST=${SAVE_BEST:-true}
export RUN_SUFFIX=${RUN_SUFFIX:-"gemma270_siglip2_robotwin_pi05_blocks_ranking_all_30epochs_bs512_neg${BIG_NEGATIVE}_stride${FRAME_STRIDE}_$(date +%H%M%S)"}

exec bash "${PROJECT_ROOT}/scripts/value_training/robotwin/run_value_robotwin_gemma_with_RL_T_bs2x.sh"
