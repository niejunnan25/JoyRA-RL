#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Success-heavy block-ranking value training:
#   blocks_ranking_rgb_success80_failure20:  1,896,006 frames
#   blocks_ranking_size_success80_failure20: 2,640,151 frames
#   total: 4,536,157 frames, success/failure ~= 80/20 by frame count
#
# 8 GPUs x batch_size 64 = global batch size 512.
# 30 traditional epochs over train_split=0.9:
#   4,536,157 * 0.9 * 30 / 512 ~= 239,212 optimizer steps.
export DATA_ROOT_DIR=${DATA_ROOT_DIR:-/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot}
export DATA_MIX=${DATA_MIX:-robotwin_pi05_blocks_ranking_success80_failure20}
export BIG_NEGATIVE=${BIG_NEGATIVE:-600}
export BATCH_SIZE=${BATCH_SIZE:-64}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-239212}
export NUM_BINS=${NUM_BINS:-201}
export FRAME_STRIDE=${FRAME_STRIDE:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT:-29545}
export SAVE_STEPS=${SAVE_STEPS:-2500}
export EVAL_STEPS=${EVAL_STEPS:-2500}
export SAVE_BEST=${SAVE_BEST:-true}
export RUN_SUFFIX=${RUN_SUFFIX:-"gemma270_siglip2_robotwin_pi05_blocks_ranking_success80_failure20_30epochs_bs512_neg${BIG_NEGATIVE}_$(date +%H%M%S)"}

exec bash "${PROJECT_ROOT}/scripts/value_training/robotwin/run_value_robotwin_gemma_with_RL_T_bs2x.sh"
