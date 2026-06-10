#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/mnt/workspace/users/niejunnan/workspace/JoyRA-RL
cd "${PROJECT_ROOT}"
export CONFIG_YAML=${CONFIG_YAML:-examples/Suqian_agibot/train_files/starvla_gemma_value_function_51bins.yaml}
export NUM_BINS=${NUM_BINS:-51}
export TARGET_EPOCHS=${TARGET_EPOCHS:-30}
export BATCH_SIZE=${BATCH_SIZE:-128}
export RUN_SUFFIX=${RUN_SUFFIX:-"gemma270_siglip2_libero_all_30epochs_neg250_51bins_$(date +%H%M%S)"}
exec bash "${PROJECT_ROOT}/scripts/value_training/libero/run_value_libero_all_gemma_10epochs.sh"
