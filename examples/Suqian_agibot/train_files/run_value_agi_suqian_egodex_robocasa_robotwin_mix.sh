#!/usr/bin/env bash

set -e

# ===== 基础环境与分布式配置（按需修改）=====
NUM_NODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${PET_MASTER_PORT:-29510}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

source /mnt/workspace/envs/conda3/bin/activate starVLA_1
cd /mnt/workspace/users/daiyixiang/JoyRA-RL

export PYTHONPATH=/mnt/workspace/users/daiyixiang/JoyRA-RL:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML=examples/Suqian_agibot/train_files/starvla_value_function.yaml

DATA_ROOT_DIR=/mnt/workspace/datasets
# 使用你要求的五个数据集的混合
DATA_MIX=agi_suqian_egodex_robocasa_robotwin_mix

# ===== Return 归一化配置 =====
NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false

# ===== 训练超参数 =====
EPOCHS=10
STEPS_PER_EPOCH=100000
BATCH_SIZE=8
LR=3e-5
NUM_WORKERS=4

# 时序降采样步长（帧间隔），例如 3 表示每 3 帧取一帧
FRAME_STRIDE=${FRAME_STRIDE:-3}

# 输出与缓存
OUTPUT_DIR=/mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_agi_suqian_egodex_robocasa_robotwin_mix
RETURNS_CACHE_DIR=/mnt/workspace/users/daiyixiang/JoyRA-RL/returns_cache_agi_suqian_egodex_robocasa_robotwin_mix

SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=20

TRAIN_SPLIT=0.9
EVAL_INTERVAL=1
VAL_NUM_SAMPLES=2000
SAVE_BEST=false

# ===== 构建归一化参数 =====
NORMALIZE_ARGS=""
BIN_RANGE_ARGS=""

if [ "${NORMALIZE_RETURNS_PER_TASK}" = "true" ]; then
  NORMALIZE_ARGS="--normalize_returns_per_task --normalize_use_big_negative_in_denom"
  echo "Using per-task normalized returns mode WITH (H + big_negative) denom: returns in [-1.0, 0.0], bin_min=-1.0, bin_max=0.0."
elif [ "${NORMALIZE_RETURNS}" = "true" ]; then
  NORMALIZE_ARGS="--normalize_returns"
  echo "Using per-episode normalized returns mode: returns will be in [-1.0, 0.0] range, bin_min=-1.0, bin_max=0.0"
else
  if [ -n "${BIN_MIN}" ] && [ -n "${BIN_MAX}" ]; then
    BIN_RANGE_ARGS="--bin_min ${BIN_MIN} --bin_max ${BIN_MAX}"
    echo "Using fixed bin range: bin_min=${BIN_MIN}, bin_max=${BIN_MAX}"
  elif [ -n "${BIN_RANGE_JSON}" ] && [ -f "${BIN_RANGE_JSON}" ]; then
    BIN_RANGE_ARGS="--bin_range_json ${BIN_RANGE_JSON}"
    echo "Using fixed bin range from: ${BIN_RANGE_JSON}"
  else
    echo "Warning: No bin range specified. Using data-driven mode (sampling)."
    echo "  Set BIN_MIN/BIN_MAX or run 'bash examples/Suqian_agibot/train_files/compute_bin_range.sh' first."
  fi
fi

echo "Using frame_stride=${FRAME_STRIDE} for temporal downsampling."

env IS_TORCHRUN=1 torchrun \
  --nnodes=${NUM_NODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  starVLA/training/train_value.py \
  --config_yaml "${CONFIG_YAML}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --data_mix "${DATA_MIX}" \
  ${NORMALIZE_ARGS} \
  ${BIN_RANGE_ARGS} \
  --epochs ${EPOCHS} \
  ${STEPS_PER_EPOCH:+--steps_per_epoch ${STEPS_PER_EPOCH}} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_workers ${NUM_WORKERS} \
  --output_dir "${OUTPUT_DIR}" \
  ${RETURNS_CACHE_DIR:+--returns_cache_dir "${RETURNS_CACHE_DIR}"} \
  --train_split ${TRAIN_SPLIT} \
  --eval_interval ${EVAL_INTERVAL} \
  ${VAL_NUM_SAMPLES:+--val_num_samples ${VAL_NUM_SAMPLES}} \
  ${SAVE_BEST:+--save_best} \
  ${SAVE_STEPS:+--save_steps ${SAVE_STEPS}} \
  ${SAVE_TOTAL_LIMIT:+--save_total_limit ${SAVE_TOTAL_LIMIT}} \
  --frame_stride ${FRAME_STRIDE}

