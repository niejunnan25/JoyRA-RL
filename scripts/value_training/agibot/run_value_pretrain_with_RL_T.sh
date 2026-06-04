#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_env.sh"
setup_value_training_distributed 29510 8
setup_value_training_project "${BASH_SOURCE[0]}"

CONFIG_YAML=${CONFIG_YAML:-examples/Suqian_agibot/train_files/starvla_value_function.yaml}

DATA_ROOT_DIR=${DATA_ROOT_DIR:-/mnt/workspace/datasets}
# 混合数据集名称由命令行传入: bash run_value_T.sh <DATA_MIX>
DATA_MIX="${1:?用法: $0 <DATA_MIX>}"

OUTPUTS_BASE="${PROJECT_ROOT}/outputs"
DATE_STR=$(date +%Y%m%d)
mkdir -p "${OUTPUTS_BASE}"
OUTPUT_DIR="${OUTPUTS_BASE}/value/${DATA_MIX}_${DATE_STR}"
RETURNS_CACHE_DIR="${OUTPUTS_BASE}/cache/${DATA_MIX}"
echo "DATA_MIX=${DATA_MIX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RETURNS_CACHE_DIR=${RETURNS_CACHE_DIR}"

# Return 归一化配置：按 task 最大长度归一化（论文风格），值域 [-1, 0]
NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false

# Bin range（在 normalize 模式下自动设置为 [-1,0]，这里留空即可）
#BIN_MIN=-1200.0
#BIN_MAX=0.0

EPOCHS=10
STEPS_PER_EPOCH=100000
BATCH_SIZE=8
LR=3e-5
NUM_WORKERS=4

# 时序降采样步长（帧间隔），例如 3 表示每 3 帧取一帧（与 examples/Suqian_agibot/train_files/run_value_agi_suqian_egodex_robocasa_robotwin_mix.sh 对齐）
FRAME_STRIDE=${FRAME_STRIDE:-3}

SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=20

# 训练集/测试集划分
TRAIN_SPLIT=0.9
EVAL_INTERVAL=1
# 验证集采样数量，加速验证
VAL_NUM_SAMPLES=2000
# 是否根据验证集 loss 保存最佳模型
SAVE_BEST=false

# ===== 构建 value 训练参数 =====
NORMALIZE_ARGS="--normalize_returns_per_task --normalize_use_big_negative_in_denom"
BIN_RANGE_ARGS=""

if [ "${NORMALIZE_RETURNS_PER_TASK}" = "true" ]; then
    NORMALIZE_ARGS="--normalize_returns_per_task --normalize_use_big_negative_in_denom"
    echo "Using per-task normalized returns mode WITH (H + big_negative) denom: returns in [-1.0, 0.0], bin_min=-1.0, bin_max=0.0."
elif [ "${NORMALIZE_RETURNS}" = "true" ]; then
    NORMALIZE_ARGS="--normalize_returns"
    echo "Using per-episode normalized returns mode: returns will be in [-1.0, 0.0] range, bin_min=-1.0, bin_max=0.0"
else
    if [ -n "${BIN_MIN:-}" ] && [ -n "${BIN_MAX:-}" ]; then
        BIN_RANGE_ARGS="--bin_min ${BIN_MIN} --bin_max ${BIN_MAX}"
        echo "Using fixed bin range: bin_min=${BIN_MIN}, bin_max=${BIN_MAX}"
    elif [ -n "${BIN_RANGE_JSON:-}" ] && [ -f "${BIN_RANGE_JSON}" ]; then
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