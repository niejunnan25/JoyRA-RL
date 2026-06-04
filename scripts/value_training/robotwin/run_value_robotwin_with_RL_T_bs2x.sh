#!/usr/bin/env bash
# 值函数训练（与 run_value_T.sh 相同：DATA_MIX 外参 + outputs 目录）。用法:
#   bash run_value_robotwin_with_RL_T.sh <DATA_MIX>
# 产出: outputs/value_<DATA_MIX>_YYYYMMDD/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_env.sh"
setup_value_training_distributed 29510 8
setup_value_training_project "${BASH_SOURCE[0]}"

CONFIG_YAML=${CONFIG_YAML:-examples/Suqian_agibot/train_files/starvla_value_function.yaml}

DATA_ROOT_DIR=${DATA_ROOT_DIR:-/mnt/workspace/datasets}
# 混合数据集名称由命令行传入（见 starVLA/dataloader/gr00t_lerobot/mixtures.py）
DATA_MIX="${1:?用法: $0 <DATA_MIX>}"

OUTPUTS_BASE="${PROJECT_ROOT}/outputs"
DATE_STR=$(date +%Y%m%d)
mkdir -p "${OUTPUTS_BASE}"
RUN_SUFFIX=${RUN_SUFFIX:-bs2x_hplusneg}
OUTPUT_DIR="${OUTPUTS_BASE}/value/${DATA_MIX}_${DATE_STR}_${RUN_SUFFIX}"
RETURNS_CACHE_DIR="${OUTPUTS_BASE}/cache/${DATA_MIX}"
echo "DATA_MIX=${DATA_MIX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RETURNS_CACHE_DIR=${RETURNS_CACHE_DIR}"

# Return 归一化配置：按 task 最大长度归一化（论文风格），值域 [-1, 0]
NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false
# π0.6-style failed terminal penalty. This run uses H + C_fail normalization,
# and keeps C_fail=600 for the current RobotWin comparison.
BIG_NEGATIVE=${BIG_NEGATIVE:-600}
echo "BIG_NEGATIVE=${BIG_NEGATIVE}"

# Bin range（在 normalize 模式下自动设置为 [-1,0]，这里留空即可）
#BIN_MIN=-1200.0
#BIN_MAX=0.0

# Batch size is per GPU/rank under DDP. With 8 GPUs, BATCH_SIZE=32 means global batch size 256.
BATCH_SIZE=${BATCH_SIZE:-32}
# Step-based value training, aligned with StarVLA trainer.max_train_steps.
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}
LR=${LR:-3e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PIN_MEMORY=${PIN_MEMORY:-true}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
# Disable periodic empty_cache by default; set e.g. EMPTY_CACHE_STEPS=100 if fragmentation/OOM appears.
EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS:-0}

# Save and evaluate every 2500 optimizer steps.
SAVE_STEPS=${SAVE_STEPS:-2500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-20}

# 训练集/测试集划分
TRAIN_SPLIT=0.9
EVAL_STEPS=${EVAL_STEPS:-2500}
# 验证集采样数量，加速验证
VAL_NUM_SAMPLES=${VAL_NUM_SAMPLES:-2000}
# 是否根据验证集 loss 保存最佳模型
SAVE_BEST=${SAVE_BEST:-true}
SAVE_BEST_ARGS=""
if [ "${SAVE_BEST}" = "true" ]; then
    SAVE_BEST_ARGS="--save_best"
fi

echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "BATCH_SIZE_PER_GPU=${BATCH_SIZE}"
echo "GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS_PER_NODE * NUM_NODES))"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "PIN_MEMORY=${PIN_MEMORY}"
echo "PERSISTENT_WORKERS=${PERSISTENT_WORKERS}"
echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
echo "EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS}"

PIN_MEMORY_ARGS=""
if [ "${PIN_MEMORY}" = "true" ]; then
    PIN_MEMORY_ARGS="--pin_memory"
fi

PERSISTENT_WORKERS_ARGS=""
if [ "${PERSISTENT_WORKERS}" = "true" ]; then
    PERSISTENT_WORKERS_ARGS="--persistent_workers"
fi

# ===== 构建 value 训练参数 =====
NORMALIZE_ARGS=""
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
  --big_negative ${BIG_NEGATIVE} \
  ${NORMALIZE_ARGS} \
  ${BIN_RANGE_ARGS} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_workers ${NUM_WORKERS} \
  ${PIN_MEMORY_ARGS} \
  ${PERSISTENT_WORKERS_ARGS} \
  ${PREFETCH_FACTOR:+--prefetch_factor ${PREFETCH_FACTOR}} \
  --empty_cache_steps ${EMPTY_CACHE_STEPS} \
  --output_dir "${OUTPUT_DIR}" \
  ${RETURNS_CACHE_DIR:+--returns_cache_dir "${RETURNS_CACHE_DIR}"} \
  --train_split ${TRAIN_SPLIT} \
  --eval_steps ${EVAL_STEPS} \
  ${VAL_NUM_SAMPLES:+--val_num_samples ${VAL_NUM_SAMPLES}} \
  ${SAVE_BEST_ARGS} \
  ${SAVE_STEPS:+--save_steps ${SAVE_STEPS}} \
  ${SAVE_TOTAL_LIMIT:+--save_total_limit ${SAVE_TOTAL_LIMIT}}

python examples/Suqian_agibot/train_files/plot_training_losses.py \
  --loss_file "${OUTPUT_DIR}/training_losses.json" \
  --output "${OUTPUT_DIR}/training_losses_curve.png"
