#!/usr/bin/env bash

RDMA_DEVICES=$(ls /sys/class/infiniband)
if [ -z "$RDMA_DEVICES" ]; then
  echo "ERROR: No active RDMA devices found. Exiting script." >&2
  exit 1
fi

# 设置RDMA设备列表 (逗号分隔)
NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | grep mlx5_gdr_ | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA
echo "Detected RDMA devices: $NCCL_IB_HCA"

# 获取GID_INDEX
NCCL_IB_GID_INDEX=""
output=$(show_gids | grep v2)
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    break
  fi
done <<<"$output"

export NCCL_SOCKET_IFNAME="eth0"

# 分布式训练相关环境变量
export NCCL_IB_HCA=${NCCL_IB_HCA}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# ==== 多节点配置 ====
NUM_NODES=${PET_NNODES}
NODE_RANK=${PET_NODE_RANK}
MASTER_ADDR=${PET_MASTER_ADDR}
MASTER_PORT=${PET_MASTER_PORT}
NUM_GPUS_PER_NODE=8

source /mnt/workspace/envs/conda3/bin/activate starVLA_1
cd /mnt/workspace/users/daiyixiang/JoyRA-RL

export PYTHONPATH=/mnt/workspace/users/daiyixiang/JoyRA-RL:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML=examples/Suqian_agibot/train_files/starvla_value_function.yaml

DATA_ROOT_DIR=/mnt/workspace/datasets
# 只使用 AgibotWorld-Beta-LeRobot 数据集
# 对应 mixtures.py 里的 "agi_beta"
DATA_MIX=agi_beta

# Return 归一化配置：按 task 最大长度归一化（论文风格），值域 [-1, 0]
NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false

# Bin range（在 normalize 模式下自动设置为 [-1,0]，这里留空即可）
#BIN_MIN=-1200.0
#BIN_MAX=0.0

EPOCHS=20
# STEPS_PER_EPOCH=20000
BATCH_SIZE=8
LR=3e-5
NUM_WORKERS=4

# 模型保存目录（agibotworld 专用）
OUTPUT_DIR=/mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_agibotworld_only

# Return 缓存目录
RETURNS_CACHE_DIR=/mnt/workspace/users/daiyixiang/JoyRA-RL/returns_cache_agibotworld_only

SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=3

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
    # 这里同时打开 per-task 归一化 和 H+big_negative 作为分母的开关
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

env IS_TORCHRUN=1 torchrun \
  --nnodes=${PET_NNODES:-1} \
  --node_rank=${PET_NODE_RANK:-0} \
  --master_addr=${PET_MASTER_ADDR:-127.0.0.1} \
  --master_port=${PET_MASTER_PORT:-29510} \
  --nproc_per_node=${NUM_GPUS_PER_NODE:-8} \
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
  ${SAVE_TOTAL_LIMIT:+--save_total_limit ${SAVE_TOTAL_LIMIT}}