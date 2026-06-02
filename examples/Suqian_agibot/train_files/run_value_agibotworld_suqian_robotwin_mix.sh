RDMA_DEVICES=$(ls /sys/class/infiniband)
if [ -z "$RDMA_DEVICES" ]; then
  log "ERROR: No active RDMA devices found. Exiting script." >&2
  exit 1
fi

# 设置RDMA设备列表 (逗号分隔)
NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | grep mlx5_gdr_ | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA
echo "Detected RDMA devices: $NCCL_IB_HCA"

# 获取GID_INDEX
NCCL_IB_GID_INDEX=""
output=$(show_gids | grep v2)
# 遍历每一行
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  # 检查IPv4地址是否有值（不是空字符串且不是全0地址）
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    # 找到第一个匹配项后立即退出循环
    break
  fi
done <<<"$output"

export NCCL_SOCKET_IFNAME="eth0"

# 处理分布式训练需要使用到的相关环境变量
export NCCL_IB_HCA=${NCCL_IB_HCA}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
export NCCL_NET_GDR_LEVEL=2
# export NCCL_DEBUG_SUBSYS=ALL
# 加速
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

# ==== 多节点配置 ====
# 总节点数（根据实际情况修改）
NUM_NODES=${PET_NNODES}
# 当前节点索引（0-based，每个节点启动时需指定不同值）
NODE_RANK=${PET_NODE_RANK}
# 主节点IP地址（确保所有节点可访问）
MASTER_ADDR=${PET_MASTER_ADDR}  # 替换为实际主节点IP
# 主节点端口（保持与代码中一致）
MASTER_PORT=${PET_MASTER_PORT}
NUM_GPUS_PER_NODE=8

source /mnt/workspace/envs/conda3/bin/activate starVLA_1
cd /mnt/workspace/users/daiyixiang/JoyRA-RL


export PYTHONPATH=/mnt/workspace/users/daiyixiang/JoyRA-RL:${PYTHONPATH}

# 禁用 HuggingFace tokenizers 的并行处理警告（多进程环境下避免死锁）
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML=examples/Suqian_agibot/train_files/starvla_value_function.yaml

DATA_ROOT_DIR=/mnt/workspace/datasets
# 使用混合数据集：agibot-world + Suqian_agibot + robotwin_aloha_agilex_550_mix
DATA_MIX=agibotworld_suqian_robotwin_mix

# Return 归一化配置
# 1) NORMALIZE_RETURNS_PER_TASK=true: 按 task 的最大 episode 长度做归一化（论文版），值域在 [-1, 0]
# 2) 否则如果 NORMALIZE_RETURNS=true: 按 episode 长度做归一化（per-episode），值域在 [-1, 0]
# 两种模式都会自动设置 bin_min=-1.0, bin_max=0.0，不需要手动设置 bin 范围
NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false

# Bin range 配置（仅在 NORMALIZE_RETURNS=false 时生效，三种方式，按优先级）：
# BIN_MIN=-1200.0
# BIN_MAX=0.0


EPOCHS=50
STEPS_PER_EPOCH=20000
BATCH_SIZE=8
LR=3e-5
NUM_WORKERS=4

# 模型保存目录
OUTPUT_DIR=/mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_agibotworld_suqian_robotwin_mix

# Return 缓存目录（可选，如果设置将保存/加载计算好的 return 值）
# 设置为空字符串或注释掉则不启用缓存功能
RETURNS_CACHE_DIR=/mnt/workspace/users/daiyixiang/JoyRA-RL/returns_cache_agibotworld_suqian_robotwin_mix


SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=3

# 训练集/测试集划分配置
TRAIN_SPLIT=0.9
EVAL_INTERVAL=1
# 验证集采样数量（可选，如果设置则只评估前 N 个样本，加速验证过程）
VAL_NUM_SAMPLES=2000
# 是否根据测试集 loss 保存最佳模型（true/false）。默认 false（不保存）
SAVE_BEST=false

# ===== 启动分布式 value function 训练（torchrun） =====
# 构建参数
NORMALIZE_ARGS=""
BIN_RANGE_ARGS=""

if [ "${NORMALIZE_RETURNS_PER_TASK}" = "true" ]; then
    NORMALIZE_ARGS="--normalize_returns_per_task"
    echo "Using per-task normalized returns mode: returns will be in [-1.0, 0.0] range, bin_min=-1.0, bin_max=0.0 (per task denom)."
elif [ "${NORMALIZE_RETURNS}" = "true" ]; then
    NORMALIZE_ARGS="--normalize_returns"
    echo "Using per-episode normalized returns mode: returns will be in [-1.0, 0.0] range, bin_min=-1.0, bin_max=0.0"
else
    # 构建 bin range 参数（优先级：命令行参数 > JSON 文件 > 数据驱动）
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
