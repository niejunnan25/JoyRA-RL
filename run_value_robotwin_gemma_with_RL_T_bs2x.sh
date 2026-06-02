#!/usr/bin/env bash
# GemmaValue bs2x value-function training for RobotWin.
# Usage:
#   bash run_value_robotwin_gemma_with_RL_T_bs2x.sh
#   bash run_value_robotwin_gemma_with_RL_T_bs2x.sh <DATA_MIX>

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
output=$(show_gids 2>/dev/null | grep v2 || true)
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    break
  fi
done <<<"$output"
if [ -z "${NCCL_IB_GID_INDEX}" ]; then
  echo "WARN: show_gids did not report a usable v2 GID. Continuing with NCCL_IB_GID_INDEX unset/empty."
fi

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
NUM_NODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${PET_MASTER_PORT:-29520}
NUM_GPUS_PER_NODE=8

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
source /mnt/workspace/envs/conda3/bin/activate starVLA_1

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML=examples/Suqian_agibot/train_files/starvla_gemma_value_function.yaml

DATA_ROOT_DIR=/mnt/workspace/datasets
# 混合数据集名称由命令行传入（见 starVLA/dataloader/gr00t_lerobot/mixtures.py）
DATA_MIX="${1:-robotwin_orig_plus_offline_v2}"

OUTPUTS_BASE="${PROJECT_ROOT}/outputs"
DATE_STR=$(date +%Y%m%d)
mkdir -p "${OUTPUTS_BASE}"
RUN_SUFFIX="gemma270_siglip2_bs2x_neg600_hplusneg_$(date +%H%M%S)"
OUTPUT_NAME="${DATA_MIX}_${DATE_STR}"
if [ -n "${RUN_SUFFIX}" ]; then
  OUTPUT_NAME="${OUTPUT_NAME}_${RUN_SUFFIX}"
fi
OUTPUT_DIR="${OUTPUTS_BASE}/value/${OUTPUT_NAME}"
RETURNS_CACHE_DIR="${OUTPUTS_BASE}/cache/${DATA_MIX}"
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "DATA_MIX=${DATA_MIX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RETURNS_CACHE_DIR=${RETURNS_CACHE_DIR}"

# ==== 自动检测实际可见 GPU 数量，避免 invalid device ordinal ====
GPU_COUNT_PY=$(python - <<'PY'
import os, sys
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
if [[ -z "${GPU_COUNT_PY}" || "${GPU_COUNT_PY}" -lt 1 ]]; then
  # 尝试用 nvidia-smi 兜底
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT_PY=$(nvidia-smi -L 2>/dev/null | wc -l | xargs)
  else
    GPU_COUNT_PY=0
  fi
fi

# 若用户未显式指定 NUM_GPUS_PER_NODE，则用检测到的数量
if [[ -z "${PET_NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${GPU_COUNT_PY}" && "${GPU_COUNT_PY}" -gt 0 ]]; then
    DETECTED_GPUS=${GPU_COUNT_PY}
    if [[ -n "${NUM_GPUS_PER_NODE}" && "${NUM_GPUS_PER_NODE}" -gt "${DETECTED_GPUS}" ]]; then
      echo "WARN: Requested NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} > detected ${DETECTED_GPUS}. Using ${DETECTED_GPUS}."
      NUM_GPUS_PER_NODE=${DETECTED_GPUS}
    elif [[ -z "${NUM_GPUS_PER_NODE}" || "${NUM_GPUS_PER_NODE}" -lt 1 ]]; then
      NUM_GPUS_PER_NODE=${DETECTED_GPUS}
    fi
  else
    # 无 GPU 或未能检测到，退化到 1 以避免越界（仍可能因无 GPU 而失败）
    echo "WARN: No GPUs detected by torch/nvidia-smi. Falling back to NUM_GPUS_PER_NODE=1."
    NUM_GPUS_PER_NODE=1
  fi
fi

# Return 归一化配置：按 task 最大长度归一化（论文风格），值域 [-1, 0]
NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false
# π0.6 要求 C_fail 足够大，使失败 episode value 明显偏低。
# RobotWin 常见 horizon 约 1200 steps，因此默认使用同量级失败惩罚。
BIG_NEGATIVE=600
echo "BIG_NEGATIVE=${BIG_NEGATIVE}"

# Bin range（在 normalize 模式下自动设置为 [-1,0]，这里留空即可）
#BIN_MIN=-1200.0
#BIN_MAX=0.0

# Batch size is per GPU/rank under DDP. With 8 GPUs, BATCH_SIZE=32 means global batch size 256.
BATCH_SIZE=32
# Step-based value training, aligned with StarVLA trainer.max_train_steps.
MAX_TRAIN_STEPS=50000
LR=3e-5
NUM_WORKERS=8
PREFETCH_FACTOR=4
PIN_MEMORY=true
PERSISTENT_WORKERS=true
# Disable periodic empty_cache; it slows this training loop noticeably.
EMPTY_CACHE_STEPS=0

SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=20

# 训练集/测试集划分
TRAIN_SPLIT=0.9
EVAL_STEPS=5000
# 验证集采样数量，加速验证
VAL_NUM_SAMPLES=2000
# 是否根据验证集 loss 保存最佳模型
SAVE_BEST=true
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
  --nnodes=${NUM_NODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${NUM_GPUS_PER_NODE:-8} \
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
