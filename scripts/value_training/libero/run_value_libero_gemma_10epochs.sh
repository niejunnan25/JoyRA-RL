#!/usr/bin/env bash
# Generic GemmaValue value-function training for LIBERO PI0 LeRobot datasets.
# Runs about TARGET_EPOCHS traditional epochs over train_split samples.

set -euo pipefail

RDMA_DEVICES=$(ls /sys/class/infiniband 2>/dev/null || true)
if [ -z "$RDMA_DEVICES" ]; then
  echo "WARN: No RDMA devices found. Continuing with NCCL_IB_DISABLE=1 for single-node training."
  export NCCL_IB_DISABLE=1
else
  NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | grep mlx5_gdr_ | tr '\n' ',' | sed 's/,$//' || true)
  export NCCL_IB_HCA
  echo "Detected RDMA devices: ${NCCL_IB_HCA}"
  export NCCL_IB_DISABLE=0
fi

NCCL_IB_GID_INDEX=""
output=$(show_gids 2>/dev/null | grep v2 || true)
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    break
  fi
done <<<"$output"
if [ -n "${NCCL_IB_GID_INDEX}" ]; then
  export NCCL_IB_GID_INDEX
else
  echo "WARN: show_gids did not report a usable v2 GID. Continuing with NCCL_IB_GID_INDEX unset/empty."
fi

export NCCL_SOCKET_IFNAME="eth0"
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

LIBERO_SUITE=${LIBERO_SUITE:-spatial}
case "${LIBERO_SUITE}" in
  spatial)
    DATA_MIX_DEFAULT=libero_spatial_pi0_20260530
    DATASET_NAME_DEFAULT=libero_spatial_pi0_20260530_lerobot
    DATASET_NAMES_DEFAULT=libero_spatial_pi0_20260530_lerobot
    LABEL_DEFAULT=libero_spatial
    MASTER_PORT_DEFAULT=29530
    ;;
  goal)
    DATA_MIX_DEFAULT=libero_goal_pi0_20260530
    DATASET_NAME_DEFAULT=libero_goal_pi0_20260530_lerobot
    DATASET_NAMES_DEFAULT=libero_goal_pi0_20260530_lerobot
    LABEL_DEFAULT=libero_goal
    MASTER_PORT_DEFAULT=29531
    ;;
  object)
    DATA_MIX_DEFAULT=libero_object_pi0_20260530
    DATASET_NAME_DEFAULT=libero_object_pi0_20260530_lerobot
    DATASET_NAMES_DEFAULT=libero_object_pi0_20260530_lerobot
    LABEL_DEFAULT=libero_object
    MASTER_PORT_DEFAULT=29532
    ;;
  libero_10|libero10|10)
    DATA_MIX_DEFAULT=libero_10_pi0_20260603_merged
    DATASET_NAME_DEFAULT=libero_10_pi0_20260603_merged_lerobot
    DATASET_NAMES_DEFAULT=libero_10_pi0_20260603_merged_lerobot
    LABEL_DEFAULT=libero_10
    MASTER_PORT_DEFAULT=29533
    ;;
  all|libero_all)
    DATA_MIX_DEFAULT=libero_all_pi0_20260603
    DATASET_NAME_DEFAULT=libero_spatial_pi0_20260530_lerobot
    DATASET_NAMES_DEFAULT="libero_spatial_pi0_20260530_lerobot libero_goal_pi0_20260530_lerobot libero_object_pi0_20260530_lerobot libero_10_pi0_20260603_merged_lerobot"
    LABEL_DEFAULT=libero_all
    MASTER_PORT_DEFAULT=29534
    ;;
  *)
    echo "ERROR: Unknown LIBERO_SUITE=${LIBERO_SUITE}. Expected spatial|goal|object|libero_10|all." >&2
    exit 1
    ;;
esac

NUM_NODES=${NUM_NODES:-${PET_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-${MASTER_PORT_DEFAULT}}}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-${PET_NPROC_PER_NODE:-8}}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /mnt/workspace/envs/conda3/bin/activate starVLA_1

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML=${CONFIG_YAML:-examples/Suqian_agibot/train_files/starvla_gemma_value_function.yaml}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/mnt/workspace/users/niejunnan/lerobot_datasets/libero}
DATA_MIX=${DATA_MIX:-${DATA_MIX_DEFAULT}}
DATASET_NAME=${DATASET_NAME:-${DATASET_NAME_DEFAULT}}
DATASET_NAMES=${DATASET_NAMES:-${DATASET_NAMES_DEFAULT:-${DATASET_NAME}}}
DATASET_DIR="${DATA_ROOT_DIR}/${DATASET_NAME}"
LABEL=${LABEL:-${LABEL_DEFAULT}}
OUTPUTS_BASE="${PROJECT_ROOT}/outputs"
DATE_STR=$(date +%Y%m%d)
BIG_NEGATIVE=${BIG_NEGATIVE:-250}
RUN_SUFFIX=${RUN_SUFFIX:-"gemma270_siglip2_${LABEL}_10epochs_neg${BIG_NEGATIVE}_$(date +%H%M%S)"}
OUTPUT_NAME="${DATA_MIX}_${DATE_STR}_${RUN_SUFFIX}"
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUTS_BASE}/value/${OUTPUT_NAME}"}
RETURNS_CACHE_DIR=${RETURNS_CACHE_DIR:-"${OUTPUTS_BASE}/cache/${DATA_MIX}"}
mkdir -p "${OUTPUTS_BASE}" "${RETURNS_CACHE_DIR}"

for _dataset_name in ${DATASET_NAMES}; do
  _info_path="${DATA_ROOT_DIR}/${_dataset_name}/meta/info.json"
  if [ ! -f "${_info_path}" ]; then
    echo "ERROR: Missing dataset meta/info.json: ${_info_path}" >&2
    exit 1
  fi
done

GPU_COUNT_PY=$(python - <<'PYCHECK'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PYCHECK
)
if [[ -z "${GPU_COUNT_PY}" || "${GPU_COUNT_PY}" -lt 1 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT_PY=$(nvidia-smi -L 2>/dev/null | wc -l | xargs)
  else
    GPU_COUNT_PY=0
  fi
fi
if ! [[ "${NUM_GPUS_PER_NODE}" =~ ^[0-9]+$ ]]; then
  if [[ "${NUM_GPUS_PER_NODE}" == "auto" && -n "${GPU_COUNT_PY}" && "${GPU_COUNT_PY}" -gt 0 ]]; then
    echo "INFO: NUM_GPUS_PER_NODE=auto; using detected GPU count ${GPU_COUNT_PY}."
    NUM_GPUS_PER_NODE=${GPU_COUNT_PY}
  else
    echo "ERROR: NUM_GPUS_PER_NODE/PET_NPROC_PER_NODE must be an integer or auto, got: ${NUM_GPUS_PER_NODE}" >&2
    exit 1
  fi
fi
if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NUM_NODES/PET_NNODES must be an integer, got: ${NUM_NODES}" >&2
  exit 1
fi
if [[ -z "${PET_NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${GPU_COUNT_PY}" && "${GPU_COUNT_PY}" -gt 0 && "${NUM_GPUS_PER_NODE}" -gt "${GPU_COUNT_PY}" ]]; then
    echo "WARN: Requested NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} > detected ${GPU_COUNT_PY}. Using ${GPU_COUNT_PY}."
    NUM_GPUS_PER_NODE=${GPU_COUNT_PY}
  fi
fi

NORMALIZE_RETURNS_PER_TASK=${NORMALIZE_RETURNS_PER_TASK:-true}
NORMALIZE_RETURNS=${NORMALIZE_RETURNS:-false}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_BINS=${NUM_BINS:-201}
TARGET_EPOCHS=${TARGET_EPOCHS:-10}
LR=${LR:-3e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PIN_MEMORY=${PIN_MEMORY:-true}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS:-0}
SAVE_STEPS=${SAVE_STEPS:-2500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-20}
TRAIN_SPLIT=${TRAIN_SPLIT:-0.9}
EVAL_STEPS=${EVAL_STEPS:-2500}
VAL_NUM_SAMPLES=${VAL_NUM_SAMPLES:-2000}
SAVE_BEST=${SAVE_BEST:-true}

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS_PER_NODE * NUM_NODES))
MAX_TRAIN_STEPS=$(python - <<PY_STEPS
import json, math
from pathlib import Path
data_root=Path("${DATA_ROOT_DIR}")
dataset_names="${DATASET_NAMES}".split()
total=sum(int(json.load(open(data_root/name/"meta/info.json"))["total_frames"]) for name in dataset_names)
train_split=float("${TRAIN_SPLIT}")
target_epochs=float("${TARGET_EPOCHS}")
global_batch=int("${GLOBAL_BATCH_SIZE}")
train_frames=max(1, int(total * train_split))
print(math.ceil(train_frames * target_epochs / global_batch))
PY_STEPS
)
DATASET_TOTAL_FRAMES=$(python - <<PY_FRAMES
import json
from pathlib import Path
data_root=Path("${DATA_ROOT_DIR}")
dataset_names="${DATASET_NAMES}".split()
print(sum(int(json.load(open(data_root/name/"meta/info.json"))["total_frames"]) for name in dataset_names))
PY_FRAMES
)

SAVE_BEST_ARGS=""
if [ "${SAVE_BEST}" = "true" ]; then
  SAVE_BEST_ARGS="--save_best"
fi
PIN_MEMORY_ARGS=""
if [ "${PIN_MEMORY}" = "true" ]; then
  PIN_MEMORY_ARGS="--pin_memory"
fi
PERSISTENT_WORKERS_ARGS=""
if [ "${PERSISTENT_WORKERS}" = "true" ]; then
  PERSISTENT_WORKERS_ARGS="--persistent_workers"
fi
NORMALIZE_ARGS=""
BIN_RANGE_ARGS=""
if [ "${NORMALIZE_RETURNS_PER_TASK}" = "true" ]; then
  NORMALIZE_ARGS="--normalize_returns_per_task --normalize_use_big_negative_in_denom"
elif [ "${NORMALIZE_RETURNS}" = "true" ]; then
  NORMALIZE_ARGS="--normalize_returns"
fi

cat <<EOF
Launching GemmaValue LIBERO training
  HOST=$(hostname)
  LIBERO_SUITE=${LIBERO_SUITE}
  PROJECT_ROOT=${PROJECT_ROOT}
  CONFIG_YAML=${CONFIG_YAML}
  DATA_ROOT_DIR=${DATA_ROOT_DIR}
  DATA_MIX=${DATA_MIX}
  DATASET_NAME=${DATASET_NAME}
  DATASET_NAMES=${DATASET_NAMES}
  DATASET_DIR=${DATASET_DIR}
  DATASET_TOTAL_FRAMES=${DATASET_TOTAL_FRAMES}
  OUTPUT_DIR=${OUTPUT_DIR}
  RETURNS_CACHE_DIR=${RETURNS_CACHE_DIR}
  BIG_NEGATIVE=${BIG_NEGATIVE}
  TARGET_EPOCHS=${TARGET_EPOCHS}
  BATCH_SIZE_PER_GPU=${BATCH_SIZE}
  NUM_BINS=${NUM_BINS}
  NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}
  NUM_NODES=${NUM_NODES}
  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
  MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}
  TRAIN_SPLIT=${TRAIN_SPLIT}
  MASTER_PORT=${MASTER_PORT}
  NUM_WORKERS=${NUM_WORKERS}
  PIN_MEMORY=${PIN_MEMORY}
  PERSISTENT_WORKERS=${PERSISTENT_WORKERS}
  PREFETCH_FACTOR=${PREFETCH_FACTOR}
  EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS}
  SAVE_STEPS=${SAVE_STEPS}
  EVAL_STEPS=${EVAL_STEPS}
EOF

python - <<PY_MIXPRINT
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
key = "${DATA_MIX}"
print("Mixture spec:", DATASET_NAMED_MIXTURES[key])
PY_MIXPRINT

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
  --num_bins ${NUM_BINS} \
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
