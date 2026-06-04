#!/usr/bin/env bash
# Agibot G1（宿迁 KingKong）desk_organization_combine_pnp 价值函数训练。
# 数据与 Robotwin 无关；mixture 见 mixtures.py 中 agibot_g1_desk_organization_combine_pnp。
# 用法:
#   bash run_value_agibot_g1_desk_organization_T.sh
# 可选覆盖（默认已指向 workspace1 数据根目录与本 mixture）:
#   DATA_ROOT_DIR=... DATA_MIX=... bash run_value_agibot_g1_desk_organization_T.sh
# 预训练权重（可由 run_value_pretrain_with_RL_T.sh agi_suqian_egodex_robocasa_robotwin_mix 训练得到）:
#   默认 INIT_CHECKPOINT 见下方；不加载则 INIT_CHECKPOINT= bash ...
# 产出: outputs/value/<DATA_MIX>_YYYYMMDD/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_env.sh"
setup_value_training_distributed 29510 8
setup_value_training_project "${BASH_SOURCE[0]}"

CONFIG_YAML=${CONFIG_YAML:-examples/Suqian_agibot/train_files/starvla_value_function.yaml}

# 数据在 /mnt/workspace1/datasets/suqian_agibot_kingkong/...；若迁到 /mnt/workspace/datasets 可改此处或做软链
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/workspace1/datasets}"
DATA_MIX="${DATA_MIX:-agibot_g1_desk_organization_combine_pnp}"

# agi_suqian_egodex_robocasa_robotwin_mix 预训练 checkpoint；置空则随机初始化
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${PROJECT_ROOT}/outputs/value/agi_suqian_egodex_robocasa_robotwin_mix_20260413/checkpoint_step_1000000.pt}"

OUTPUTS_BASE="${PROJECT_ROOT}/outputs"
DATE_STR=$(date +%Y%m%d)
mkdir -p "${OUTPUTS_BASE}"
OUTPUT_DIR="${OUTPUTS_BASE}/value/${DATA_MIX}_${DATE_STR}"
RETURNS_CACHE_DIR="${OUTPUTS_BASE}/cache/${DATA_MIX}"
echo "DATA_ROOT_DIR=${DATA_ROOT_DIR}"
echo "DATA_MIX=${DATA_MIX}"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT:-<none>}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RETURNS_CACHE_DIR=${RETURNS_CACHE_DIR}"
if [ -n "${INIT_CHECKPOINT:-}" ] && [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: INIT_CHECKPOINT does not exist: ${INIT_CHECKPOINT}" >&2
  echo "Set INIT_CHECKPOINT= to train from the VLM base, or pass INIT_CHECKPOINT=/abs/path/to/checkpoint.pt." >&2
  exit 1
fi

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
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT_PY=$(nvidia-smi -L 2>/dev/null | wc -l | xargs)
  else
    GPU_COUNT_PY=0
  fi
fi

if [[ -z "${PET_NPROC_PER_NODE}" ]]; then
  if [[ -n "${GPU_COUNT_PY}" && "${GPU_COUNT_PY}" -gt 0 ]]; then
    DETECTED_GPUS=${GPU_COUNT_PY}
    if [[ -n "${NUM_GPUS_PER_NODE}" && "${NUM_GPUS_PER_NODE}" -gt "${DETECTED_GPUS}" ]]; then
      echo "WARN: Requested NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} > detected ${DETECTED_GPUS}. Using ${DETECTED_GPUS}."
      NUM_GPUS_PER_NODE=${DETECTED_GPUS}
    elif [[ -z "${NUM_GPUS_PER_NODE}" || "${NUM_GPUS_PER_NODE}" -lt 1 ]]; then
      NUM_GPUS_PER_NODE=${DETECTED_GPUS}
    fi
  else
    echo "WARN: No GPUs detected by torch/nvidia-smi. Falling back to NUM_GPUS_PER_NODE=1."
    NUM_GPUS_PER_NODE=1
  fi
fi

NORMALIZE_RETURNS_PER_TASK=true
NORMALIZE_RETURNS=false

EPOCHS=10
STEPS_PER_EPOCH=20000
BATCH_SIZE=8
LR=3e-5
NUM_WORKERS=4

SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=20

TRAIN_SPLIT=0.9
EVAL_INTERVAL=1
VAL_NUM_SAMPLES=2000
SAVE_BEST=false

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
  --language_prefix "tidy up the desk" \
  --skip_invalid_subtask_frames \
  ${INIT_CHECKPOINT:+--init_from_checkpoint "${INIT_CHECKPOINT}"} \
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

python examples/Suqian_agibot/train_files/plot_training_losses.py \
  --loss_file "${OUTPUT_DIR}/training_losses.json" \
  --output "${OUTPUT_DIR}/training_losses_curve.png"
