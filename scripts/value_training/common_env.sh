#!/usr/bin/env bash
# Shared setup for value-training launchers.

setup_value_training_project() {
  local script_path="${1:?script path is required}"
  PROJECT_ROOT="$(cd "$(dirname "${script_path}")/../../.." && pwd)"
  cd "${PROJECT_ROOT}"
  source /mnt/workspace/envs/conda3/bin/activate starVLA_1
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false
}

setup_value_training_distributed() {
  local default_master_port="${1:-29510}"
  local default_num_gpus="${2:-8}"

  local rdma_devices
  rdma_devices=$(ls /sys/class/infiniband 2>/dev/null || true)
  if [ -z "${rdma_devices}" ]; then
    echo "WARN: No RDMA devices found. Continuing with NCCL_IB_DISABLE=1 for single-node training."
    export NCCL_IB_DISABLE=1
    unset NCCL_IB_HCA NCCL_IB_GID_INDEX
  else
    local nccl_ib_hca
    nccl_ib_hca=$(echo "${rdma_devices}" | grep mlx5_gdr_ | tr '\n' ',' | sed 's/,$//' || true)
    if [ -n "${nccl_ib_hca}" ]; then
      export NCCL_IB_HCA="${nccl_ib_hca}"
      export NCCL_IB_DISABLE=0
      echo "Detected RDMA devices: ${NCCL_IB_HCA}"
    else
      echo "WARN: RDMA devices exist but no mlx5_gdr_* device was found. Continuing with NCCL_IB_DISABLE=1."
      export NCCL_IB_DISABLE=1
      unset NCCL_IB_HCA
    fi
  fi

  if [ "${NCCL_IB_DISABLE:-0}" = "0" ]; then
    local gid_index=""
    local output line ipv4
    output=$(show_gids 2>/dev/null | grep v2 || true)
    while IFS= read -r line; do
      ipv4=$(echo "${line}" | awk '{print $5}')
      if [[ -n "${ipv4}" && "${ipv4}" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "${ipv4}" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        gid_index=$(echo "${line}" | awk '{print $3}')
        break
      fi
    done <<<"${output}"
    if [ -n "${gid_index}" ]; then
      export NCCL_IB_GID_INDEX="${gid_index}"
    else
      echo "WARN: show_gids did not report a usable v2 GID. Continuing with NCCL_IB_GID_INDEX unset/empty."
      unset NCCL_IB_GID_INDEX
    fi
  fi

  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  unset NCCL_DEBUG_SUBSYS
  export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

  NUM_NODES="${NUM_NODES:-${PET_NNODES:-1}}"
  NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-${default_master_port}}}"
  NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${PET_NPROC_PER_NODE:-${default_num_gpus}}}"

  local detected_gpus
  detected_gpus=$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
  if [[ -z "${detected_gpus}" || "${detected_gpus}" -lt 1 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      detected_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | xargs)
    else
      detected_gpus=0
    fi
  fi

  if [ "${NUM_GPUS_PER_NODE}" = "auto" ]; then
    if [[ -n "${detected_gpus}" && "${detected_gpus}" -gt 0 ]]; then
      echo "INFO: NUM_GPUS_PER_NODE=auto; using detected GPU count ${detected_gpus}."
      NUM_GPUS_PER_NODE="${detected_gpus}"
    else
      echo "WARN: No GPUs detected by torch/nvidia-smi. Falling back to NUM_GPUS_PER_NODE=1."
      NUM_GPUS_PER_NODE=1
    fi
  fi

  if ! [[ "${NUM_GPUS_PER_NODE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_GPUS_PER_NODE/PET_NPROC_PER_NODE must be an integer or auto, got: ${NUM_GPUS_PER_NODE}" >&2
    exit 1
  fi
  if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_NODES/PET_NNODES must be an integer, got: ${NUM_NODES}" >&2
    exit 1
  fi

  if [[ "${NUM_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "ERROR: NUM_GPUS_PER_NODE must be >= 1, got: ${NUM_GPUS_PER_NODE}" >&2
    exit 1
  fi

  if [[ -n "${detected_gpus}" && "${detected_gpus}" -gt 0 && "${NUM_GPUS_PER_NODE}" -gt "${detected_gpus}" ]]; then
    echo "WARN: Requested NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} > detected ${detected_gpus}. Using ${detected_gpus}."
    NUM_GPUS_PER_NODE="${detected_gpus}"
  fi
}
