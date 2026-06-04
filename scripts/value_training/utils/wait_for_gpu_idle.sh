#!/usr/bin/env bash
set -euo pipefail

#
# 等待 GPU “空闲”后退出 0，便于用：
#   bash wait_for_gpu_idle.sh && bash your_train.sh
#
# 判定“空闲”的条件（对所有目标 GPU 都满足）：
# - GPU 利用率 <= GPU_UTIL_MAX（默认 5%）
# - 显存占用 <= MEM_USED_MAX_MB（默认 1024MB）
# 并且需要连续满足 IDLE_SECS 秒（默认 60s），每隔 INTERVAL 秒采样一次（默认 5s）。
#
# 目标 GPU 集合：
# - 若设置了 CUDA_VISIBLE_DEVICES（形如 "0,1,2" 或 "0"），则按其物理 GPU id 过滤
# - 否则默认检查所有 GPU
#

GPU_UTIL_MAX="${GPU_UTIL_MAX:-5}"            # %
MEM_USED_MAX_MB="${MEM_USED_MAX_MB:-1024}"  # MB
IDLE_SECS="${IDLE_SECS:-600}"                # seconds
INTERVAL="${INTERVAL:-5}"                   # seconds
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"           # 0 means no timeout

QUIET="${QUIET:-0}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() {
  if [[ "${QUIET}" != "1" ]]; then
    echo "[$(ts)] $*" >&2
  fi
}
die() { echo "[$(ts)] ERROR: $*" >&2; exit 2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

require_cmd nvidia-smi

trim() {
  local s="$1"
  # remove leading/trailing whitespace
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

get_target_gpu_indices() {
  # prints space-separated physical GPU indices
  local indices=()

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # CUDA_VISIBLE_DEVICES is expected to be physical ids in your environment (common for training launchers).
    # We treat it as a comma-separated list and pass each entry to nvidia-smi -i.
    IFS=',' read -r -a indices <<<"${CUDA_VISIBLE_DEVICES}"
    for i in "${!indices[@]}"; do
      indices[$i]="$(trim "${indices[$i]}")"
      [[ -n "${indices[$i]}" ]] || unset 'indices[$i]'
    done
  else
    # All GPUs
    mapfile -t indices < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true)
  fi

  if [[ "${#indices[@]}" -eq 0 ]]; then
    die "no target GPUs detected (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')"
  fi

  echo "${indices[*]}"
}

query_one_gpu() {
  # args: gpu_index
  # output: "util mem_used mem_total" (all integers, MB for mem)
  local idx="$1"
  local out
  out="$(nvidia-smi -i "${idx}" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  out="$(trim "${out}")"
  [[ -n "${out}" ]] || return 1

  # out example: "0, 120, 81920"
  local util mem_used mem_total
  IFS=',' read -r util mem_used mem_total <<<"${out}"
  util="$(trim "${util}")"
  mem_used="$(trim "${mem_used}")"
  mem_total="$(trim "${mem_total}")"
  [[ "${util}" =~ ^[0-9]+$ ]] || return 1
  [[ "${mem_used}" =~ ^[0-9]+$ ]] || return 1
  [[ "${mem_total}" =~ ^[0-9]+$ ]] || return 1
  echo "${util} ${mem_used} ${mem_total}"
}

all_gpus_idle_once() {
  # returns 0 if all target GPUs satisfy thresholds in this sample
  local indices=("$@")
  local idx util mem_used mem_total
  local ok=0

  for idx in "${indices[@]}"; do
    if ! read -r util mem_used mem_total < <(query_one_gpu "${idx}"); then
      log "GPU ${idx}: failed to query"
      return 1
    fi

    if (( util > GPU_UTIL_MAX )); then
      ok=1
    fi
    if (( mem_used > MEM_USED_MAX_MB )); then
      ok=1
    fi

    if [[ "${QUIET}" != "1" ]]; then
      echo "  GPU ${idx}: util=${util}% mem=${mem_used}/${mem_total}MB" >&2
    fi
  done

  [[ "${ok}" -eq 0 ]]
}

main() {
  local indices
  read -r -a indices <<<"$(get_target_gpu_indices)"
  log "Watching GPUs: ${indices[*]}"
  log "Thresholds: util<=${GPU_UTIL_MAX}% mem_used<=${MEM_USED_MAX_MB}MB, require idle for ${IDLE_SECS}s (interval ${INTERVAL}s), timeout=${TIMEOUT_SECS}s"

  local idle_acc=0
  local start_ts now_ts elapsed
  start_ts="$(date +%s)"

  while true; do
    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))

    if (( TIMEOUT_SECS > 0 )) && (( elapsed >= TIMEOUT_SECS )); then
      die "timeout after ${TIMEOUT_SECS}s waiting for idle GPUs"
    fi

    if all_gpus_idle_once "${indices[@]}"; then
      idle_acc=$(( idle_acc + INTERVAL ))
      log "Idle sample OK. accumulated_idle=${idle_acc}/${IDLE_SECS}s"
    else
      idle_acc=0
      log "Not idle yet. accumulated_idle reset to 0/${IDLE_SECS}s"
    fi

    if (( idle_acc >= IDLE_SECS )); then
      log "GPUs are idle for >= ${IDLE_SECS}s. Proceed."
      exit 0
    fi

    sleep "${INTERVAL}"
  done
}

main "$@"

