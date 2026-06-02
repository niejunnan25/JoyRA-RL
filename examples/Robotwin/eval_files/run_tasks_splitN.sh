#!/bin/bash
set -u

######################################
# 用法：
# ./run_tasks_splitN.sh <part_id> <num_parts> [options]
#
# options:
#   --ckpt <name>       ckpt_setting (default: my_test_v1)
#   --gpu <id>          GPU id (default: 0)
#   --port <port>       server port (default: 5694)
#   --logdir <dir>      log directory (default: eval_logs_splitN)
#   --seed <seed>       seed (default: 0)
#   --rerun             force rerun even if log exists
######################################
# export CUROBO_TORCH_COMPILE_DISABLE=0
source /mnt/workspace/envs/conda3/bin/activate robotwin2_n

# ---------- 必选参数 ----------
PART_ID="${1:-}"
NUM_PARTS="${2:-}"
shift 2 || true

# ---------- 默认参数 ----------
CKPT_SETTING="my_test_v1"
GPU_ID=0
PORT=5694
LOG_DIR="eval_logs_splitN"
SEED=0
RERUN=0

# ---------- 解析可选参数 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT_SETTING="$2"; shift 2;;
    --gpu)
      GPU_ID="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --logdir)
      LOG_DIR="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --rerun)
      RERUN=1; shift 1;;
    --model)
      MODEL_PATH="$2"; shift 2;;
    *)
      echo "Unknown option: $1"
      exit 1;;
  esac
done

# ---------- 合法性检查 ----------
if [[ -z "$PART_ID" || -z "$NUM_PARTS" ]]; then
  echo "Usage: $0 <part_id> <num_parts> [options]"
  exit 1
fi
if ! [[ "$PART_ID" =~ ^[0-9]+$ && "$NUM_PARTS" =~ ^[0-9]+$ ]]; then
  echo "part_id and num_parts must be integers."
  exit 1
fi
if (( NUM_PARTS < 1 )); then
  echo "num_parts must be >= 1"
  exit 1
fi
if (( PART_ID < 1 || PART_ID > NUM_PARTS )); then
  echo "part_id must be in [1, num_parts]"
  exit 1
fi

# ---------- 固定参数 ----------
TASK_CONFIG="demo_clean"
EVAL_SCRIPT="eval.sh"

mkdir -p "$LOG_DIR"

# ---------- 任务列表 ----------
TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle_horizontally
  shake_bottle
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

TOTAL=${#TASKS[@]}
CHUNK=$(( (TOTAL + NUM_PARTS - 1) / NUM_PARTS ))
START=$(( (PART_ID - 1) * CHUNK ))
END=$(( START + CHUNK - 1 ))
if (( START >= TOTAL )); then
  echo "This part has no tasks."
  exit 0
fi
if (( END >= TOTAL )); then END=$((TOTAL - 1)); fi

echo "======================================"
echo "Run part ${PART_ID}/${NUM_PARTS}"
echo "Tasks index range: ${START}..${END}"
echo "CKPT_SETTING: ${CKPT_SETTING}"
echo "GPU_ID: ${GPU_ID}"
echo "PORT: ${PORT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "Skip existing logs: $((1-RERUN))"
echo "======================================"

FAILED_LIST="${LOG_DIR}/failed_part.txt"
: > "$FAILED_LIST"

for (( i=START; i<=END; i++ )); do
  TASK="${TASKS[$i]}"
  LOG_FILE="${LOG_DIR}/${TASK}.log"

  if [[ "$RERUN" -eq 0 && -s "$LOG_FILE" ]]; then
    echo "[SKIP] ${TASK}"
    continue
  fi

  echo "[RUN ] ${TASK}"
  bash "$EVAL_SCRIPT" \
    "$TASK" \
    "$TASK_CONFIG" \
    "$CKPT_SETTING" \
    "$SEED" \
    "$GPU_ID" \
    "$PORT" \
    "$MODEL_PATH" \
    > "$LOG_FILE" 2>&1

  RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "[FAIL] ${TASK}"
    echo "$TASK" >> "$FAILED_LIST"
  else
    echo "[DONE] ${TASK}"
  fi
done

echo "======================================"
echo "Finished part ${PART_ID}/${NUM_PARTS}"
if [[ -s "$FAILED_LIST" ]]; then
  echo "Failed tasks:"
  cat "$FAILED_LIST"
else
  echo "No failed tasks."
fi
echo "======================================"
