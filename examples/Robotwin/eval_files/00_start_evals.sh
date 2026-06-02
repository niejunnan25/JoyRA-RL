#!/usr/bin/env bash
set -euo pipefail

########################################
# 用法：分别是并行2任务，25个screen；并行1任务，50个screen。默认50
#   bash start_evals.sh 25
#   bash start_evals.sh 50
########################################


N="${1:-50}"
if [[ "${N}" != "25" && "${N}" != "50" ]]; then
  echo "Usage: bash $0 [25|50]   (default: 50)"
  exit 1
fi

# 修改自己的root目录
JOYRA_ROOT="/mnt/workspace/users/weiziming/projects/JoyRA" 
EVAL_DIR="${JOYRA_ROOT}/examples/Robotwin/eval_files"
EVAL_SCRIPT="run_tasks_splitN.sh"

# checkpoint & model （跑不同的设置，记得修改这里！）
CKPT_TAG="260131-robotwin-Groot-setting7-plusrandom"
MODEL_PATH="${JOYRA_ROOT}/results/robotwin2/${CKPT_TAG}/robotwin/final_model/pytorch_model.pt"

# logdir
LOGDIR="eval-logs/${CKPT_TAG}"

# eval1->5801, eval2->5802 ...
BASE_PORT=5800

NGPU=8

# rem=N%8，offset=(8-rem)%8
rem=$(( N % NGPU ))
offset=$(( (NGPU - rem) % NGPU ))

echo "[INFO] Starting ${N} evals with screen..."
echo "[INFO] EVAL_DIR=${EVAL_DIR}"
echo "[INFO] CKPT_TAG=${CKPT_TAG}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] LOGDIR=${LOGDIR}"
echo

for i in $(seq 1 "${N}"); do
  port=$(( BASE_PORT + i ))
  gpu=$(( ( (i - 1 + offset) % NGPU ) ))
  sname="eval${i}"

  if screen -list | grep -q "\.${sname}[[:space:]]"; then
    echo "[SKIP] screen ${sname} already exists."
    continue
  fi

  # i 是 split id；N 是 split 总数（25/50）
  cmd="cd ${EVAL_DIR} && bash ${EVAL_SCRIPT} ${i} ${N} --ckpt ${CKPT_TAG} --gpu ${gpu} --port ${port} --logdir ${LOGDIR} --model ${MODEL_PATH}"

  echo "[START] ${sname}  gpu=${gpu}  port=${port}  split=${i}/${N}"
  screen -dmS "${sname}" bash -lc "${cmd}"
done

echo
echo "[DONE] All eval screens launched."
echo "Tips:"
echo "  screen -ls"
echo "  screen -r eval1"
