#!/usr/bin/env bash
set -euo pipefail

########################################
# 用法：分别是并行2任务，25个screen；并行1任务，50个screen。默认50
#   bash start_servers.sh 25
#   bash start_servers.sh 50
########################################

N="${1:-50}"
if [[ "${N}" != "25" && "${N}" != "50" ]]; then
  echo "Usage: bash $0 [25|50]   (default: 50)"
  exit 1
fi

# 修改自己的root目录
JOYRA_ROOT="/mnt/workspace/users/weiziming/projects/JoyRA"
SERVER_SCRIPT="examples/Robotwin/eval_files/run_policy_server.sh"

# checkpoint & model （跑不同的设置，记得修改这里！）
CKPT_TAG="260131-robotwin-Groot-setting7-plusrandom"
MODEL_PATH="${JOYRA_ROOT}/results/robotwin2/${CKPT_TAG}/robotwin/final_model/pytorch_model.pt"

# server1->5801, server2->5802 ...
BASE_PORT=5800

NGPU=8

# rem=N%8，offset=(8-rem)%8
rem=$(( N % NGPU ))
offset=$(( (NGPU - rem) % NGPU ))

echo "[INFO] Starting ${N} servers with screen..."
echo "[INFO] JOYRA_ROOT=${JOYRA_ROOT}"
echo "[INFO] CKPT_TAG=${CKPT_TAG}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] PORT: ${BASE_PORT}+i"
echo "[INFO] GPU spread over ${NGPU} gpus (extra goes to last gpus)"
echo

for i in $(seq 1 "${N}"); do
  port=$(( BASE_PORT + i ))
  gpu=$(( ( (i - 1 + offset) % NGPU ) ))
  sname="server${i}"

  # 如果同名screen已存在就跳过（避免重复启动）
  if screen -list | grep -q "\.${sname}[[:space:]]"; then
    echo "[SKIP] screen ${sname} already exists."
    continue
  fi

  cmd="cd ${JOYRA_ROOT} && bash ${SERVER_SCRIPT} ${gpu} ${port} ${MODEL_PATH}"

  echo "[START] ${sname}  gpu=${gpu}  port=${port}"
  screen -dmS "${sname}" bash -lc "${cmd}"
done

echo
echo "[DONE] All server screens launched."
echo "Tips:"
echo "  screen -ls"
echo "  screen -r server1"
