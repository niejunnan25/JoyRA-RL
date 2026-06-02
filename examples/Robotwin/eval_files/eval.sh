#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# export LD_LIBRARY_PATH=/mnt/workspace/lib/nvidia_egl:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/mnt/workspace/lib/nvidia_egl:$LD_LIBRARY_PATH

export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MUJOCO_GL=egl
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export ASSETS_PATH=/mnt/workspace/users/weiziming/projects/RoboTwin # 修改成自己的

# 修改成自己的Robotwin目录
ROBOTWIN_PATH=/mnt/workspace/users/weiziming/projects/RoboTwin

policy_name="model2robotwin_interface"

task_name=${1}
task_config=${2}
ckpt_setting=${3:-starvla_demo}
seed=${4:-0}
gpu_id=${5:-0} # default is 0
server_port=${6:-5694}
model_path=${7:-""}   

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH=$(pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

cd $ROBOTWIN_PATH

echo "PYTHONPATH: $PYTHONPATH"

DEPLOY_POLICY_PATH_TO_USE="$DEPLOY_POLICY_PATH"

if [[ -n "${model_path}" ]]; then
  if [[ ! -f "${model_path}" ]]; then
    echo -e "\033[31m[ERROR] model not found: ${model_path}\033[0m"
    exit 1
  fi

  tmp_cfg=$(mktemp ./deploy_policy.XXXXXX.yml)

  cp "$DEPLOY_POLICY_PATH" "$tmp_cfg"

  sed -i "0,/^[[:space:]]*policy_ckpt_path:/s|^[[:space:]]*policy_ckpt_path:.*|policy_ckpt_path: \"${model_path}\"|" "$tmp_cfg"

  DEPLOY_POLICY_PATH_TO_USE="$tmp_cfg"

  echo -e "\033[33m[override] policy_ckpt_path -> ${model_path}\033[0m"
  echo -e "\033[33m[temp config] ${tmp_cfg}\033[0m"

  trap "rm -f $tmp_cfg" EXIT INT TERM
fi


PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config $DEPLOY_POLICY_PATH_TO_USE \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --port ${server_port}
