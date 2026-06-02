#!/bin/bash

############################################
# 用法：
#   ./start_server.sh [gpu_id] [port] [ckpt_path]
#
# 示例：
#   ./start_server.sh 3 5697 /path/to/pytorch_model.pt
############################################
# export CUROBO_TORCH_COMPILE_DISABLE=0
export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/workspace/envs/conda3/envs/starVLA_1/bin/python
source /mnt/workspace/envs/conda3/bin/activate starVLA_1

# ---------- 默认值（不传参数就用这些） ----------
DEFAULT_CKPT="/mnt/workspace/users/weiziming/projects/JoyRA/results/robotwin2/260113-robotwin-train-setting7/robotwin/final_model/pytorch_model.pt"
DEFAULT_GPU_ID=3
DEFAULT_PORT=5697

# ---------- 从命令行读取 ----------
gpu_id="${1:-$DEFAULT_GPU_ID}"
port="${2:-$DEFAULT_PORT}"
your_ckpt="${3:-$DEFAULT_CKPT}"

echo "========================================"
echo "Starting StarVLA Policy Server"
echo "GPU_ID : ${gpu_id}"
echo "PORT   : ${port}"
echo "CKPT   : ${your_ckpt}"
echo "========================================"

################# star Policy Server ######################

CUDA_VISIBLE_DEVICES=${gpu_id} \
${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16
