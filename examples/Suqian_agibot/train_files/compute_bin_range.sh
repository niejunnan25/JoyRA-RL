#!/bin/bash

# 计算数据集的 value bin 范围（min/max）
# 用法：bash examples/Suqian_agibot/train_files/compute_bin_range.sh

source /mnt/workspace/envs/conda3/bin/activate starVLA_1
cd /mnt/workspace/users/daiyixiang/JoyRA-RL

# ===== 配置区域（按需修改） =====
DATA_ROOT_DIR=/mnt/workspace/datasets
DATA_MIX=sq_agi_beta
OUTPUT_JSON=examples/Suqian_agibot/train_files/value_bin_range.json

# 采样参数
SAMPLE_SIZE=-1 
GAMMA=1.0
BIG_NEGATIVE=100.0
SUCCESS_COL=episode_success

# ===== 计算 bin range =====
python starVLA/training/compute_value_bin_range.py \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --data_mix "${DATA_MIX}" \
    --output_json "${OUTPUT_JSON}" \
    --sample_size ${SAMPLE_SIZE} \
    --gamma ${GAMMA} \
    --big_negative ${BIG_NEGATIVE} \
    --success_col "${SUCCESS_COL}"

echo ""
echo "Bin range computed and saved to: ${OUTPUT_JSON}"
echo "You can now use this file in run_value.sh with --bin_range_json ${OUTPUT_JSON}"
