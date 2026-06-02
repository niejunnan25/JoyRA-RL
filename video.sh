source /mnt/workspace/envs/conda3/bin/activate starVLA_1
cd /mnt/workspace1/users/tangyili/Projects/JoyRA-RL
export PYTHONPATH=/mnt/workspace1/users/tangyili/Projects/JoyRA-RL:$PYTHONPATH
python scripts/scan_lerobot_videos_decord.py \
  --data_root_dir /mnt/workspace/datasets \
  --data_mix robotwin_orig_plus_offline_v2 \
  --output_json /tmp/bad_videos.json