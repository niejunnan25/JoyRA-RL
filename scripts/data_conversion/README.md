# Data Conversion Scripts

Data conversion utilities live here. These scripts produce LeRobot-format
datasets consumed by `starVLA/training/train_value.py`.

## RobotWin PI0.5 Qpos

- `convert_robotwin_pi05_pkl_to_lerobot.py`: convert raw PI0.5 RobotWin pkl rollouts into one merged LeRobot dataset.
- `split_lerobot_by_success_copy.py`: split a merged dataset into physical success/failure datasets.
- `split_lerobot_by_task_copy.py`: split a merged dataset into one physical LeRobot dataset per `task_name`.

Current generated datasets:

```text
/mnt/workspace/users/niejunnan/datasets/robotwin_pi05_demo_clean_qpos_rollout
/mnt/workspace/users/niejunnan/datasets/robotwin_pi05_demo_clean_qpos_rollout_success
/mnt/workspace/users/niejunnan/datasets/robotwin_pi05_demo_clean_qpos_rollout_failure
/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot
```

`robotwin_rollout_lerobot` contains 50 task subdirectories, each with success
and failure episodes packed together.
