# RobotWin Value Training

Canonical RobotWin launchers live in this directory. Prefer these paths over
root-level scripts.

## Main Entrypoints

- `run_value_robotwin_gemma_with_RL_T_bs2x.sh`: configurable Gemma value launcher.
- `run_robotwin_pi05_qpos_gemma_10epochs_bs1024.sh`: new 25k pi0.5 demo_clean qpos dataset, global BS 1024, 10 traditional epochs.
- `run_value_robotwin_with_RL_T.sh`: archived Qwen-style launcher.
- `run_value_robotwin_with_RL_T_bs2x.sh`: archived bs2x launcher.

## New PI0.5 Qpos Dataset

Default dataset:

```text
/mnt/workspace/users/niejunnan/datasets/robotwin_pi05_demo_clean_qpos_rollout
```

Default command:

```bash
cd /mnt/workspace/users/niejunnan/workspace/JoyRA-RL
bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_10epochs_bs1024.sh
```

Useful overrides:

```bash
FRAME_STRIDE=2 MAX_TRAIN_STEPS=42409 bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_10epochs_bs1024.sh
DATA_MIX=robotwin_pi05_demo_clean_qpos_rollout_success_failure bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_10epochs_bs1024.sh
```
