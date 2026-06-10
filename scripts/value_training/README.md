# Value Training Scripts

This directory is the canonical place for value-function training launchers.
Root-level `run_value_*.sh` files should be compatibility wrappers only.

## Layout

- `libero/`: LIBERO PI0 rollout value experiments and bin/epoch ablations.
- `robotwin/`: RobotWin value experiments, including the new PI0.5 qpos rollout dataset.
- `agibot/`: archived Agibot/pretrain value launchers from the project root.
- `utils/`: helper scripts that are not dataset-specific.

## Canonical Commands

LIBERO spatial 30 epochs:

```bash
cd /mnt/workspace/users/niejunnan/workspace/JoyRA-RL
bash scripts/value_training/libero/run_libero_spatial_30epochs_201bins.sh
bash scripts/value_training/libero/run_libero_spatial_30epochs_51bins.sh
```

RobotWin PI0.5 qpos 25k rollout, global BS 512, 30 traditional epochs, 101 bins:

```bash
cd /mnt/workspace/users/niejunnan/workspace/JoyRA-RL
bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_30epochs_101bins_bs512.sh
```

## Dataset Mix Names

Common `DATA_MIX` values:

- `libero_spatial_pi0_20260530`
- `libero_goal_pi0_20260530`
- `libero_object_pi0_20260530`
- `libero_10_pi0_20260603_merged`
- `libero_all_pi0_20260603`
- `robotwin_pi05_demo_clean_qpos_rollout`
- `robotwin_pi05_demo_clean_qpos_rollout_success`
- `robotwin_pi05_demo_clean_qpos_rollout_failure`
- `robotwin_pi05_demo_clean_qpos_rollout_success_failure`

## Rules

1. Put new launchers under `scripts/value_training/<dataset_family>/`.
2. Keep root-level scripts as thin wrappers only when old commands need to keep working.
3. Prefer environment-variable overrides over duplicating full scripts.
4. Put data conversion scripts under `scripts/data_conversion/`.
