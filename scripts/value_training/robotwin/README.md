# RobotWin Value Training

Canonical RobotWin launchers live in this directory. Prefer these paths over
root-level scripts.

## Main Entrypoints

- `run_value_robotwin_gemma_with_RL_T_bs2x.sh`: configurable Gemma value launcher.
- `run_robotwin_pi05_qpos_gemma_30epochs_101bins_bs512.sh`: new 25k pi0.5 demo_clean qpos dataset, global BS 512, 30 traditional epochs, 101 bins.
- `run_robotwin_pi05_blocks_ranking_gemma_30epochs_bs512.sh`: merged `blocks_ranking_rgb` + `blocks_ranking_size`, global BS 512, 30 traditional epochs, `BIG_NEGATIVE=600`.
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
bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_30epochs_101bins_bs512.sh
```

Useful overrides:

```bash
FRAME_STRIDE=2 MAX_TRAIN_STEPS=254451 bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_30epochs_101bins_bs512.sh
DATA_MIX=robotwin_pi05_demo_clean_qpos_rollout_success_failure bash scripts/value_training/robotwin/run_robotwin_pi05_qpos_gemma_30epochs_101bins_bs512.sh
```

## PI0.5 Blocks Ranking

Merged block-ranking command:

```bash
cd /mnt/workspace/users/niejunnan/workspace/JoyRA-RL
bash scripts/value_training/robotwin/run_robotwin_pi05_blocks_ranking_gemma_30epochs_bs512.sh
```

Available qpos left-right block-ranking mixes:

```text
robotwin_pi05_blocks_ranking_rgb
robotwin_pi05_blocks_ranking_size
robotwin_pi05_blocks_ranking_all
```
