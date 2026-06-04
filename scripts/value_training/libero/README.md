# LIBERO Value Training

Canonical LIBERO launchers live in this directory. Root-level `run_value_libero*.sh`
files are compatibility wrappers only.

## Main Entrypoints

- `run_value_libero_gemma_10epochs.sh`: generic suite-aware launcher. Select the suite with `LIBERO_SUITE`.
- `run_value_libero_spatial_gemma_10epochs.sh`: spatial, `big_negative=110`.
- `run_value_libero_10_gemma_10epochs.sh`: libero_10, `big_negative=300`.
- `run_value_libero_all_gemma_10epochs.sh`: spatial + goal + object + libero_10, `big_negative=250`.
- `run_libero_spatial_30epochs_201bins.sh`: spatial 30 epochs, 201 bins, global BS 1024 on 8 GPUs.
- `run_libero_spatial_30epochs_51bins.sh`: spatial 30 epochs, 51 bins, global BS 1024 on 8 GPUs.

## Examples

```bash
cd /mnt/workspace/users/niejunnan/workspace/JoyRA-RL
bash scripts/value_training/libero/run_libero_spatial_30epochs_201bins.sh
bash scripts/value_training/libero/run_libero_spatial_30epochs_51bins.sh
LIBERO_SUITE=all bash scripts/value_training/libero/run_value_libero_gemma_10epochs.sh
```

The generic launcher computes `MAX_TRAIN_STEPS` from dataset frame count, `TARGET_EPOCHS`,
`TRAIN_SPLIT`, and global batch size.
