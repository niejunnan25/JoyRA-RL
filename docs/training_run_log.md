# Training Run Log

## 2026-05-29 22:49:39 CST

Purpose: rerun the RobotWin value critic with higher throughput while keeping the total number of training samples aligned with the previous setup.

Remote workspace:

```bash
/mnt/workspace/users/niejunnan/workspace/JoyRA-RL
```

Command run:

```bash
cd /mnt/workspace/users/niejunnan/workspace/JoyRA-RL

BIG_NEGATIVE=1200 \
BATCH_SIZE=16 \
EPOCHS=10 \
STEPS_PER_EPOCH=10000 \
NUM_WORKERS=8 \
PREFETCH_FACTOR=4 \
PIN_MEMORY=true \
PERSISTENT_WORKERS=true \
EMPTY_CACHE_STEPS=0 \
bash run_value_robotwin_with_RL_T.sh robotwin_orig_plus_offline_v2
```

Training-volume equivalence:

```text
Previous: per-GPU batch 8  * 8 GPUs * 20000 steps/epoch * 10 epochs = 12,800,000 samples
Current:  per-GPU batch 16 * 8 GPUs * 10000 steps/epoch * 10 epochs = 12,800,000 samples
```

Notes:

- `BIG_NEGATIVE=1200` is the failure penalty used for the pi0.6-style empirical-return critic target.
- `BATCH_SIZE` is per GPU/rank under DDP.
- `EMPTY_CACHE_STEPS=0` disables periodic `torch.cuda.empty_cache()` during training.
