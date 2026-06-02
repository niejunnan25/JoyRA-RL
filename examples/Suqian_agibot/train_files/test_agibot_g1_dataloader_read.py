#!/usr/bin/env python3
"""
抽样校验 LeRobot 单数据集读取是否正确（Agibot G1 Kingkong / subtask 场景）。

检查项：
  1) 开启 skip_invalid_subtask_frames 时，all_steps 中每一步对应 parquet 行的 subtask_index >= 0
  2) 该步语言（annotation.human.subtask_description）与 meta/subtasks.jsonl 中 subtask_index 一致
     （与数据集内 _process_task_text 处理后的文案一致）

用法示例：
  cd /mnt/workspace1/users/tangyili/Projects/JoyRA-RL
  source /mnt/workspace/envs/conda3/bin/activate starVLA_1
  export PYTHONPATH=$PWD:$PYTHONPATH

  python examples/Suqian_agibot/train_files/test_agibot_g1_dataloader_read.py \\
      --data_root /mnt/workspace1/datasets \\
      --data_name suqian_agibot_kingkong/desk_organization_combine_pnp/G1_task_4196_subtask_new_eepose \\
      --skip_invalid_subtask_frames \\
      --num_checks 50
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def _int_cell(v) -> int:
    if hasattr(v, "item"):
        return int(v.item())
    return int(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="相对 data_root 的数据集路径，如 suqian_agibot_kingkong/desk_organization_combine_pnp/G1_task_xxx",
    )
    parser.add_argument(
        "--robot_type",
        default="agibot_g1_kingkong_eepose",
        help="ROBOT_TYPE_CONFIG_MAP 中的名字",
    )
    parser.add_argument(
        "--skip_invalid_subtask_frames",
        action="store_true",
        help="与 train_value 一致：不将 subtask_index<0 的帧加入 all_steps",
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--num_checks", type=int, default=40, help="随机抽样条数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    data_cfg = {
        "lerobot_version": "v2.0",
        "frame_stride": max(1, args.frame_stride),
        "skip_invalid_subtask_frames": args.skip_invalid_subtask_frames,
    }

    ds = make_LeRobotSingleDataset(
        args.data_root,
        args.data_name,
        args.robot_type,
        data_cfg=data_cfg,
    )

    if ds.subtasks is None:
        print("ERROR: meta/subtasks.jsonl 未加载，无法校验子任务语言。", file=sys.stderr)
        sys.exit(2)

    lang_key = ds.modality_keys["language"][0]
    print(f"数据集: {ds.dataset_name}")
    print(f"len(all_steps)={len(ds)}, language_key={lang_key}")
    print(f"skip_invalid_subtask_frames={args.skip_invalid_subtask_frames}")
    print()

    rng = random.Random(args.seed)
    n = min(args.num_checks, len(ds))
    if n == 0:
        print("ERROR: 数据集步数为 0", file=sys.stderr)
        sys.exit(3)

    indices = rng.sample(range(len(ds)), n)
    errors: list[str] = []

    for k, idx in enumerate(indices):
        traj_id, base = ds.all_steps[idx]
        traj = ds.get_trajectory_data(traj_id)
        st_raw = traj["subtask_index"].iloc[base]
        st = _int_cell(st_raw)

        if args.skip_invalid_subtask_frames and st < 0:
            errors.append(f"idx={idx} traj={traj_id} base={base}: subtask_index={st} (期望 >=0)")
            continue

        raw = ds.get_step_data(traj_id, base)
        lang_list = raw[lang_key]
        lang = lang_list[0] if lang_list else ""

        if st < 0:
            # 未开 skip 时允许 -1，语言应为空串
            if lang != "":
                errors.append(
                    f"idx={idx} subtask_index={st}: 期望语言为空串，得到 {lang[:80]!r}"
                )
            continue

        if st not in ds.subtasks.index:
            errors.append(f"idx={idx} subtask_index={st} 不在 subtasks 表索引中")
            continue

        expected = ds.subtasks.loc[st]["task"]
        if expected is None:
            expected = ""
        if lang != expected:
            errors.append(
                f"idx={idx} traj={traj_id} base={base} st={st}:\n"
                f"  语言不一致\n"
                f"  get_step_data: {lang[:200]!r}\n"
                f"  subtasks 表   : {str(expected)[:200]!r}"
            )

        if k < 5:
            print(f"[样本 {k}] idx={idx} traj={traj_id} step={base} subtask_index={st}")
            print(f"  language: {lang[:160]}{'...' if len(lang) > 160 else ''}")
            print()

    if errors:
        print(f"失败 {len(errors)} / {n}", file=sys.stderr)
        for e in errors[:20]:
            print(e, file=sys.stderr)
        if len(errors) > 20:
            print(f"... 另有 {len(errors) - 20} 条", file=sys.stderr)
        sys.exit(1)

    print(f"PASS：{n} 条随机抽样校验通过。")


if __name__ == "__main__":
    main()
