#!/usr/bin/env python3
"""
遍历 LeRobot 数据集（与 train_value 相同的 data_root_dir + data_mix 或单个子集），
用 decord 打开每个 episode 所需的视频文件，找出缺失、空文件或 decord 无法解码的文件。

典型报错（与训练一致）：
  DECORDError: ... cannot find video stream with wanted index: -1

用法示例：
  python scripts/scan_lerobot_videos_decord.py \\
    --data_root_dir /mnt/workspace/datasets \\
    --data_mix robotwin_orig_plus_offline_v2

  # 只扫一个子目录 + robot_type
  python scripts/scan_lerobot_videos_decord.py \\
    --data_root_dir /mnt/workspace/datasets \\
    --dataset_name rl_offline_ee3/blocks_ranking_size \\
    --robot_type my_robotwin_reversed

  # 尝试读第 0 帧（更慢，但能发现部分仅头部损坏的文件）
  python scripts/scan_lerobot_videos_decord.py ... --read_first_frame
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_on_path()

from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


@dataclass
class VideoRef:
    path: str
    dataset_name: str
    episode_index: int
    video_key: str


def _dedupe_mixture(spec: list[tuple]) -> list[tuple]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple] = []
    for item in spec:
        if len(item) == 3:
            d_name, d_weight, robot_type = item
        elif len(item) == 4:
            d_name, d_weight, robot_type, _ = item
        else:
            raise ValueError(f"Invalid mixture item: {item}")
        key = (d_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        out.append((d_name, d_weight, robot_type))
    return out


def collect_unique_video_paths(
    dataset_name: str,
    ds,
) -> dict[str, VideoRef]:
    """path_str -> 一个代表引用（用于报错定位）。"""
    by_path: dict[str, VideoRef] = {}
    video_keys = list(ds.lerobot_modality_meta.video.keys())
    for tid in ds.trajectory_ids:
        epi = int(tid)
        for vk in video_keys:
            p = ds.get_video_path(epi, vk)
            key = p.resolve().as_posix()
            if key not in by_path:
                by_path[key] = VideoRef(
                    path=key,
                    dataset_name=dataset_name,
                    episode_index=epi,
                    video_key=vk,
                )
    return by_path


def check_one_video(
    path: str,
    video_backend_kwargs: dict,
    read_first_frame: bool,
) -> tuple[str | None, str | None]:
    """
    Returns:
        (error_kind, detail) — 正常时 (None, None)
        error_kind: 'missing' | 'empty' | 'decord' | 'read_frame'
    """
    p = Path(path)
    if not p.is_file():
        return "missing", "not a file or does not exist"
    if p.stat().st_size == 0:
        return "empty", "file size 0"

    try:
        import decord
    except ImportError:
        return "decord", "decord not installed"

    try:
        vr = decord.VideoReader(path, **video_backend_kwargs)
        n = len(vr)
        if n <= 0:
            return "decord", "VideoReader reports 0 frames"
    except Exception as e:
        return "decord", repr(e)

    if read_first_frame:
        try:
            _ = vr.get_batch([0]).asnumpy()
        except Exception as e:
            return "read_frame", repr(e)

    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root_dir", type=str, required=True, help="数据集根目录（与 train_value 一致）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_mix",
        type=str,
        default=None,
        help="mixtures.DATASET_NAMED_MIXTURES 中的名称",
    )
    group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="相对 data_root_dir 的单个子数据集目录名，例如 rl_offline_ee3/blocks_ranking_size",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default=None,
        help="与 --dataset_name 联用；省略时仅允许使用 --data_mix",
    )
    parser.add_argument("--lerobot_version", type=str, default="v2.0", choices=("v2.0", "v3.0"))
    parser.add_argument(
        "--video_backend_kwargs_json",
        type=str,
        default=None,
        help='传给 decord.VideoReader 的 JSON 对象，例如 "{}"',
    )
    parser.add_argument(
        "--read_first_frame",
        action="store_true",
        help="在成功打开 VideoReader 后再读第 0 帧（更慢）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="将问题视频列表写入 JSON",
    )
    parser.add_argument("--quiet", action="store_true", help="减少进度条与过程输出")
    args = parser.parse_args()

    data_root = Path(args.data_root_dir)
    if not data_root.is_dir():
        print(f"ERROR: data_root_dir is not a directory: {data_root}", file=sys.stderr)
        return 2

    vb_kwargs: dict = {}
    if args.video_backend_kwargs_json:
        vb_kwargs = json.loads(args.video_backend_kwargs_json)

    data_cfg = {"lerobot_version": args.lerobot_version}

    if args.data_mix:
        if args.data_mix not in DATASET_NAMED_MIXTURES:
            print(f"ERROR: unknown data_mix '{args.data_mix}'", file=sys.stderr)
            return 2
        spec = _dedupe_mixture(DATASET_NAMED_MIXTURES[args.data_mix])
        to_build: list[tuple[str, str]] = [(d_name, robot_type) for d_name, _, robot_type in spec]
    else:
        if not args.robot_type:
            print("ERROR: --dataset_name requires --robot_type", file=sys.stderr)
            return 2
        to_build = [(args.dataset_name, args.robot_type)]

    all_refs: dict[str, VideoRef] = {}
    for d_name, robot_type in to_build:
        ds_path = data_root / d_name
        if not ds_path.is_dir():
            print(f"WARN: skip missing dataset path: {ds_path}", file=sys.stderr)
            continue
        if not args.quiet:
            print(f"Loading metadata: {d_name} ({robot_type}) ...")
        try:
            ds = make_LeRobotSingleDataset(
                data_root_dir=data_root,
                data_name=d_name,
                robot_type=robot_type,
                delete_pause_frame=False,
                data_cfg=data_cfg,
            )
        except Exception as e:
            print(f"ERROR: failed to open dataset {d_name}: {e}", file=sys.stderr)
            return 1
        sub = collect_unique_video_paths(d_name, ds)
        for k, ref in sub.items():
            if k not in all_refs:
                all_refs[k] = ref

    if not all_refs:
        print("No video paths collected (datasets missing or empty).", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Unique video files to check: {len(all_refs)}")

    bad: list[dict] = []
    iterator: Iterable[str] = all_refs.keys()
    if not args.quiet:
        iterator = tqdm(sorted(all_refs.keys()), desc="decord check")

    for path in iterator:
        ref = all_refs[path]
        kind, detail = check_one_video(path, vb_kwargs, args.read_first_frame)
        if kind is not None:
            bad.append(
                {
                    "kind": kind,
                    "detail": detail,
                    "path": path,
                    "dataset_name": ref.dataset_name,
                    "episode_index": ref.episode_index,
                    "video_key": ref.video_key,
                }
            )

    if bad:
        print(f"\nFound {len(bad)} problematic video file(s):\n")
        for row in bad:
            print(
                f"[{row['kind']}] {row['path']}\n"
                f"    dataset={row['dataset_name']} episode={row['episode_index']} key={row['video_key']}\n"
                f"    {row['detail']}\n"
            )
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(bad, f, indent=2, ensure_ascii=False)
            print(f"Wrote {args.output_json}")
        return 1

    print("All collected videos passed decord checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
