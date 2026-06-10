#!/usr/bin/env python3
"""Build frame-balanced success/failure RobotWin block-ranking datasets.

The value trainer samples frames, not episodes. For these tasks, failed
episodes are much longer than successful ones, so episode-level 1:1 balancing
would still leave the frame distribution failure-heavy. This script keeps all
successful episodes and samples failed episodes until failure frames are close
to the successful-frame count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_dir_for_episode(episode_index: int, chunks_size: int) -> str:
    return f"chunk-{episode_index // chunks_size:03d}"


def source_chunk_dir(ep: dict[str, Any], chunks_size: int) -> str:
    if ep.get("chunk_dir"):
        return str(ep["chunk_dir"])
    if ep.get("chunk_id") is not None:
        return f"chunk-{int(ep['chunk_id']):03d}"
    return chunk_dir_for_episode(int(ep["episode_index"]), chunks_size)


def steps_config_key(dataset_name: str) -> str:
    config_dict = {
        "delete_pause_frame": False,
        "dataset_name": dataset_name,
        "step_stride": 1,
        "skip_invalid_subtask_frames": False,
    }
    return hashlib.md5(str(sorted(config_dict.items())).encode()).hexdigest()[:12]


def choose_failures_by_frame_count(
    failures: list[dict[str, Any]],
    target_frames: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    shuffled = list(failures)
    rng.shuffle(shuffled)

    chosen: list[dict[str, Any]] = []
    current = 0
    for ep in shuffled:
        prev = current
        chosen.append(ep)
        current += int(ep["length"])
        if current >= target_frames:
            without_last = chosen[:-1]
            if without_last and abs(prev - target_frames) < abs(current - target_frames):
                return without_last
            return chosen
    return chosen


def to_2d_array(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(series.to_numpy()).astype(np.float64)
    return series.to_numpy(dtype=np.float64).reshape(-1, 1)


def compute_stats(frames: dict[str, list[np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    stats: dict[str, dict[str, list[float]]] = {}
    for key, chunks in frames.items():
        arr = np.concatenate(chunks, axis=0)
        stats[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


def copy_dataset(
    src: Path,
    dst: Path,
    selected: list[dict[str, Any]],
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    info = read_json(src / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "meta" / "modality.json", dst / "meta" / "modality.json")
    shutil.copy2(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")

    new_episodes: list[dict[str, Any]] = []
    steps: list[tuple[int, int]] = []
    stats_frames: dict[str, list[np.ndarray]] = {}
    global_index = 0

    for new_ep_idx, ep in enumerate(tqdm(selected, desc=f"copy {dst.name}")):
        old_ep_idx = int(ep["episode_index"])
        length = int(ep["length"])
        old_chunk = source_chunk_dir(ep, chunks_size)
        new_chunk = chunk_dir_for_episode(new_ep_idx, chunks_size)

        old_parquet = src / "data" / old_chunk / f"episode_{old_ep_idx:06d}.parquet"
        new_parquet = dst / "data" / new_chunk / f"episode_{new_ep_idx:06d}.parquet"
        if not old_parquet.exists():
            raise FileNotFoundError(old_parquet)

        df = pd.read_parquet(old_parquet)
        if len(df) != length:
            raise ValueError(f"Length mismatch for episode {old_ep_idx}: meta={length}, parquet={len(df)}")

        df = df.copy()
        df["episode_index"] = new_ep_idx
        df["index"] = range(global_index, global_index + len(df))
        new_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(new_parquet, index=False)

        for column in df.columns:
            stats_frames.setdefault(column, []).append(to_2d_array(df[column]))

        for video_key in VIDEO_KEYS:
            old_video = src / "videos" / old_chunk / video_key / f"episode_{old_ep_idx:06d}.mp4"
            new_video = dst / "videos" / new_chunk / video_key / f"episode_{new_ep_idx:06d}.mp4"
            if not old_video.exists():
                raise FileNotFoundError(old_video)
            new_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_video, new_video)

        new_ep = dict(ep)
        new_ep["episode_index"] = new_ep_idx
        new_ep["chunk_id"] = new_ep_idx // chunks_size
        new_ep["chunk_dir"] = new_chunk
        new_ep["source_episode_index"] = old_ep_idx
        new_ep["balanced_source_dataset"] = src.name
        new_episodes.append(new_ep)
        steps.extend((new_ep_idx, i) for i in range(length))
        global_index += length

    total_frames = sum(int(ep["length"]) for ep in new_episodes)
    total_episodes = len(new_episodes)
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes else 0

    out_info = dict(info)
    out_info["total_episodes"] = total_episodes
    out_info["total_frames"] = total_frames
    out_info["total_videos"] = total_episodes * len(VIDEO_KEYS)
    out_info["total_chunks"] = total_chunks
    out_info["splits"] = {"train": f"0:{total_episodes}"}
    out_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    out_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_json(dst / "meta" / "info.json", out_info)
    write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)
    write_json(dst / "meta" / "stats.json", compute_stats(stats_frames))

    config_key = steps_config_key(dst.name)
    steps_cache = {
        "config_key": config_key,
        "steps": steps,
        "num_trajectories": total_episodes,
        "total_steps": len(steps),
        "source_dataset": str(src),
        "delete_pause_frame": False,
        "step_stride": 1,
        "balanced_by": "frame_count",
        "seed": seed,
    }
    for name in (f"steps_{config_key}.pkl", "steps_data_index.pkl"):
        with (dst / "meta" / name).open("wb") as f:
            pickle.dump(steps_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    success_eps = [ep for ep in new_episodes if bool(ep.get("success"))]
    failure_eps = [ep for ep in new_episodes if not bool(ep.get("success"))]
    manifest = {
        "source_dataset": str(src),
        "output_dataset": str(dst),
        "balanced_by": "frame_count",
        "seed": seed,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "success_episodes": len(success_eps),
        "failure_episodes": len(failure_eps),
        "success_frames": sum(int(ep["length"]) for ep in success_eps),
        "failure_frames": sum(int(ep["length"]) for ep in failure_eps),
        "selected_source_success_episode_indices": [int(ep["source_episode_index"]) for ep in success_eps],
        "selected_source_failure_episode_indices": [int(ep["source_episode_index"]) for ep in failure_eps],
        "copied_files_are_physical": True,
        "uses_symlink_or_hardlink": False,
    }
    write_json(dst / "source_manifest.json", manifest)
    return manifest


def build_one(source_root: Path, output_root: Path, task_name: str, seed: int, overwrite: bool) -> dict[str, Any]:
    src = source_root / task_name
    episodes = read_jsonl(src / "meta" / "episodes.jsonl")
    success = [ep for ep in episodes if bool(ep.get("success"))]
    failure = [ep for ep in episodes if not bool(ep.get("success"))]

    rng = random.Random(seed + sum(ord(c) for c in task_name))
    selected_failures = choose_failures_by_frame_count(
        failure,
        target_frames=sum(int(ep["length"]) for ep in success),
        rng=rng,
    )
    selected = list(success) + selected_failures
    rng.shuffle(selected)

    dst = output_root / f"{task_name}_success_failure_balanced"
    print(
        f"[select] {task_name}: success_eps={len(success)}, failure_eps={len(selected_failures)}, "
        f"success_frames={sum(int(ep['length']) for ep in success)}, "
        f"failure_frames={sum(int(ep['length']) for ep in selected_failures)}",
        flush=True,
    )
    return copy_dataset(src, dst, selected, seed=seed, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot"),
    )
    parser.add_argument("--tasks", nargs="+", default=["blocks_ranking_rgb", "blocks_ranking_size"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifests = [
        build_one(
            source_root=args.source_root.resolve(),
            output_root=args.output_root.resolve(),
            task_name=task_name,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        for task_name in args.tasks
    ]

    summary = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tasks": manifests,
        "total_episodes": sum(m["total_episodes"] for m in manifests),
        "total_frames": sum(m["total_frames"] for m in manifests),
        "success_episodes": sum(m["success_episodes"] for m in manifests),
        "failure_episodes": sum(m["failure_episodes"] for m in manifests),
        "success_frames": sum(m["success_frames"] for m in manifests),
        "failure_frames": sum(m["failure_frames"] for m in manifests),
        "balanced_by": "frame_count",
        "seed": args.seed,
    }
    write_json(args.output_root / "robotwin_pi05_blocks_ranking_success_failure_balanced_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
