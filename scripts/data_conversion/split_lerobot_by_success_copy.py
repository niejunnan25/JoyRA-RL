#!/usr/bin/env python3
"""Split a LeRobot dataset into physical success/failure datasets.

This creates independent dataset directories: parquet files are rewritten with
sequential episode_index/index values, and videos are copied to matching names.
No symlinks or hardlinks are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

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


def steps_config_key(dataset_name: str) -> str:
    config_dict = {
        "delete_pause_frame": False,
        "dataset_name": dataset_name,
        "step_stride": 1,
        "skip_invalid_subtask_frames": False,
    }
    return hashlib.md5(str(sorted(config_dict.items())).encode()).hexdigest()[:12]


def chunk_dir_for_episode(episode_index: int, chunks_size: int) -> str:
    return f"chunk-{episode_index // chunks_size:03d}"


def source_chunk_dir(ep: dict[str, Any], chunks_size: int) -> str:
    if ep.get("chunk_dir"):
        return str(ep["chunk_dir"])
    if ep.get("chunk_id") is not None:
        return f"chunk-{int(ep['chunk_id']):03d}"
    return chunk_dir_for_episode(int(ep["episode_index"]), chunks_size)


def copy_metadata_files(src: Path, dst: Path) -> None:
    for name in ["modality.json", "tasks.jsonl", "stats.json"]:
        shutil.copy2(src / "meta" / name, dst / "meta" / name)


def build_split(src: Path, dst: Path, episodes: list[dict[str, Any]], label: str, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    copy_metadata_files(src, dst)

    info = read_json(src / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    total_frames = sum(int(ep["length"]) for ep in episodes)
    total_episodes = len(episodes)
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes else 0

    new_episodes: list[dict[str, Any]] = []
    steps: list[tuple[int, int]] = []
    global_index = 0

    for new_ep_idx, ep in enumerate(tqdm(episodes, desc=f"copy {label}")):
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
        new_episodes.append(new_ep)
        steps.extend((new_ep_idx, i) for i in range(length))
        global_index += length

    info = dict(info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes * len(VIDEO_KEYS)
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_json(dst / "meta" / "info.json", info)
    write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)

    config_key = steps_config_key(dst.name)
    steps_cache = {
        "config_key": config_key,
        "steps": steps,
        "num_trajectories": total_episodes,
        "total_steps": len(steps),
        "split_label": label,
        "source_dataset": str(src),
        "delete_pause_frame": False,
        "step_stride": 1,
    }
    for name in (f"steps_{config_key}.pkl", "steps_data_index.pkl"):
        with (dst / "meta" / name).open("wb") as f:
            pickle.dump(steps_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {}
    src_manifest = src / "source_manifest.json"
    if src_manifest.exists():
        manifest = read_json(src_manifest)
    manifest.update(
        {
            "split_label": label,
            "source_dataset": str(src),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "success": sum(1 for ep in new_episodes if bool(ep.get("success"))),
            "failure": sum(1 for ep in new_episodes if not bool(ep.get("success"))),
            "copied_files_are_physical": True,
            "uses_symlink_or_hardlink": False,
        }
    )
    write_json(dst / "source_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--success-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.source.resolve()
    episodes = read_jsonl(src / "meta" / "episodes.jsonl")
    success = [ep for ep in episodes if bool(ep.get("success"))]
    failure = [ep for ep in episodes if not bool(ep.get("success"))]

    print(f"source={src}")
    print(f"success episodes={len(success)}, frames={sum(int(ep['length']) for ep in success)}")
    print(f"failure episodes={len(failure)}, frames={sum(int(ep['length']) for ep in failure)}")

    build_split(src, args.success_output.resolve(), success, "success", args.overwrite)
    build_split(src, args.failure_output.resolve(), failure, "failure", args.overwrite)

    print("done")


if __name__ == "__main__":
    main()
