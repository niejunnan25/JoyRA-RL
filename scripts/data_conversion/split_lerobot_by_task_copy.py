#!/usr/bin/env python3
"""Split a LeRobot dataset into one physical dataset per task_name.

Each output task directory is a standalone LeRobot dataset. Parquet files are
rewritten with sequential episode_index/index values and videos are copied to
matching sequential episode names. No symlinks or hardlinks are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import shutil
from collections import defaultdict
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


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        raise ValueError(f"Invalid empty task folder name from {name!r}")
    return cleaned


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


def copy_common_metadata(src: Path, dst: Path, task_rows: list[dict[str, Any]]) -> None:
    shutil.copy2(src / "meta" / "modality.json", dst / "meta" / "modality.json")
    shutil.copy2(src / "meta" / "stats.json", dst / "meta" / "stats.json")
    write_jsonl(dst / "meta" / "tasks.jsonl", task_rows)


def build_task_dataset(
    src: Path,
    output_root: Path,
    task_name: str,
    episodes: list[dict[str, Any]],
    task_rows_by_index: dict[int, dict[str, Any]],
    info_template: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    folder_name = safe_name(task_name)
    dst = output_root / folder_name
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    chunks_size = int(info_template.get("chunks_size", 1000))
    task_indices = sorted({int(ep["task_index"]) for ep in episodes if ep.get("task_index") is not None})
    task_rows = [task_rows_by_index[i] for i in task_indices if i in task_rows_by_index]
    if not task_rows:
        raise ValueError(f"No task rows found for task_name={task_name}, task_indices={task_indices}")
    copy_common_metadata(src, dst, task_rows)

    total_frames = sum(int(ep["length"]) for ep in episodes)
    total_episodes = len(episodes)
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    new_episodes: list[dict[str, Any]] = []
    steps: list[tuple[int, int]] = []
    global_index = 0

    for new_ep_idx, ep in enumerate(tqdm(episodes, desc=f"copy {folder_name}", leave=False)):
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

    info = dict(info_template)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes * len(VIDEO_KEYS)
    info["total_chunks"] = total_chunks
    info["total_tasks"] = len(task_rows)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_json(dst / "meta" / "info.json", info)
    write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)

    config_key = steps_config_key(folder_name)
    cache_data = {
        "config_key": config_key,
        "steps": steps,
        "num_trajectories": total_episodes,
        "total_steps": len(steps),
        "task_name": task_name,
        "source_dataset": str(src),
        "delete_pause_frame": False,
        "step_stride": 1,
    }
    for name in (f"steps_{config_key}.pkl", "steps_data_index.pkl"):
        with (dst / "meta" / name).open("wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "task_name": task_name,
        "folder_name": folder_name,
        "source_dataset": str(src),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "success": sum(1 for ep in new_episodes if bool(ep.get("success"))),
        "failure": sum(1 for ep in new_episodes if not bool(ep.get("success"))),
        "copied_files_are_physical": True,
        "uses_symlink_or_hardlink": False,
    }
    write_json(dst / "source_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.source.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    info = read_json(src / "meta" / "info.json")
    episodes = read_jsonl(src / "meta" / "episodes.jsonl")
    task_rows = read_jsonl(src / "meta" / "tasks.jsonl")
    task_rows_by_index = {int(row["task_index"]): row for row in task_rows}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        task_name = ep.get("task_name")
        if task_name is None:
            task_index = ep.get("task_index")
            row = task_rows_by_index.get(int(task_index))
            task_name = row.get("task_name") if row else f"task_{task_index}"
        grouped[str(task_name)].append(ep)

    print(f"source={src}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"tasks={len(grouped)}, episodes={len(episodes)}, frames={sum(int(ep['length']) for ep in episodes)}", flush=True)

    manifests = []
    for task_name in sorted(grouped):
        task_eps = sorted(grouped[task_name], key=lambda e: int(e["episode_index"]))
        frames = sum(int(ep["length"]) for ep in task_eps)
        success = sum(1 for ep in task_eps if bool(ep.get("success")))
        failure = len(task_eps) - success
        print(
            f"[task] {task_name}: episodes={len(task_eps)}, frames={frames}, success={success}, failure={failure}",
            flush=True,
        )
        manifest = build_task_dataset(
            src=src,
            output_root=output_root,
            task_name=task_name,
            episodes=task_eps,
            task_rows_by_index=task_rows_by_index,
            info_template=info,
            overwrite=args.overwrite,
        )
        manifests.append(manifest)

    summary = {
        "source_dataset": str(src),
        "output_root": str(output_root),
        "total_tasks": len(manifests),
        "total_episodes": sum(m["total_episodes"] for m in manifests),
        "total_frames": sum(m["total_frames"] for m in manifests),
        "tasks": manifests,
    }
    write_json(output_root / "split_by_task_manifest.json", summary)
    print("done", flush=True)


if __name__ == "__main__":
    main()
