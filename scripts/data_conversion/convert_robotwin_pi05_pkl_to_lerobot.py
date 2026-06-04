#!/usr/bin/env python3
"""Convert RobotWin pi0.5 raw pickle rollouts into one LeRobot v2 dataset.

The source pickle format is `robotwin_openpi_raw_pickle_v1` with keys:
images, states, actions, timestamps, frame_indices, task_name, instruction, success.
The output is a single multi-task LeRobot dataset that StarVLA's value trainer can
read directly through a one-entry mixture.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import hashlib
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/workspace/users/niejunnan/datasets/robotwin_rollouts/demo_clean_pi05"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/workspace/users/niejunnan/datasets/robotwin_pi05_demo_clean_qpos_rollout"
)

CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


@dataclass(frozen=True)
class EpisodeJob:
    source_path: str
    episode_index: int
    task_index: int
    output_dir: str
    chunks_size: int
    fps: int
    video_codec: str
    skip_existing: bool


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _count_pickles(run_dir: Path) -> int:
    return sum(1 for _ in run_dir.glob("*/episodes/*.pkl"))


def discover_complete_runs(source_root: Path, expected_pickles: int) -> list[Path]:
    runs: list[tuple[int, str, Path]] = []
    for run_dir in sorted(source_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        pkl_count = _count_pickles(run_dir)
        if pkl_count != expected_pickles:
            continue
        seed = int(manifest.get("seed_arg_start", 10**9))
        runs.append((seed, run_dir.name, run_dir))
    return [p for _, _, p in sorted(runs)]


def build_task_table(run_dirs: list[Path]) -> tuple[dict[str, int], dict[str, str]]:
    task_names = sorted({p.parent.parent.name for run in run_dirs for p in run.glob("*/episodes/*.pkl")})
    task_to_index = {task_name: i for i, task_name in enumerate(task_names)}
    task_to_instruction: dict[str, str] = {}

    for task_name in task_names:
        for run in run_dirs:
            matches = sorted((run / task_name / "episodes").glob("*.pkl"))
            if not matches:
                continue
            with matches[0].open("rb") as f:
                obj = pickle.load(f)
            instruction = obj.get("instruction") or obj.get("policy_task_name") or task_name
            task_to_instruction[task_name] = str(instruction)
            break
        task_to_instruction.setdefault(task_name, task_name)

    return task_to_index, task_to_instruction


def gather_jobs(
    run_dirs: list[Path],
    output_dir: Path,
    task_to_index: dict[str, int],
    chunks_size: int,
    fps: int,
    video_codec: str,
    skip_existing: bool,
    limit_episodes: int | None,
) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    episode_index = 0
    for run in run_dirs:
        for task_name in sorted(task_to_index):
            episode_dir = run / task_name / "episodes"
            if not episode_dir.exists():
                continue
            for pkl_path in sorted(episode_dir.glob("*.pkl")):
                jobs.append(
                    EpisodeJob(
                        source_path=str(pkl_path),
                        episode_index=episode_index,
                        task_index=task_to_index[task_name],
                        output_dir=str(output_dir),
                        chunks_size=chunks_size,
                        fps=fps,
                        video_codec=video_codec,
                        skip_existing=skip_existing,
                    )
                )
                episode_index += 1
                if limit_episodes is not None and len(jobs) >= limit_episodes:
                    return jobs
    return jobs


def _stats_for_array(arr: np.ndarray) -> dict[str, np.ndarray | int]:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr.astype(np.float64, copy=False)
    return {
        "count": int(arr.shape[0]),
        "sum": arr.sum(axis=0),
        "sumsq": np.square(arr).sum(axis=0),
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
    }


def _merge_stats_item(dst: dict[str, Any], src: dict[str, Any]) -> None:
    if not dst:
        dst.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in src.items()})
        return
    dst["count"] += src["count"]
    dst["sum"] += src["sum"]
    dst["sumsq"] += src["sumsq"]
    dst["min"] = np.minimum(dst["min"], src["min"])
    dst["max"] = np.maximum(dst["max"], src["max"])


def _finalize_stats(stats_acc: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for key, stat in stats_acc.items():
        count = max(1, int(stat["count"]))
        mean = stat["sum"] / count
        var = stat["sumsq"] / count - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        out[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": stat["min"].tolist(),
            "max": stat["max"].tolist(),
            # Current RobotWin qpos config uses min_max normalization. Keep q01/q99
            # present for schema compatibility; exact quantiles are not required here.
            "q01": stat["min"].tolist(),
            "q99": stat["max"].tolist(),
        }
    return out


def _write_video(video_path: Path, frames: np.ndarray, fps: int, codec: str) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames [T,H,W,3], got {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
    if not writer.isOpened() and codec != "mp4v":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def convert_one_episode(job: EpisodeJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    chunk_id = job.episode_index // job.chunks_size
    chunk_dir = f"chunk-{chunk_id:03d}"
    data_path = output_dir / "data" / chunk_dir / f"episode_{job.episode_index:06d}.parquet"

    if job.skip_existing and data_path.exists():
        raise FileExistsError(f"Skipping existing episode parquet is not enough for metadata: {data_path}")

    with open(job.source_path, "rb") as f:
        obj = pickle.load(f)

    states = np.asarray(obj["states"], dtype=np.float32)
    actions = np.asarray(obj["actions"], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"{job.source_path}: expected states [T,14], got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"{job.source_path}: expected actions [T,14], got {actions.shape}")
    length = int(obj.get("length", len(actions)))
    if length != len(states) or length != len(actions):
        raise ValueError(
            f"{job.source_path}: length mismatch length={length}, states={len(states)}, actions={len(actions)}"
        )

    timestamps = np.asarray(obj.get("timestamps", np.arange(length, dtype=np.float32) / job.fps), dtype=np.float32)
    frame_indices = np.asarray(obj.get("frame_indices", np.arange(length)), dtype=np.int64)
    if len(timestamps) != length:
        raise ValueError(f"{job.source_path}: timestamps length {len(timestamps)} != episode length {length}")
    if len(frame_indices) != length:
        raise ValueError(f"{job.source_path}: frame_indices length {len(frame_indices)} != episode length {length}")
    success = bool(obj.get("success", "success" in Path(job.source_path).name))

    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "observation.state": list(states),
            "action": list(actions),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": np.full(length, job.episode_index, dtype=np.int64),
            "index": np.arange(length, dtype=np.int64),
            "task_index": np.full(length, job.task_index, dtype=np.int64),
            "success": np.full(length, success, dtype=bool),
        }
    )
    df.to_parquet(data_path, index=False)

    images = obj["images"]
    for camera_key in CAMERA_KEYS:
        if camera_key not in images:
            raise KeyError(f"{job.source_path}: missing camera key {camera_key}")
        if len(images[camera_key]) != length:
            raise ValueError(
                f"{job.source_path}: {camera_key} has {len(images[camera_key])} frames, "
                f"expected {length}"
            )
        video_path = (
            output_dir
            / "videos"
            / chunk_dir
            / camera_key
            / f"episode_{job.episode_index:06d}.mp4"
        )
        _write_video(video_path, np.asarray(images[camera_key]), job.fps, job.video_codec)

    task_name = str(obj.get("task_name") or Path(job.source_path).parent.parent.name)
    instruction = str(obj.get("instruction") or obj.get("policy_task_name") or task_name)
    episode_meta = {
        "episode_index": job.episode_index,
        "tasks": [instruction],
        "length": length,
        "task_index": job.task_index,
        "chunk_id": chunk_id,
        "chunk_dir": chunk_dir,
        "success": success,
        "task_name": task_name,
        "seed": obj.get("seed"),
        "source_run": Path(job.source_path).parents[2].name,
        "source_path": job.source_path,
    }

    scalar_episode = np.full(length, job.episode_index, dtype=np.int64)
    scalar_task = np.full(length, job.task_index, dtype=np.int64)
    scalar_success = np.full(length, int(success), dtype=np.int64)
    stats = {
        "observation.state": _stats_for_array(states),
        "action": _stats_for_array(actions),
        "timestamp": _stats_for_array(timestamps),
        "frame_index": _stats_for_array(frame_indices),
        "episode_index": _stats_for_array(scalar_episode),
        "index": _stats_for_array(np.arange(length, dtype=np.int64)),
        "task_index": _stats_for_array(scalar_task),
        "success": _stats_for_array(scalar_success),
    }
    return {
        "episode": episode_meta,
        "stats": stats,
        "length": length,
        "success": success,
        "task_name": task_name,
    }


def write_static_metadata(
    output_dir: Path,
    task_to_index: dict[str, int],
    task_to_instruction: dict[str, str],
    fps: int,
    chunks_size: int,
    total_episodes: int,
    total_frames: int,
    video_codec: str,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with (meta_dir / "tasks.jsonl").open("w") as f:
        for task_name, task_index in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            row = {
                "task_index": task_index,
                "task": task_to_instruction.get(task_name, task_name),
                "task_name": task_name,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    modality = {
        "state": {
            "left_arm": {"original_key": "observation.state", "start": 0, "end": 6, "dtype": "float32"},
            "left_gripper": {"original_key": "observation.state", "start": 6, "end": 7, "dtype": "float32"},
            "right_arm": {"original_key": "observation.state", "start": 7, "end": 13, "dtype": "float32"},
            "right_gripper": {"original_key": "observation.state", "start": 13, "end": 14, "dtype": "float32"},
        },
        "action": {
            "left_arm": {"original_key": "action", "start": 0, "end": 6, "dtype": "float32"},
            "left_gripper": {"original_key": "action", "start": 6, "end": 7, "dtype": "float32"},
            "right_arm": {"original_key": "action", "start": 7, "end": 13, "dtype": "float32"},
            "right_gripper": {"original_key": "action", "start": 13, "end": 14, "dtype": "float32"},
        },
        "video": {
            "image_high": {"original_key": "observation.images.cam_high"},
            "image_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "image_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    with (meta_dir / "modality.json").open("w") as f:
        json.dump(modality, f, indent=4)

    video_feature = {
        "dtype": "video",
        "shape": [3, 240, 320],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 240,
            "video.width": 320,
            "video.codec": video_codec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [JOINT_NAMES]},
        "action": {"dtype": "float32", "shape": [14], "names": [JOINT_NAMES]},
        "observation.images.cam_high": video_feature,
        "observation.images.cam_left_wrist": video_feature,
        "observation.images.cam_right_wrist": video_feature,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "success": {"dtype": "bool", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_episodes * len(CAMERA_KEYS),
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)


def write_steps_cache(output_dir: Path, episodes: list[dict[str, Any]]) -> None:
    steps: list[tuple[int, int]] = []
    for ep in sorted(episodes, key=lambda row: int(row["episode_index"])):
        episode_index = int(ep["episode_index"])
        for i in range(int(ep["length"])):
            steps.append((episode_index, i))
    config_dict = {
        "delete_pause_frame": False,
        "dataset_name": output_dir.name,
        "step_stride": 1,
        "skip_invalid_subtask_frames": False,
    }
    config_key = hashlib.md5(str(sorted(config_dict.items())).encode()).hexdigest()[:12]
    payload = {
        "config_key": config_key,
        "steps": steps,
        "num_trajectories": len(episodes),
        "total_steps": len(steps),
        "computed_timestamp": pd.Timestamp.now().isoformat(),
        "delete_pause_frame": False,
        "step_stride": 1,
    }
    for name in (f"steps_{config_key}.pkl", "steps_data_index.pkl"):
        with (output_dir / "meta" / name).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_dirs", nargs="*", default=None, help="Specific run directory names under source_root.")
    parser.add_argument("--expected_pickles_per_run", type=int, default=5000)
    parser.add_argument("--chunks_size", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video_codec", type=str, default="mp4v")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_dir = args.output_dir

    if args.run_dirs:
        run_dirs = [source_root / name for name in args.run_dirs]
    else:
        run_dirs = discover_complete_runs(source_root, args.expected_pickles_per_run)

    if not run_dirs:
        raise RuntimeError(f"No complete run dirs found under {source_root}")
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    task_to_index, task_to_instruction = build_task_table(run_dirs)
    jobs = gather_jobs(
        run_dirs=run_dirs,
        output_dir=output_dir,
        task_to_index=task_to_index,
        chunks_size=args.chunks_size,
        fps=args.fps,
        video_codec=args.video_codec,
        skip_existing=False,
        limit_episodes=args.limit_episodes,
    )
    if not jobs:
        raise RuntimeError("No episode jobs to convert")

    print(f"source_root={source_root}")
    print("run_dirs=" + ",".join(run.name for run in run_dirs))
    print(f"output_dir={output_dir}")
    print(f"tasks={len(task_to_index)} episodes={len(jobs)} workers={args.num_workers}")

    episodes: list[dict[str, Any]] = []
    stats_acc: dict[str, dict[str, Any]] = {}
    success_count = 0
    failure_count = 0
    total_frames = 0
    per_task_counts = {task_name: {"episodes": 0, "success": 0, "failure": 0} for task_name in task_to_index}

    max_workers = max(1, int(args.num_workers))
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(convert_one_episode, job): job for job in jobs}
        for fut in tqdm(futures.as_completed(future_to_job), total=len(future_to_job), desc="Converting"):
            result = fut.result()
            episodes.append(result["episode"])
            total_frames += int(result["length"])
            if result["success"]:
                success_count += 1
                per_task_counts[result["task_name"]]["success"] += 1
            else:
                failure_count += 1
                per_task_counts[result["task_name"]]["failure"] += 1
            per_task_counts[result["task_name"]]["episodes"] += 1
            for key, stat in result["stats"].items():
                _merge_stats_item(stats_acc.setdefault(key, {}), stat)

    episodes.sort(key=lambda row: int(row["episode_index"]))
    with (output_dir / "meta" / "episodes.jsonl").open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False, default=_json_default) + "\n")

    stats = _finalize_stats(stats_acc)
    with (output_dir / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=4)

    write_static_metadata(
        output_dir=output_dir,
        task_to_index=task_to_index,
        task_to_instruction=task_to_instruction,
        fps=args.fps,
        chunks_size=args.chunks_size,
        total_episodes=len(episodes),
        total_frames=total_frames,
        video_codec=args.video_codec,
    )
    write_steps_cache(output_dir, episodes)

    source_manifest = {
        "format": "robotwin_pi05_demo_clean_qpos_rollout_lerobot_v2",
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "run_dirs": [str(run) for run in run_dirs],
        "total_tasks": len(task_to_index),
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "success": success_count,
        "failure": failure_count,
        "per_task_counts": per_task_counts,
        "camera_keys": CAMERA_KEYS,
        "state_action_order": "left_arm[0:6], left_gripper[6:7], right_arm[7:13], right_gripper[13:14]",
    }
    with (output_dir / "source_manifest.json").open("w") as f:
        json.dump(source_manifest, f, ensure_ascii=False, indent=2, default=_json_default)

    print("Conversion complete")
    print(json.dumps(source_manifest, ensure_ascii=False, indent=2, default=_json_default)[:4000])


if __name__ == "__main__":
    # Avoid OpenCV internal threading fighting with process-level parallelism.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
