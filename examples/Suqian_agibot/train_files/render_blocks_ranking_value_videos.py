#!/usr/bin/env python3
"""
Render RobotWin value-function videos with a Robo-Dopamine-style progress panel.

For each selected episode, the script writes an mp4 where:
  - the left panel shows the current multi-view observation;
  - the right panel shows target return_to_go and predicted V(s_t);
  - a red cursor marks the current step.

Default task:
  data_mix  = robotwin_orig_plus_offline_v2
  data_name = rl_offline_ee3/blocks_ranking_size

Example:
  cd /mnt/workspace1/users/tangyili/Projects/JoyRA-RL
  source /mnt/workspace/envs/conda3/bin/activate starVLA_1
  export PYTHONPATH=$PWD:$PYTHONPATH
  python examples/Suqian_agibot/train_files/render_blocks_ranking_value_videos.py \
    --checkpoint_path outputs/value/robotwin_orig_plus_offline_v2_20260413/qwen_value_best.pt \
    --num_videos 10 \
    --frame_interval 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
for _p in (_THIS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_episode_value_print import resolve_data_name_from_mixture  # noqa: E402
from visualize_episode_value_advantage import (  # noqa: E402
    TrainingAlignedDatasetCache,
    _valid_episode_index_set,
    build_episode_examples_training_aligned,
    collect_episode_entries,
    load_model,
    resolve_robot_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 10 blocks_ranking_size videos with value curves."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=(
            "outputs/value/robotwin_orig_plus_offline_v2_20260413/"
            "qwen_value_best.pt"
        ),
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/Suqian_agibot/train_files/starvla_value_function.yaml",
    )
    parser.add_argument("--base_vlm", type=str, default=None)
    parser.add_argument("--data_root_dir", type=str, default="/mnt/workspace/datasets")
    parser.add_argument("--data_mix", type=str, default="robotwin_orig_plus_offline_v2")
    parser.add_argument(
        "--task_name",
        type=str,
        default="rl_offline_ee3/blocks_ranking_size",
        help="Substring used to resolve the dataset from the mixture.",
    )
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help="Comma-separated episode ids. If omitted, the first --num_videos ids are used.",
    )
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Render every Nth frame, like Robo-Dopamine's frame_interval.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--bin_min", type=float, default=-1.0)
    parser.add_argument("--bin_max", type=float, default=0.0)
    parser.add_argument("--big_negative", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--success_col", type=str, default="episode_success")
    parser.add_argument(
        "--normalize_returns",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize_returns_per_task",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize_use_big_negative_in_denom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep this aligned with training. Existing robotwin *_T scripts only pass "
            "--normalize_returns_per_task, so default is False."
        ),
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_invalid_subtask_frames", action="store_true")
    parser.add_argument("--language_prefix", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/blocks_ranking_size_value_videos",
    )
    parser.add_argument(
        "--curve_width",
        type=int,
        default=760,
        help="Width of the value-curve panel. Camera views keep original video size.",
    )
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def parse_episode_indices(args: argparse.Namespace, data_name: str) -> List[int]:
    valid = sorted(_valid_episode_index_set(args.data_root_dir, data_name))
    if args.episode_indices:
        wanted = [int(x.strip()) for x in args.episode_indices.split(",") if x.strip()]
    else:
        wanted = [ep for ep in valid if ep >= args.episode_start][: args.num_videos]

    valid_set = set(valid)
    missing = [ep for ep in wanted if ep not in valid_set]
    if missing:
        raise ValueError(f"episode ids not found in {data_name}: {missing}")
    return wanted


def to_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    elif torch.is_tensor(image):
        x = image.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        arr = x.numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def fit_rgb(image: Any, size: tuple[int, int]) -> np.ndarray:
    """Resize into a fixed box without distorting the source aspect ratio."""
    arr = to_rgb_array(image)
    target_w, target_h = size
    src_h, src_w = arr.shape[:2]
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_text(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_dashed_polyline(
    canvas: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 10,
) -> None:
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = map(float, p0)
        x1, y1 = map(float, p1)
        length = max(1.0, float(np.hypot(x1 - x0, y1 - y0)))
        n = max(1, int(length // dash))
        for i in range(0, n, 2):
            a = i / n
            b = min(1.0, (i + 1) / n)
            q0 = (int(x0 + (x1 - x0) * a), int(y0 + (y1 - y0) * a))
            q1 = (int(x0 + (x1 - x0) * b), int(y0 + (y1 - y0) * b))
            cv2.line(canvas, q0, q1, color, thickness, cv2.LINE_AA)


def value_points(
    values: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    bin_min: float,
    bin_max: float,
) -> np.ndarray:
    n = len(values)
    if n <= 1:
        xs = np.array([x0], dtype=np.float32)
    else:
        xs = np.linspace(x0, x1, n, dtype=np.float32)
    denom = max(1e-6, float(bin_max - bin_min))
    ys_norm = np.clip((values.astype(np.float32) - bin_min) / denom, 0.0, 1.0)
    ys = y1 - ys_norm * (y1 - y0)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def render_curve_panel(
    rows: List[dict[str, Any]],
    idx: int,
    width: int,
    height: int,
    bin_min: float,
    bin_max: float,
) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 70, 90, width - 35, height - 120

    pred = np.array([r["value_t"] for r in rows], dtype=np.float32)
    target = np.array([r["return_to_go"] for r in rows], dtype=np.float32)
    idx = int(np.clip(idx, 0, len(rows) - 1))

    put_text(panel, "Value Progress", (30, 35), scale=0.9, thickness=2)
    put_text(panel, "blue: predicted V(s_t)   orange: target return", (30, 65), scale=0.5)

    cv2.rectangle(panel, (left, top), (right, bottom), (230, 230, 230), 1)
    for v in np.linspace(bin_min, bin_max, 5):
        y = int(bottom - ((v - bin_min) / (bin_max - bin_min)) * (bottom - top))
        cv2.line(panel, (left, y), (right, y), (235, 235, 235), 1)
        put_text(panel, f"{v:.2f}", (12, y + 5), scale=0.42, color=(80, 80, 80))

    target_pts = value_points(target, left, top, right, bottom, bin_min, bin_max)
    pred_pts = value_points(pred, left, top, right, bottom, bin_min, bin_max)
    draw_dashed_polyline(panel, target_pts, (255, 140, 0), thickness=2)
    if idx >= 1:
        cv2.polylines(panel, [pred_pts[: idx + 1]], False, (31, 119, 180), 3, cv2.LINE_AA)

    cursor_x = int(pred_pts[idx, 0])
    cv2.line(panel, (cursor_x, top), (cursor_x, bottom), (220, 40, 40), 2)
    cv2.circle(panel, tuple(pred_pts[idx]), 6, (31, 119, 180), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(target_pts[idx]), 6, (255, 140, 0), -1, cv2.LINE_AA)

    progress = 100.0 * idx / max(1, len(rows) - 1)
    cur = rows[idx]
    put_text(panel, f"step {cur['step']} / {rows[-1]['step']}  ({progress:.1f}%)", (40, height - 78), scale=0.65, thickness=2)
    put_text(panel, f"pred={cur['value_t']:.4f}   target={cur['return_to_go']:.4f}", (40, height - 48), scale=0.65, thickness=2)
    put_text(panel, f"abs_err={abs(cur['value_t'] - cur['return_to_go']):.4f}", (40, height - 20), scale=0.65, thickness=2)
    return panel


def render_observation_panel(
    images: Iterable[Any],
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    imgs = list(images)
    while len(imgs) < 3:
        imgs.append(imgs[-1] if imgs else np.zeros((224, 224, 3), dtype=np.uint8))

    margin = 18
    label_h = 24
    # Keep each camera view near 4:3. The previous implementation filled the
    # full left-panel width, which made each stacked view look like a thin strip.
    cell_h = (height - margin * 4 - label_h * 3 - 46) // 3
    cell_w = min(width - 2 * margin, int(round(cell_h * 4 / 3)))
    cell_x = (width - cell_w) // 2
    y = 56
    put_text(panel, title[:52], (margin, 34), scale=0.58, thickness=2)
    labels = ["high", "left/wrist", "right/wrist"]
    for img, label in zip(imgs[:3], labels):
        view = fit_rgb(img, (cell_w, cell_h))
        panel[y : y + cell_h, cell_x : cell_x + cell_w] = view
        cv2.rectangle(panel, (cell_x, y), (cell_x + cell_w, y + cell_h), (210, 210, 210), 1)
        put_text(panel, label, (cell_x, y + cell_h + 18), scale=0.55)
        y += cell_h + label_h + margin
    return panel


def render_observation_panel_original_size(
    images: Iterable[Any],
    title: str,
) -> np.ndarray:
    """Render camera views at their original pixel size, without resize."""
    imgs = [to_rgb_array(img) for img in images]
    while len(imgs) < 3:
        imgs.append(imgs[-1] if imgs else np.zeros((240, 320, 3), dtype=np.uint8))

    margin = 18
    label_h = 24
    title_h = 56
    labels = ["high", "left/wrist", "right/wrist"]
    view_w = max(int(img.shape[1]) for img in imgs[:3])
    panel_w = view_w + 2 * margin
    panel_h = title_h + margin + sum(int(img.shape[0]) + label_h + margin for img in imgs[:3])
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    put_text(panel, title[:52], (margin, 34), scale=0.58, thickness=2)
    y = title_h
    for img, label in zip(imgs[:3], labels):
        h, w = img.shape[:2]
        x = margin + (view_w - w) // 2
        panel[y : y + h, x : x + w] = img
        cv2.rectangle(panel, (x, y), (x + w, y + h), (210, 210, 210), 1)
        put_text(panel, label, (x, y + h + 18), scale=0.55)
        y += h + label_h + margin
    return panel


def load_raw_video_frames(
    args: argparse.Namespace,
    episode_index: int,
    step_ids: List[int],
) -> dict[int, List[np.ndarray]]:
    """Read original LeRobot mp4 frames for display only."""
    dataset_dir = Path(args.data_root_dir) / args.data_name
    meta_dir = dataset_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    episodes = []
    with (meta_dir / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    episode_rec = None
    for ep in episodes:
        if int(ep.get("episode_index", -1)) == int(episode_index):
            episode_rec = ep
            break
    if episode_rec is None:
        raise ValueError(f"episode_index={episode_index} 不存在于 {dataset_dir}")

    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // max(1, chunks_size)
    data_path = dataset_dir / info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    )
    df = pd.read_parquet(data_path)
    length = int(episode_rec.get("length", len(df)))
    if len(df) > length:
        df = df.iloc[:length].copy()
    frame_indices = (
        df["frame_index"].astype(int).tolist()
        if "frame_index" in df.columns
        else list(range(len(df)))
    )

    video_keys: list[tuple[str, str]] = []
    for key, feature in info.get("features", {}).items():
        if key.startswith("observation.images."):
            video_keys.append((key.split("observation.images.", 1)[1], key))
        elif key.startswith("observation.") and feature.get("dtype") == "video":
            video_keys.append((key.split("observation.", 1)[1], key))
    if not video_keys:
        raise ValueError(
            f"{meta_dir / 'info.json'} 未找到 observation.images.* 或 observation.* video 键"
        )

    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for display_key, video_key in video_keys:
            candidate_keys = [
                video_key,
                display_key,
                f"observation.images.{display_key}",
                f"observation.{display_key}",
            ]
            candidates = [
                dataset_dir
                / info["video_path"].format(
                    episode_chunk=episode_chunk,
                    episode_index=int(episode_index),
                    video_key=candidate_key,
                )
                for candidate_key in candidate_keys
            ]
            video_path = next((p for p in candidates if p.exists()), None)
            if video_path is None:
                raise FileNotFoundError(
                    "找不到视频，尝试过:\n" + "\n".join(str(p) for p in candidates)
                )
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {video_path}")
            captures[display_key] = cap

        frames_by_step: dict[int, List[np.ndarray]] = {}
        for step in step_ids:
            if step < 0 or step >= len(frame_indices):
                continue
            frame_idx = int(frame_indices[step])
            views: List[np.ndarray] = []
            for display_key, _video_key in video_keys:
                cap = captures[display_key]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"读取视频帧失败: key={display_key}, frame={frame_idx}")
                views.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames_by_step[int(step)] = views
        return frames_by_step
    finally:
        for cap in captures.values():
            cap.release()


def write_episode_video(
    examples: List[dict[str, Any]],
    rows: List[dict[str, Any]],
    meta: dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    frame_interval: int,
    fps: float,
    bin_min: float,
    bin_max: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step_ids = list(range(0, len(rows), max(1, int(frame_interval))))
    if step_ids[-1] != len(rows) - 1:
        step_ids.append(len(rows) - 1)

    task = Path(str(meta.get("data_name", "episode"))).name
    ep = int(meta.get("episode_index", -1))
    succ = int(bool(meta.get("success", True)))
    title = f"{task} | ep={ep} | success={succ}"

    raw_frames = load_raw_video_frames(args, ep, step_ids)
    first_obs = render_observation_panel_original_size(raw_frames[step_ids[0]], title)
    left_w = first_obs.shape[1]
    height = first_obs.shape[0]
    right_w = int(args.curve_width)
    width = left_w + right_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {output_path}")

    for idx in tqdm(step_ids, desc=f"写视频 ep{ep}", unit="frame", leave=False):
        display_images = raw_frames.get(idx)
        if display_images is None:
            obs = render_observation_panel(examples[idx]["image"], left_w, height, title)
        else:
            obs = render_observation_panel_original_size(display_images, title)
        curve = render_curve_panel(rows, idx, right_w, height, bin_min, bin_max)
        frame_rgb = np.concatenate([obs, curve], axis=1)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    args = parse_args()
    if args.data_name is None:
        args.data_name = resolve_data_name_from_mixture(
            args.data_mix,
            args.data_root_dir,
            task_name=args.task_name,
        )
    episode_ids = parse_episode_indices(args, args.data_name)

    device = "cpu"
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if args.cuda_device < 0 or args.cuda_device >= n_gpu:
            raise ValueError(f"--cuda_device={args.cuda_device} invalid; visible GPUs={n_gpu}")
        device = f"cuda:{args.cuda_device}"
        torch.cuda.set_device(args.cuda_device)

    cfg = OmegaConf.load(args.config_yaml)
    if args.base_vlm:
        cfg.framework.qwenvl.base_vlm = args.base_vlm

    print(f"[Info] device={device}")
    print(f"[Info] data_name={args.data_name}")
    print(f"[Info] episodes={episode_ids}")
    model = load_model(str(Path(args.checkpoint_path)), cfg, device)

    robot_type = resolve_robot_type(args.data_mix, args.data_name)
    ds_cache = TrainingAlignedDatasetCache(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "data_mix": args.data_mix,
        "data_name": args.data_name,
        "frame_interval": args.frame_interval,
        "videos": [],
    }

    for ep in tqdm(episode_ids, desc="episodes", disable=args.no_progress):
        examples, meta = build_episode_examples_training_aligned(
            args=args,
            episode_index=int(ep),
            robot_type=robot_type,
            ds_cache=ds_cache,
        )
        rows = collect_episode_entries(
            model=model,
            examples=examples,
            batch_size=args.batch_size,
            bin_min=args.bin_min,
            bin_max=args.bin_max,
            progress_desc=f"推理 ep{ep}",
            show_progress=not args.no_progress,
        )

        safe_task = Path(args.data_name).name
        video_path = output_dir / f"{safe_task}_ep{int(ep):04d}_value.mp4"
        json_path = output_dir / f"{safe_task}_ep{int(ep):04d}_value.json"
        write_episode_video(
            examples=examples,
            rows=rows,
            meta=meta,
            args=args,
            output_path=video_path,
            frame_interval=args.frame_interval,
            fps=args.fps,
            bin_min=args.bin_min,
            bin_max=args.bin_max,
        )

        payload = {"meta": meta, "rows": rows, "video_path": str(video_path)}
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        index_payload["videos"].append(
            {
                "episode_index": int(ep),
                "success": bool(meta.get("success", True)),
                "num_steps": len(rows),
                "video_path": str(video_path),
                "json_path": str(json_path),
            }
        )
        print(f"[Done] ep{ep}: {video_path}")

    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)
    print(f"[Done] index: {index_path}")


if __name__ == "__main__":
    main()
