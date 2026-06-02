#!/usr/bin/env python3
"""
针对单条 episode 逐帧可视化 value / advantage 曲线。

加 ``--no_advantage`` 时只画预测 ``value_t`` 与目标 ``return_to_go``（单图、无 advantage）。
``--num_episodes N`` 时从 ``--episode_index`` 起连续画 N 条轨迹（**同一张图**、每条约不同颜色；
实线为 pred、虚线为 target，仅支持 ``--no_advantage``）。

不依赖 build_value_dataloader / robot_type transforms，直接按原始 LeRobot 数据读取：
- meta/info.json / episodes.jsonl / tasks.jsonl
- videos/*.mp4 逐帧读取

优势计算逻辑参考 eval_qwen_nstep_advantage.py:
A_t = (G_t - gamma^n * G_{t+n}) + gamma^n * V_{t+n} - V_t
当 t+n 超出轨迹末尾时:
- n_step_reward = G_t
- bootstrap_value = 0

cd /mnt/workspace1/users/tangyili/Projects/JoyRA-RL && source /mnt/workspace/envs/conda3/bin/activate starVLA_1 && export PYTHONPATH=/mnt/workspace1/users/tangyili/Projects/JoyRA-RL:$PYTHONPATH && python examples/Suqian_agibot/train_files/visualize_episode_value_advantage.py   --checkpoint_path /mnt/workspace1/users/tangyili/Projects/JoyRA-RL/outputs/value/robotwin_orig_plus_offline_v2_20260413/checkpoint_step_200000.pt   --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml   --data_root_dir /mnt/workspace/datasets   --data_name robotwin_dataset/dataset_lerobot/aloha-agilex_clean_50_single_task_eepose/adjust_bottle-aloha-agilex_clean_50-50_ee   --episode_index 0   --num_episodes 50   --bin_min -1.0   --bin_max 0.0   --no_advantage   --output_png ./eval_results/episode_value_50.png   --output_json ./eval_results/episode_value_50.json

"""

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.value_targets_wrapper import (
    _with_language_prefix,
    augment_traj_df_success_for_returns,
    compute_normalized_returns_from_traj,
    compute_rewards_and_returns_from_traj,
    load_episode_success_from_jsonl,
    resolve_success_bool_from_traj_df,
)
from starVLA.training.train_value import build_value_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化单条 episode 的 value / advantage（逐帧推理）")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--base_vlm", type=str, default=None, help="可选：覆盖 config.framework.qwenvl.base_vlm")

    parser.add_argument("--data_root_dir", type=str, default="/mnt/workspace/datasets/rl_offline_ee", help="数据根目录")
    parser.add_argument(
        "--data_name",
        type=str,
        default="place_a2b_right",
        help=(
            "相对 data_root_dir 的 LeRobot 数据集子路径。Robotwin 单任务在 mixtures 中一般为 "
            "robotwin_dataset/dataset_lerobot/<task_folder>/<run_name>，勿漏前缀。"
        ),
    )

    parser.add_argument(
        "--episode_index",
        type=int,
        required=True,
        help="起始 episode id；与 --num_episodes 联用时从该 id 起连续取多条",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="连续绘制多少个 episode（默认 1）。>1 时建议配合 --no_advantage，图为多子图拼接",
    )

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n_step", type=int, default=50)
    parser.add_argument("--adv_scale", type=float, default=1.0)

    parser.add_argument("--big_negative", type=float, default=100.0)
    parser.add_argument(
        "--normalize_returns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用与 value 训练一致的归一化 return（默认开启）",
    )
    parser.add_argument(
        "--normalize_use_big_negative_in_denom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="归一化分母是否使用 H + big_negative；与 train_value 一致时默认关闭（仅 H），"
        "run_value_robotwin_with_RL_T 当前未传该开关。若训练显式加了 --normalize_use_big_negative_in_denom 再打开。",
    )
    parser.add_argument(
        "--normalize_returns_per_task",
        action="store_true",
        help="与 train_value 的 --normalize_returns_per_task 一致：按 task 内最长轨迹设 H 做 [-1,0] 归一化；"
        "robotwin 类训练一般需开此开关，否则 target 曲线与训练监督不一致。",
    )
    parser.add_argument(
        "--success_col",
        type=str,
        default="episode_success",
        help="parquet 中成功列名，与 LeRobotWithValueTarget 一致；无列时会用 episodes.jsonl 的 success 注入",
    )

    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--bin_min", type=float, default=-1.0)
    parser.add_argument("--bin_max", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--output_png", type=str, default="./eval_results/episode_value_advantage.png")
    parser.add_argument("--output_json", type=str, default="./eval_results/episode_value_advantage.json")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no_advantage",
        action="store_true",
        help="只画单条轨迹的预测 value 与 return_to_go（目标），不计算、不绘制 advantage",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, cfg: Any, device: str):
    model = build_value_model(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 仅去掉 DDP 的 module. 前缀；不可对全部 key 做 k[7:]，否则会误删
    # qwen_vl_interface / value_head 等前缀的前 7 个字符。
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Warn] missing keys: {missing}")
    if unexpected:
        print(f"[Warn] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _valid_episode_index_set(data_root_dir: str, data_name: str) -> Set[int]:
    """从 meta/episodes.jsonl 读取本数据集实际存在的 episode_index 集合。"""
    meta_path = Path(data_root_dir) / data_name / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        hint = ""
        if "robotwin_dataset" not in data_name and "aloha-agilex" in data_name:
            alt = Path(data_root_dir) / "robotwin_dataset" / "dataset_lerobot" / data_name / "meta" / "episodes.jsonl"
            hint = (
                f"\n  若你按 train_value 的 robotwin_orig_plus_offline(_v2) 等 mixture 选数据，"
                f"完整相对路径通常带前缀 `robotwin_dataset/dataset_lerobot/`。"
                f"\n  可尝试是否存在: {alt}"
            )
        raise FileNotFoundError(f"找不到 {meta_path}{hint}")
    out: Set[int] = set()
    for ep in _load_jsonl(meta_path):
        if "episode_index" in ep:
            out.add(int(ep["episode_index"]))
    return out


def build_denom_per_traj_from_dataset(dataset_dir: Path) -> Dict[int, int]:
    """
    与 LeRobotWithValueTarget._build_task_denom_map 对齐：按 task 统计 max length，
    每条轨迹 denom = max(1, max_len - 1)。
    """
    episodes_jsonl = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"找不到 {episodes_jsonl}")

    task_to_max_len: Dict[str, int] = {}
    traj_task_pairs: List[Tuple[int, str, int]] = []

    for ep in _load_jsonl(episodes_jsonl):
        traj_id = int(ep.get("episode_index", -1))
        if traj_id < 0:
            continue
        length = int(ep.get("length", 0))
        if "tasks" in ep and ep["tasks"]:
            task_id = str(ep["tasks"][0])
        elif "task" in ep:
            task_id = str(ep["task"])
        elif "task_index" in ep:
            task_id = f"idx_{ep['task_index']}"
        else:
            task_id = f"ep_{traj_id}"

        traj_task_pairs.append((traj_id, task_id, length))
        prev = task_to_max_len.get(task_id, 0)
        if length > prev:
            task_to_max_len[task_id] = length

    denom_per_traj: Dict[int, int] = {}
    for traj_id, task_id, _length in traj_task_pairs:
        max_len = task_to_max_len.get(task_id, 1)
        denom_per_traj[int(traj_id)] = max(1, max_len - 1)
    return denom_per_traj


def resolve_robot_type(data_mix: str, data_name: str) -> str:
    """从 mixture 定义解析子数据集的 robot_type（与 train_value 一致）。"""
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"data_mix '{data_mix}' 不在 DATASET_NAMED_MIXTURES 中")
    for d_name, _weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
        if d_name == data_name:
            return robot_type
    raise ValueError(f"data_name '{data_name}' 不在 mixture '{data_mix}' 中")


def _extract_images_and_language_from_transformed(
    data: Dict[str, Any],
    dataset: Any,
) -> Tuple[List[Any], str]:
    """与 LeRobotMixtureWithValueTarget 中 video/language 处理一致。"""
    prim_images: List[Any] = []
    wrist_views: List[Any] = []

    if "video" in data and not any(k.startswith("video.") for k in data.keys()):
        video_data = data["video"]
        for view_idx, video_key in enumerate(dataset.modality_keys["video"]):
            image = video_data[0, view_idx, :, :, :]
            if "wrist" not in video_key and "hand" not in video_key:
                prim_images.append(image)
            else:
                wrist_views.append(image)
    else:
        for video_key in dataset.modality_keys["video"]:
            image = data[video_key][0]
            image = Image.fromarray(image).resize((224, 224))
            if "wrist" not in video_key and "hand" not in video_key:
                prim_images.append(image)
            else:
                wrist_views.append(image)

    all_images = prim_images + wrist_views
    language = data[dataset.modality_keys["language"][0]][0]
    return all_images, language


@dataclass
class _DatasetEntry:
    base_ds: Any
    returns_pkl_path: Optional[Path]
    denom_per_traj: Optional[Dict[int, int]]
    episode_success_jsonl: Dict[int, bool]
    returns_mem: Dict[int, np.ndarray] = field(default_factory=dict)
    success_mem: Dict[int, bool] = field(default_factory=dict)
    full_pkl_loaded: bool = False


class TrainingAlignedDatasetCache:
    """
    按 data_name 缓存 LeRobotSingleDataset（transforms.eval）。
    return 按 episode 惰性加载/计算，避免 LeRobotWithValueTarget 对全库轨迹预计算导致 CPU 爆满。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._entries: Dict[str, _DatasetEntry] = {}

    def _returns_pkl_path(self, data_name: str) -> Optional[Path]:
        cache_dir = getattr(self.args, "returns_cache_dir", None)
        if not cache_dir:
            return None
        safe_d_name = data_name.replace("/", "_").replace("\\", "_")
        skip_tag = (
            "_skip_subinv"
            if getattr(self.args, "skip_invalid_subtask_frames", False)
            else ""
        )
        return Path(cache_dir) / (
            f"returns_cache_{safe_d_name}{skip_tag}_gamma{self.args.gamma}_neg"
            f"{self.args.big_negative}_success{self.args.success_col}.pkl"
        )

    def _ensure_entry(self, data_name: str, robot_type: str) -> _DatasetEntry:
        if data_name in self._entries:
            return self._entries[data_name]

        data_cfg = {
            "frame_stride": max(1, int(getattr(self.args, "frame_stride", 1))),
            "skip_invalid_subtask_frames": bool(
                getattr(self.args, "skip_invalid_subtask_frames", False)
            ),
        }
        base_ds = make_LeRobotSingleDataset(
            data_root_dir=Path(self.args.data_root_dir),
            data_name=data_name,
            robot_type=robot_type,
            delete_pause_frame=False,
            data_cfg=data_cfg,
        )
        base_ds.transforms.eval()

        pkl_path = self._returns_pkl_path(data_name)
        denom: Optional[Dict[int, int]] = None
        if self.args.normalize_returns_per_task:
            denom = build_denom_per_traj_from_dataset(base_ds.dataset_path)

        entry = _DatasetEntry(
            base_ds=base_ds,
            returns_pkl_path=pkl_path if pkl_path and pkl_path.is_file() else None,
            denom_per_traj=denom,
            episode_success_jsonl=load_episode_success_from_jsonl(base_ds.dataset_path),
            returns_mem={},
            success_mem={},
            full_pkl_loaded=False,
        )
        if entry.returns_pkl_path:
            print(f"[Info] 将按需从 returns 缓存读取: {entry.returns_pkl_path}")
        self._entries[data_name] = entry
        return entry

    def _load_full_pkl_if_needed(self, entry: _DatasetEntry) -> None:
        if entry.full_pkl_loaded or entry.returns_pkl_path is None:
            return
        t0 = time.time()
        with entry.returns_pkl_path.open("rb") as f:
            cache_data = pickle.load(f)
        entry.returns_mem.update(
            {int(k): np.asarray(v) for k, v in cache_data.get("returns_per_traj", {}).items()}
        )
        for k, v in cache_data.get("success_per_traj", {}).items():
            entry.success_mem[int(k)] = bool(v)
        entry.full_pkl_loaded = True
        print(
            f"[Info] 已加载 returns 缓存 ({len(entry.returns_mem)} 条轨迹, "
            f"{time.time() - t0:.1f}s)"
        )

    def get_episode_returns_and_success(
        self, data_name: str, robot_type: str, episode_index: int
    ) -> Tuple[np.ndarray, bool]:
        entry = self._ensure_entry(data_name, robot_type)
        ep = int(episode_index)

        if ep in entry.returns_mem:
            return entry.returns_mem[ep], entry.success_mem.get(ep, True)

        if entry.returns_pkl_path and not entry.full_pkl_loaded:
            self._load_full_pkl_if_needed(entry)
            if ep in entry.returns_mem:
                return entry.returns_mem[ep], entry.success_mem.get(ep, True)

        traj_df = augment_traj_df_success_for_returns(
            entry.base_ds.get_trajectory_data(ep),
            ep,
            self.args.success_col,
            entry.episode_success_jsonl,
        )
        denom = None
        if self.args.normalize_returns_per_task and entry.denom_per_traj:
            denom = entry.denom_per_traj.get(ep)

        use_norm = bool(self.args.normalize_returns or self.args.normalize_returns_per_task)
        if use_norm:
            _, returns = compute_normalized_returns_from_traj(
                traj_df,
                success_col=self.args.success_col,
                big_negative=self.args.big_negative,
                denom=denom,
                use_big_negative_in_denom=self.args.normalize_use_big_negative_in_denom,
                gamma=self.args.gamma,
            )
        else:
            _, returns = compute_rewards_and_returns_from_traj(
                traj_df,
                success_col=self.args.success_col,
                gamma=self.args.gamma,
                big_negative=self.args.big_negative,
            )

        resolved = resolve_success_bool_from_traj_df(traj_df, self.args.success_col)
        success_bool = True if resolved is None else bool(resolved)

        entry.returns_mem[ep] = returns
        entry.success_mem[ep] = success_bool
        return returns, success_bool

    def get_base_ds(self, data_name: str, robot_type: str) -> Any:
        return self._ensure_entry(data_name, robot_type).base_ds


def build_episode_examples_training_aligned(
    args: argparse.Namespace,
    episode_index: int,
    robot_type: str,
    ds_cache: TrainingAlignedDatasetCache,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    与 train_value / eval_qwen_nstep_advantage 一致：
    get_step_data -> transforms.eval()（ResizePad，无 ColorJitter）-> predict；
    value_target 按单条 episode 计算（或从训练 returns 缓存读取），不全库预计算。
    """
    base_ds = ds_cache.get_base_ds(args.data_name, robot_type)
    returns_traj, success_bool = ds_cache.get_episode_returns_and_success(
        args.data_name, robot_type, episode_index
    )
    length = len(returns_traj)

    lang_prefix = getattr(args, "language_prefix", None)
    if isinstance(lang_prefix, str):
        lang_prefix = lang_prefix.strip() or None

    short_task = Path(args.data_name).name
    examples: List[Dict[str, Any]] = []
    step_iter = tqdm(
        range(length),
        desc=f"解码 {short_task} ep{episode_index}",
        unit="frame",
        leave=False,
        disable=bool(getattr(args, "no_progress", False)),
    )
    for step in step_iter:
        raw_data = base_ds.get_step_data(episode_index, step)
        data = base_ds.transforms(raw_data)
        all_images, language = _extract_images_and_language_from_transformed(data, base_ds)
        lang_text = _with_language_prefix(lang_prefix, language)

        examples.append(
            {
                "image": all_images,
                "lang": lang_text,
                "dataset_key": str(base_ds.dataset_path),
                "trajectory_id": int(episode_index),
                "step": int(step),
                "value_target": float(returns_traj[step]),
                "success": success_bool,
            }
        )

    meta = {
        "episode_index": int(episode_index),
        "data_name": args.data_name,
        "dataset_dir": str(base_ds.dataset_path),
        "success": success_bool,
        "length": length,
        "robot_type": robot_type,
        "training_aligned": True,
    }
    return examples, meta


def _read_frame_rgb(capture: cv2.VideoCapture, frame_index: int) -> Image.Image:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"读取视频帧失败，frame_index={frame_index}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def build_episode_examples(
    args: argparse.Namespace,
    episode_index: int,
    denom_per_traj: Optional[Dict[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_dir = Path(args.data_root_dir) / args.data_name
    meta_dir = dataset_dir / "meta"

    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    episodes = _load_jsonl(meta_dir / "episodes.jsonl")
    tasks = _load_jsonl(meta_dir / "tasks.jsonl")

    episode_rec = None
    for ep in episodes:
        if int(ep.get("episode_index", -1)) == episode_index:
            episode_rec = ep
            break
    if episode_rec is None:
        raise ValueError(f"episode_index={episode_index} 不存在")

    length = int(episode_rec.get("length", 0))
    if length <= 0:
        raise ValueError(f"episode length 非法: {length}")

    task_index = int(episode_rec.get("task_index", 0))
    task_map = {int(t["task_index"]): t["task"] for t in tasks if "task_index" in t and "task" in t}
    instruction = task_map.get(task_index, "")

    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // max(1, chunks_size)

    data_path_tpl = info["data_path"]
    video_path_tpl = info["video_path"]

    parquet_path = dataset_dir / data_path_tpl.format(episode_chunk=episode_chunk, episode_index=episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"找不到 parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if len(df) < length:
        raise ValueError(f"parquet 行数 {len(df)} < meta length {length}")
    if len(df) > length:
        df = df.iloc[:length].copy()
    frame_indices = df["frame_index"].astype(int).tolist() if "frame_index" in df.columns else list(range(length))
    if len(frame_indices) != length:
        frame_indices = frame_indices[:length]

    jsonl_succ = load_episode_success_from_jsonl(dataset_dir)
    traj_df = augment_traj_df_success_for_returns(
        df, episode_index, args.success_col, jsonl_succ
    )

    use_norm = bool(args.normalize_returns or args.normalize_returns_per_task)
    denom: Optional[int] = None
    if args.normalize_returns_per_task and denom_per_traj is not None:
        denom = denom_per_traj.get(episode_index)

    if use_norm:
        _, returns_arr = compute_normalized_returns_from_traj(
            traj_df,
            success_col=args.success_col,
            big_negative=args.big_negative,
            denom=denom,
            use_big_negative_in_denom=args.normalize_use_big_negative_in_denom,
            gamma=args.gamma,
        )
    else:
        _, returns_arr = compute_rewards_and_returns_from_traj(
            traj_df,
            success_col=args.success_col,
            gamma=args.gamma,
            big_negative=args.big_negative,
        )

    resolved_succ = resolve_success_bool_from_traj_df(traj_df, args.success_col)
    success_bool = True if resolved_succ is None else resolved_succ

    video_keys = []
    for k in info.get("features", {}).keys():
        if k.startswith("observation.images."):
            video_keys.append(k.split("observation.images.", 1)[1])

    if not video_keys:
        raise ValueError("meta/info.json 未找到 observation.images.* 视频键")

    captures: Dict[str, cv2.VideoCapture] = {}
    try:
        for key in video_keys:
            # 兼容两种目录命名：
            # 1) .../{video_key}/episode_xxx.mp4，其中 video_key=cam_high
            # 2) .../{video_key}/episode_xxx.mp4，其中 video_key=observation.images.cam_high
            candidate_keys = [key, f"observation.images.{key}"]
            candidate_paths = [
                dataset_dir
                / video_path_tpl.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=k,
                )
                for k in candidate_keys
            ]

            vpath = None
            for p in candidate_paths:
                if p.exists():
                    vpath = p
                    break

            if vpath is None:
                tried = "\n".join([str(p) for p in candidate_paths])
                raise FileNotFoundError(f"找不到视频，尝试过:\n{tried}")

            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {vpath}")
            captures[key] = cap

        examples: List[Dict[str, Any]] = []
        short_task = Path(args.data_name).name
        for step, frame_idx in tqdm(
            enumerate(frame_indices),
            total=len(frame_indices),
            desc=f"读视频 {short_task} ep{episode_index}",
            unit="frame",
            leave=False,
            disable=bool(getattr(args, "no_progress", False)),
        ):
            images = [_read_frame_rgb(captures[k], frame_idx) for k in video_keys]
            examples.append(
                {
                    "image": images,
                    "lang": instruction,
                    "dataset_key": str(dataset_dir),
                    "trajectory_id": int(episode_index),
                    "step": int(step),
                    "value_target": float(returns_arr[step]),
                    "success": bool(success_bool),
                }
            )

        meta = {
            "episode_index": int(episode_index),
            "dataset_dir": str(dataset_dir),
            "task_index": task_index,
            "instruction": instruction,
            "success": success_bool,
            "success_jsonl": bool(episode_rec.get("success", True)),
            "length": length,
            "video_keys": video_keys,
            "denom_H": denom,
            "normalize_returns_per_task": bool(args.normalize_returns_per_task),
        }
        return examples, meta
    finally:
        for cap in captures.values():
            cap.release()


def compute_n_step_reward(cur_return: float, future_return: Optional[float], gamma: float, n_step: int) -> float:
    if future_return is None:
        return cur_return
    return cur_return - (gamma**n_step) * future_return


def collect_episode_entries(
    model,
    examples: List[Dict[str, Any]],
    batch_size: int,
    bin_min: float,
    bin_max: float,
    progress_desc: Optional[str] = "GPU 推理",
    *,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bs = max(1, int(batch_size))
    batch_starts = range(0, len(examples), bs)

    with torch.no_grad():
        for i in tqdm(
            batch_starts,
            desc=progress_desc or "GPU 推理",
            unit="batch",
            leave=False,
            disable=not show_progress,
        ):
            sub_batch = examples[i : i + bs]
            result = model.predict_value(examples=sub_batch, bin_min=bin_min, bin_max=bin_max)
            pred_values = result["values"]
            if isinstance(pred_values, torch.Tensor):
                pred_values = pred_values.tolist()

            for ex, pred_value in zip(sub_batch, pred_values):
                rows.append(
                    {
                        "dataset_key": str(ex.get("dataset_key", "__single_dataset__")),
                        "trajectory_id": int(ex["trajectory_id"]),
                        "step": int(ex["step"]),
                        "value_t": float(pred_value),
                        "return_to_go": float(ex.get("value_target", 0.0)),
                        "success": bool(ex.get("success", True)),
                    }
                )

    rows.sort(key=lambda x: x["step"])
    return rows


def attach_advantages(rows: List[Dict[str, Any]], gamma: float, n_step: int, adv_scale: float) -> List[Dict[str, Any]]:
    if adv_scale <= 0:
        raise ValueError(f"adv_scale must be positive, got {adv_scale}")

    step_to_row = {r["step"]: r for r in rows}
    out = []
    for r in rows:
        step = r["step"]
        future = step_to_row.get(step + n_step)

        future_return = None if future is None else future["return_to_go"]
        bootstrap_value = 0.0 if future is None else future["value_t"]
        bootstrap_weight = 0.0 if future is None else (gamma**n_step)

        n_step_reward = compute_n_step_reward(r["return_to_go"], future_return, gamma, n_step)
        advantage_raw = n_step_reward + bootstrap_weight * bootstrap_value - r["value_t"]

        out.append(
            {
                **r,
                "n_step_reward": float(n_step_reward),
                "bootstrap_value": float(bootstrap_value),
                "advantage_raw": float(advantage_raw),
                "advantage": float(advantage_raw / adv_scale),
            }
        )

    return out


def plot_curves(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    output_png: Path,
    dpi: int,
    include_advantage: bool,
):
    steps = [r["step"] for r in rows]
    values = [r["value_t"] for r in rows]
    returns = [r["return_to_go"] for r in rows]

    task_line = meta.get("dataset_dir", "")
    tid = rows[0]["trajectory_id"] if rows else -1
    succ = meta.get("success")
    title = f"Task: {task_line} | Trajectory: {tid} | Success: {succ}"

    if not include_advantage:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(
            steps,
            values,
            label="value_t (pred)",
            linewidth=1.8,
            color="#1f77b4",
            marker="s",
            markersize=3,
        )
        ax.plot(
            steps,
            returns,
            label="return_to_go (target)",
            linewidth=1.4,
            color="#ff7f0e",
            alpha=0.9,
            marker="o",
            markersize=3,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=dpi)
        plt.close(fig)
        return

    advs = [r["advantage"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(steps, values, label="V(s_t)", linewidth=1.8, color="#1f77b4")
    axes[0].plot(steps, returns, label="G_t (return_to_go)", linewidth=1.4, color="#ff7f0e", alpha=0.9)
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, advs, label="A_t", linewidth=1.8, color="#2ca02c")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("advantage")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        "Episode Value / Advantage\n"
        f"trajectory_id={rows[0]['trajectory_id']} | success={meta.get('success')} | task={meta.get('task_index')}",
        fontsize=12,
    )
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def plot_curves_multi_value_only(
    episode_blocks: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    output_png: Path,
    dpi: int,
) -> None:
    """多条 episode：单坐标轴，每条轨迹一种颜色；实线=pred，虚线=target（同色）。"""
    n = len(episode_blocks)
    task_line = episode_blocks[0][1].get("dataset_dir", "") if episode_blocks else ""

    if n <= 10:
        colors = [plt.cm.tab10(i / max(n - 1, 1)) for i in range(n)]
    elif n <= 20:
        colors = [plt.cm.tab20(i / max(n - 1, 1)) for i in range(n)]
    else:
        colors = [plt.cm.hsv(i / max(n, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (rows, meta) in enumerate(episode_blocks):
        color = colors[idx]
        steps = [x["step"] for x in rows]
        values = [x["value_t"] for x in rows]
        returns = [x["return_to_go"] for x in rows]
        ep_id = meta.get("episode_index", rows[0]["trajectory_id"] if rows else -1)
        succ = meta.get("success")
        ax.plot(
            steps,
            values,
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"ep {ep_id} (succ={succ})",
        )
        ax.plot(
            steps,
            returns,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)
    fig.suptitle(
        f"Task: {task_line}\n"
        "实线 = 预测 value_t，虚线 = return_to_go 目标（同色 = 同一条 episode）",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    cfg = OmegaConf.load(args.config_yaml)

    if args.base_vlm is not None:
        cfg.framework.qwenvl.base_vlm = args.base_vlm

    model_id = str(cfg.framework.qwenvl.get("base_vlm", ""))
    if model_id and (model_id.startswith("./") or model_id.startswith("../")):
        resolved = (Path(args.config_yaml).resolve().parent / model_id).resolve()
        if resolved.exists():
            cfg.framework.qwenvl.base_vlm = str(resolved)

    print(f"[Info] base_vlm = {cfg.framework.qwenvl.get('base_vlm')}")

    model = load_model(args.checkpoint_path, cfg, device)

    if args.num_episodes < 1:
        raise ValueError("--num_episodes 必须 >= 1")
    if args.num_episodes > 1 and not args.no_advantage:
        raise ValueError("绘制多条 episode 时请使用 --no_advantage（单条可同时画 advantage）")

    valid_eps = _valid_episode_index_set(args.data_root_dir, args.data_name)
    wanted = list(range(args.episode_index, args.episode_index + args.num_episodes))
    episode_indices = [ei for ei in wanted if ei in valid_eps]
    skipped = [ei for ei in wanted if ei not in valid_eps]
    if skipped:
        n_show = min(30, len(skipped))
        tail = " ..." if len(skipped) > n_show else ""
        print(
            f"[Warn] 以下 episode 在 episodes.jsonl 中不存在，已跳过（共 {len(skipped)} 个）: "
            f"{skipped[:n_show]}{tail}"
        )
    if not episode_indices:
        raise ValueError(
            f"请求的区间 [{args.episode_index}, {args.episode_index + args.num_episodes}) "
            f"内没有有效 episode；数据集 episode_index 范围约为 "
            f"[{min(valid_eps)}, {max(valid_eps)}]（共 {len(valid_eps)} 条）"
        )
    if len(episode_indices) < args.num_episodes:
        print(
            f"[Info] 请求 {args.num_episodes} 条，实际可画 {len(episode_indices)} 条（已排除不存在的 id）"
        )

    dataset_dir = Path(args.data_root_dir) / args.data_name
    denom_per_traj: Optional[Dict[int, int]] = None
    if args.normalize_returns_per_task:
        denom_per_traj = build_denom_per_traj_from_dataset(dataset_dir)
        if episode_indices:
            ei0 = episode_indices[0]
            print(
                f"[Info] normalize_returns_per_task: 已构建 denom(H) 表，示例 ep {ei0} 的 denom={denom_per_traj.get(ei0)}"
            )
    elif args.normalize_returns:
        print(
            "[Info] 使用 per-episode 归一化 target；若训练时开了 --normalize_returns_per_task，"
            "请对本脚本同样加上该参数以对齐虚线目标。"
        )

    episode_blocks: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []

    for ei in episode_indices:
        examples, meta = build_episode_examples(args, ei, denom_per_traj=denom_per_traj)
        print(
            f"[Info] episode={ei}, steps={len(examples)}, success={meta.get('success')}, "
            f"normalize_returns={args.normalize_returns}, "
            f"normalize_returns_per_task={args.normalize_returns_per_task}, "
            f"denom_H={meta.get('denom_H')}, "
            f"normalize_use_big_negative_in_denom={args.normalize_use_big_negative_in_denom}"
        )

        rows = collect_episode_entries(
            model=model,
            examples=examples,
            batch_size=args.batch_size,
            bin_min=args.bin_min,
            bin_max=args.bin_max,
        )

        if not rows:
            raise ValueError(f"episode {ei} 没有可用帧")

        if not args.no_advantage:
            rows = attach_advantages(rows, gamma=args.gamma, n_step=args.n_step, adv_scale=args.adv_scale)

        episode_blocks.append((rows, meta))

    output_png = Path(args.output_png)
    output_json = Path(args.output_json)

    if len(episode_blocks) == 1:
        rows_0, meta_0 = episode_blocks[0]
        plot_curves(
            rows_0,
            meta=meta_0,
            output_png=output_png,
            dpi=args.dpi,
            include_advantage=not args.no_advantage,
        )
    else:
        plot_curves_multi_value_only(episode_blocks, output_png=output_png, dpi=args.dpi)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    if len(episode_blocks) == 1:
        rows_0, meta_0 = episode_blocks[0]
        payload: Dict[str, Any] = {
            "episode_index": args.episode_index,
            "num_steps": len(rows_0),
            "meta": meta_0,
            "frames": rows_0,
            "no_advantage": bool(args.no_advantage),
        }
        if not args.no_advantage:
            payload["n_step"] = args.n_step
            payload["gamma"] = args.gamma
            payload["adv_scale"] = args.adv_scale
    else:
        payload = {
            "episode_indices": episode_indices,
            "skipped_episode_indices": skipped,
            "no_advantage": True,
            "episodes": [
                {
                    "episode_index": meta["episode_index"],
                    "num_steps": len(rows),
                    "meta": {k: v for k, v in meta.items()},
                    "frames": rows,
                }
                for rows, meta in episode_blocks
            ],
        }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved plot to: {output_png}")
    print(f"[Done] Saved per-frame values to: {output_json}")


if __name__ == "__main__":
    main()
