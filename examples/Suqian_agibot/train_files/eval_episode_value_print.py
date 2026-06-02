#!/usr/bin/env python3
"""
按 episode 评估 value 模型，在终端打印每条轨迹的指标表与汇总统计。

只需提供 checkpoint 路径时，会从父目录名推断 data_mix（例如
``robotwin_orig_plus_offline_v2``），并在该 mixture 的**全部子数据集**上按权重
随机抽 10 条 episode（5×2 图，与训练混采一致：dataset 权重 × 轨迹长度，无放回）。

最简用法（``robotwin_orig_plus_offline_v2`` 内 clean_50 + rl_offline_ee3 等混采 10 条）::

  python examples/Suqian_agibot/train_files/eval_episode_value_print.py \\
    --checkpoint_path outputs/value/robotwin_orig_plus_offline_v2_20260413/checkpoint_step_200000.pt

只评 mixture 内某一个任务（连续 ep 0–24，不混采）::

  --episode_source single --task_name adjust_bottle

列出 mixture 内所有可用数据集后退出::

  --list_mixture_datasets

默认与 train_value / run_value_*_T.sh 一致：LeRobotSingleDataset + transforms.eval()
（ResizePad 224，无 ColorJitter）；可用 --legacy_raw_video 回退到 cv2 直读原视频。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_TRAIN_FILES = Path(__file__).resolve().parent
if str(_TRAIN_FILES) not in sys.path:
    sys.path.insert(0, str(_TRAIN_FILES))

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES  # noqa: E402

from visualize_episode_value_advantage import (  # noqa: E402
    TrainingAlignedDatasetCache,
    _load_jsonl,
    _valid_episode_index_set,
    build_denom_per_traj_from_dataset,
    build_episode_examples,
    build_episode_examples_training_aligned,
    collect_episode_entries,
    load_model,
    resolve_robot_type,
)

_DEFAULT_CONFIG = "examples/Suqian_agibot/train_files/starvla_value_function.yaml"


def infer_data_mix_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """从 ``outputs/value/{data_mix}_{YYYYMMDD}/checkpoint_*.pt`` 推断 mixture 名。"""
    run_dir = Path(checkpoint_path).resolve().parent.name
    keys = sorted(DATASET_NAMED_MIXTURES.keys(), key=len, reverse=True)
    for key in keys:
        if run_dir == key or run_dir.startswith(f"{key}_"):
            return key
    return None


def list_existing_mixture_datasets(
    data_mix: str,
    data_root_dir: str,
) -> List[Tuple[int, str]]:
    """返回 mixture 内在 data_root 下存在的 (index, data_name) 列表。"""
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"data_mix '{data_mix}' 不在 DATASET_NAMED_MIXTURES 中；"
            f"可用: {sorted(DATASET_NAMED_MIXTURES.keys())[:8]}..."
        )
    root = Path(data_root_dir)
    out: List[Tuple[int, str]] = []
    for idx, (d_name, _w, _robot) in enumerate(DATASET_NAMED_MIXTURES[data_mix]):
        if (root / d_name).is_dir():
            out.append((idx, d_name))
    return out


def _episode_length_map(data_root_dir: str, data_name: str) -> Dict[int, int]:
    meta_path = Path(data_root_dir) / data_name / "meta" / "episodes.jsonl"
    out: Dict[int, int] = {}
    for ep in _load_jsonl(meta_path):
        if "episode_index" in ep:
            out[int(ep["episode_index"])] = max(1, int(ep.get("length", 1)))
    return out


def _short_task_label(data_name: str) -> str:
    return data_name.rstrip("/").split("/")[-1]


def sample_episodes_from_mixture(
    data_mix: str,
    data_root_dir: str,
    num_episodes: int,
    seed: int = 42,
    balance_trajectory: bool = True,
) -> List[Tuple[str, int]]:
    """
    在 mixture 全部子数据集上按权重随机抽 episode（无放回）。

    权重与 LeRobotMixtureDataset 的边际一致：P(ep) ∝ mixture_weight × episode_length。
    """
    spec = DATASET_NAMED_MIXTURES[data_mix]
    root = Path(data_root_dir)
    pool: List[Tuple[str, int, float]] = []

    for d_name, weight, _robot in spec:
        if not (root / d_name).is_dir():
            continue
        try:
            ep_ids = sorted(_valid_episode_index_set(data_root_dir, d_name))
        except FileNotFoundError:
            continue
        if not ep_ids:
            continue
        lengths = _episode_length_map(data_root_dir, d_name) if balance_trajectory else {}
        for ep in ep_ids:
            w = float(weight)
            if balance_trajectory:
                w *= float(lengths.get(ep, 1))
            pool.append((d_name, ep, w))

    if not pool:
        raise FileNotFoundError(
            f"data_mix={data_mix} 在 {data_root_dir} 下没有可用的 episode，请检查数据路径"
        )

    weights = np.array([p[2] for p in pool], dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum()

    rng = np.random.default_rng(seed)
    n = min(int(num_episodes), len(pool))
    chosen = rng.choice(len(pool), size=n, replace=False, p=weights)
    jobs = [(pool[i][0], pool[i][1]) for i in chosen]
    print(
        f"[Info] mixture 混采: 从 {len(pool)} 条候选 episode 中无放回抽取 {n} 条 "
        f"(seed={seed}, balance_trajectory={balance_trajectory})"
    )
    return jobs


def resolve_episode_source(args: argparse.Namespace) -> str:
    if args.episode_source != "auto":
        return args.episode_source
    if args.data_name or args.task_name:
        return "single"
    if args.data_mix:
        return "mixture"
    return "single"


def resolve_data_name_from_mixture(
    data_mix: str,
    data_root_dir: str,
    mixture_dataset_index: int = 0,
    task_name: Optional[str] = None,
) -> str:
    existing = list_existing_mixture_datasets(data_mix, data_root_dir)
    if not existing:
        raise FileNotFoundError(
            f"data_mix={data_mix} 在 {data_root_dir} 下没有找到任何数据集目录，"
            f"请检查 --data_root_dir 或手动指定 --data_name"
        )

    if task_name:
        matched = [(i, n) for i, n in existing if task_name in n]
        if not matched:
            names = "\n  ".join(n for _, n in existing[:20])
            raise ValueError(
                f"--task_name '{task_name}' 在 mixture '{data_mix}' 的现有数据集中无匹配。\n"
                f"已有（前 20）:\n  {names}"
            )
        if len(matched) > 1:
            print(
                f"[Warn] task_name '{task_name}' 匹配到 {len(matched)} 个数据集，使用第一个: {matched[0][1]}"
            )
        return matched[0][1]

    for idx, name in existing:
        if idx == mixture_dataset_index:
            return name

    raise IndexError(
        f"mixture_dataset_index={mixture_dataset_index} 越界；"
        f"mixture '{data_mix}' 在磁盘上仅有 {len(existing)} 个数据集 (index 0..{len(existing)-1})"
    )


def resolve_run_configuration(args: argparse.Namespace) -> argparse.Namespace:
    ckpt = Path(args.checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")

    run_dir = ckpt.parent

    if args.data_mix is None:
        args.data_mix = infer_data_mix_from_checkpoint(str(ckpt))
        if args.data_mix:
            print(f"[Info] 从目录名推断 data_mix={args.data_mix}")
        else:
            print(
                f"[Warn] 无法从 {run_dir.name} 推断 data_mix，"
                "请显式传 --data_mix 或 --data_name"
            )

    if args.list_mixture_datasets:
        if not args.data_mix:
            raise ValueError("--list_mixture_datasets 需要可推断或显式指定的 --data_mix")
        existing = list_existing_mixture_datasets(args.data_mix, args.data_root_dir)
        print(f"[Mixture] {args.data_mix} @ {args.data_root_dir}")
        for idx, name in existing:
            print(f"  [{idx}] {name}")
        sys.exit(0)

    args.episode_source = resolve_episode_source(args)
    print(f"[Info] episode_source={args.episode_source}")

    if args.episode_source == "single":
        if args.data_name is None:
            if not args.data_mix:
                raise ValueError("请指定 --data_name，或提供可推断 data_mix 的 checkpoint 路径")
            args.data_name = resolve_data_name_from_mixture(
                args.data_mix,
                args.data_root_dir,
                mixture_dataset_index=args.mixture_dataset_index,
                task_name=args.task_name,
            )
            print(f"[Info] 单任务数据集: {args.data_name}")
        if args.output_png is None:
            safe_task = args.data_name.replace("/", "_")
            args.output_png = str(run_dir / f"episode_value_grid_{safe_task[-80:]}.png")
    else:
        if not args.data_mix:
            raise ValueError("mixture 混采模式需要 --data_mix 或可从 checkpoint 目录推断")
        if args.num_episodes is None:
            args.num_episodes = args.plot_num_episodes
        if args.output_png is None:
            args.output_png = str(
                run_dir / f"episode_value_grid_{args.data_mix}_2x5.png"
            )

    if args.normalize_returns_per_task:
        print(
            "[Info] normalize_returns_per_task=True（与 run_value_*_T.sh 训练脚本一致）；"
            "如需关闭请显式传 --no-normalize_returns_per_task"
        )

    if args.returns_cache_dir is None and args.data_mix:
        ckpt = Path(args.checkpoint_path).resolve()
        project_root = ckpt.parent.parent.parent
        cache_candidate = project_root / "outputs" / "cache" / args.data_mix
        if cache_candidate.is_dir():
            args.returns_cache_dir = str(cache_candidate)
            print(f"[Info] 使用 returns 缓存目录: {args.returns_cache_dir}")

    if not args.legacy_raw_video:
        print(
            "[Info] 数据通路=训练同款（LeRobotSingleDataset + transforms.eval，无 ColorJitter）"
        )
    else:
        print("[Warn] --legacy_raw_video：cv2 原视频直读，pred 与训练不一致，仅适合调试")

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 episode 打印 value 模型评估指标（MAE/RMSE、首末帧 pred vs target 等）"
    )

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=_DEFAULT_CONFIG,
        help="模型配置 yaml",
    )
    parser.add_argument("--base_vlm", type=str, default=None)
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA 设备编号，例如 1 表示 cuda:1；也可用环境变量 CUDA_VISIBLE_DEVICES",
    )

    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/mnt/workspace/datasets",
        help="与 run_value_agibot_with_RL_T.sh 等训练脚本一致（默认 /mnt/workspace/datasets）",
    )
    parser.add_argument(
        "--returns_cache_dir",
        type=str,
        default=None,
        help="与训练相同的 returns 缓存目录，默认尝试 outputs/cache/<data_mix>",
    )
    parser.add_argument(
        "--legacy_raw_video",
        action="store_true",
        help="回退到 cv2 直读原视频（不经训练 transforms，pred 与训练不一致）",
    )
    parser.add_argument(
        "--language_prefix",
        type=str,
        default=None,
        help="与 train_value --language_prefix 一致（通常留空）",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default=None,
        help="mixtures.DATASET_NAMED_MIXTURES 名称；省略则从 checkpoint 父目录名推断",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="相对 data_root_dir 的单个子数据集；省略则从 data_mix 中选取",
    )
    parser.add_argument(
        "--mixture_dataset_index",
        type=int,
        default=0,
        help="data_mix 内第几个在磁盘上存在的数据集（默认 0）",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="在 mixture 各 data_name 子串匹配任务名，例如 adjust_bottle、rl_offline_ee3/press_stapler",
    )
    parser.add_argument(
        "--list_mixture_datasets",
        action="store_true",
        help="列出 data_mix 内可用数据集路径后退出",
    )
    parser.add_argument(
        "--episode_source",
        type=str,
        choices=("auto", "single", "mixture"),
        default="auto",
        help="auto：有 data_mix 且无 task_name/data_name 时对整个 mixture 混采；"
        "single：只评一个子数据集；mixture：强制 mixture 混采",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="mixture 混采 episode 时的随机种子",
    )
    parser.add_argument(
        "--no_balance_trajectory",
        action="store_true",
        help="mixture 混采时不对长轨迹加大权重（默认按 episode length 加权）",
    )

    ep_group = parser.add_mutually_exclusive_group()
    ep_group.add_argument(
        "--all_episodes",
        action="store_true",
        help="评估 meta/episodes.jsonl 中的全部 episode（默认行为）",
    )
    ep_group.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="起始 episode id；与 --num_episodes 联用",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="从 --episode_index 起连续评估 N 条；未指定且给了 --episode_index 时默认为 1",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="在 --all_episodes 模式下最多评估前 N 条（按 episode_index 排序）",
    )

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--big_negative", type=float, default=100.0)
    parser.add_argument(
        "--normalize_returns",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize_use_big_negative_in_denom",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--normalize_returns_per_task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启，与 run_value_*_T.sh 训练脚本一致；如需关闭传 --no-normalize_returns_per_task",
    )
    parser.add_argument("--success_col", type=str, default="episode_success")

    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--bin_min", type=float, default=-1.0)
    parser.add_argument("--bin_max", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)

    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="保存网格图路径；默认写到 checkpoint 同目录；传 '' 则不保存",
    )
    parser.add_argument("--grid_rows", type=int, default=2, help="子图网格行数（默认 2）")
    parser.add_argument("--grid_cols", type=int, default=5, help="子图网格列数（默认 5）")
    parser.add_argument(
        "--plot_num_episodes",
        type=int,
        default=None,
        help="写入网格图的 episode 数，默认 grid_rows * grid_cols（10）",
    )
    parser.add_argument("--dpi", type=int, default=120, help="输出 PNG 分辨率")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="关闭 tqdm 进度条",
    )

    args = parser.parse_args()

    if args.plot_num_episodes is None:
        args.plot_num_episodes = args.grid_rows * args.grid_cols

    if args.num_episodes is None:
        args.num_episodes = args.plot_num_episodes

    if args.episode_index is not None and args.num_episodes is None:
        args.num_episodes = 1

    if args.episode_index is not None and args.num_episodes < 1:
        raise ValueError("--num_episodes 必须 >= 1")

    return args


def _episode_metrics(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    preds = np.array([r["value_t"] for r in rows], dtype=np.float64)
    tgts = np.array([r["return_to_go"] for r in rows], dtype=np.float64)
    err = preds - tgts

    t0 = 0
    t_last = len(rows) - 1

    return {
        "data_name": str(meta.get("data_name", "")),
        "episode_index": int(meta["episode_index"]),
        "success": bool(meta.get("success", True)),
        "num_steps": int(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_err": float(np.max(np.abs(err))),
        "pred_t0": float(preds[t0]),
        "target_t0": float(tgts[t0]),
        "err_t0": float(err[t0]),
        "pred_last": float(preds[t_last]),
        "target_last": float(tgts[t_last]),
        "err_last": float(err[t_last]),
        "mean_pred": float(np.mean(preds)),
        "mean_target": float(np.mean(tgts)),
    }


def _resolve_episode_indices(args: argparse.Namespace) -> Tuple[List[int], List[int]]:
    valid_eps = sorted(_valid_episode_index_set(args.data_root_dir, args.data_name))

    if args.all_episodes:
        indices = list(valid_eps)
        if args.max_episodes is not None:
            indices = indices[: args.max_episodes]
        return indices, []

    assert args.episode_index is not None
    wanted = list(range(args.episode_index, args.episode_index + args.num_episodes))
    indices = [ei for ei in wanted if ei in valid_eps]
    skipped = [ei for ei in wanted if ei not in valid_eps]
    return indices, skipped


def _print_table(rows: List[Dict[str, Any]], show_task: bool = False) -> None:
    if show_task:
        header = (
            f"{'task':<28} {'ep':>5} {'succ':>5} {'steps':>6} {'mae':>8} {'rmse':>8} "
            f"{'pred0':>8} {'tgt0':>8}"
        )
    else:
        header = (
            f"{'ep':>5} {'succ':>5} {'steps':>6} {'mae':>8} {'rmse':>8} "
            f"{'pred0':>8} {'tgt0':>8} {'e0':>8} {'predL':>8} {'tgtL':>8}"
        )
    print(header)
    print("-" * len(header))
    for r in rows:
        if show_task:
            task = _short_task_label(r.get("data_name", ""))[:28]
            print(
                f"{task:<28} {r['episode_index']:5d} {int(r['success']):5d} "
                f"{r['num_steps']:6d} {r['mae']:8.4f} {r['rmse']:8.4f} "
                f"{r['pred_t0']:8.4f} {r['target_t0']:8.4f}"
            )
        else:
            print(
                f"{r['episode_index']:5d} "
                f"{int(r['success']):5d} "
                f"{r['num_steps']:6d} "
                f"{r['mae']:8.4f} "
                f"{r['rmse']:8.4f} "
                f"{r['pred_t0']:8.4f} "
                f"{r['target_t0']:8.4f} "
                f"{r['err_t0']:8.4f} "
                f"{r['pred_last']:8.4f} "
                f"{r['target_last']:8.4f}"
            )


def _print_summary(metrics: List[Dict[str, Any]]) -> None:
    if not metrics:
        print("\n[Summary] 无有效 episode")
        return

    mae_all = [m["mae"] for m in metrics]
    succ = [m for m in metrics if m["success"]]
    fail = [m for m in metrics if not m["success"]]

    print(f"\n[Summary] episodes={len(metrics)}  overall_mae={np.mean(mae_all):.4f}  "
          f"overall_rmse={np.mean([m['rmse'] for m in metrics]):.4f}")

    if succ:
        print(
            f"  success (n={len(succ)}): mae={np.mean([m['mae'] for m in succ]):.4f}  "
            f"mean_pred_t0={np.mean([m['pred_t0'] for m in succ]):.4f}  "
            f"mean_target_t0={np.mean([m['target_t0'] for m in succ]):.4f}"
        )
    if fail:
        print(
            f"  failure (n={len(fail)}): mae={np.mean([m['mae'] for m in fail]):.4f}  "
            f"mean_pred_t0={np.mean([m['pred_t0'] for m in fail]):.4f}  "
            f"mean_target_t0={np.mean([m['target_t0'] for m in fail]):.4f}"
        )

    if succ and fail:
        gap = np.mean([m["pred_t0"] for m in succ]) - np.mean([m["pred_t0"] for m in fail])
        print(f"  pred_t0 gap (succ - fail) = {gap:.4f}")


def _save_outputs(
    metrics: List[Dict[str, Any]],
    args: argparse.Namespace,
    skipped: List[int],
    output_json: Optional[str],
    output_csv: Optional[str],
) -> None:
    payload = {
        "checkpoint_path": args.checkpoint_path,
        "data_mix": getattr(args, "data_mix", None),
        "data_root_dir": args.data_root_dir,
        "data_name": args.data_name,
        "bin_min": args.bin_min,
        "bin_max": args.bin_max,
        "normalize_returns": args.normalize_returns,
        "normalize_returns_per_task": args.normalize_returns_per_task,
        "skipped_episode_indices": skipped,
        "episodes": metrics,
        "summary": {
            "count": len(metrics),
            "overall_mae": float(np.mean([m["mae"] for m in metrics])) if metrics else None,
            "overall_rmse": float(np.mean([m["rmse"] for m in metrics])) if metrics else None,
        },
    }

    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] JSON -> {p}")

    if output_csv:
        p = Path(output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(metrics[0].keys()) if metrics else []
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
        print(f"[Done] CSV -> {p}")


def plot_episode_grid(
    episode_blocks: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    output_png: Path,
    grid_rows: int,
    grid_cols: int,
    dpi: int,
    data_name: str,
) -> None:
    """Grid of per-episode value curves (solid=pred, dashed=target)."""
    n_slots = grid_rows * grid_cols
    blocks = episode_blocks[:n_slots]

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3.2 * grid_cols, 2.6 * grid_rows))
    if grid_rows == 1 and grid_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.ravel(axes))

    for slot in range(n_slots):
        ax = axes_flat[slot]
        if slot >= len(blocks):
            ax.axis("off")
            continue

        rows, meta = blocks[slot]
        steps = [r["step"] for r in rows]
        values = [r["value_t"] for r in rows]
        returns = [r["return_to_go"] for r in rows]
        ep_id = int(meta.get("episode_index", -1))
        succ = meta.get("success")
        mae = float(np.mean(np.abs(np.array(values) - np.array(returns))))
        task_lbl = _short_task_label(str(meta.get("data_name", data_name)))[:18]

        ax.plot(steps, values, color="#1f77b4", linewidth=1.2, linestyle="-")
        ax.plot(steps, returns, color="#ff7f0e", linewidth=1.0, linestyle="--", alpha=0.85)
        ax.set_title(
            f"{task_lbl}\nep{ep_id} succ={int(bool(succ))} mae={mae:.3f}",
            fontsize=7,
        )
        if slot % grid_cols == 0:
            ax.set_ylabel("Value", fontsize=7)
        if slot >= (grid_rows - 1) * grid_cols:
            ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

    for slot in range(len(blocks), n_slots):
        axes_flat[slot].axis("off")

    fig.suptitle(
        f"{data_name}\npred (solid) / target (dashed) — {len(blocks)}/{n_slots} episodes",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args = resolve_run_configuration(args)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if args.cuda_device < 0 or args.cuda_device >= n_gpu:
            raise ValueError(
                f"--cuda_device={args.cuda_device} 无效，当前可见 GPU 数量={n_gpu}"
            )
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"
    print(f"[Info] device={device}")

    cfg = OmegaConf.load(args.config_yaml)
    if args.base_vlm is not None and "qwenvl" in cfg.framework:
        cfg.framework.qwenvl.base_vlm = args.base_vlm

    if "qwenvl" in cfg.framework:
        model_id = str(cfg.framework.qwenvl.get("base_vlm", ""))
        if model_id and (model_id.startswith("./") or model_id.startswith("../")):
            resolved = (Path(args.config_yaml).resolve().parent / model_id).resolve()
            if resolved.exists():
                cfg.framework.qwenvl.base_vlm = str(resolved)
    elif "gemma_value" in cfg.framework:
        for key in ("vision_model", "text_model", "tokenizer"):
            model_id = str(cfg.framework.gemma_value.get(key, ""))
            if model_id and (model_id.startswith("./") or model_id.startswith("../")):
                resolved = (Path(args.config_yaml).resolve().parent / model_id).resolve()
                if resolved.exists():
                    cfg.framework.gemma_value[key] = str(resolved)

    model = load_model(args.checkpoint_path, cfg, device)

    if args.episode_source == "mixture":
        episode_jobs = sample_episodes_from_mixture(
            args.data_mix,
            args.data_root_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            balance_trajectory=not args.no_balance_trajectory,
        )
        skipped: List[int] = []
    else:
        if args.episode_index is None and not args.all_episodes:
            args.episode_index = 0
        episode_indices, skipped = _resolve_episode_indices(args)
        if skipped:
            n_show = min(20, len(skipped))
            tail = " ..." if len(skipped) > n_show else ""
            print(
                f"[Warn] 跳过不存在的 episode（共 {len(skipped)}）: "
                f"{skipped[:n_show]}{tail}"
            )
        if not episode_indices:
            raise ValueError("没有可评估的 episode")
        episode_jobs = [(args.data_name, ei) for ei in episode_indices]
        print(
            f"[Info] 单任务: {args.data_name}，连续评估 {len(episode_jobs)} 条 "
            f"(ep {episode_indices[0]}..{episode_indices[-1]})"
        )

    if args.normalize_returns_per_task:
        print("[Info] normalize_returns_per_task=True")
    elif args.normalize_returns:
        print(
            "[Info] per-episode 归一化；若训练用了 --normalize_returns_per_task 请加上该参数"
        )

    denom_cache: Dict[str, Optional[Dict[int, int]]] = {}
    ds_cache = TrainingAlignedDatasetCache(args) if not args.legacy_raw_video else None
    all_metrics: List[Dict[str, Any]] = []
    episode_blocks: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    need_plot = bool(args.output_png)
    plot_limit = args.plot_num_episodes if need_plot else 0

    def _short_task(name: str) -> str:
        return name.rstrip("/").split("/")[-1][:28]

    ep_iter = episode_jobs
    if not args.no_progress:
        ep_iter = tqdm(
            episode_jobs,
            desc="评估 episodes",
            unit="ep",
            total=len(episode_jobs),
        )

    for job_data_name, ei in ep_iter:
        args.data_name = job_data_name
        if args.legacy_raw_video:
            denom_per_traj = None
            if args.normalize_returns_per_task:
                if job_data_name not in denom_cache:
                    denom_cache[job_data_name] = build_denom_per_traj_from_dataset(
                        Path(args.data_root_dir) / job_data_name
                    )
                denom_per_traj = denom_cache[job_data_name]
            examples, meta = build_episode_examples(args, ei, denom_per_traj=denom_per_traj)
        else:
            if not args.data_mix:
                raise ValueError("训练同款 eval 需要 --data_mix（或可从 checkpoint 目录推断）")
            robot_type = resolve_robot_type(args.data_mix, job_data_name)
            examples, meta = build_episode_examples_training_aligned(
                args, ei, robot_type, ds_cache
            )
        meta["data_name"] = job_data_name
        infer_desc = f"推理 {_short_task(job_data_name)} ep{ei}"
        rows = collect_episode_entries(
            model=model,
            examples=examples,
            batch_size=args.batch_size,
            bin_min=args.bin_min,
            bin_max=args.bin_max,
            progress_desc=infer_desc,
            show_progress=not args.no_progress,
        )
        if not rows:
            print(f"[Warn] {job_data_name} ep {ei} 无有效帧，跳过")
            continue

        m = _episode_metrics(rows, meta)
        all_metrics.append(m)
        if need_plot and len(episode_blocks) < plot_limit:
            episode_blocks.append((rows, meta))

    all_metrics.sort(key=lambda x: (x.get("data_name", ""), x["episode_index"]))
    episode_blocks.sort(key=lambda x: (x[1].get("data_name", ""), x[1]["episode_index"]))

    mix_tag = f" mix={args.data_mix}" if args.data_mix else ""
    src_tag = f" source={args.episode_source}"
    data_tag = "" if args.episode_source == "mixture" else f" data={args.data_name}"
    print(f"\n[Episode metrics]{mix_tag}{src_tag}{data_tag}")
    _print_table(all_metrics, show_task=(args.episode_source == "mixture"))
    _print_summary(all_metrics)

    if args.output_json or args.output_csv:
        _save_outputs(
            all_metrics,
            args,
            skipped,
            args.output_json,
            args.output_csv,
        )

    if args.output_png:
        if not episode_blocks:
            print("[Warn] 无可用曲线，跳过保存 PNG")
        else:
            out_png = Path(args.output_png)
            grid_title = args.data_mix if args.episode_source == "mixture" else args.data_name
            plot_episode_grid(
                episode_blocks,
                out_png,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                dpi=args.dpi,
                data_name=grid_title,
            )
            print(f"[Done] PNG ({args.grid_rows}x{args.grid_cols}, "
                  f"{min(len(episode_blocks), args.plot_num_episodes)} eps) -> {out_png}")


if __name__ == "__main__":
    main()
