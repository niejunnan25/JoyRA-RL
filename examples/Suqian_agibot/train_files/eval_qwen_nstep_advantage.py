"""
Compute RECAP-style n-step advantages with a learned value model.

Compared with eval_qwen_value_fit.py, this script is closer to the
advantage form used in pi*0.6 / RECAP:

    A_t = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n V(s_{t+n}) - V(s_t)

The dataset only provides return-to-go targets G_t, so the script
reconstructs the n-step reward sum from returns:

    sum_{k=0}^{n-1} gamma^k r_{t+k} = G_t - gamma^n G_{t+n}

If t + n goes past the end of the trajectory, it falls back to:

    reward_sum = G_t
    bootstrap_value = 0

For distributed runs, every rank only predicts values for its own shard.
Rank 0 then merges all shard outputs and computes advantages globally, so
bootstrap states from other ranks are not lost.

只测 value、不算 advantage：加 ``--value_only``，会在 checkpoint 同目录写出
``value_predictions.json``（每步 pred_value / return_to_go），并保存
``value_fit_scatter.png`` / ``value_pred_histogram.png``；不再写
``advantages_all_train1.json``。与 ``eval_qwen_value_fit.py`` 的区别是：本脚本仍走
与 n-step 评估相同的数据管线（默认 mode=train、train_split 等）。
可用 ``--max_eval_steps N`` 只推理约 N 条 step（多卡时按 rank 均分），做快速抽检。
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.training.train_value import build_value_dataloader, build_value_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, required=True)

    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--data_mix", type=str, default=None)
    parser.add_argument("--returns_cache_dir", type=str, default=None)

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--big_negative", type=float, default=100.0)
    parser.add_argument("--success_col", type=str, default="episode_success")

    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--bin_min", type=float, default=-1.0)
    parser.add_argument("--bin_max", type=float, default=0.0)

    parser.add_argument(
        "--train_split",
        type=float,
        default=1.0,
        help="与 train_value / LeRobotMixtureWithValueTarget 一致：mode=train 时取前 "
        "train_split 比例作为训练索引区间；默认 1.0 表示全部 step 用于 train。若设为 0，"
        "则训练集长度为 0，会触发 DataLoader 报错。",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--n_step", type=int, default=50)
    parser.add_argument(
        "--adv_scale",
        type=float,
        default=1.0,
        help="Optional positive scale divisor applied to the final advantage.",
    )
    parser.add_argument(
        "--normalize_returns",
        action="store_true",
        help="按 episode 长度做 [-1, 0] 归一化（与 train_value.py 一致）",
    )
    parser.add_argument(
        "--normalize_returns_per_task",
        action="store_true",
        help="按 task 的最大 episode 长度做 [-1, 0] 归一化",
    )
    parser.add_argument(
        "--normalize_use_big_negative_in_denom",
        action="store_true",
        help="归一化分母中使用 H + big_negative",
    )
    parser.add_argument(
        "--value_only",
        action="store_true",
        help="只合并并保存每步 pred_value 与 return_to_go，不计算 n-step advantage",
    )
    parser.add_argument(
        "--value_scatter_path",
        type=str,
        default=None,
        help="value_only：return_to_go vs pred_value 散点图路径，默认 checkpoint 目录下 value_fit_scatter.png",
    )
    parser.add_argument(
        "--value_hist_path",
        type=str,
        default=None,
        help="value_only：预测 value 分布直方图路径，默认 checkpoint 目录下 value_pred_histogram.png",
    )
    parser.add_argument(
        "--value_plot_max_points",
        type=int,
        default=20000,
        help="value_only：散点图最多绘制的点数（随机子采样）",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=None,
        help="最多评估多少个 step（全进程合计近似值）。多卡时每卡 ceil(total/world_size) 条，"
        "用于快速抽检；不设则跑满整个 dataloader。",
    )

    return parser.parse_args()


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    is_dist = False
    rank = 0
    world_size = 1

    if not dist.is_available():
        return is_dist, rank, world_size, local_rank

    if dist.is_initialized():
        is_dist = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return is_dist, rank, world_size, local_rank

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        is_dist = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return is_dist, rank, world_size, local_rank


def load_model(checkpoint_path, cfg, device):
    """加载训练好的 QwenValue 模型，并加载权重。"""
    if not hasattr(cfg, "framework"):
        raise ValueError("Config 文件中必须包含 framework 段用于构建 QwenValue / Qwen-VL。")

    model = build_value_model(cfg)

    print(f"[Eval] 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print(
            f"  - step={ckpt.get('step','N/A')} "
            f"epoch={ckpt.get('epoch','N/A')} "
            f"loss={ckpt.get('loss','N/A')}"
        )
    else:
        state_dict = ckpt

    # 去掉 DDP 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Eval] Warning: missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[Eval] Warning: unexpected keys in state_dict: {unexpected}")

    model.to(device)
    model.eval()
    print(f"[Eval] 模型已加载到 {device}")
    return model


def build_val_loader(args, world_size):
    per_proc_batch = max(1, args.batch_size // max(1, world_size))

    data_args = argparse.Namespace(
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        gamma=args.gamma,
        big_negative=args.big_negative,
        success_col=args.success_col,
        num_bins=args.num_bins,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        bin_range_json=None,
        returns_cache_dir=args.returns_cache_dir,
        train_split=args.train_split,
        batch_size=per_proc_batch,
        num_workers=args.num_workers,
        normalize_returns=args.normalize_returns,
        normalize_returns_per_task=args.normalize_returns_per_task,
        normalize_use_big_negative_in_denom=args.normalize_use_big_negative_in_denom,
    )

    val_loader, _ = build_value_dataloader(
        data_args,
        distributed=(world_size > 1),
        mode="train",
        seed=123,
    )

    n_steps = len(val_loader.dataset)
    if n_steps == 0:
        raise ValueError(
            "数据集在 mode=train 下长度为 0。LeRobotMixtureWithValueTarget 用 "
            "train_split 划分索引：train 区间长度为 total * train_split；若 train_split=0 "
            "则 train 为空。请使用 --train_split 1.0（默认，全量 step 作为 train）。"
        )
    print(f"[Eval] dataloader dataset length (train partition): {n_steps}")

    return val_loader


def collect_local_predictions(
    model, dataloader, rank, bin_min, bin_max, max_samples=None
):
    entries = []

    total_batches = len(dataloader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            elapsed = time.time() - start_time
            n_done = batch_idx + 1
            if max_samples is None:
                avg_time = elapsed / n_done
                remaining = avg_time * (total_batches - batch_idx - 1)
                if rank == 0:
                    print(
                        f"[PREDICT] {n_done}/{total_batches} | "
                        f"ETA: {remaining / 60:.2f} min",
                        end="\r",
                    )
            elif rank == 0:
                print(
                    f"[PREDICT] batch={n_done} samples={len(entries)}/{max_samples} | "
                    f"elapsed={elapsed:.1f}s",
                    end="\r",
                )

            result = model.predict_value(
                examples=batch,
                bin_min=bin_min,
                bin_max=bin_max,
            )

            pred_values = result["values"]
            if isinstance(pred_values, torch.Tensor):
                pred_values = pred_values.tolist()

            for ex, pred_value in zip(batch, pred_values):
                dataset_key = ex.get("dataset_key")
                if dataset_key is None:
                    dataset_key = "__legacy_single_dataset__"
                success = ex.get("success", True)
                if isinstance(success, bool):
                    pass
                else:
                    success = str(success).lower() in ("true", "1", "yes")
                entries.append(
                    {
                        "dataset_key": str(dataset_key),
                        "trajectory_id": int(ex["trajectory_id"]),
                        "step": int(ex["step"]),
                        "pred_value": float(pred_value),
                        "return_to_go": float(ex.get("value_target", 0.0)),
                        "success": success,
                    }
                )
                if max_samples is not None and len(entries) >= max_samples:
                    break

            if max_samples is not None and len(entries) >= max_samples:
                break

    print(f"\n[rank {rank}] prediction pass done. samples={len(entries)}")
    return entries


def wait_for_rank_files(output_dir, world_size, prefix, timeout_seconds=24000):
    paths = []
    wait_start = time.time()

    for rank in range(world_size):
        path = output_dir / f"{prefix}_rank{rank}.json"
        while not path.exists():
            if time.time() - wait_start > timeout_seconds:
                raise RuntimeError(f"Timeout waiting for {path}")
            time.sleep(2)
        paths.append(path)

    return paths


def merge_entries(paths):
    merged = {}
    saw_legacy_entry = False

    for path in paths:
        with open(path, "r") as file:
            data = json.load(file)

        for item in data["entries"]:
            dataset_key = item.get("dataset_key")
            if dataset_key is None:
                dataset_key = "__legacy_single_dataset__"
                saw_legacy_entry = True

            key = (str(dataset_key), int(item["trajectory_id"]), int(item["step"]))
            merged[key] = {
                "dataset_key": str(dataset_key),
                "pred_value": float(item["pred_value"]),
                "return_to_go": float(item["return_to_go"]),
                "success": item.get("success", True),
            }

    if saw_legacy_entry:
        print(
            "[merge_entries] Found legacy entries without dataset_key. "
            "They are grouped under '__legacy_single_dataset__'; this is only safe for single-dataset inputs."
        )

    return merged


def merged_to_value_predictions(entries_by_key):
    """将 merge_entries 的结果转为按 step 排序的列表（仅 value 评估用）。"""
    rows = []
    for dataset_key, trajectory_id, step in sorted(entries_by_key.keys()):
        e = entries_by_key[(dataset_key, trajectory_id, step)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "trajectory_id": trajectory_id,
                "step": step,
                "pred_value": e["pred_value"],
                "return_to_go": e["return_to_go"],
                "success": e.get("success", True),
            }
        )
    return rows


def _compute_ranks(values: np.ndarray) -> np.ndarray:
    """带 ties 平均处理的秩，用于 Spearman 相关（与 eval_qwen_value_fit 一致）。"""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, c in enumerate(counts):
            if c <= 1:
                continue
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    return ranks


def plot_value_only_figures(
    predictions: list,
    bin_min: float,
    bin_max: float,
    scatter_path: Path,
    hist_path: Path,
    max_points: int,
) -> None:
    """绘制 return_to_go vs pred_value 散点图与预测值直方图。"""
    y = np.array([float(p["return_to_go"]) for p in predictions], dtype=np.float32)
    pred = np.array([float(p["pred_value"]) for p in predictions], dtype=np.float32)
    n = len(y)
    if n == 0:
        print("[value_only plot] 无样本，跳过作图")
        return

    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        y_s, p_s = y[idx], pred[idx]
    else:
        y_s, p_s = y, pred

    mask = y_s >= bin_min
    filtered = int((~mask).sum())
    if filtered > 0:
        print(
            f"[value_only plot] 过滤掉 {filtered} 个点：return_to_go < bin_min ({bin_min})"
        )
    y_f = y_s[mask]
    p_f = p_s[mask]
    if len(y_f) == 0:
        print("[value_only plot] 过滤后无样本，跳过作图")
        return

    denom = float(bin_max - bin_min) if bin_max != bin_min else 1.0
    norm_y = np.clip((y_f - bin_min) / denom, 0.0, 1.0)
    norm_p = np.clip((p_f - bin_min) / denom, 0.0, 1.0)
    mse = float(np.mean((norm_y - norm_p) ** 2))
    mae = float(np.mean(np.abs(norm_y - norm_p)))
    if len(y_f) > 1:
        pearson_r = float(np.corrcoef(norm_y, norm_p)[0, 1])
        tr = _compute_ranks(norm_y)
        pr = _compute_ranks(norm_p)
        spearman_r = float(np.corrcoef(tr, pr)[0, 1])
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")
    print(
        f"[value_only plot] N={len(y_f)} (scatter), total_steps={n}, "
        f"MSE={mse:.4f}, MAE={mae:.4f}, Pearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}"
    )

    scatter_path = Path(scatter_path)
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_f, p_f, s=4, alpha=0.35)
    lo = min(float(y_f.min()), float(p_f.min()))
    hi = max(float(y_f.max()), float(p_f.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    plt.xlabel("return_to_go (target)")
    plt.ylabel("pred_value")
    plt.title(
        f"Value fit (N={len(y_f)}, r={pearson_r:.3f}, ρ={spearman_r:.3f})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"[value_only plot] 散点图: {scatter_path}")

    hist_path = Path(hist_path)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(pred, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    plt.xlabel("pred_value")
    plt.ylabel("count")
    plt.title(f"Predicted value distribution (N={n})")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[value_only plot] 直方图: {hist_path}")


def compute_n_step_reward(cur_return, future_return, gamma, n_step):
    if future_return is None:
        return cur_return
    return cur_return - (gamma ** n_step) * future_return


def compute_advantages(entries_by_key, n_step, gamma, adv_scale):
    if adv_scale <= 0:
        raise ValueError(f"adv_scale must be positive, got {adv_scale}")

    advantages = []
    sorted_keys = sorted(entries_by_key.keys())

    for dataset_key, trajectory_id, step in sorted_keys:
        cur = entries_by_key[(dataset_key, trajectory_id, step)]
        future = entries_by_key.get((dataset_key, trajectory_id, step + n_step))

        future_return = None if future is None else future["return_to_go"]
        bootstrap_value = 0.0 if future is None else future["pred_value"]

        n_step_reward = compute_n_step_reward(
            cur_return=cur["return_to_go"],
            future_return=future_return,
            gamma=gamma,
            n_step=n_step,
        )
        bootstrap_weight = gamma ** n_step if future is not None else 0.0
        advantage_raw = (
            n_step_reward
            + bootstrap_weight * bootstrap_value
            - cur["pred_value"]
        )
        print(cur.get("success"), flush=True)

        advantages.append(
            {
                "dataset_key": dataset_key,
                "trajectory_id": trajectory_id,
                "step": step,
                "n_step_reward": n_step_reward,
                "bootstrap_value": bootstrap_value,
                "value_t": cur["pred_value"],
                "advantage_raw": advantage_raw,
                "advantage": advantage_raw / adv_scale,
                "success": cur.get("success", True),
            }
        )

    return advantages


def main():
    args = parse_args()

    is_dist, rank, world_size, local_rank = setup_distributed()

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    print(f"rank={rank}, world_size={world_size}, device={device}")

    cfg = OmegaConf.load(args.config_yaml)

    vla_cfg = cfg.get("datasets", {}).get("vla_data", {})
    if args.data_mix is None:
        args.data_mix = vla_cfg.get("data_mix")
    if args.data_root_dir is None:
        args.data_root_dir = vla_cfg.get("data_root_dir")

    if not args.data_mix or not args.data_root_dir:
        raise ValueError(
            "--data_mix and --data_root_dir must be provided via CLI or "
            "set in config_yaml under datasets.vla_data"
        )

    model = load_model(args.checkpoint_path, cfg, device)
    val_loader = build_val_loader(args, world_size)

    per_rank_cap = None
    if args.max_eval_steps is not None:
        if args.max_eval_steps <= 0:
            raise ValueError("--max_eval_steps 必须为正整数")
        per_rank_cap = (args.max_eval_steps + world_size - 1) // world_size
        if rank == 0:
            print(
                f"[Eval] max_eval_steps={args.max_eval_steps} "
                f"(约每 rank ≤{per_rank_cap} 条, world_size={world_size})"
            )

    local_entries = collect_local_predictions(
        model,
        val_loader,
        rank,
        args.bin_min,
        args.bin_max,
        max_samples=per_rank_cap,
    )

    output_dir = Path(args.checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = output_dir / f"train_nstep_inputs1_rank{rank}.json"
    with open(local_path, "w") as file:
        json.dump(
            {
                "n_step": args.n_step,
                "gamma": args.gamma,
                "entries": local_entries,
            },
            file,
            indent=4,
        )

    print(f"[rank {rank}] saved local inputs to {local_path}")

    if rank == 0:
        input_paths = wait_for_rank_files(
            output_dir=output_dir,
            world_size=world_size,
            prefix="train_nstep_inputs1",
        )
        entries_by_key = merge_entries(input_paths)

        if args.value_only:
            predictions = merged_to_value_predictions(entries_by_key)
            output_path = output_dir / "value_predictions.json"
            with open(output_path, "w") as file:
                json.dump(
                    {
                        "gamma": args.gamma,
                        "value_only": True,
                        "max_eval_steps": args.max_eval_steps,
                        "num_steps": len(predictions),
                        "predictions": predictions,
                    },
                    file,
                    indent=4,
                )
            print(f"[rank 0] saved value-only predictions to {output_path}")

            scatter_out = (
                Path(args.value_scatter_path)
                if args.value_scatter_path
                else output_dir / "value_fit_scatter.png"
            )
            hist_out = (
                Path(args.value_hist_path)
                if args.value_hist_path
                else output_dir / "value_pred_histogram.png"
            )
            plot_value_only_figures(
                predictions=predictions,
                bin_min=args.bin_min,
                bin_max=args.bin_max,
                scatter_path=scatter_out,
                hist_path=hist_out,
                max_points=args.value_plot_max_points,
            )
        else:
            advantages = compute_advantages(
                entries_by_key=entries_by_key,
                n_step=args.n_step,
                gamma=args.gamma,
                adv_scale=args.adv_scale,
            )

            output_path = output_dir / "advantages_all_train1.json"
            with open(output_path, "w") as file:
                json.dump(
                    {
                        "n_step": args.n_step,
                        "gamma": args.gamma,
                        "adv_scale": args.adv_scale,
                        "advantages": advantages,
                    },
                    file,
                    indent=4,
                )

            print(f"[rank 0] saved merged advantages to {output_path}")

    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
