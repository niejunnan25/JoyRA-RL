"""
可视化 QwenValue 值函数的拟合效果：
在验证集上抽样若干点，绘制「真实 return（value_target）」vs「模型预测 value」的散点图，
并计算相关系数 / MSE 等指标。

用法示例（与 aloha-agilex_550_mix 训练配置一致）::

  python examples/Suqian_agibot/train_files/eval_qwen_value_fit.py \
    --checkpoint_path /mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
    --config_yaml   /mnt/workspace/users/daiyixiang/JoyRA-RL/examples/Suqian_agibot/train_files/starvla_value_function.yaml \
    --data_root_dir /mnt/workspace/datasets \
    --data_mix robotwin_aloha_agilex_550_mix \
    --bin_min -600.0 --bin_max 0.0 \
    --train_split 0.9 \
    --num_samples 2000 \
    --output /mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_aloha_agilex_550_mix/value_fit_scatter.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from starVLA.training.train_value import build_value_dataloader, build_value_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="在验证集上可视化 QwenValue 值函数拟合效果"
    )

    # 模型 / 配置
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="训练好的 checkpoint 路径（支持 train_value.py 保存的 checkpoint_step_*.pt 或 qwen_value_best.pt）",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="与训练时相同的 config YAML（用于构建 QwenValue / Qwen-VL）",
    )

    # 数据与 bin 配置（需与训练时一致）
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/mnt/workspace/datasets",
        help="数据根目录（与训练时相同）",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default="robotwin_aloha_agilex_550_mix",
        help="使用的数据混合名称（在 mixtures.py 中定义）",
    )
    parser.add_argument(
        "--returns_cache_dir",
        type=str,
        default=None,
        help="return 缓存目录（与训练时相同，可选）",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="折扣因子 gamma（需与训练时一致）",
    )
    parser.add_argument(
        "--big_negative",
        type=float,
        default=100.0,
        help="失败惩罚 big_negative（需与训练时一致）",
    )
    parser.add_argument(
        "--success_col",
        type=str,
        default="episode_success",
        help="成功标记列名（需与训练时一致）",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=201,
        help="value bin 数量（需与训练时一致）",
    )
    # 与训练保持一致的归一化选项
    parser.add_argument(
        "--normalize_returns",
        action="store_true",
        help="按 episode 长度做 [-1, 0] 归一化（与 train_value.py 中 --normalize_returns 相同）",
    )
    parser.add_argument(
        "--normalize_returns_per_task",
        action="store_true",
        help="按 task 的最大 episode 长度做 [-1, 0] 归一化（与 train_value.py 中 --normalize_returns_per_task 相同）",
    )
    parser.add_argument(
        "--normalize_use_big_negative_in_denom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启，与当前 value 训练脚本一致：归一化分母使用 H + big_negative。"
        "如需评估旧 H-only checkpoint，传 --no-normalize_use_big_negative_in_denom。",
    )
    parser.add_argument(
        "--bin_min",
        type=float,
        default=-600.0,
        help="bin 最小值（与训练时一致）",
    )
    parser.add_argument(
        "--bin_max",
        type=float,
        default=0.0,
        help="bin 最大值（与训练时一致）",
    )

    # 数据划分 / 抽样
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="训练 / 验证划分比例（与训练时一致，用于取验证集部分）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="评估时的 batch 大小",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="在验证集上最多抽样多少个 step 做可视化",
    )

    # 其它
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备（cuda 或 cpu）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="value_fit_scatter.png",
        help="输出散点图文件路径",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, cfg_path: str, device: str = "cuda"):
    """加载训练好的 QwenValue 模型，并加载权重。"""
    print(f"[Eval] 加载 config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if not hasattr(cfg, "framework"):
        raise ValueError("Config 文件中必须包含 framework 段用于构建 QwenValue / Qwen-VL。")

    # 使用与 train_value.py 相同的构建方式
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


def build_val_dataloader(args):
    """基于 train_value.py 的 build_value_dataloader 构建验证集 DataLoader。"""
    # 构造一个简化的 args Namespace，满足 build_value_dataloader 所需字段
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        # 与训练对齐的归一化开关
        # - 如果训练开了 --normalize_returns / --normalize_returns_per_task
        #   那么评估时也应该传同样的 flag，确保：
        #   1）数据集内部的 value_target 计算逻辑一致；
        #   2）bin_min/bin_max 与训练时保持同步（例如 [-1, 0]）。
        normalize_returns=args.normalize_returns,
        normalize_returns_per_task=args.normalize_returns_per_task,
        normalize_use_big_negative_in_denom=args.normalize_use_big_negative_in_denom,
    )

    val_loader, _ = build_value_dataloader(
        data_args,
        distributed=False,
        mode="val",
        seed=123,
    )

    print(
        f"[Eval] 验证集大小: {len(val_loader.dataset)}, "
        f"batch_size={args.batch_size}"
    )
    # 读取实际使用的 bin 范围（以防数据驱动模式下被调整）
    dataset = val_loader.dataset
    bin_min_used = getattr(dataset, "_bin_min", args.bin_min)
    bin_max_used = getattr(dataset, "_bin_max", args.bin_max)
    print(f"[Eval] 使用的 bin 范围: [{bin_min_used:.3f}, {bin_max_used:.3f}]")

    return val_loader, bin_min_used, bin_max_used


def _compute_ranks(values: np.ndarray) -> np.ndarray:
    """
    计算带平均处理 ties 的秩，用于 Spearman 相关系数。
    """
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)

    # 处理重复值：对相同值取平均秩
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, c in enumerate(counts):
            if c <= 1:
                continue
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    return ranks


def evaluate_and_plot(model, val_loader, bin_min, bin_max, num_samples: int, device: str, output_path: str):
    """在验证集上跑若干样本，绘制 true vs predicted 的散点图。"""
    all_true = []
    all_pred = []

    model.to(device)

    with torch.no_grad():
        for batch in val_loader:
            # batch 是 List[dict]，每个元素中包含 "value_target"
            true_vals = [float(ex["value_target"]) for ex in batch]

            # 模型预测连续 value（通过分布的期望值）
            result = model.predict_value(
                examples=batch,
                bin_min=bin_min,
                bin_max=bin_max,
            )
            pred_vals = result["values"].tolist()

            all_true.extend(true_vals)
            all_pred.extend(pred_vals)

            if len(all_true) >= num_samples:
                break

    all_true = np.array(all_true[:num_samples], dtype=np.float32)
    all_pred = np.array(all_pred[:num_samples], dtype=np.float32)

    # 截断：只保留大于等于 bin_min 的样本（例如 -600 以上），
    # 与训练时的有效 value 区间保持一致，避免极端长轨迹早期步干扰可视化。
    mask = all_true >= bin_min
    filtered = int((~mask).sum())
    if filtered > 0:
        print(f"[Eval] Filtered out {filtered} samples with true return < bin_min ({bin_min}).")
    raw_true = all_true[mask]
    raw_pred = all_pred[mask]

    print(raw_pred)

    print(f"[Eval] 实际用于可视化的样本数: {len(raw_true)}")

    # ===== 按 [-600, 0] 做归一化到 [0, 1] 后再计算指标 =====
    denom = float(bin_max - bin_min) if bin_max != bin_min else 1.0
    norm_true = (raw_true - bin_min) / denom
    norm_pred = (raw_pred - bin_min) / denom
    norm_true = np.clip(norm_true, 0.0, 1.0)
    norm_pred = np.clip(norm_pred, 0.0, 1.0)

    # 计算整体指标（在归一化空间中）
    mse = float(np.mean((norm_true - norm_pred) ** 2))
    mae = float(np.mean(np.abs(norm_true - norm_pred)))
    if len(norm_true) > 1:
        pearson_r = float(np.corrcoef(norm_true, norm_pred)[0, 1])

        # Spearman: 对秩做 Pearson 相关（线性缩放不影响秩，这里用归一化后的值）
        true_rank = _compute_ranks(norm_true)
        pred_rank = _compute_ranks(norm_pred)
        spearman_r = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")

    print(
        f"[Eval] MSE = {mse:.4f}, MAE = {mae:.4f}, "
        f"Pearson r = {pearson_r:.4f}, Spearman r = {spearman_r:.4f}"
    )

    # 按真实 return 区间分桶统计 MAE/MSE（但误差在归一化空间中计算）
    bins = [-600, -400, -200, -100, -50, 0]
    print("\n[Eval] Bucketed errors by true return:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        bmask = (raw_true >= lo) & (raw_true < hi) if i < len(bins) - 2 else (raw_true >= lo) & (raw_true <= hi)
        if not np.any(bmask):
            continue
        bt = norm_true[bmask]
        bp = norm_pred[bmask]
        bucket_mse = float(np.mean((bt - bp) ** 2))
        bucket_mae = float(np.mean(np.abs(bt - bp)))
        print(
            f"  [{lo:6.1f}, {hi:6.1f}]  "
            f"count={len(bt):6d}, MSE={bucket_mse:8.4f}, MAE={bucket_mae:8.4f}"
        )

    # 绘制散点图
    plt.figure(figsize=(6, 6))
    # 散点图仍然在原始 return 空间中展示，便于直观理解
    plt.scatter(raw_true, raw_pred, s=5, alpha=0.4)
    min_val = min(raw_true.min(), raw_pred.min())
    max_val = max(raw_true.max(), raw_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
    plt.xlabel("True Return (value_target)")
    plt.ylabel("Predicted Value")
    plt.title(f"Value Function Fit (N={len(all_true)}, r={pearson_r:.3f}, ρ={spearman_r:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Eval] 散点图已保存到: {output_path}")


def main():
    args = parse_args()

    # 1) 加载模型
    model = load_model(args.checkpoint_path, args.config_yaml, args.device)

    # 2) 构建验证集 DataLoader
    val_loader, bin_min_used, bin_max_used = build_val_dataloader(args)

    # 3) 在验证集上评估并绘图
    evaluate_and_plot(
        model=model,
        val_loader=val_loader,
        bin_min=bin_min_used,
        bin_max=bin_max_used,
        num_samples=args.num_samples,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
