"""
简化版 value function 训练脚本。

功能：
  - 使用 framework.name 指定的价值网络（例如 QwenValue / GemmaValue）
  - 使用 LeRobotSingleDataset 读取两个数据集：
        /mnt/workspace/datasets/AgiBotWorld-Beta-LeRobot
        /mnt/workspace/datasets/suqian_agibot_lerobot_data/task_170_modified
    （可通过命令行参数修改）
  - 使用 LeRobotWithValueTarget 在线生成每个 step 的 value_target
  - 纯 PyTorch 单机单卡训练循环（不依赖 accelerate / deepspeed）
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import (
    make_LeRobotSingleDataset,
    collate_fn as lerobot_collate_fn,
)
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.value_targets_wrapper import (
    LeRobotWithValueTarget,
    LeRobotMixtureWithValueTarget,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified value-function training script")

    # 配置与模型
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="YAML 配置文件路径（需包含 framework.qwenvl.base_vlm 等）",
    )
    parser.add_argument(
        "--framework_name",
        type=str,
        default=None,
        help="可选：覆盖 YAML 中的 framework.name，例如 QwenValue 或 GemmaValue。",
    )

    # 数据配置
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/mnt/workspace/datasets",
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default="sq_agi_beta",
        help="使用 DATASET_NAMED_MIXTURES 中定义的 mixture 名称，例如 sq_agi_beta",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="时间下采样步长：frame_stride=1 表示不过采，=3 表示每隔 3 帧取一次（Hz 降为原来的 1/3）。",
    )
    parser.add_argument(
        "--skip_invalid_subtask_frames",
        action="store_true",
        help="不加入训练/采样索引：跳过 parquet 中 subtask_index < 0 的帧（无效帧）。需要数据含 subtask_index 列。",
    )
    parser.add_argument(
        "--language_prefix",
        type=str,
        default=None,
        help="若设置，则在每条训练样本的语言指令前拼接此前缀（与原文本以「, 」分隔）。",
    )

    # value target 相关
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="计算 return 的折扣因子 gamma",
    )
    parser.add_argument(
        "--big_negative",
        type=float,
        default=100.0,
        help="失败末步 reward 的绝对值（r_T=-big_negative）",
    )
    parser.add_argument(
        "--success_col",
        type=str,
        default="episode_success",
        help=(
            "episode 成功标记列名；若 parquet 中无该列，会自动尝试列 success（RL rollout）"
            "与 episode_success；仍无则读 meta/episodes.jsonl 的 success；皆无则视为成功"
        ),
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=201,
        help="将 return 离散为的 bin 数，默认 201（与 π0.6* 一致）",
    )
    parser.add_argument(
        "--bin_range_json",
        type=str,
        default=None,
        help="包含 bin_min/bin_max 的 JSON 文件路径（由 compute_value_bin_range.py 生成）。"
        "如果提供，将使用固定范围；否则使用数据驱动模式。",
    )
    parser.add_argument(
        "--bin_min",
        type=float,
        default=None,
        help="直接指定 bin_min（固定范围模式）。如果提供，将优先使用此值而不是 JSON 文件。",
    )
    parser.add_argument(
        "--bin_max",
        type=float,
        default=None,
        help="直接指定 bin_max（固定范围模式）。如果提供，将优先使用此值而不是 JSON 文件。",
    )
    parser.add_argument(
        "--normalize_returns",
        action="store_true",
        help="如果启用，使用归一化 return 计算（值在 [-1, 0] 范围内），"
        "并自动设置 bin_min=-1.0, bin_max=0.0。此时不需要手动设置 bin 范围。",
    )
    parser.add_argument(
        "--normalize_returns_per_task",
        action="store_true",
        help="如果启用，则按 task 的最大 episode 长度进行归一化，值仍在 [-1, 0]。"
        "该选项隐含开启 normalize_returns，并自动设置 bin_min=-1.0, bin_max=0.0。",
    )
    parser.add_argument(
        "--normalize_use_big_negative_in_denom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启：归一化分母使用 H + big_negative（最大任务长度 + 失败罚分）。"
        "如需旧版 H-only 行为，传 --no-normalize_use_big_negative_in_denom。",
    )

    # 训练超参
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=(
            "最大 optimizer update 步数。推荐使用该参数控制 value 训练长度，"
            "语义与 StarVLA 主线 trainer.max_train_steps 对齐。"
        ),
    )
    parser.add_argument("--epochs", type=int, default=1, help="旧版兼容参数：训练阶段数")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help=(
            "旧版兼容参数：每个训练阶段的最大步数上限。"
            "如果未设置 --max_train_steps，则总步数 = epochs * steps_per_epoch。"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory. Useful when batches contain tensors copied to CUDA.",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when num_workers > 0.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="DataLoader prefetch_factor when num_workers > 0. PyTorch default is 2.",
    )
    parser.add_argument(
        "--empty_cache_steps",
        type=int,
        default=100,
        help=(
            "Call torch.cuda.empty_cache() every N train steps. "
            "Set <= 0 to disable periodic empty_cache during training."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_value",
        help="模型保存目录",
    )
    parser.add_argument(
        "--returns_cache_dir",
        type=str,
        default=None,
        help="Return 缓存目录。如果提供，将在此目录下为每个数据集保存/加载 return 缓存文件。"
        "如果为 None，不启用缓存功能。",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="训练集比例（0-1之间），剩余部分作为测试集。默认 0.8（80%% 训练，20%% 测试）。",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="旧版兼容参数：每 N 个训练阶段评估一次。推荐新脚本使用 --eval_steps。",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="每隔 N 个 optimizer steps 做一次验证。语义与 save_steps 对齐。",
    )
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=None,
        help="验证集评估时采样的样本数量。如果为 None，则评估整个验证集。设置此值可以加速验证过程。",
    )
    parser.add_argument(
        "--save_best",
        action="store_true",
        help="如果启用，将根据测试集 loss 保存最佳模型（保存为 qwen_value_best.pt）。",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="每隔 N 个 step 保存一次 checkpoint。如果为 None，不按 step 保存。保存为 checkpoint_step_{step}.pt",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="最多保留多少个 checkpoint 文件。超过限制时，删除最旧的 checkpoint。默认 3。",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=None,
        help=(
            "从已有 .pt 初始化模型权重（在 DDP 包装之前加载）。"
            "支持本脚本保存的 dict（含 model_state_dict）或纯 state_dict。"
            "不恢复 optimizer / global_step，仅用于在新 mixture 上微调。"
        ),
    )

    return parser.parse_args()


def build_value_dataloader(
    args, distributed: bool = False, mode: str = "train", seed: int = 42
) -> tuple[DataLoader, DistributedSampler | None]:
    """
    构建带 value_target / value_bin 的 DataLoader。
    使用 DATASET_NAMED_MIXTURES[data_mix] 里的多个数据集，按 mixture 形式导入。
    
    Args:
        args: 命令行参数
        distributed: 是否使用分布式训练
        mode: "train" 或 "val"，用于区分训练集和验证集（通过不同的 seed 实现）
        seed: 随机种子，train 和 val 使用不同的 seed 来划分数据
    """
    data_root = Path(args.data_root_dir)

    # 简单 data_cfg，用于 make_LeRobotSingleDataset
    # 这里可以通过命令行控制时间下采样的步长（frame_stride）
    data_cfg = {
        "frame_stride": max(1, int(getattr(args, "frame_stride", 1))),
        "skip_invalid_subtask_frames": bool(getattr(args, "skip_invalid_subtask_frames", False)),
    }

    if args.data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"data_mix '{args.data_mix}' not found in DATASET_NAMED_MIXTURES.")

    mixture_spec = DATASET_NAMED_MIXTURES[args.data_mix]

    # 读取 bin 范围（优先级：命令行参数 > JSON 文件 > 数据驱动）
    bin_min = None
    bin_max = None
    
    # 优先级 1: 命令行直接指定
    if args.bin_min is not None and args.bin_max is not None:
        bin_min = args.bin_min
        bin_max = args.bin_max
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"[build_value_dataloader] Using fixed bin range from command line:")
            print(f"  bin_min: {bin_min:.6f}, bin_max: {bin_max:.6f}")
        elif not (dist.is_available() and dist.is_initialized()):
            print(f"[build_value_dataloader] Using fixed bin range from command line:")
            print(f"  bin_min: {bin_min:.6f}, bin_max: {bin_max:.6f}")
    # 优先级 2: JSON 文件
    elif args.bin_range_json is not None:
        bin_range_path = Path(args.bin_range_json)
        if not bin_range_path.exists():
            raise FileNotFoundError(f"Bin range JSON file not found: {bin_range_path}")
        
        with open(bin_range_path, "r") as f:
            bin_range_data = json.load(f)
        
        bin_min = bin_range_data["bin_min"]
        bin_max = bin_range_data["bin_max"]
        
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"[build_value_dataloader] Loaded bin range from {bin_range_path}:")
            print(f"  bin_min: {bin_min:.6f}, bin_max: {bin_max:.6f}")
        elif not (dist.is_available() and dist.is_initialized()):
            print(f"[build_value_dataloader] Loaded bin range from {bin_range_path}:")
            print(f"  bin_min: {bin_min:.6f}, bin_max: {bin_max:.6f}")
    # 优先级 3: 数据驱动模式（在 LeRobotWithValueTarget 中处理）
    else:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"[build_value_dataloader] Using data-driven mode (sampling to estimate bin range)")
        elif not (dist.is_available() and dist.is_initialized()):
            print(f"[build_value_dataloader] Using data-driven mode (sampling to estimate bin range)")

    # 构建 Mixture + ValueTarget 版本的数据集：
    #   - Mixture 部分复用 LeRobotMixtureDataset 的采样策略（多数据集按权重采样）
    #   - ValueTarget 部分复用 LeRobotWithValueTarget（含 return 计算和缓存）
    mixture_datasets = []
    for d_name, d_weight, robot_type in mixture_spec:
        base_ds = make_LeRobotSingleDataset(
            data_root_dir=data_root,
            data_name=d_name,
            robot_type=robot_type,
            delete_pause_frame=False,
            data_cfg=data_cfg,
        )

        # 生成缓存路径（如果启用）
        returns_cache_path = None
        if args.returns_cache_dir is not None:
            cache_dir = Path(args.returns_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # 使用数据集名称和参数生成唯一的缓存文件名
            # 清理数据集名称中的特殊字符（如斜杠）
            safe_d_name = d_name.replace("/", "_").replace("\\", "_")
            skip_tag = "_skip_subinv" if getattr(args, "skip_invalid_subtask_frames", False) else ""
            cache_filename = (
                f"returns_cache_{safe_d_name}{skip_tag}_gamma{args.gamma}_neg{args.big_negative}_"
                f"success{args.success_col}.pkl"
            )
            returns_cache_path = str(cache_dir / cache_filename)

        # 记录 (dataset, weight, cache_path)，后续交给 LeRobotMixtureWithValueTarget 处理
        mixture_datasets.append((base_ds, d_weight, returns_cache_path))

    _lp = getattr(args, "language_prefix", None)
    language_prefix = _lp.strip() if isinstance(_lp, str) and _lp.strip() else None

    dataset = LeRobotMixtureWithValueTarget(
        mixture_datasets=mixture_datasets,
        gamma=args.gamma,
        big_negative=args.big_negative,
        success_col=args.success_col,
        num_bins=args.num_bins,
        bin_min=bin_min,
        bin_max=bin_max,
        # 采样估计 bin range / return 时使用的参数（仅在未显式提供 bin_min/max 时生效）
        sample_size=1000,
        bin_margin=0.1,
        data_cfg=data_cfg,
        seed=seed,
        train_split=args.train_split,
        mode=mode,
        normalize_returns=args.normalize_returns,
        normalize_returns_per_task=args.normalize_returns_per_task,
        normalize_use_big_negative_in_denom=args.normalize_use_big_negative_in_denom,
        language_prefix=language_prefix,
    )
    
    # 对于 val 模式，设置 mixture 的 mode 为 "val"（确保每个 epoch 返回相同的样本）
    if mode == "val":
        dataset.mixture.mode = "val"

    sampler: DistributedSampler | None = None
    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=(mode == "train"),  # 训练集 shuffle，测试集不 shuffle
        )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": (sampler is None and mode == "train"),  # 训练集 shuffle，测试集不 shuffle
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": lerobot_collate_fn,  # 返回 List[dict]，与 QwenValue.forward 兼容
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader, sampler


def build_value_model(cfg, framework_name: str | None = None) -> torch.nn.Module:
    """
    使用现有框架工厂函数构建 value 模型。
    默认尊重 YAML 中的 framework.name；framework_name 非空时才覆盖。
    """
    if framework_name:
        cfg.framework.name = framework_name
    elif not hasattr(cfg.framework, "name"):
        cfg.framework.name = "QwenValue"
    model = build_framework(cfg)
    return model


def main():
    args = parse_args()

    # ==== 分布式初始化（torchrun） ====
    distributed = False
    local_rank = 0
    
    # 检查进程组是否已经初始化（torchrun 会自动初始化）
    if dist.is_available() and dist.is_initialized():
        # 进程组已经初始化（torchrun 或其他方式），直接使用
        distributed = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 手动初始化（非 torchrun 场景，且进程组未初始化）
        distributed = True
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    # 1) 加载配置（只用于构建 QwenValue / Qwen-VL）
    cfg = OmegaConf.load(args.config_yaml)
    if not hasattr(cfg, "framework"):
        raise ValueError("Config 文件中必须包含 framework 段用于构建 QwenValue / Qwen-VL。")

    model = build_value_model(cfg, framework_name=args.framework_name)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    is_main_process_pre = (not distributed) or dist.get_rank() == 0
    if args.init_from_checkpoint:
        ckpt_path = Path(args.init_from_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"init_from_checkpoint 不存在或不是文件: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        # 兼容外层带 module. 的保存格式
        if sd and all(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main_process_pre:
            print(f"已从 checkpoint 初始化权重: {ckpt_path}")
            if missing:
                print(f"  load_state_dict missing keys ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
            if unexpected:
                print(f"  load_state_dict unexpected keys ({len(unexpected)}): {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")

    if distributed:
        # 为了兼容 Qwen3-VL 中可能存在的未参与 value_loss 的参数（如 lm_head 等），
        # 这里重新启用 find_unused_parameters=True，避免 DDP 报错。
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    model.train()

    # 2) 构建训练集和测试集 DataLoader
    train_dataloader, train_sampler = build_value_dataloader(
        args, distributed=distributed, mode="train", seed=42
    )
    val_dataloader, val_sampler = build_value_dataloader(
        args, distributed=distributed, mode="val", seed=42
    )

    # 3) 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4) 简单训练循环
    global_step = 0
    is_main_process = (not distributed) or dist.get_rank() == 0
    
    # 计算训练步数。新语义优先使用 max_train_steps；旧参数仅作为兼容 fallback。
    dataset_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is not None:
        total_steps = int(args.max_train_steps)
        legacy_steps_per_epoch = None
    else:
        if args.steps_per_epoch is not None:
            legacy_steps_per_epoch = min(dataset_steps_per_epoch, args.steps_per_epoch)
        else:
            legacy_steps_per_epoch = dataset_steps_per_epoch
        total_steps = legacy_steps_per_epoch * args.epochs

    if total_steps <= 0:
        raise ValueError(f"total training steps must be positive, got {total_steps}")

    eval_steps = args.eval_steps
    if eval_steps is None and legacy_steps_per_epoch is not None and args.eval_interval is not None:
        eval_steps = legacy_steps_per_epoch * args.eval_interval
    
    # 训练速度统计
    start_time = time.time()
    step_times = []
    data_times = []
    model_times = []
    log_interval = 10  # 每 10 步打印一次
    
    # Loss 记录（用于可视化）
    train_losses = []  # [{step, epoch, loss}, ...]
    val_losses = []    # [{epoch, loss}, ...]
    loss_log_file = None  # 将在主进程中初始化
    
    # 最佳模型跟踪（用于 save_best）
    best_val_loss = float("inf")
    
    # Checkpoint 保存相关
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 loss 日志文件（仅主进程）
    if is_main_process:
        loss_log_file = output_dir / "training_losses.json"
        # 如果文件已存在，读取已有数据（用于断点续训）
        if loss_log_file.exists():
            try:
                with open(loss_log_file, "r") as f:
                    existing_data = json.load(f)
                    train_losses = existing_data.get("train_losses", [])
                    val_losses = existing_data.get("val_losses", [])
                    print(f"  -> 从 {loss_log_file} 加载已有 loss 记录: {len(train_losses)} 个训练点, {len(val_losses)} 个验证点")
            except Exception as e:
                print(f"  -> 警告: 无法读取已有 loss 记录: {e}，将重新开始记录")
    
    world_size = dist.get_world_size() if distributed and dist.is_available() and dist.is_initialized() else 1
    global_batch_size = args.batch_size * world_size

    def _traditional_epoch(step: int) -> float:
        denom = max(len(train_dataloader.dataset), 1)
        return (step * global_batch_size) / denom

    if is_main_process:
        if args.max_train_steps is None:
            print(
                "WARN: 使用旧版 epochs/steps_per_epoch 语义。"
                "推荐改用 --max_train_steps / --eval_steps。"
            )
        print(f"开始训练: max_train_steps={total_steps}")
        print(f"完整训练集步数(当前 world_size/batch 下): {dataset_steps_per_epoch}")
        if legacy_steps_per_epoch is not None and legacy_steps_per_epoch < dataset_steps_per_epoch:
            print(f"  (旧版训练阶段步数限制: {legacy_steps_per_epoch})")
        print(f"等价传统 epoch 估计: {_traditional_epoch(total_steps):.3f}")
        print(f"训练集大小: {len(train_dataloader.dataset)}, 测试集大小: {len(val_dataloader.dataset)}")
        print(
            f"Batch size per GPU/rank={args.batch_size}, world_size={world_size}, "
            f"global batch size={global_batch_size}, Learning rate={args.learning_rate}"
        )
        print(
            f"DataLoader: num_workers={args.num_workers}, pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers if args.num_workers > 0 else False}, "
            f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else None}, "
            f"empty_cache_steps={args.empty_cache_steps}"
        )
        print(f"训练/测试集划分: {args.train_split:.1%} / {1-args.train_split:.1%}")
        if args.save_steps is not None:
            print(f"Checkpoint 保存: 每 {args.save_steps} 步保存一次，最多保留 {args.save_total_limit} 个")
        if eval_steps is not None and eval_steps > 0:
            print(f"验证评估: 每 {eval_steps} 步评估一次")
        print(f"输出目录: {output_dir}")
        print("-" * 80)

    def _write_loss_log() -> None:
        if not is_main_process or loss_log_file is None:
            return
        try:
            with open(loss_log_file, "w") as f:
                json.dump({"train_losses": train_losses, "val_losses": val_losses}, f, indent=2)
        except Exception as e:
            print(f"  -> 警告: 保存 loss 记录失败: {e}")

    def _save_checkpoint(step: int, loss_value: float) -> None:
        if not is_main_process:
            return
        checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
        model_to_save = model.module if isinstance(model, DDP) else model
        checkpoint = {
            "step": step,
            "epoch": _traditional_epoch(step),
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_value,
            "learning_rate": args.learning_rate,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  -> 保存 checkpoint 到: {checkpoint_path} (step={step}, loss={loss_value:.4f})")

        checkpoint_files = sorted(
            output_dir.glob("checkpoint_step_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1]),
            reverse=True,
        )
        if len(checkpoint_files) > args.save_total_limit:
            for old_checkpoint in checkpoint_files[args.save_total_limit:]:
                old_checkpoint.unlink()
                print(f"  -> 删除旧 checkpoint: {old_checkpoint.name}")

    def _run_validation(step: int) -> None:
        nonlocal best_val_loss
        model.eval()
        val_batch_losses = []
        val_start_time = time.time()

        val_batches_to_eval = None
        if args.val_num_samples is not None:
            val_batches_to_eval = (args.val_num_samples + args.batch_size - 1) // args.batch_size
            if is_main_process:
                print(f"验证集采样评估: 限制为 {args.val_num_samples} 个样本 ({val_batches_to_eval} 个 batch)")

        with torch.no_grad():
            for batch_idx, val_batch in enumerate(val_dataloader):
                if val_batches_to_eval is not None and batch_idx >= val_batches_to_eval:
                    break
                outputs = model(val_batch)
                val_loss = outputs["value_loss"]
                val_batch_losses.append(val_loss.item())
                del outputs, val_loss

        torch.cuda.empty_cache()

        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses) if val_batch_losses else float("inf")
        val_time = time.time() - val_start_time

        if is_main_process:
            val_losses.append({
                "step": step,
                "epoch": _traditional_epoch(step),
                "loss": avg_val_loss,
            })
            print(
                f"Step {step}/{total_steps} 验证集评估: "
                f"Loss={avg_val_loss:.4f}, 耗时: {val_time:.2f}秒"
            )

            if args.save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_save_path = output_dir / "qwen_value_best.pt"
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), best_save_path)
                print(f"  -> 保存最佳模型到: {best_save_path} (val_loss={best_val_loss:.4f})")

            _write_loss_log()

        model.train()

    if train_sampler is not None:
        train_sampler.set_epoch(0)
    if val_sampler is not None:
        val_sampler.set_epoch(0)

    model.train()
    data_epoch = 0
    train_iter = iter(train_dataloader)
    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            data_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(data_epoch)
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            if is_main_process:
                print(f"DataLoader 完成一次完整遍历，进入 data_epoch={data_epoch}")

        step_start_time = time.time()
        t_model_start = time.time()

        outputs = model(batch)
        value_loss = outputs["value_loss"]

        optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        optimizer.step()

        t_model_end = time.time()
        model_time = t_model_end - t_model_start
        loss_value = value_loss.item()

        del outputs, value_loss

        if args.empty_cache_steps > 0 and (global_step + 1) % args.empty_cache_steps == 0:
            torch.cuda.empty_cache()

        global_step += 1

        if is_main_process:
            train_losses.append({
                "step": global_step,
                "epoch": _traditional_epoch(global_step),
                "loss": loss_value,
            })

        if args.save_steps is not None and args.save_steps > 0 and global_step % args.save_steps == 0:
            _save_checkpoint(global_step, loss_value)

        step_time = time.time() - step_start_time
        data_time = max(step_time - model_time, 0.0)
        step_times.append(step_time)
        data_times.append(data_time)
        model_times.append(model_time)
        if len(step_times) > 100:
            step_times.pop(0)
            data_times.pop(0)
            model_times.pop(0)

        if is_main_process and global_step % log_interval == 0:
            avg_step_time = sum(step_times) / len(step_times) if step_times else step_time
            avg_data_time = sum(data_times) / len(data_times) if data_times else data_time
            avg_model_time = sum(model_times) / len(model_times) if model_times else model_time
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

            progress = global_step / total_steps if total_steps > 0 else 0
            elapsed_time = time.time() - start_time
            if progress > 0:
                eta_seconds = elapsed_time / progress - elapsed_time
                eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = "N/A"

            print(
                f"Step {global_step}/{total_steps} ({progress*100:.1f}%) | "
                f"epoch~{_traditional_epoch(global_step):.3f} | "
                f"Loss: {loss_value:.4f} | "
                f"Speed: {steps_per_sec:.2f} steps/s | "
                f"data_time: {avg_data_time:.3f}s | "
                f"model_time: {avg_model_time:.3f}s | "
                f"ETA: {eta_str}"
            )
            _write_loss_log()

        if (
            eval_steps is not None
            and eval_steps > 0
            and (global_step % eval_steps == 0 or global_step == total_steps)
        ):
            _run_validation(global_step)
            if is_main_process:
                print("-" * 80)

    # 5) 仅在主进程保存最终模型参数和 loss 记录
    if is_main_process:
        save_path = output_dir / "qwen_value_final.pt"
        # DDP 包装下需要取 .module
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save(model_to_save.state_dict(), save_path)
        print(f"训练完成，模型已保存到: {save_path}")
        
        # 保存最终的 loss 记录
        try:
            with open(loss_log_file, "w") as f:
                json.dump({
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }, f, indent=2)
            print(f"Loss 记录已保存到: {loss_log_file}")
            print(f"  - 训练 loss 记录数: {len(train_losses)}")
            print(f"  - 验证 loss 记录数: {len(val_losses)}")
        except Exception as e:
            print(f"  -> 警告: 保存最终 loss 记录失败: {e}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
