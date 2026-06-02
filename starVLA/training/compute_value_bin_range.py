"""
单独计算数据集的 value return 的 min/max，用于确定 bin 范围。

用法：
    python starVLA/training/compute_value_bin_range.py \
        --data_root_dir /mnt/workspace/datasets \
        --data_mix sq_agi_beta \
        --output_json value_bin_range.json \
        --sample_size 1000 \
        --gamma 1.0 \
        --big_negative 100.0
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.value_targets_wrapper import (
    augment_traj_df_success_for_returns,
    compute_rewards_and_returns_from_traj,
    load_episode_success_from_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_bin_range(
    data_root_dir: str,
    data_mix: str,
    sample_size: int = 1000,
    gamma: float = 1.0,
    big_negative: float = 100.0,
    success_col: str = "episode_success",
) -> Dict[str, float]:
    """
    计算数据集的 return min/max。

    Args:
        data_root_dir: 数据集根目录
        data_mix: 数据集混合名称（在 mixtures.py 中定义）
        sample_size: 采样轨迹数量（用于估计，如果为 -1 则使用全部）
        gamma: discount factor
        big_negative: 失败 episode 的惩罚值
        success_col: success 列名

    Returns:
        dict: {"bin_min": float, "bin_max": float, "num_trajs_sampled": int, "num_steps_sampled": int}
    """
    logger.info(f"Loading dataset mixture: {data_mix}")
    
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise ValueError(
            f"Unknown data mixture: {data_mix}. "
            f"Available: {list(DATASET_NAMED_MIXTURES.keys())}"
        )
    
    mixture = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"Mixture contains {len(mixture)} datasets")
    
    # 确保 data_root_dir 是 Path 对象
    data_root_dir = Path(data_root_dir)
    
    # 加载所有数据集
    all_datasets = []
    for dataset_name, weight, robot_type in mixture:
        logger.info(f"Loading dataset: {dataset_name} (robot_type={robot_type})")
        ds = make_LeRobotSingleDataset(
            data_root_dir=data_root_dir,
            data_name=dataset_name,
            robot_type=robot_type,
            delete_pause_frame=False,
            data_cfg={},
        )
        all_datasets.append(ds)
        logger.info(f"  - Loaded {len(ds)} steps from {len(ds.trajectory_ids)} trajectories")

    episode_success_maps = [
        load_episode_success_from_jsonl(ds.dataset_path) for ds in all_datasets
    ]
    
    # 合并所有轨迹 ID 和长度
    all_traj_ids = []
    all_traj_lens = []
    dataset_offsets = [0]  # 每个数据集在合并列表中的起始索引
    
    for ds in all_datasets:
        dataset_offsets.append(dataset_offsets[-1] + len(ds.trajectory_ids))
        all_traj_ids.extend(ds.trajectory_ids)
        all_traj_lens.extend(ds.trajectory_lengths)
    
    total_trajs = len(all_traj_ids)
    logger.info(f"Total trajectories across all datasets: {total_trajs}")
    
    # 决定采样数量
    if sample_size == -1:
        actual_sample_size = total_trajs
        logger.info("Using all trajectories (sample_size=-1)")
    else:
        actual_sample_size = min(sample_size, total_trajs)
        logger.info(f"Sampling {actual_sample_size} trajectories (out of {total_trajs})")
    
    # 随机采样轨迹索引
    sampled_indices = np.random.choice(
        total_trajs,
        size=actual_sample_size,
        replace=False,
    )
    
    # 计算采样轨迹的 return
    all_returns = []
    num_steps_sampled = 0
    
    logger.info("Computing returns for sampled trajectories...")
    for idx in tqdm(sampled_indices, desc="Processing trajectories"):
        # 确定这个轨迹属于哪个数据集
        ds_idx = 0
        for i, offset in enumerate(dataset_offsets[1:], start=0):
            if idx < offset:
                ds_idx = i
                break
        
        traj_id = all_traj_ids[idx]
        ds = all_datasets[ds_idx]
        
        # 获取轨迹数据（与 value 训练一致：支持 success / jsonl 标签）
        traj_df = augment_traj_df_success_for_returns(
            ds.get_trajectory_data(int(traj_id)),
            int(traj_id),
            success_col,
            episode_success_maps[ds_idx],
        )
        _, returns = compute_rewards_and_returns_from_traj(
            traj_df,
            success_col=success_col,
            gamma=gamma,
            big_negative=big_negative,
        )
        
        all_returns.append(returns)
        num_steps_sampled += len(returns)
    
    # 计算 min/max
    all_returns_concat = np.concatenate(all_returns, axis=0)
    bin_min = float(all_returns_concat.min())
    bin_max = float(all_returns_concat.max())
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Bin Range Statistics:")
    logger.info(f"  - Trajectories sampled: {actual_sample_size} / {total_trajs}")
    logger.info(f"  - Steps sampled: {num_steps_sampled}")
    logger.info(f"  - Return min: {bin_min:.6f}")
    logger.info(f"  - Return max: {bin_max:.6f}")
    logger.info(f"  - Return range: {bin_max - bin_min:.6f}")
    logger.info(f"  - Return mean: {all_returns_concat.mean():.6f}")
    logger.info(f"  - Return std: {all_returns_concat.std():.6f}")
    logger.info(f"{'='*60}\n")
    
    return {
        "bin_min": bin_min,
        "bin_max": bin_max,
        "num_trajs_sampled": actual_sample_size,
        "num_trajs_total": total_trajs,
        "num_steps_sampled": num_steps_sampled,
        "gamma": gamma,
        "big_negative": big_negative,
        "data_mix": data_mix,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute value bin range (min/max) for dataset"
    )
    
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="数据集根目录",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        required=True,
        help="数据集混合名称（在 mixtures.py 中定义）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="value_bin_range.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="采样轨迹数量（用于估计 min/max，-1 表示使用全部）",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor",
    )
    parser.add_argument(
        "--big_negative",
        type=float,
        default=100.0,
        help="失败 episode 的惩罚值",
    )
    parser.add_argument(
        "--success_col",
        type=str,
        default="episode_success",
        help="Success 列名",
    )
    
    args = parser.parse_args()
    
    # 计算 bin 范围
    result = compute_bin_range(
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        sample_size=args.sample_size,
        gamma=args.gamma,
        big_negative=args.big_negative,
        success_col=args.success_col,
    )
    
    # 保存到 JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved bin range to: {output_path}")
    logger.info(f"  bin_min: {result['bin_min']:.6f}")
    logger.info(f"  bin_max: {result['bin_max']:.6f}")


if __name__ == "__main__":
    main()
