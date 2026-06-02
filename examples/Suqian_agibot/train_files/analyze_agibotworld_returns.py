"""
统计 AgiBotWorld 数据集中 return 的分布

用法:
    python analyze_agibotworld_returns.py \
        --data_root_dir /mnt/workspace/datasets \
        --output_dir ./agibotworld_analysis \
        --gamma 1.0 \
        --big_negative 100.0
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.value_targets_wrapper import compute_rewards_and_returns_from_traj


def analyze_dataset_returns(
    data_root_dir: str,
    dataset_name: str = "AgiBotWorld-Beta-LeRobot",
    robot_type: str = "agibot_genie1_joint",
    gamma: float = 1.0,
    big_negative: float = 100.0,
    success_col: str = "episode_success",
    output_dir: str = "./agibotworld_analysis",
    force_recompute: bool = False,
):
    """
    分析数据集的 return 分布
    
    Args:
        data_root_dir: 数据集根目录
        dataset_name: 数据集名称
        robot_type: 机器人类型
        gamma: discount factor
        big_negative: 失败 episode 的惩罚值
        success_col: success 列名
        output_dir: 输出目录
    """
    data_root_dir = Path(data_root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有保存的数据
    stats_file = output_dir / "return_statistics.json"
    data_file = output_dir / "return_data.npz"
    
    use_saved_data = False
    if not force_recompute and stats_file.exists() and data_file.exists():
        print(f"发现已保存的统计数据，检查配置...")
        print(f"  统计文件: {stats_file}")
        print(f"  数据文件: {data_file}")
        
        # 加载统计信息检查配置
        with open(stats_file, "r") as f:
            saved_stats = json.load(f)
        
        saved_config = saved_stats.get("config", {})
        if (saved_config.get("gamma") == gamma and 
            saved_config.get("big_negative") == big_negative and
            saved_config.get("success_col") == success_col):
            print(f"  配置匹配，使用已保存的数据")
            use_saved_data = True
            
            # 加载数据
            stats = saved_stats
            data = np.load(data_file)
            all_returns = data["all_returns"]
            all_rewards = data["all_rewards"]
            traj_returns = data["traj_returns"]
            traj_lengths = data["traj_lengths"]
        else:
            print(f"  ⚠️  警告: 保存的数据配置与当前参数不匹配！")
            print(f"     保存的配置: gamma={saved_config.get('gamma')}, big_negative={saved_config.get('big_negative')}, success_col={saved_config.get('success_col')}")
            print(f"     当前配置: gamma={gamma}, big_negative={big_negative}, success_col={success_col}")
            print(f"     将重新计算...")
    
    if not use_saved_data:
        print(f"加载数据集: {dataset_name}")
        print(f"  路径: {data_root_dir / dataset_name}")
        print(f"  机器人类型: {robot_type}")
        
        # 加载数据集
        dataset = make_LeRobotSingleDataset(
            data_root_dir=data_root_dir,
            data_name=dataset_name,
            robot_type=robot_type,
            delete_pause_frame=False,
            data_cfg={},
        )
        
        num_trajs = len(dataset.trajectory_ids)
        print(f"\n数据集信息:")
        print(f"  - 轨迹数量: {num_trajs}")
        print(f"  - 总步数: {len(dataset)}")
        print(f"  - 平均轨迹长度: {len(dataset) / num_trajs:.2f}")
        
        # 计算所有轨迹的 return
        print(f"\n计算所有轨迹的 return...")
        all_returns = []
        all_rewards = []
        traj_returns = []  # 每个轨迹的平均 return
        traj_lengths = []
        success_count = 0
        fail_count = 0
        
        for traj_id in tqdm(dataset.trajectory_ids, desc="处理轨迹"):
            traj_df = dataset.get_trajectory_data(int(traj_id))
            traj_length = len(traj_df)
            traj_lengths.append(traj_length)
            
            # 检查是否成功
            if success_col in traj_df.columns:
                is_success = bool(traj_df.iloc[0][success_col])
            else:
                is_success = True  # 默认成功
            
            if is_success:
                success_count += 1
            else:
                fail_count += 1
            
            # 计算 return
            rewards, returns = compute_rewards_and_returns_from_traj(
                traj_df,
                success_col=success_col,
                gamma=gamma,
                big_negative=big_negative,
            )
            
            all_returns.extend(returns.tolist())
            all_rewards.extend(rewards.tolist())
            traj_returns.append(float(returns.mean()))  # 每个轨迹的平均 return
        
        all_returns = np.array(all_returns)
        all_rewards = np.array(all_rewards)
        traj_returns = np.array(traj_returns)
        traj_lengths = np.array(traj_lengths)
        
        # 统计信息
        stats = {
            "dataset_name": dataset_name,
            "num_trajectories": int(num_trajs),
            "total_steps": int(len(all_returns)),
            "success_count": int(success_count),
            "fail_count": int(fail_count),
            "success_rate": float(success_count / num_trajs) if num_trajs > 0 else 0.0,
            "return_stats": {
                "min": float(all_returns.min()),
                "max": float(all_returns.max()),
                "mean": float(all_returns.mean()),
                "median": float(np.median(all_returns)),
                "std": float(all_returns.std()),
                "percentile_1": float(np.percentile(all_returns, 1)),
                "percentile_5": float(np.percentile(all_returns, 5)),
                "percentile_25": float(np.percentile(all_returns, 25)),
                "percentile_75": float(np.percentile(all_returns, 75)),
                "percentile_95": float(np.percentile(all_returns, 95)),
                "percentile_99": float(np.percentile(all_returns, 99)),
            },
            "traj_return_stats": {
                "min": float(traj_returns.min()),
                "max": float(traj_returns.max()),
                "mean": float(traj_returns.mean()),
                "median": float(np.median(traj_returns)),
                "std": float(traj_returns.std()),
            },
            "traj_length_stats": {
                "min": int(traj_lengths.min()),
                "max": int(traj_lengths.max()),
                "mean": float(traj_lengths.mean()),
                "median": float(np.median(traj_lengths)),
                "std": float(traj_lengths.std()),
            },
            "reward_stats": {
                "min": float(all_rewards.min()),
                "max": float(all_rewards.max()),
                "mean": float(all_rewards.mean()),
                "median": float(np.median(all_rewards)),
                "std": float(all_rewards.std()),
            },
            "config": {
                "gamma": gamma,
                "big_negative": big_negative,
                "success_col": success_col,
            },
        }
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"Return 分布统计 (所有 step)")
    print(f"{'='*80}")
    print(f"最小值:     {stats['return_stats']['min']:.4f}")
    print(f"最大值:     {stats['return_stats']['max']:.4f}")
    print(f"均值:       {stats['return_stats']['mean']:.4f}")
    print(f"中位数:     {stats['return_stats']['median']:.4f}")
    print(f"标准差:     {stats['return_stats']['std']:.4f}")
    print(f"1% 分位数:  {stats['return_stats']['percentile_1']:.4f}")
    print(f"5% 分位数:  {stats['return_stats']['percentile_5']:.4f}")
    print(f"25% 分位数: {stats['return_stats']['percentile_25']:.4f}")
    print(f"75% 分位数: {stats['return_stats']['percentile_75']:.4f}")
    print(f"95% 分位数: {stats['return_stats']['percentile_95']:.4f}")
    print(f"99% 分位数: {stats['return_stats']['percentile_99']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"轨迹平均 Return 分布")
    print(f"{'='*80}")
    print(f"最小值:     {stats['traj_return_stats']['min']:.4f}")
    print(f"最大值:     {stats['traj_return_stats']['max']:.4f}")
    print(f"均值:       {stats['traj_return_stats']['mean']:.4f}")
    print(f"中位数:     {stats['traj_return_stats']['median']:.4f}")
    print(f"标准差:     {stats['traj_return_stats']['std']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Episode 统计")
    print(f"{'='*80}")
    print(f"成功数量:   {stats['success_count']}")
    print(f"失败数量:   {stats['fail_count']}")
    print(f"成功率:     {stats['success_rate']:.2%}")
    
    # 保存统计信息到 JSON（仅在重新计算时保存）
    if not use_saved_data:
        stats_file = output_dir / "return_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n统计信息已保存到: {stats_file}")
        
        # 保存原始数据（可选，用于进一步分析）
        data_file = output_dir / "return_data.npz"
        np.savez(
            data_file,
            all_returns=all_returns,
            all_rewards=all_rewards,
            traj_returns=traj_returns,
            traj_lengths=traj_lengths,
        )
        print(f"原始数据已保存到: {data_file}")
    
    # 绘制分布图
    print(f"\n绘制分布图...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Return distribution histogram (all steps)
    axes[0, 0].hist(all_returns, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(stats['return_stats']['mean'], color='red', linestyle='--', label=f'Mean: {stats["return_stats"]["mean"]:.2f}')
    axes[0, 0].axvline(stats['return_stats']['median'], color='green', linestyle='--', label=f'Median: {stats["return_stats"]["median"]:.2f}')
    axes[0, 0].set_xlabel("Return")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Return Distribution (All Steps)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Return distribution histogram (log scale)
    axes[0, 1].hist(all_returns, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel("Return")
    axes[0, 1].set_ylabel("Frequency (Log Scale)")
    axes[0, 1].set_title("Return Distribution (Log Scale)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Trajectory average return distribution
    axes[0, 2].hist(traj_returns, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 2].axvline(stats['traj_return_stats']['mean'], color='red', linestyle='--', label=f'Mean: {stats["traj_return_stats"]["mean"]:.2f}')
    axes[0, 2].axvline(stats['traj_return_stats']['median'], color='green', linestyle='--', label=f'Median: {stats["traj_return_stats"]["median"]:.2f}')
    axes[0, 2].set_xlabel("Trajectory Average Return")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Trajectory Average Return Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Reward distribution
    axes[1, 0].hist(all_rewards, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].axvline(stats['reward_stats']['mean'], color='red', linestyle='--', label=f'Mean: {stats["reward_stats"]["mean"]:.2f}')
    axes[1, 0].set_xlabel("Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Trajectory length distribution
    axes[1, 1].hist(traj_lengths, bins=50, alpha=0.7, edgecolor='black', color='teal')
    axes[1, 1].axvline(stats['traj_length_stats']['mean'], color='red', linestyle='--', label=f'Mean: {stats["traj_length_stats"]["mean"]:.2f}')
    axes[1, 1].set_xlabel("Trajectory Length (Steps)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Trajectory Length Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Return vs trajectory length scatter plot
    scatter = axes[1, 2].scatter(traj_lengths, traj_returns, alpha=0.5, s=10)
    axes[1, 2].set_xlabel("Trajectory Length")
    axes[1, 2].set_ylabel("Trajectory Average Return")
    axes[1, 2].set_title("Return vs Trajectory Length")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_file = output_dir / "return_distribution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"分布图已保存到: {plot_file}")
    
    print(f"\n分析完成！所有结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="分析 AgiBotWorld 数据集的 return 分布")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/mnt/workspace/datasets",
        help="数据集根目录",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="AgiBotWorld-Beta-LeRobot",
        help="数据集名称",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="agibot_genie1_joint",
        help="机器人类型",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./agibotworld_analysis",
        help="输出目录",
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
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="强制重新计算，即使已存在保存的数据",
    )
    
    args = parser.parse_args()
    
    analyze_dataset_returns(
        data_root_dir=args.data_root_dir,
        dataset_name=args.dataset_name,
        robot_type=args.robot_type,
        gamma=args.gamma,
        big_negative=args.big_negative,
        success_col=args.success_col,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
