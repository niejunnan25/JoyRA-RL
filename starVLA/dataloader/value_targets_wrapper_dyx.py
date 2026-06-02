from typing import Dict, Tuple, Optional, Sequence, List
import logging
import time
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    LeRobotMixtureDataset,
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT3_EPISODE_FILENAME,
)
from starVLA.dataloader.value_targets_wrapper import (
    augment_traj_df_success_for_returns,
    compute_normalized_returns_from_traj,
    compute_rewards_and_returns_from_traj,
    load_episode_success_from_jsonl,
)

logger = logging.getLogger(__name__)


class LeRobotWithValueTarget(Dataset):
    """
    一个“外包”的 Dataset：
      - 包装现有的 LeRobotSingleDataset（不修改原实现）
      - 在 __init__ 时预先为每条 trajectory 计算 per-step return
      - 在 __getitem__ 时为 sample 动态添加 "value_target"

    使用方式示例（伪代码）::

        base_ds = make_LeRobotSingleDataset(...)
        value_ds = LeRobotWithValueTarget(
            base_ds,
            gamma=1.0,
            big_negative=100.0,
            success_col="episode_success",
        )
        sample = value_ds[0]
        sample["value_target"]  # 标量，作为 QwenValue 的监督信号
    """

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        gamma: float = 1.0,
        big_negative: float = 100.0,
        success_col: str = "episode_success",
        num_bins: int = 201,
        bin_min: Optional[float] = None,
        bin_max: Optional[float] = None,
        sample_size: int = 1000,
        bin_margin: float = 0.1,
        returns_cache_path: Optional[str] = None,
        normalize_returns: bool = False,
        normalize_returns_per_task: bool = False,
        normalize_use_big_negative_in_denom: bool = True,
    ) -> None:
        """
        Args:
            bin_min: 固定 bin 最小值（C51 风格）。如果为 None，则从数据中统计（数据驱动）。
            bin_max: 固定 bin 最大值（C51 风格）。如果为 None，则从数据中统计（数据驱动）。
            sample_size: 数据驱动模式下，用于估计 min/max 的采样轨迹数量（默认 1000）。
            bin_margin: 数据驱动模式下，在估计的 min/max 基础上添加的 margin 比例（默认 0.1，即 10%）。
            returns_cache_path: Return 缓存文件路径。如果提供且文件存在，将尝试加载缓存的 return 值。
                               如果文件不存在，计算完 return 后将保存到此路径。
            normalize_returns: 如果为 True，使用归一化 return 计算（值在 [-1, 0] 范围内），
                               并自动设置 bin_min=-1.0, bin_max=0.0。此时不需要手动设置 bin 范围。
        """
        super().__init__()
        self.base = base_dataset
        self.gamma = float(gamma)
        self.big_negative = float(big_negative)
        self.success_col = success_col
        self.num_bins = int(num_bins)
        self.returns_cache_path = returns_cache_path
        # 两个模式：
        # - normalize_returns=True 且 normalize_returns_per_task=False: per-episode 归一化
        # - normalize_returns_per_task=True: 按 task（更准确说按同一任务簇）共享 denom 归一化
        self.normalize_returns_per_task = bool(normalize_returns_per_task)
        self.normalize_returns = bool(normalize_returns or normalize_returns_per_task)
        # 是否在归一化分母中显式加入 big_negative（即使用 H + big_negative）
        self.normalize_use_big_negative_in_denom = bool(
            normalize_use_big_negative_in_denom
        )
        
        # 如果启用归一化，自动设置 bin 范围
        if self.normalize_returns:
            if bin_min is not None or bin_max is not None:
                logger.warning(
                    "[LeRobotWithValueTarget] normalize_returns=True, "
                    "ignoring user-specified bin_min/bin_max, using [-1.0, 0.0]"
                )
            bin_min = -1.0
            bin_max = 0.0
            if self.normalize_returns_per_task:
                logger.info(
                    "[LeRobotWithValueTarget] Using per-task normalized returns mode: "
                    "returns will be in [-1.0, 0.0] range, bin_min=-1.0, bin_max=0.0, "
                    "denom is max episode length per task."
                )
            else:
                logger.info(
                    "[LeRobotWithValueTarget] Using per-episode normalized returns mode: "
                    "returns will be in [-1.0, 0.0] range, bin_min=-1.0, bin_max=0.0"
                )

        # 如果需要按 task 归一化，预先构建「每条 trajectory 对应的 denom」
        # 这里我们直接从 episodes 元数据中读取 (episode_index, length, task_index/tasks)
        # 以 task_index 或任务文本为 key 统计该 task 的 max episode length。
        self._denom_per_traj: Dict[int, int] = {}
        if self.normalize_returns_per_task:
            try:
                self._build_task_denom_map()
            except Exception as e:
                logger.error(
                    f"[LeRobotWithValueTarget] Failed to build per-task denom map, "
                    f"fallback to per-episode normalization. Error: {e}"
                )
                self.normalize_returns_per_task = False

        # 为每条 trajectory 预先计算 return 序列
        # key: trajectory_id -> np.ndarray[T]
        self._returns_per_traj: Dict[int, np.ndarray] = {}
        self._episode_success_from_jsonl = load_episode_success_from_jsonl(
            self.base.dataset_path
        )

        # 尝试从缓存加载 return 值
        if self.returns_cache_path is not None:
            cache_path = Path(self.returns_cache_path)
            if cache_path.exists():
                logger.info(f"[LeRobotWithValueTarget] 尝试从缓存加载 return 值: {cache_path}")
                if self._load_returns_cache(cache_path):
                    logger.info(
                        f"[LeRobotWithValueTarget] 成功从缓存加载 {len(self._returns_per_traj)} 条轨迹的 return 值"
                    )
                else:
                    logger.warning(
                        f"[LeRobotWithValueTarget] 缓存加载失败或参数不匹配，将重新计算 return 值"
                    )
                    self._returns_per_traj = {}
        
        # 确定 bin 范围：支持两种模式
        # 模式1（C51 风格）：固定范围（用户指定 bin_min/bin_max）
        # 模式2（数据驱动）：从采样数据中估计 min/max
        if bin_min is not None and bin_max is not None:
            # 固定范围模式（C51 风格）
            self._bin_min = float(bin_min)
            self._bin_max = float(bin_max)
            if self._bin_max <= self._bin_min:
                raise ValueError(f"bin_max ({bin_max}) must be > bin_min ({bin_min})")
            logger.info(
                f"[LeRobotWithValueTarget] Using fixed bin range: "
                f"bin_min={self._bin_min:.4f}, bin_max={self._bin_max:.4f}"
            )
        else:
            # 数据驱动模式：先采样估计 min/max
            total_trajs = len(self.base.trajectory_ids)
            actual_sample_size = min(sample_size, total_trajs)
            
            logger.info(
                f"[LeRobotWithValueTarget] Estimating bin range from {actual_sample_size} "
                f"sampled trajectories (out of {total_trajs} total trajectories)..."
            )
            
            # 采样轨迹 ID（随机采样）
            sampled_indices = np.random.choice(
                len(self.base.trajectory_ids),
                size=actual_sample_size,
                replace=False,
            )
            sampled_traj_ids = [self.base.trajectory_ids[i] for i in sampled_indices]
            sampled_traj_lens = [self.base.trajectory_lengths[i] for i in sampled_indices]
            
            # 计算采样轨迹的 return（带进度显示）
            sampled_returns = []
            sample_start_time = time.time()
            for idx, (traj_id, traj_len) in enumerate(zip(sampled_traj_ids, sampled_traj_lens)):
                traj_df = self._traj_df_for_return_computation(
                    self.base.get_trajectory_data(int(traj_id)), int(traj_id)
                )
                if self.normalize_returns:
                    # per-task 模式下，优先使用同一 task 的共享 denom；否则退化为 per-episode
                    denom = None
                    if self.normalize_returns_per_task and int(traj_id) in self._denom_per_traj:
                        denom = self._denom_per_traj[int(traj_id)]
                    _, returns = compute_normalized_returns_from_traj(
                        traj_df,
                        success_col=self.success_col,
                        big_negative=self.big_negative,
                        denom=denom,
                        use_big_negative_in_denom=self.normalize_use_big_negative_in_denom,
                    )
                else:
                    _, returns = compute_rewards_and_returns_from_traj(
                        traj_df,
                        success_col=self.success_col,
                        gamma=self.gamma,
                        big_negative=self.big_negative,
                    )
                sampled_returns.append(returns)
                
                # 每处理 10% 或每 100 个轨迹打印一次进度
                if (idx + 1) % max(1, actual_sample_size // 10) == 0 or (idx + 1) % 100 == 0 or (idx + 1) == actual_sample_size:
                    elapsed = time.time() - sample_start_time
                    speed = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (actual_sample_size - idx - 1) / speed if speed > 0 else 0
                    logger.info(
                        f"[LeRobotWithValueTarget] 采样计算进度: {idx + 1}/{actual_sample_size} "
                        f"({(idx + 1) / actual_sample_size * 100:.1f}%) | "
                        f"速度: {speed:.1f} traj/s | "
                        f"ETA: {int(eta // 60):02d}:{int(eta % 60):02d}"
                    )
            
            # 从采样数据估计 min/max
            sampled_returns_concat = np.concatenate(sampled_returns, axis=0)
            estimated_min = float(sampled_returns_concat.min())
            estimated_max = float(sampled_returns_concat.max())
            
            # 添加 margin 以防万一
            range_size = estimated_max - estimated_min
            margin = range_size * bin_margin
            self._bin_min = estimated_min - margin
            self._bin_max = estimated_max + margin
            
            # 避免除零
            if self._bin_max <= self._bin_min:
                self._bin_max = self._bin_min + 1.0
            
            logger.info(
                f"[LeRobotWithValueTarget] Estimated bin range from {actual_sample_size} samples:\n"
                f"  - Sampled min: {estimated_min:.4f}, max: {estimated_max:.4f}\n"
                f"  - Final bin_min: {self._bin_min:.4f}, bin_max: {self._bin_max:.4f} "
                f"(margin={bin_margin*100:.1f}%)\n"
                f"  - Bin delta: {(self._bin_max - self._bin_min) / max(self.num_bins - 1, 1):.4f}"
            )
        
        self._bin_delta = (self._bin_max - self._bin_min) / max(self.num_bins - 1, 1)
        
        # 为所有轨迹计算 return（用于 __getitem__）
        total_trajs = len(self.base.trajectory_ids)
        logger.info(
            f"[LeRobotWithValueTarget] Computing returns for all {total_trajs} trajectories..."
        )
        
        compute_start_time = time.time()
        processed_count = 0
        for idx, (traj_id, traj_len) in enumerate(zip(self.base.trajectory_ids, self.base.trajectory_lengths)):
            if int(traj_id) not in self._returns_per_traj:
                traj_df = self._traj_df_for_return_computation(
                    self.base.get_trajectory_data(int(traj_id)), int(traj_id)
                )
                if self.normalize_returns:
                    denom = None
                    if self.normalize_returns_per_task and int(traj_id) in self._denom_per_traj:
                        denom = self._denom_per_traj[int(traj_id)]
                    _, returns = compute_normalized_returns_from_traj(
                        traj_df,
                        success_col=self.success_col,
                        big_negative=self.big_negative,
                        denom=denom,
                        use_big_negative_in_denom=self.normalize_use_big_negative_in_denom,
                    )
                else:
                    _, returns = compute_rewards_and_returns_from_traj(
                        traj_df,
                        success_col=self.success_col,
                        gamma=self.gamma,
                        big_negative=self.big_negative,
                    )
                if len(returns) != int(traj_len):
                    raise ValueError(
                        f"Returns length ({len(returns)}) != trajectory_length ({traj_len}) "
                        f"for traj_id={traj_id}"
                    )
                self._returns_per_traj[int(traj_id)] = returns
                processed_count += 1
            
            # 每处理 10% 或每 1000 个轨迹打印一次进度
            if (idx + 1) % max(1, total_trajs // 10) == 0 or (idx + 1) % 1000 == 0 or (idx + 1) == total_trajs:
                elapsed = time.time() - compute_start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_trajs - idx - 1) / speed if speed > 0 else 0
                total_steps_computed = sum(len(r) for r in self._returns_per_traj.values())
                logger.info(
                    f"[LeRobotWithValueTarget] Return 计算进度: {idx + 1}/{total_trajs} "
                    f"({(idx + 1) / total_trajs * 100:.1f}%) | "
                    f"已计算轨迹: {processed_count} | "
                    f"总步数: {total_steps_computed} | "
                    f"速度: {speed:.1f} traj/s | "
                    f"ETA: {int(eta // 60):02d}:{int(eta % 60):02d}"
                )
        
        logger.info(
            f"[LeRobotWithValueTarget] Initialization complete. "
            f"Total trajectories: {len(self._returns_per_traj)}, "
            f"Total steps: {sum(len(r) for r in self._returns_per_traj.values())}"
        )
        
        # 如果提供了缓存路径且缓存不存在，保存计算好的 return 值
        if self.returns_cache_path is not None:
            cache_path = Path(self.returns_cache_path)
            if not cache_path.exists():
                logger.info(f"[LeRobotWithValueTarget] 保存 return 缓存到: {cache_path}")
                self._save_returns_cache(cache_path)
                logger.info(f"[LeRobotWithValueTarget] Return 缓存保存完成")

    def _build_task_denom_map(self) -> None:
        """
        按 task 统计「该 task 的最大 episode 长度」，并为每条 trajectory 生成对应的 denom。
        这里的 task 粒度：
          - 优先使用 episodes.jsonl/episodes.parquet 中的 tasks[0] 或 task 文本（子任务描述）
          - 如果没有 tasks / task 文本，则退化为使用 task_index
        """
        dataset_path: Path = self.base.dataset_path

        task_to_max_len: Dict[str, int] = {}
        traj_task_pairs: List[Tuple[int, str, int]] = []

        episodes_jsonl = dataset_path / LE_ROBOT_EPISODE_FILENAME
        episodes_parquet_paths = list(dataset_path.glob(LE_ROBOT3_EPISODE_FILENAME))

        if episodes_jsonl.exists():
            # v2.0 风格：episodes.jsonl
            with open(episodes_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ep = json.loads(line)
                    traj_id = int(ep.get("episode_index"))
                    length = int(ep.get("length", 0))

                    # 优先使用子任务描述（tasks / task），从而实现「按子任务」归一化
                    if "tasks" in ep and ep["tasks"]:
                        task_id = str(ep["tasks"][0])
                    elif "task" in ep:
                        task_id = str(ep["task"])
                    elif "task_index" in ep:
                        task_id = f"idx_{ep['task_index']}"
                    else:
                        # 实在没有 task 信息，就退化为「每个 episode 自己是一个 task」
                        task_id = f"ep_{traj_id}"

                    traj_task_pairs.append((traj_id, task_id, length))
                    prev = task_to_max_len.get(task_id, 0)
                    if length > prev:
                        task_to_max_len[task_id] = length

        elif episodes_parquet_paths:
            # v3.0 风格：meta/episodes/*/*.parquet
            for p in episodes_parquet_paths:
                df_ep = pd.read_parquet(p)
                for _, ep in df_ep.iterrows():
                    traj_id = int(ep["episode_index"])
                    length = int(ep["length"])

                    # 优先使用子任务描述（tasks / task）
                    if "tasks" in ep and ep["tasks"]:
                        # tasks 可能是 list，也可能是 str
                        tasks_val = ep["tasks"]
                        if isinstance(tasks_val, (list, tuple)) and len(tasks_val) > 0:
                            task_id = str(tasks_val[0])
                        else:
                            task_id = str(tasks_val)
                    elif "task" in ep:
                        task_id = str(ep["task"])
                    elif "task_index" in ep:
                        task_id = f"idx_{int(ep['task_index'])}"
                    else:
                        task_id = f"ep_{traj_id}"

                    traj_task_pairs.append((traj_id, task_id, length))
                    prev = task_to_max_len.get(task_id, 0)
                    if length > prev:
                        task_to_max_len[task_id] = length
        else:
            raise FileNotFoundError(
                f"Neither episodes.jsonl nor episodes parquet files found under {dataset_path}"
            )

        if not traj_task_pairs:
            raise RuntimeError(
                f"No episode records found when building per-task denom map for dataset {dataset_path}"
            )

        # 第二遍：为每个 trajectory 生成 denom（max(1, max_len(task) - 1)）
        self._denom_per_traj = {}
        for traj_id, task_id, _length in traj_task_pairs:
            max_len = task_to_max_len.get(task_id, 1)
            denom = max(1, max_len - 1)
            self._denom_per_traj[int(traj_id)] = denom

        logger.info(
            f"[LeRobotWithValueTarget] Built per-task denom map: "
            f"{len(task_to_max_len)} unique tasks, {len(self._denom_per_traj)} trajectories."
        )

    def _traj_df_for_return_computation(
        self, traj_df: pd.DataFrame, traj_id: int
    ) -> pd.DataFrame:
        return augment_traj_df_success_for_returns(
            traj_df,
            traj_id,
            self.success_col,
            self._episode_success_from_jsonl,
        )
    
    def _save_returns_cache(self, cache_path: Path) -> None:
        """
        保存 return 缓存到文件。
        
        Args:
            cache_path: 缓存文件路径
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备缓存数据
        cache_data = {
            "returns_per_traj": self._returns_per_traj,
            "metadata": {
                "gamma": self.gamma,
                "big_negative": self.big_negative,
                "success_col": self.success_col,
                "normalize_returns": self.normalize_returns,
                "normalize_returns_per_task": self.normalize_returns_per_task,
                "trajectory_ids": list(self.base.trajectory_ids),
                "trajectory_lengths": {
                    int(traj_id): int(length)
                    for traj_id, length in zip(self.base.trajectory_ids, self.base.trajectory_lengths)
                },
            },
        }
        
        # 使用 pickle 保存（支持 numpy array）
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
    
    def _load_returns_cache(self, cache_path: Path) -> bool:
        """
        从文件加载 return 缓存。
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            bool: 是否成功加载（如果参数不匹配，返回 False）
        """
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            
            # 验证元数据是否匹配
            metadata = cache_data.get("metadata", {})
            if (
                abs(metadata.get("gamma", 0) - self.gamma) > 1e-6
                or abs(metadata.get("big_negative", 0) - self.big_negative) > 1e-6
                or metadata.get("success_col", "") != self.success_col
                or bool(metadata.get("normalize_returns", False)) != bool(self.normalize_returns)
                or bool(metadata.get("normalize_returns_per_task", False)) != bool(self.normalize_returns_per_task)
            ):
                logger.warning(
                    f"[LeRobotWithValueTarget] 缓存参数不匹配: "
                    f"gamma={metadata.get('gamma')} vs {self.gamma}, "
                    f"big_negative={metadata.get('big_negative')} vs {self.big_negative}, "
                    f"success_col={metadata.get('success_col')} vs {self.success_col}"
                )
                return False
            
            # 验证轨迹 ID 是否匹配
            cached_traj_ids = set(metadata.get("trajectory_ids", []))
            current_traj_ids = set(self.base.trajectory_ids)
            
            if cached_traj_ids != current_traj_ids:
                logger.warning(
                    f"[LeRobotWithValueTarget] 缓存轨迹 ID 不匹配: "
                    f"缓存中有 {len(cached_traj_ids)} 条轨迹，当前数据集有 {len(current_traj_ids)} 条轨迹"
                )
                # 只加载匹配的轨迹
                common_traj_ids = cached_traj_ids & current_traj_ids
                if len(common_traj_ids) == 0:
                    logger.error("[LeRobotWithValueTarget] 没有匹配的轨迹，无法使用缓存")
                    return False
                else:
                    logger.info(
                        f"[LeRobotWithValueTarget] 找到 {len(common_traj_ids)} 条匹配的轨迹，将只加载这些轨迹的缓存"
                    )
                    returns_per_traj = cache_data.get("returns_per_traj", {})
                    self._returns_per_traj = {
                        int(traj_id): returns_per_traj[int(traj_id)]
                        for traj_id in common_traj_ids
                        if int(traj_id) in returns_per_traj
                    }
                    return True
            
            # 完全匹配，加载所有 return
            self._returns_per_traj = cache_data.get("returns_per_traj", {})
            
            # 验证轨迹长度是否匹配
            trajectory_lengths = metadata.get("trajectory_lengths", {})
            for traj_id, returns in self._returns_per_traj.items():
                cached_len = trajectory_lengths.get(traj_id)
                if cached_len is not None and len(returns) != cached_len:
                    logger.warning(
                        f"[LeRobotWithValueTarget] 轨迹 {traj_id} 的长度不匹配: "
                        f"缓存中 {cached_len}，return 数组 {len(returns)}"
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"[LeRobotWithValueTarget] 加载缓存失败: {e}")
            return False

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        """
        调用底层 LeRobotSingleDataset 得到 sample，然后根据 (traj_id, base_index)
        查表取出对应的 return 作为 value_target，加入 sample。
        """
        sample = self.base[index]  # dict(action=..., image=..., language=...) 等

        # 兼容键名：如果底层用 "language"，而上层模型期望 "lang"，这里做一个别名映射
        if "lang" not in sample and "language" in sample:
            sample["lang"] = sample["language"]

        # 利用 LeRobotSingleDataset 的 all_steps 将 index 映射到 (trajectory_id, base_index)
        traj_id, base_index = self.base.all_steps[index]
        traj_id = int(traj_id)
        base_index = int(base_index)

        if traj_id not in self._returns_per_traj:
            raise KeyError(f"Trajectory id {traj_id} not found in returns cache.")

        returns_traj = self._returns_per_traj[traj_id]
        if base_index < 0 or base_index >= len(returns_traj):
            raise IndexError(
                f"base_index {base_index} out of range for trajectory {traj_id} "
                f"with length {len(returns_traj)}"
            )

        value_target = float(returns_traj[base_index])
        sample["value_target"] = value_target  # 保留连续 return 以便调试 / 可视化

        # 将连续 return 离散化为 [0, num_bins-1] 的整数 bin（与 QwenValue 对应）
        rel = (value_target - self._bin_min) / self._bin_delta
        bin_idx = int(np.round(rel))
        
        # Clip 到有效范围（类似 C51 的投影操作）
        original_bin_idx = bin_idx
        bin_idx = max(0, min(self.num_bins - 1, bin_idx))
        
        # 如果超出范围，记录警告（但只在第一次发生时打印，避免日志过多）
        if original_bin_idx != bin_idx:
            if not hasattr(self, "_out_of_range_warned"):
                self._out_of_range_warned = set()
            
            if value_target < self._bin_min:
                if "below_min" not in self._out_of_range_warned:
                    logger.warning(
                        f"[LeRobotWithValueTarget] Return value {value_target:.2f} < bin_min {self._bin_min:.2f}. "
                        f"Clamping to bin_index=0. Consider expanding bin_min range."
                    )
                    self._out_of_range_warned.add("below_min")
            elif value_target > self._bin_max:
                if "above_max" not in self._out_of_range_warned:
                    logger.warning(
                        f"[LeRobotWithValueTarget] Return value {value_target:.2f} > bin_max {self._bin_max:.2f}. "
                        f"Clamping to bin_index={self.num_bins - 1}. Consider expanding bin_max range."
                    )
                    self._out_of_range_warned.add("above_max")
        
        sample["value_bin"] = bin_idx

        return sample


class LeRobotMixtureWithValueTarget(Dataset):
    """
    针对 value training 的混合数据集版本：
      - 内部使用 LeRobotMixtureDataset 进行「按数据集权重」采样
      - 同时为每个底层 LeRobotSingleDataset 构造一个 LeRobotWithValueTarget，
        复用其中的 return 计算与缓存逻辑
      - __getitem__ 会按 Mixture 的采样规则选一个 (dataset, trajectory_id, base_index)，
        然后：
          * 用底层 dataset / transforms 构造 obs（image / lang / action / state）
          * 用对应的 LeRobotWithValueTarget 提供 value_target / value_bin
    """

    def __init__(
        self,
        mixture_datasets: Sequence[Tuple[LeRobotSingleDataset, float, Optional[str]]],
        gamma: float = 1.0,
        big_negative: float = 100.0,
        success_col: str = "episode_success",
        num_bins: int = 201,
        bin_min: Optional[float] = None,
        bin_max: Optional[float] = None,
        sample_size: int = 1000,
        bin_margin: float = 0.1,
        data_cfg: Optional[dict] = None,
        seed: int = 42,
        train_split: float = 1.0,
        mode: str = "train",
        normalize_returns: bool = False,
        normalize_returns_per_task: bool = False,
        normalize_use_big_negative_in_denom: bool = True,
    ) -> None:
        """
        Args:
            mixture_datasets: 底层数据集 + 采样权重 + 可选的 return 缓存路径，
                形如 [(LeRobotSingleDataset, weight, returns_cache_path), ...]
            train_split: 训练集比例（0-1之间）。剩余部分会被平均划分为验证集和测试集。
                        例如 train_split=0.8，则约 80% 训练，10% 验证，10% 测试。
                        默认 1.0（全部作为训练集，不划分 val/test）。
            mode: "train"、"val" 或 "test"，用于选择使用训练 / 验证 / 测试集部分。
            normalize_returns: 如果为 True，使用归一化 return 计算（值在 [-1, 0] 范围内），
                               并自动设置 bin_min=-1.0, bin_max=0.0（per-episode 模式）。
            normalize_returns_per_task: 如果为 True，则按 task 的最大 episode 长度做归一化，
                               仍然落在 [-1, 0]，并同样设置 bin_min=-1.0, bin_max=0.0。
            其余参数与 LeRobotWithValueTarget 基本一致。
        """
        super().__init__()
        self.gamma = float(gamma)
        self.big_negative = float(big_negative)
        self.success_col = success_col
        self.num_bins = int(num_bins)
        self.train_split = float(train_split)
        self.mode = mode
        self.normalize_returns_per_task = bool(normalize_returns_per_task)
        self.normalize_use_big_negative_in_denom = bool(
            normalize_use_big_negative_in_denom
        )

        # === 1) 构建底层 MixtureDataset，用于「按权重」采样 step ===
        data_mixture = [(ds, weight) for (ds, weight, _cache) in mixture_datasets]
        self.mixture = LeRobotMixtureDataset(
            data_mixture=data_mixture,
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=True,
            seed=seed,
            data_cfg=data_cfg or {},
        )

        # === 2) 为每个底层数据集构造一个 LeRobotWithValueTarget，用于 return + bin 逻辑 ===
        self._value_wrappers: List[LeRobotWithValueTarget] = []
        # 通过 id(dataset) 建立到 wrapper 的索引映射，__getitem__ 中可 O(1) 查到
        self._dataset_id_to_idx: Dict[int, int] = {}

        for idx, (base_ds, _weight, returns_cache_path) in enumerate(mixture_datasets):
            vw = LeRobotWithValueTarget(
                base_dataset=base_ds,
                gamma=self.gamma,
                big_negative=self.big_negative,
                success_col=self.success_col,
                num_bins=self.num_bins,
                bin_min=bin_min,
                bin_max=bin_max,
                sample_size=sample_size,
                bin_margin=bin_margin,
                returns_cache_path=returns_cache_path,
                normalize_returns=normalize_returns,
                normalize_returns_per_task=self.normalize_returns_per_task,
                normalize_use_big_negative_in_denom=self.normalize_use_big_negative_in_denom,
            )
            self._value_wrappers.append(vw)
            self._dataset_id_to_idx[id(base_ds)] = idx

        # 约定：如果传入了固定 bin_min / bin_max，则所有 wrapper 使用同一套范围
        # 这里直接从第一个 wrapper 读取，方便调试
        ref_vw = self._value_wrappers[0]
        self._bin_min = ref_vw._bin_min
        self._bin_max = ref_vw._bin_max
        self._bin_delta = ref_vw._bin_delta
        
        # 计算数据集总长度和划分点
        self._total_length = len(self.mixture)
        # 训练集长度
        self._train_length = int(self._total_length * self.train_split)
        # 剩余部分平均分成验证集和测试集
        remaining = max(self._total_length - self._train_length, 0)
        self._val_length = remaining // 2
        self._test_length = remaining - self._val_length

    def __len__(self) -> int:
        # 根据 mode 返回训练 / 验证 / 测试集的长度
        if self.mode == "train":
            return self._train_length
        elif self.mode == "val":
            return self._val_length
        elif self.mode == "test":
            return self._test_length
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Expected 'train', 'val' or 'test'.")

    def set_epoch(self, epoch: int) -> None:
        """外部（如 DistributedSampler）可以调用此函数，以便 Mixture 使用新的 epoch Seed。"""
        if hasattr(self.mixture, "set_epoch"):
            self.mixture.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        """
        基本流程参考 LeRobotMixtureDataset.__getitem__：
          1. 用 mixture.sample_step(index) 采样 (dataset, trajectory_id, base_index)
          2. 用底层 dataset 的 get_step_data + transforms 构造 obs dict
          3. 用对应的 LeRobotWithValueTarget 提供 value_target / value_bin
        """
        # === 0) 根据 mode 调整索引：
        #   - 训练集: [0, train_length)
        #   - 验证集: [train_length, train_length + val_length)
        #   - 测试集: [train_length + val_length, total_length)
        if self.mode == "train":
            actual_index = index
        elif self.mode == "val":
            actual_index = self._train_length + index
        elif self.mode == "test":
            actual_index = self._train_length + self._val_length + index
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Expected 'train', 'val' or 'test'.")
        
        # === 1) 从 Mixture 中采样一个 step ===
        # 这里直接调用 LeRobotMixtureDataset.sample_step，保证与原实现一致的采样分布
        dataset, trajectory_id, step = self.mixture.sample_step(actual_index)

        # === 2) 构造观测（基本复制自 LeRobotMixtureDataset.__getitem__） ===
        # 注意：我们假设底层 dataset 是 LeRobotSingleDataset
        raw_data = dataset.get_step_data(trajectory_id, step)
        data = dataset.transforms(raw_data)

        # --- 处理 video -> image 列表（与 MixtureDataset 中逻辑等价）---
        prim_images = []
        wrist_views = []

        if "video" in data and not any(k.startswith("video.") for k in data.keys()):
            # ConcatTransform 已经把多路视频合在 "video" 里
            video_data = data["video"]  # [T, V, H, W, C]
            num_views = video_data.shape[1]

            for view_idx, video_key in enumerate(dataset.modality_keys["video"]):
                # 取第 0 帧，指定视角
                image = video_data[0, view_idx, :, :, :]  # [H, W, C]
                if "wrist" not in video_key and "hand" not in video_key:
                    prim_images.append(image)
                else:
                    wrist_views.append(image)
        else:
            # 未 concat，按每个 video key 单独取帧
            from PIL import Image

            for video_key in dataset.modality_keys["video"]:
                image = data[video_key][0]  # 取第 0 帧
                image = Image.fromarray(image).resize((224, 224))
                if "wrist" not in video_key and "hand" not in video_key:
                    prim_images.append(image)
                else:
                    wrist_views.append(image)

        all_images = prim_images + wrist_views

        # --- language ---
        language = data[dataset.modality_keys["language"][0]][0]

        # --- action / state，基本复制 MixtureDataset 的逻辑 ---
        def to_numpy_float16(x):
            if hasattr(x, "cpu") and hasattr(x, "numpy"):
                return x.cpu().numpy().astype(np.float16)
            else:
                return x.astype(np.float16)

        # action
        # 优先使用已经拼接好的 data["action"]（例如经过 ConcatStateAction 之后），
        # 只有在没有该键时，才根据 per-key action.* 再拼接一次。
        if "action" in data:
            action = to_numpy_float16(data["action"])
        else:
            action_list = []
            for action_key in dataset.modality_keys["action"]:
                action_list.append(data[action_key])
            action = np.concatenate(action_list, axis=1).astype(np.float16)

        # state（如果需要）
        state = None
        if self.mixture.data_cfg is not None and self.mixture.data_cfg.get("include_state", False) not in [
            "False",
            False,
        ]:
            if "state" in data and not any(k.startswith("state.") for k in data.keys()):
                state = to_numpy_float16(data["state"])
            else:
                state_list = []
                for state_key in dataset.modality_keys["state"]:
                    state_list.append(data[state_key])
                state = np.concatenate(state_list, axis=1).astype(np.float16)

        sample = dict(action=action, image=all_images, lang=language, embodiment_tag=dataset.tag_index)
        if state is not None:
            sample["state"] = state

        # === 3) 查表获取 value_target / value_bin ===
        ds_idx = self._dataset_id_to_idx.get(id(dataset), None)
        if ds_idx is None:
            raise KeyError("Internal error: dataset not found in _dataset_id_to_idx.")

        vw = self._value_wrappers[ds_idx]
        traj_id_int = int(trajectory_id)
        base_index_int = int(step)

        if traj_id_int not in vw._returns_per_traj:
            raise KeyError(f"Trajectory id {traj_id_int} not found in returns cache for this dataset.")

        returns_traj = vw._returns_per_traj[traj_id_int]
        if base_index_int < 0 or base_index_int >= len(returns_traj):
            raise IndexError(
                f"base_index {base_index_int} out of range for trajectory {traj_id_int} "
                f"with length {len(returns_traj)}"
            )

        value_target = float(returns_traj[base_index_int])
        sample["value_target"] = value_target

        # 与 LeRobotWithValueTarget 中相同的 bin 离散化与截断逻辑
        rel = (value_target - vw._bin_min) / vw._bin_delta
        bin_idx = int(np.round(rel))

        original_bin_idx = bin_idx
        bin_idx = max(0, min(vw.num_bins - 1, bin_idx))

        if original_bin_idx != bin_idx:
            if not hasattr(vw, "_out_of_range_warned"):
                vw._out_of_range_warned = set()

            if value_target < vw._bin_min:
                if "below_min" not in vw._out_of_range_warned:
                    logger.warning(
                        f"[LeRobotMixtureWithValueTarget] Return value {value_target:.2f} < bin_min {vw._bin_min:.2f}. "
                        f"Clamping to bin_index=0. Consider expanding bin_min range."
                    )
                    vw._out_of_range_warned.add("below_min")
            elif value_target > vw._bin_max:
                if "above_max" not in vw._out_of_range_warned:
                    logger.warning(
                        f"[LeRobotMixtureWithValueTarget] Return value {value_target:.2f} > bin_max {vw._bin_max:.2f}. "
                        f"Clamping to bin_index={vw.num_bins - 1}. Consider expanding bin_max range."
                    )
                    vw._out_of_range_warned.add("above_max")

        sample["value_bin"] = bin_idx

        return sample

