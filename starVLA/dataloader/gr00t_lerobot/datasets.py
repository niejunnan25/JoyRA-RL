# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import os
import hashlib
import json, torch
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import os, random
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag, EMBODIMENT_TAG_MAPPING
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.state_action_key_synonyms import (
    STATE_ACTION_CANONICAL_TO_RAW_SUBKEY,
    STATE_ACTION_KEY_SYNONYMS,
    resolve_metadata_subkey,
)

from starVLA.dataloader.gr00t_lerobot.relative_action_stats.relative_stats import generate_rel_stats

from functools import partial
from typing import Tuple, List
import pickle

# LeRobot v2.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_SUBTASKS_FILENAME = "meta/subtasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
EPSILON = 5e-4

# State/action synonym helpers: ``state_action_key_synonyms`` (metadata keys are canonical).


def _normalize_state_action_modality_cfg(cfg: dict) -> dict:
    """Rename state/action modality sub-keys to canonical names (same rules as mixture merge)."""
    normalized: dict[str, dict] = {}
    for k, v in cfg.items():
        canonical_k = STATE_ACTION_KEY_SYNONYMS.get(k, k)
        if canonical_k not in normalized:
            normalized[canonical_k] = v
    return normalized


def _rename_state_action_statistics_keys(stats_by_subkey: dict) -> dict:
    """Keep statistics dict keys in sync with `_normalize_state_action_modality_cfg`."""
    return {STATE_ACTION_KEY_SYNONYMS.get(k, k): v for k, v in stats_by_subkey.items()}


def _resolve_lerobot_state_action_subkey(le_cfg: dict, canonical_subkey: str) -> str:
    """Map a canonical sub-key to the name present in on-disk LeRobot modality.json."""
    if canonical_subkey in le_cfg:
        return canonical_subkey
    raw = STATE_ACTION_CANONICAL_TO_RAW_SUBKEY.get(canonical_subkey)
    if raw is not None and raw in le_cfg:
        return raw
    raise KeyError(
        f"canonical sub-key {canonical_subkey!r} not found in LeRobot splits "
        f"(also tried raw synonym); available: {list(le_cfg.keys())}"
    )


def _state_action_disk_key_candidates(full_key: str) -> list[str]:
    """Candidates for `lerobot_modality_meta.get_key_meta` (canonical then raw synonym)."""
    parts = full_key.split(".", 1)
    if len(parts) != 2:
        return [full_key]
    modality, subkey = parts
    if modality not in ("state", "action"):
        return [full_key]
    out = [full_key]
    raw = STATE_ACTION_CANONICAL_TO_RAW_SUBKEY.get(subkey)
    if raw is not None and raw != subkey:
        out.append(f"{modality}.{raw}")
    return out


#  LeRobot v3.0 dataset file names
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"
from starVLA.dataloader.gr00t_lerobot.embodiment_configs import MODALITY_CONFIGS
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.types import ActionRepresentation





def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        # 检查数据是否为空或无效
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
            )
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg = None,
        enable_relative_action: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        self.enable_relative_action = enable_relative_action
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict lerobot version
        self._lerobot_version = self.data_cfg.get("lerobot_version", "v2.0")  # self._indict_lerobot_version(**kwargs)

        self.delete_pause_frame = delete_pause_frame

        # Optional temporal downsampling along time dimension.
        # If data_cfg contains "frame_stride", we keep every `frame_stride`-th step in each trajectory
        # when building `all_steps` (see `_get_all_steps_single_process`).
        self.step_stride = 1
        if isinstance(self.data_cfg, dict):
            frame_stride = self.data_cfg.get("frame_stride", 1)
            try:
                self.step_stride = max(1, int(frame_stride))
            except (TypeError, ValueError):
                print(f"[LeRobotSingleDataset] Invalid frame_stride={frame_stride}, fallback to 1")
                self.step_stride = 1

        self._skip_invalid_subtask_frames = bool(
            self.data_cfg.get("skip_invalid_subtask_frames", False)
        ) if isinstance(self.data_cfg, dict) else False
        if self._skip_invalid_subtask_frames:
            print(
                "[LeRobotSingleDataset] skip_invalid_subtask_frames=True: "
                "steps 将不包含 subtask_index < 0 的帧（需 parquet 含 subtask_index 列）"
            )

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
            self.tag_index = EMBODIMENT_TAG_MAPPING[self.tag]
        else:
            self.tag = embodiment_tag
            self.tag_index = EMBODIMENT_TAG_MAPPING[EmbodimentTag.NEW_EMBODIMENT.value]


        self._metadata = self._get_metadata(EmbodimentTag(self.tag), self.enable_relative_action)

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self._subtasks = self._get_subtasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    @property
    def subtasks(self) -> pd.DataFrame | None:
        """Optional subtask index -> instruction table from meta/subtasks.jsonl."""
        return self._subtasks

    def _get_subtasks(self) -> pd.DataFrame | None:
        """Load meta/subtasks.jsonl when present (subtask_index -> text in column 'task')."""
        if self._lerobot_version != "v2.0":
            return None
        subtasks_path = self.dataset_path / LE_ROBOT_SUBTASKS_FILENAME
        if not subtasks_path.exists():
            return None
        with open(subtasks_path, "r") as f:
            rows = [json.loads(line) for line in f]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if "subtask_index" not in df.columns or "task" not in df.columns:
            return None
        df["task"] = df["task"].apply(self._process_task_text)
        return df.set_index("subtask_index")

    @staticmethod
    def _scalar_int_from_series_cell(val) -> int:
        if isinstance(val, (int, np.integer, float, np.floating)):
            return int(val)
        return int(val.item())

    def _get_metadata(self, embodiment_tag: EmbodimentTag, enable_relative_action: bool = True) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Args:
            embodiment_tag: The embodiment tag for the dataset.
            enable_relative_action: Whether to compute relative action statistics.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            parquet_files_filtered = []
            #  parquet_files[0].name = "episode_033675.parquet" is broken file
            for pf in parquet_files:
                if "episode_033675.parquet" in pf.name:
                    continue
                parquet_files_filtered.append(pf)

            le_statistics = calculate_dataset_statistics(parquet_files_filtered)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)

        
        # Optionally generate relative action statistics
        if enable_relative_action:
            rel_stats = generate_rel_stats(self.dataset_path, EmbodimentTag(self.tag))
        else:
            rel_stats = {}
            print(f"Skipping relative action statistics calculation (enable_relative_action=False)")

        
        #这段代码的作用是将 LeRobot 原始格式的扁平统计数据 转换为 按语义分组的层次化统计数据。
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # Align single-dataset metadata/statistics keys with mixture merge (gripper→hand, pose→arm).
        for our_modality in ["state", "action"]:
            simplified_modality_meta[our_modality] = _normalize_state_action_modality_cfg(
                simplified_modality_meta[our_modality]
            )
            dataset_statistics[our_modality] = _rename_state_action_statistics_keys(
                dataset_statistics[our_modality]
            )

        # Initialize relative_action statistics only if enabled
        dataset_statistics["relative_action"] = {}
        if enable_relative_action:
            for key, value in rel_stats.items():
                dataset_statistics["relative_action"][key] = value

        #import pdb; pdb.set_trace()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    # TODO auto map key? just map to file_path and file_from_index
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/observation.images.wrist/from_timestamp": episode["videos/observation.images.wrist/from_timestamp"],
                    }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            # 这里应该可以直接读取到 save index 信息
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """
        # Create a hash key based on configuration to ensure cache validity
        config_key = self._get_steps_config_key()

        # skip_invalid_subtask_frames 会依赖 parquet 中 subtask_index，缓存易过期且易与
        # steps_data_index.pkl 混淆；该模式下不使用任何 steps pkl（不读、不写）。
        skip_no_steps_cache = getattr(self, "_skip_invalid_subtask_frames", False)
        meta_dir = self.dataset_path / "meta"
        steps_path = meta_dir / f"steps_{config_key}.pkl"
        legacy_steps_path = meta_dir / "steps_data_index.pkl"

        if not skip_no_steps_cache:
            # Try to load the config-specific cache first. Fall back to the
            # legacy cache only when its config key matches; otherwise stride
            # changes can silently reuse stale step indices.
            try:
                for candidate in (steps_path, legacy_steps_path):
                    if not candidate.exists():
                        continue
                    with open(candidate, "rb") as f:
                        cached_data = pickle.load(f)
                    cached_config_key = cached_data.get("config_key")
                    if cached_config_key == config_key:
                        return cached_data["steps"]
                    print(
                        f"Cached steps config mismatch for {candidate}: "
                        f"cached={cached_config_key}, expected={config_key}. Recomputing..."
                    )
            except (FileNotFoundError, pickle.PickleError, KeyError) as e:
                print(f"Failed to load cached steps: {e}")
                print("Computing steps from scratch...")
        else:
            print(
                "[LeRobotSingleDataset] skip_invalid_subtask_frames=True: "
                "不使用 meta/steps*.pkl，每次启动重新扫描 steps。"
            )

        # Compute steps using single process
        all_steps = self._get_all_steps_single_process()

        if not skip_no_steps_cache:
            # Cache the computed steps under a config-specific filename.
            try:
                cache_data = {
                    "config_key": config_key,
                    "steps": all_steps,
                    "num_trajectories": len(self.trajectory_ids),
                    "total_steps": len(all_steps),
                    "computed_timestamp": pd.Timestamp.now().isoformat(),
                    "delete_pause_frame": self.delete_pause_frame,
                    "step_stride": getattr(self, "step_stride", 1),
                }

                # Ensure the meta directory exists
                steps_path.parent.mkdir(parents=True, exist_ok=True)

                with open(steps_path, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Cached steps saved to {steps_path}")
            except Exception as e:
                print(f"Failed to cache steps: {e}")

        return all_steps

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
            "step_stride": getattr(self, "step_stride", 1),
            "skip_invalid_subtask_frames": getattr(self, "_skip_invalid_subtask_frames", False),
        }
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #


    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0

        has_language_modality = ("language" in self.modality_keys) and (len(self.modality_keys["language"]) > 0)
        lang_key = self.modality_keys["language"][0] if has_language_modality else None

        for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc="Getting All Step (fast)",
        ):
            try:
                # 关键：只读一次（如果你必须依赖 parquet 的 language mapping）
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                else:
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)
                self.curr_traj_data = data

                if has_language_modality:
                    # 只检查一次：base_index=0
                    try:
                        language_instruction = self.get_language(trajectory_id, lang_key, 0)
                        valid = bool(language_instruction) and len(language_instruction) > 0 and (language_instruction[0] not in ["", None])
                    except Exception:
                        valid = False

                    if not valid:
                        skipped_trajectories += 1
                        continue

                # Temporal downsampling: use step_stride to subsample steps along each trajectory.
                # step_stride=1 -> keep every frame; >1 -> keep every N frames.
                step_stride = max(1, getattr(self, "step_stride", 1))
                if getattr(self, "_skip_invalid_subtask_frames", False) and "subtask_index" in data.columns:
                    col = data["subtask_index"]
                    for i in range(0, int(trajectory_length), step_stride):
                        if i >= len(data):
                            break
                        iv = self._scalar_int_from_series_cell(col.iloc[i])
                        if iv >= 0:
                            all_steps.append((trajectory_id, i))
                else:
                    all_steps.extend(
                        (trajectory_id, i)
                        for i in range(0, int(trajectory_length), step_stride)
                    )
                processed_trajectories += 1

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                continue

        print(f"Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} trajectories")
        print(f"Total steps: {len(all_steps)}")
        return all_steps


    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])

        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']

        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy()
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _process_task_text(self, task_text: str) -> str:
        """Process task text to extract English part after @ and validate.
        
        Args:
            task_text: Original task text, may contain Chinese and English separated by @
            
        Returns:
            Processed task text (English part only), or None if invalid
        """
        if not isinstance(task_text, str):
            return None

        # Extract English part after @
        if "@" in task_text:
            parts = task_text.split("@", 1)
            if len(parts) > 1:
                english_part = parts[1].strip()
            else:
                english_part = task_text.strip()
        else:
            # If no @, use the whole text
            english_part = task_text.strip()

        # Filter out invalid instructions
        invalid_keywords = ["null", "unqualified", "qualified"]
        if english_part.lower() in invalid_keywords:
            return None

        # Return empty string if result is empty
        if not english_part:
            return None

        return english_part

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            # Process task text: extract English part and filter invalid ones
            df["task"] = df["task"].apply(self._process_task_text)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # 把索引变成一列，列名通常为 'index'
            df = df.rename(columns={'index': 'task'})  # 把 'index' 列重命名为 'task'
            df = df[['task_index', 'task']]  # 调整列顺序
            # Process task text: extract English part and filter invalid ones
            df["task"] = df["task"].apply(self._process_task_text)
            return df
    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Config uses canonical names; on-disk modality.json may still use synonyms (e.g. left_gripper).
                last_e: Exception | None = None
                matched = False
                for cand in _state_action_disk_key_candidates(key):
                    try:
                        self.lerobot_modality_meta.get_key_meta(cand)
                        matched = True
                        break
                    except Exception as e:
                        last_e = e
                if not matched:
                    raise ValueError(
                        ERROR_MSG_HEADER
                        + f"Unable to find key {key} in modality metadata:\n{last_e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        data = self.get_step_data(trajectory_id, base_index)

        # Process all video keys dynamically
        images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]

            # Apply image cropping if enabled and the video key is base_view
            # Note: crop_obs_camera functionality has been removed

            # image = Image.fromarray(image).resize((224, 224))
            images.append(image)

        # Get language and action data
        # Language instruction should already be validated in _get_all_steps_single_process
        language_list = data[self.modality_keys["language"][0]]
        # Get the first language instruction (should be valid after filtering)
        language = language_list[0] if language_list and language_list[0] is not None else "Perform the task."

        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1)

        return dict(action=action, image=images, language=language)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id     # ✅ 关键：让缓存真的生效
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                df = pd.read_parquet(parquet_path)
                self.curr_traj_id = trajectory_id
                self.curr_traj_data = df
                return df
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else: #TODO check detail later
            chunk_index = self.get_episode_chunk(trajectory_id)

            file_index = self.get_episode_file_index(trajectory_id)
            # file_from_index = self.get_episode_file_from_index(trajectory_id)


            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)

            # filter by trajectory_id
            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()

            # fix timestamp from epis index to file index
            from_timestamp = self.trajectory_ids_to_metadata[trajectory_id]["videos/observation.images.wrist/from_timestamp"]
            episode_data["timestamp"] = episode_data["timestamp"] + from_timestamp

            return episode_data


    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size
    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]


    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        # import ipdb; ipdb.set_trace()
        video_timestamp = timestamp[step_indices]
        # try:
        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend, # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )
        # except Exception as e:
        #     print("\n[VIDEO READ FAIL]")
        #     print("video_path:", video_path.as_posix())
        #     print("trajectory_id:", trajectory_id, "key:", key)
        #     print("base_index:", base_index)
        #     print("step_indices:", step_indices[:10], "...", "len=", len(step_indices))
        #     print("video_timestamp:", video_timestamp[:10], "...", "min/max=", float(np.min(video_timestamp)), float(np.max(video_timestamp)))
        #     raise

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Canonical sub-key (matches self.metadata.modalities after _get_metadata normalization).
        subkey = key.replace(modality + ".", "")
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_subkey = _resolve_lerobot_state_action_subkey(le_state_or_action_cfg, subkey)
        le_key = le_state_or_action_cfg[le_subkey].original_key
        if le_key is None:
            le_key = le_subkey
        # Get the data array, shape: (T, D)

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        # We only materialize the rows we need (step_indices), with padding logic.
        col = self.curr_traj_data[le_key]     # pandas Series, each element is array-like

        # Determine feature slice for this subkey
        le_indices = np.arange(
            le_state_or_action_cfg[le_subkey].start,
            le_state_or_action_cfg[le_subkey].end,
        )

        # Padding / normalization metadata uses canonical sub-keys (after _get_metadata rename).
        modality_meta = getattr(self.metadata.modalities, modality)
        meta_subkey = resolve_metadata_subkey(subkey, modality_meta)
        state_or_action_cfg = modality_meta[meta_subkey]
        pad_mode = "first_last" if state_or_action_cfg.absolute else "zero"

        # Helper: fetch one row -> np.ndarray (D,)
        def _row_to_1d(idx: int) -> np.ndarray:
            v = col.iloc[idx]
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            # scalar -> (1,)
            if v.ndim == 0:
                v = v.reshape(1)
            # if (D,) OK; if (1, D) or something odd, flatten
            if v.ndim > 1:
                v = v.reshape(-1)
            return v

        T = int(max_length)

        # Pre-fetch pad rows if needed
        if pad_mode == "first_last":
            first_row = _row_to_1d(0)[le_indices]
            last_row = _row_to_1d(T - 1)[le_indices]

        out = []
        for si in step_indices:
            if 0 <= int(si) < T:
                out.append(_row_to_1d(int(si))[le_indices])
            else:
                if pad_mode == "zero":
                    out.append(np.zeros(len(le_indices), dtype=np.float32))
                else:
                    # first_last
                    out.append(first_row if int(si) < 0 else last_row)

        data_out = np.stack(out, axis=0)

        return data_out

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
                若列为 task_index/subtask_index 且值为 -1，对应位置为空串「不做相邻帧回填」；训练请配合
                data_cfg skip_invalid_subtask_frames 从采样索引中排除无效帧。
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations (integer ids used to index tasks.jsonl / subtasks.jsonl)
        ann_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]
            v = self._scalar_int_from_series_cell(value)
            ann_indices.append(v)

        # Get language instructions and filter out None values (invalid instructions)
        if original_key == "subtask_index":
            if self._subtasks is None:
                raise ValueError(
                    f"Language key {key} uses subtask_index but {LE_ROBOT_SUBTASKS_FILENAME} is missing "
                    f"under {self.dataset_path}"
                )
            lookup = self._subtasks
        else:
            lookup = self.tasks

        language_list: list[str] = []
        for ai in ann_indices:
            if ai < 0:
                language_list.append("")
                continue
            if ai not in lookup.index:
                raise KeyError(
                    f"annotation lookup id {ai} not in table index for {key} "
                    f"(dataset {self.dataset_name})"
                )
            language_list.append(lookup.loc[ai]["task"])
        language_list = [lang for lang in language_list if lang is not None]

        return language_list

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        import pdb; pdb.set_trace()

        # Build the data structure to save
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                #import pdb; pdb.set_trace()

                combined_action_stats = combine_modality_stats(filtered_action_stats)

                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys()
                )
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats

                 ### 相对动作加入统计值中（仅在启用时）
                if self.enable_relative_action:
                    action_config = MODALITY_CONFIGS[tag]["action"]
                    if action_config.action_configs is not None:
                        relative_action_keys = [
                            key
                            for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
                            if action_config.rep == ActionRepresentation.RELATIVE
                        ]
                        relative_action_stats = self.metadata.statistics.relative_action
                        for key in relative_action_keys:
                            # Convert DatasetStatisticalValues to dict for JSON serialization
                            stats_obj = relative_action_stats[key]
                            stats_dict = {
                                "mean": stats_obj.mean.tolist() if hasattr(stats_obj.mean, 'tolist') else stats_obj.mean,
                                "std": stats_obj.std.tolist() if hasattr(stats_obj.std, 'tolist') else stats_obj.std,
                                "max": stats_obj.max.tolist() if hasattr(stats_obj.max, 'tolist') else stats_obj.max,
                                "min": stats_obj.min.tolist() if hasattr(stats_obj.min, 'tolist') else stats_obj.min,
                                "q01": stats_obj.q01.tolist() if hasattr(stats_obj.q01, 'tolist') else stats_obj.q01,
                                "q99": stats_obj.q99.tolist() if hasattr(stats_obj.q99, 'tolist') else stats_obj.q99,
                            }
                            tag_stats[f"relative_action.{key}"] = stats_dict

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed

                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }

    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object

        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))

    return combined_stats

def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.
    
    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.
        
    Returns:
        list[bool]: List of mask values
    """
    mask = []

    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()

            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True

    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []

    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    # Extract state keys (remove "state." prefix)
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        print(f"Dataset sampling weights: {self._dataset_sampling_weights}")
        print(f"balance_dataset_weights: {self.balance_dataset_weights}")
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        print(f"After balance_dataset_weights: {self._dataset_sampling_weights}")
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum
        print(f'After normalize weights: {self._dataset_sampling_weights}')
        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)

            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum

            self._trajectory_sampling_weights.append(trajectory_sampling_weights)
        # print(f'After trajectory sampling weights: {self._trajectory_sampling_weights}')
        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

        # === Deterministic full-coverage sampling over all (dataset, trajectory_id, step) ===
        # 对于 train/val/test 三种模式，我们都构建一个固定长度的 sampled_steps，
        # 然后在一个 epoch 内按 index 顺序访问时，每个下采样后的 step 恰好被访问一次。
        # 不再在 train 模式下按 index+epoch 在线随机采样，这样可以保证“完整遍历 + 打乱顺序”。
        all_steps: list[tuple[int, int, int]] = []  # (dataset_index, trajectory_id, base_index)
        for ds_idx, dataset in enumerate(self.datasets):
            for (traj_id, base_index) in dataset.all_steps:
                all_steps.append((ds_idx, int(traj_id), int(base_index)))

        # 使用 safe_hash(epoch, seed) 生成可复现的 shuffle
        rng_seed = safe_hash((self.seed, epoch, "mixture_full_scan"))
        rng = np.random.default_rng(rng_seed)
        indices = np.arange(len(all_steps))
        rng.shuffle(indices)

        # 重排后的顺序作为本 epoch 的采样顺序
        self.sampled_steps = [all_steps[i] for i in indices]

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # 现在直接走 set_epoch 构建好的 sampled_steps，实现“完整遍历 + 打乱顺序”
        if not hasattr(self, "sampled_steps"):
            # 兜底：如果外部忘记调用 set_epoch，则按 epoch=0 构建一次
            self.set_epoch(0)

        # index 可能来自外部 DataLoader 的 [0, len(self))，这里做一次取模防御
        if len(self.sampled_steps) == 0:
            raise RuntimeError("LeRobotMixtureDataset.sampled_steps is empty.")

        idx = int(index) % len(self.sampled_steps)
        ds_idx, trajectory_id, base_index = self.sampled_steps[idx]
        dataset = self.datasets[ds_idx]
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True: # @DUG
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)

                # Process video: check if ConcatTransform has been applied
                prim_images = []
                wrist_views = []

                if "video" in data and not any(k.startswith("video.") for k in data.keys()):
                    # Videos have been concatenated by ConcatTransform
                    # data["video"] shape: [T, V, H, W, C] where V is number of views
                    video_data = data["video"]
                    num_views = video_data.shape[1]  # Number of video views

                    for view_idx, video_key in enumerate(dataset.modality_keys["video"]):
                        # Extract the view at view_idx
                        image = video_data[0, view_idx, :, :, :]  # Get first frame [H, W, C]
                        # image = Image.fromarray(image)

                        if "wrist" not in video_key and "hand" not in video_key:
                            prim_images.append(image)
                        else:
                            wrist_views.append(image)
                else:
                    # Videos are still individual keys (no ConcatTransform applied)
                    for video_key in dataset.modality_keys["video"]:
                        image = data[video_key][0]  # Get first frame
                        image = Image.fromarray(image).resize((224, 224))

                        if "wrist" not in video_key and "hand" not in video_key:
                            prim_images.append(image)
                        else:
                            wrist_views.append(image)

                all_images = prim_images + wrist_views

                # Get language data
                language = data[dataset.modality_keys["language"][0]][0]
                def to_numpy_float16(x):
                    if isinstance(x, torch.Tensor):
                        return x.cpu().numpy().astype(np.float16)
                    else:
                        return x.astype(np.float16)
                # Get action data: check if ConcatTransform has been applied
                if "action" in data and not any(k.startswith("action.") for k in data.keys()):
                    # Actions have been concatenated
                    # action = data["action"].astype(np.float16)
                    action = to_numpy_float16(data["action"])
                else:
                    # Individual action keys
                    action = []
                    for action_key in dataset.modality_keys["action"]:
                        action.append(data[action_key])
                    action = np.concatenate(action, axis=1).astype(np.float16)

                # Get state data if needed
                state = None
                if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
                    if "state" in data and not any(k.startswith("state.") for k in data.keys()):
                        # States have been concatenated
                        # state = data["state"].astype(np.float16)
                        state = to_numpy_float16(data["state"])
                    else:
                        # Individual state keys
                        state = []
                        for state_key in dataset.modality_keys["state"]:
                            state.append(data[state_key])
                        state = np.concatenate(state, axis=1).astype(np.float16)

                    return dict(action=action, image=all_images, lang=language, state=state, embodiment_tag=dataset.tag_index)

                return dict(action=action, image=all_images, lang=language, embodiment_tag=dataset.tag_index)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0

        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0

        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0

        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys - use union of all keys from all datasets
        all_modality_keys = set()
        for task_stats in per_task_stats:
            all_modality_keys.update(task_stats.keys())

        # For each modality key, only include datasets that have this key
        for modality in all_modality_keys:
            # Find which datasets have this modality
            datasets_with_modality = []
            weights_for_modality = []

            for task_idx, task_stats in enumerate(per_task_stats):
                if modality in task_stats:
                    datasets_with_modality.append(task_idx)
                    weights_for_modality.append(normalized_weights[task_idx])

            # Skip if no dataset has this modality (shouldn't happen)
            if not datasets_with_modality:
                continue

            # Re-normalize weights for this modality
            weights_for_modality = np.array(weights_for_modality)
            weights_for_modality = weights_for_modality / weights_for_modality.sum()

            # Number of dimensions (from first dataset that has this modality)
            #num_dims = len(per_task_stats[datasets_with_modality[0]][modality]["mean"])

            first_mean = np.array(per_task_stats[datasets_with_modality[0]][modality]["mean"])

            stats_shape = first_mean.shape



            # Initialize accumulators for means and variances
            weighted_means = np.zeros(stats_shape)
            weighted_squares = np.zeros(stats_shape)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for i, task_idx in enumerate(datasets_with_modality):
                w_i = weights_for_modality[i]
                task_stats = per_task_stats[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        merged_metadata = {}

        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        
        # Merge relative_action statistics if available
        relative_action_stats_list = [m["statistics"].get("relative_action", {}) for m in metadata_dicts]
        if any(relative_action_stats_list):  # If at least one dataset has relative_action stats
            # Filter out empty dicts and corresponding weights
            valid_indices = [i for i, stats in enumerate(relative_action_stats_list) if stats]
            valid_relative_action_stats = [relative_action_stats_list[i] for i in valid_indices]
            valid_weights = [dataset_sampling_weights[i] for i in valid_indices]

            #import pdb; pdb.set_trace()
            
            if valid_relative_action_stats:
                dataset_statistics["relative_action"] = LeRobotMixtureDataset.compute_overall_statistics(
                    per_task_stats=valid_relative_action_stats,
                    dataset_sampling_weights=valid_weights,
                    percentile_mixing_method=percentile_mixing_method,
                )
        
        merged_metadata["statistics"] = dataset_statistics

        # ---- Merge modality configs ----
        modality_configs = defaultdict(list)  # store as list of dicts (not set of strings)
        for md in metadata_dicts:
            for modality, configs in md["modalities"].items():
                modality_configs[modality].append(configs)

        # Mapping between semantically equivalent video keys so that datasets
        # with different camera naming conventions can still be merged.
        #
        # NOTE:
        # - Some DatasetMetadata instances store keys with a "video." prefix
        #   (e.g. "video.image_high"), while others store bare keys
        #   (e.g. "image_high"). We therefore provide mappings for BOTH forms.
        # - Canonical form is chosen to be the Agibot/Suqian naming:
        #     image_head_color, image_hand_left, image_hand_right
        # - 一些单视角数据集只有通用 key "image"（或 "video.image"），在这里统一归一到
        #   头部相机 "image_head_color"，便于与 Agibot/Suqian/Robotwin 等多视角数据对齐。
        VIDEO_KEY_SYNONYMS: dict[str, str] = {
            # Robotwin Agilex -> Agibot/Suqian style (bare keys)
            "image_high": "image_head_color",
            "image_left_wrist": "image_hand_left",
            "image_right_wrist": "image_hand_right",
            # 单视角通用名字 -> 头部相机
            "image": "image_head_color",
            # LIBERO converted rollouts use primary_image / wrist_image.
            "primary_image": "image_head_color",
            "wrist_image": "image_hand_right",
            # 一些数据集（例如某些 EgoDex / 自采集数据）使用 "ego_view" 作为单视角名，
            # 语义上也是「第一人称主视角」，这里同样归一到头部相机。
            "ego_view": "image_head_color",
            # Robotwin Agilex -> Agibot/Suqian style (prefixed keys)
            "video.image_high": "video.image_head_color",
            "video.image_left_wrist": "video.image_hand_left",
            "video.image_right_wrist": "video.image_hand_right",
            # 单视角通用名字（带前缀） -> 头部相机
            "video.image": "video.image_head_color",
            # LIBERO converted rollouts with video. prefix.
            "video.primary_image": "video.image_head_color",
            "video.wrist_image": "video.image_hand_right",
            # 带前缀的 ego_view
            "video.ego_view": "video.image_head_color",
        }

        # 统一视角顺序：头部视角必备，左右手视角可选。
        # 这个列表不会改变现有的键名行为，只是提供一个「多视角顺序」的统一规范，
        # 方便上层 collate_fn / 模型根据 metadata 构造 [B, V, C, H, W] + view_mask。
        CANONICAL_VIDEO_VIEWS: list[str] = [
            "image_head_color",   # 必须存在（经由同义词归一化后）
            "image_hand_left",    # 可选
            "image_hand_right",   # 可选
        ]

        def normalize_video_cfg(video_cfg: dict) -> dict:
            """Map synonym keys to a canonical form for comparison/merging."""
            normalized: dict[str, dict] = {}
            for k, v in video_cfg.items():
                canonical_k = VIDEO_KEY_SYNONYMS.get(k, k)
                normalized[canonical_k] = v
            return normalized

        # ---- State / action key synonyms（与模块级 STATE_ACTION_KEY_SYNONYMS /
        # _normalize_state_action_modality_cfg 一致；单数据集 _get_metadata 同步归一）----
        merged_metadata["modalities"] = {}

        for modality, cfg_list in modality_configs.items():
            if modality != "video":
                # 对 state / action 先做 key 同义词归一化，再检查一致性或取交集
                if modality in ("state", "action"):
                    norm_cfg_list = [_normalize_state_action_modality_cfg(c) for c in cfg_list]
                else:
                    norm_cfg_list = cfg_list

                # 默认：所有非 video 模态的配置必须完全一致
                cfg_strs = {json.dumps(c, sort_keys=True) for c in norm_cfg_list}

                # 对于 "state" 和 "action" 模态，混合不同机器人/数据集时，
                # 经常会出现一些数据集多出额外的字段。这里放宽策略：
                # - 如果 state/action 配置不一致，则取「所有数据集的公共 key 的交集」。
                # - 这样可以保证每个数据集都至少具备这些字段。
                if modality in ("state", "action") and len(cfg_strs) > 1:
                    common_keys = set(norm_cfg_list[0].keys())
                    for c in norm_cfg_list[1:]:
                        common_keys &= set(c.keys())

                    merged_cfg: dict[str, dict] = {}
                    for k in sorted(common_keys):
                        # 所有配置中该 key 的定义应当一致，这里直接取第一个
                        merged_cfg[k] = norm_cfg_list[0][k]

                    print(
                        f"[WARN] Multiple {modality} modality configs found; "
                        "using intersection of keys across datasets (after normalizing synonyms): "
                        f"{sorted(list(common_keys))}"
                    )
                    merged_metadata["modalities"][modality] = merged_cfg
                    continue

                # 其它非 video 模态仍然保持严格一致性约束
                assert len(cfg_strs) == 1, (
                    f"Multiple modality configs for modality {modality}: {list(cfg_strs)}"
                )
                merged_metadata["modalities"][modality] = json.loads(next(iter(cfg_strs)))
                continue

            # ---- video: allow different resolutions, unify, and support key synonyms ----
            # 1) normalize key sets so that synonymous keys are treated as identical
            normalized_cfg_list = [normalize_video_cfg(c) for c in cfg_list]

            # 记录每个数据集在「规范化后」真正拥有的视角 key，用于后续 available_views 统计。
            key_sets = [set(c.keys()) for c in normalized_cfg_list]

            # 确保每个数据集在归一化后都至少具备头部视角（单视角数据集通常会被映射到 image_head_color）
            for ks in key_sets:
                assert "image_head_color" in ks, (
                    "Each dataset in the mixture must provide at least a head camera view "
                    "(canonical key 'image_head_color' after VIDEO_KEY_SYNONYMS mapping). "
                    f"Got key set: {sorted(list(ks))}"
                )

            # 统计所有可能出现过的视角（并集），用于上层构建多视角输入顺序。
            all_available_views: set[str] = set().union(*key_sets)
            # 保持一个稳定、有意义的顺序：先按 CANONICAL_VIDEO_VIEWS 排，再加上其它剩余视角。
            ordered_available_views: list[str] = []
            for v in CANONICAL_VIDEO_VIEWS:
                if v in all_available_views:
                    ordered_available_views.append(v)
                    all_available_views.discard(v)
            # 其余非常见视角（如果有）按字典序追加，避免信息丢失。
            ordered_available_views.extend(sorted(all_available_views))

            # 继续沿用原有逻辑：在真正的 video modality 配置里仍然只保留「所有数据集都具备的 key（交集）」，
            # 以保证 transforms / dataset 在访问这些 key 时不会因某个数据集缺失而报错。
            key_sets_tuple = [tuple(sorted(ks)) for ks in key_sets]
            key_set_strs = {json.dumps(ks) for ks in key_sets_tuple}

            # 如果不同数据集的视频 key 集合不同（例如：有的只有 "image"(->image_head_color)，
            # 有的是三视角 image_head_color/image_hand_left/image_hand_right），
            # 尝试取所有数据集共享的交集 key，保证每个数据集都至少有这些视角。
            if len(key_set_strs) > 1:
                common_keys = set(key_sets_tuple[0])
                for ks in key_sets_tuple[1:]:
                    common_keys &= set(ks)

                if common_keys:
                    print(
                        "[WARN] Multiple video key sets found (after normalizing synonyms); "
                        f"using intersection of keys across datasets for actual video modality: {sorted(list(common_keys))}"
                    )
                    # 只保留交集里的 key，后续以这些 key 作为 canonical 视角集合
                    filtered_normalized_cfg_list = []
                    for cfg in normalized_cfg_list:
                        filtered = {k: v for k, v in cfg.items() if k in common_keys}
                        filtered_normalized_cfg_list.append(filtered)
                    normalized_cfg_list = filtered_normalized_cfg_list
                    key_sets_tuple = [tuple(sorted(c.keys())) for c in normalized_cfg_list]
                    key_set_strs = {json.dumps(ks) for ks in key_sets_tuple}
                else:
                    # 如果完全没有交集（极少见），退化为保留第一个数据集的 key 集合，
                    # 并给出明显告警。
                    print(
                        "[WARN] Multiple video key sets found with no common keys "
                        "(after normalizing synonyms); falling back to first dataset's key set "
                        "for actual video modality: "
                        f"{list(key_sets_tuple[0])}"
                    )
                    first_keys = set(key_sets_tuple[0])
                    filtered_normalized_cfg_list = []
                    for cfg in normalized_cfg_list:
                        filtered = {k: v for k, v in cfg.items() if k in first_keys}
                        filtered_normalized_cfg_list.append(filtered)
                    normalized_cfg_list = filtered_normalized_cfg_list
                    key_sets_tuple = [tuple(sorted(c.keys())) for c in normalized_cfg_list]
                    key_set_strs = {json.dumps(ks) for ks in key_sets_tuple}

            # 到这里，不同数据集的视频 key 集合应当已经一致
            assert len(key_set_strs) == 1, (
                f"Multiple video key sets found (after normalization and intersection): {list(key_set_strs)}"
            )

            # 2) pick canonical config by smallest total pixel area (sum over views)
            def total_pixels(video_cfg: dict) -> int:
                s = 0
                for _, v in video_cfg.items():
                    w, h = v["resolution"]
                    s += int(w) * int(h)
                return s

            canonical_normalized = min(normalized_cfg_list, key=total_pixels)

            # 3) build final video config: include both canonical keys and their synonyms
            final_video_cfg: dict[str, dict] = {}
            # first, add canonical keys
            for key, cfg in canonical_normalized.items():
                final_video_cfg[key] = cfg
            # then, for any known alias that maps to these canonical keys, add alias entries
            for alias, canonical in VIDEO_KEY_SYNONYMS.items():
                if canonical in canonical_normalized:
                    final_video_cfg[alias] = canonical_normalized[canonical]

            merged_metadata["modalities"]["video"] = final_video_cfg

            # 额外记录一个「所有数据集中曾经出现过的规范视角列表」，
            # 上层可以用它来构建多视角输入顺序和 view_mask。
            # 这里使用一个保留字段名，避免与原有 schema 冲突。
            merged_metadata["modalities"]["video_available_views"] = ordered_available_views

            # helpful warning
            cfg_strs = {json.dumps(c, sort_keys=True) for c in normalized_cfg_list}
            if len(cfg_strs) > 1:
                print(
                    f"[WARN] Multiple video modality configs found; "
                    f"unified to canonical (smallest resolution) with key synonyms."
                )

        return DatasetMetadata.model_validate(merged_metadata)


    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:

                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats

                    
                    ### 相对动作加入统计值中（检查各数据集是否启用）
                    # Check if any dataset for this tag has relative action enabled
                    any_dataset_has_relative_action = any(
                        dataset.enable_relative_action 
                        for dataset in self.datasets 
                        if dataset.tag == tag
                    )
                    
                    if any_dataset_has_relative_action:
                        action_config = MODALITY_CONFIGS[tag]["action"]
                        if action_config.action_configs is not None:
                            relative_action_keys = [
                                key
                                for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
                                if action_config.rep == ActionRepresentation.RELATIVE
                            ]
                            # Check if relative_action statistics exist
                            if hasattr(merged_metadata.statistics, 'relative_action') and merged_metadata.statistics.relative_action:
                                relative_action_stats = merged_metadata.statistics.relative_action
                                for key in relative_action_keys:
                                    if key in relative_action_stats:
                                        # Convert DatasetStatisticalValues to dict for JSON serialization
                                        stats_obj = relative_action_stats[key]
                                        stats_dict = {
                                            "mean": stats_obj.mean.tolist() if hasattr(stats_obj.mean, 'tolist') else stats_obj.mean,
                                            "std": stats_obj.std.tolist() if hasattr(stats_obj.std, 'tolist') else stats_obj.std,
                                            "max": stats_obj.max.tolist() if hasattr(stats_obj.max, 'tolist') else stats_obj.max,
                                            "min": stats_obj.min.tolist() if hasattr(stats_obj.min, 'tolist') else stats_obj.min,
                                            "q01": stats_obj.q01.tolist() if hasattr(stats_obj.q01, 'tolist') else stats_obj.q01,
                                            "q99": stats_obj.q99.tolist() if hasattr(stats_obj.q99, 'tolist') else stats_obj.q99,
                                        }
                                        tag_stats[f"relative_action.{key}"] = stats_dict

            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")


    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0

        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)

            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return

        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue

            # Convert back to DatasetMetadata format
            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }

            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data

            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data

            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
