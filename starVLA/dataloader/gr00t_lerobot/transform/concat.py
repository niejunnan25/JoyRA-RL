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

from typing import Optional

import cv2
import numpy as np
import torch
from pydantic import Field

from ..schema import DatasetMetadata, StateActionMetadata
from ..state_action_key_synonyms import resolve_metadata_subkey
from .base import InvertibleModalityTransform


class ConcatTransform(InvertibleModalityTransform):
    """
    Concatenate the keys according to specified order.
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )

    video_concat_order: list[str] = Field(
        ...,
        description="Concatenation order for each video modality. "
        "Format: ['video.ego_view_pad_res224_freq20', ...]",
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. "
        "Format: ['state.position', 'state.velocity', ...].",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. "
        "Format: ['action.position', 'action.velocity', ...].",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys.",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys.",
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        if "video" in grouped_keys:
            # Check if keys in video_concat_order, state_concat_order, action_concat_order are
            # ineed contained in the data. If not, then the keys are misspecified
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None, f"{self.video_concat_order=}, {video_keys=}"
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            # Process each video view
            unsqueezed_videos: list[np.ndarray] = []
            target_hw = None
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)  # [..., H, W, C]

                # 在多数据集混合的情况下，即使我们在 VideoTransform 里做了 resize，
                # 仍然可能因为元数据合并/视角过滤等原因，导致不同视角最终的 H, W 不一致。
                # 这里在 concat 之前做一次「最后兜底」的 resize：把所有视角统一到
                # 第一个视角的分辨率，避免再出现 224 vs 480 这种错误。
                h, w = video_data.shape[-3:-1]
                if target_hw is None:
                    target_hw = (h, w)
                else:
                    th, tw = target_hw
                    if h != th or w != tw:
                        # 逐帧 resize 到目标分辨率
                        # 这里假设数据类型为 uint8 / float32，使用 OpenCV 进行双线性插值
                        orig_shape = video_data.shape
                        t, c = orig_shape[0], orig_shape[-1]
                        # [T, H, W, C] -> [T, H, W, C] (resize H,W)
                        resized_frames = []
                        for frame in video_data:
                            resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                            resized_frames.append(resized)
                        video_data = np.stack(resized_frames, axis=0)

                unsqueezed_video = np.expand_dims(
                    video_data, axis=-4
                )  # [..., H, W, C] -> [..., 1, H, W, C]
                unsqueezed_videos.append(unsqueezed_video)
            # Concatenate along the new axis
            unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

            # Video
            data["video"] = unsqueezed_video

        # "state"
        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            # 如果在 set_metadata 阶段由于元数据合并而没有留下任何可用的 state 键，
            # 则直接跳过 state 的 concat，让后续逻辑使用原始的 per-key state.*
            if not self.state_concat_order:
                # 不做任何修改，保持 state.left_arm 等原始键
                return data

            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            assert all(
                item in state_keys for item in self.state_concat_order
            ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
            # Check the state dims
            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(6)  # Allow for rotation_6d
                # if key in ["state.right_arm", "state.right_hand"]:
                target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                assert (
                    data[key].shape[-1] in target_shapes
                ), f"State dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
            # Concatenate the state keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["state"] = torch.cat(
                [data.pop(key) for key in self.state_concat_order], dim=-1
            )  # [T, D_state]

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            # 如果在 set_metadata 阶段由于元数据合并而没有留下任何可用的 action 键，
            # 则直接跳过 action 的 concat，让后续逻辑使用原始的 per-key action.*
            if not self.action_concat_order:
                # 不做任何修改，保持 action.left_arm_joint 等原始键
                return data

            assert self.action_concat_order is not None, f"{self.action_concat_order=}"
            # Check if all keys in concat_order are present（允许有额外的 action key，比如 gripper）
            concat_keys_set = set(self.action_concat_order)
            action_keys_set = set(action_keys)
            # 要求 concat_order 中的 key 至少都在实际的 action_keys 里
            assert concat_keys_set.issubset(
                action_keys_set
            ), f"{concat_keys_set=} is not subset of {action_keys_set=}"
            # 如果还有额外的 action key，没有出现在 concat_order 里，就忽略它们，并打印一个提示
            extra_keys = action_keys_set - concat_keys_set
            # if extra_keys:
            #     print(
            #         "[WARN] Extra action keys found that are not in action_concat_order; "
            #         f"they will be ignored by ConcatStateAction: {sorted(extra_keys)}"
            #     )
            # Record the action dims
            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(3)  # Allow for axis angle
                assert (
                    self.action_dims[key] == data[key].shape[-1]
                ), f"Action dim mismatch for {key=}, {self.action_dims[key]=}, {data[key].shape[-1]=}"
            # Concatenate the action keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["action"] = torch.cat(
                [data.pop(key) for key in self.action_concat_order], dim=-1
            )  # [T, D_action]

        return data

    def unapply(self, data: dict) -> dict:
        start_dim = 0
        assert "action" in data, f"{data.keys()=}"
        # For those dataset without actions (LAPA), we'll never run unapply
        assert self.action_concat_order is not None, f"{self.action_concat_order=}"
        action_tensor = data.pop("action")
        for key in self.action_concat_order:
            if key not in self.action_dims:
                raise ValueError(f"Action dim {key} not found in action_dims.")
            end_dim = start_dim + self.action_dims[key]
            data[key] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim
        if "state" in data:
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                end_dim = start_dim + self.state_dims[key]
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".", 1)
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        resolved = resolve_metadata_subkey(subkey, modality_config)
        assert isinstance(
            modality_config[resolved], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {resolved=}, got {type(modality_config[resolved])=}"
        return modality_config[resolved]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        # Pre-compute the dimensions of the state and action keys
        if self.action_concat_order is not None:
            filtered_action_order = []
            for key in self.action_concat_order:
                try:
                    self.action_dims[key] = self.get_state_action_dims(key)
                    filtered_action_order.append(key)
                except AssertionError as e:
                    # 在混合多数据集时，某些动作键（例如 action.left_arm_joint）可能
                    # 在合并后的 metadata 中被丢弃，这里安全地跳过这些键。
                    print(
                        "[WARN] Action key missing in merged metadata; "
                        f"skipping from concat order: {key} ({e})"
                    )
                    continue
            self.action_concat_order = filtered_action_order
        if self.state_concat_order is not None:
            filtered_state_order = []
            for key in self.state_concat_order:
                try:
                    self.state_dims[key] = self.get_state_action_dims(key)
                    filtered_state_order.append(key)
                except AssertionError as e:
                    # 同理，某些 state 键在合并后的 metadata 中可能不存在，直接跳过
                    print(
                        "[WARN] State key missing in merged metadata; "
                        f"skipping from concat order: {key} ({e})"
                    )
                    continue
            self.state_concat_order = filtered_state_order
