# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared state/action sub-key synonyms for LeRobot datasets and transforms.

Disk ``meta/modality.json`` may name splits ``left_gripper`` / ``left_pose`` while
training configs often use ``left_hand`` / ``left_arm``. Single-dataset
``DatasetMetadata`` normalizes modality + statistics keys to the canonical names below
(see ``datasets._normalize_state_action_modality_cfg``). Transforms still receive
``apply_to`` keys from ``data_config`` (either raw or canonical); resolve them via
``resolve_metadata_subkey`` when indexing merged/normalized metadata.
"""

from __future__ import annotations

from typing import Any, Mapping

STATE_ACTION_KEY_SYNONYMS: dict[str, str] = {
    "left_gripper": "left_hand",
    "right_gripper": "right_hand",
    "left_pose": "left_arm",
    "right_pose": "right_arm",
}

STATE_ACTION_CANONICAL_TO_RAW_SUBKEY: dict[str, str] = {
    canonical: raw for raw, canonical in STATE_ACTION_KEY_SYNONYMS.items()
}


def resolve_metadata_subkey(subkey: str, modality_cfg: Mapping[str, Any]) -> str:
    """Map a config/dataloader sub-key to the key stored in ``DatasetMetadata.modalities.{state|action}``.

    Prefer canonical names after synonym normalization; fall back to the raw sub-key if still present.
    """
    canonical = STATE_ACTION_KEY_SYNONYMS.get(subkey, subkey)
    if canonical in modality_cfg:
        return canonical
    if subkey in modality_cfg:
        return subkey
    raise KeyError(
        f"state/action sub-key {subkey!r} not in metadata (tried canonical {canonical!r}); "
        f"available: {list(modality_cfg.keys())}"
    )
