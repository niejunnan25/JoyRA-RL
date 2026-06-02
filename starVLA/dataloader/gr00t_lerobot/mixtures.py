"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

import os
from typing import Dict, List, Tuple


def _build_robotwin_rl_offline_entries() -> List[Tuple[str, float, str]]:
    """
    Dynamically build RL-offline entries for Robotwin from the actual directory
    structure under /mnt/workspace/datasets/rl_offline, to avoid hard-coding
    task names that might be inconsistent with the filesystem.
    """
    base_dir = "/mnt/workspace/datasets/rl_offline_ee"
    entries: List[Tuple[str, float, str]] = []
    if not os.path.isdir(base_dir):
        return entries

    for name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            # dataset_name is relative to data_root_dir (/mnt/workspace/datasets)
            entries.append((f"rl_offline_ee/{name}", 1.0, "my_robotwin_reversed"))
    return entries

def _replace_robot_type(
    mixture_items: list[tuple],
    robot_type: str,
) -> list[tuple]:
    """Rewrite robot_type while preserving optional d_downsample."""
    converted: list[tuple] = []
    for item in mixture_items:
        if len(item) == 3:
            dataset_name, sampling_weight, _ = item
            d_downsample = 1
        elif len(item) == 4:
            dataset_name, sampling_weight, _, d_downsample = item
        else:
            raise ValueError(f"Invalid mixture item length {len(item)}: {item}")

        # converted.append((dataset_name, sampling_weight, robot_type, d_downsample))
        converted.append(("robotwin_dataset/dataset_lerobot/" + dataset_name, sampling_weight, robot_type))
    return converted


def _build_robocasa_24tasks_entries() -> List[Tuple[str, float, str]]:
    """
    动态构建 RoboCasa 24 tasks 的条目：
    目录结构假设为：
      /mnt/workspace/datasets/robocasa_24tasks_datasets/pick_and_place_lerobot_task24/<subdir>
    data_root_dir 一般设为 /mnt/workspace/datasets，因此 dataset_name 需要是
      robocasa_24tasks_datasets/pick_and_place_lerobot_task24/<subdir>
    """
    base_dir = "/mnt/workspace/datasets/robocasa_24tasks_datasets/pick_and_place_lerobot_task24"
    entries: List[Tuple[str, float, str]] = []
    if not os.path.isdir(base_dir):
        return entries

    for name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            entries.append((f"robocasa_24tasks_datasets/pick_and_place_lerobot_task24/{name}", 1.0, "fourier_gr1_arms_waist"))
    return entries


def _build_agibot_g1_desk_organization_combine_pnp_entries() -> List[Tuple[str, float, str]]:
    """
    Agibot G1（宿迁 KingKong）desk_organization_combine_pnp：父目录下多份子任务 LeRobot 数据。
    使用 `agibot_g1_kingkong_eepose` 配置，与各子集 meta/modality.json、meta/subtasks.jsonl 一致（子任务语言）。
    data_root_dir 需包含 suqian_agibot_kingkong（常见 /mnt/workspace1/datasets），
    相对路径：<data_root_dir>/suqian_agibot_kingkong/desk_organization_combine_pnp/<子目录>
    """
    base_rel = "suqian_agibot_kingkong/desk_organization_combine_pnp"
    candidate_roots = ("/mnt/workspace1/datasets", "/mnt/workspace/datasets")
    base_dir = ""
    for root in candidate_roots:
        cand = os.path.join(root, base_rel)
        if os.path.isdir(cand):
            base_dir = cand
            break
    entries: List[Tuple[str, float, str]] = []
    if not base_dir:
        return entries

    for name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue
        if not os.path.isdir(os.path.join(full_path, "meta")):
            continue
        entries.append((f"{base_rel}/{name}", 1.0, "agibot_g1_kingkong_eepose"))
    return entries


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAME_TO_PREFIX_small = {
    "robotwin_clean_random": [
        ("aloha-agilex_clean_50_single_task_eepose/adjust_bottle-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/beat_block_hammer-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/blocks_ranking_rgb-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/blocks_ranking_size-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/click_alarmclock-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/click_bell-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/dump_bin_bigbin-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/grab_roller-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/handover_block-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/handover_mic-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/hanging_mug-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/lift_pot-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/move_can_pot-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/move_pillbottle_pad-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/move_playingcard_away-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/move_stapler_pad-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/open_laptop-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/open_microwave-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/pick_diverse_bottles-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/pick_dual_bottles-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_a2b_left-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_a2b_right-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_bread_basket-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_bread_skillet-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_burger_fries-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_can_basket-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_cans_plasticbox-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_container_plate-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_dual_shoes-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_empty_cup-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_fan-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_mouse_pad-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_object_basket-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_object_scale-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_object_stand-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_phone_stand-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/place_shoe-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/press_stapler-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/put_bottles_dustbin-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/put_object_cabinet-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/rotate_qrcode-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/scan_object-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/shake_bottle-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/shake_bottle_horizontally-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/stack_blocks_three-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/stack_blocks_two-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/stack_bowls_three-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/stack_bowls_two-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/stamp_seal-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task_eepose/turn_switch-aloha-agilex_clean_50-50_ee", 1.0, "robotwin"),

        ("aloha-agilex_randomized_500_single_task_eepose/adjust_bottle-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/beat_block_hammer-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/blocks_ranking_rgb-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/blocks_ranking_size-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/click_alarmclock-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/click_bell-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/dump_bin_bigbin-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/grab_roller-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/handover_block-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/handover_mic-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/hanging_mug-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/lift_pot-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/move_can_pot-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/move_pillbottle_pad-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/move_playingcard_away-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/move_stapler_pad-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/open_laptop-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/open_microwave-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/pick_diverse_bottles-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/pick_dual_bottles-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_a2b_left-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_a2b_right-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_bread_basket-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_bread_skillet-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_burger_fries-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_can_basket-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_cans_plasticbox-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_container_plate-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_dual_shoes-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_empty_cup-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_fan-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_mouse_pad-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_object_basket-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_object_scale-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_object_stand-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_phone_stand-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/place_shoe-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/press_stapler-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/put_bottles_dustbin-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/put_object_cabinet-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/rotate_qrcode-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/scan_object-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/shake_bottle_horizontally-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/shake_bottle-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/stack_blocks_three-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/stack_blocks_two-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/stack_bowls_three-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/stack_bowls_two-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/stamp_seal-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task_eepose/turn_switch-aloha-agilex_randomized_500-500_ee", 1.0, "robotwin"),
    ],
}

DATASET_NAMED_MIXTURES = {
    "suqian_test_mouse_right": [("suqian_agibot_lerobot_data/task_3463_mouse_right", 1.0, "agibot_genie1"),],
    # G1 LeRobot：鼠标右侧 task_3463 + task_3540。路径相对 data_root_dir（例如 /mnt/workspace1/datasets）。
    # 绝对路径：.../G1_task_3463/task_3463_mouse_right 与 .../G1_task_3540/task_3540_mouse_right
    "g1_task_3463_3540_mouse_right": [
        ("G1_task_3463/task_3463_mouse_right", 1.0, "agibot_genie1"),
        ("G1_task_3540/task_3540_mouse_right", 1.0, "agibot_genie1"),
    ],
    "suqian_test_office_1": [
        ("suqian_agibot_lerobot_data/headphone_mouse_pen/task_3463_mouse", 1.0, "agibot_genie1"),
        ("suqian_agibot_lerobot_data/headphone_mouse_pen/task_3540_mouse", 1.0, "agibot_genie1"),
    ],
    "suqian_test_chaji": [("suqian_agibot_lerobot_data/task_3593_filtered", 1.0, "agibot_genie1"),],
    "suqian_test_chaji_2": [("suqian_agibot_lerobot_data/task_3593_full_episode_modified", 1.0, "agibot_genie1"),],
    "suqian_test_table_clean": [("suqian_agibot_lerobot_data/task_3571_full_episode_modified", 1.0, "agibot_genie1"),],
    # data_root_dir=/mnt/workspace/datasets 时对应 .../suqian_agibot_lerobot_data/task_3799_full_episode_modified
    "suqian_task_3799_full_episode_modified": [
        ("suqian_agibot_lerobot_data/task_3799_full_episode_modified", 1.0, "agibot_genie1"),
    ],
    "suqian_test_pack_1": [("suqian_agibot_lerobot_data/task_3667_3708_999_cold_remedy", 1.0, "agibot_genie1"),],
    "suqian_test_headphone_r": [("suqian_agibot_lerobot_data/task_3463_headphone_right", 1.0, "agibot_genie1"),],
    "suqian_test_headphone_l": [("suqian_agibot_lerobot_data/task_3463_headphone_left", 1.0, "agibot_genie1"),],
    "suqian_train": [("suqian_agibot_lerobot_data/task_170_modified", 1.0, "agibot_genie1_joint"),],
    "robocasa_300_ee": [
        ("robocasa_24tasks_datasets/pick_and_place_lerobot_task24_sampled_300_epose_merged", 1.0, "fourier_gr1_arms_waist"),
    ],
    "sq_agi_beta": [
        ("suqian_agibot_lerobot_data/task_170_modified", 1.0, "agibot_genie1_joint"),
        ("AgiBotWorld-Beta-LeRobot", 1.0, "agibot_genie1_joint"),
    ],
    "agi_beta": [("AgiBotWorld-Beta-LeRobot", 1.0, "agibot_genie1"),],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],
    "ego_dex": [("", 1.0, "ego_dex"),],
    "galaxea_suqian_agi": [
        ("datasets/suqian_agibot_lerobot_data/task_170_modified", 1.0, "agibot_genie1"),
        ("users/chenzengjue/starVLA2/merge_test", 1.0, "galaxea_r1lite"),
    ],
    "ego_dex_suqian_agi": [
        ("egodex30w_datasets/egodex_merge30w_Agibot", 1.0, "ego_dex"),
        ("suqian_agibot_lerobot_data/task_170_modified", 1.0, "agibot_genie1"),
    ],
    "place_a2b_right": [("stack_blocks_two", 1.0, "my_robotwin_reversed"),],

    "interndata_a1_aloha": [
        ("collect_the_shoes", 1.0, "interndata_a1_aloha"),
        ("hang_the_cup_on_rack_left_arm", 1.0, "interndata_a1_aloha"),
        ("hang_the_cup_on_rack_right_arm", 1.0, "interndata_a1_aloha"),
        ("organize_the_brushes", 1.0, "interndata_a1_aloha"),
        ("pick_ham_sandwich_on_conveyor", 1.0, "interndata_a1_aloha"),
        ("pick_the_priced_item", 1.0, "interndata_a1_aloha"),
        ("pour_Baijiu_right_arm", 1.0, "interndata_a1_aloha"),
        ("pour_redwine_left_arm", 1.0, "interndata_a1_aloha"),
        ("pour_redwine_right_arm", 1.0, "interndata_a1_aloha"),
        ("pour_water_left_arm", 1.0, "interndata_a1_aloha"),
        ("sort_the_table_waste", 1.0, "interndata_a1_aloha"),
        ("stack_the_boxes_part1", 1.0, "interndata_a1_aloha"),
        ("stack_the_boxes_part2", 1.0, "interndata_a1_aloha"),
        ("stack_the_boxes_part3", 1.0, "interndata_a1_aloha"),
        ("stack_the_boxes_part6", 1.0, "interndata_a1_aloha"),
        ("watering_plants_right_arm", 1.0, "interndata_a1_aloha"),
    ],


    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
     # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "bridge": [("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],
    "demo_sim_pick_place": [("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),],
    "custom_dataset": [("custom_dataset_name", 1.0, "custom_robot_config"),],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],
    "fourier_gr1_unified_1000_debug": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],
    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0,
         "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0,
         "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],
    "BEHAVIOR_challenge": [("BEHAVIOR_challenge", 1.0, "R1Pro"),],
    "SO101_pick": [("pick_dataset_name", 1.0, "SO101"),],
    "arx_x5": [("arx_x5", 1.0, "arx_x5"),],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],

    # Mix original Robotwin Aloha-Agilex 550 dataset and RL-offline data for Robotwin.
    # Assumes:
    #   - Original mixed dataset is under: /mnt/workspace/datasets/robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix
    #     (this is already used by 'robotwin_aloha_agilex_550_mix' mix)
    #   - RL-offline per-task datasets are under: /mnt/workspace/datasets/rl_offline/<task_name>
    # And you set data_root_dir to: /mnt/workspace/datasets
    "robotwin_orig_plus_offline": [
        # ---- 原始 Robotwin Aloha-Agilex 550 混合数据 ----
        # ("robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix", 1.0, "robotwin"),
        *_replace_robot_type(DATASET_NAME_TO_PREFIX_small["robotwin_clean_random"],"my_robotwin_reversed",),
        # ---- RL-offline Robotwin datasets (discovered dynamically from filesystem) ----
        *_build_robotwin_rl_offline_entries(),
    ],

    "robotwin_orig_plus_offline_v2": [
        # ---- 原始 Robotwin Aloha-Agilex 550 混合数据 ----
        # ("robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix", 1.0, "robotwin"),
        *_replace_robot_type(DATASET_NAME_TO_PREFIX_small["robotwin_clean_random"],"my_robotwin_reversed",),
        # ---- 指定仅包含以下 RL-offline 任务（不再自动发现）----
        ("rl_offline_ee3/blocks_ranking_size", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/click_alarmclock", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/click_bell", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/hanging_mug", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/pick_diverse_bottles", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_bread_basket", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_bread_skillet", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_can_basket", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_container_plate", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_fan", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_object_stand", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/place_phone_stand", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/press_stapler", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/put_bottles_dustbin", 1.0, "my_robotwin_reversed"),
        # ("rl_offline_ee3/put_object_cabinet", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/rotate_qrcode", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/stack_blocks_three", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/stack_bowls_three", 1.0, "my_robotwin_reversed"),
        ("rl_offline_ee3/turn_switch", 1.0, "my_robotwin_reversed"),

        # ("rl_offline_ee3/dump_bin_bigbin", 1.0, "my_robotwin_reversed"),
    ],

    # Agibot G1（非 Robotwin）：desk_organization_combine_pnp 下多子目录；robot_type=agibot_g1_kingkong_eepose
    # 启动示例：bash run_value_agibot_g1_desk_organization_T.sh
    "agibot_g1_desk_organization_combine_pnp": [
        *_build_agibot_g1_desk_organization_combine_pnp_entries(),
    ],
    
    "robotwin_aloha_agilex_550_mix": [
        ("robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix", 1.0, "robotwin"),
    ],
    "agibotworld_suqian_robotwin_mix": [
        ("AgiBotWorld-Beta-LeRobot", 1.0, "agibot_genie1_joint"),
        ("suqian_agibot_lerobot_data/task_170_modified", 1.0, "agibot_genie1_joint"),
        ("robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix", 1.0, "robotwin"),
    ],
    # 混合 AgiBotWorld Beta、宿迁 Suqian、EgoDex AlohaGripper、RoboCasa 24tasks 以及 Robotwin Aloha-Agilex 550 数据
    # 对应的绝对路径（在 data_root_dir=/mnt/workspace/datasets 前提下）为：
    #   - /mnt/workspace/datasets/AgiBotWorld-Beta-LeRobot
    #   - /mnt/workspace/datasets/suqian_agibot_lerobot_data/task_170_modified
    #   - /mnt/workspace/datasets/egodex30w_datasets/egodex_merge30w_AlohaGripper
    #   - /mnt/workspace/datasets/robocasa_24tasks_datasets/pick_and_place_lerobot_task24
    #   - /mnt/workspace/datasets/robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix
    "agi_suqian_egodex_robocasa_robotwin_mix": [
        ("AgiBotWorld-Beta-LeRobot", 1.0, "agibot_genie1_joint"),
        ("suqian_agibot_lerobot_data/task_170_modified", 1.0, "agibot_genie1_joint"),
        ("egodex30w_datasets/egodex_merge30w_AlohaGripper", 1.0, "ego_dex"),
        # 动态加入 robocasa 24 tasks 下面所有子任务目录
        *_build_robocasa_24tasks_entries(),
        ("robotwin_dataset/dataset_lerobot/aloha-agilex_550_mix", 1.0, "robotwin"),
    ],
    "robotwin_aloha_agilex-clean": [
        ("aloha-agilex_clean_50_single_task/adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/blocks_ranking_rgb-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/blocks_ranking_size-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/click_alarmclock-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/click_bell-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/handover_block-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/handover_mic-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/hanging_mug-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_can_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_pillbottle_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_stapler_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/open_laptop-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/open_microwave-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/pick_diverse_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_a2b_left-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_a2b_right-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_bread_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_bread_skillet-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_burger_fries-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_can_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_cans_plasticbox-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_container_plate-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_dual_shoes-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_empty_cup-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_fan-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_mouse_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_object_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_object_scale-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_phone_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_shoe-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/press_stapler-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/put_bottles_dustbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/put_object_cabinet-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/rotate_qrcode-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/scan_object-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/shake_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/shake_bottle_horizontally-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_blocks_three-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_blocks_two-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_bowls_three-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_bowls_two-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stamp_seal-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/turn_switch-aloha-agilex_clean_50-50", 1.0, "robotwin"),
    ],

    "robotwin_aloha_agilex-all": [
        ("aloha-agilex_clean_50_single_task/adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/blocks_ranking_rgb-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/blocks_ranking_size-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/click_alarmclock-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/click_bell-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/handover_block-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/handover_mic-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/hanging_mug-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_can_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_pillbottle_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/move_stapler_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/open_laptop-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/open_microwave-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/pick_diverse_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_a2b_left-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_a2b_right-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_bread_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_bread_skillet-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_burger_fries-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_can_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_cans_plasticbox-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_container_plate-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_dual_shoes-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_empty_cup-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_fan-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_mouse_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_object_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_object_scale-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_phone_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/place_shoe-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/press_stapler-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/put_bottles_dustbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/put_object_cabinet-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/rotate_qrcode-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/scan_object-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/shake_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/shake_bottle_horizontally-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_blocks_three-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_blocks_two-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_bowls_three-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stack_bowls_two-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/stamp_seal-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("aloha-agilex_clean_50_single_task/turn_switch-aloha-agilex_clean_50-50", 1.0, "robotwin"),

        ("aloha-agilex_randomized_500_single_task/adjust_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/beat_block_hammer-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/blocks_ranking_rgb-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/blocks_ranking_size-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/click_alarmclock-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/click_bell-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/dump_bin_bigbin-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/grab_roller-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/handover_block-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/handover_mic-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/hanging_mug-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/lift_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/move_can_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/move_pillbottle_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/move_playingcard_away-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/move_stapler_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/open_laptop-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/open_microwave-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/pick_diverse_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/pick_dual_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_a2b_left-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_a2b_right-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_bread_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_bread_skillet-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_burger_fries-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_can_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_cans_plasticbox-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_container_plate-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_dual_shoes-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_empty_cup-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_fan-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_mouse_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_object_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_object_scale-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_object_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_phone_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/place_shoe-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/press_stapler-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/put_bottles_dustbin-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/put_object_cabinet-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/rotate_qrcode-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/scan_object-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/shake_bottle_horizontally-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/shake_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/stack_blocks_three-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/stack_blocks_two-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/stack_bowls_three-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/stack_bowls_two-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/stamp_seal-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("aloha-agilex_randomized_500_single_task/turn_switch-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
    ],

    "robotwin_task1": [("adjust_bottle", 1.0, "robotwin"),],
    "robotwin_task2": [
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
    ],
    "multi_robot": [("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
     # ("OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
                   ],
}
