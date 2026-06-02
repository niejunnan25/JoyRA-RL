#!/usr/bin/env python3
"""
简易自检：成功/失败标记解析是否与 value_targets_wrapper 一致。

默认不 import starVLA（避免 gr00t / numpydantic 等重依赖），内置一份与
`starVLA/dataloader/value_targets_wrapper.py` 同步的逻辑副本；若你改了后者，
请同步更新本脚本中带「MIRROR」注释的函数。

用法:
  python scripts/test_read_episode_success_labels.py

可选：抽查磁盘上的 LeRobot 目录（需 pandas；读 parquet 建议安装 pyarrow）:
  python scripts/test_read_episode_success_labels.py \\
    --dataset_dir /mnt/workspace/datasets/rl_offline_ee3/blocks_ranking_size

可选：在已配置好依赖的环境中，校验「包内实现」与副本一致:
  PYTHONPATH=. python scripts/test_read_episode_success_labels.py --verify_starVLA_import
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ----- MIRROR: 与 starVLA/dataloader/gr00t_lerobot/datasets.py 中常量一致 -----
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"


# ----- MIRROR: 与 value_targets_wrapper.py 中同名函数保持行为一致 -----


def load_episode_success_from_jsonl(dataset_path: Path) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    path = dataset_path / LE_ROBOT_EPISODE_FILENAME
    if not path.exists():
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                traj_id = int(ep.get("episode_index", -1))
                if traj_id >= 0 and "success" in ep:
                    out[traj_id] = bool(ep["success"])
    except OSError as e:
        print(f"WARN: 读取 {path} 失败: {e}")
    return out


def resolve_success_bool_from_traj_df(
    traj_df: pd.DataFrame, success_col: str
) -> Optional[bool]:
    if success_col in traj_df.columns:
        return bool(traj_df.iloc[0][success_col])
    if "success" in traj_df.columns:
        return bool(traj_df.iloc[0]["success"])
    if "episode_success" in traj_df.columns:
        return bool(traj_df.iloc[0]["episode_success"])
    return None


def augment_traj_df_success_for_returns(
    traj_df: pd.DataFrame,
    traj_id: int,
    success_col: str,
    episode_success_from_jsonl: Dict[int, bool],
) -> pd.DataFrame:
    if resolve_success_bool_from_traj_df(traj_df, success_col) is not None:
        return traj_df
    tid = int(traj_id)
    if tid not in episode_success_from_jsonl:
        return traj_df
    out = traj_df.copy()
    out[success_col] = bool(episode_success_from_jsonl[tid])
    return out


def compute_rewards_and_returns_from_traj(
    traj_df: pd.DataFrame,
    success_col: str = "episode_success",
    gamma: float = 1.0,
    big_negative: float = 100.0,
):
    T = len(traj_df)
    rewards = np.full(T, -1.0, dtype=np.float32)
    resolved = resolve_success_bool_from_traj_df(traj_df, success_col)
    success = True if resolved is None else resolved
    if success:
        rewards[-1] = 0.0
    else:
        rewards[-1] = -float(big_negative)
    returns = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return rewards, returns


# ----- tests -----


def _run_synthetic_tests() -> int:
    fails = []

    df = pd.DataFrame({"x": [1, 2], "success": [True, True]})
    if resolve_success_bool_from_traj_df(df, "episode_success") is not True:
        fails.append("success=True 应被解析为成功")

    df = pd.DataFrame({"x": [1, 2], "success": [False, False]})
    if resolve_success_bool_from_traj_df(df, "episode_success") is not False:
        fails.append("success=False 应被解析为失败")

    df = pd.DataFrame({"episode_success": [False, False]})
    if resolve_success_bool_from_traj_df(df, "episode_success") is not False:
        fails.append("episode_success=False 应失败")

    df = pd.DataFrame({"obs": [0.0, 0.0]})
    if resolve_success_bool_from_traj_df(df, "episode_success") is not None:
        fails.append("无 success 相关列时应为 None")
    _, ret = compute_rewards_and_returns_from_traj(
        df, success_col="episode_success", big_negative=100.0
    )
    if abs(float(ret[-1])) > 1e-5:
        fails.append(f"无标签应算成功，末步 return 期望约 0，得到 {ret[-1]}")

    df = pd.DataFrame({"obs": [0.0, 0.0]})
    m = {0: False}
    aug = augment_traj_df_success_for_returns(df, 0, "episode_success", m)
    _, ret = compute_rewards_and_returns_from_traj(
        aug, success_col="episode_success", big_negative=10.0
    )
    if float(ret[-1]) > -5.0:
        fails.append(f"jsonl 失败应对应强负 return，得到 {ret[-1]}")

    if fails:
        for msg in fails:
            print(f"FAIL: {msg}")
        return 1
    print("OK: 合成用例全部通过（resolve / augment / compute_rewards）")
    return 0


def _run_dataset_smoke(dataset_dir: Path) -> int:
    if not dataset_dir.is_dir():
        print(f"FAIL: 目录不存在: {dataset_dir}")
        return 1

    meta = load_episode_success_from_jsonl(dataset_dir)
    print(f"  episodes.jsonl 中读到 {len(meta)} 条带 success 的记录")

    parquets = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        print("  WARN: 未找到 data/chunk-*/episode_*.parquet，跳过 parquet 抽查")
        return 0

    p = parquets[0]
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"FAIL: 无法读取 {p}（可尝试 pip install pyarrow）: {e}")
        return 1

    stem = p.stem
    if not stem.startswith("episode_"):
        print(f"WARN: 非预期 parquet 命名 {p.name}，跳过 episode_index 对齐检查")
        return 0
    ep_idx = int(stem.replace("episode_", ""))

    r = resolve_success_bool_from_traj_df(df, "episode_success")
    jsonl_s = meta.get(ep_idx)

    print(f"  抽查文件: {p.relative_to(dataset_dir)}")
    print(f"    episode_index={ep_idx}")
    print(f"    resolve(..., episode_success) = {r}")
    print(f"    jsonl['success']  = {jsonl_s!r}")

    if jsonl_s is not None and r is None:
        aug = augment_traj_df_success_for_returns(
            df, ep_idx, "episode_success", meta
        )
        r2 = resolve_success_bool_from_traj_df(aug, "episode_success")
        if r2 != jsonl_s:
            print(
                f"FAIL: parquet 无显式列时应可由 jsonl 注入，期望 {jsonl_s}，注入后 resolve={r2}"
            )
            return 1
        print(f"    augment 后 resolve = {r2}（与 jsonl 一致）")

    if jsonl_s is not None and r is not None and r != jsonl_s:
        print(
            f"WARN: parquet 解析结果 {r} 与 jsonl {jsonl_s} 不一致，训练时以 parquet 优先"
        )

    print("OK: 真实数据集目录抽查完成")
    return 0


def _run_starvla_import_mirror_check() -> int:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from starVLA.dataloader import value_targets_wrapper as vw

    cases = [
        (pd.DataFrame({"a": [1], "success": [False]}), "episode_success", False),
        (pd.DataFrame({"episode_success": [True]}), "episode_success", True),
        (pd.DataFrame({"x": [1]}), "episode_success", None),
    ]
    for df, col, exp in cases:
        a = resolve_success_bool_from_traj_df(df, col)
        b = vw.resolve_success_bool_from_traj_df(df, col)
        if a != b:
            print(f"FAIL: 与包内 resolve 不一致: 副本={a!r} 包内={b!r} df={df.columns.tolist()}")
            return 1
        if a != exp:
            print(f"FAIL: 期望值 {exp!r} 得到 {a!r}")
            return 1

    df = pd.DataFrame({"x": [1]})
    m = {0: False}
    a = augment_traj_df_success_for_returns(df, 0, "episode_success", m)
    b = vw.augment_traj_df_success_for_returns(df, 0, "episode_success", m)
    if not a.equals(b):
        print("FAIL: augment 结果与包内不一致")
        return 1

    print("OK: --verify_starVLA_import 与包内实现一致")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="可选：LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--verify_starVLA_import",
        action="store_true",
        help="可选：import starVLA 并对比本脚本副本与 value_targets_wrapper",
    )
    args = parser.parse_args()

    print("== 合成测试（内置副本，无需 starVLA）==")
    code = _run_synthetic_tests()
    if code != 0:
        return code

    if args.dataset_dir is not None:
        print()
        print("== 可选：真实数据集目录 ==")
        code = _run_dataset_smoke(args.dataset_dir.resolve())
        if code != 0:
            return code

    if args.verify_starVLA_import:
        print()
        print("== 可选：与 starVLA 包内实现对齐检查 ==")
        try:
            return _run_starvla_import_mirror_check()
        except ImportError as e:
            print(f"SKIP: 无法 import starVLA（依赖未就绪）: {e}")
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
