#!/usr/bin/env python3
"""
扫描 LeRobot v2 数据集里 parquet 的 subtask_index 列分布（含 -1 占位帧）。

示例：
  # 单个子数据集目录（含 data/chunk-*/*.parquet）
  python examples/Suqian_agibot/train_files/inspect_subtask_index.py \\
      /mnt/workspace1/datasets/suqian_agibot_kingkong/desk_organization_combine_pnp/G1_task_4196_subtask_new_eepose

  # desk_organization_combine_pnp 下所有子目录分别统计 + 汇总
  python examples/Suqian_agibot/train_files/inspect_subtask_index.py \\
      /mnt/workspace1/datasets/suqian_agibot_kingkong/desk_organization_combine_pnp --scan-subdirs
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def _is_lerobot_dataset_root(p: Path) -> bool:
    return (p / "meta" / "modality.json").is_file() and (p / "data").is_dir()


def _gather_parquets(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("data/**/*.parquet"))


def _accumulate_subtask_counts(parquet_paths: list[Path], column: str) -> tuple[Counter, int, int]:
    """Returns (value_counts as Counter, files_read, files_missing_column)."""
    total: Counter = Counter()
    missing_col = 0
    for i, fp in enumerate(parquet_paths):
        try:
            df = pd.read_parquet(fp, columns=[column])
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}", file=sys.stderr)
            continue
        if column not in df.columns:
            missing_col += 1
            continue
        vc = df[column].value_counts(dropna=False)
        for k, v in vc.items():
            # 统一成 int，避免 float / numpy 类型混在 key 里
            try:
                ik = int(k) if pd.notna(k) else k
            except (TypeError, ValueError):
                ik = k
            total[ik] += int(v)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(parquet_paths)} parquet files", flush=True)
    return total, len(parquet_paths), missing_col


def _print_report(name: str, counts: Counter, n_files: int, missing_col: int, column: str) -> None:
    total_frames = sum(counts.values())
    neg1 = counts.get(-1, 0)
    print(f"\n=== {name} ===")
    print(f"parquet 文件数: {n_files} | 缺列文件数: {missing_col}")
    print(f"总行数(帧): {total_frames}")
    if total_frames == 0:
        print("(无数据)")
        return
    print(f"{column} == -1: {neg1} ({100.0 * neg1 / total_frames:.2f}%)")
    # 非负索引分布（摘要）
    nonneg = {k: v for k, v in counts.items() if isinstance(k, int) and k >= 0}
    if nonneg:
        mx = max(nonneg.keys())
        mn = min(nonneg.keys())
        print(f"{column} 范围(非负): [{mn}, {mx}]，不同取值个数: {len(nonneg)}")
    # 最常见的若干个值（含 -1）
    print("Top 15 取值:")
    for k, v in counts.most_common(15):
        print(f"  {k!r}: {v} ({100.0 * v / total_frames:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 LeRobot parquet 中 subtask_index 分布")
    parser.add_argument(
        "path",
        type=Path,
        help="单个数据集根目录，或父目录（配合 --scan-subdirs）",
    )
    parser.add_argument(
        "--scan-subdirs",
        action="store_true",
        help="若 path 为父目录，对其下每个子目录单独统计并最后汇总",
    )
    parser.add_argument(
        "--column",
        default="subtask_index",
        help="列名，默认 subtask_index",
    )
    args = parser.parse_args()
    root: Path = args.path.expanduser().resolve()
    col = args.column

    if not root.exists():
        print(f"路径不存在: {root}", file=sys.stderr)
        sys.exit(1)

    if args.scan_subdirs:
        subs = sorted([p for p in root.iterdir() if p.is_dir() and _is_lerobot_dataset_root(p)])
        if not subs:
            print(f"未在 {root} 下找到含 meta/modality.json + data/ 的子目录", file=sys.stderr)
            sys.exit(2)
        global_counts: Counter = Counter()
        total_files = 0
        total_missing_col = 0
        for sub in subs:
            pqs = _gather_parquets(sub)
            c, nf, mc = _accumulate_subtask_counts(pqs, col)
            _print_report(sub.name, c, nf, mc, col)
            global_counts.update(c)
            total_files += nf
            total_missing_col += mc
        _print_report(f"{root.name} [ALL SUBDIRS SUM]", global_counts, total_files, total_missing_col, col)
        return

    if not _is_lerobot_dataset_root(root):
        print(
            f"不是 LeRobot 数据集根目录（需要 meta/modality.json 与 data/）: {root}\n"
            f"若要对父目录下多个子集扫描，请使用 --scan-subdirs",
            file=sys.stderr,
        )
        sys.exit(2)

    pqs = _gather_parquets(root)
    if not pqs:
        print(f"未找到 parquet: {root}/data/**/*.parquet", file=sys.stderr)
        sys.exit(2)
    counts, nf, mc = _accumulate_subtask_counts(pqs, col)
    _print_report(root.name, counts, nf, mc, col)


if __name__ == "__main__":
    main()
