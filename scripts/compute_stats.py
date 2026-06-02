#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute LeRobot meta/stats.json for datasets like AgiBotWorld-Beta-LeRobot.

Works with schema like:
- frame_index int64
- observation.state list<element: double>
- action list<element: double>
- camera_params.* list<element: double>
...

It streams parquet row-groups; no pandas concat; no OOM.
mean/std/min/max are exact (Welford); q01/q99 are approx via reservoir sampling.

Usage:
  python compute_stats.py \
    --dataset_root /mnt/workspace/datasets/AgiBotWorld-Beta-LeRobot \
    --output /mnt/workspace/datasets/AgiBotWorld-Beta-LeRobot/meta/stats.json \
    --sample_size 100000 \
    --seed 42 \
    --max_files 20000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

EPS = 1e-8


def is_numeric_scalar(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)


def is_list_of_numeric(t: pa.DataType) -> bool:
    return (pa.types.is_list(t) or pa.types.is_large_list(t)) and is_numeric_scalar(t.value_type)


class RunningStats:
    """Vector running stats (Welford) + min/max + reservoir sample for quantiles."""

    def __init__(self, dim: int, sample_size: int, rng: np.random.Generator):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.minv = np.full(dim, np.inf, dtype=np.float64)
        self.maxv = np.full(dim, -np.inf, dtype=np.float64)

        self.sample_size = int(sample_size)
        self.rng = rng
        self._sample_n = 0
        if self.sample_size > 0:
            self._sample = np.empty((self.sample_size, dim), dtype=np.float64)
        else:
            self._sample = None

    def update_batch(self, x: np.ndarray):
        """
        x: [B] or [B, D]
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        assert x.shape[1] == self.dim, (x.shape, self.dim)
        b = x.shape[0]
        if b == 0:
            return

        self.minv = np.minimum(self.minv, np.min(x, axis=0))
        self.maxv = np.maximum(self.maxv, np.max(x, axis=0))

        # Welford per-row (robust, fast enough)
        for i in range(b):
            self.n += 1
            xi = x[i]
            delta = xi - self.mean
            self.mean += delta / self.n
            delta2 = xi - self.mean
            self.M2 += delta * delta2

        # Reservoir sampling for quantiles
        if self.sample_size > 0:
            for i in range(b):
                if self._sample_n < self.sample_size:
                    self._sample[self._sample_n] = x[i]
                    self._sample_n += 1
                else:
                    j = self.rng.integers(0, self.n)
                    if j < self.sample_size:
                        self._sample[j] = x[i]

    def finalize(self) -> Dict[str, Any]:
        if self.n <= 1:
            std = np.ones(self.dim, dtype=np.float64)
        else:
            var = self.M2 / (self.n - 1)
            std = np.sqrt(np.maximum(var, EPS))

        if self.sample_size > 0 and self._sample_n > 0:
            samp = self._sample[:self._sample_n]
            q01 = np.quantile(samp, 0.01, axis=0)
            q99 = np.quantile(samp, 0.99, axis=0)
        else:
            q01 = self.minv.copy()
            q99 = self.maxv.copy()

        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.minv.tolist(),
            "max": self.maxv.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def list_array_to_2d_numpy(col: pa.Array) -> Optional[np.ndarray]:
    """
    Convert Arrow ListArray (list<double>) to numpy [B, D] if lengths are fixed.
    Returns None if ragged or null-heavy.
    """
    if not (pa.types.is_list(col.type) or pa.types.is_large_list(col.type)):
        return None

    # Convert each row to python list. This is not zero-copy, but safe and simple.
    py = col.to_pylist()
    if len(py) == 0:
        return None

    # Find first non-null row to infer dim
    dim = None
    for row in py:
        if row is not None:
            dim = len(row)
            break
    if dim is None:
        return None

    # Check fixed length
    for row in py:
        if row is None:
            continue
        if len(row) != dim:
            # ragged -> skip
            return None

    # Build numpy, fill null rows with zeros
    out = np.zeros((len(py), dim), dtype=np.float64)
    for i, row in enumerate(py):
        if row is None:
            continue
        out[i] = np.asarray(row, dtype=np.float64)
    return out


def compute_stats(
    parquet_files: list[Path],
    sample_size: int,
    seed: int,
    max_files: int,
    include_cols: Optional[list[str]] = None,
    exclude_cols: Optional[list[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    col_stats: Dict[str, RunningStats] = {}

    if max_files and max_files > 0:
        parquet_files = parquet_files[:max_files]

    for path in tqdm(parquet_files, desc="Streaming parquet for stats"):
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"[WARN] cannot open parquet {path}: {e}")
            continue

        schema = pf.schema_arrow
        schema_names = set(schema.names)

        # Determine columns to read
        cols = schema.names
        if include_cols is not None:
            cols = [c for c in cols if c in set(include_cols)]
        if exclude_cols is not None:
            cols = [c for c in cols if c not in set(exclude_cols)]

        # Read row groups streaming
        for rg in range(pf.num_row_groups):
            try:
                table = pf.read_row_group(rg, columns=cols)
            except Exception as e:
                print(f"[WARN] read row_group {rg} failed for {path}: {e}")
                continue

            # Process column by column to avoid large intermediate numpy
            for col_name in table.schema.names:
                field = table.schema.field(col_name)
                t = field.type
                col = table[col_name]

                # Skip obvious non-feature text stuff
                if "task_info" in col_name:
                    continue

                # scalar numeric
                if is_numeric_scalar(t):
                    arr = col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
                    dim = 1
                    if col_name not in col_stats:
                        col_stats[col_name] = RunningStats(dim=dim, sample_size=sample_size, rng=rng)
                    col_stats[col_name].update_batch(arr.reshape(-1, 1))
                    continue

                # list<double> (state/action/etc.)
                if is_list_of_numeric(t):
                    mat = list_array_to_2d_numpy(col)
                    if mat is None:
                        # ragged or weird -> skip
                        continue
                    dim = mat.shape[1]
                    if col_name not in col_stats:
                        col_stats[col_name] = RunningStats(dim=dim, sample_size=sample_size, rng=rng)
                    col_stats[col_name].update_batch(mat)
                    continue

                # otherwise skip
                continue

    # finalize
    out: Dict[str, Dict[str, Any]] = {}
    for k, rs in col_stats.items():
        out[k] = rs.finalize()
    return out


def main():
    DEFAULT_INCLUDE = [
        "observation.state",
        "action",
        "original_state",
        "original_action",
        "frame_index",
        "timestamp",
    ]


    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--sample_size", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=20000)
    ap.add_argument(
        "--include",
        type=str,
        default=",".join(DEFAULT_INCLUDE),
        help=(
            "Comma-separated column names to include. "
            "Default computes only core training columns: "
            + ", ".join(DEFAULT_INCLUDE)
        ),
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated column names to exclude.",
    )
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_path = Path(args.output) if args.output else (root / "meta" / "stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(root.glob("data/*/*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {root}/data/*/*.parquet")

    include_cols = [s.strip() for s in args.include.split(",") if s.strip()] or None
    exclude_cols = [s.strip() for s in args.exclude.split(",") if s.strip()] or None

    stats = compute_stats(
        parquet_files=parquet_files,
        sample_size=args.sample_size,
        seed=args.seed,
        max_files=args.max_files,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote stats: {out_path}")
    print(f"[OK] columns computed: {len(stats)}")
    # quick sanity print
    for k in ["observation.state", "action", "original_state", "original_action"]:
        if k in stats:
            print(f"[OK] {k} dim={len(stats[k]['mean'])} n_fields={len(stats[k])}")
        else:
            print(f"[WARN] missing {k} in output (maybe excluded or non-numeric?)")


if __name__ == "__main__":
    main()
