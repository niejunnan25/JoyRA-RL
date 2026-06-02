#!/usr/bin/env python3
"""
读取 advantages_all_train_labeled.json 并统计预测 value 的分布。
兼容两种结构：
1) {"advantages": [ {...}, ... ]}
2) 直接是 list[dict]

字段优先级：value_t > pred_value
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot predicted value distribution")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--output_stats_json", type=str, required=True)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_values(data: Any) -> List[float]:
    if isinstance(data, dict) and isinstance(data.get("advantages"), list):
        items = data["advantages"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON 顶层不是 list 或 {advantages: list} 结构")

    values: List[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "value_t" in item:
            values.append(float(item["value_t"]))
        elif "pred_value" in item:
            values.append(float(item["pred_value"]))
    return values


def _stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"找不到文件: {json_path}")

    data = _load_json(json_path)
    values = _extract_values(data)
    if not values:
        raise ValueError("没有在 JSON 中找到 value_t / pred_value 字段")

    values_np = np.asarray(values, dtype=np.float32)
    stats = _stats(values_np)

    fig = plt.figure(figsize=(10, 5))
    plt.hist(values_np, bins=args.bins, color="#4C78A8", alpha=0.85)
    plt.axvline(stats["mean"], color="#F58518", linewidth=1.5, label=f"mean={stats['mean']:.4f}")
    plt.axvline(stats["p50"], color="#54A24B", linewidth=1.5, label=f"median={stats['p50']:.4f}")
    plt.xlabel("predicted value")
    plt.ylabel("count")
    plt.title("Predicted Value Distribution")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=args.dpi)
    plt.close(fig)

    output_stats = Path(args.output_stats_json)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    with output_stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[Done] Saved plot: {output_png}")
    print(f"[Done] Saved stats: {output_stats}")


if __name__ == "__main__":
    main()
