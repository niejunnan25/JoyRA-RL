#!/usr/bin/env python3
"""
按任务（dataset_key 的 basename）对 advantages JSON 打标。

规则（每个任务独立）：
  - 子集 P = { advantage > 0 的条目 }
  - P 按 advantage 降序排序，取前 ceil(fraction * |P|) 条，且至少 1 条（当 |P|>0）
  - 上述条目标记为 positive，其余为 negative

输入为 RLinf/JoyRA 风格的根对象：n_step, gamma, adv_scale, advantages[]。

内存说明：默认 json.load 会一次性载入整个文件；3GB 量级的 JSON 在 Python 中往往需要数倍文件大小的 RAM。
可选 --parser ijson：元数据仅解析到 advantages 数组开始，advantages 仍会被完整收集到内存（峰值与全量相近，但可避免先构建完整 dict 的额外开销）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT = (
    "/mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_robotwin_plus_offline_with_neg/"
    "advantages_all_train1.json"
)


def read_root_scalars_ijson(path: str) -> Dict[str, Any]:
    """只解析到 advantages 数组开始，读取 n_step / gamma / adv_scale。"""
    import ijson

    meta: Dict[str, Any] = {}
    with open(path, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if event == "number" and prefix in ("n_step", "gamma", "adv_scale"):
                meta[prefix] = value
            if prefix == "advantages" and event == "start_array":
                break
    for key in ("n_step", "gamma", "adv_scale"):
        if key not in meta:
            raise ValueError(f"未在文件开头解析到根字段 {key!r}，请确认 JSON 结构或使用 --parser json")
    return meta


def load_advantages_ijson(path: str) -> List[Dict[str, Any]]:
    import ijson

    with open(path, "rb") as f:
        return list(ijson.items(f, "advantages.item"))


def coerce_ijson_numbers(obj: Any) -> Any:
    """ijson 常把数字解析为 Decimal，需转为 float/int 才能 json.dump。"""
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral() else float(obj)
    if isinstance(obj, dict):
        return {k: coerce_ijson_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_ijson_numbers(v) for v in obj]
    return obj


def load_json_full(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("根对象必须是 JSON object")
    advantages = data.get("advantages")
    if not isinstance(advantages, list):
        raise ValueError("缺少 advantages 数组")
    meta = {k: data[k] for k in ("n_step", "gamma", "adv_scale") if k in data}
    for key in ("n_step", "gamma", "adv_scale"):
        if key not in meta:
            raise ValueError(f"缺少根字段 {key!r}")
    return meta, advantages


def load_input(path: str, parser: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    parser = parser.lower()
    if parser == "json":
        return load_json_full(path)
    if parser == "ijson":
        try:
            import ijson  # noqa: F401
        except ImportError as e:
            raise SystemExit("未安装 ijson，请 pip install ijson 或改用 --parser json") from e
        meta = coerce_ijson_numbers(read_root_scalars_ijson(path))
        advantages = coerce_ijson_numbers(load_advantages_ijson(path))
        return meta, advantages
    raise ValueError(f"未知 parser: {parser}")


def task_name(item: Dict[str, Any]) -> str:
    dk = item.get("dataset_key")
    if dk is None:
        raise KeyError("条目缺少 dataset_key")
    return os.path.basename(str(dk))


def assign_labels(
    advantages: List[Dict[str, Any]],
    *,
    adv_key: str,
    label_key: str,
    fraction: float,
) -> None:
    """原地写入 label_key。"""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction 应在 (0, 1] 内")

    by_task: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, item in enumerate(advantages):
        if adv_key not in item:
            raise KeyError(f"条目缺少字段 {adv_key!r} (index={i})")
        adv = float(item[adv_key])
        by_task[task_name(item)].append((i, adv))

    positive_indices: set[int] = set()
    for task, pairs in by_task.items():
        pos_pairs = [(idx, a) for idx, a in pairs if a > 0.0]
        if not pos_pairs:
            continue
        pos_pairs.sort(key=lambda x: -x[1])
        k = max(1, math.ceil(fraction * len(pos_pairs)))
        for idx, _ in pos_pairs[:k]:
            positive_indices.add(idx)

    for i, item in enumerate(advantages):
        item[label_key] = "positive" if i in positive_indices else "negative"


def validate_counts(
    advantages: List[Dict[str, Any]],
    *,
    adv_key: str,
    label_key: str,
    fraction: float,
) -> None:
    by_task: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for item in advantages:
        by_task[task_name(item)].append((float(item[adv_key]), str(item[label_key])))

    for task, rows in sorted(by_task.items()):
        total = len(rows)
        pos_adv = sum(1 for a, _ in rows if a > 0.0)
        pos_l = sum(1 for _, lb in rows if lb == "positive")
        neg_l = sum(1 for _, lb in rows if lb == "negative")
        assert pos_l + neg_l == total, (task, pos_l, neg_l, total)
        if pos_adv == 0:
            assert pos_l == 0, task
        else:
            expect = max(1, math.ceil(fraction * pos_adv))
            assert pos_l == expect, (task, pos_l, expect, pos_adv)


def collect_stats(
    advantages: List[Dict[str, Any]],
    *,
    adv_key: str,
    label_key: str,
) -> List[Dict[str, Any]]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for item in advantages:
        t = task_name(item)
        a = float(item[adv_key])
        lb = item[label_key]
        if t not in by_task:
            by_task[t] = {
                "task": t,
                "total": 0,
                "advantage_gt_0": 0,
                "positive": 0,
                "negative": 0,
                "sum_advantage": 0.0,
                "min_advantage": a,
                "max_advantage": a,
            }
        s = by_task[t]
        s["total"] += 1
        s["sum_advantage"] += a
        s["min_advantage"] = min(s["min_advantage"], a)
        s["max_advantage"] = max(s["max_advantage"], a)
        if a > 0.0:
            s["advantage_gt_0"] += 1
        if lb == "positive":
            s["positive"] += 1
        else:
            s["negative"] += 1

    rows: List[Dict[str, Any]] = []
    for t, s in by_task.items():
        s["mean_advantage"] = s["sum_advantage"] / s["total"] if s["total"] else 0.0
        rows.append(s)
    return rows


def print_stats_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    if sort_by == "name":
        rows = sorted(rows, key=lambda r: r["task"])
    elif sort_by == "mean_adv":
        rows = sorted(rows, key=lambda r: -r["mean_advantage"])
    else:
        raise ValueError(sort_by)

    header = (
        f"{'task':<40} {'total':>8} {'adv>0':>8} {'pos':>8} {'neg':>8} "
        f"{'mean_adv':>12} {'min_adv':>12} {'max_adv':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['task']:<40} {r['total']:>8} {r['advantage_gt_0']:>8} "
            f"{r['positive']:>8} {r['negative']:>8} "
            f"{r['mean_advantage']:>12.6f} {r['min_advantage']:>12.6f} {r['max_advantage']:>12.6f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按任务对 advantage 分位打标并写回 JSON")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT, help="输入 advantages JSON 路径")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="输出路径；默认与输入同目录下的 advantages_all_train_labeled.json",
    )
    p.add_argument("--fraction", type=float, default=0.4, help="advantage>0 子集中取前多少比例标为 positive")
    p.add_argument("--adv-key", type=str, default="advantage", help="用于排序与统计的数值字段名")
    p.add_argument("--label-key", type=str, default="adv_quality_label", help="写入的标签字段名")
    p.add_argument(
        "--parser",
        type=str,
        choices=("json", "ijson"),
        default="json",
        help="json：json.load 全量解析；ijson：两遍读取（需 pip install ijson）",
    )
    p.add_argument(
        "--sort-by",
        type=str,
        choices=("name", "mean_adv"),
        default="name",
        help="终端统计表的排序方式",
    )
    p.add_argument("--indent", type=int, default=None, help="输出 JSON indent；默认紧凑（None）以减小体积")
    p.add_argument(
        "--stats-json",
        type=str,
        default="",
        help="若指定路径，将按任务的统计写入该 JSON 文件",
    )
    p.add_argument("--no-validate", action="store_true", help="跳过打标后的计数校验")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(inp)), "advantages_all_train_labeled.json")

    meta, advantages = load_input(inp, args.parser)
    assign_labels(
        advantages,
        adv_key=args.adv_key,
        label_key=args.label_key,
        fraction=args.fraction,
    )
    if not args.no_validate:
        validate_counts(
            advantages,
            adv_key=args.adv_key,
            label_key=args.label_key,
            fraction=args.fraction,
        )

    stats_rows = collect_stats(advantages, adv_key=args.adv_key, label_key=args.label_key)
    print_stats_table(stats_rows, args.sort_by)

    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as sf:
            json.dump(stats_rows, sf, indent=2)
            sf.write("\n")

    payload = {
        "n_step": meta["n_step"],
        "gamma": meta["gamma"],
        "adv_scale": meta["adv_scale"],
        "advantages": advantages,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as wf:
        json.dump(payload, wf, indent=args.indent)
        wf.write("\n")

    print(f"\n已写入: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
