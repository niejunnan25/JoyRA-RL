import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Label each sample in advantages json with adv_quality_label. "
            "For each dataset_key/task, among samples with advantage > 0, "
            "top ratio are labeled positive, all others negative."
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input advantages_all_train1.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output labeled json",
    )
    parser.add_argument(
        "--top_ratio",
        type=float,
        default=0.3,
        help="Top ratio among advantage > 0 samples in each task to mark as positive. Default: 0.3",
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.0,
        help="Only samples with advantage > positive_threshold are candidates for positive. Default: 0.0",
    )
    parser.add_argument(
        "--min_positive_per_task",
        type=int,
        default=1,
        help="If a task has any valid positive-advantage samples, keep at least this many as positive. Default: 1",
    )
    return parser.parse_args()


def load_json(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "advantages" not in data:
        raise ValueError(f"Invalid format: key 'advantages' not found in {path}")

    if not isinstance(data["advantages"], list):
        raise ValueError("Invalid format: data['advantages'] must be a list")

    return data


def group_indices_by_task(advantages):
    task_to_indices = defaultdict(list)
    for idx, item in enumerate(advantages):
        dataset_key = item.get("dataset_key", "__unknown_task__")
        task_to_indices[dataset_key].append(idx)
    return task_to_indices


def assign_labels(data, top_ratio=0.3, positive_threshold=0.0, min_positive_per_task=1):
    advantages = data["advantages"]

    # 先默认全设为 negative
    for item in advantages:
        item["adv_quality_label"] = "negative"

    task_to_indices = group_indices_by_task(advantages)

    summary = {}

    for dataset_key, indices in task_to_indices.items():
        # 只保留 advantage > threshold 的样本作为候选
        positive_candidates = []
        for idx in indices:
            adv = advantages[idx].get("advantage", None)
            if adv is None:
                continue
            if adv > positive_threshold:
                positive_candidates.append((idx, float(adv)))

        # 按 advantage 从大到小排序
        positive_candidates.sort(key=lambda x: x[1], reverse=True)

        num_candidates = len(positive_candidates)

        if num_candidates == 0:
            summary[dataset_key] = {
                "total": len(indices),
                "adv_gt_threshold": 0,
                "positive_labeled": 0,
                "negative_labeled": len(indices),
            }
            continue

        keep_n = int(math.ceil(num_candidates * top_ratio))
        keep_n = max(min_positive_per_task, keep_n)
        keep_n = min(keep_n, num_candidates)

        positive_indices = set(idx for idx, _ in positive_candidates[:keep_n])

        for idx in positive_indices:
            advantages[idx]["adv_quality_label"] = "positive"

        summary[dataset_key] = {
            "total": len(indices),
            "adv_gt_threshold": num_candidates,
            "positive_labeled": len(positive_indices),
            "negative_labeled": len(indices) - len(positive_indices),
        }

    return data, summary


def print_summary(summary):
    print("\n========== Label Summary ==========")
    total_all = 0
    total_candidates = 0
    total_positive = 0
    total_negative = 0

    for dataset_key, stats in sorted(summary.items(), key=lambda x: x[0]):
        total_all += stats["total"]
        total_candidates += stats["adv_gt_threshold"]
        total_positive += stats["positive_labeled"]
        total_negative += stats["negative_labeled"]

        print(
            f"[{dataset_key}] "
            f"total={stats['total']} | "
            f"adv>thr={stats['adv_gt_threshold']} | "
            f"positive={stats['positive_labeled']} | "
            f"negative={stats['negative_labeled']}"
        )

    print("-----------------------------------")
    print(f"ALL total={total_all}")
    print(f"ALL adv>thr={total_candidates}")
    print(f"ALL positive={total_positive}")
    print(f"ALL negative={total_negative}")
    print("===================================\n")


def save_json(data, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved labeled json to: {path}")


def main():
    args = parse_args()

    if not (0 < args.top_ratio <= 1):
        raise ValueError(f"--top_ratio must be in (0, 1], got {args.top_ratio}")

    if args.min_positive_per_task < 0:
        raise ValueError(
            f"--min_positive_per_task must be >= 0, got {args.min_positive_per_task}"
        )

    data = load_json(args.input_json)

    data, summary = assign_labels(
        data=data,
        top_ratio=args.top_ratio,
        positive_threshold=args.positive_threshold,
        min_positive_per_task=args.min_positive_per_task,
    )

    print_summary(summary)
    save_json(data, args.output_json)


if __name__ == "__main__":
    main()