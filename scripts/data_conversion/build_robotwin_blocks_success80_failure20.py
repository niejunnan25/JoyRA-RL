#!/usr/bin/env python3
"""Build RobotWin block-ranking datasets with all failures and 4x success frames."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any


HELPER_PATH = Path(__file__).with_name("build_robotwin_blocks_balanced_success_failure.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("robotwin_blocks_balance_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_successes_for_ratio(
    successes: list[dict[str, Any]],
    target_frames: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    current = 0
    cycle = list(successes)
    while current < target_frames:
        rng.shuffle(cycle)
        for ep in cycle:
            prev = current
            chosen.append(ep)
            current += int(ep["length"])
            if current >= target_frames:
                without_last = chosen[:-1]
                if without_last and abs(prev - target_frames) < abs(current - target_frames):
                    return without_last
                return chosen
    return chosen


def build_one(
    helper,
    source_root: Path,
    output_root: Path,
    task_name: str,
    success_to_failure_frame_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    src = source_root / task_name
    episodes = helper.read_jsonl(src / "meta" / "episodes.jsonl")
    successes = [ep for ep in episodes if bool(ep.get("success"))]
    failures = [ep for ep in episodes if not bool(ep.get("success"))]
    failure_frames = sum(int(ep["length"]) for ep in failures)
    target_success_frames = round(failure_frames * success_to_failure_frame_ratio)

    rng = random.Random(seed + sum(ord(c) for c in task_name))
    selected_successes = choose_successes_for_ratio(successes, target_success_frames, rng)
    selected = list(failures) + selected_successes
    rng.shuffle(selected)

    success_frames = sum(int(ep["length"]) for ep in selected_successes)
    dst = output_root / f"{task_name}_success80_failure20"
    print(
        f"[select] {task_name}: failure_eps={len(failures)}, failure_frames={failure_frames}, "
        f"success_eps_copied={len(selected_successes)}, success_frames={success_frames}, "
        f"success/failure_frame_ratio={success_frames / max(1, failure_frames):.4f}",
        flush=True,
    )
    manifest = helper.copy_dataset(src, dst, selected, seed=seed, overwrite=overwrite)
    manifest.update(
        {
            "balanced_by": "all_failures_success_to_failure_frame_ratio",
            "success_to_failure_frame_ratio_target": success_to_failure_frame_ratio,
            "success_to_failure_frame_ratio_actual": success_frames / max(1, failure_frames),
            "all_failures_used": True,
            "success_episodes_are_oversampled": len(selected_successes) > len(successes),
            "unique_success_source_episodes": len({int(ep["episode_index"]) for ep in selected_successes}),
            "success_episode_copies": len(selected_successes),
            "failure_episode_copies": len(failures),
        }
    )
    helper.write_json(dst / "source_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/workspace/users/niejunnan/datasets/robotwin_rollout_lerobot"),
    )
    parser.add_argument("--tasks", nargs="+", default=["blocks_ranking_rgb", "blocks_ranking_size"])
    parser.add_argument("--success-to-failure-frame-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    helper = load_helper()
    manifests = [
        build_one(
            helper=helper,
            source_root=args.source_root.resolve(),
            output_root=args.output_root.resolve(),
            task_name=task_name,
            success_to_failure_frame_ratio=args.success_to_failure_frame_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        for task_name in args.tasks
    ]

    summary = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tasks": manifests,
        "total_episodes": sum(m["total_episodes"] for m in manifests),
        "total_frames": sum(m["total_frames"] for m in manifests),
        "success_episodes": sum(m["success_episodes"] for m in manifests),
        "failure_episodes": sum(m["failure_episodes"] for m in manifests),
        "success_frames": sum(m["success_frames"] for m in manifests),
        "failure_frames": sum(m["failure_frames"] for m in manifests),
        "success_to_failure_frame_ratio_target": args.success_to_failure_frame_ratio,
        "success_to_failure_frame_ratio_actual": (
            sum(m["success_frames"] for m in manifests) / max(1, sum(m["failure_frames"] for m in manifests))
        ),
        "all_failures_used": True,
        "seed": args.seed,
    }
    helper.write_json(args.output_root / "robotwin_pi05_blocks_ranking_success80_failure20_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
