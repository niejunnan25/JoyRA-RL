import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read advantages_all_train1.json and visualize one task / one trajectory."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to advantages_all_train1.json",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="dataset_key / task name. Can also be a full path; script will auto use the last part.",
    )
    parser.add_argument(
        "--trajectory_id",
        type=int,
        default=None,
        help="trajectory_id to visualize",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output image path, e.g. ./traj_vis.png",
    )
    parser.add_argument(
        "--show_raw_adv",
        action="store_true",
        help="Also plot advantage_raw",
    )
    parser.add_argument(
        "--show_bootstrap",
        action="store_true",
        help="Also plot bootstrap_value and n_step_reward",
    )
    parser.add_argument(
        "--list_only",
        action="store_true",
        help="Only print all tasks and trajectories, do not plot",
    )
    parser.add_argument(
        "--print_task_trajs",
        action="store_true",
        help="Print each task and its trajectory ids",
    )
    return parser.parse_args()


def load_advantages(json_path: str) -> pd.DataFrame:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "advantages" not in data:
        raise ValueError(f"Invalid json format: key 'advantages' not found in {json_path}")

    df = pd.DataFrame(data["advantages"])

    required_cols = ["dataset_key", "trajectory_id", "step", "value_t", "advantage"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in json: {missing}")

    return df


def print_all_tasks(df: pd.DataFrame):
    tasks = sorted(df["dataset_key"].astype(str).unique().tolist())
    print("\n=== ALL TASKS IN JSON ===")
    for t in tasks:
        print(t)
    print("========================\n")


def print_task_and_trajs(df: pd.DataFrame):
    print("\n=== TASK -> TRAJECTORIES ===")
    grouped = df.groupby("dataset_key")["trajectory_id"].unique()
    for task in sorted(grouped.index.tolist()):
        trajs = sorted([int(x) for x in grouped[task]])
        preview = trajs[:20]
        suffix = " ..." if len(trajs) > 20 else ""
        print(f"{task}: {preview}{suffix}")
    print("===========================\n")


def normalize_task_input(task: str) -> str:
    if task is None:
        return None
    task = str(task).strip()
    # 支持完整路径输入，只取最后一段
    return task.split("/")[-1]


def match_task(df: pd.DataFrame, task: str):
    """
    先精确匹配原始 task；
    再匹配 split('/')[-1] 后的 task 名称；
    再做 contains 模糊匹配。
    """
    if task is None:
        return None, df.iloc[0:0].copy()

    raw_task = str(task).strip()
    short_task = normalize_task_input(raw_task)

    dataset_keys = df["dataset_key"].astype(str)

    # 1) 精确匹配完整输入
    df_task = df[dataset_keys == raw_task].copy()
    if len(df_task) > 0:
        return raw_task, df_task

    # 2) 精确匹配最后一段
    df_task = df[dataset_keys == short_task].copy()
    if len(df_task) > 0:
        return short_task, df_task

    # 3) 用最后一段去匹配 dataset_key 的最后一段
    matched_mask = dataset_keys.apply(lambda x: x.split("/")[-1] == short_task)
    df_task = df[matched_mask].copy()
    if len(df_task) > 0:
        matched_names = sorted(df_task["dataset_key"].astype(str).unique().tolist())
        if len(matched_names) == 1:
            return matched_names[0], df_task

    # 4) contains 模糊匹配
    fuzzy_mask = dataset_keys.str.contains(short_task, case=False, regex=False)
    df_task = df[fuzzy_mask].copy()
    if len(df_task) > 0:
        matched_names = sorted(df_task["dataset_key"].astype(str).unique().tolist())
        if len(matched_names) == 1:
            return matched_names[0], df_task
        else:
            print("\n[Info] No exact match, but found multiple fuzzy-matched tasks:")
            for name in matched_names:
                print(f"  {name}")
            print()
            return None, df.iloc[0:0].copy()

    return None, df.iloc[0:0].copy()


def print_available_trajs(df_task: pd.DataFrame, task_name: str):
    print(f"\n=== AVAILABLE TRAJECTORIES FOR TASK: {task_name} ===")
    traj_lengths = (
        df_task.groupby("trajectory_id")["step"]
        .max()
        .sort_index()
    )
    for traj_id, max_step in traj_lengths.items():
        num_steps = len(df_task[df_task["trajectory_id"] == traj_id])
        print(f"trajectory_id={int(traj_id):4d}, num_points={num_steps:4d}, max_step={int(max_step):4d}")
    print("====================================================\n")


def plot_trajectory(
    df: pd.DataFrame,
    task: str,
    trajectory_id: int,
    show_raw_adv: bool = False,
    show_bootstrap: bool = False,
    save_path: str = None,
):
    matched_task_name, df_task = match_task(df, task)

    if len(df_task) == 0:
        print_all_tasks(df)
        raise ValueError(f"Task '{task}' not found in json.")

    df_traj = df_task[df_task["trajectory_id"] == trajectory_id].copy()
    if len(df_traj) == 0:
        print_available_trajs(df_task, matched_task_name)
        raise ValueError(
            f"trajectory_id={trajectory_id} not found under task '{matched_task_name}'."
        )

    df_traj = df_traj.sort_values("step").reset_index(drop=True)

    steps = df_traj["step"].values

    plt.figure(figsize=(12, 6))
    plt.plot(steps, df_traj["advantage"].values, marker="o", label="advantage")
    plt.plot(steps, df_traj["value_t"].values, marker="s", label="value_t")

    if show_raw_adv and "advantage_raw" in df_traj.columns:
        plt.plot(steps, df_traj["advantage_raw"].values, marker="^", label="advantage_raw")

    if show_bootstrap:
        if "bootstrap_value" in df_traj.columns:
            plt.plot(steps, df_traj["bootstrap_value"].values, marker="x", label="bootstrap_value")
        if "n_step_reward" in df_traj.columns:
            plt.plot(steps, df_traj["n_step_reward"].values, marker="d", label="n_step_reward")

    success_flag = df_traj["success"].iloc[0] if "success" in df_traj.columns else "N/A"

    plt.title(
        f"Task: {matched_task_name} | Trajectory: {trajectory_id} | Success: {success_flag}"
    )
    plt.xlabel("Step")
    plt.ylabel("Value / Advantage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    print("\n=== TRAJECTORY SUMMARY ===")
    print(f"matched task  : {matched_task_name}")
    print(f"trajectory_id : {trajectory_id}")
    print(f"num_steps     : {len(df_traj)}")
    print(f"step range    : [{df_traj['step'].min()}, {df_traj['step'].max()}]")
    if "success" in df_traj.columns:
        print(f"success       : {success_flag}")
    print("==========================\n")

    cols_to_show = ["step", "value_t", "advantage"]
    if show_raw_adv and "advantage_raw" in df_traj.columns:
        cols_to_show.append("advantage_raw")
    if show_bootstrap:
        if "bootstrap_value" in df_traj.columns:
            cols_to_show.append("bootstrap_value")
        if "n_step_reward" in df_traj.columns:
            cols_to_show.append("n_step_reward")

    print(df_traj[cols_to_show].head(30).to_string(index=False))

    if save_path is not None:
        save_path = str(Path(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to: {save_path}")

    plt.show()


def main():
    args = parse_args()

    df = load_advantages(args.json_path)

    # 先打印所有 task，方便你直接复制名字
    print_all_tasks(df)

    if args.print_task_trajs:
        print_task_and_trajs(df)

    if args.list_only:
        return

    if args.task is None:
        raise ValueError("Please provide --task, or use --list_only to only print tasks.")

    matched_task_name, df_task = match_task(df, args.task)
    if len(df_task) == 0:
        raise ValueError(f"Task '{args.task}' not found in json.")

    # 如果只给 task，不给 trajectory_id，就先打印这个 task 下所有轨迹
    if args.trajectory_id is None:
        print_available_trajs(df_task, matched_task_name)
        return

    plot_trajectory(
        df=df,
        task=args.task,
        trajectory_id=args.trajectory_id,
        show_raw_adv=args.show_raw_adv,
        show_bootstrap=args.show_bootstrap,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()