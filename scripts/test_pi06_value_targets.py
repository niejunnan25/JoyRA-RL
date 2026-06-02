import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_value_target_functions():
    datasets_stub = types.ModuleType("starVLA.dataloader.gr00t_lerobot.datasets")
    datasets_stub.LeRobotSingleDataset = object
    datasets_stub.LeRobotMixtureDataset = object
    datasets_stub.LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
    datasets_stub.LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/**/*.parquet"

    sys.modules.setdefault("starVLA", types.ModuleType("starVLA"))
    sys.modules.setdefault("starVLA.dataloader", types.ModuleType("starVLA.dataloader"))
    sys.modules.setdefault(
        "starVLA.dataloader.gr00t_lerobot",
        types.ModuleType("starVLA.dataloader.gr00t_lerobot"),
    )
    sys.modules["starVLA.dataloader.gr00t_lerobot.datasets"] = datasets_stub

    module_path = REPO_ROOT / "starVLA" / "dataloader" / "value_targets_wrapper.py"
    spec = importlib.util.spec_from_file_location("value_targets_wrapper_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.compute_normalized_returns_from_traj,
        module.compute_rewards_and_returns_from_traj,
    )


compute_normalized_returns_from_traj, compute_rewards_and_returns_from_traj = (
    load_value_target_functions()
)


def make_traj(length: int, success: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_success": [success] * length,
            "step": list(range(length)),
        }
    )


def test_raw_pi06_returns() -> None:
    _, success_returns = compute_rewards_and_returns_from_traj(
        make_traj(4, True),
        big_negative=10.0,
    )
    np.testing.assert_allclose(success_returns, np.array([-3.0, -2.0, -1.0, 0.0]))

    _, failure_returns = compute_rewards_and_returns_from_traj(
        make_traj(4, False),
        big_negative=10.0,
    )
    np.testing.assert_allclose(failure_returns, np.array([-13.0, -12.0, -11.0, -10.0]))


def test_normalized_success_terminal_uses_empirical_return() -> None:
    _, returns = compute_normalized_returns_from_traj(
        make_traj(400, True),
        big_negative=1200.0,
        denom=1199,
        use_big_negative_in_denom=False,
    )
    assert returns[-1] == 0.0
    np.testing.assert_allclose(returns[-2], -1.0 / 1199.0, rtol=1e-6)


def test_failure_penalty_propagates_before_terminal() -> None:
    _, returns = compute_normalized_returns_from_traj(
        make_traj(1200, False),
        big_negative=1200.0,
        denom=1199,
        use_big_negative_in_denom=False,
    )
    assert returns[-1] == -1.0
    assert returns[-2] == -1.0
    assert returns[0] == -1.0


if __name__ == "__main__":
    test_raw_pi06_returns()
    test_normalized_success_terminal_uses_empirical_return()
    test_failure_penalty_propagates_before_terminal()
    print("pi06 value target tests passed")
