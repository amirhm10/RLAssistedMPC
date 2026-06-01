from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS
from utils.rewards import make_reward_fn_relative_QR


BASELINE_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
OUT_DIR = ROOT / "report" / "figures" / "distillation_ofmpc_current_reward_20260601"


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _moving_average(values: np.ndarray, width: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < width:
        return arr.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _jsonable_reward_params(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            out[key] = value.astype(float).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def recompute_ofmpc_reward(bundle: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    n_inputs = int(len(np.asarray(bundle["steady_states"]["ss_inputs"], dtype=float)))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float).reshape(1, -1)
    y_range = (data_max[n_inputs:] - data_min[n_inputs:]).reshape(1, -1)
    y_sp_phys = y_ss + np.asarray(bundle["y_sp"], dtype=float) * y_range

    reward_params, reward_fn = make_reward_fn_relative_QR(
        data_min,
        data_max,
        n_inputs,
        **RL_REWARD_DEFAULTS,
    )
    delta_y = np.asarray(bundle["delta_y_storage"], dtype=float)
    delta_u = np.asarray(bundle["delta_u_storage"], dtype=float)
    rewards = np.asarray(
        [reward_fn(delta_y[i, :], delta_u[i, :], y_sp_phys[i, :]) for i in range(delta_y.shape[0])],
        dtype=float,
    )

    steps_per_episode = int(bundle["time_in_sub_episodes"])
    n_episodes = int(rewards.size // steps_per_episode)
    avg_rewards = rewards[: n_episodes * steps_per_episode].reshape(n_episodes, steps_per_episode).mean(axis=1)
    return rewards, avg_rewards, reward_params


def plot_avg_reward(avg_rewards: np.ndarray, reward_params: dict) -> Path:
    episodes = np.arange(1, avg_rewards.size + 1)
    smooth = _moving_average(avg_rewards, width=10)
    tail20 = float(np.mean(avg_rewards[-20:]))
    final_reward = float(avg_rewards[-1])

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.plot(episodes, avg_rewards, color="#4c78a8", linewidth=1.25, alpha=0.72, label="OF-MPC avg reward")
    ax.plot(episodes, smooth, color="#f58518", linewidth=2.1, label="10-episode moving average")
    ax.axhline(tail20, color="#54a24b", linewidth=1.3, linestyle="--", label=f"Tail-20 mean = {tail20:.3f}")
    ax.scatter([episodes[-1]], [final_reward], color="#e45756", s=38, zorder=4, label=f"Final = {final_reward:.3f}")
    ax.set_title("OF-MPC Average Reward With Current Distillation Reward", fontsize=14)
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.set_xlim(1, int(episodes[-1]))
    ax.grid(True, alpha=0.24)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    q = np.asarray(reward_params["Q_diag"], dtype=float)
    r = np.asarray(reward_params["R_diag"], dtype=float)
    text = (
        f"Q = [{q[0]:.0f}, {q[1]:.0f}], R = [{r[0]:.0f}, {r[1]:.0f}], "
        f"beta = {float(reward_params['beta']):.1f}, scale = {float(reward_params['reward_scale']):.1f}"
    )
    ax.text(
        0.01,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )
    fig.tight_layout()
    path = OUT_DIR / "fig_ofmpc_avg_reward_current_reward.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = _load_pickle(BASELINE_PATH)
    step_rewards, avg_rewards, reward_params = recompute_ofmpc_reward(bundle)
    episodes = np.arange(1, avg_rewards.size + 1)
    pd.DataFrame(
        {
            "episode": episodes,
            "avg_reward_current": avg_rewards,
            "moving_avg_10": _moving_average(avg_rewards, width=10),
        }
    ).to_csv(OUT_DIR / "ofmpc_current_reward_avg.csv", index=False)

    fig_path = plot_avg_reward(avg_rewards, reward_params)
    summary = {
        "source_data": str(BASELINE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "figure": str(fig_path.relative_to(ROOT)).replace("\\", "/"),
        "episodes": int(avg_rewards.size),
        "steps_per_episode": int(bundle["time_in_sub_episodes"]),
        "step_reward_mean": float(np.mean(step_rewards)),
        "avg_reward_mean": float(np.mean(avg_rewards)),
        "avg_reward_tail20": float(np.mean(avg_rewards[-20:])),
        "avg_reward_final": float(avg_rewards[-1]),
        "reward_params": _jsonable_reward_params(reward_params),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
