from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_latest_run_20260509"

LATEST_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260509_023119" / "input_data.pkl"
LATEST_COMPARE = (
    REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_markov" / "20260509_023133" / "input_data.pkl"
)
PREVIOUS_RUN = (
    REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260508_123902" / "input_data.pkl"
)
CANONICAL_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"

WINDOW = 20
Z_BOUND = 0.05
SOURCE_NAMES = {
    1: "warm_start_ls",
    2: "td3_accepted",
    3: "ls_fallback",
    4: "nominal_fallback",
}
SOURCE_COLORS = {
    1: "#8c564b",
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#7f7f7f",
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def bonus_phi(z: np.ndarray, kind: str, k: float, p: float, c: float) -> np.ndarray:
    z = np.clip(z, 0.0, 1.0)
    if kind == "linear":
        return 1.0 - z
    if kind == "quadratic":
        return (1.0 - z) ** 2
    if kind == "exp":
        return (np.exp(-k * z) - np.exp(-k)) / (1.0 - np.exp(-k))
    if kind == "power":
        return 1.0 - np.power(z, p)
    if kind == "log":
        return np.log1p(c * (1.0 - z)) / np.log1p(c)
    raise ValueError(f"unknown bonus kind: {kind}")


def vector_shared_reward(
    e_scaled: np.ndarray,
    du_scaled: np.ndarray,
    y_sp_phys: np.ndarray,
    reward_params: dict,
    dy_phys: np.ndarray,
) -> np.ndarray:
    e_scaled = np.asarray(e_scaled, float)
    du_scaled = np.asarray(du_scaled, float)
    y_sp_phys = np.asarray(y_sp_phys, float)

    k_rel = np.asarray(reward_params["k_rel"], float).reshape(1, -1)
    band_floor_phys = np.asarray(reward_params["band_floor_phys"], float).reshape(1, -1)
    q_diag = np.asarray(reward_params["Q_diag"], float).reshape(1, -1)
    r_diag = np.asarray(reward_params["R_diag"], float).reshape(1, -1)

    band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor_phys)
    band_scaled = band_phys / np.maximum(dy_phys.reshape(1, -1), 1.0e-12)
    tau_scaled = float(reward_params["tau_frac"]) * band_scaled

    abs_e = np.abs(e_scaled)
    s_i = sigmoid((band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12))
    gate = str(reward_params["gate"])
    if gate == "prod":
        w_in = np.prod(s_i, axis=1)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=1)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])
    else:
        raise ValueError(f"unknown gate: {gate}")

    err_quad = np.sum(q_diag * e_scaled**2, axis=1)
    lam_in = float(reward_params["lam_in"])
    err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)
    move = np.sum(r_diag * du_scaled**2, axis=1)

    slope_at_edge = 2.0 * q_diag * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    gamma_out = float(reward_params["gamma_out"])
    gamma_in = float(reward_params["gamma_in"])
    lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow, axis=1)
    lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag, axis=1)

    qb2 = q_diag * band_scaled**2
    z = abs_e / np.maximum(band_scaled, 1.0e-12)
    phi = bonus_phi(
        z,
        kind=str(reward_params["bonus_kind"]),
        k=float(reward_params["bonus_k"]),
        p=float(reward_params["bonus_p"]),
        c=float(reward_params["bonus_c"]),
    )
    beta = float(reward_params["beta"])
    bonus = w_in * beta * np.sum(qb2 * phi, axis=1)
    reward_scale = float(reward_params["reward_scale"])
    return (-(err_eff + move + lin_out + lin_in) + bonus) * reward_scale


def physical_to_scaled_abs(values: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    return (np.asarray(values, float) - np.asarray(data_min, float)) / np.maximum(
        np.asarray(data_max, float) - np.asarray(data_min, float), 1.0e-12
    )


def rewards_from_trajectories(
    y_phys: np.ndarray,
    u_phys: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
    reward_params: dict,
) -> np.ndarray:
    y_phys = np.asarray(y_phys, float)
    u_phys = np.asarray(u_phys, float)
    y_sp_scaled = np.asarray(y_sp_scaled, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    n_inputs = u_phys.shape[1]

    ss_u_phys = np.asarray(steady_states["ss_inputs"], float)
    ss_y_phys = np.asarray(steady_states["y_ss"], float)
    ss_u_scaled = physical_to_scaled_abs(ss_u_phys, data_min[:n_inputs], data_max[:n_inputs])
    ss_y_scaled = physical_to_scaled_abs(ss_y_phys, data_min[n_inputs:], data_max[n_inputs:])

    u_scaled = physical_to_scaled_abs(u_phys, data_min[:n_inputs], data_max[:n_inputs])
    u_dev = u_scaled - ss_u_scaled.reshape(1, -1)
    du = np.vstack([np.zeros((1, n_inputs), dtype=float), np.diff(u_dev, axis=0)])

    y_scaled = physical_to_scaled_abs(y_phys[1:], data_min[n_inputs:], data_max[n_inputs:])
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    e_scaled = y_dev - y_sp_scaled
    y_sp_phys = (y_sp_scaled + ss_y_scaled.reshape(1, -1)) * (data_max[n_inputs:] - data_min[n_inputs:]).reshape(
        1, -1
    ) + data_min[n_inputs:].reshape(1, -1)
    dy_phys = data_max[n_inputs:] - data_min[n_inputs:]
    return vector_shared_reward(e_scaled, du, y_sp_phys, reward_params, dy_phys)


def moving_average(values: np.ndarray, width: int = 10) -> np.ndarray:
    values = np.asarray(values, float)
    if values.size < width:
        return values.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def compute_episode_mae(y_phys: np.ndarray, y_sp_scaled: np.ndarray, steady_states: dict, data_min, data_max):
    y_phys = np.asarray(y_phys, float)
    y_sp_scaled = np.asarray(y_sp_scaled, float)
    ss_y_scaled = physical_to_scaled_abs(
        np.asarray(steady_states["y_ss"], float), np.asarray(data_min, float)[2:], np.asarray(data_max, float)[2:]
    )
    y_scaled = physical_to_scaled_abs(y_phys[1:], np.asarray(data_min, float)[2:], np.asarray(data_max, float)[2:])
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    return np.abs(y_dev - y_sp_scaled)


def reshape_run(current: dict, baseline: dict | None = None) -> dict:
    if "rewards_markov" in current:
        n_episodes = len(np.asarray(current["avg_rewards_markov"], float))
        n_steps = len(np.asarray(current["rewards_markov"], float)) // n_episodes
        return {
            "kind": "previous",
            "n_episodes": n_episodes,
            "n_steps_per_episode": n_steps,
            "reward_markov": np.asarray(current["rewards_markov"], float).reshape(n_episodes, n_steps),
            "reward_nominal": np.asarray(current["rewards_nominal"], float).reshape(n_episodes, n_steps),
            "y_markov": np.asarray(current["y_markov"], float),
            "u_markov": np.asarray(current["u_markov"], float),
            "y_nominal": np.asarray(current["y_nominal"], float),
            "u_nominal": np.asarray(current["u_nominal"], float),
            "y_sp": np.asarray(current["y_sp"], float).reshape(n_episodes, n_steps, -1),
            "steady_states": current["steady_states"],
            "data_min": np.asarray(current["data_min"], float),
            "data_max": np.asarray(current["data_max"], float),
            "action_source": np.asarray(current["rl_action_source_log"], int).reshape(n_episodes, n_steps),
            "z_executed": np.asarray(current["z_executed_log"], float).reshape(n_episodes, n_steps, -1),
            "z_requested": np.asarray(current["rl_requested_z_log"], float).reshape(n_episodes, n_steps, -1),
            "z_ls": np.asarray(current["rl_ls_z_log"], float).reshape(n_episodes, n_steps, -1),
            "scores": np.asarray(current["s_pred_log"], float).reshape(n_episodes, n_steps),
            "drift": np.asarray(current["gain_drift_log"], float).reshape(n_episodes, n_steps),
            "reward_delta_original": np.asarray(current["rewards_markov"], float).reshape(n_episodes, n_steps).mean(
                axis=1
            )
            - np.asarray(current["rewards_nominal"], float).reshape(n_episodes, n_steps).mean(axis=1),
        }

    if baseline is None:
        raise ValueError("baseline bundle is required for current unified run analysis")

    n_episodes = len(np.asarray(current["avg_rewards"], float))
    n_steps = len(np.asarray(current["rewards_step"], float)) // n_episodes
    return {
        "kind": "latest",
        "n_episodes": n_episodes,
        "n_steps_per_episode": n_steps,
        "reward_markov": np.asarray(current["rewards_step"], float).reshape(n_episodes, n_steps),
        "reward_nominal": np.asarray(baseline["rewards_step"], float).reshape(n_episodes, n_steps),
        "y_markov": np.asarray(current["y"], float),
        "u_markov": np.asarray(current["u"], float),
        "y_nominal": np.asarray(baseline["y_mpc"], float),
        "u_nominal": np.asarray(baseline["u_mpc"], float),
        "y_sp": np.asarray(current["y_sp"], float).reshape(n_episodes, n_steps, -1),
        "steady_states": current["steady_states"],
        "data_min": np.asarray(current["data_min"], float),
        "data_max": np.asarray(current["data_max"], float),
        "action_source": np.asarray(current["rl_action_source_log"], int).reshape(n_episodes, n_steps),
        "z_executed": np.asarray(current["z_executed_log"], float).reshape(n_episodes, n_steps, -1),
        "z_requested": np.asarray(current["rl_requested_z_log"], float).reshape(n_episodes, n_steps, -1),
        "z_ls": np.asarray(current["rl_ls_z_log"], float).reshape(n_episodes, n_steps, -1),
        "scores": np.asarray(current["s_pred_log"], float).reshape(n_episodes, n_steps),
        "drift": np.asarray(current["gain_drift_log"], float).reshape(n_episodes, n_steps),
        "reward_delta_original": np.asarray(current["rewards_step"], float).reshape(n_episodes, n_steps).mean(axis=1)
        - np.asarray(baseline["rewards_step"], float).reshape(n_episodes, n_steps).mean(axis=1),
    }


def episode_input_movement(u_phys: np.ndarray, n_episodes: int, n_steps: int) -> np.ndarray:
    u = np.asarray(u_phys, float).reshape(n_episodes, n_steps, -1)
    return np.mean(np.linalg.norm(np.diff(u, axis=1), axis=2), axis=1)


def episode_mae(y_phys: np.ndarray, y_sp: np.ndarray, steady_states, data_min, data_max, n_episodes, n_steps):
    abs_err = compute_episode_mae(y_phys, y_sp.reshape(n_episodes * n_steps, -1), steady_states, data_min, data_max)
    return abs_err.reshape(n_episodes, n_steps, -1).mean(axis=1)


def source_fraction_by_window(action_source: np.ndarray, code: int) -> np.ndarray:
    n_episodes = action_source.shape[0]
    out = []
    for start in range(0, n_episodes, WINDOW):
        stop = min(start + WINDOW, n_episodes)
        out.append(float((action_source[start:stop] == code).mean()))
    return np.asarray(out, float)


def window_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    out = []
    for start in range(0, len(values), WINDOW):
        stop = min(start + WINDOW, len(values))
        out.append(float(values[start:stop].mean()))
    return np.asarray(out, float)


def summarize_run(label: str, run: dict, shared_reward_markov: np.ndarray | None = None, shared_reward_nominal: np.ndarray | None = None):
    n_episodes = run["n_episodes"]
    n_steps = run["n_steps_per_episode"]
    reward_delta = run["reward_delta_original"]
    y_sp = run["y_sp"]

    mae_markov = episode_mae(
        run["y_markov"], y_sp, run["steady_states"], run["data_min"], run["data_max"], n_episodes, n_steps
    )
    mae_nominal = episode_mae(
        run["y_nominal"], y_sp, run["steady_states"], run["data_min"], run["data_max"], n_episodes, n_steps
    )
    move_markov = episode_input_movement(run["u_markov"], n_episodes, n_steps)
    move_nominal = episode_input_movement(run["u_nominal"], n_episodes, n_steps)

    z_abs = np.abs(run["z_executed"])
    z_sat_any = (z_abs.max(axis=2) >= 0.98 * Z_BOUND).astype(float)

    summary = {
        "label": label,
        "reward_delta_mean": float(reward_delta.mean()),
        "reward_delta_last20": float(reward_delta[-20:].mean()),
        "reward_better_frac": float((reward_delta > 0.0).mean()),
        "reward_better_last20_frac": float((reward_delta[-20:] > 0.0).mean()),
        "output1_mae_delta_mean": float((mae_markov[:, 0] - mae_nominal[:, 0]).mean()),
        "output2_mae_delta_mean": float((mae_markov[:, 1] - mae_nominal[:, 1]).mean()),
        "output1_mae_delta_last20": float((mae_markov[-20:, 0] - mae_nominal[-20:, 0]).mean()),
        "output2_mae_delta_last20": float((mae_markov[-20:, 1] - mae_nominal[-20:, 1]).mean()),
        "input_move_delta_mean": float((move_markov - move_nominal).mean()),
        "input_move_delta_last20": float((move_markov[-20:] - move_nominal[-20:]).mean()),
        "td3_fraction": float((run["action_source"] == 2).mean()),
        "ls_fraction": float((run["action_source"] == 3).mean()),
        "nominal_fraction": float((run["action_source"] == 4).mean()),
        "warm_fraction": float((run["action_source"] == 1).mean()),
        "z_sat_any98_mean": float(z_sat_any.mean()),
        "z_sat_any98_last20": float(z_sat_any[-20:].mean()),
        "z_abs_mean_1": float(z_abs[..., 0].mean()),
        "z_abs_mean_2": float(z_abs[..., 1].mean()),
        "z_abs_mean_3": float(z_abs[..., 2].mean()),
        "z_abs_mean_4": float(z_abs[..., 3].mean()),
        "score_mean": float(run["scores"].mean()),
        "drift_mean": float(run["drift"].mean()),
    }
    if shared_reward_markov is not None and shared_reward_nominal is not None:
        shared_delta = shared_reward_markov.reshape(n_episodes, n_steps).mean(axis=1) - shared_reward_nominal.reshape(
            n_episodes, n_steps
        ).mean(axis=1)
        summary["shared_reward_delta_mean"] = float(shared_delta.mean())
        summary["shared_reward_delta_last20"] = float(shared_delta[-20:].mean())
        summary["shared_reward_better_frac"] = float((shared_delta > 0.0).mean())
        summary["shared_reward_better_last20_frac"] = float((shared_delta[-20:] > 0.0).mean())
    return summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_reward_delta_compare(latest_summary: dict, previous_summary: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    series = [
        ("Latest Unified Run", latest_summary["reward_delta_original"]),
        ("Previous Prototype Run", previous_summary["reward_delta_original"]),
    ]
    for ax, (title, delta) in zip(axes, series):
        episodes = np.arange(1, len(delta) + 1)
        ax.plot(episodes, delta, color="#9ecae1", linewidth=1.0, alpha=0.8, label="episode delta")
        ax.plot(episodes, moving_average(delta, 10), color="#08519c", linewidth=2.2, label="10-episode mean")
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Markov minus MPC reward")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tail_outputs_compare(latest_summary: dict, previous_summary: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col", constrained_layout=True)
    series = [("Latest Unified Run", latest_summary), ("Previous Prototype Run", previous_summary)]
    labels = ["Viscosity", "Temperature"]
    for col, (title, summary) in enumerate(series):
        n_steps = summary["n_steps_per_episode"]
        delta_t = float(summary["delta_t"])
        time_hours = np.arange(n_steps) * delta_t
        y_markov = summary["y_markov"][-n_steps:, :]
        y_nominal = summary["y_nominal"][-n_steps:, :]
        y_sp_phys = summary["y_sp_phys_last"]
        for row in range(2):
            ax = axes[row, col]
            ax.plot(time_hours, y_markov[:, row], color="#1f77b4", linewidth=2.0, label="Markov")
            ax.plot(time_hours, y_nominal[:, row], color="#ff7f0e", linewidth=1.8, label="MPC")
            ax.plot(time_hours, y_sp_phys[:, row], color="#2ca02c", linestyle="--", linewidth=1.6, label="Setpoint")
            ax.set_title(title if row == 0 else "")
            ax.set_ylabel(labels[row])
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("Time in final episode [h]")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_action_mix_and_saturation(latest_summary: dict, previous_summary: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, (title, summary) in enumerate(
        [("Latest Unified Run", latest_summary), ("Previous Prototype Run", previous_summary)]
    ):
        windows = np.arange(len(summary["reward_delta_window"]))
        base = np.zeros_like(windows, dtype=float)
        ax_top = axes[0, col]
        for code in [1, 2, 3, 4]:
            vals = summary["source_window"][code]
            ax_top.bar(
                windows,
                vals,
                bottom=base,
                color=SOURCE_COLORS[code],
                width=0.8,
                label=SOURCE_NAMES[code],
            )
            base += vals
        ax_top.set_title(title)
        ax_top.set_ylabel("Fraction of steps")
        ax_top.set_ylim(0.0, 1.0)
        ax_top.set_xticks(windows)
        ax_top.set_xticklabels(summary["window_labels"], rotation=30)
        ax_top.grid(axis="y", alpha=0.25)
        if col == 0:
            ax_top.legend(loc="upper right", fontsize=8)

        ax_bot = axes[1, col]
        ax_bot.plot(windows, summary["reward_delta_window"], color="#08519c", marker="o", linewidth=2.0)
        ax_bot.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax_bot.set_ylabel("Window reward delta")
        ax_bot.set_xlabel("Episode window")
        ax_bot.set_xticks(windows)
        ax_bot.set_xticklabels(summary["window_labels"], rotation=30)
        ax_bot.grid(alpha=0.25)
        ax_sat = ax_bot.twinx()
        ax_sat.plot(
            windows,
            summary["z_sat_any98_window"],
            color="#d62728",
            marker="s",
            linewidth=1.8,
            alpha=0.85,
        )
        ax_sat.set_ylabel("Any-coordinate saturation")
        ax_sat.set_ylim(0.0, 1.05)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_z_usage_compare(latest_summary: dict, previous_summary: dict, out_path: Path):
    labels = ["z1", "z2", "z3", "z4"]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].bar(
        x - width / 2,
        latest_summary["coord_abs_mean"] / Z_BOUND,
        width=width,
        color="#1f77b4",
        label="latest",
    )
    axes[0].bar(
        x + width / 2,
        previous_summary["coord_abs_mean"] / Z_BOUND,
        width=width,
        color="#ff7f0e",
        label="previous",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(r"Mean $|z_i| / z_{\max}$")
    axes[0].set_title("Average correction usage")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].bar(
        x - width / 2,
        latest_summary["coord_sat_frac"],
        width=width,
        color="#1f77b4",
        label="latest",
    )
    axes[1].bar(
        x + width / 2,
        previous_summary["coord_sat_frac"],
        width=width,
        color="#ff7f0e",
        label="previous",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Fraction with |z_i| >= 0.98 zmax")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Coordinate-wise saturation")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_reward_rescoring(summary_rows: list[dict], out_path: Path):
    full_values = [
        summary_rows[1]["reward_delta_mean"],
        summary_rows[1]["shared_reward_delta_mean"],
        summary_rows[0]["shared_reward_delta_mean"],
    ]
    tail_values = [
        summary_rows[1]["reward_delta_last20"],
        summary_rows[1]["shared_reward_delta_last20"],
        summary_rows[0]["shared_reward_delta_last20"],
    ]
    labels = ["Previous original", "Previous rescored", "Latest unified"]
    colors = ["#ff7f0e", "#9467bd", "#1f77b4"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, vals, title in zip(
        axes,
        [full_values, tail_values],
        ["Full-run reward delta", "Last-20-episode reward delta"],
    ):
        bars = ax.bar(labels, vals, color=colors)
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Markov minus MPC reward")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        span = max(abs(min(vals)), abs(max(vals)), 1.0)
        for bar, value in zip(bars, vals):
            y = value + 0.03 * span if value >= 0.0 else value - 0.05 * span
            va = "bottom" if value >= 0.0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{value:.4f}",
                ha="center",
                va=va,
                fontsize=9,
                rotation=90,
            )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_plot_summary(run: dict, raw_bundle: dict) -> dict:
    n_episodes = run["n_episodes"]
    n_steps = run["n_steps_per_episode"]
    labels = []
    for start in range(0, n_episodes, WINDOW):
        stop = min(start + WINDOW, n_episodes)
        labels.append(f"{start + 1}-{stop}")

    z_abs = np.abs(run["z_executed"])
    coord_abs_mean = z_abs.reshape(n_episodes * n_steps, -1).mean(axis=0)
    coord_sat_frac = (z_abs.reshape(n_episodes * n_steps, -1) >= 0.98 * Z_BOUND).mean(axis=0)
    y_sp_scaled_last = run["y_sp"][-1]
    ss_y_scaled = physical_to_scaled_abs(
        np.asarray(run["steady_states"]["y_ss"], float), run["data_min"][2:], run["data_max"][2:]
    )
    y_sp_phys_last = (y_sp_scaled_last + ss_y_scaled.reshape(1, -1)) * (
        run["data_max"][2:] - run["data_min"][2:]
    ).reshape(1, -1) + run["data_min"][2:].reshape(1, -1)
    return {
        "n_steps_per_episode": n_steps,
        "delta_t": float(raw_bundle["delta_t"]),
        "y_markov": run["y_markov"],
        "y_nominal": run["y_nominal"],
        "y_sp_phys_last": y_sp_phys_last,
        "reward_delta_original": run["reward_delta_original"],
        "reward_delta_window": window_mean(run["reward_delta_original"]),
        "z_sat_any98_window": window_mean((z_abs.max(axis=2) >= 0.98 * Z_BOUND).astype(float)),
        "source_window": {code: source_fraction_by_window(run["action_source"], code) for code in SOURCE_NAMES},
        "window_labels": labels,
        "coord_abs_mean": coord_abs_mean,
        "coord_sat_frac": coord_sat_frac,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    latest_raw = load_pickle(LATEST_RUN)
    latest_compare = load_pickle(LATEST_COMPARE)
    previous_raw = load_pickle(PREVIOUS_RUN)
    canonical_raw = load_pickle(CANONICAL_BASELINE)

    latest_run = reshape_run(latest_raw, canonical_raw)
    previous_run = reshape_run(previous_raw)

    latest_shared_markov = np.asarray(latest_raw["rewards_step"], float)
    latest_shared_nominal = np.asarray(canonical_raw["rewards_step"], float)
    previous_shared_markov = rewards_from_trajectories(
        previous_raw["y_markov"],
        previous_raw["u_markov"],
        previous_raw["y_sp"],
        previous_raw["steady_states"],
        latest_raw["data_min"],
        latest_raw["data_max"],
        latest_raw["reward_params"],
    )
    previous_shared_nominal = rewards_from_trajectories(
        previous_raw["y_nominal"],
        previous_raw["u_nominal"],
        previous_raw["y_sp"],
        previous_raw["steady_states"],
        latest_raw["data_min"],
        latest_raw["data_max"],
        latest_raw["reward_params"],
    )
    canonical_shared_nominal = rewards_from_trajectories(
        canonical_raw["y_mpc"],
        canonical_raw["u_mpc"],
        canonical_raw["y_sp"],
        previous_raw["steady_states"],
        latest_raw["data_min"],
        latest_raw["data_max"],
        latest_raw["reward_params"],
    )

    latest_summary = summarize_run(
        "latest_unified",
        latest_run,
        shared_reward_markov=latest_shared_markov,
        shared_reward_nominal=latest_shared_nominal,
    )
    previous_summary = summarize_run(
        "previous_prototype",
        previous_run,
        shared_reward_markov=previous_shared_markov,
        shared_reward_nominal=previous_shared_nominal,
    )

    comparison_rows = [latest_summary, previous_summary]
    write_csv(FIG_DIR / "comparison_summary.csv", comparison_rows)

    previous_nominal_vs_canonical = (
        previous_shared_nominal.reshape(previous_run["n_episodes"], previous_run["n_steps_per_episode"]).mean(axis=1)
        - canonical_shared_nominal.reshape(previous_run["n_episodes"], previous_run["n_steps_per_episode"]).mean(axis=1)
    )
    baseline_rows = [
        {
            "comparison": "previous_nominal_vs_canonical_baseline",
            "shared_reward_delta_mean": float(previous_nominal_vs_canonical.mean()),
            "shared_reward_delta_last20": float(previous_nominal_vs_canonical[-20:].mean()),
            "output_max_abs_diff": float(np.max(np.abs(previous_raw["y_nominal"] - canonical_raw["y_mpc"]))),
            "input_max_abs_diff": float(np.max(np.abs(previous_raw["u_nominal"] - canonical_raw["u_mpc"]))),
        }
    ]
    write_csv(FIG_DIR / "baseline_reference_difference.csv", baseline_rows)

    window_rows = []
    for run_label, run in [("latest_unified", latest_run), ("previous_prototype", previous_run)]:
        reward_delta = run["reward_delta_original"]
        z_sat = (np.abs(run["z_executed"]).max(axis=2) >= 0.98 * Z_BOUND).astype(float)
        for idx, start in enumerate(range(0, run["n_episodes"], WINDOW)):
            stop = min(start + WINDOW, run["n_episodes"])
            row = {
                "run": run_label,
                "episode_window": f"{start + 1}-{stop}",
                "reward_delta_mean": float(reward_delta[start:stop].mean()),
                "reward_better_fraction": float((reward_delta[start:stop] > 0.0).mean()),
                "z_sat_any98_fraction": float(z_sat[start:stop].mean()),
            }
            for code, name in SOURCE_NAMES.items():
                row[name] = float((run["action_source"][start:stop] == code).mean())
            window_rows.append(row)
    write_csv(FIG_DIR / "window_metrics.csv", window_rows)

    plot_latest = build_plot_summary(latest_run, latest_raw)
    plot_previous = build_plot_summary(previous_run, previous_raw)

    plot_reward_delta_compare(plot_latest, plot_previous, FIG_DIR / "reward_delta_compare.png")
    plot_tail_outputs_compare(plot_latest, plot_previous, FIG_DIR / "tail_output_compare.png")
    plot_action_mix_and_saturation(plot_latest, plot_previous, FIG_DIR / "action_mix_and_saturation.png")
    plot_z_usage_compare(plot_latest, plot_previous, FIG_DIR / "z_usage_compare.png")
    plot_reward_rescoring(comparison_rows, FIG_DIR / "reward_rescoring_compare.png")

    compare_rows = [
        {
            "rl_dir": str(latest_compare["rl_dir"]),
            "mpc_path_or_dir": str(latest_compare["mpc_path_or_dir"]),
            "latest_compare_dir": str(LATEST_COMPARE.parent),
            "latest_run_dir": str(LATEST_RUN.parent),
            "previous_run_dir": str(PREVIOUS_RUN.parent),
        }
    ]
    write_csv(FIG_DIR / "run_references.csv", compare_rows)

    print(FIG_DIR)


if __name__ == "__main__":
    main()
