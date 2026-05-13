from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_markov_latest_20260513"

RUN_DIR = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_unified"
    / "20260512_090635"
)
RUN_BUNDLE = RUN_DIR / "input_data.pkl"
COMPARE_BUNDLE = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_markov_td3_disturb_fluctuation"
    / "20260512_090646"
    / "input_data.pkl"
)
BASELINE_BUNDLE = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"

SOURCE_COLORS = {
    "td3_accepted": "#0f766e",
    "ls_fallback": "#ea580c",
    "nominal": "#6b7280",
}
WINDOW = 20
MOVING_AVERAGE = 10


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, float)
    if values.size < width:
        return values.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def bootstrap_ci(values: np.ndarray, seed: int = 20260513, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float).reshape(-1)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def reconstruct_setpoint_phys(bundle: dict) -> np.ndarray:
    n_inputs = len(np.asarray(bundle["steady_states"]["ss_inputs"], float))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float).reshape(1, -1)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_span = (data_max[n_inputs:] - data_min[n_inputs:]).reshape(1, -1)
    return y_ss + np.asarray(bundle["y_sp"], float) * y_span


def reshape_steps(bundle: dict) -> tuple[int, int, int]:
    n_episodes = len(np.asarray(bundle["avg_rewards"], float))
    steps = int(bundle["time_in_sub_episodes"])
    warm_start_episodes = int(bundle["warm_start_step"]) // steps
    return n_episodes, steps, warm_start_episodes


def episode_metrics(
    rl_bundle: dict, baseline_bundle: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_episodes, steps, _warm_start_episodes = reshape_steps(rl_bundle)

    y_sp_phys = reconstruct_setpoint_phys(rl_bundle).reshape(n_episodes, steps, -1)
    y_rl = np.asarray(rl_bundle["y"][1:], float).reshape(n_episodes, steps, -1)
    y_mpc = np.asarray(baseline_bundle["y"][1:], float).reshape(n_episodes, steps, -1)
    u_rl = np.asarray(rl_bundle["u"], float).reshape(n_episodes, steps, -1)
    u_mpc = np.asarray(baseline_bundle["u"], float).reshape(n_episodes, steps, -1)

    mae_rl = np.mean(np.abs(y_rl - y_sp_phys), axis=1)
    mae_mpc = np.mean(np.abs(y_mpc - y_sp_phys), axis=1)
    rmse_rl = np.sqrt(np.mean((y_rl - y_sp_phys) ** 2, axis=1))
    rmse_mpc = np.sqrt(np.mean((y_mpc - y_sp_phys) ** 2, axis=1))
    last_err_rl = np.abs(y_rl[:, -1, :] - y_sp_phys[:, -1, :])
    last_err_mpc = np.abs(y_mpc[:, -1, :] - y_sp_phys[:, -1, :])

    input_span = np.asarray(rl_bundle["data_max"], float)[: u_rl.shape[2]] - np.asarray(rl_bundle["data_min"], float)[
        : u_rl.shape[2]
    ]
    du_rl = np.diff(u_rl, axis=1) / input_span.reshape(1, 1, -1)
    du_mpc = np.diff(u_mpc, axis=1) / input_span.reshape(1, 1, -1)
    move_rl = np.mean(np.linalg.norm(du_rl, axis=2), axis=1)
    move_mpc = np.mean(np.linalg.norm(du_mpc, axis=2), axis=1)

    return mae_rl, mae_mpc, rmse_rl, rmse_mpc, last_err_rl, last_err_mpc, move_rl, move_mpc


def compute_summary(rl_bundle: dict, baseline_bundle: dict, compare_bundle: dict) -> dict:
    n_episodes, steps, warm_start_episodes = reshape_steps(rl_bundle)
    reward_rl = np.asarray(compare_bundle["avg_rewards_rl"], float)
    reward_mpc = np.asarray(compare_bundle["avg_rewards_mpc"], float)
    reward_delta = reward_rl - reward_mpc

    (
        mae_rl,
        mae_mpc,
        rmse_rl,
        rmse_mpc,
        last_err_rl,
        last_err_mpc,
        move_rl,
        move_mpc,
    ) = episode_metrics(rl_bundle, baseline_bundle)

    y_sp_phys = reconstruct_setpoint_phys(rl_bundle).reshape(n_episodes, steps, -1)
    y_rl = np.asarray(rl_bundle["y"][1:], float).reshape(n_episodes, steps, -1)
    y_mpc = np.asarray(baseline_bundle["y"][1:], float).reshape(n_episodes, steps, -1)

    block_slices = {
        "sp1": slice(0, steps // 2),
        "sp2": slice(steps // 2, steps),
    }
    block_metrics = {}
    for name, slc in block_slices.items():
        block_mae_rl = np.mean(np.abs(y_rl[:, slc, :] - y_sp_phys[:, slc, :]), axis=1)
        block_mae_mpc = np.mean(np.abs(y_mpc[:, slc, :] - y_sp_phys[:, slc, :]), axis=1)
        block_metrics[name] = {
            "rl": block_mae_rl,
            "mpc": block_mae_mpc,
            "delta": block_mae_rl - block_mae_mpc,
        }

    source_log = np.asarray(rl_bundle["rl_action_source_log"], int).reshape(n_episodes, steps)
    accepted_log = np.asarray(rl_bundle["accepted_log"], int).reshape(n_episodes, steps)
    executed_z = np.asarray(rl_bundle["z_executed_log"], float).reshape(n_episodes, steps, -1)
    prediction_score = np.asarray(rl_bundle["s_pred_log"], float).reshape(n_episodes, steps)
    gain_drift = np.asarray(rl_bundle["gain_drift_log"], float).reshape(n_episodes, steps)

    td3_mask = source_log == 2
    ls_mask = (source_log == 3) | (source_log == 5)
    nominal_mask = (source_log == 0) | (source_log == 4)
    executed_z_norm = np.linalg.norm(executed_z, axis=2)

    post_live = slice(warm_start_episodes, None)
    tail20 = slice(-20, None)
    tail10 = slice(-10, None)

    def phase_row(label: str, indexer: slice) -> dict:
        reward_phase = reward_delta[indexer]
        mae_delta = mae_rl[indexer] - mae_mpc[indexer]
        rmse_delta = rmse_rl[indexer] - rmse_mpc[indexer]
        move_delta = move_rl[indexer] - move_mpc[indexer]
        return {
            "label": label,
            "reward_rl": float(np.mean(reward_rl[indexer])),
            "reward_mpc": float(np.mean(reward_mpc[indexer])),
            "reward_delta": float(np.mean(reward_phase)),
            "output1_mae_rl": float(np.mean(mae_rl[indexer, 0])),
            "output1_mae_mpc": float(np.mean(mae_mpc[indexer, 0])),
            "output1_mae_delta": float(np.mean(mae_delta[:, 0])),
            "output2_mae_rl": float(np.mean(mae_rl[indexer, 1])),
            "output2_mae_mpc": float(np.mean(mae_mpc[indexer, 1])),
            "output2_mae_delta": float(np.mean(mae_delta[:, 1])),
            "output1_rmse_delta": float(np.mean(rmse_delta[:, 0])),
            "output2_rmse_delta": float(np.mean(rmse_delta[:, 1])),
            "move_rl": float(np.mean(move_rl[indexer])),
            "move_mpc": float(np.mean(move_mpc[indexer])),
            "move_delta": float(np.mean(move_delta)),
            "move_ratio": float(np.mean(move_rl[indexer] / np.maximum(move_mpc[indexer], 1.0e-12))),
        }

    phases = {
        "post_live": phase_row("post_live", post_live),
        "tail20": phase_row("tail20", tail20),
        "tail10": phase_row("tail10", tail10),
    }

    ci = {
        "post_live_reward_delta": bootstrap_ci(reward_delta[post_live]),
        "post_live_output1_mae_delta": bootstrap_ci((mae_rl[post_live, 0] - mae_mpc[post_live, 0])),
        "post_live_output2_mae_delta": bootstrap_ci((mae_rl[post_live, 1] - mae_mpc[post_live, 1])),
        "post_live_move_delta": bootstrap_ci(move_rl[post_live] - move_mpc[post_live]),
    }

    final_episode = {
        "reward_delta": float(reward_delta[-1]),
        "output_diff_max_abs": np.max(
            np.abs(np.asarray(rl_bundle["y"][1:], float).reshape(n_episodes, steps, -1)[-1] - y_mpc[-1]),
            axis=0,
        ).tolist(),
        "output_diff_mean_abs": np.mean(
            np.abs(np.asarray(rl_bundle["y"][1:], float).reshape(n_episodes, steps, -1)[-1] - y_mpc[-1]),
            axis=0,
        ).tolist(),
        "input_diff_max_abs": np.max(
            np.abs(np.asarray(rl_bundle["u"], float).reshape(n_episodes, steps, -1)[-1] - np.asarray(baseline_bundle["u"], float).reshape(n_episodes, steps, -1)[-1]),
            axis=0,
        ).tolist(),
        "input_diff_mean_abs": np.mean(
            np.abs(np.asarray(rl_bundle["u"], float).reshape(n_episodes, steps, -1)[-1] - np.asarray(baseline_bundle["u"], float).reshape(n_episodes, steps, -1)[-1]),
            axis=0,
        ).tolist(),
        "last_err_delta": np.mean(last_err_rl[-20:] - last_err_mpc[-20:], axis=0).tolist(),
    }

    return {
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "compare_bundle": str(COMPARE_BUNDLE.relative_to(REPO_ROOT)),
        "baseline_bundle": str(BASELINE_BUNDLE.relative_to(REPO_ROOT)),
        "n_episodes": n_episodes,
        "steps_per_episode": steps,
        "warm_start_episodes": warm_start_episodes,
        "reward_delta": reward_delta.tolist(),
        "mae_delta": (mae_rl - mae_mpc).tolist(),
        "rmse_delta": (rmse_rl - rmse_mpc).tolist(),
        "phases": phases,
        "ci": ci,
        "block_tail20": {
            name: {
                "output1_rl": float(np.mean(values["rl"][-20:, 0])),
                "output1_mpc": float(np.mean(values["mpc"][-20:, 0])),
                "output1_delta": float(np.mean(values["delta"][-20:, 0])),
                "output2_rl": float(np.mean(values["rl"][-20:, 1])),
                "output2_mpc": float(np.mean(values["mpc"][-20:, 1])),
                "output2_delta": float(np.mean(values["delta"][-20:, 1])),
            }
            for name, values in block_metrics.items()
        },
        "action_sources": {
            "post_live_accepted_fraction": float(np.mean(accepted_log[post_live])),
            "post_live_td3_fraction": float(np.mean(td3_mask[post_live])),
            "post_live_ls_fraction": float(np.mean(ls_mask[post_live])),
            "post_live_nominal_fraction": float(np.mean(nominal_mask[post_live])),
            "tail20_accepted_fraction": float(np.mean(accepted_log[tail20])),
            "tail20_td3_fraction": float(np.mean(td3_mask[tail20])),
            "tail20_ls_fraction": float(np.mean(ls_mask[tail20])),
            "tail20_nominal_fraction": float(np.mean(nominal_mask[tail20])),
            "post_live_mean_executed_z_norm": float(np.nanmean(executed_z_norm[post_live])),
            "tail20_mean_executed_z_norm": float(np.nanmean(executed_z_norm[tail20])),
            "post_live_mean_accepted_z_norm": float(np.nanmean(executed_z_norm[post_live][accepted_log[post_live].astype(bool)])),
            "tail20_mean_accepted_z_norm": float(np.nanmean(executed_z_norm[tail20][accepted_log[tail20].astype(bool)])),
            "post_live_mean_prediction_score": float(np.nanmean(prediction_score[post_live])),
            "tail20_mean_prediction_score": float(np.nanmean(prediction_score[tail20])),
            "post_live_mean_gain_drift": float(np.nanmean(gain_drift[post_live])),
            "tail20_mean_gain_drift": float(np.nanmean(gain_drift[tail20])),
        },
        "final_episode": final_episode,
    }


def plot_reward_and_error_deltas(summary: dict, rl_bundle: dict, baseline_bundle: dict, compare_bundle: dict) -> str:
    n_episodes = summary["n_episodes"]
    _steps = summary["steps_per_episode"]
    warm_start_episodes = summary["warm_start_episodes"]

    reward_delta = np.asarray(summary["reward_delta"], float)
    mae_delta = np.asarray(summary["mae_delta"], float)
    episodes = np.arange(1, n_episodes + 1)

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 11.0), sharex=True, constrained_layout=True)
    panels = [
        (reward_delta, "Reward delta vs disturbance MPC", "Reward delta"),
        (mae_delta[:, 0], "Tray-24 ethane composition MAE delta", "Delta (-)"),
        (mae_delta[:, 1], "Tray-85 temperature MAE delta", "Delta (K)"),
    ]
    for ax, (values, title, ylabel) in zip(axes, panels):
        ax.plot(episodes, values, color="#94a3b8", linewidth=1.0, alpha=0.55, label="Episode value")
        ax.plot(
            episodes,
            moving_average(values, MOVING_AVERAGE),
            color="#0f172a",
            linewidth=2.0,
            label=f"{MOVING_AVERAGE}-episode moving average",
        )
        ax.axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--")
        ax.axvline(warm_start_episodes + 0.5, color="#dc2626", linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Sub-episode")

    out_path = OUT_DIR / "fig_reward_and_output_error_deltas.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def plot_final_episode_tracking(summary: dict, rl_bundle: dict, baseline_bundle: dict) -> str:
    n_episodes = summary["n_episodes"]
    steps = summary["steps_per_episode"]
    metadata = rl_bundle["system_metadata"]
    y_sp_phys = reconstruct_setpoint_phys(rl_bundle).reshape(n_episodes, steps, -1)
    y_rl = np.asarray(rl_bundle["y"][1:], float).reshape(n_episodes, steps, -1)
    y_mpc = np.asarray(baseline_bundle["y"][1:], float).reshape(n_episodes, steps, -1)
    u_rl = np.asarray(rl_bundle["u"], float).reshape(n_episodes, steps, -1)
    u_mpc = np.asarray(baseline_bundle["u"], float).reshape(n_episodes, steps, -1)
    xs = np.arange(steps)
    switch_step = steps // 2

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]

    for idx, ax in enumerate(axes[0]):
        ax.step(xs, y_sp_phys[-1, :, idx], where="post", color="#111827", linewidth=1.5, linestyle="--", label="Setpoint")
        ax.plot(xs, y_mpc[-1, :, idx], color="#7c3aed", linewidth=1.8, label="Disturbance MPC")
        ax.plot(xs, y_rl[-1, :, idx], color="#059669", linewidth=1.8, label="Markov RL")
        ax.axvline(switch_step, color="#94a3b8", linewidth=1.0, linestyle=":")
        ax.set_title(output_labels[idx])
        ax.set_ylabel("Output")
        ax.grid(alpha=0.25, linewidth=0.6)

    for idx, ax in enumerate(axes[1]):
        ax.step(xs, u_mpc[-1, :, idx], where="post", color="#7c3aed", linewidth=1.8, label="Disturbance MPC")
        ax.step(xs, u_rl[-1, :, idx], where="post", color="#059669", linewidth=1.8, label="Markov RL")
        ax.axvline(switch_step, color="#94a3b8", linewidth=1.0, linestyle=":")
        ax.set_title(input_labels[idx])
        ax.set_ylabel("Input")
        ax.set_xlabel("Step within final sub-episode")
        ax.grid(alpha=0.25, linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    out_path = OUT_DIR / "fig_final_episode_tracking.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def plot_action_source_trends(summary: dict, rl_bundle: dict) -> str:
    n_episodes = summary["n_episodes"]
    steps = summary["steps_per_episode"]
    warm_start_episodes = summary["warm_start_episodes"]

    source_log = np.asarray(rl_bundle["rl_action_source_log"], int).reshape(n_episodes, steps)
    accepted_log = np.asarray(rl_bundle["accepted_log"], int).reshape(n_episodes, steps)
    executed_z = np.asarray(rl_bundle["z_executed_log"], float).reshape(n_episodes, steps, -1)
    prediction_score = np.asarray(rl_bundle["s_pred_log"], float).reshape(n_episodes, steps)
    gain_drift = np.asarray(rl_bundle["gain_drift_log"], float).reshape(n_episodes, steps)
    z_bound = float(rl_bundle.get("markov_z_bound", 0.05))
    drift_limit = float(rl_bundle.get("markov_gain_drift_max", 0.10))

    td3_mask = source_log == 2
    ls_mask = (source_log == 3) | (source_log == 5)
    nominal_mask = (source_log == 0) | (source_log == 4)
    executed_z_norm = np.linalg.norm(executed_z, axis=2)

    def window_means(values: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for start in range(warm_start_episodes, n_episodes, WINDOW):
            stop = min(start + WINDOW, n_episodes)
            xs.append(start + 1 + 0.5 * (stop - start - 1))
            chunk = values[start:stop]
            if mask is None:
                ys.append(float(np.nanmean(chunk)))
            else:
                chunk_mask = mask[start:stop]
                ys.append(float(np.nanmean(chunk[chunk_mask])) if np.any(chunk_mask) else np.nan)
        return np.asarray(xs, float), np.asarray(ys, float)

    xs, td3_frac = window_means(td3_mask.astype(float))
    _xs, ls_frac = window_means(ls_mask.astype(float))
    _xs, nominal_frac = window_means(nominal_mask.astype(float))
    _xs, accepted_frac = window_means(accepted_log.astype(float))
    _xs, z_all = window_means(executed_z_norm / max(z_bound, 1.0e-12))
    _xs, z_accepted = window_means(executed_z_norm / max(z_bound, 1.0e-12), mask=accepted_log.astype(bool))
    _xs, drift_ratio = window_means(gain_drift / max(drift_limit, 1.0e-12))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    axes[0].plot(xs, td3_frac, marker="o", linewidth=2.0, color=SOURCE_COLORS["td3_accepted"], label="TD3 accepted")
    axes[0].plot(xs, ls_frac, marker="s", linewidth=2.0, color=SOURCE_COLORS["ls_fallback"], label="LS fallback")
    axes[0].plot(xs, nominal_frac, marker="^", linewidth=2.0, color=SOURCE_COLORS["nominal"], label="Nominal fallback")
    axes[0].set_title("20-episode action-source fractions")
    axes[0].set_xlabel("Sub-episode")
    axes[0].set_ylabel("Fraction of steps")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(loc="best")

    axes[1].plot(xs, accepted_frac, marker="o", linewidth=2.0, color="#2563eb", label="Accepted fraction")
    axes[1].plot(xs, z_all, marker="s", linewidth=2.0, color="#16a34a", label="Mean executed z norm / z_bound")
    axes[1].plot(xs, z_accepted, marker="^", linewidth=2.0, color="#ca8a04", label="Accepted-step z norm / z_bound")
    axes[1].plot(xs, drift_ratio, marker="d", linewidth=2.0, color="#dc2626", label="Mean gain drift / limit")
    axes[1].set_title("Correction magnitude stays conservative")
    axes[1].set_xlabel("Sub-episode")
    axes[1].set_ylabel("Fraction of configured limit")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(loc="best")

    out_path = OUT_DIR / "fig_action_source_and_correction_limits.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def write_summary_files(summary: dict) -> None:
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    rows = []
    for phase_name, phase in summary["phases"].items():
        rows.extend(
            [
                {
                    "phase": phase_name,
                    "metric": "reward_mean",
                    "markov_rl": phase["reward_rl"],
                    "mpc": phase["reward_mpc"],
                    "delta": phase["reward_delta"],
                    "units": "reward units",
                },
                {
                    "phase": phase_name,
                    "metric": "output1_mae",
                    "markov_rl": phase["output1_mae_rl"],
                    "mpc": phase["output1_mae_mpc"],
                    "delta": phase["output1_mae_delta"],
                    "units": "composition fraction",
                },
                {
                    "phase": phase_name,
                    "metric": "output2_mae",
                    "markov_rl": phase["output2_mae_rl"],
                    "mpc": phase["output2_mae_mpc"],
                    "delta": phase["output2_mae_delta"],
                    "units": "K",
                },
                {
                    "phase": phase_name,
                    "metric": "normalized_input_move",
                    "markov_rl": phase["move_rl"],
                    "mpc": phase["move_mpc"],
                    "delta": phase["move_delta"],
                    "units": "mean normalized step 2-norm",
                },
            ]
        )

    for block_name, values in summary["block_tail20"].items():
        rows.extend(
            [
                {
                    "phase": "tail20",
                    "metric": f"{block_name}_output1_mae",
                    "markov_rl": values["output1_rl"],
                    "mpc": values["output1_mpc"],
                    "delta": values["output1_delta"],
                    "units": "composition fraction",
                },
                {
                    "phase": "tail20",
                    "metric": f"{block_name}_output2_mae",
                    "markov_rl": values["output2_rl"],
                    "mpc": values["output2_mpc"],
                    "delta": values["output2_delta"],
                    "units": "K",
                },
            ]
        )

    csv_path = OUT_DIR / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "metric", "markov_rl", "mpc", "delta", "units"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rl_bundle = load_pickle(RUN_BUNDLE)
    baseline_bundle = load_pickle(BASELINE_BUNDLE)
    compare_bundle = load_pickle(COMPARE_BUNDLE)

    summary = compute_summary(rl_bundle, baseline_bundle, compare_bundle)
    summary["figures"] = {
        "reward_and_output_error_deltas": plot_reward_and_error_deltas(summary, rl_bundle, baseline_bundle, compare_bundle),
        "final_episode_tracking": plot_final_episode_tracking(summary, rl_bundle, baseline_bundle),
        "action_source_and_correction_limits": plot_action_source_trends(summary, rl_bundle),
    }
    write_summary_files(summary)

    print(json.dumps(
        {
            "out_dir": str(OUT_DIR.relative_to(REPO_ROOT)),
            "run_dir": summary["run_dir"],
            "post_live_reward_delta": summary["phases"]["post_live"]["reward_delta"],
            "tail20_reward_delta": summary["phases"]["tail20"]["reward_delta"],
            "post_live_output1_mae_delta": summary["phases"]["post_live"]["output1_mae_delta"],
            "post_live_output2_mae_delta": summary["phases"]["post_live"]["output2_mae_delta"],
            "post_live_move_delta": summary["phases"]["post_live"]["move_delta"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
