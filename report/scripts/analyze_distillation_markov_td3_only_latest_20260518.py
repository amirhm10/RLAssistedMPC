from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_markov_td3_only_latest_20260518"

LATEST_RUN = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified"
    / "20260518_091937"
)
PRIOR_RUN = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified"
    / "20260516_200353"
)
LATEST_COMPARE = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_markov_td3_disturb_fluctuation_td3_only_no_safeguard"
    / "20260518_091949"
)
BASELINE_BUNDLE = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"

TAIL_EPISODES = 20
ROLLING_WINDOW = 10


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def moving_average(values: np.ndarray, width: int = ROLLING_WINDOW) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < width:
        return arr.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def n_inputs(bundle: dict) -> int:
    return int(len(np.asarray(bundle["steady_states"]["ss_inputs"], dtype=float)))


def episode_shape(bundle: dict) -> tuple[int, int]:
    steps = int(bundle["time_in_sub_episodes"])
    n_episodes = int(len(np.asarray(bundle["avg_rewards"], dtype=float)))
    return n_episodes, steps


def output_arrays(bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_episodes, steps = episode_shape(bundle)
    n_out = int(len(np.asarray(bundle["steady_states"]["y_ss"], dtype=float)))
    n_in = n_inputs(bundle)
    y = np.asarray(bundle["y"][1:], dtype=float).reshape(n_episodes, steps, n_out)
    u = np.asarray(bundle["u"], dtype=float).reshape(n_episodes, steps, n_in)
    y_sp = y_sp_phys(bundle).reshape(n_episodes, steps, n_out)
    return y, u, y_sp


def baseline_arrays(reference_bundle: dict, baseline_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    n_episodes, steps = episode_shape(reference_bundle)
    n_out = int(len(np.asarray(reference_bundle["steady_states"]["y_ss"], dtype=float)))
    n_in = n_inputs(reference_bundle)
    y = np.asarray(baseline_bundle["y"][1:], dtype=float).reshape(n_episodes, steps, n_out)
    u = np.asarray(baseline_bundle["u"], dtype=float).reshape(n_episodes, steps, n_in)
    return y, u


def y_sp_phys(bundle: dict) -> np.ndarray:
    n_in = n_inputs(bundle)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    y_range = data_max[n_in:] - data_min[n_in:]
    return y_ss.reshape(1, -1) + np.asarray(bundle["y_sp"], dtype=float) * y_range.reshape(1, -1)


def reward_band_phys(bundle: dict, reward_params: dict) -> np.ndarray:
    y_sp = y_sp_phys(bundle)
    n_episodes, steps = episode_shape(bundle)
    k_rel = np.asarray(reward_params["k_rel"], dtype=float).reshape(1, 1, -1)
    band_floor = np.asarray(reward_params["band_floor_phys"], dtype=float).reshape(1, 1, -1)
    return np.maximum(k_rel * np.abs(y_sp.reshape(n_episodes, steps, -1)), band_floor)


def score_trajectory(bundle: dict, reward_params: dict) -> np.ndarray:
    y, u, y_sp = output_arrays(bundle)
    n_ep, steps, n_out = y.shape
    n_in = n_inputs(bundle)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    dy = data_max[n_in:] - data_min[n_in:]
    du_span = data_max[:n_in] - data_min[:n_in]

    q_diag = np.asarray(reward_params["Q_diag"], dtype=float).reshape(1, 1, n_out)
    r_diag = np.asarray(reward_params["R_diag"], dtype=float).reshape(1, 1, n_in)
    k_rel = np.asarray(reward_params["k_rel"], dtype=float).reshape(1, 1, n_out)
    band_floor = np.asarray(reward_params["band_floor_phys"], dtype=float).reshape(1, 1, n_out)
    tau_frac = float(reward_params["tau_frac"])
    gamma_out = float(reward_params["gamma_out"])
    gamma_in = float(reward_params["gamma_in"])
    beta = float(reward_params["beta"])
    gate = str(reward_params["gate"])
    bonus_kind = str(reward_params["bonus_kind"])
    bonus_k = float(reward_params["bonus_k"])
    bonus_p = float(reward_params["bonus_p"])
    bonus_c = float(reward_params["bonus_c"])
    reward_scale = float(reward_params["reward_scale"])

    e_scaled = (y - y_sp) / dy.reshape(1, 1, n_out)
    du = np.diff(np.concatenate([u[:, 0:1, :], u], axis=1), axis=1)
    du_scaled = du / du_span.reshape(1, 1, n_in)
    band_phys = np.maximum(k_rel * np.abs(y_sp), band_floor)
    band_scaled = band_phys / dy.reshape(1, 1, n_out)
    tau_scaled = tau_frac * band_scaled
    abs_e = np.abs(e_scaled)
    sig_arg = (band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12)
    s_i = 1.0 / (1.0 + np.exp(-np.clip(sig_arg, -60.0, 60.0)))

    if gate == "prod":
        w_in = np.prod(s_i, axis=2)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=2)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=2) ** (1.0 / float(s_i.shape[2]))
    else:
        raise ValueError(f"Unsupported reward gate: {gate}")

    z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
    if bonus_kind == "linear":
        phi = 1.0 - z
    elif bonus_kind == "quadratic":
        phi = (1.0 - z) ** 2
    elif bonus_kind == "exp":
        phi = (np.exp(-bonus_k * z) - np.exp(-bonus_k)) / (1.0 - np.exp(-bonus_k))
    elif bonus_kind == "power":
        phi = 1.0 - np.power(z, bonus_p)
    elif bonus_kind == "log":
        phi = np.log1p(bonus_c * (1.0 - z)) / np.log1p(bonus_c)
    else:
        raise ValueError(f"Unsupported bonus kind: {bonus_kind}")

    err_quad = np.sum(q_diag * (e_scaled**2), axis=2)
    move = np.sum(r_diag * (du_scaled**2), axis=2)
    slope_at_edge = 2.0 * q_diag * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    lin_out = np.sum((1.0 - w_in).reshape(n_ep, steps, 1) * gamma_out * slope_at_edge * overflow, axis=2)
    lin_in = np.sum(w_in.reshape(n_ep, steps, 1) * gamma_in * slope_at_edge * inside_mag, axis=2)
    qb2 = q_diag * (band_scaled**2)
    bonus = np.sum(w_in.reshape(n_ep, steps, 1) * beta * qb2 * phi, axis=2)
    return (-(err_quad + move + lin_out + lin_in) + bonus) * reward_scale


def block_slices(steps: int) -> dict[str, slice]:
    return {"SP1": slice(0, steps // 2), "SP2": slice(steps // 2, steps)}


def mae_by_block(y: np.ndarray, y_sp: np.ndarray, episodes: slice) -> dict[str, dict[str, float]]:
    _, steps, _ = y.shape
    err = np.abs(y - y_sp)
    out: dict[str, dict[str, float]] = {}
    for block, slc in block_slices(steps).items():
        out[block] = {
            "comp_mae": float(np.mean(err[episodes, slc, 0])),
            "temp_mae": float(np.mean(err[episodes, slc, 1])),
            "comp_max": float(np.max(err[episodes, slc, 0])),
            "temp_max": float(np.max(err[episodes, slc, 1])),
        }
    return out


def inside_band_by_block(y: np.ndarray, y_sp: np.ndarray, band: np.ndarray, episodes: slice) -> dict[str, dict[str, float]]:
    _, steps, _ = y.shape
    err = np.abs(y - y_sp)
    out: dict[str, dict[str, float]] = {}
    for block, slc in block_slices(steps).items():
        out[block] = {
            "comp_inside": float(np.mean(err[episodes, slc, 0] <= band[episodes, slc, 0])),
            "temp_inside": float(np.mean(err[episodes, slc, 1] <= band[episodes, slc, 1])),
        }
    return out


def diagnostic_summary(stage_df: pd.DataFrame, steps: int, tail_episodes: int = TAIL_EPISODES) -> dict[str, float]:
    tail = stage_df.iloc[-tail_episodes * steps :]
    return {
        "td3_fraction": float(np.mean(tail["action_source_name"].astype(str) == "td3_accepted")),
        "executed_prediction_score_mean": float(np.nanmean(tail["executed_prediction_score"])),
        "executed_prediction_score_p05": float(np.nanpercentile(tail["executed_prediction_score"], 5.0)),
        "executed_gain_drift_mean": float(np.nanmean(tail["executed_gain_drift"])),
        "executed_gain_drift_p95": float(np.nanpercentile(tail["executed_gain_drift"], 95.0)),
        "executed_cost_margin_mean": float(np.nanmean(tail["executed_cost_margin"])),
        "executed_cost_margin_p95": float(np.nanpercentile(tail["executed_cost_margin"], 95.0)),
        "executed_z_norm_mean": float(np.nanmean(tail["executed_z_norm"])),
        "executed_z_norm_p95": float(np.nanpercentile(tail["executed_z_norm"], 95.0)),
    }


def summarize(
    latest: dict,
    prior: dict,
    baseline: dict,
    compare: dict,
    latest_stage: pd.DataFrame,
    prior_stage: pd.DataFrame,
) -> dict:
    latest_params = latest["reward_params"]
    latest_rewards_native = np.asarray(latest["avg_rewards"], dtype=float)
    prior_rewards_native = np.asarray(prior["avg_rewards"], dtype=float)
    latest_rescored = np.mean(score_trajectory(latest, latest_params), axis=1)
    prior_rescored = np.mean(score_trajectory(prior, latest_params), axis=1)
    mpc_rewards_latest = np.asarray(compare["avg_rewards_mpc"], dtype=float)

    latest_y, _latest_u, latest_sp = output_arrays(latest)
    prior_y, _prior_u, prior_sp = output_arrays(prior)
    baseline_y, _baseline_u = baseline_arrays(latest, baseline)
    band = reward_band_phys(latest, latest_params)
    prior_band = reward_band_phys(prior, latest_params)
    n_ep, steps = episode_shape(latest)
    tail = slice(max(0, n_ep - TAIL_EPISODES), n_ep)
    final = slice(n_ep - 1, n_ep)

    latest_mpc_final = mae_by_block(baseline_y, latest_sp, final)
    latest_mpc_tail = mae_by_block(baseline_y, latest_sp, tail)

    return {
        "paths": {
            "latest_run": str(LATEST_RUN.relative_to(REPO_ROOT)),
            "prior_run": str(PRIOR_RUN.relative_to(REPO_ROOT)),
            "latest_compare": str(LATEST_COMPARE.relative_to(REPO_ROOT)),
            "baseline_bundle": str(BASELINE_BUNDLE.relative_to(REPO_ROOT)),
        },
        "reward_params_latest": {
            "Q_diag": np.asarray(latest_params["Q_diag"], dtype=float).tolist(),
            "R_diag": np.asarray(latest_params["R_diag"], dtype=float).tolist(),
            "k_rel": np.asarray(latest_params["k_rel"], dtype=float).tolist(),
            "band_floor_phys": np.asarray(latest_params["band_floor_phys"], dtype=float).tolist(),
        },
        "reward_native": {
            "latest_mean": float(np.mean(latest_rewards_native)),
            "latest_tail20": float(np.mean(latest_rewards_native[tail])),
            "latest_final": float(latest_rewards_native[-1]),
            "latest_best": float(np.max(latest_rewards_native)),
            "latest_best_episode": int(np.argmax(latest_rewards_native) + 1),
            "prior_mean": float(np.mean(prior_rewards_native)),
            "prior_tail20": float(np.mean(prior_rewards_native[tail])),
            "prior_final": float(prior_rewards_native[-1]),
            "mpc_latest_reward_tail20": float(np.mean(mpc_rewards_latest[tail])),
            "mpc_latest_reward_final": float(mpc_rewards_latest[-1]),
        },
        "reward_rescored_may18_geometry": {
            "latest_tail20": float(np.mean(latest_rescored[tail])),
            "prior_tail20": float(np.mean(prior_rescored[tail])),
            "latest_final": float(latest_rescored[-1]),
            "prior_final": float(prior_rescored[-1]),
            "latest_minus_prior_tail20": float(np.mean(latest_rescored[tail] - prior_rescored[tail])),
            "latest_minus_mpc_tail20": float(np.mean(latest_rescored[tail] - mpc_rewards_latest[tail])),
        },
        "tracking": {
            "latest_final": mae_by_block(latest_y, latest_sp, final),
            "latest_tail20": mae_by_block(latest_y, latest_sp, tail),
            "prior_final": mae_by_block(prior_y, prior_sp, final),
            "prior_tail20": mae_by_block(prior_y, prior_sp, tail),
            "mpc_final": latest_mpc_final,
            "mpc_tail20": latest_mpc_tail,
            "latest_inside_final": inside_band_by_block(latest_y, latest_sp, band, final),
            "latest_inside_tail20": inside_band_by_block(latest_y, latest_sp, band, tail),
            "prior_inside_final_rescored": inside_band_by_block(prior_y, prior_sp, prior_band, final),
            "prior_inside_tail20_rescored": inside_band_by_block(prior_y, prior_sp, prior_band, tail),
            "mpc_inside_final": inside_band_by_block(baseline_y, latest_sp, band, final),
            "mpc_inside_tail20": inside_band_by_block(baseline_y, latest_sp, band, tail),
        },
        "diagnostics_tail20": {
            "latest": diagnostic_summary(latest_stage, steps),
            "prior": diagnostic_summary(prior_stage, steps),
        },
        "shock": {
            "latest_min_reward_first20": float(np.min(latest_rewards_native[:20])),
            "latest_min_reward_first20_episode": int(np.argmin(latest_rewards_native[:20]) + 1),
            "latest_negative_episodes_first20": [int(i + 1) for i, val in enumerate(latest_rewards_native[:20]) if val < 0.0],
            "prior_min_reward_first20": float(np.min(prior_rewards_native[:20])),
            "prior_min_reward_first20_episode": int(np.argmin(prior_rewards_native[:20]) + 1),
        },
    }


def plot_rewards(latest: dict, prior: dict, compare: dict, summary: dict) -> Path:
    latest_params = latest["reward_params"]
    episodes = np.arange(1, len(latest["avg_rewards"]) + 1)
    latest_native = np.asarray(latest["avg_rewards"], dtype=float)
    prior_native = np.asarray(prior["avg_rewards"], dtype=float)
    latest_rescored = np.mean(score_trajectory(latest, latest_params), axis=1)
    prior_rescored = np.mean(score_trajectory(prior, latest_params), axis=1)
    mpc_latest = np.asarray(compare["avg_rewards_mpc"], dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True)
    axs[0].plot(episodes, moving_average(prior_native), color="#7C3AED", linewidth=2.0, linestyle="--", label="May 16 TD3-only native")
    axs[0].plot(episodes, moving_average(latest_native), color="#0B6E4F", linewidth=2.4, label="May 18 TD3-only native")
    axs[0].plot(episodes, moving_average(mpc_latest), color="#4C4C4C", linewidth=2.0, linestyle=":", label="MPC under May 18 reward")
    axs[0].axhline(0.0, color="0.3", linewidth=0.8)
    axs[0].set_title("Native reward traces")
    axs[0].set_ylabel("Average reward")

    axs[1].plot(episodes, moving_average(prior_rescored), color="#7C3AED", linewidth=2.0, linestyle="--", label="May 16 trajectory rescored")
    axs[1].plot(episodes, moving_average(latest_rescored), color="#0B6E4F", linewidth=2.4, label="May 18 trajectory")
    axs[1].plot(episodes, moving_average(mpc_latest), color="#4C4C4C", linewidth=2.0, linestyle=":", label="MPC")
    axs[1].axhline(0.0, color="0.3", linewidth=0.8)
    axs[1].set_title("Both TD3 trajectories scored with May 18 reward geometry")

    for ax in axs:
        ax.axvline(10.5, color="0.55", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Sub-episode")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        "Distillation TD3-only: the May 18 reward retuning produces a stronger late controller but still has early shock episodes",
        y=1.02,
    )
    out = OUT_DIR / "fig_latest_reward_vs_prior.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_final_tracking(latest: dict, prior: dict, baseline: dict) -> Path:
    latest_y, latest_u, latest_sp = output_arrays(latest)
    prior_y, prior_u, _prior_sp = output_arrays(prior)
    baseline_y, baseline_u = baseline_arrays(latest, baseline)
    steps = latest_y.shape[1]
    t = np.arange(steps)
    output_labels = list(latest["system_metadata"]["output_labels"])
    input_labels = list(latest["system_metadata"]["input_labels"])

    fig, axs = plt.subplots(2, 2, figsize=(13.8, 8.6), sharex="col")
    for idx, ax in enumerate(axs[0]):
        ax.step(t, latest_sp[-1, :, idx], where="post", color="#111111", linewidth=1.4, linestyle="--", label="Setpoint")
        ax.plot(t, baseline_y[-1, :, idx], color="#4C4C4C", linewidth=1.8, linestyle=":", label="MPC")
        ax.plot(t, prior_y[-1, :, idx], color="#7C3AED", linewidth=1.7, linestyle="--", label="May 16 TD3-only")
        ax.plot(t, latest_y[-1, :, idx], color="#0B6E4F", linewidth=2.1, label="May 18 TD3-only")
        ax.axvline(steps // 2, color="0.55", linewidth=1.0, linestyle=":")
        ax.set_title(output_labels[idx])
        ax.set_ylabel("Output")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="best", fontsize=8.5)

    for idx, ax in enumerate(axs[1]):
        ax.step(t, baseline_u[-1, :, idx], where="post", color="#4C4C4C", linewidth=1.8, linestyle=":", label="MPC")
        ax.step(t, prior_u[-1, :, idx], where="post", color="#7C3AED", linewidth=1.7, linestyle="--", label="May 16 TD3-only")
        ax.step(t, latest_u[-1, :, idx], where="post", color="#0B6E4F", linewidth=2.0, label="May 18 TD3-only")
        ax.axvline(steps // 2, color="0.55", linewidth=1.0, linestyle=":")
        ax.set_title(input_labels[idx])
        ax.set_ylabel("Input")
        ax.set_xlabel("Step within final sub-episode")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Final episode tracking: May 18 TD3-only is much cleaner, especially in the second setpoint block", y=1.02)
    out = OUT_DIR / "fig_latest_final_episode_tracking.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_block_metrics(summary: dict) -> Path:
    groups = ["MPC", "May 16 TD3-only", "May 18 TD3-only"]
    colors = ["#4C4C4C", "#7C3AED", "#0B6E4F"]
    x = np.arange(len(groups))

    def vals(metric_group: str, block: str, metric: str) -> list[float]:
        return [
            summary["tracking"]["mpc_tail20"][block][metric],
            summary["tracking"]["prior_tail20"][block][metric],
            summary["tracking"]["latest_tail20"][block][metric],
        ]

    def inside(block: str, metric: str) -> list[float]:
        return [
            summary["tracking"]["mpc_inside_tail20"][block][metric],
            summary["tracking"]["prior_inside_tail20_rescored"][block][metric],
            summary["tracking"]["latest_inside_tail20"][block][metric],
        ]

    fig, axs = plt.subplots(2, 2, figsize=(13.6, 8.8))
    axs = axs.ravel()
    specs = [
        ("SP1", "temp_mae", "Tail-20 SP1 temperature MAE", "K"),
        ("SP2", "temp_mae", "Tail-20 SP2 temperature MAE", "K"),
        ("SP1", "comp_mae", "Tail-20 SP1 composition MAE", "fraction"),
        ("SP2", "comp_mae", "Tail-20 SP2 composition MAE", "fraction"),
    ]
    for ax, (block, metric, title, ylabel) in zip(axs, specs):
        ax.bar(x, vals("tail20", block, metric), color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=12)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Tail-20 tracking: retuned TD3-only improves the old TD3 trajectory but does not remove every tradeoff", y=1.02)
    out = OUT_DIR / "fig_latest_tail20_block_tracking.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    for ax, block in zip(axs, ["SP1", "SP2"]):
        width = 0.34
        ax.bar(x - width / 2.0, inside(block, "comp_inside"), width, color="#1976D2", label="composition")
        ax.bar(x + width / 2.0, inside(block, "temp_inside"), width, color="#D97706", label="temperature")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=12)
        ax.set_ylim(0.0, 1.03)
        ax.set_title(f"{block} fraction inside May 18 reward band")
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=9)
    axs[0].set_ylabel("Fraction of tail-20 steps")
    out2 = OUT_DIR / "fig_latest_tail20_inside_band.png"
    fig.tight_layout()
    fig.savefig(out2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_diagnostics(summary: dict) -> Path:
    latest = summary["diagnostics_tail20"]["latest"]
    prior = summary["diagnostics_tail20"]["prior"]
    labels = ["pred. score", "gain drift", "cost margin", "z norm"]
    latest_vals = [
        latest["executed_prediction_score_mean"],
        latest["executed_gain_drift_mean"],
        latest["executed_cost_margin_mean"],
        latest["executed_z_norm_mean"],
    ]
    prior_vals = [
        prior["executed_prediction_score_mean"],
        prior["executed_gain_drift_mean"],
        prior["executed_cost_margin_mean"],
        prior["executed_z_norm_mean"],
    ]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    ax.bar(x - width / 2.0, prior_vals, width, color="#7C3AED", label="May 16 TD3-only")
    ax.bar(x + width / 2.0, latest_vals, width, color="#0B6E4F", label="May 18 TD3-only")
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Tail-20 Markov diagnostics: May 18 is less aggressive and less model-negative")
    ax.set_ylabel("Mean diagnostic value")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")
    out = OUT_DIR / "fig_latest_tail20_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = load_pickle(LATEST_RUN / "input_data.pkl")
    prior = load_pickle(PRIOR_RUN / "input_data.pkl")
    compare = load_pickle(LATEST_COMPARE / "input_data.pkl")
    baseline = load_pickle(BASELINE_BUNDLE)
    latest_stage = pd.read_csv(LATEST_RUN / "markov_stage_diagnostics.csv")
    prior_stage = pd.read_csv(PRIOR_RUN / "markov_stage_diagnostics.csv")

    summary = summarize(latest, prior, baseline, compare, latest_stage, prior_stage)
    figures = {
        "reward": str(plot_rewards(latest, prior, compare, summary).relative_to(REPO_ROOT)),
        "final_tracking": str(plot_final_tracking(latest, prior, baseline).relative_to(REPO_ROOT)),
        "block_tracking": str(plot_block_metrics(summary).relative_to(REPO_ROOT)),
        "diagnostics": str(plot_diagnostics(summary).relative_to(REPO_ROOT)),
        "inside_band": str((OUT_DIR / "fig_latest_tail20_inside_band.png").relative_to(REPO_ROOT)),
    }
    summary["figures"] = figures
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
