import collections
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from utils.helpers import apply_min_max, reverse_min_max


def _set_plot_style(style_profile="hybrid"):
    style_profile = str(style_profile).lower()
    if style_profile not in {"hybrid", "paper", "debug"}:
        style_profile = "hybrid"

    profile_map = {
        "hybrid": {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "figure.dpi": 120,
        },
        "paper": {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "figure.dpi": 140,
        },
        "debug": {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.3,
            "lines.markersize": 5,
            "figure.dpi": 120,
        },
    }
    palette = ["#0B3954", "#C81D25", "#2D6A4F", "#6A4C93", "#F4A259", "#7D8597", "#3A86FF", "#8338EC"]
    rc = {
        "axes.labelweight": "bold",
        "axes.titlesize": profile_map[style_profile]["axes.labelsize"],
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.35,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=palette),
    }
    rc.update(profile_map[style_profile])
    plt.rcParams.update(rc)


def _make_axes_bold(ax):
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def _save_fig(fig, out_dir, fname_base, save_pdf=False):
    fig.savefig(os.path.join(out_dir, fname_base + ".png"), dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(os.path.join(out_dir, fname_base + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def normalize_result_bundle(result_bundle):
    bundle = dict(result_bundle)

    y = bundle.get("y", bundle.get("y_rl", bundle.get("y_mpc")))
    u = bundle.get("u", bundle.get("u_rl", bundle.get("u_mpc")))
    if y is None or u is None:
        raise KeyError("result_bundle must contain y/u or legacy y_rl/u_rl or y_mpc/u_mpc keys.")

    bundle["y"] = np.asarray(y, float)
    bundle["u"] = np.asarray(u, float)
    bundle["avg_rewards"] = np.asarray(bundle.get("avg_rewards", []), float)
    bundle["avg_rewards_mpc"] = np.asarray(bundle.get("avg_rewards_mpc", []), float)
    bundle["data_min"] = np.asarray(bundle["data_min"], float)
    bundle["data_max"] = np.asarray(bundle["data_max"], float)
    bundle["nFE"] = int(bundle["nFE"])
    bundle["delta_t"] = float(bundle["delta_t"])
    bundle["time_in_sub_episodes"] = int(bundle["time_in_sub_episodes"])
    bundle["y_sp"] = np.asarray(bundle["y_sp"], float)
    bundle["rewards_step"] = None if bundle.get("rewards_step") is None else np.asarray(bundle["rewards_step"], float)
    bundle["rewards_mpc"] = None if bundle.get("rewards_mpc") is None else np.asarray(bundle["rewards_mpc"], float)
    bundle["delta_y_storage"] = None if bundle.get("delta_y_storage") is None else np.asarray(bundle["delta_y_storage"], float)
    bundle["delta_u_storage"] = None if bundle.get("delta_u_storage") is None else np.asarray(bundle["delta_u_storage"], float)
    bundle["horizon_trace"] = None if bundle.get("horizon_trace") is None else np.asarray(bundle["horizon_trace"], float)
    bundle["action_trace"] = None if bundle.get("action_trace") is None else np.asarray(bundle["action_trace"], int)
    bundle["horizon_action_trace"] = (
        None if bundle.get("horizon_action_trace") is None else np.asarray(bundle["horizon_action_trace"], int).reshape(-1)
    )
    bundle["horizon_decision_log"] = (
        None if bundle.get("horizon_decision_log") is None else np.asarray(bundle["horizon_decision_log"], int).reshape(-1)
    )
    bundle["yhat"] = None if bundle.get("yhat") is None else np.asarray(bundle["yhat"], float)
    bundle["xhatdhat"] = None if bundle.get("xhatdhat") is None else np.asarray(bundle["xhatdhat"], float)
    bundle["alpha_log"] = None if bundle.get("alpha_log") is None else np.asarray(bundle["alpha_log"], float).reshape(-1)
    bundle["delta_log"] = None if bundle.get("delta_log") is None else np.asarray(bundle["delta_log"], float)
    bundle["matrix_alpha_log"] = (
        None if bundle.get("matrix_alpha_log") is None else np.asarray(bundle["matrix_alpha_log"], float).reshape(-1)
    )
    bundle["matrix_delta_log"] = (
        None if bundle.get("matrix_delta_log") is None else np.asarray(bundle["matrix_delta_log"], float)
    )
    bundle["matrix_decision_log"] = (
        None if bundle.get("matrix_decision_log") is None else np.asarray(bundle["matrix_decision_log"], int).reshape(-1)
    )
    bundle["weight_log"] = None if bundle.get("weight_log") is None else np.asarray(bundle["weight_log"], float)
    bundle["weight_decision_log"] = (
        None if bundle.get("weight_decision_log") is None else np.asarray(bundle["weight_decision_log"], int).reshape(-1)
    )
    bundle["residual_exec_log"] = (
        None if bundle.get("residual_exec_log") is None else np.asarray(bundle["residual_exec_log"], float)
    )
    bundle["residual_raw_log"] = (
        None if bundle.get("residual_raw_log") is None else np.asarray(bundle["residual_raw_log"], float)
    )
    bundle["residual_decision_log"] = (
        None if bundle.get("residual_decision_log") is None else np.asarray(bundle["residual_decision_log"], int).reshape(-1)
    )
    bundle["rho_log"] = None if bundle.get("rho_log") is None else np.asarray(bundle["rho_log"], float).reshape(-1)
    bundle["innovation_log"] = None if bundle.get("innovation_log") is None else np.asarray(bundle["innovation_log"], float)
    bundle["tracking_error_log"] = (
        None if bundle.get("tracking_error_log") is None else np.asarray(bundle["tracking_error_log"], float)
    )
    for key in (
        "horizon_innovation_log",
        "horizon_tracking_error_log",
        "matrix_innovation_log",
        "matrix_tracking_error_log",
        "weight_innovation_log",
        "weight_tracking_error_log",
        "residual_innovation_log",
        "residual_tracking_error_log",
    ):
        bundle[key] = None if bundle.get(key) is None else np.asarray(bundle[key], float)
    bundle["u_base"] = None if bundle.get("u_base") is None else np.asarray(bundle["u_base"], float)
    bundle["actor_losses"] = None if bundle.get("actor_losses") is None else np.asarray(bundle["actor_losses"], float).reshape(-1)
    bundle["critic_losses"] = None if bundle.get("critic_losses") is None else np.asarray(bundle["critic_losses"], float).reshape(-1)
    bundle["alpha_losses"] = None if bundle.get("alpha_losses") is None else np.asarray(bundle["alpha_losses"], float).reshape(-1)
    bundle["alphas"] = None if bundle.get("alphas") is None else np.asarray(bundle["alphas"], float).reshape(-1)
    for key in (
        "matrix_actor_losses",
        "matrix_critic_losses",
        "matrix_alpha_losses",
        "matrix_alphas",
        "weight_actor_losses",
        "weight_critic_losses",
        "weight_alpha_losses",
        "weight_alphas",
        "residual_actor_losses",
        "residual_critic_losses",
        "residual_alpha_losses",
        "residual_alphas",
    ):
        bundle[key] = None if bundle.get(key) is None else np.asarray(bundle[key], float).reshape(-1)
    bundle["low_coef"] = None if bundle.get("low_coef") is None else np.asarray(bundle["low_coef"], float).reshape(-1)
    bundle["high_coef"] = None if bundle.get("high_coef") is None else np.asarray(bundle["high_coef"], float).reshape(-1)
    bundle["matrix_low_coef"] = (
        None if bundle.get("matrix_low_coef") is None else np.asarray(bundle["matrix_low_coef"], float).reshape(-1)
    )
    bundle["matrix_high_coef"] = (
        None if bundle.get("matrix_high_coef") is None else np.asarray(bundle["matrix_high_coef"], float).reshape(-1)
    )
    bundle["weight_low_coef"] = (
        None if bundle.get("weight_low_coef") is None else np.asarray(bundle["weight_low_coef"], float).reshape(-1)
    )
    bundle["weight_high_coef"] = (
        None if bundle.get("weight_high_coef") is None else np.asarray(bundle["weight_high_coef"], float).reshape(-1)
    )
    bundle["residual_low_coef"] = (
        None if bundle.get("residual_low_coef") is None else np.asarray(bundle["residual_low_coef"], float).reshape(-1)
    )
    bundle["residual_high_coef"] = (
        None if bundle.get("residual_high_coef") is None else np.asarray(bundle["residual_high_coef"], float).reshape(-1)
    )
    bundle["state_mode"] = str(bundle.get("state_mode", "standard")).lower()

    if bundle["y"].shape[0] == bundle["nFE"]:
        bundle["y_line_full"] = np.vstack([bundle["y"], bundle["y"][-1:, :]])
    else:
        bundle["y_line_full"] = bundle["y"][: bundle["nFE"] + 1, :]

    bundle["u_step_full"] = bundle["u"][: bundle["nFE"], :]
    bundle["n_inputs"] = bundle["u_step_full"].shape[1]
    bundle["n_outputs"] = bundle["y_line_full"].shape[1]

    if bundle["delta_log"] is not None:
        if bundle["delta_log"].ndim == 1:
            bundle["delta_log"] = bundle["delta_log"][:, None]
        if bundle["delta_log"].shape[0] > bundle["nFE"]:
            bundle["delta_log"] = bundle["delta_log"][: bundle["nFE"], :]

    if bundle["alpha_log"] is not None and bundle["alpha_log"].shape[0] > bundle["nFE"]:
        bundle["alpha_log"] = bundle["alpha_log"][: bundle["nFE"]]
    if bundle["matrix_alpha_log"] is not None and bundle["matrix_alpha_log"].shape[0] > bundle["nFE"]:
        bundle["matrix_alpha_log"] = bundle["matrix_alpha_log"][: bundle["nFE"]]

    if bundle["weight_log"] is not None:
        if bundle["weight_log"].ndim == 1:
            bundle["weight_log"] = bundle["weight_log"][:, None]
        if bundle["weight_log"].shape[0] > bundle["nFE"]:
            bundle["weight_log"] = bundle["weight_log"][: bundle["nFE"], :]
    if bundle["matrix_delta_log"] is not None:
        if bundle["matrix_delta_log"].ndim == 1:
            bundle["matrix_delta_log"] = bundle["matrix_delta_log"][:, None]
        if bundle["matrix_delta_log"].shape[0] > bundle["nFE"]:
            bundle["matrix_delta_log"] = bundle["matrix_delta_log"][: bundle["nFE"], :]

    if bundle["residual_exec_log"] is not None:
        if bundle["residual_exec_log"].ndim == 1:
            bundle["residual_exec_log"] = bundle["residual_exec_log"][:, None]
        if bundle["residual_exec_log"].shape[0] > bundle["nFE"]:
            bundle["residual_exec_log"] = bundle["residual_exec_log"][: bundle["nFE"], :]

    if bundle["residual_raw_log"] is not None:
        if bundle["residual_raw_log"].ndim == 1:
            bundle["residual_raw_log"] = bundle["residual_raw_log"][:, None]
        if bundle["residual_raw_log"].shape[0] > bundle["nFE"]:
            bundle["residual_raw_log"] = bundle["residual_raw_log"][: bundle["nFE"], :]

    if bundle["u_base"] is not None:
        if bundle["u_base"].shape[0] > bundle["nFE"]:
            bundle["u_base"] = bundle["u_base"][: bundle["nFE"], :]

    for key in ("innovation_log", "tracking_error_log"):
        if bundle[key] is not None:
            if bundle[key].ndim == 1:
                bundle[key] = bundle[key][:, None]
            if bundle[key].shape[0] > bundle["nFE"]:
                bundle[key] = bundle[key][: bundle["nFE"], :]
    for key in (
        "horizon_innovation_log",
        "horizon_tracking_error_log",
        "matrix_innovation_log",
        "matrix_tracking_error_log",
        "weight_innovation_log",
        "weight_tracking_error_log",
        "residual_innovation_log",
        "residual_tracking_error_log",
    ):
        if bundle[key] is not None:
            if bundle[key].ndim == 1:
                bundle[key] = bundle[key][:, None]
            if bundle[key].shape[0] > bundle["nFE"]:
                bundle[key] = bundle[key][: bundle["nFE"], :]

    return bundle


def normalize_external_bundle(result_bundle, reference_bundle):
    merged = dict(result_bundle)
    for key in ("y_sp", "steady_states", "nFE", "delta_t", "time_in_sub_episodes", "data_min", "data_max"):
        if key not in merged:
            merged[key] = reference_bundle[key]
    return normalize_result_bundle(merged)


def build_storage_bundle(bundle, start_episode):
    stored = dict(bundle)
    stored["y_rl"] = bundle["y_line_full"]
    stored["u_rl"] = bundle["u_step_full"]
    stored["y_mpc"] = bundle["y_line_full"]
    stored["u_mpc"] = bundle["u_step_full"]
    stored["y"] = bundle["y_line_full"]
    stored["u"] = bundle["u_step_full"]
    stored["start_episode_plotted"] = int(start_episode)
    return stored


def create_output_dir(directory, prefix_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_bundle_pickle(out_dir, stored_bundle):
    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as handle:
        pickle.dump(stored_bundle, handle)


def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs):
    y_ss_phys = np.asarray(steady_states["y_ss"], float)
    y_ss_scaled = apply_min_max(y_ss_phys, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled = np.asarray(y_sp_scaled_dev, float) + y_ss_scaled
    return reverse_min_max(y_sp_scaled, data_min[n_inputs:], data_max[n_inputs:])


def slice_avg_rewards(avg_rewards, n_ep_total, start_episode):
    avg = np.asarray(avg_rewards, float)
    if len(avg) == n_ep_total + 1:
        avg = avg[1:]
    start_episode = int(max(1, start_episode))
    start_idx = max(0, start_episode - 1)
    avg = avg[start_idx:]
    x = np.arange(start_episode, start_episode + len(avg))
    return x, avg


def episode_spans(test_train_dict, nFE):
    if not test_train_dict:
        return []
    starts = sorted(int(k) for k in test_train_dict.keys())
    spans = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else int(nFE)
        spans.append((start, end, bool(test_train_dict[start])))
    return spans


def shade_test_regions(ax, spans, delta_t):
    for start, end, is_test in spans:
        if is_test:
            ax.axvspan(start * delta_t, end * delta_t, color="orange", alpha=0.12, linewidth=0)


def load_pickle(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "input_data.pkl")
        if os.path.exists(candidate):
            with open(candidate, "rb") as handle:
                return pickle.load(handle)
        for name in sorted(os.listdir(path)):
            if name.endswith((".pickle", ".pkl")):
                with open(os.path.join(path, name), "rb") as handle:
                    return pickle.load(handle)
        raise FileNotFoundError(f"No pickle files found in directory: {path}")

    with open(path, "rb") as handle:
        return pickle.load(handle)


def recompute_step_rewards(y_line_full, u_step, y_sp, steady_states, data_min, data_max, reward_fn):
    n_inputs = u_step.shape[1]
    nFE = min(len(y_sp), len(u_step))
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])

    y_scaled_abs = apply_min_max(y_line_full[: nFE + 1, :], data_min[n_inputs:], data_max[n_inputs:])
    y_scaled_dev = y_scaled_abs - y_ss_scaled
    u_scaled_abs = apply_min_max(u_step[:nFE, :], data_min[:n_inputs], data_max[:n_inputs])

    delta_y = y_scaled_dev[1:, :] - y_sp[:nFE, :]
    delta_u = np.zeros((nFE, n_inputs))
    delta_u[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
    if nFE > 1:
        delta_u[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

    y_sp_phys = ysp_scaled_dev_to_phys(y_sp[:nFE, :], steady_states, data_min, data_max, n_inputs)
    rewards = np.zeros(nFE)
    for idx in range(nFE):
        rewards[idx] = reward_fn(delta_y[idx], delta_u[idx], y_sp_phys=y_sp_phys[idx])
    return rewards, delta_y, delta_u


def _plot_mismatch_diagnostics(bundle, out_dir, prefix, t_step, t_step_blk, start_step, W, s_last, last_steps, spans, delta_t, save_pdf):
    innovation_log = bundle.get("innovation_log")
    tracking_error_log = bundle.get("tracking_error_log")
    if innovation_log is None and tracking_error_log is None:
        return

    n_outputs = bundle["n_outputs"]

    def _plot_series(series, ylabel_stem, title_stem, fname_base, t_axis):
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.3 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_axis, series[:, idx])
            if len(t_axis) == len(t_step):
                shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"{ylabel_stem}{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, f"{prefix}_{fname_base}", save_pdf=save_pdf)

    if innovation_log is not None:
        innov_seg = innovation_log[start_step : start_step + W, :]
        _plot_series(innov_seg, "innov", "Innovation", "innovation_full", t_step)
        _plot_series(innov_seg[s_last : s_last + last_steps, :], "innov", "Innovation", "innovation_last_block", t_step_blk)

    if tracking_error_log is not None:
        terr_seg = tracking_error_log[start_step : start_step + W, :]
        _plot_series(terr_seg, "etrack", "Tracking Error", "tracking_state_full", t_step)
        _plot_series(
            terr_seg[s_last : s_last + last_steps, :],
            "etrack",
            "Tracking Error",
            "tracking_state_last_block",
            t_step_blk,
        )


def plot_baseline_mpc_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "mpc_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    output_labels = [r"$\eta$ (L/g)", r"$T$ (K)"] if n_outputs == 2 else [f"y{idx + 1}" for idx in range(n_outputs)]
    input_labels = [r"$Q_c$ (L/h)", r"$Q_m$ (L/h)"] if n_inputs == 2 else [f"u{idx + 1}" for idx in range(n_inputs)]

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="MPC")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_mpc_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="MPC")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_mpc_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_mpc_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_mpc_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    x_ep_mpc, y_ep_mpc = slice_avg_rewards(bundle.get("avg_rewards_mpc", []), n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-", label="Shared reward")
    if len(y_ep_mpc) > 0:
        ax.plot(x_ep_mpc, y_ep_mpc, "s--", label="Quadratic MPC reward")
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_mpc_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.plot(t_step, rewards_seg, label="Shared reward")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_mpc_step_rewards", save_pdf=save_pdf)

    rewards_mpc = bundle.get("rewards_mpc")
    if rewards_mpc is not None:
        rewards_mpc_seg = rewards_mpc[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.plot(t_step, rewards_mpc_seg, label="Quadratic MPC reward")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_mpc_step_rewards_quadratic", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        dy_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.2, 4.0 + 2.0 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, dy_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        axs[-1].plot(t_step, np.linalg.norm(dy_seg, axis=1))
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel(r"$||e||_2$")
        axs[-1].set_xlabel("Time (h)")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_mpc_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_mpc_delta_u", save_pdf=save_pdf)

    if bundle.get("yhat") is not None:
        yhat_seg = bundle["yhat"][:, start_step : start_step + W].T
        y_meas_seg = y_line[1 : 1 + W, :]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_step, y_meas_seg[:, idx], label="Measured")
            ax.plot(t_step, yhat_seg[:, idx], linestyle="--", label="Observer")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_mpc_observer_overlay", save_pdf=save_pdf)

    if bundle.get("disturbance_profile") is not None:
        disturbance = bundle["disturbance_profile"]
        fig, axs = plt.subplots(3, 1, figsize=(8.2, 7.0), sharex=True)
        keys = [("qi", r"$Q_i$"), ("qs", r"$Q_s$"), ("ha", r"$hA$")]
        for ax, (key, label) in zip(axs, keys):
            ax.plot(t_step, np.asarray(disturbance[key], float)[start_step : start_step + W])
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_mpc_disturbance_profile", save_pdf=save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_horizon_results_core(result_bundle, plot_cfg):
    _set_plot_style()
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "horizon_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    recipe_counts = bool(plot_cfg.get("recipe_counts", True))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_horizon_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_horizon_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_horizon_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_horizon_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_horizon_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_horizon_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel("Time (h)")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_horizon_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"du{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_horizon_delta_u", save_pdf=save_pdf)

    horizon_trace = bundle.get("horizon_trace")
    mpc_horizons = bundle.get("mpc_horizons")
    if horizon_trace is not None:
        ht_full = np.asarray(horizon_trace, float)
        if ht_full.shape[0] > nFE:
            ht_full = ht_full[:nFE, :]
        ht = ht_full[start_step : start_step + W, :]
        Hp = ht[:, 0].astype(int)
        Hc = ht[:, 1].astype(int)
        Hp0 = None if not mpc_horizons else int(mpc_horizons[0])
        Hc0 = None if not mpc_horizons else int(mpc_horizons[1])

        fig, axs = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
        axs[0].step(t_step, Hp, where="post")
        axs[1].step(t_step, Hc, where="post")
        for ax, val, label in [(axs[0], Hp0, "Hp"), (axs[1], Hc0, "Hc")]:
            if val is not None:
                ax.axhline(float(val), color="red", linestyle=":", linewidth=2.0)
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_horizon_horizons_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
        axs[0].step(t_step_blk, Hp[s_last : s_last + last_steps], where="post")
        axs[1].step(t_step_blk, Hc[s_last : s_last + last_steps], where="post")
        for ax, val, label in [(axs[0], Hp0, "Hp"), (axs[1], Hc0, "Hc")]:
            if val is not None:
                ax.axhline(float(val), color="red", linestyle=":", linewidth=2.0)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_horizon_horizons_last_block", save_pdf=save_pdf)

        if recipe_counts:
            counts = collections.Counter(list(zip(Hp.tolist(), Hc.tolist())))
            items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            top = items[:20]
            key0 = None if Hp0 is None or Hc0 is None else (Hp0, Hc0)
            if key0 is not None and key0 not in dict(top):
                top.append((key0, counts.get(key0, 0)))
            labels = [f"Hp={k[0]},Hc={k[1]}" for k, _ in top]
            freqs = [v for _, v in top]
            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            if freqs:
                bars = ax.bar(np.arange(len(freqs)), freqs)
                if key0 is not None:
                    for idx, (k, _) in enumerate(top):
                        if k == key0:
                            bars[idx].set_color("red")
                            ax.text(idx, freqs[idx], "MPC", ha="center", va="bottom", fontweight="bold")
                            break
                ax.set_xticks(np.arange(len(freqs)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_horizon_recipe_counts", save_pdf=save_pdf)

    action_trace = bundle.get("action_trace")
    if action_trace is not None:
        action_seg = np.asarray(action_trace, int)[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.step(t_step, action_seg, where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Action")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_horizon_action_trace", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"yhat{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_horizon_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        keys = [key for key in ("qi", "qs", "ha") if key in disturbance_profile]
        if keys:
            fig, axs = plt.subplots(len(keys), 1, figsize=(8.6, 3.0 + 2.0 * max(1, len(keys) - 1)), sharex=True)
            if len(keys) == 1:
                axs = [axs]
            for idx, key in enumerate(keys):
                series = np.asarray(disturbance_profile[key], float)[start_step : start_step + W]
                axs[idx].plot(t_step, series)
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(key)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel("Time (h)")
            _save_fig(fig, out_dir, "fig_horizon_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_horizon",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    stored_bundle = build_storage_bundle(bundle, start_episode)
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_matrix_multiplier_results_core(result_bundle, plot_cfg):
    _set_plot_style()
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "matrix_multiplier_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_matrix_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_matrix_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_matrix_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_matrix_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_matrix_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_matrix_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel("Time (h)")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_matrix_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"du{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_matrix_delta_u", save_pdf=save_pdf)

    alpha_log = bundle.get("alpha_log")
    delta_log = bundle.get("delta_log")
    low_coef = bundle.get("low_coef")
    high_coef = bundle.get("high_coef")
    if alpha_log is not None:
        alpha_seg = alpha_log[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        if low_coef is not None and high_coef is not None and low_coef.size > 0:
            ax.fill_between(t_step, float(low_coef[0]), float(high_coef[0]), color="tab:blue", alpha=0.12, step="post")
            ax.axhline(float(low_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
            ax.axhline(float(high_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
        ax.step(t_step, alpha_seg, where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("alpha")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_matrix_alpha_full", save_pdf=save_pdf)

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        if low_coef is not None and high_coef is not None and low_coef.size > 0:
            ax.fill_between(t_step_blk, float(low_coef[0]), float(high_coef[0]), color="tab:blue", alpha=0.12, step="post")
            ax.axhline(float(low_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
            ax.axhline(float(high_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
        ax.step(t_step_blk, alpha_seg[s_last : s_last + last_steps], where="post")
        ax.set_ylabel("alpha")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_matrix_alpha_last_block", save_pdf=save_pdf)

    if delta_log is not None:
        delta_seg = delta_log[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and high_coef.size > idx + 1:
                ax.fill_between(
                    t_step,
                    float(low_coef[idx + 1]),
                    float(high_coef[idx + 1]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
            ax.step(t_step, delta_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"delta{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_matrix_delta_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and high_coef.size > idx + 1:
                ax.fill_between(
                    t_step_blk,
                    float(low_coef[idx + 1]),
                    float(high_coef[idx + 1]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
            ax.step(t_step_blk, delta_seg[s_last : s_last + last_steps, idx], where="post")
            ax.set_ylabel(f"delta{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_matrix_delta_last_block", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"yhat{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_matrix_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        keys = [key for key in ("qi", "qs", "ha") if key in disturbance_profile]
        if keys:
            fig, axs = plt.subplots(len(keys), 1, figsize=(8.6, 3.0 + 2.0 * max(1, len(keys) - 1)), sharex=True)
            if len(keys) == 1:
                axs = [axs]
            for idx, key in enumerate(keys):
                series = np.asarray(disturbance_profile[key], float)[start_step : start_step + W]
                axs[idx].plot(t_step, series)
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(key)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel("Time (h)")
            _save_fig(fig, out_dir, "fig_matrix_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_matrix",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    diag_series = []
    if bundle.get("actor_losses") is not None and len(bundle["actor_losses"]) > 0:
        diag_series.append(("actor_loss", bundle["actor_losses"]))
    if bundle.get("critic_losses") is not None and len(bundle["critic_losses"]) > 0:
        diag_series.append(("critic_loss", bundle["critic_losses"]))
    if bundle.get("alphas") is not None and len(bundle["alphas"]) > 0:
        diag_series.append(("sac_alpha", bundle["alphas"]))
    if bundle.get("alpha_losses") is not None and len(bundle["alpha_losses"]) > 0:
        diag_series.append(("sac_alpha_loss", bundle["alpha_losses"]))
    if diag_series:
        fig, axs = plt.subplots(len(diag_series), 1, figsize=(8.4, 3.0 + 2.2 * max(1, len(diag_series) - 1)), sharex=False)
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_matrix_training_diagnostics", save_pdf=save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    stored_bundle.update(
        {
            "alpha_log": bundle.get("alpha_log"),
            "delta_log": bundle.get("delta_log"),
            "low_coef": bundle.get("low_coef"),
            "high_coef": bundle.get("high_coef"),
            "agent_kind": bundle.get("agent_kind"),
            "run_mode": bundle.get("run_mode"),
            "test_train_dict": bundle.get("test_train_dict"),
            "disturbance_profile": bundle.get("disturbance_profile"),
            "actor_losses": bundle.get("actor_losses"),
            "critic_losses": bundle.get("critic_losses"),
            "alpha_losses": bundle.get("alpha_losses"),
            "alphas": bundle.get("alphas"),
            "mpc_path_or_dir": bundle.get("mpc_path_or_dir"),
        }
    )
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_weight_multiplier_results_core(result_bundle, plot_cfg):
    _set_plot_style()
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "weight_multiplier_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_weights_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_weights_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_weights_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_weights_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_weights_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_weights_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel("Time (h)")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_weights_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"du{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_weights_delta_u", save_pdf=save_pdf)

    weight_log = bundle.get("weight_log")
    low_coef = bundle.get("low_coef")
    high_coef = bundle.get("high_coef")
    if weight_log is not None:
        w_seg = weight_log[start_step : start_step + W, :]
        names = ["Q1", "Q2", "R1", "R2"]
        fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == 4 and high_coef.size == 4:
                ax.fill_between(
                    t_step,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:purple",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
            ax.step(t_step, w_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(names[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_weights_multipliers_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == 4 and high_coef.size == 4:
                ax.fill_between(
                    t_step_blk,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:purple",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
            ax.step(t_step_blk, w_seg[s_last : s_last + last_steps, idx], where="post")
            ax.set_ylabel(names[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_weights_multipliers_last_block", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"yhat{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_weights_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        keys = [key for key in ("qi", "qs", "ha") if key in disturbance_profile]
        if keys:
            fig, axs = plt.subplots(len(keys), 1, figsize=(8.6, 3.0 + 2.0 * max(1, len(keys) - 1)), sharex=True)
            if len(keys) == 1:
                axs = [axs]
            for idx, key in enumerate(keys):
                series = np.asarray(disturbance_profile[key], float)[start_step : start_step + W]
                axs[idx].plot(t_step, series)
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(key)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel("Time (h)")
            _save_fig(fig, out_dir, "fig_weights_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_weights",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    diag_series = []
    if bundle.get("actor_losses") is not None and len(bundle["actor_losses"]) > 0:
        diag_series.append(("actor_loss", bundle["actor_losses"]))
    if bundle.get("critic_losses") is not None and len(bundle["critic_losses"]) > 0:
        diag_series.append(("critic_loss", bundle["critic_losses"]))
    if bundle.get("alphas") is not None and len(bundle["alphas"]) > 0:
        diag_series.append(("sac_alpha", bundle["alphas"]))
    if bundle.get("alpha_losses") is not None and len(bundle["alpha_losses"]) > 0:
        diag_series.append(("sac_alpha_loss", bundle["alpha_losses"]))
    if diag_series:
        fig, axs = plt.subplots(len(diag_series), 1, figsize=(8.4, 3.0 + 2.2 * max(1, len(diag_series) - 1)), sharex=False)
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_weights_training_diagnostics", save_pdf=save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    stored_bundle.update(
        {
            "weight_log": bundle.get("weight_log"),
            "low_coef": bundle.get("low_coef"),
            "high_coef": bundle.get("high_coef"),
            "agent_kind": bundle.get("agent_kind"),
            "run_mode": bundle.get("run_mode"),
            "test_train_dict": bundle.get("test_train_dict"),
            "disturbance_profile": bundle.get("disturbance_profile"),
            "actor_losses": bundle.get("actor_losses"),
            "critic_losses": bundle.get("critic_losses"),
            "alpha_losses": bundle.get("alpha_losses"),
            "alphas": bundle.get("alphas"),
            "mpc_path_or_dir": bundle.get("mpc_path_or_dir"),
        }
    )
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_residual_results_core(result_bundle, plot_cfg):
    _set_plot_style()
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "residual_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_residual_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_residual_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_residual_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_residual_inputs_last_block", save_pdf=save_pdf)

    u_base = bundle.get("u_base")
    if u_base is not None:
        u_base_seg = u_base[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.2 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, u_line[:, idx], where="post", label="Applied")
            ax.step(t_step, u_base_seg[:, idx], where="post", linestyle="--", label="MPC baseline")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"u{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_inputs_overlay_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.2 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post", label="Applied")
            ax.step(
                t_step_blk,
                u_base_seg[s_last : s_last + last_steps, idx],
                where="post",
                linestyle="--",
                label="MPC baseline",
            )
            ax.set_ylabel(f"u{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_inputs_overlay_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_residual_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_residual_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel("Time (h)")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_residual_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"du{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_residual_delta_u", save_pdf=save_pdf)

    residual_exec_log = bundle.get("residual_exec_log")
    residual_raw_log = bundle.get("residual_raw_log")
    low_coef = bundle.get("low_coef")
    high_coef = bundle.get("high_coef")
    if residual_exec_log is not None:
        exec_seg = residual_exec_log[start_step : start_step + W, :]
        raw_seg = None if residual_raw_log is None else residual_raw_log[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == n_inputs and high_coef.size == n_inputs:
                ax.fill_between(
                    t_step,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
            if raw_seg is not None:
                ax.step(t_step, raw_seg[:, idx], where="post", linestyle=":", label="Raw")
            ax.step(t_step, exec_seg[:, idx], where="post", label="Executed")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"r{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_correction_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == n_inputs and high_coef.size == n_inputs:
                ax.fill_between(
                    t_step_blk,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
            if raw_seg is not None:
                ax.step(
                    t_step_blk,
                    raw_seg[s_last : s_last + last_steps, idx],
                    where="post",
                    linestyle=":",
                    label="Raw",
                )
            ax.step(
                t_step_blk,
                exec_seg[s_last : s_last + last_steps, idx],
                where="post",
                label="Executed",
            )
            ax.set_ylabel(f"r{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_correction_last_block", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"yhat{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        keys = [key for key in ("qi", "qs", "ha") if key in disturbance_profile]
        if keys:
            fig, axs = plt.subplots(len(keys), 1, figsize=(8.6, 3.0 + 2.0 * max(1, len(keys) - 1)), sharex=True)
            if len(keys) == 1:
                axs = [axs]
            for idx, key in enumerate(keys):
                series = np.asarray(disturbance_profile[key], float)[start_step : start_step + W]
                axs[idx].plot(t_step, series)
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(key)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel("Time (h)")
            _save_fig(fig, out_dir, "fig_residual_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_residual",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    diag_series = []
    if bundle.get("actor_losses") is not None and len(bundle["actor_losses"]) > 0:
        diag_series.append(("actor_loss", bundle["actor_losses"]))
    if bundle.get("critic_losses") is not None and len(bundle["critic_losses"]) > 0:
        diag_series.append(("critic_loss", bundle["critic_losses"]))
    if bundle.get("alphas") is not None and len(bundle["alphas"]) > 0:
        diag_series.append(("sac_alpha", bundle["alphas"]))
    if bundle.get("alpha_losses") is not None and len(bundle["alpha_losses"]) > 0:
        diag_series.append(("sac_alpha_loss", bundle["alpha_losses"]))
    if diag_series:
        fig, axs = plt.subplots(
            len(diag_series),
            1,
            figsize=(8.4, 3.0 + 2.2 * max(1, len(diag_series) - 1)),
            sharex=False,
        )
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_residual_training_diagnostics", save_pdf=save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    stored_bundle.update(
        {
            "residual_exec_log": bundle.get("residual_exec_log"),
            "residual_raw_log": bundle.get("residual_raw_log"),
            "u_base": bundle.get("u_base"),
            "low_coef": bundle.get("low_coef"),
            "high_coef": bundle.get("high_coef"),
            "agent_kind": bundle.get("agent_kind"),
            "run_mode": bundle.get("run_mode"),
            "test_train_dict": bundle.get("test_train_dict"),
            "disturbance_profile": bundle.get("disturbance_profile"),
            "actor_losses": bundle.get("actor_losses"),
            "critic_losses": bundle.get("critic_losses"),
            "alpha_losses": bundle.get("alpha_losses"),
            "alphas": bundle.get("alphas"),
            "mpc_path_or_dir": bundle.get("mpc_path_or_dir"),
        }
    )
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_combined_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "combined_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    compare_start_episode = int(plot_cfg.get("compare_start_episode", start_episode))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    reward_fn = plot_cfg.get("reward_fn")
    include_baseline_compare = bool(plot_cfg.get("include_baseline_compare", True))
    compare_mode = str(plot_cfg.get("compare_mode", bundle.get("run_mode", "nominal"))).lower()
    compare_prefix = plot_cfg.get("compare_prefix", "baseline_compare")
    out_dir = create_output_dir(directory, prefix_name)

    active_agents = dict(bundle.get("active_agents", {}))
    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)
    debug_mode = style_profile == "debug"

    output_labels = [r"$\eta$ (L/g)", r"$T$ (K)"] if n_outputs == 2 else [f"y{idx + 1}" for idx in range(n_outputs)]
    input_labels = [r"$Q_c$ (L/h)", r"$Q_m$ (L/h)"] if n_inputs == 2 else [f"u{idx + 1}" for idx in range(n_inputs)]

    def shade_segment(ax, segment_start, segment_len):
        for span_start, span_end, is_test in spans:
            if not is_test:
                continue
            clipped_start = max(segment_start, span_start)
            clipped_end = min(segment_start + segment_len, span_end)
            if clipped_end > clipped_start:
                ax.axvspan(
                    (clipped_start - segment_start) * delta_t,
                    (clipped_end - segment_start) * delta_t,
                    color="orange",
                    alpha=0.12,
                    linewidth=0,
                )

    def plot_outputs(prefix, y_seg, ysp_seg, t_line_local, t_step_local, segment_start, segment_len):
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.4, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_line_local, y_seg[:, idx], label="Combined RL")
            ax.step(t_step_local, ysp_seg[:, idx], where="post", linestyle="--", label="Setpoint")
            shade_segment(ax, segment_start, segment_len)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, prefix, save_pdf=save_pdf)

    def plot_inputs(prefix, u_seg, t_step_local, segment_start, segment_len):
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step_local, u_seg[:, idx], where="post")
            shade_segment(ax, segment_start, segment_len)
            ax.set_ylabel(input_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, prefix, save_pdf=save_pdf)

    plot_outputs("fig_combined_outputs_full", y_line, y_sp_phys, t_line, t_step, start_step, W)
    plot_outputs(
        "fig_combined_outputs_last_block",
        y_line[s_last : s_last + last_steps + 1, :],
        y_sp_phys[s_last : s_last + last_steps, :],
        t_line_blk,
        t_step_blk,
        start_step + s_last,
        last_steps,
    )
    plot_inputs("fig_combined_inputs_full", u_line, t_step, start_step, W)
    plot_inputs(
        "fig_combined_inputs_last_block",
        u_line[s_last : s_last + last_steps, :],
        t_step_blk,
        start_step + s_last,
        last_steps,
    )

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_combined_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.plot(t_step, rewards_seg)
        shade_segment(ax, start_step, W)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_combined_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        dy_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.4, 4.2 + 2.1 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, dy_seg[:, idx])
            shade_segment(axs[idx], start_step, W)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        axs[-1].plot(t_step, np.linalg.norm(dy_seg, axis=1))
        shade_segment(axs[-1], start_step, W)
        axs[-1].set_ylabel(r"$||e||_2$")
        axs[-1].set_xlabel("Time (h)")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_combined_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_combined_delta_u", save_pdf=save_pdf)

    if bundle.get("yhat") is not None:
        yhat_seg = bundle["yhat"][:, start_step : start_step + W].T
        y_meas_seg = y_line[1 : 1 + W, :]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.4, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_step, y_meas_seg[:, idx], label="Measured")
            ax.plot(t_step, yhat_seg[:, idx], linestyle="--", label="Observer")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_combined_observer_overlay", save_pdf=save_pdf)

    if bundle.get("disturbance_profile") is not None:
        disturbance = bundle["disturbance_profile"]
        fig, axs = plt.subplots(3, 1, figsize=(8.3, 7.2), sharex=True)
        keys = [("qi", r"$Q_i$"), ("qs", r"$Q_s$"), ("ha", r"$hA$")]
        for ax, (key, label) in zip(axs, keys):
            ax.plot(t_step, np.asarray(disturbance[key], float)[start_step : start_step + W])
            shade_segment(ax, start_step, W)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_combined_disturbance_profile", save_pdf=save_pdf)

    if active_agents.get("horizon") and bundle.get("horizon_trace") is not None:
        horizon_trace = np.asarray(bundle["horizon_trace"], float)
        ht_seg = horizon_trace[start_step : start_step + W, :]
        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].step(t_step, ht_seg[:, 0], where="post")
        shade_segment(axs[0], start_step, W)
        axs[0].set_ylabel("Hp")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])
        axs[1].step(t_step, ht_seg[:, 1], where="post")
        shade_segment(axs[1], start_step, W)
        axs[1].set_ylabel("Hc")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, out_dir, "fig_combined_horizons", save_pdf=save_pdf)

        if bundle.get("horizon_action_trace") is not None:
            a_seg = bundle["horizon_action_trace"][start_step : start_step + W]
            fig, ax = plt.subplots(figsize=(8.2, 4.7))
            ax.step(t_step, a_seg, where="post")
            shade_segment(ax, start_step, W)
            ax.set_ylabel("Action Index")
            ax.set_xlabel("Time (h)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_combined_horizon_action_trace", save_pdf=save_pdf)

        if debug_mode and bundle.get("horizon_action_trace") is not None:
            counts = collections.Counter(bundle["horizon_action_trace"][start_step : start_step + W].tolist())
            labels = [str(k) for k, _ in sorted(counts.items())]
            values = [v for _, v in sorted(counts.items())]
            fig, ax = plt.subplots(figsize=(7.8, 4.5))
            ax.bar(np.arange(len(values)), values)
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Count")
            ax.set_xlabel("Horizon Action")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_combined_horizon_action_hist", save_pdf=save_pdf)

    if active_agents.get("matrix") and bundle.get("matrix_alpha_log") is not None:
        alpha_seg = bundle["matrix_alpha_log"][start_step : start_step + W]
        delta_seg = bundle["matrix_delta_log"][start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs + 1, 1, figsize=(8.4, 4.0 + 2.0 * n_inputs), sharex=True)
        axs[0].plot(t_step, alpha_seg)
        if bundle.get("matrix_low_coef") is not None:
            axs[0].axhline(bundle["matrix_low_coef"][0], linestyle=":", color="#7D8597")
            axs[0].axhline(bundle["matrix_high_coef"][0], linestyle=":", color="#7D8597")
        shade_segment(axs[0], start_step, W)
        axs[0].set_ylabel(r"$\alpha$")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])
        for idx in range(n_inputs):
            axs[idx + 1].plot(t_step, delta_seg[:, idx])
            if bundle.get("matrix_low_coef") is not None:
                axs[idx + 1].axhline(bundle["matrix_low_coef"][idx + 1], linestyle=":", color="#7D8597")
                axs[idx + 1].axhline(bundle["matrix_high_coef"][idx + 1], linestyle=":", color="#7D8597")
            shade_segment(axs[idx + 1], start_step, W)
            axs[idx + 1].set_ylabel(rf"$\delta_{{{idx + 1}}}$")
            axs[idx + 1].spines["top"].set_visible(False)
            axs[idx + 1].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx + 1])
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_combined_matrix_multipliers", save_pdf=save_pdf)

    if active_agents.get("weights") and bundle.get("weight_log") is not None:
        weight_seg = bundle["weight_log"][start_step : start_step + W, :]
        labels = ["Q1", "Q2", "R1", "R2"]
        fig, axs = plt.subplots(4, 1, figsize=(8.4, 9.0), sharex=True)
        for idx, ax in enumerate(axs):
            ax.plot(t_step, weight_seg[:, idx])
            if bundle.get("weight_low_coef") is not None:
                ax.axhline(bundle["weight_low_coef"][idx], linestyle=":", color="#7D8597")
                ax.axhline(bundle["weight_high_coef"][idx], linestyle=":", color="#7D8597")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_combined_weight_multipliers", save_pdf=save_pdf)

    if active_agents.get("residual") and bundle.get("residual_exec_log") is not None:
        raw_seg = bundle["residual_raw_log"][start_step : start_step + W, :]
        exec_seg = bundle["residual_exec_log"][start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_step, raw_seg[:, idx], label="Raw")
            ax.plot(t_step, exec_seg[:, idx], linestyle="--", label="Executed")
            if bundle.get("residual_low_coef") is not None:
                ax.axhline(bundle["residual_low_coef"][idx], linestyle=":", color="#7D8597")
                ax.axhline(bundle["residual_high_coef"][idx], linestyle=":", color="#7D8597")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(rf"$u^r_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_combined_residual_traces", save_pdf=save_pdf)

        if bundle.get("u_base") is not None:
            u_base_seg = bundle["u_base"][start_step : start_step + W, :]
            fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
            if n_inputs == 1:
                axs = [axs]
            for idx, ax in enumerate(axs):
                ax.step(t_step, u_line[:, idx], where="post", label="Applied")
                ax.step(t_step, u_base_seg[:, idx], where="post", linestyle="--", label="MPC baseline")
                shade_segment(ax, start_step, W)
                ax.set_ylabel(input_labels[idx])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
            axs[-1].set_xlabel("Time (h)")
            axs[0].legend(loc="best")
            _save_fig(fig, out_dir, "fig_combined_residual_input_overlay", save_pdf=save_pdf)

    decision_series = []
    decision_labels = []
    for name, key in (
        ("Horizon", "horizon_decision_log"),
        ("Matrix", "matrix_decision_log"),
        ("Weights", "weight_decision_log"),
        ("Residual", "residual_decision_log"),
    ):
        values = bundle.get(key)
        enabled = active_agents.get(name.lower(), False) if name != "Weights" else active_agents.get("weights", False)
        if values is not None and enabled:
            decision_series.append(np.asarray(values, int)[start_step : start_step + W])
            decision_labels.append(name)
    if decision_series:
        fig, axs = plt.subplots(len(decision_series), 1, figsize=(8.2, 2.2 + 1.5 * len(decision_series)), sharex=True)
        if len(decision_series) == 1:
            axs = [axs]
        for ax, label, series in zip(axs, decision_labels, decision_series):
            ax.step(t_step, series, where="post")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(label)
            ax.set_ylim(-0.05, 1.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_combined_decision_timeline", save_pdf=save_pdf)

    if debug_mode and bundle.get("rho_log") is not None:
        rho_seg = bundle["rho_log"][start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.plot(t_step, rho_seg)
        shade_segment(ax, start_step, W)
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel("Time (h)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_combined_rho_trace", save_pdf=save_pdf)

    if debug_mode:
        def plot_named_mismatch(agent_key, title_key):
            innov = bundle.get(f"{agent_key}_innovation_log")
            terr = bundle.get(f"{agent_key}_tracking_error_log")
            if innov is None and terr is None:
                return
            series_items = []
            if innov is not None:
                series_items.append((innov[start_step : start_step + W, :], f"{title_key} innovation", f"{agent_key}_innovation"))
            if terr is not None:
                series_items.append((terr[start_step : start_step + W, :], f"{title_key} tracking", f"{agent_key}_tracking"))
            for series, label_stem, fname in series_items:
                fig, axs = plt.subplots(n_outputs, 1, figsize=(8.4, 3.0 + 2.2 * max(1, n_outputs - 1)), sharex=True)
                if n_outputs == 1:
                    axs = [axs]
                for idx, ax in enumerate(axs):
                    ax.plot(t_step, series[:, idx])
                    shade_segment(ax, start_step, W)
                    ax.set_ylabel(f"{label_stem} {idx + 1}")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    _make_axes_bold(ax)
                axs[-1].set_xlabel("Time (h)")
                _save_fig(fig, out_dir, f"fig_combined_{fname}", save_pdf=save_pdf)

        for key in ("horizon", "matrix", "weight", "residual"):
            plot_named_mismatch(key, key.capitalize())

        def plot_losses(prefix, label):
            actor_losses = bundle.get(f"{prefix}_actor_losses")
            critic_losses = bundle.get(f"{prefix}_critic_losses")
            alpha_losses = bundle.get(f"{prefix}_alpha_losses")
            alphas = bundle.get(f"{prefix}_alphas")
            plots = [
                (actor_losses, f"{label} actor loss", f"fig_combined_{prefix}_actor_loss"),
                (critic_losses, f"{label} critic loss", f"fig_combined_{prefix}_critic_loss"),
                (alpha_losses, f"{label} alpha loss", f"fig_combined_{prefix}_alpha_loss"),
                (alphas, f"{label} alpha", f"fig_combined_{prefix}_alpha_trace"),
            ]
            for values, ylabel, fname in plots:
                if values is None or len(values) == 0:
                    continue
                fig, ax = plt.subplots(figsize=(7.8, 4.6))
                ax.plot(np.arange(1, len(values) + 1), values)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Training step")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
                _save_fig(fig, out_dir, fname, save_pdf=save_pdf)

        for prefix, label in (("matrix", "Matrix"), ("weight", "Weights"), ("residual", "Residual")):
            plot_losses(prefix, label)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    save_bundle_pickle(out_dir, stored_bundle)

    mpc_path_or_dir = plot_cfg.get("mpc_path_or_dir", bundle.get("mpc_path_or_dir"))
    if include_baseline_compare and reward_fn is not None and mpc_path_or_dir is not None:
        compare_mpc_rl_from_dirs_core(
            rl_dir=out_dir,
            mpc_path_or_dir=mpc_path_or_dir,
            reward_fn=reward_fn,
            directory=out_dir,
            prefix_name=compare_prefix,
            compare_mode=compare_mode,
            start_episode=compare_start_episode,
            n_inputs=n_inputs,
            save_pdf=save_pdf,
            style_profile=style_profile,
        )

    return out_dir


def compare_mpc_rl_from_dirs_core(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    compare_mode="nominal",
    start_episode=1,
    n_inputs=2,
    save_pdf=False,
    style_profile="hybrid",
):
    _set_plot_style(style_profile=style_profile)
    rl_bundle = normalize_result_bundle(load_pickle(rl_dir))
    mpc_data = load_pickle(mpc_path_or_dir)
    mpc_bundle = normalize_external_bundle(mpc_data, rl_bundle)

    out_dir = create_output_dir(os.fspath(directory), prefix_name)

    rl_y = rl_bundle["y_line_full"]
    rl_u = rl_bundle["u_step_full"]
    mpc_y = mpc_bundle["y_line_full"]
    mpc_u = mpc_bundle["u_step_full"]

    delta_t = rl_bundle["delta_t"]
    time_in_sub_episodes = rl_bundle["time_in_sub_episodes"]
    y_sp_phys_full = ysp_scaled_dev_to_phys(
        rl_bundle["y_sp"],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        rl_u.shape[1],
    )
    nFE_sp = int(len(y_sp_phys_full))
    steps_rl = int(max(1, rl_y.shape[0] - 1))
    steps_mpc = int(max(1, mpc_y.shape[0] - 1))

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE_sp - 1)))
    max_start = int(max(0, min(nFE_sp, steps_rl, steps_mpc) - 1))
    start_step = int(min(start_step, max_start))

    W = int(min(nFE_sp - start_step, max(1, steps_rl - start_step), max(1, steps_mpc - start_step)))
    W = int(max(1, W))

    rl_y_seg = rl_y[start_step : start_step + W + 1, :]
    mpc_y_seg = mpc_y[start_step : start_step + W + 1, :]
    rl_u_seg = rl_u[start_step : start_step + W, :]
    mpc_u_seg = mpc_u[start_step : start_step + W, :]
    y_sp_phys = y_sp_phys_full[start_step : start_step + W, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    tail = int(min(max(20, time_in_sub_episodes), W))
    tail_start = max(0, W - tail)
    t_line_blk = np.linspace(0.0, tail * delta_t, tail + 1)
    t_step_blk = t_line_blk[:-1]

    # The "last episode" comparison must always mean:
    # final episode of the RL run vs final episode of the MPC run,
    # not the tail of the currently selected overlap window.
    last_episode_steps = int(min(max(1, time_in_sub_episodes), steps_rl, steps_mpc, nFE_sp))
    last_episode_steps = int(max(1, last_episode_steps))
    t_line_last = np.linspace(0.0, last_episode_steps * delta_t, last_episode_steps + 1)
    t_step_last = t_line_last[:-1]
    rl_y_last = rl_y[-(last_episode_steps + 1) :, :]
    mpc_y_last = mpc_y[-(last_episode_steps + 1) :, :]
    rl_u_last = rl_u[-last_episode_steps:, :]
    mpc_u_last = mpc_u[-last_episode_steps:, :]
    sp_last = y_sp_phys_full[-last_episode_steps:, :]

    fig, axs = plt.subplots(rl_y_seg.shape[1], 1, figsize=(8.6, 3.0 + 2.5 * max(1, rl_y_seg.shape[1] - 1)), sharex=True)
    if rl_y_seg.shape[1] == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, rl_y_seg[:, idx], label="RL")
        ax.plot(t_line, mpc_y_seg[:, idx], linestyle="--", label="MPC")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle=":", label="Setpoint")
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(rl_y_seg.shape[1], 1, figsize=(8.6, 3.0 + 2.5 * max(1, rl_y_seg.shape[1] - 1)), sharex=True)
    if rl_y_seg.shape[1] == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_last, rl_y_last[:, idx], label="RL")
        ax.plot(t_line_last, mpc_y_last[:, idx], linestyle="--", label="MPC")
        ax.step(t_step_last, sp_last[:, idx], where="post", linestyle=":", label="Setpoint")
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_outputs_last_episode", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_last, rl_u_last[:, idx], where="post", label="RL")
        ax.step(t_step_last, mpc_u_last[:, idx], where="post", linestyle="--", label="MPC")
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_inputs_last_episode", save_pdf=save_pdf)

    rl_rewards, _, _ = recompute_step_rewards(
        rl_bundle["y_line_full"],
        rl_bundle["u_step_full"],
        rl_bundle["y_sp"],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        reward_fn,
    )
    mpc_rewards, _, _ = recompute_step_rewards(
        mpc_bundle["y_line_full"],
        mpc_bundle["u_step_full"],
        rl_bundle["y_sp"],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        reward_fn,
    )

    n_ep_rl = len(rl_rewards) // time_in_sub_episodes
    n_ep_mpc = len(mpc_rewards) // time_in_sub_episodes
    avg_rl = rl_rewards[: n_ep_rl * time_in_sub_episodes].reshape(n_ep_rl, time_in_sub_episodes).mean(axis=1) if n_ep_rl else np.asarray([])
    avg_mpc = mpc_rewards[: n_ep_mpc * time_in_sub_episodes].reshape(n_ep_mpc, time_in_sub_episodes).mean(axis=1) if n_ep_mpc else np.asarray([])

    x_rl, y_rl = slice_avg_rewards(avg_rl, n_ep_rl, start_episode)
    x_mpc, y_mpc = slice_avg_rewards(avg_mpc, n_ep_mpc, start_episode)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_rl) > 0:
        ax.plot(x_rl, y_rl, "o-", label="RL")
    if compare_mode == "nominal" and len(y_mpc) > 0 and len(x_rl) > 0:
        ax.plot(x_rl, np.full(x_rl.shape, float(y_mpc[-1])), "s--", label="MPC")
    elif len(y_mpc) > 0:
        ax.plot(x_mpc, y_mpc, "s--", label="MPC")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "compare_rewards", save_pdf=save_pdf)

    save_bundle_pickle(
        out_dir,
        {
            "rl_dir": rl_dir,
            "mpc_path_or_dir": mpc_path_or_dir,
            "compare_mode": compare_mode,
            "avg_rewards_rl": avg_rl,
            "avg_rewards_mpc": avg_mpc,
        },
    )
    return out_dir
