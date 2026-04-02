import collections
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from utils.helpers import apply_min_max, reverse_min_max


def _set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120,
        }
    )


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
    bundle["data_min"] = np.asarray(bundle["data_min"], float)
    bundle["data_max"] = np.asarray(bundle["data_max"], float)
    bundle["nFE"] = int(bundle["nFE"])
    bundle["delta_t"] = float(bundle["delta_t"])
    bundle["time_in_sub_episodes"] = int(bundle["time_in_sub_episodes"])
    bundle["y_sp"] = np.asarray(bundle["y_sp"], float)
    bundle["rewards_step"] = None if bundle.get("rewards_step") is None else np.asarray(bundle["rewards_step"], float)
    bundle["delta_y_storage"] = None if bundle.get("delta_y_storage") is None else np.asarray(bundle["delta_y_storage"], float)
    bundle["delta_u_storage"] = None if bundle.get("delta_u_storage") is None else np.asarray(bundle["delta_u_storage"], float)
    bundle["horizon_trace"] = None if bundle.get("horizon_trace") is None else np.asarray(bundle["horizon_trace"], float)
    bundle["action_trace"] = None if bundle.get("action_trace") is None else np.asarray(bundle["action_trace"], int)
    bundle["yhat"] = None if bundle.get("yhat") is None else np.asarray(bundle["yhat"], float)
    bundle["xhatdhat"] = None if bundle.get("xhatdhat") is None else np.asarray(bundle["xhatdhat"], float)
    bundle["alpha_log"] = None if bundle.get("alpha_log") is None else np.asarray(bundle["alpha_log"], float).reshape(-1)
    bundle["delta_log"] = None if bundle.get("delta_log") is None else np.asarray(bundle["delta_log"], float)
    bundle["actor_losses"] = None if bundle.get("actor_losses") is None else np.asarray(bundle["actor_losses"], float).reshape(-1)
    bundle["critic_losses"] = None if bundle.get("critic_losses") is None else np.asarray(bundle["critic_losses"], float).reshape(-1)
    bundle["alpha_losses"] = None if bundle.get("alpha_losses") is None else np.asarray(bundle["alpha_losses"], float).reshape(-1)
    bundle["alphas"] = None if bundle.get("alphas") is None else np.asarray(bundle["alphas"], float).reshape(-1)
    bundle["low_coef"] = None if bundle.get("low_coef") is None else np.asarray(bundle["low_coef"], float).reshape(-1)
    bundle["high_coef"] = None if bundle.get("high_coef") is None else np.asarray(bundle["high_coef"], float).reshape(-1)

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
):
    _set_plot_style()
    rl_bundle = normalize_result_bundle(load_pickle(rl_dir))
    mpc_data = load_pickle(mpc_path_or_dir)
    mpc_bundle = normalize_external_bundle(mpc_data, rl_bundle)

    out_dir = create_output_dir(os.fspath(directory), prefix_name)

    rl_y = rl_bundle["y_line_full"]
    rl_u = rl_bundle["u_step_full"]
    mpc_y = mpc_bundle["y_line_full"]
    mpc_u = mpc_bundle["u_step_full"]

    nFE = min(rl_bundle["nFE"], mpc_bundle["nFE"])
    delta_t = rl_bundle["delta_t"]
    time_in_sub_episodes = rl_bundle["time_in_sub_episodes"]
    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(max(1, nFE - start_step))

    rl_y_seg = rl_y[start_step : start_step + W + 1, :]
    mpc_y_seg = mpc_y[start_step : start_step + W + 1, :]
    rl_u_seg = rl_u[start_step : start_step + W, :]
    mpc_u_seg = mpc_u[start_step : start_step + W, :]
    y_sp_phys = ysp_scaled_dev_to_phys(
        rl_bundle["y_sp"][start_step : start_step + W, :],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        rl_u.shape[1],
    )

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    tail = int(min(max(20, time_in_sub_episodes), W))
    tail_start = max(0, W - tail)
    t_line_blk = np.linspace(0.0, tail * delta_t, tail + 1)
    t_step_blk = t_line_blk[:-1]

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
        ax.plot(t_line_blk, rl_y_seg[tail_start : tail_start + tail + 1, idx], label="RL")
        ax.plot(t_line_blk, mpc_y_seg[tail_start : tail_start + tail + 1, idx], linestyle="--", label="MPC")
        ax.step(t_step_blk, y_sp_phys[tail_start : tail_start + tail, idx], where="post", linestyle=":", label="Setpoint")
        ax.set_ylabel(f"y{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, rl_u_seg[tail_start : tail_start + tail, idx], where="post", label="RL")
        ax.step(t_step_blk, mpc_u_seg[tail_start : tail_start + tail, idx], where="post", linestyle="--", label="MPC")
        ax.set_ylabel(f"u{idx + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_inputs_last_block", save_pdf=save_pdf)

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
