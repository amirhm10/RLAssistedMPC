def plot_rl_results_dqn(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_mpc,
    u_mpc,
    avg_rewards,
    data_min,
    data_max,
    reward_fn=None,
    horizon_trace=None,
    mpc_horizons=None,
    recipe_counts=True,
    start_episode=1,
    prefix_name="agent_result",
    directory=None,
    save_pdf=False
):
    import os
    import pickle
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import collections
    from utils.helpers import apply_min_max, reverse_min_max

    def set_paper_plot_style(font_size=14, label_size=16, tick_size=14, legend_size=13, lw=3.0, ms=6):
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.labelweight": "bold",
            "axes.titlesize": label_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120
        })

    def _make_axes_bold(ax):
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    def _save_fig(fig, out_dir, fname_base, dpi=300, save_pdf=False):
        fig.savefig(os.path.join(out_dir, fname_base + ".png"), dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(out_dir, fname_base + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs=2):
        y_sp_scaled_dev = np.asarray(y_sp_scaled_dev, float)
        data_min = np.asarray(data_min, float)
        data_max = np.asarray(data_max, float)
        y_ss_phys = np.asarray(steady_states["y_ss"], float)
        y_ss_scaled = apply_min_max(y_ss_phys, data_min[n_inputs:], data_max[n_inputs:])
        y_sp_scaled = y_sp_scaled_dev + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_scaled, data_min[n_inputs:], data_max[n_inputs:])
        return y_sp_phys

    def _parse_mpc_horizons(mpc_horizons):
        Hp0 = None
        Hc0 = None
        if mpc_horizons is None:
            return Hp0, Hc0
        if isinstance(mpc_horizons, dict):
            Hp0 = mpc_horizons.get("Hp", None)
            Hc0 = mpc_horizons.get("Hc", None)
            return Hp0, Hc0
        try:
            Hp0 = mpc_horizons[0]
            Hc0 = mpc_horizons[1]
        except Exception:
            Hp0, Hc0 = None, None
        return Hp0, Hc0

    def _slice_avg_rewards(avg, n_ep_total, start_episode):
        avg = np.asarray(avg, float)
        if len(avg) == n_ep_total + 1:
            avg = avg[1:]
        start_episode = int(max(1, start_episode))
        s = max(0, start_episode - 1)
        avg = avg[s:]
        x = np.arange(start_episode, start_episode + len(avg))
        return x, avg

    if directory is None:
        directory = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    nFE = int(nFE)
    delta_t = float(delta_t)
    time_in_sub_episodes = int(time_in_sub_episodes)
    start_episode = int(max(1, start_episode))

    y_sp = np.asarray(y_sp, float)
    y_mpc = np.asarray(y_mpc, float)
    u_mpc = np.asarray(u_mpc, float)

    if y_mpc.shape[0] == nFE:
        y_line_full = np.vstack([y_mpc, y_mpc[-1:, :]])
    else:
        y_line_full = y_mpc[:nFE + 1, :]

    n_inputs = 2
    y_sp_phys_full = ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs=n_inputs)

    start_step = (start_episode - 1) * time_in_sub_episodes
    start_step = int(min(max(0, start_step), max(0, nFE - 1)))

    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_mpc[start_step:, :]

    W = int(len(y_sp_phys))
    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]

    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s0 = W - last_steps
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]

    Hp0, Hc0 = _parse_mpc_horizons(mpc_horizons)

    set_paper_plot_style()

    # 1) Outputs vs Setpoints (full, from start_episode)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line, y_line[:, 0], "-")
    axs[0].step(t_step, y_sp_phys[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line, y_line[:, 1], "-")
    axs[1].step(t_step, y_sp_phys[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_outputs_full", save_pdf=save_pdf)

    # 2) Outputs vs Setpoints (last block of plotted segment)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line_blk, y_line[s0:s0 + last_steps + 1, 0], "-")
    axs[0].step(t_step_blk, y_sp_phys[s0:s0 + last_steps, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line_blk, y_line[s0:s0 + last_steps + 1, 1], "-")
    axs[1].step(t_step_blk, y_sp_phys[s0:s0 + last_steps, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_outputs_last_block", save_pdf=save_pdf)

    # 3) Inputs (full, from start_episode)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].step(t_step, u_line[:, 0], where="post")
    axs[0].set_ylabel(r"$Q_c$ (L/h)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    _make_axes_bold(axs[0])

    axs[1].step(t_step, u_line[:, 1], where="post")
    axs[1].set_ylabel(r"$Q_m$ (L/h)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_inputs_full", save_pdf=save_pdf)

    # 4) Avg reward per block (from start_episode)
    n_ep_total = int(nFE // time_in_sub_episodes)
    x_ep, y_ep = _slice_avg_rewards(avg_rewards, n_ep_total, start_episode)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-", lw=2.8, ms=5)
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_rl_rewards", save_pdf=save_pdf)

    # 5) Horizons (Hp, Hc) over time + highlight MPC horizons + histogram with MPC bar red
    if horizon_trace is not None:
        ht_full = np.asarray(horizon_trace, float)
        if ht_full.shape[0] >= nFE:
            ht_full = ht_full[:nFE, :]
        ht = ht_full[start_step:start_step + W, :]

        Hp = ht[:, 0].astype(int)
        Hc = ht[:, 1].astype(int)

        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].step(t_step, Hp, where="post")
        if Hp0 is not None:
            axs[0].axhline(float(Hp0), color="red", linestyle=":", linewidth=2.2)
        axs[0].set_ylabel("Hp")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])

        axs[1].step(t_step, Hc, where="post")
        if Hc0 is not None:
            axs[1].axhline(float(Hc0), color="red", linestyle=":", linewidth=2.2)
        axs[1].set_ylabel("Hc")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, out_dir, "HPHC", save_pdf=save_pdf)

        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].step(t_step_blk, Hp[s0:s0 + last_steps], where="post")
        if Hp0 is not None:
            axs[0].axhline(float(Hp0), color="red", linestyle=":", linewidth=2.2)
        axs[0].set_ylabel("Hp")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])

        axs[1].step(t_step_blk, Hc[s0:s0 + last_steps], where="post")
        if Hc0 is not None:
            axs[1].axhline(float(Hc0), color="red", linestyle=":", linewidth=2.2)
        axs[1].set_ylabel("Hc")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, out_dir, "HPHCLast", save_pdf=save_pdf)

        if recipe_counts:
            counts = collections.Counter(list(zip(Hp.tolist(), Hc.tolist())))
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top = items[:20] if len(items) > 0 else []

            key0 = None
            if Hp0 is not None and Hc0 is not None:
                key0 = (int(Hp0), int(Hc0))
                if key0 not in dict(top):
                    top.append((key0, counts.get(key0, 0)))

            labels = [f"Hp={k[0]},Hc={k[1]}" for k, _ in top]
            freqs = [v for _, v in top]

            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            if len(freqs) > 0:
                bars = ax.bar(np.arange(len(freqs)), freqs)
                if key0 is not None:
                    idx0 = None
                    for i, (k, _) in enumerate(top):
                        if k == key0:
                            idx0 = i
                            break
                    if idx0 is not None:
                        bars[idx0].set_color("red")
                        ax.text(idx0, freqs[idx0], "MPC", ha="center", va="bottom", fontweight="bold")
                ax.set_xticks(np.arange(len(freqs)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "HPHCcounts", save_pdf=save_pdf)

    # 6) recompute rewards step-by-step (full storage; plot from start_episode)
    rewards_step = None
    delta_y_storage = None
    delta_u_storage = None
    avg_rewards_recomputed = None

    if reward_fn is not None:
        y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
        ss_inputs = np.asarray(steady_states["ss_inputs"], float)
        ss_scaled_inputs = apply_min_max(ss_inputs, data_min[:n_inputs], data_max[:n_inputs])

        y_scaled_abs = apply_min_max(y_line_full, data_min[n_inputs:], data_max[n_inputs:])
        y_scaled_dev = y_scaled_abs - y_ss_scaled

        u_scaled_abs = apply_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])

        delta_y_storage = y_scaled_dev[1:, :] - y_sp[:nFE, :]

        delta_u_storage = np.zeros((nFE, n_inputs))
        delta_u_storage[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
        if nFE > 1:
            delta_u_storage[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

        rewards_step = np.zeros(nFE)
        for i in range(nFE):
            rewards_step[i] = reward_fn(delta_y_storage[i], delta_u_storage[i], y_sp_phys=y_sp_phys_full[i])

        n_ep = int(nFE // time_in_sub_episodes)
        if n_ep > 0:
            rr = rewards_step[:n_ep * time_in_sub_episodes].reshape(n_ep, time_in_sub_episodes)
            avg_rewards_recomputed = rr.mean(axis=1)

            x_ep2, y_ep2 = _slice_avg_rewards(avg_rewards_recomputed, n_ep, start_episode)
            fig, ax = plt.subplots(figsize=(7.8, 5.0))
            if len(y_ep2) > 0:
                ax.plot(x_ep2, y_ep2, "o-", lw=2.8, ms=5)
            ax.set_ylabel("Avg. Reward (recomputed)")
            ax.set_xlabel("Episode #")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_rl_rewards_recomputed", save_pdf=save_pdf)

    input_data = {
        "y_sp": y_sp,
        "y_sp_phys": y_sp_phys_full,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_mpc": y_line_full,
        "u_mpc": u_mpc,
        "y_rl": y_line_full,
        "u_rl": u_mpc,
        "avg_rewards": avg_rewards,
        "avg_rewards_recomputed": avg_rewards_recomputed,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "rewards_step": rewards_step,
        "data_min": data_min,
        "data_max": data_max,
        "horizon_trace": horizon_trace,
        "mpc_horizons": mpc_horizons,
        "start_episode_plotted": start_episode
    }

    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as f:
        pickle.dump(input_data, f)

    return out_dir


def _load_pickle(path):
    import os
    import glob
    import pickle

    if os.path.isdir(path):
        cand = os.path.join(path, "input_data.pkl")
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                return pickle.load(f)

        files = []
        files += glob.glob(os.path.join(path, "*.pickle"))
        files += glob.glob(os.path.join(path, "*.pkl"))
        if len(files) == 0:
            raise FileNotFoundError("No pickle files found in directory: " + path)

        files = sorted(files, key=lambda p: os.path.getmtime(p))
        with open(files[-1], "rb") as f:
            return pickle.load(f)

    with open(path, "rb") as f:
        return pickle.load(f)


def _recompute_mpc_rewards_from_storage(mpc_data, rl_data, reward_fn, n_inputs=2):
    import numpy as np
    from utils.helpers import apply_min_max, reverse_min_max

    def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs=2):
        y_sp_scaled_dev = np.asarray(y_sp_scaled_dev, float)
        data_min = np.asarray(data_min, float)
        data_max = np.asarray(data_max, float)
        y_ss_phys = np.asarray(steady_states["y_ss"], float)
        y_ss_scaled = apply_min_max(y_ss_phys, data_min[n_inputs:], data_max[n_inputs:])
        y_sp_scaled = y_sp_scaled_dev + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_scaled, data_min[n_inputs:], data_max[n_inputs:])
        return y_sp_phys

    y_sp_scaled_dev = np.asarray(rl_data["y_sp"], float)
    steady_states = rl_data["steady_states"]
    data_min = np.asarray(rl_data["data_min"], float)
    data_max = np.asarray(rl_data["data_max"], float)
    time_in_sub_episodes = int(rl_data["time_in_sub_episodes"])

    y_sp_phys = ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs=n_inputs)

    dy_mpc = None
    du_mpc = None

    if "delta_y_storage" in mpc_data and mpc_data["delta_y_storage"] is not None:
        dy_mpc = np.asarray(mpc_data["delta_y_storage"], float)

    if "delta_u_storage" in mpc_data and mpc_data["delta_u_storage"] is not None:
        du_mpc = np.asarray(mpc_data["delta_u_storage"], float)
    if du_mpc is None and "delat_u_storage" in mpc_data and mpc_data["delat_u_storage"] is not None:
        du_mpc = np.asarray(mpc_data["delat_u_storage"], float)

    if dy_mpc is None or du_mpc is None:
        y_mpc = mpc_data.get("y_mpc", mpc_data.get("y_rl", None))
        u_mpc = mpc_data.get("u_mpc", mpc_data.get("u_rl", None))
        if y_mpc is None or u_mpc is None:
            raise KeyError("MPC pickle must contain y_mpc/u_mpc or delta_y_storage/delta_u_storage.")

        y_mpc = np.asarray(y_mpc, float)
        u_mpc = np.asarray(u_mpc, float)

        nFE = int(min(len(y_sp_scaled_dev), len(u_mpc)))
        if y_mpc.shape[0] == nFE:
            y_line = np.vstack([y_mpc, y_mpc[-1:, :]])
        else:
            y_line = y_mpc[:nFE + 1, :]

        y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
        ss_inputs = np.asarray(steady_states["ss_inputs"], float)
        ss_scaled_inputs = apply_min_max(ss_inputs, data_min[:n_inputs], data_max[:n_inputs])

        y_scaled_abs = apply_min_max(y_line, data_min[n_inputs:], data_max[n_inputs:])
        y_scaled_dev = y_scaled_abs - y_ss_scaled

        u_scaled_abs = apply_min_max(u_mpc[:nFE, :], data_min[:n_inputs], data_max[:n_inputs])

        dy_mpc = y_scaled_dev[1:, :] - y_sp_scaled_dev[:nFE, :]

        du_mpc = np.zeros((nFE, n_inputs))
        du_mpc[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
        if nFE > 1:
            du_mpc[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

    n = int(min(len(dy_mpc), len(du_mpc), len(y_sp_phys)))
    r = np.zeros(n)
    for i in range(n):
        r[i] = reward_fn(dy_mpc[i], du_mpc[i], y_sp_phys=y_sp_phys[i])

    n_ep = int(n // time_in_sub_episodes)
    if n_ep <= 0:
        return r, np.asarray([], float)

    rr = r[:n_ep * time_in_sub_episodes].reshape(n_ep, time_in_sub_episodes)
    avg = rr.mean(axis=1)
    return r, avg


def compare_mpc_rl_nominal_from_dirs(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    start_episode=1,
    start_idx=None,
    n_inputs=2,
    save_pdf=False
):
    import os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    def set_paper_plot_style(font_size=14, label_size=16, tick_size=14, legend_size=13, lw=3.0, ms=6):
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.labelweight": "bold",
            "axes.titlesize": label_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120
        })

    def _make_axes_bold(ax):
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    def _save_fig(fig, fname):
        fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(out_dir, fname + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def _y_to_line(y):
        y = np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        return y

    def _slice_segment_line(y_line, s0, W):
        y_line = _y_to_line(y_line)
        if y_line.shape[0] >= s0 + W + 1:
            return y_line[s0:s0 + W + 1, :]
        if y_line.shape[0] >= s0 + W:
            seg = y_line[s0:s0 + W, :]
            return np.vstack([seg, seg[-1:, :]])
        seg = y_line[s0:, :]
        if seg.shape[0] == 0:
            return np.repeat(y_line[-1:, :], W + 1, axis=0)
        pad = np.repeat(seg[-1:, :], (W + 1 - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _slice_segment_step(a_step, s0, W):
        a_step = np.asarray(a_step, float)
        if a_step.ndim == 1:
            a_step = a_step[:, None]
        if a_step.shape[0] >= s0 + W:
            return a_step[s0:s0 + W, :]
        seg = a_step[s0:, :]
        if seg.shape[0] == 0:
            return np.repeat(a_step[-1:, :], W, axis=0)
        pad = np.repeat(seg[-1:, :], (W - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    rl_data = _load_pickle(rl_dir)
    mpc_data = _load_pickle(mpc_path_or_dir)

    y_rl = np.asarray(rl_data.get("y_rl", rl_data.get("y_mpc")), float)
    y_mpc = np.asarray(mpc_data.get("y_mpc", mpc_data.get("y_rl")), float)

    u_rl = rl_data.get("u_rl", rl_data.get("u_mpc", None))
    u_mpc = mpc_data.get("u_mpc", mpc_data.get("u_rl", None))

    delta_t = float(rl_data["delta_t"])
    time_in_sub_episodes = int(rl_data["time_in_sub_episodes"])

    y_sp_phys = rl_data.get("y_sp_phys", None)
    if y_sp_phys is None:
        raise KeyError("RL input_data.pkl must include y_sp_phys.")
    y_sp_phys = np.asarray(y_sp_phys, float)

    _, avg_mpc = _recompute_mpc_rewards_from_storage(mpc_data, rl_data, reward_fn, n_inputs=n_inputs)

    avg_rl = rl_data.get("avg_rewards_recomputed", None)
    if avg_rl is None:
        avg_rl = rl_data.get("avg_rewards", [])
    avg_rl = np.asarray(avg_rl, float)

    nFE_sp = int(len(y_sp_phys))
    start_episode = int(max(1, start_episode))
    if start_idx is None:
        s0 = (start_episode - 1) * time_in_sub_episodes
    else:
        s0 = int(start_idx) if start_idx >= 0 else max(0, nFE_sp + int(start_idx))
    s0 = int(min(max(0, s0), max(0, nFE_sp - 1)))

    steps_rl = int(max(0, y_rl.shape[0] - 1))
    steps_mpc = int(max(0, y_mpc.shape[0] - 1))
    W = int(min(nFE_sp - s0, max(1, steps_rl - s0), max(1, steps_mpc - s0)))
    W = int(max(1, W))

    sp_seg = y_sp_phys[s0:s0 + W, :]
    t_line = np.linspace(0.0, W * float(delta_t), W + 1)
    t_step = t_line[:-1]

    set_paper_plot_style()

    yrl_seg = _slice_segment_line(y_rl, s0, W)
    ympc_seg = _slice_segment_line(y_mpc, s0, W)

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line, yrl_seg[:, 0], "-")
    axs[0].plot(t_line, ympc_seg[:, 0], "--")
    axs[0].step(t_step, sp_seg[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line, yrl_seg[:, 1], "-")
    axs[1].plot(t_line, ympc_seg[:, 1], "--")
    axs[1].step(t_step, sp_seg[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, ts)
    os.makedirs(out_dir, exist_ok=True)
    _save_fig(fig, "outputs_compare_nominal")

    # LAST EPISODE compare (tail-based, no padding to RL length)
    ep_len = int(max(1, time_in_sub_episodes))
    steps_rl2 = int(max(1, y_rl.shape[0] - 1))
    steps_mpc2 = int(max(1, y_mpc.shape[0] - 1))
    W2 = int(min(ep_len, len(y_sp_phys), steps_rl2, steps_mpc2))
    W2 = int(max(1, W2))

    sp_last = y_sp_phys[-W2:, :]
    t_line2 = np.linspace(0.0, W2 * float(delta_t), W2 + 1)
    t_step2 = t_line2[:-1]

    yrl_last = _slice_segment_line(y_rl, int(max(0, steps_rl2 - W2)), W2)
    ympc_last = _slice_segment_line(y_mpc, int(max(0, steps_mpc2 - W2)), W2)

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line2, yrl_last[:, 0], "-")
    axs[0].plot(t_line2, ympc_last[:, 0], "--")
    axs[0].step(t_step2, sp_last[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line2, yrl_last[:, 1], "-")
    axs[1].plot(t_line2, ympc_last[:, 1], "--")
    axs[1].step(t_step2, sp_last[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, "outputs_compare_last_episode")

    if u_rl is not None and u_mpc is not None:
        u_rl = np.asarray(u_rl, float)
        u_mpc = np.asarray(u_mpc, float)
        url_last = _slice_segment_step(u_rl, int(max(0, u_rl.shape[0] - W2)), W2)
        umpc_last = _slice_segment_step(u_mpc, int(max(0, u_mpc.shape[0] - W2)), W2)

        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].plot(t_step2, url_last[:, 0], "-")
        axs[0].plot(t_step2, umpc_last[:, 0], "--")
        axs[0].set_ylabel(r"$Q_c$ (L/h)")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])

        axs[1].plot(t_step2, url_last[:, 1], "-")
        axs[1].plot(t_step2, umpc_last[:, 1], "--")
        axs[1].set_ylabel(r"$Q_m$ (L/h)")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, "inputs_compare_last_episode")

    # rewards compare (nominal MPC constant line)
    n_ep_total = int(nFE_sp // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0

    def _slice_avg(avg, n_ep_total, start_episode):
        avg = np.asarray(avg, float)
        if len(avg) == n_ep_total + 1:
            avg = avg[1:]
        start_episode = int(max(1, start_episode))
        s = max(0, start_episode - 1)
        avg = avg[s:]
        x = np.arange(start_episode, start_episode + len(avg))
        return x, avg

    x_rl, y_rl_avg = _slice_avg(avg_rl, n_ep_total, start_episode)
    _, y_mpc_avg = _slice_avg(avg_mpc, n_ep_total, start_episode)

    if len(y_rl_avg) > 0:
        mpc_const = float(y_mpc_avg[-1]) if len(y_mpc_avg) > 0 else 0.0
        y_mpc_line = np.full(len(y_rl_avg), mpc_const, dtype=float)

        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.plot(x_rl, y_rl_avg, "o-", lw=2.8, ms=5)
        ax.plot(x_rl, y_mpc_line, "s--", lw=2.8, ms=5)
        ax.set_ylabel("Avg. Reward")
        ax.set_xlabel("Episode #")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        _make_axes_bold(ax)
        _save_fig(fig, "reward_compare_nominal")

    return out_dir


def compare_mpc_rl_disturb_from_dirs(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    start_episode=1,
    start_idx=None,
    n_inputs=2,
    save_pdf=False
):
    import os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    def set_paper_plot_style(font_size=14, label_size=16, tick_size=14, legend_size=13, lw=3.0, ms=6):
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.labelweight": "bold",
            "axes.titlesize": label_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120
        })

    def _make_axes_bold(ax):
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    def _save_fig(fig, fname):
        fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(out_dir, fname + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def _y_to_line(y):
        y = np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        return y

    def _slice_segment_line(y_line, s0, W):
        y_line = _y_to_line(y_line)
        if y_line.shape[0] >= s0 + W + 1:
            return y_line[s0:s0 + W + 1, :]
        if y_line.shape[0] >= s0 + W:
            seg = y_line[s0:s0 + W, :]
            return np.vstack([seg, seg[-1:, :]])
        seg = y_line[s0:, :]
        if seg.shape[0] == 0:
            return np.repeat(y_line[-1:, :], W + 1, axis=0)
        pad = np.repeat(seg[-1:, :], (W + 1 - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _slice_segment_step(a_step, s0, W):
        a_step = np.asarray(a_step, float)
        if a_step.ndim == 1:
            a_step = a_step[:, None]
        if a_step.shape[0] >= s0 + W:
            return a_step[s0:s0 + W, :]
        seg = a_step[s0:, :]
        if seg.shape[0] == 0:
            return np.repeat(a_step[-1:, :], W, axis=0)
        pad = np.repeat(seg[-1:, :], (W - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    rl_data = _load_pickle(rl_dir)
    mpc_data = _load_pickle(mpc_path_or_dir)

    y_rl = np.asarray(rl_data.get("y_rl", rl_data.get("y_mpc")), float)
    y_mpc = np.asarray(mpc_data.get("y_mpc", mpc_data.get("y_rl")), float)

    u_rl = rl_data.get("u_rl", rl_data.get("u_mpc", None))
    u_mpc = mpc_data.get("u_mpc", mpc_data.get("u_rl", None))

    delta_t = float(rl_data["delta_t"])
    time_in_sub_episodes = int(rl_data["time_in_sub_episodes"])

    y_sp_phys = rl_data.get("y_sp_phys", None)
    if y_sp_phys is None:
        raise KeyError("RL input_data.pkl must include y_sp_phys.")
    y_sp_phys = np.asarray(y_sp_phys, float)

    _, avg_mpc = _recompute_mpc_rewards_from_storage(mpc_data, rl_data, reward_fn, n_inputs=n_inputs)

    avg_rl = rl_data.get("avg_rewards_recomputed", None)
    if avg_rl is None:
        avg_rl = rl_data.get("avg_rewards", [])
    avg_rl = np.asarray(avg_rl, float)

    nFE_sp = int(len(y_sp_phys))
    start_episode = int(max(1, start_episode))
    if start_idx is None:
        s0 = (start_episode - 1) * time_in_sub_episodes
    else:
        s0 = int(start_idx) if start_idx >= 0 else max(0, nFE_sp + int(start_idx))
    s0 = int(min(max(0, s0), max(0, nFE_sp - 1)))

    steps_rl = int(max(0, y_rl.shape[0] - 1))
    steps_mpc = int(max(0, y_mpc.shape[0] - 1))
    W = int(min(nFE_sp - s0, max(1, steps_rl - s0), max(1, steps_mpc - s0)))
    W = int(max(1, W))

    sp_seg = y_sp_phys[s0:s0 + W, :]
    t_line = np.linspace(0.0, W * float(delta_t), W + 1)
    t_step = t_line[:-1]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, ts)
    os.makedirs(out_dir, exist_ok=True)

    set_paper_plot_style()

    yrl_seg = _slice_segment_line(y_rl, s0, W)
    ympc_seg = _slice_segment_line(y_mpc, s0, W)

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line, yrl_seg[:, 0], "-")
    axs[0].plot(t_line, ympc_seg[:, 0], "--")
    axs[0].step(t_step, sp_seg[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line, yrl_seg[:, 1], "-")
    axs[1].plot(t_line, ympc_seg[:, 1], "--")
    axs[1].step(t_step, sp_seg[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, "outputs_compare_dist")

    # LAST EPISODE compare (tail-based, no padding to RL length)
    ep_len = int(max(1, time_in_sub_episodes))
    steps_rl2 = int(max(1, y_rl.shape[0] - 1))
    steps_mpc2 = int(max(1, y_mpc.shape[0] - 1))
    W2 = int(min(ep_len, len(y_sp_phys), steps_rl2, steps_mpc2))
    W2 = int(max(1, W2))

    sp_last = y_sp_phys[-W2:, :]
    t_line2 = np.linspace(0.0, W2 * float(delta_t), W2 + 1)
    t_step2 = t_line2[:-1]

    yrl_last = _slice_segment_line(y_rl, int(max(0, steps_rl2 - W2)), W2)
    ympc_last = _slice_segment_line(y_mpc, int(max(0, steps_mpc2 - W2)), W2)

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line2, yrl_last[:, 0], "-")
    axs[0].plot(t_line2, ympc_last[:, 0], "--")
    axs[0].step(t_step2, sp_last[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line2, yrl_last[:, 1], "-")
    axs[1].plot(t_line2, ympc_last[:, 1], "--")
    axs[1].step(t_step2, sp_last[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, "outputs_compare_last_episode")

    if u_rl is not None and u_mpc is not None:
        u_rl = np.asarray(u_rl, float)
        u_mpc = np.asarray(u_mpc, float)
        url_last = _slice_segment_step(u_rl, int(max(0, u_rl.shape[0] - W2)), W2)
        umpc_last = _slice_segment_step(u_mpc, int(max(0, u_mpc.shape[0] - W2)), W2)

        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].plot(t_step2, url_last[:, 0], "-")
        axs[0].plot(t_step2, umpc_last[:, 0], "--")
        axs[0].set_ylabel(r"$Q_c$ (L/h)")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])

        axs[1].plot(t_step2, url_last[:, 1], "-")
        axs[1].plot(t_step2, umpc_last[:, 1], "--")
        axs[1].set_ylabel(r"$Q_m$ (L/h)")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, "inputs_compare_last_episode")

    # reward compare (disturb: both curves)
    n_ep_total = int(nFE_sp // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0

    def _slice_avg(avg, n_ep_total, start_episode):
        avg = np.asarray(avg, float)
        if len(avg) == n_ep_total + 1:
            avg = avg[1:]
        start_episode = int(max(1, start_episode))
        s = max(0, start_episode - 1)
        avg = avg[s:]
        x = np.arange(start_episode, start_episode + len(avg))
        return x, avg

    x_rl, y_rl_avg = _slice_avg(avg_rl, n_ep_total, start_episode)
    _, y_mpc_avg = _slice_avg(avg_mpc, n_ep_total, start_episode)

    n = int(min(len(y_rl_avg), len(y_mpc_avg)))
    if n > 0:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.plot(x_rl[:n], y_rl_avg[:n], "o-", lw=2.8, ms=5)
        ax.plot(x_rl[:n], y_mpc_avg[:n], "s--", lw=2.8, ms=5)
        ax.set_ylabel("Avg. Reward")
        ax.set_xlabel("Episode #")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        _make_axes_bold(ax)
        _save_fig(fig, "reward_compare_dist")

    return out_dir


def plot_rl_results_td3_weights_dqnstyle(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_rl,
    u_rl,
    avg_rewards,
    data_min,
    data_max,
    mult_log=None,
    low_coef=None,
    high_coef=None,
    reward_fn=None,
    start_episode=1,
    prefix_name="td3_weights",
    directory=None,
    save_pdf=False
):
    import os
    import pickle
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from utils.helpers import apply_min_max, reverse_min_max

    def set_paper_plot_style(font_size=14, label_size=16, tick_size=14, legend_size=13, lw=3.0, ms=6):
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.labelweight": "bold",
            "axes.titlesize": label_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120
        })

    def _make_axes_bold(ax):
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    def _save_fig(fig, out_dir, fname_base, dpi=300, save_pdf=False):
        fig.savefig(os.path.join(out_dir, fname_base + ".png"), dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(out_dir, fname_base + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs=2):
        y_sp_scaled_dev = np.asarray(y_sp_scaled_dev, float)
        data_min = np.asarray(data_min, float)
        data_max = np.asarray(data_max, float)
        y_ss_phys = np.asarray(steady_states["y_ss"], float)
        y_ss_scaled = apply_min_max(y_ss_phys, data_min[n_inputs:], data_max[n_inputs:])
        y_sp_scaled = y_sp_scaled_dev + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_scaled, data_min[n_inputs:], data_max[n_inputs:])
        return y_sp_phys

    def _slice_avg_rewards(avg, n_ep_total, start_episode):
        avg = np.asarray(avg, float)
        if len(avg) == n_ep_total + 1:
            avg = avg[1:]
        start_episode = int(max(1, start_episode))
        s = max(0, start_episode - 1)
        avg = avg[s:]
        x = np.arange(start_episode, start_episode + len(avg))
        return x, avg

    def _to_2d(a):
        a = np.asarray(a, float)
        if a.ndim == 1:
            a = a[:, None]
        return a

    def _slice_line(y_line, start_step, W):
        y_line = _to_2d(y_line)
        if y_line.shape[0] >= start_step + W + 1:
            return y_line[start_step:start_step + W + 1, :]
        if y_line.shape[0] >= start_step + W:
            seg = y_line[start_step:start_step + W, :]
            return np.vstack([seg, seg[-1:, :]])
        seg = y_line[start_step:, :]
        if seg.shape[0] == 0:
            return np.repeat(y_line[-1:, :], W + 1, axis=0)
        pad = np.repeat(seg[-1:, :], (W + 1 - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _slice_step(a_step, start_step, W):
        a_step = _to_2d(a_step)
        if a_step.shape[0] >= start_step + W:
            return a_step[start_step:start_step + W, :]
        seg = a_step[start_step:, :]
        if seg.shape[0] == 0:
            return np.repeat(a_step[-1:, :], W, axis=0)
        pad = np.repeat(seg[-1:, :], (W - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _prep_weights(mult_log, nFE):
        if mult_log is None:
            return None
        m = np.asarray(mult_log, float)
        if m.ndim == 1:
            m = m[:, None]
        if m.shape[0] > nFE:
            m = m[:nFE, :]
        if m.shape[1] != 4:
            raise ValueError("mult_log must have shape [nFE, 4] for [Q1,Q2,R1,R2].")
        return m

    def _prep_bounds(low, high):
        if low is None or high is None:
            return None, None
        lo = np.asarray(low, float).reshape(-1)
        hi = np.asarray(high, float).reshape(-1)
        if lo.size != 4 or hi.size != 4:
            raise ValueError("low_coef/high_coef must each have length 4 for [Q1,Q2,R1,R2] multipliers.")
        return lo, hi

    def _plot_weights(t_step, m_seg, lo, hi, out_dir, fname, save_pdf):
        names = ["Q1", "Q2", "R1", "R2"]
        fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for k in range(4):
            ax = axs[k]
            if lo is not None and hi is not None:
                ax.fill_between(t_step, float(lo[k]), float(hi[k]), alpha=0.15, step="post")
                ax.hlines([float(lo[k]), float(hi[k])], t_step[0], t_step[-1], linestyles="--", linewidth=1.0)
            ax.step(t_step, m_seg[:, k], where="post")
            ax.set_ylabel(names[k])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, fname, save_pdf=save_pdf)

    if directory is None:
        directory = os.getcwd()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, ts)
    os.makedirs(out_dir, exist_ok=True)

    nFE = int(nFE)
    delta_t = float(delta_t)
    time_in_sub_episodes = int(time_in_sub_episodes)
    start_episode = int(max(1, start_episode))

    y_sp = np.asarray(y_sp, float)
    y_rl = np.asarray(y_rl, float)
    u_rl = np.asarray(u_rl, float)

    n_inputs = 2

    if y_rl.shape[0] == nFE:
        y_line_full = np.vstack([y_rl, y_rl[-1:, :]])
    else:
        y_line_full = y_rl[:nFE + 1, :]

    y_sp_phys_full = ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs=n_inputs)

    start_step = (start_episode - 1) * time_in_sub_episodes
    start_step = int(min(max(0, start_step), max(0, nFE - 1)))

    W = int(max(1, len(y_sp_phys_full) - start_step))
    y_line = _slice_line(y_line_full, start_step, W)
    u_line = _slice_step(u_rl, start_step, W)
    sp_line = _slice_step(y_sp_phys_full, start_step, W)

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]

    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s0 = W - last_steps
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]

    set_paper_plot_style()

    # 1) Outputs vs setpoints (full from start_episode)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line, y_line[:, 0], "-")
    axs[0].step(t_step, sp_line[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line, y_line[:, 1], "-")
    axs[1].step(t_step, sp_line[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_outputs_full", save_pdf=save_pdf)

    # 2) Outputs vs setpoints (last block)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line_blk, y_line[s0:s0 + last_steps + 1, 0], "-")
    axs[0].step(t_step_blk, sp_line[s0:s0 + last_steps, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line_blk, y_line[s0:s0 + last_steps + 1, 1], "-")
    axs[1].step(t_step_blk, sp_line[s0:s0 + last_steps, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_outputs_last_block", save_pdf=save_pdf)

    # 3) Inputs (full from start_episode)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].step(t_step, u_line[:, 0], where="post")
    axs[0].set_ylabel(r"$Q_c$ (L/h)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    _make_axes_bold(axs[0])

    axs[1].step(t_step, u_line[:, 1], where="post")
    axs[1].set_ylabel(r"$Q_m$ (L/h)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_inputs_full", save_pdf=save_pdf)

    # 4) Avg reward per block (from start_episode)
    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = _slice_avg_rewards(avg_rewards, n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-", lw=2.8, ms=5)
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_rl_rewards", save_pdf=save_pdf)

    # 5) Weight multipliers (full + last block)
    m_full = _prep_weights(mult_log, nFE)
    lo, hi = _prep_bounds(low_coef, high_coef)

    if m_full is not None:
        m_seg = _slice_step(m_full, start_step, W)
        _plot_weights(t_step, m_seg, lo, hi, out_dir, "fig_weights_full", save_pdf)

        m_blk = m_seg[s0:s0 + last_steps, :]
        _plot_weights(t_step_blk, m_blk, lo, hi, out_dir, "fig_weights_last_block", save_pdf)

    # 6) Optional: recompute rewards step-by-step (same as DQN logic)
    rewards_step = None
    delta_y_storage = None
    delta_u_storage = None
    avg_rewards_recomputed = None

    if reward_fn is not None:
        y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
        ss_inputs = np.asarray(steady_states["ss_inputs"], float)
        ss_scaled_inputs = apply_min_max(ss_inputs, data_min[:n_inputs], data_max[:n_inputs])

        y_scaled_abs = apply_min_max(y_line_full, data_min[n_inputs:], data_max[n_inputs:])
        y_scaled_dev = y_scaled_abs - y_ss_scaled

        u_scaled_abs = apply_min_max(u_rl[:nFE, :], data_min[:n_inputs], data_max[:n_inputs])

        delta_y_storage = y_scaled_dev[1:, :] - y_sp[:nFE, :]

        delta_u_storage = np.zeros((nFE, n_inputs))
        delta_u_storage[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
        if nFE > 1:
            delta_u_storage[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

        rewards_step = np.zeros(nFE)
        for i in range(nFE):
            rewards_step[i] = reward_fn(delta_y_storage[i], delta_u_storage[i], y_sp_phys=y_sp_phys_full[i])

        n_ep = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
        if n_ep > 0:
            rr = rewards_step[:n_ep * time_in_sub_episodes].reshape(n_ep, time_in_sub_episodes)
            avg_rewards_recomputed = rr.mean(axis=1)

            x_ep2, y_ep2 = _slice_avg_rewards(avg_rewards_recomputed, n_ep, start_episode)
            fig, ax = plt.subplots(figsize=(7.8, 5.0))
            if len(y_ep2) > 0:
                ax.plot(x_ep2, y_ep2, "o-", lw=2.8, ms=5)
            ax.set_ylabel("Avg. Reward (recomputed)")
            ax.set_xlabel("Episode #")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_rl_rewards_recomputed", save_pdf=save_pdf)

    input_data = {
        "y_sp": y_sp,
        "y_sp_phys": y_sp_phys_full,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_rl": y_line_full,
        "u_rl": u_rl,
        "y_mpc": y_line_full,
        "u_mpc": u_rl,
        "avg_rewards": avg_rewards,
        "avg_rewards_recomputed": avg_rewards_recomputed,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "rewards_step": rewards_step,
        "data_min": data_min,
        "data_max": data_max,
        "start_episode_plotted": start_episode,
        "mult_log": None if m_full is None else m_full,
        "low_coef": None if lo is None else lo,
        "high_coef": None if hi is None else hi
    }

    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as f:
        pickle.dump(input_data, f)

    return out_dir


def plot_rl_results_td3_multipliers_dqnstyle(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_rl,
    u_rl,
    avg_rewards,
    data_min,
    data_max,
    reward_fn=None,
    coef_alpha=None,
    coef_delta=None,
    low_coef=None,
    high_coef=None,
    start_episode=1,
    prefix_name="agent_result",
    directory=None,
    save_pdf=False
):
    import os
    import pickle
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from utils.helpers import apply_min_max, reverse_min_max

    def set_paper_plot_style(font_size=14, label_size=16, tick_size=14, legend_size=13, lw=3.0, ms=6):
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.labelweight": "bold",
            "axes.titlesize": label_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120
        })

    def _make_axes_bold(ax):
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    def _save_fig(fig, out_dir, fname_base, dpi=300, save_pdf=False):
        fig.savefig(os.path.join(out_dir, fname_base + ".png"), dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(out_dir, fname_base + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs=2):
        y_sp_scaled_dev = np.asarray(y_sp_scaled_dev, float)
        data_min = np.asarray(data_min, float)
        data_max = np.asarray(data_max, float)
        y_ss_phys = np.asarray(steady_states["y_ss"], float)
        y_ss_scaled = apply_min_max(y_ss_phys, data_min[n_inputs:], data_max[n_inputs:])
        y_sp_scaled = y_sp_scaled_dev + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_scaled, data_min[n_inputs:], data_max[n_inputs:])
        return y_sp_phys

    def _slice_avg_rewards(avg, n_ep_total, start_episode):
        avg = np.asarray(avg, float)
        if len(avg) == n_ep_total + 1:
            avg = avg[1:]
        start_episode = int(max(1, start_episode))
        s = max(0, start_episode - 1)
        avg = avg[s:]
        x = np.arange(start_episode, start_episode + len(avg))
        return x, avg

    def _to_2d(a):
        a = np.asarray(a, float)
        if a.ndim == 1:
            a = a[:, None]
        return a

    def _slice_step(a_step, start_step, W):
        a_step = _to_2d(a_step)
        if a_step.shape[0] >= start_step + W:
            return a_step[start_step:start_step + W, :]
        seg = a_step[start_step:, :]
        if seg.shape[0] == 0:
            return np.repeat(a_step[-1:, :], W, axis=0)
        pad = np.repeat(seg[-1:, :], (W - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _slice_line(y_line, start_step, W):
        y_line = _to_2d(y_line)
        if y_line.shape[0] >= start_step + W + 1:
            return y_line[start_step:start_step + W + 1, :]
        if y_line.shape[0] >= start_step + W:
            seg = y_line[start_step:start_step + W, :]
            return np.vstack([seg, seg[-1:, :]])
        seg = y_line[start_step:, :]
        if seg.shape[0] == 0:
            return np.repeat(y_line[-1:, :], W + 1, axis=0)
        pad = np.repeat(seg[-1:, :], (W + 1 - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _prep_coeff_alpha(a, nFE):
        if a is None:
            return None
        a = np.asarray(a, float).reshape(-1)
        if len(a) >= nFE + 1:
            a = a[:nFE]
        if len(a) > nFE:
            a = a[:nFE]
        return a

    def _prep_coeff_delta(d, nFE, nu):
        if d is None:
            return None
        d = np.asarray(d, float)
        if d.ndim == 1:
            d = d[:, None]
        if d.shape[0] >= nFE + 1:
            d = d[:nFE, :]
        if d.shape[0] > nFE:
            d = d[:nFE, :]
        if d.shape[1] != nu and d.shape[1] == 1 and nu > 1:
            d = np.repeat(d, nu, axis=1)
        return d

    def _plot_multipliers(t_step, alpha_seg, delta_seg, low, high, out_dir, fname, save_pdf):
        nu_local = 0 if delta_seg is None else int(delta_seg.shape[1])
        rows = (1 if alpha_seg is not None else 0) + nu_local
        if rows == 0:
            return

        fig, axs = plt.subplots(rows, 1, figsize=(8.2, max(5.0, 2.2 * rows)), sharex=True)
        if rows == 1:
            axs = [axs]

        have_bounds = (low is not None) and (high is not None)
        r = 0

        if alpha_seg is not None:
            ax = axs[r]
            if have_bounds:
                ax.fill_between(t_step, float(low[0]), float(high[0]), alpha=0.15, step="post")
                ax.hlines([float(low[0]), float(high[0])], t_step[0], t_step[-1], linestyles="--", linewidth=1.0)
            ax.step(t_step, alpha_seg, where="post")
            ax.set_ylabel(r"$\alpha$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            r += 1

        if delta_seg is not None:
            for j in range(delta_seg.shape[1]):
                ax = axs[r]
                if have_bounds and low.size >= 1 + nu_local:
                    lo = float(low[1 + j])
                    hi = float(high[1 + j])
                    ax.fill_between(t_step, lo, hi, alpha=0.15, step="post")
                    ax.hlines([lo, hi], t_step[0], t_step[-1], linestyles="--", linewidth=1.0)
                ax.step(t_step, delta_seg[:, j], where="post")
                ax.set_ylabel(r"$\delta_{" + str(j + 1) + r"}$")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
                r += 1

        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, fname, save_pdf=save_pdf)

    if directory is None:
        directory = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    nFE = int(nFE)
    delta_t = float(delta_t)
    time_in_sub_episodes = int(time_in_sub_episodes)
    start_episode = int(max(1, start_episode))

    y_sp = np.asarray(y_sp, float)
    y_rl = np.asarray(y_rl, float)
    u_rl = np.asarray(u_rl, float)

    nu = int(u_rl.shape[1])
    n_inputs = 2

    if y_rl.shape[0] == nFE:
        y_line_full = np.vstack([y_rl, y_rl[-1:, :]])
    else:
        y_line_full = y_rl[:nFE + 1, :]

    y_sp_phys_full = ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs=n_inputs)

    start_step = (start_episode - 1) * time_in_sub_episodes
    start_step = int(min(max(0, start_step), max(0, nFE - 1)))

    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_rl[start_step:, :]

    W = int(len(y_sp_phys))
    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]

    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s0 = W - last_steps
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]

    set_paper_plot_style()

    # 1) Outputs vs setpoints (full, from start_episode)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line, y_line[:, 0], "-")
    axs[0].step(t_step, y_sp_phys[:, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line, y_line[:, 1], "-")
    axs[1].step(t_step, y_sp_phys[:, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_outputs_full", save_pdf=save_pdf)

    # 2) Outputs vs setpoints (last block of plotted segment)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].plot(t_line_blk, y_line[s0:s0 + last_steps + 1, 0], "-")
    axs[0].step(t_step_blk, y_sp_phys[s0:s0 + last_steps, 0], where="post", linestyle="--")
    axs[0].set_ylabel(r"$\mathbf{\eta}$ (L/g)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[0].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[0])

    axs[1].plot(t_line_blk, y_line[s0:s0 + last_steps + 1, 1], "-")
    axs[1].step(t_step_blk, y_sp_phys[s0:s0 + last_steps, 1], where="post", linestyle="--")
    axs[1].set_ylabel(r"$\mathbf{T}$ (K)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(mtick.MaxNLocator(6))
    axs[1].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_outputs_last_block", save_pdf=save_pdf)

    # 3) Inputs (full, from start_episode)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axs[0].step(t_step, u_line[:, 0], where="post")
    axs[0].set_ylabel(r"$Q_c$ (L/h)")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    _make_axes_bold(axs[0])

    axs[1].step(t_step, u_line[:, 1], where="post")
    axs[1].set_ylabel(r"$Q_m$ (L/h)")
    axs[1].set_xlabel("Time (h)")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_rl_inputs_full", save_pdf=save_pdf)

    # 4) Avg reward per block (from start_episode)
    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = _slice_avg_rewards(avg_rewards, n_ep_total, start_episode)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-", lw=2.8, ms=5)
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_rl_rewards", save_pdf=save_pdf)

    # 5) Multipliers (full + last block), respecting start_episode
    alpha_full = _prep_coeff_alpha(coef_alpha, nFE)
    delta_full = _prep_coeff_delta(coef_delta, nFE, nu)

    low = None
    high = None
    if low_coef is not None and high_coef is not None:
        low = np.asarray(low_coef, float).reshape(-1)
        high = np.asarray(high_coef, float).reshape(-1)
        if low.size != high.size:
            low = None
            high = None
        if low is not None and low.size not in [1, 1 + nu]:
            low = None
            high = None

    if alpha_full is not None or delta_full is not None:
        a_seg = None if alpha_full is None else _slice_step(alpha_full, start_step, W)[:, 0]
        d_seg = None if delta_full is None else _slice_step(delta_full, start_step, W)
        _plot_multipliers(t_step, a_seg, d_seg, low, high, out_dir, "fig_multipliers_full", save_pdf)

        a_blk = None if alpha_full is None else a_seg[s0:s0 + last_steps]
        d_blk = None if d_seg is None else d_seg[s0:s0 + last_steps, :]
        _plot_multipliers(t_step_blk, a_blk, d_blk, low, high, out_dir, "fig_multipliers_last_block", save_pdf)

    # 6) Optional: recompute rewards step-by-step (same logic as DQN version)
    rewards_step = None
    delta_y_storage = None
    delta_u_storage = None
    avg_rewards_recomputed = None

    if reward_fn is not None:
        y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
        ss_inputs = np.asarray(steady_states["ss_inputs"], float)
        ss_scaled_inputs = apply_min_max(ss_inputs, data_min[:n_inputs], data_max[:n_inputs])

        y_scaled_abs = apply_min_max(y_line_full, data_min[n_inputs:], data_max[n_inputs:])
        y_scaled_dev = y_scaled_abs - y_ss_scaled

        u_scaled_abs = apply_min_max(u_rl, data_min[:n_inputs], data_max[:n_inputs])

        delta_y_storage = y_scaled_dev[1:, :] - y_sp[:nFE, :]

        delta_u_storage = np.zeros((nFE, n_inputs))
        delta_u_storage[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
        if nFE > 1:
            delta_u_storage[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

        y_sp_phys_full2 = ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs=n_inputs)

        rewards_step = np.zeros(nFE)
        for i in range(nFE):
            rewards_step[i] = reward_fn(delta_y_storage[i], delta_u_storage[i], y_sp_phys=y_sp_phys_full2[i])

        n_ep = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
        if n_ep > 0:
            rr = rewards_step[:n_ep * time_in_sub_episodes].reshape(n_ep, time_in_sub_episodes)
            avg_rewards_recomputed = rr.mean(axis=1)

            x_ep2, y_ep2 = _slice_avg_rewards(avg_rewards_recomputed, n_ep, start_episode)
            fig, ax = plt.subplots(figsize=(7.8, 5.0))
            if len(y_ep2) > 0:
                ax.plot(x_ep2, y_ep2, "o-", lw=2.8, ms=5)
            ax.set_ylabel("Avg. Reward (recomputed)")
            ax.set_xlabel("Episode #")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_rl_rewards_recomputed", save_pdf=save_pdf)

    input_data = {
        "y_sp": y_sp,
        "y_sp_phys": y_sp_phys_full,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_rl": y_line_full,
        "u_rl": u_rl,
        "y_mpc": y_line_full,
        "u_mpc": u_rl,
        "avg_rewards": avg_rewards,
        "avg_rewards_recomputed": avg_rewards_recomputed,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "rewards_step": rewards_step,
        "data_min": data_min,
        "data_max": data_max,
        "start_episode_plotted": start_episode,
        "multiplier_coefs_alpha": None if coef_alpha is None else np.asarray(coef_alpha, float),
        "multiplier_coefs_delta": None if coef_delta is None else np.asarray(coef_delta, float),
        "low_coef": None if low_coef is None else np.asarray(low_coef, float),
        "high_coef": None if high_coef is None else np.asarray(high_coef, float)
    }

    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as f:
        pickle.dump(input_data, f)

    return out_dir

def plot_rl_results_multiagent_dqnstyle(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_rl,
    u_rl,
    avg_rewards,
    data_min,
    data_max,
    reward_fn=None,
    horizon_trace=None,
    mpc_horizons=None,
    recipe_counts=True,
    horizon_recipes=None,
    alpha_log=None,
    delta_log=None,
    weight_log=None,
    model_low=None,
    model_high=None,
    weights_low=None,
    weights_high=None,
    mpc_path_or_dir=None,
    mpc_reward_mode="auto",
    start_episode=1,
    prefix_name="multi_agent_run",
    directory=None,
    save_pdf=False
):
    import os
    import glob
    import pickle
    import collections
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from utils.helpers import apply_min_max, reverse_min_max

    def set_paper_plot_style(font_size=14, label_size=16, tick_size=14, legend_size=13, lw=3.0, ms=6):
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.labelweight": "bold",
            "axes.titlesize": label_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "figure.dpi": 120
        })

    def _make_axes_bold(ax):
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")

    def _save_fig(fig, out_dir, fname_base, dpi=300, save_pdf=False):
        fig.savefig(os.path.join(out_dir, fname_base + ".png"), dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(out_dir, fname_base + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def _to_2d(a):
        a = np.asarray(a, float)
        if a.ndim == 1:
            a = a[:, None]
        return a

    def _y_to_line(y, nFE_local):
        y = _to_2d(y)
        if y.shape[0] == nFE_local:
            return np.vstack([y, y[-1:, :]])
        if y.shape[0] >= nFE_local + 1:
            return y[:nFE_local + 1, :]
        if y.shape[0] == 0:
            return np.zeros((nFE_local + 1, 1))
        pad = np.repeat(y[-1:, :], (nFE_local + 1 - y.shape[0]), axis=0)
        return np.vstack([y, pad])

    def _slice_segment_line(y_line, s0, W):
        y_line = _to_2d(y_line)
        if y_line.shape[0] >= s0 + W + 1:
            return y_line[s0:s0 + W + 1, :]
        if y_line.shape[0] >= s0 + W:
            seg = y_line[s0:s0 + W, :]
            return np.vstack([seg, seg[-1:, :]])
        seg = y_line[s0:, :]
        if seg.shape[0] == 0:
            return np.repeat(y_line[-1:, :], W + 1, axis=0)
        pad = np.repeat(seg[-1:, :], (W + 1 - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _slice_segment_step(a_step, s0, W):
        a_step = _to_2d(a_step)
        if a_step.shape[0] >= s0 + W:
            return a_step[s0:s0 + W, :]
        seg = a_step[s0:, :]
        if seg.shape[0] == 0:
            return np.repeat(a_step[-1:, :], W, axis=0)
        pad = np.repeat(seg[-1:, :], (W - seg.shape[0]), axis=0)
        return np.vstack([seg, pad])

    def _slice_avg_rewards(avg, n_ep_total, start_episode_local):
        avg = np.asarray(avg, float)
        if len(avg) == n_ep_total + 1:
            avg = avg[1:]
        start_episode_local = int(max(1, start_episode_local))
        s = max(0, start_episode_local - 1)
        avg = avg[s:]
        x = np.arange(start_episode_local, start_episode_local + len(avg))
        return x, avg

    def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states_local, data_min_local, data_max_local, n_inputs_local):
        y_sp_scaled_dev = np.asarray(y_sp_scaled_dev, float)
        data_min_local = np.asarray(data_min_local, float)
        data_max_local = np.asarray(data_max_local, float)
        y_ss_phys = np.asarray(steady_states_local["y_ss"], float)
        y_ss_scaled = apply_min_max(y_ss_phys, data_min_local[n_inputs_local:], data_max_local[n_inputs_local:])
        y_sp_scaled = y_sp_scaled_dev + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_scaled, data_min_local[n_inputs_local:], data_max_local[n_inputs_local:])
        return y_sp_phys

    def _parse_mpc_horizons(mpc_horizons_local):
        Hp0 = None
        Hc0 = None
        if mpc_horizons_local is None:
            return Hp0, Hc0
        if isinstance(mpc_horizons_local, dict):
            Hp0 = mpc_horizons_local.get("Hp", None)
            Hc0 = mpc_horizons_local.get("Hc", None)
            return Hp0, Hc0
        try:
            Hp0 = mpc_horizons_local[0]
            Hc0 = mpc_horizons_local[1]
        except Exception:
            Hp0 = None
            Hc0 = None
        return Hp0, Hc0

    def _load_pickle(path):
        if path is None:
            return None
        if os.path.isdir(path):
            cand = os.path.join(path, "input_data.pkl")
            if os.path.exists(cand):
                with open(cand, "rb") as f:
                    return pickle.load(f)

            files = []
            files += glob.glob(os.path.join(path, "*.pickle"))
            files += glob.glob(os.path.join(path, "*.pkl"))
            if len(files) == 0:
                raise FileNotFoundError("No pickle files found in directory: " + path)

            files = sorted(files, key=lambda p: os.path.getmtime(p))
            with open(files[-1], "rb") as f:
                return pickle.load(f)

        with open(path, "rb") as f:
            return pickle.load(f)

    def _recompute_step_rewards(y_phys_line, u_phys_step, y_sp_scaled_dev, steady_states_local, data_min_local, data_max_local, reward_fn_local):
        y_phys_line = np.asarray(y_phys_line, float)
        u_phys_step = np.asarray(u_phys_step, float)
        y_sp_scaled_dev = np.asarray(y_sp_scaled_dev, float)

        n_inputs_local = int(u_phys_step.shape[1])
        nFE_local = int(min(u_phys_step.shape[0], y_sp_scaled_dev.shape[0]))

        y_ss_scaled = apply_min_max(np.asarray(steady_states_local["y_ss"], float),
                                    np.asarray(data_min_local[n_inputs_local:], float),
                                    np.asarray(data_max_local[n_inputs_local:], float))
        ss_inputs = np.asarray(steady_states_local["ss_inputs"], float)
        ss_scaled_inputs = apply_min_max(ss_inputs,
                                         np.asarray(data_min_local[:n_inputs_local], float),
                                         np.asarray(data_max_local[:n_inputs_local], float))

        y_line = _y_to_line(y_phys_line, nFE_local)
        y_scaled_abs = apply_min_max(y_line, data_min_local[n_inputs_local:], data_max_local[n_inputs_local:])
        y_scaled_dev = y_scaled_abs - y_ss_scaled

        u_scaled_abs = apply_min_max(u_phys_step[:nFE_local, :], data_min_local[:n_inputs_local], data_max_local[:n_inputs_local])

        dy = y_scaled_dev[1:, :] - y_sp_scaled_dev[:nFE_local, :]

        du = np.zeros((nFE_local, n_inputs_local))
        du[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
        if nFE_local > 1:
            du[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

        y_sp_phys = ysp_scaled_dev_to_phys(y_sp_scaled_dev[:nFE_local, :],
                                           steady_states_local,
                                           data_min_local,
                                           data_max_local,
                                           n_inputs_local)

        r = np.zeros(nFE_local)
        for k in range(nFE_local):
            r[k] = float(reward_fn_local(dy[k], du[k], y_sp_phys=y_sp_phys[k]))
        return r

    def _plot_bounds(ax, t_step_local, lo, hi, alpha_fill=0.15):
        ax.fill_between(t_step_local, float(lo), float(hi), alpha=alpha_fill, step="post")
        ax.hlines([float(lo), float(hi)], t_step_local[0], t_step_local[-1], linestyles="--", linewidth=1.0)

    # ------------------------ setup ------------------------
    if reward_fn is None:
        raise ValueError("reward_fn must be provided")

    if directory is None:
        directory = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    nFE = int(nFE)
    delta_t = float(delta_t)
    time_in_sub_episodes = int(time_in_sub_episodes)
    start_episode = int(max(1, start_episode))

    y_sp = np.asarray(y_sp, float)
    y_rl = np.asarray(y_rl, float)
    u_rl = np.asarray(u_rl, float)

    n_inputs = int(u_rl.shape[1])
    n_outputs = int(y_rl.shape[1]) if y_rl.ndim > 1 else 1

    y_line_full = _y_to_line(y_rl, nFE)
    y_sp_phys_full = ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs)

    start_step = (start_episode - 1) * time_in_sub_episodes
    start_step = int(min(max(0, start_step), max(0, nFE - 1)))

    y_line = y_line_full[start_step:, :]
    u_line = u_rl[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]

    W = int(max(1, y_sp_phys.shape[0]))
    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]

    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = W - last_steps
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]

    Hp0, Hc0 = _parse_mpc_horizons(mpc_horizons)

    set_paper_plot_style()

    # ------------------------ core plots ------------------------
    # 1) Outputs vs setpoints (full from start_episode)
    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 2.8 + 2.8 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for j in range(n_outputs):
        axs[j].plot(t_line, y_line[:, j], "-")
        axs[j].step(t_step, y_sp_phys[:, j], where="post", linestyle="--")
        axs[j].spines["top"].set_visible(False)
        axs[j].spines["right"].set_visible(False)
        axs[j].xaxis.set_major_locator(mtick.MaxNLocator(6))
        axs[j].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        axs[j].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        axs[j].set_ylabel(f"y{j + 1}")
        _make_axes_bold(axs[j])
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_rl_outputs_full", save_pdf=save_pdf)

    # 2) Outputs vs setpoints (last block)
    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 2.8 + 2.8 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for j in range(n_outputs):
        axs[j].plot(t_line_blk, y_line[s_last:s_last + last_steps + 1, j], "-")
        axs[j].step(t_step_blk, y_sp_phys[s_last:s_last + last_steps, j], where="post", linestyle="--")
        axs[j].spines["top"].set_visible(False)
        axs[j].spines["right"].set_visible(False)
        axs[j].xaxis.set_major_locator(mtick.MaxNLocator(6))
        axs[j].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        axs[j].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        axs[j].set_ylabel(f"y{j + 1}")
        _make_axes_bold(axs[j])
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_rl_outputs_last_block", save_pdf=save_pdf)

    # 3) Inputs (full from start_episode)
    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 2.8 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for j in range(n_inputs):
        axs[j].step(t_step, u_line[:, j], where="post")
        axs[j].spines["top"].set_visible(False)
        axs[j].spines["right"].set_visible(False)
        axs[j].set_ylabel(f"u{j + 1}")
        _make_axes_bold(axs[j])
    axs[-1].set_xlabel("Time (h)")
    _save_fig(fig, out_dir, "fig_rl_inputs_full", save_pdf=save_pdf)

    # 4) Avg reward per episode/block (given avg_rewards)
    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = _slice_avg_rewards(avg_rewards, n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-", lw=2.8, ms=5)
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_rl_rewards", save_pdf=save_pdf)

    # 5) Horizons: traces + last block + counts
    ht_seg = None
    if horizon_trace is not None:
        ht_full = np.asarray(horizon_trace, float)
        if ht_full.shape[0] > nFE:
            ht_full = ht_full[:nFE, :]
        ht_seg = ht_full[start_step:start_step + W, :]

        Hp = ht_seg[:, 0].astype(int)
        Hc = ht_seg[:, 1].astype(int)

        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].step(t_step, Hp, where="post")
        if Hp0 is not None:
            axs[0].axhline(float(Hp0), color="red", linestyle=":", linewidth=2.2)
        axs[0].set_ylabel("Hp")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])

        axs[1].step(t_step, Hc, where="post")
        if Hc0 is not None:
            axs[1].axhline(float(Hc0), color="red", linestyle=":", linewidth=2.2)
        axs[1].set_ylabel("Hc")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, out_dir, "HPHC_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].step(t_step_blk, Hp[s_last:s_last + last_steps], where="post")
        if Hp0 is not None:
            axs[0].axhline(float(Hp0), color="red", linestyle=":", linewidth=2.2)
        axs[0].set_ylabel("Hp")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])

        axs[1].step(t_step_blk, Hc[s_last:s_last + last_steps], where="post")
        if Hc0 is not None:
            axs[1].axhline(float(Hc0), color="red", linestyle=":", linewidth=2.2)
        axs[1].set_ylabel("Hc")
        axs[1].set_xlabel("Time (h)")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, out_dir, "HPHC_last_block", save_pdf=save_pdf)

        if recipe_counts:
            counts = collections.Counter(list(zip(Hp.tolist(), Hc.tolist())))
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top = items[:20] if len(items) > 0 else []

            key0 = None
            if Hp0 is not None and Hc0 is not None:
                key0 = (int(Hp0), int(Hc0))
                if key0 not in dict(top):
                    top.append((key0, counts.get(key0, 0)))

            labels = [f"Hp={k[0]},Hc={k[1]}" for k, _ in top]
            freqs = [v for _, v in top]

            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            if len(freqs) > 0:
                bars = ax.bar(np.arange(len(freqs)), freqs)
                if key0 is not None:
                    idx0 = None
                    for ii, (k, _) in enumerate(top):
                        if k == key0:
                            idx0 = ii
                            break
                    if idx0 is not None:
                        bars[idx0].set_color("red")
                        ax.text(idx0, freqs[idx0], "MPC", ha="center", va="bottom", fontweight="bold")
                ax.set_xticks(np.arange(len(freqs)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "HPHC_counts", save_pdf=save_pdf)

    # 6) Alpha/Delta logs (bounds shaded by model_low/high)
    def _plot_alpha_delta(t_step_local, a_seg, d_seg, lo, hi, fname):
        rows = 0
        if a_seg is not None:
            rows += 1
        if d_seg is not None:
            rows += int(d_seg.shape[1])
        if rows == 0:
            return

        fig, axs = plt.subplots(rows, 1, figsize=(8.2, max(5.0, 2.2 * rows)), sharex=True)
        if rows == 1:
            axs = [axs]

        r = 0
        have_bounds = (lo is not None) and (hi is not None)

        if a_seg is not None:
            ax = axs[r]
            if have_bounds:
                _plot_bounds(ax, t_step_local, lo[0], hi[0])
            ax.step(t_step_local, a_seg, where="post")
            ax.set_ylabel("alpha")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            r += 1

        if d_seg is not None:
            for j in range(d_seg.shape[1]):
                ax = axs[r]
                if have_bounds and lo.size >= 1 + d_seg.shape[1]:
                    _plot_bounds(ax, t_step_local, lo[1 + j], hi[1 + j])
                ax.step(t_step_local, d_seg[:, j], where="post")
                ax.set_ylabel(f"delta{j + 1}")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
                r += 1

        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, fname, save_pdf=save_pdf)

    a_seg = None
    d_seg = None
    if alpha_log is not None:
        a_full = np.asarray(alpha_log, float).reshape(-1)
        if a_full.size > nFE:
            a_full = a_full[:nFE]
        a_seg = _slice_segment_step(a_full, start_step, W)[:, 0]

    if delta_log is not None:
        d_full = np.asarray(delta_log, float)
        if d_full.ndim == 1:
            d_full = d_full[:, None]
        if d_full.shape[0] > nFE:
            d_full = d_full[:nFE, :]
        if d_full.shape[1] != n_inputs and d_full.shape[1] == 1 and n_inputs > 1:
            d_full = np.repeat(d_full, n_inputs, axis=1)
        d_seg = _slice_segment_step(d_full, start_step, W)

    lo_m = None
    hi_m = None
    if model_low is not None and model_high is not None:
        lo_m = np.asarray(model_low, float).reshape(-1)
        hi_m = np.asarray(model_high, float).reshape(-1)

    if (a_seg is not None) or (d_seg is not None):
        _plot_alpha_delta(t_step, a_seg, d_seg, lo_m, hi_m, "fig_model_multipliers_full")
        a_blk = None if a_seg is None else a_seg[s_last:s_last + last_steps]
        d_blk = None if d_seg is None else d_seg[s_last:s_last + last_steps, :]
        _plot_alpha_delta(t_step_blk, a_blk, d_blk, lo_m, hi_m, "fig_model_multipliers_last_block")

    # 7) Weights logs (Q1,Q2,R1,R2) with bounds shaded
    def _plot_weights(t_step_local, w_seg, lo, hi, fname):
        if w_seg is None:
            return
        names = ["Q1", "Q2", "R1", "R2"]
        fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for k in range(4):
            ax = axs[k]
            if lo is not None and hi is not None:
                _plot_bounds(ax, t_step_local, lo[k], hi[k])
            ax.step(t_step_local, w_seg[:, k], where="post")
            ax.set_ylabel(names[k])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, fname, save_pdf=save_pdf)

    w_seg = None
    if weight_log is not None:
        w_full = np.asarray(weight_log, float)
        if w_full.ndim == 1:
            w_full = w_full[:, None]
        if w_full.shape[0] > nFE:
            w_full = w_full[:nFE, :]
        if w_full.shape[1] != 4:
            raise ValueError("weight_log must have shape [nFE, 4] for [Q1,Q2,R1,R2].")
        w_seg = _slice_segment_step(w_full, start_step, W)

    lo_w = None
    hi_w = None
    if weights_low is not None and weights_high is not None:
        lo_w = np.asarray(weights_low, float).reshape(-1)
        hi_w = np.asarray(weights_high, float).reshape(-1)
        if lo_w.size != 4 or hi_w.size != 4:
            lo_w = None
            hi_w = None

    if w_seg is not None:
        _plot_weights(t_step, w_seg, lo_w, hi_w, "fig_weights_full")
        _plot_weights(t_step_blk, w_seg[s_last:s_last + last_steps, :], lo_w, hi_w, "fig_weights_last_block")

    # 8) Recompute RL rewards step-by-step and plot (more reliable than avg_rewards if needed)
    rl_rewards_step = _recompute_step_rewards(y_line_full, u_rl[:nFE, :], y_sp[:nFE, :], steady_states, data_min, data_max, reward_fn)
    n_ep2 = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    avg_rewards_recomputed = None
    if n_ep2 > 0:
        rr2 = rl_rewards_step[:n_ep2 * time_in_sub_episodes].reshape(n_ep2, time_in_sub_episodes)
        avg_rewards_recomputed = rr2.mean(axis=1)

        x_ep2, y_ep2 = _slice_avg_rewards(avg_rewards_recomputed, n_ep2, start_episode)
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        if len(y_ep2) > 0:
            ax.plot(x_ep2, y_ep2, "o-", lw=2.8, ms=5)
        ax.set_ylabel("Avg. Reward (recomputed)")
        ax.set_xlabel("Episode #")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_rl_rewards_recomputed", save_pdf=save_pdf)

    # ------------------------ MPC comparison inside same out_dir ------------------------
    mpc_data = None
    avg_mpc = None
    mpc_rewards_step = None
    if mpc_path_or_dir is not None:
        mpc_data = _load_pickle(mpc_path_or_dir)

        y_mpc = np.asarray(mpc_data.get("y_mpc", mpc_data.get("y_rl", None)), float)
        u_mpc = np.asarray(mpc_data.get("u_mpc", mpc_data.get("u_rl", None)), float)
        if y_mpc is None or u_mpc is None:
            raise KeyError("MPC pickle must contain y_mpc and u_mpc (or y_rl/u_rl).")

        u_mpc_step = np.asarray(u_mpc, float)
        y_mpc = np.asarray(y_mpc, float)

        # actual MPC usable length in steps
        nFE_mpc = int(min(u_mpc_step.shape[0], max(0, y_mpc.shape[0] - 1)))
        if nFE_mpc <= 0:
            raise ValueError("MPC data is too short to compare.")

        u_mpc_step = u_mpc_step[:nFE_mpc, :]
        y_mpc_line_full = _y_to_line(y_mpc, nFE_mpc)

        # segment selection (same logic as compare functions)
        nFE_sp = int(y_sp_phys_full.shape[0])
        s0 = start_step
        s0 = int(min(max(0, s0), max(0, nFE_sp - 1)))

        steps_rl = int(max(1, y_line_full.shape[0] - 1))
        steps_mpc = int(max(1, nFE_mpc))
        if steps_mpc <= s0:
            Wc = 0
        else:
            Wc = int(min(nFE_sp - s0, max(1, steps_rl - s0), max(1, steps_mpc - s0)))
            Wc = int(max(1, Wc))

        if Wc > 0:
            sp_seg = y_sp_phys_full[s0:s0 + Wc, :]
            t_line_c = np.linspace(0.0, Wc * delta_t, Wc + 1)
            t_step_c = t_line_c[:-1]

            yrl_seg = _slice_segment_line(y_line_full, s0, Wc)
            ympc_seg = _slice_segment_line(y_mpc_line_full, s0, Wc)

            # outputs compare (full segment)
            fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 2.8 + 2.8 * max(1, n_outputs - 1)), sharex=True)
            if n_outputs == 1:
                axs = [axs]
            for j in range(n_outputs):
                axs[j].plot(t_line_c, yrl_seg[:, j], "-")
                axs[j].plot(t_line_c, ympc_seg[:, j], "--")
                axs[j].step(t_step_c, sp_seg[:, j], where="post", linestyle="--")
                axs[j].spines["top"].set_visible(False)
                axs[j].spines["right"].set_visible(False)
                axs[j].xaxis.set_major_locator(mtick.MaxNLocator(6))
                axs[j].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
                axs[j].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
                axs[j].set_ylabel(f"y{j + 1}")
                _make_axes_bold(axs[j])
            axs[-1].set_xlabel("Time (h)")
            _save_fig(fig, out_dir, "compare_outputs_full", save_pdf=save_pdf)

        # last episode compare (tail based)
        ep_len = int(max(1, time_in_sub_episodes))
        W2 = int(min(ep_len, y_sp_phys_full.shape[0], steps_rl, nFE_mpc))
        W2 = int(max(1, W2))

        sp_last = y_sp_phys_full[-W2:, :]
        t_line2 = np.linspace(0.0, W2 * delta_t, W2 + 1)
        t_step2 = t_line2[:-1]

        yrl_last = _slice_segment_line(y_line_full, int(max(0, steps_rl - W2)), W2)
        ympc_last = _slice_segment_line(y_mpc_line_full, int(max(0, steps_mpc - W2)), W2)

        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 2.8 + 2.8 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for j in range(n_outputs):
            axs[j].plot(t_line2, yrl_last[:, j], "-")
            axs[j].plot(t_line2, ympc_last[:, j], "--")
            axs[j].step(t_step2, sp_last[:, j], where="post", linestyle="--")
            axs[j].spines["top"].set_visible(False)
            axs[j].spines["right"].set_visible(False)
            axs[j].xaxis.set_major_locator(mtick.MaxNLocator(6))
            axs[j].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            axs[j].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
            axs[j].set_ylabel(f"y{j + 1}")
            _make_axes_bold(axs[j])
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "compare_outputs_last_episode", save_pdf=save_pdf)

        # inputs compare (last episode)
        url_last = _slice_segment_step(u_rl, int(max(0, u_rl.shape[0] - W2)), W2)
        umpc_last = _slice_segment_step(u_mpc_step, int(max(0, u_mpc_step.shape[0] - W2)), W2)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 2.8 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for j in range(n_inputs):
            axs[j].plot(t_step2, url_last[:, j], "-")
            axs[j].plot(t_step2, umpc_last[:, j], "--")
            axs[j].spines["top"].set_visible(False)
            axs[j].spines["right"].set_visible(False)
            axs[j].set_ylabel(f"u{j + 1}")
            _make_axes_bold(axs[j])
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "compare_inputs_last_episode", save_pdf=save_pdf)

        # reward compare
        mpc_rewards_step = _recompute_step_rewards(
            y_mpc_line_full,
            u_mpc_step,
            y_sp[:nFE, :],
            steady_states,
            data_min,
            data_max,
            reward_fn
        )

        n_ep_m = int(len(mpc_rewards_step) // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
        if n_ep_m > 0:
            rr_m = mpc_rewards_step[:n_ep_m * time_in_sub_episodes].reshape(n_ep_m, time_in_sub_episodes)
            avg_mpc = rr_m.mean(axis=1)
        else:
            avg_mpc = np.asarray([], float)

        avg_rl_use = avg_rewards_recomputed if avg_rewards_recomputed is not None else np.asarray(avg_rewards, float)
        avg_mpc_use = np.asarray(avg_mpc, float)

        x_rl = np.arange(1, len(avg_rl_use) + 1)
        mask_rl = x_rl >= start_episode
        x_rl = x_rl[mask_rl]
        rl_cmp = avg_rl_use[mask_rl]

        x_mpc = np.arange(1, len(avg_mpc_use) + 1)
        mask_mpc = x_mpc >= start_episode
        x_mpc = x_mpc[mask_mpc]
        mpc_cmp = avg_mpc_use[mask_mpc]

        use_mode = mpc_reward_mode
        if use_mode == "auto":
            # if MPC has only 1-2 episodes, treat it as nominal baseline by default
            if len(avg_mpc_use) <= 2:
                use_mode = "nominal"
            elif len(mpc_cmp) > 1 and np.std(mpc_cmp) > 1e-10:
                use_mode = "disturb"
            else:
                use_mode = "nominal"

        if len(x_rl) > 0:
            fig, ax = plt.subplots(figsize=(7.8, 5.0))
            ax.plot(x_rl, rl_cmp, "o-", lw=2.8, ms=5)

            if use_mode == "nominal":
                if len(mpc_cmp) > 0:
                    mpc_const = float(mpc_cmp[-1])
                elif len(avg_mpc_use) > 0:
                    mpc_const = float(avg_mpc_use[-1])
                else:
                    mpc_const = np.nan
                if np.isfinite(mpc_const):
                    ax.plot(x_rl, np.full(x_rl.shape, mpc_const, dtype=float), "s--", lw=2.8, ms=5)
            else:
                if len(x_mpc) > 0:
                    ax.plot(x_mpc, mpc_cmp, "s--", lw=2.8, ms=5)

            ax.set_ylabel("Avg. Reward")
            ax.set_xlabel("Episode #")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "compare_rewards", save_pdf=save_pdf)

    # ------------------------ save pickle ------------------------
    input_data = {
        "y_sp": y_sp,
        "y_sp_phys": y_sp_phys_full,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_rl": y_line_full,
        "u_rl": u_rl,
        "avg_rewards": avg_rewards,
        "rl_rewards_step": rl_rewards_step,
        "avg_rewards_recomputed": avg_rewards_recomputed,
        "data_min": np.asarray(data_min, float),
        "data_max": np.asarray(data_max, float),
        "start_episode_plotted": start_episode,
        "horizon_trace": None if horizon_trace is None else np.asarray(horizon_trace),
        "mpc_horizons": mpc_horizons,
        "alpha_log": None if alpha_log is None else np.asarray(alpha_log),
        "delta_log": None if delta_log is None else np.asarray(delta_log),
        "weight_log": None if weight_log is None else np.asarray(weight_log),
        "model_low": None if model_low is None else np.asarray(model_low, float),
        "model_high": None if model_high is None else np.asarray(model_high, float),
        "weights_low": None if weights_low is None else np.asarray(weights_low, float),
        "weights_high": None if weights_high is None else np.asarray(weights_high, float),
        "mpc_path_or_dir": mpc_path_or_dir,
        "mpc_avg_rewards_recomputed": None if avg_mpc is None else np.asarray(avg_mpc, float),
        "mpc_rewards_step": None if mpc_rewards_step is None else np.asarray(mpc_rewards_step, float)
    }

    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as f:
        pickle.dump(input_data, f)

    return out_dir


from utils.plotting_core import (
    _make_axes_bold,
    _save_fig,
    _set_plot_style,
    compare_mpc_rl_from_dirs_core,
    create_output_dir,
    normalize_result_bundle,
    plot_baseline_mpc_results_core,
    plot_combined_results_core,
    plot_horizon_results_core,
    plot_matrix_multiplier_results_core,
    plot_residual_results_core,
    plot_structured_matrix_results_core,
    plot_weight_multiplier_results_core,
)


def plot_baseline_mpc_results(result_bundle, plot_cfg):
    return plot_baseline_mpc_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_horizon_results(result_bundle, plot_cfg):
    return plot_horizon_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_matrix_multiplier_results(result_bundle, plot_cfg):
    return plot_matrix_multiplier_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_structured_matrix_results(result_bundle, plot_cfg):
    return plot_structured_matrix_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_weight_multiplier_results(result_bundle, plot_cfg):
    return plot_weight_multiplier_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_residual_results(result_bundle, plot_cfg):
    return plot_residual_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_combined_results(result_bundle, plot_cfg):
    return plot_combined_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)


def compare_mpc_rl_from_dirs(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    compare_mode="nominal",
    start_episode=1,
    start_idx=None,
    n_inputs=2,
    save_pdf=False,
    style_profile="hybrid",
):
    del start_idx
    return compare_mpc_rl_from_dirs_core(
        rl_dir=rl_dir,
        mpc_path_or_dir=mpc_path_or_dir,
        reward_fn=reward_fn,
        directory=directory,
        prefix_name=prefix_name,
        compare_mode=compare_mode,
        start_episode=start_episode,
        n_inputs=n_inputs,
        save_pdf=save_pdf,
        style_profile=style_profile,
    )


def load_result_bundle(path_or_dir):
    import pickle
    from pathlib import Path

    path = Path(path_or_dir).expanduser()
    if path.is_dir():
        path = path / "input_data.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Could not find saved bundle at {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return normalize_result_bundle(bundle)


def load_result_bundles(paths_or_dirs, labels=None):
    from pathlib import Path

    paths = [Path(p).expanduser() for p in paths_or_dirs]
    labels = list(labels) if labels is not None else []
    runs = []
    for idx, path in enumerate(paths):
        label = labels[idx] if idx < len(labels) else path.parent.name if path.name == "input_data.pkl" else path.name
        runs.append(
            {
                "label": label,
                "path": str(path),
                "bundle": load_result_bundle(path),
            }
        )
    return runs


def summarize_result_bundles(paths_or_dirs, labels=None):
    import numpy as np

    runs = load_result_bundles(paths_or_dirs, labels=labels)
    rows = []
    for run in runs:
        bundle = run["bundle"]
        avg = np.asarray(bundle.get("avg_rewards", []), float).reshape(-1)
        rows.append(
            {
                "label": run["label"],
                "path": run["path"],
                "method_family": bundle.get("method_family"),
                "algorithm": bundle.get("algorithm"),
                "run_mode": bundle.get("run_mode"),
                "state_mode": bundle.get("state_mode"),
                "n_step": int(bundle.get("n_step", 1) or 1),
                "multistep_mode": bundle.get("multistep_mode"),
                "lambda_value": bundle.get("lambda_value"),
                "nFE": int(bundle.get("nFE", 0) or 0),
                "episodes": int(len(avg)),
                "final_avg_reward": float(avg[-1]) if len(avg) else float("nan"),
                "best_avg_reward": float(np.max(avg)) if len(avg) else float("nan"),
                "mean_avg_reward": float(np.mean(avg)) if len(avg) else float("nan"),
            }
        )
    return rows


def plot_multi_run_reward_summary(paths_or_dirs, plot_cfg, labels=None):
    import csv
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    runs = load_result_bundles(paths_or_dirs, labels=labels)
    if not runs:
        raise ValueError("No runs provided.")

    directory = plot_cfg.get("directory")
    prefix_name = plot_cfg.get("prefix_name", "multi_run_reward_summary")
    start_episode = int(max(1, plot_cfg.get("start_episode", 1)))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    style_profile = plot_cfg.get("style_profile", "hybrid")

    out_dir = create_output_dir(directory if directory is not None else os.getcwd(), prefix_name)
    _set_plot_style(style_profile)

    aligned = []
    final_rewards = []
    best_rewards = []
    labels_out = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        bundle = run["bundle"]
        avg = np.asarray(bundle.get("avg_rewards", []), float).reshape(-1)
        if len(avg) == 0:
            continue
        sliced = avg[start_episode - 1 :]
        if len(sliced) == 0:
            continue
        x = np.arange(start_episode, start_episode + len(sliced))
        ax.plot(x, sliced, alpha=0.6, label=run["label"])
        aligned.append(sliced)
        final_rewards.append(float(sliced[-1]))
        best_rewards.append(float(np.max(sliced)))
        labels_out.append(run["label"])

    if aligned:
        min_len = min(len(arr) for arr in aligned)
        if min_len > 0:
            stack = np.vstack([arr[:min_len] for arr in aligned])
            x = np.arange(start_episode, start_episode + min_len)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            ax.plot(x, mean, color="black", linewidth=2.5, label="Mean")
            ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.15, label="Mean ± std")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _make_axes_bold(ax)
    if len(labels_out) <= 8:
        ax.legend(frameon=False)
    _save_fig(fig, out_dir, "fig_multi_run_avg_rewards", save_pdf=save_pdf)

    if final_rewards:
        fig, ax = plt.subplots(figsize=(9, 5))
        positions = np.arange(1, len(final_rewards) + 1)
        ax.boxplot([final_rewards, best_rewards], labels=["Final avg reward", "Best avg reward"], patch_artist=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_multi_run_reward_boxplots", save_pdf=save_pdf)

        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels_out)), 5))
        ax.bar(positions - 0.18, final_rewards, width=0.36, label="Final")
        ax.bar(positions + 0.18, best_rewards, width=0.36, label="Best")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_out, rotation=30, ha="right")
        ax.set_ylabel("Reward")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        ax.legend(frameon=False)
        _save_fig(fig, out_dir, "fig_multi_run_reward_bars", save_pdf=save_pdf)

    rows = summarize_result_bundles(paths_or_dirs, labels=labels)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return out_dir


def plot_rl_results_dqn(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_mpc,
    u_mpc,
    avg_rewards,
    data_min,
    data_max,
    reward_fn=None,
    horizon_trace=None,
    mpc_horizons=None,
    recipe_counts=True,
    start_episode=1,
    prefix_name="agent_result",
    directory=None,
    save_pdf=False,
):
    result_bundle = {
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y": y_mpc,
        "u": u_mpc,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "rewards_step": None,
        "delta_y_storage": None,
        "delta_u_storage": None,
        "horizon_trace": horizon_trace,
        "action_trace": None,
        "yhat": None,
        "xhatdhat": None,
        "mpc_horizons": mpc_horizons,
    }
    plot_cfg = {
        "directory": directory if directory is not None else __import__("os").getcwd(),
        "prefix_name": prefix_name,
        "start_episode": start_episode,
        "recipe_counts": recipe_counts,
        "save_pdf": save_pdf,
    }
    return plot_horizon_results(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_rl_results_td3_multipliers_dqnstyle(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_rl,
    u_rl,
    avg_rewards,
    data_min,
    data_max,
    reward_fn=None,
    coef_alpha=None,
    coef_delta=None,
    low_coef=None,
    high_coef=None,
    start_episode=1,
    prefix_name="agent_result",
    directory=None,
    save_pdf=False,
):
    del reward_fn
    result_bundle = {
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y": y_rl,
        "u": u_rl,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "alpha_log": coef_alpha,
        "delta_log": coef_delta,
        "low_coef": low_coef,
        "high_coef": high_coef,
    }
    plot_cfg = {
        "directory": directory if directory is not None else __import__("os").getcwd(),
        "prefix_name": prefix_name,
        "start_episode": start_episode,
        "save_pdf": save_pdf,
    }
    return plot_matrix_multiplier_results(result_bundle=result_bundle, plot_cfg=plot_cfg)


def plot_rl_results_td3_weights_dqnstyle(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_rl,
    u_rl,
    avg_rewards,
    data_min,
    data_max,
    mult_log=None,
    low_coef=None,
    high_coef=None,
    reward_fn=None,
    start_episode=1,
    prefix_name="td3_weights",
    directory=None,
    save_pdf=False,
):
    del reward_fn
    result_bundle = {
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y": y_rl,
        "u": u_rl,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "weight_log": mult_log,
        "low_coef": low_coef,
        "high_coef": high_coef,
    }
    plot_cfg = {
        "directory": directory if directory is not None else __import__("os").getcwd(),
        "prefix_name": prefix_name,
        "start_episode": start_episode,
        "save_pdf": save_pdf,
    }
    return plot_weight_multiplier_results(result_bundle=result_bundle, plot_cfg=plot_cfg)


def compare_mpc_rl_nominal_from_dirs(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    start_episode=1,
    start_idx=None,
    n_inputs=2,
    save_pdf=False,
):
    return compare_mpc_rl_from_dirs(
        rl_dir=rl_dir,
        mpc_path_or_dir=mpc_path_or_dir,
        reward_fn=reward_fn,
        directory=directory,
        prefix_name=prefix_name,
        compare_mode="nominal",
        start_episode=start_episode,
        start_idx=start_idx,
        n_inputs=n_inputs,
        save_pdf=save_pdf,
    )


def compare_mpc_rl_disturb_from_dirs(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    start_episode=1,
    start_idx=None,
    n_inputs=2,
    save_pdf=False,
):
    return compare_mpc_rl_from_dirs(
        rl_dir=rl_dir,
        mpc_path_or_dir=mpc_path_or_dir,
        reward_fn=reward_fn,
        directory=directory,
        prefix_name=prefix_name,
        compare_mode="disturb",
        start_episode=start_episode,
        start_idx=start_idx,
        n_inputs=n_inputs,
        save_pdf=save_pdf,
    )


def plot_rl_results_multiagent_dqnstyle(
    y_sp,
    steady_states,
    nFE,
    delta_t,
    time_in_sub_episodes,
    y_rl,
    u_rl,
    avg_rewards,
    data_min,
    data_max,
    reward_fn=None,
    horizon_trace=None,
    mpc_horizons=None,
    recipe_counts=True,
    horizon_recipes=None,
    alpha_log=None,
    delta_log=None,
    weight_log=None,
    model_low=None,
    model_high=None,
    weights_low=None,
    weights_high=None,
    mpc_path_or_dir=None,
    mpc_reward_mode="auto",
    start_episode=1,
    prefix_name="multi_agent_run",
    directory=None,
    save_pdf=False,
    residual_raw_log=None,
    residual_exec_log=None,
    u_base=None,
    residual_low=None,
    residual_high=None,
    style_profile="hybrid",
):
    del recipe_counts
    del mpc_reward_mode
    result_bundle = {
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y": y_rl,
        "u": u_rl,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "horizon_trace": horizon_trace,
        "horizon_action_trace": None,
        "horizon_agent_kind": "dqn" if horizon_trace is not None else None,
        "horizon_state_mode": "standard",
        "horizon_recipes": horizon_recipes,
        "matrix_alpha_log": alpha_log,
        "matrix_delta_log": delta_log,
        "matrix_low_coef": model_low,
        "matrix_high_coef": model_high,
        "matrix_agent_kind": "td3" if alpha_log is not None else None,
        "matrix_state_mode": "standard",
        "weight_log": weight_log,
        "weight_low_coef": weights_low,
        "weight_high_coef": weights_high,
        "weight_agent_kind": "td3" if weight_log is not None else None,
        "weight_state_mode": "standard",
        "residual_raw_log": residual_raw_log,
        "residual_exec_log": residual_exec_log,
        "residual_low_coef": residual_low,
        "residual_high_coef": residual_high,
        "residual_agent_kind": "td3" if residual_exec_log is not None else None,
        "residual_state_mode": "standard",
        "u_base": u_base,
        "active_agents": {
            "horizon": horizon_trace is not None,
            "matrix": alpha_log is not None,
            "weights": weight_log is not None,
            "residual": residual_exec_log is not None,
        },
        "mpc_horizons": mpc_horizons,
        "mpc_path_or_dir": mpc_path_or_dir,
    }
    plot_cfg = {
        "directory": directory if directory is not None else __import__("os").getcwd(),
        "prefix_name": prefix_name,
        "start_episode": start_episode,
        "save_pdf": save_pdf,
        "style_profile": style_profile,
        "reward_fn": reward_fn,
        "include_baseline_compare": mpc_path_or_dir is not None and reward_fn is not None,
        "compare_mode": "disturb" if "dist" in str(prefix_name).lower() else "nominal",
        "compare_prefix": "baseline_compare",
        "mpc_path_or_dir": mpc_path_or_dir,
    }
    return plot_combined_results(result_bundle=result_bundle, plot_cfg=plot_cfg)
