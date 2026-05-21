from __future__ import annotations

import csv
import gc
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures" / "polymer_latest_family_runs_20260521"
REPORT_PATH = REPO_ROOT / "report" / "polymer_latest_family_runs_2026_05_21.md"

RUNS = {
    "OF-MPC": {
        "family": "baseline",
        "path": REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
        "kind": "mpc",
    },
    "Horizon DQN": {
        "family": "horizon",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "horizon_disturb_unified"
        / "20260520_222308"
        / "input_data.pkl",
        "kind": "rl",
    },
    "Dueling Horizon": {
        "family": "horizon_dueling",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "dueling_horizon_disturb_unified"
        / "20260520_225756"
        / "input_data.pkl",
        "kind": "rl",
    },
    "TD3 Weights": {
        "family": "weights",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_weights_disturb"
        / "20260520_214118"
        / "input_data.pkl",
        "kind": "rl",
    },
    "TD3 Residual": {
        "family": "residual",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_residual_disturb"
        / "20260520_214325"
        / "input_data.pkl",
        "kind": "rl",
    },
    "TD3 Markov": {
        "family": "markov",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_markov_disturb"
        / "20260521_032358"
        / "input_data.pkl",
        "kind": "rl",
    },
    "Combined": {
        "family": "combined",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "combined_disturb_h_dqn_mismatch__markov_td3_mismatch__w_td3_mismatch__r_td3_mismatch_rho"
        / "20260521_054234"
        / "input_data.pkl",
        "kind": "rl",
    },
}

OUTPUT_LABELS = ["eta", "T"]
INPUT_LABELS = ["Qc", "Qm"]


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _as_array(value, default=None):
    if value is None:
        return default
    return np.asarray(value, float)


def _reverse_minmax(x_scaled, lo, hi):
    return np.asarray(x_scaled, float) * (np.asarray(hi, float) - np.asarray(lo, float)) + np.asarray(lo, float)


def ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs=2):
    y_sp = np.asarray(y_sp, float)
    y_ss = np.asarray(steady_states["y_ss"], float)
    y_min = np.asarray(data_min[n_inputs:], float)
    y_max = np.asarray(data_max[n_inputs:], float)
    y_ss_scaled = (y_ss - y_min) / np.maximum(y_max - y_min, 1e-12)
    return _reverse_minmax(y_sp + y_ss_scaled, y_min, y_max)


def rolling_mean(values, window=5):
    values = np.asarray(values, float)
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def finite_mean(values):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_q(values, q):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def source_fractions(source_log, start_idx):
    if source_log is None:
        return {}
    src = np.asarray(source_log, int)[start_idx:]
    denom = max(1, int(src.size))
    return {
        "td3": float(np.sum(src == 2) / denom),
        "ls_fallback": float(np.sum(src == 3) / denom),
        "nominal_fallback": float(np.sum(src == 4) / denom),
        "ls_any": float(np.sum(np.isin(src, [1, 3, 5])) / denom),
    }


def load_run(label, spec):
    data = _load_pickle(spec["path"])
    nfe = int(data["nFE"])
    n_inputs = 2
    dt = float(data["delta_t"])
    time_in_sub = int(data["time_in_sub_episodes"])
    warm_start_step = int(data.get("warm_start_step", time_in_sub * 10))
    warm_episodes = int(warm_start_step // max(1, time_in_sub))
    data_min = np.asarray(data["data_min"], float)
    data_max = np.asarray(data["data_max"], float)
    steady_states = data["steady_states"]
    y_sp = np.asarray(data["y_sp"], float)[:nfe, :]
    y_sp_phys = ysp_scaled_dev_to_phys(y_sp, steady_states, data_min, data_max, n_inputs=n_inputs)

    if spec["kind"] == "mpc":
        y = np.asarray(data["y_mpc"], float)
        u = np.asarray(data["u_mpc"], float)
    else:
        y = np.asarray(data["y"], float)
        u = np.asarray(data["u"], float)

    y_eval = y[1 : nfe + 1, :]
    err_phys = y_eval - y_sp_phys
    delta_y = _as_array(data.get("delta_y_storage"))
    if delta_y is None:
        y_min = data_min[n_inputs:]
        y_max = data_max[n_inputs:]
        delta_y = err_phys / np.maximum(y_max - y_min, 1e-12)
    delta_y = delta_y[:nfe, :]
    delta_u = _as_array(data.get("delta_u_storage"), np.zeros((nfe, n_inputs), dtype=float))[:nfe, :]
    avg_rewards = np.asarray(data.get("avg_rewards", []), float)
    rewards_step = np.asarray(data.get("rewards_step", []), float)
    if rewards_step.size == 0 and data.get("rewards_mpc") is not None:
        rewards_step = np.asarray(data["rewards_mpc"], float)

    episode_count = int(avg_rewards.size)
    tail_episodes = min(10, episode_count)
    tail_steps = min(nfe, tail_episodes * time_in_sub)
    tail_start = max(0, nfe - tail_steps)
    post_warm_step = min(nfe, warm_start_step + 1)
    post_warm_episode = min(episode_count, warm_episodes)

    du_phys = np.diff(u, axis=0) if u.shape[0] > 1 else np.zeros_like(u)
    err_tail = err_phys[tail_start:nfe, :]
    dy_tail = delta_y[tail_start:nfe, :]
    du_tail = delta_u[tail_start:nfe, :]
    rewards_tail_steps = rewards_step[tail_start:nfe] if rewards_step.size else np.array([], dtype=float)

    metrics = {
        "method": label,
        "family": spec["family"],
        "path": spec["path"].relative_to(REPO_ROOT).as_posix(),
        "nFE": nfe,
        "episodes": episode_count,
        "time_in_sub_episodes": time_in_sub,
        "warm_start_episode_count": warm_episodes,
        "reward_mean": finite_mean(avg_rewards),
        "reward_post_warm_mean": finite_mean(avg_rewards[post_warm_episode:]),
        "reward_tail10_mean": finite_mean(avg_rewards[-tail_episodes:]),
        "reward_final_episode": float(avg_rewards[-1]) if avg_rewards.size else float("nan"),
        "step_reward_tail10_mean": finite_mean(rewards_tail_steps),
        "eta_rmse_tail": float(np.sqrt(np.mean(err_tail[:, 0] ** 2))),
        "T_rmse_tail": float(np.sqrt(np.mean(err_tail[:, 1] ** 2))),
        "eta_mae_tail": float(np.mean(np.abs(err_tail[:, 0]))),
        "T_mae_tail": float(np.mean(np.abs(err_tail[:, 1]))),
        "eta_max_abs_tail": float(np.max(np.abs(err_tail[:, 0]))),
        "T_max_abs_tail": float(np.max(np.abs(err_tail[:, 1]))),
        "scaled_rmse_tail": float(np.sqrt(np.mean(dy_tail**2))),
        "scaled_rmse_post_warm": float(np.sqrt(np.mean(delta_y[post_warm_step:nfe, :] ** 2))),
        "scaled_rmse_all": float(np.sqrt(np.mean(delta_y**2))),
        "mean_abs_delta_u_scaled_tail": float(np.mean(np.abs(du_tail))),
        "input_move_mean_tail": float(np.mean(np.linalg.norm(du_phys[tail_start:nfe, :], axis=1)))
        if du_phys.size
        else 0.0,
    }

    mechanism = {
        "method": label,
        "horizon_tail_Hp_mean": float("nan"),
        "horizon_tail_Hc_mean": float("nan"),
        "horizon_distinct_actions": float("nan"),
        "weight_tail_Q1_mean": float("nan"),
        "weight_tail_Q2_mean": float("nan"),
        "weight_tail_R1_mean": float("nan"),
        "weight_tail_R2_mean": float("nan"),
        "residual_tail_norm_mean": float("nan"),
        "markov_z_q95_abs": float("nan"),
        "markov_z_norm_tail_mean": float("nan"),
        "markov_td3_source_post_warm": float("nan"),
        "markov_ls_fallback_post_warm": float("nan"),
        "markov_nominal_fallback_post_warm": float("nan"),
        "markov_projection_active_fraction": float("nan"),
        "markov_coord_cap_tail_mean": float("nan"),
    }

    horizon = data.get("horizon_trace")
    if horizon is not None:
        h = np.asarray(horizon, float)[:nfe, :]
        mechanism["horizon_tail_Hp_mean"] = float(np.mean(h[tail_start:nfe, 0]))
        mechanism["horizon_tail_Hc_mean"] = float(np.mean(h[tail_start:nfe, 1]))
        mechanism["horizon_distinct_actions"] = float(len({tuple(row) for row in h[post_warm_step:nfe, :].astype(int)}))

    weights = data.get("weight_log")
    if weights is not None:
        w = np.asarray(weights, float)[:nfe, :]
        if w.size:
            means = np.mean(w[tail_start:nfe, :], axis=0)
            mechanism["weight_tail_Q1_mean"] = float(means[0])
            mechanism["weight_tail_Q2_mean"] = float(means[1])
            mechanism["weight_tail_R1_mean"] = float(means[2])
            mechanism["weight_tail_R2_mean"] = float(means[3])

    residual = data.get("residual_exec_log", data.get("delta_u_res_exec_log"))
    if residual is not None:
        r = np.asarray(residual, float)[:nfe, :]
        if r.size:
            mechanism["residual_tail_norm_mean"] = float(np.mean(np.linalg.norm(r[tail_start:nfe, :], axis=1)))

    z_log = data.get("z_log", data.get("markov_z_log"))
    if z_log is not None:
        z = np.asarray(z_log, float)[:nfe, :]
        if z.size:
            mechanism["markov_z_q95_abs"] = finite_q(np.abs(z[post_warm_step:nfe, :]).reshape(-1), 0.95)
            mechanism["markov_z_norm_tail_mean"] = float(np.mean(np.linalg.norm(z[tail_start:nfe, :], axis=1)))

    cap_log = data.get("z_safety_effective_cap_log", data.get("markov_z_safety_effective_cap_log"))
    if cap_log is not None:
        cap = np.asarray(cap_log, float)[:nfe]
        mechanism["markov_coord_cap_tail_mean"] = finite_mean(cap[tail_start:nfe])

    projection_log = data.get(
        "z_safety_requested_projection_active_log",
        data.get("markov_z_safety_requested_projection_active_log"),
    )
    if projection_log is not None:
        proj = np.asarray(projection_log, float)[:nfe]
        mechanism["markov_projection_active_fraction"] = finite_mean(proj[post_warm_step:nfe])

    source_log = data.get("rl_action_source_log", data.get("markov_action_source_log"))
    fractions = source_fractions(source_log, post_warm_step)
    if fractions:
        mechanism["markov_td3_source_post_warm"] = fractions["td3"]
        mechanism["markov_ls_fallback_post_warm"] = fractions["ls_fallback"]
        mechanism["markov_nominal_fallback_post_warm"] = fractions["nominal_fallback"]

    config = data.get("config_snapshot")
    if isinstance(config, dict):
        metrics["test_cycle"] = config.get("test_cycle")
    else:
        metrics["test_cycle"] = None
    if data.get("summary_metrics"):
        metrics["stored_summary_metrics"] = data["summary_metrics"]

    arrays = {
        "avg_rewards": avg_rewards,
        "rewards_step": rewards_step,
        "y_eval": y_eval,
        "y_sp_phys": y_sp_phys,
        "err_phys": err_phys,
        "delta_y": delta_y,
        "delta_u": delta_u,
        "u": u,
        "tail_start": tail_start,
        "post_warm_step": post_warm_step,
        "warm_episodes": warm_episodes,
        "time_in_sub": time_in_sub,
        "dt": dt,
        "horizon_trace": None if horizon is None else np.asarray(horizon, float)[:nfe, :],
        "weight_log": None if weights is None else np.asarray(weights, float)[:nfe, :],
        "residual_exec": None if residual is None else np.asarray(residual, float)[:nfe, :],
        "z_log": None if z_log is None else np.asarray(z_log, float)[:nfe, :],
        "cap_log": None if cap_log is None else np.asarray(cap_log, float)[:nfe],
        "source_log": None if source_log is None else np.asarray(source_log, int)[:nfe],
        "projection_log": None if projection_log is None else np.asarray(projection_log, int)[:nfe],
    }
    del data
    gc.collect()
    return {"metrics": metrics, "mechanism": mechanism, "arrays": arrays}


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fmt(value, digits=3):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def md_table(rows, columns):
    out = []
    out.append("| " + " | ".join(label for _, label, _ in columns) + " |")
    out.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for key, _, digits in columns:
            value = row.get(key)
            cells.append(str(value) if digits is None else fmt(value, digits))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def annotate_bars(ax, bars, digits=2):
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.{digits}f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
                rotation=90,
            )


def make_figures(results, summary_rows, mechanism_rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = list(results.keys())
    colors = {
        "OF-MPC": "#1f2937",
        "Horizon DQN": "#2563eb",
        "Dueling Horizon": "#0891b2",
        "TD3 Weights": "#7c3aed",
        "TD3 Residual": "#dc2626",
        "TD3 Markov": "#16a34a",
        "Combined": "#f97316",
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for method in methods:
        avg = results[method]["arrays"]["avg_rewards"]
        if avg.size == 0:
            continue
        x = np.arange(1, avg.size + 1)
        ax.plot(x, rolling_mean(avg, 5), label=method, color=colors.get(method), linewidth=2.0)
    warm = int(results["OF-MPC"]["arrays"]["warm_episodes"])
    ax.axvline(warm, color="#6b7280", linestyle="--", linewidth=1.3, label="warm-start end")
    ax.set_title("Episode reward trends, 5-episode rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reward_learning_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    labels = [r["method"] for r in summary_rows]
    vals = [r["reward_tail10_mean"] for r in summary_rows]
    bars = ax.bar(np.arange(len(labels)), vals, color=[colors.get(label, "#64748b") for label in labels])
    ax.axhline(summary_rows[0]["reward_tail10_mean"], color="#111827", linestyle="--", linewidth=1.2, label="OF-MPC")
    annotate_bars(ax, bars, digits=2)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Tail-10 episode mean reward")
    ax.set_title("Tail reward comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tail_reward_comparison.png", dpi=220)
    plt.close(fig)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    eta = np.array([r["eta_rmse_tail"] for r in summary_rows])
    temp = np.array([r["T_rmse_tail"] for r in summary_rows])
    ax.bar(x - width / 2, eta, width, label="eta RMSE", color="#2563eb")
    ax.bar(x + width / 2, temp, width, label="T RMSE", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Physical-unit RMSE over last 10 episodes")
    ax.set_title("Tail tracking error by output")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tail_tracking_rmse_physical.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    vals = [r["scaled_rmse_tail"] for r in summary_rows]
    bars = ax.bar(np.arange(len(labels)), vals, color=[colors.get(label, "#64748b") for label in labels])
    annotate_bars(ax, bars, digits=3)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Scaled-deviation RMSE")
    ax.set_title("Tail normalized tracking error")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tail_tracking_rmse_scaled.png", dpi=220)
    plt.close(fig)

    tail_start = int(results["OF-MPC"]["arrays"]["tail_start"])
    nfe = int(summary_rows[0]["nFE"])
    dt = float(results["OF-MPC"]["arrays"]["dt"])
    view_steps = min(4 * int(results["OF-MPC"]["arrays"]["time_in_sub"]), nfe - tail_start)
    s0 = nfe - view_steps
    t = np.arange(view_steps) * dt
    fig, axs = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    ysp = results["OF-MPC"]["arrays"]["y_sp_phys"][s0:nfe, :]
    for out_idx, ax in enumerate(axs):
        ax.step(t, ysp[:view_steps, out_idx], where="post", color="#111827", linestyle="--", linewidth=1.7, label="Setpoint")
        for method in methods:
            y = results[method]["arrays"]["y_eval"][s0:nfe, out_idx]
            ax.plot(t, y[:view_steps], label=method, color=colors.get(method), linewidth=1.35, alpha=0.9)
        ax.set_ylabel(OUTPUT_LABELS[out_idx])
        ax.grid(True, alpha=0.25)
    axs[-1].set_xlabel("Tail window time (h)")
    axs[0].set_title("Final four-episode output tracking overlay")
    axs[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_final_tail_tracking_overlay.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11.2, 6.6), sharex=True)
    move = [r["input_move_mean_tail"] for r in summary_rows]
    du = [r["mean_abs_delta_u_scaled_tail"] for r in summary_rows]
    axs[0].bar(np.arange(len(labels)), move, color="#0f766e")
    axs[0].set_ylabel("Mean ||Delta u|| physical")
    axs[0].set_title("Tail input movement")
    axs[0].grid(True, axis="y", alpha=0.25)
    axs[1].bar(np.arange(len(labels)), du, color="#7c3aed")
    axs[1].set_ylabel("Mean |Delta u| scaled")
    axs[1].grid(True, axis="y", alpha=0.25)
    axs[1].set_xticks(np.arange(len(labels)))
    axs[1].set_xticklabels(labels, rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tail_input_movement.png", dpi=220)
    plt.close(fig)

    markov_methods = [m for m in ("TD3 Markov", "Combined") if results[m]["arrays"]["z_log"] is not None]
    if markov_methods:
        fig, axs = plt.subplots(len(markov_methods), 1, figsize=(11.0, 3.4 + 2.6 * len(markov_methods)), sharex=True)
        if len(markov_methods) == 1:
            axs = [axs]
        for ax, method in zip(axs, markov_methods):
            arr = results[method]["arrays"]
            z = arr["z_log"]
            cap = arr["cap_log"]
            norm = np.linalg.norm(z, axis=1)
            s = max(0, nfe - 10 * int(arr["time_in_sub"]))
            tt = np.arange(nfe - s) * arr["dt"]
            ax.plot(tt, norm[s:nfe], color=colors.get(method), label="||z||2")
            if cap is not None:
                ax.plot(tt, cap[s:nfe], color="#111827", linestyle="--", label="coord cap")
            ax.set_ylabel(method)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        axs[-1].set_xlabel("Last 10 episodes time (h)")
        axs[0].set_title("Markov z safety activity")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_markov_z_safety_tail.png", dpi=220)
        plt.close(fig)

        source_labels = ["TD3", "LS fallback", "Nominal fallback", "Projection active"]
        source_values = []
        for method in markov_methods:
            mech = next(row for row in mechanism_rows if row["method"] == method)
            source_values.append(
                [
                    mech["markov_td3_source_post_warm"],
                    mech["markov_ls_fallback_post_warm"],
                    mech["markov_nominal_fallback_post_warm"],
                    mech["markov_projection_active_fraction"],
                ]
            )
        source_values = np.asarray(source_values, float)
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        xsrc = np.arange(len(source_labels))
        wbar = 0.8 / max(1, len(markov_methods))
        for idx, method in enumerate(markov_methods):
            ax.bar(xsrc + (idx - (len(markov_methods) - 1) / 2) * wbar, source_values[idx], wbar, label=method)
        ax.set_xticks(xsrc)
        ax.set_xticklabels(source_labels, rotation=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Post-warm fraction")
        ax.set_title("Markov source and safety projection fractions")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_markov_source_projection_fractions.png", dpi=220)
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.0))
    horizon_rows = [row for row in mechanism_rows if np.isfinite(row["horizon_tail_Hp_mean"])]
    if horizon_rows:
        xh = np.arange(len(horizon_rows))
        axs[0, 0].bar(xh - 0.18, [row["horizon_tail_Hp_mean"] for row in horizon_rows], 0.36, label="Hp")
        axs[0, 0].bar(xh + 0.18, [row["horizon_tail_Hc_mean"] for row in horizon_rows], 0.36, label="Hc")
        axs[0, 0].set_xticks(xh)
        axs[0, 0].set_xticklabels([row["method"] for row in horizon_rows], rotation=12, ha="right")
        axs[0, 0].set_title("Tail horizon choices")
        axs[0, 0].legend()
    weight_rows = [row for row in mechanism_rows if np.isfinite(row["weight_tail_Q1_mean"])]
    if weight_rows:
        xw = np.arange(4)
        for row in weight_rows:
            axs[0, 1].plot(xw, [row[f"weight_tail_{name}_mean"] for name in ("Q1", "Q2", "R1", "R2")], "o-", label=row["method"])
        axs[0, 1].set_xticks(xw)
        axs[0, 1].set_xticklabels(["Q1", "Q2", "R1", "R2"])
        axs[0, 1].set_title("Tail weight multipliers")
        axs[0, 1].legend(fontsize=8)
    residual_rows = [row for row in mechanism_rows if np.isfinite(row["residual_tail_norm_mean"])]
    if residual_rows:
        axs[1, 0].bar(np.arange(len(residual_rows)), [row["residual_tail_norm_mean"] for row in residual_rows], color="#dc2626")
        axs[1, 0].set_xticks(np.arange(len(residual_rows)))
        axs[1, 0].set_xticklabels([row["method"] for row in residual_rows], rotation=12, ha="right")
        axs[1, 0].set_title("Tail residual correction norm")
    markov_rows = [row for row in mechanism_rows if np.isfinite(row["markov_z_q95_abs"])]
    if markov_rows:
        xm = np.arange(len(markov_rows))
        axs[1, 1].bar(xm - 0.18, [row["markov_z_q95_abs"] for row in markov_rows], 0.36, label="q95 |z_i|")
        axs[1, 1].bar(xm + 0.18, [row["markov_z_norm_tail_mean"] for row in markov_rows], 0.36, label="tail mean ||z||")
        axs[1, 1].set_xticks(xm)
        axs[1, 1].set_xticklabels([row["method"] for row in markov_rows], rotation=12, ha="right")
        axs[1, 1].set_title("Markov authority")
        axs[1, 1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Controller mechanism dashboard", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_controller_mechanism_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary_rows, mechanism_rows):
    by_method = {row["method"]: row for row in summary_rows}
    mech = {row["method"]: row for row in mechanism_rows}
    mpc_tail = by_method["OF-MPC"]["reward_tail10_mean"]
    best_reward = max(summary_rows, key=lambda row: row["reward_tail10_mean"])
    best_scaled = min(summary_rows, key=lambda row: row["scaled_rmse_tail"])
    best_eta = min(summary_rows, key=lambda row: row["eta_rmse_tail"])
    best_temp = min(summary_rows, key=lambda row: row["T_rmse_tail"])

    reward_ordered = sorted(summary_rows, key=lambda row: row["reward_tail10_mean"], reverse=True)
    tracking_ordered = sorted(summary_rows, key=lambda row: row["scaled_rmse_tail"])
    provenance_rows = [
        {"method": row["method"], "path": row["path"], "episodes": row["episodes"], "warm": row["warm_start_episode_count"]}
        for row in summary_rows
    ]

    summary_table = md_table(
        summary_rows,
        [
            ("method", "Method", None),
            ("reward_tail10_mean", "Tail reward", 2),
            ("reward_final_episode", "Final reward", 2),
            ("scaled_rmse_tail", "Tail scaled RMSE", 4),
            ("eta_rmse_tail", "eta RMSE", 4),
            ("T_rmse_tail", "T RMSE", 3),
            ("mean_abs_delta_u_scaled_tail", "Tail mean abs du scaled", 5),
        ],
    )
    rank_table = md_table(
        reward_ordered,
        [
            ("method", "Method", None),
            ("reward_tail10_mean", "Tail reward", 2),
            ("scaled_rmse_tail", "Tail scaled RMSE", 4),
            ("eta_rmse_tail", "eta RMSE", 4),
            ("T_rmse_tail", "T RMSE", 3),
        ],
    )
    mechanism_table = md_table(
        mechanism_rows,
        [
            ("method", "Method", None),
            ("horizon_tail_Hp_mean", "Tail Hp", 2),
            ("horizon_tail_Hc_mean", "Tail Hc", 2),
            ("weight_tail_Q1_mean", "Q1 mult", 3),
            ("weight_tail_Q2_mean", "Q2 mult", 3),
            ("residual_tail_norm_mean", "Residual norm", 5),
            ("markov_z_q95_abs", "q95 abs z_i", 4),
            ("markov_td3_source_post_warm", "TD3 source", 3),
            ("markov_projection_active_fraction", "Proj active", 3),
        ],
    )
    provenance_table = md_table(
        provenance_rows,
        [
            ("method", "Method", None),
            ("path", "Bundle", None),
            ("episodes", "Episodes", 0),
            ("warm", "Warm episodes", 0),
        ],
    )

    lines = []
    lines.append("# Polymer Latest Family Runs Analysis")
    lines.append("")
    lines.append("Date: 2026-05-21")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This report analyzes the latest disturbed polymer CSTR runs for OF-MPC, horizon DQN, dueling horizon DQN, TD3 weights, TD3 residual, TD3 Markov, and the new combined horizon + Markov + weights + residual supervisor."
    )
    lines.append("")
    lines.append("## Data Provenance")
    lines.append("")
    lines.append(provenance_table)
    lines.append("")
    lines.append(
        "All analyzed runs use `run_mode = disturb`, 200 episodes, 800 control steps per episode, and a 10-episode warm-start window. The saved configs use `test_cycle = [False, False, False, False, False]`, so these are training-rollout comparisons rather than held-out evaluation-rollout comparisons."
    )
    lines.append("")
    lines.append("## Controller Defaults Checked")
    lines.append("")
    lines.append(
        "The current polymer family defaults use the enlarged neural networks `[512, 512, 512, 512, 512]` with `gamma = 0.99` for the active DQN/TD3 families. The combined run uses active agents `{horizon: True, markov: True, matrix: False, weights: True, residual: True}`."
    )
    lines.append("")
    lines.append(
        "The standalone and combined Markov controllers use the shared safety profile: `z_bound = 0.05`, protected cap `0.025`, ramp cap `0.035 -> 0.05`, full cap `0.05`, probation cap `0.025`, and vector norm cap `0.075`."
    )
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(
        "The polymer plant controls viscosity `eta` and reactor temperature `T` using coolant flow `Qc` and monomer flow `Qm`. The baseline is offset-free MPC in scaled-deviation coordinates. The RL supervisors modify either the MPC horizon, the output/input penalties, a residual input correction, or a Markov lifted-response correction."
    )
    lines.append("")
    lines.append("The tracking error used for comparable diagnostics is")
    lines.append("")
    lines.append("$$ e_k = y_k - r_k, \\qquad \\tilde e_k = y_{k,\\mathrm{scaled}} - r_{k,\\mathrm{scaled}}. $$")
    lines.append("")
    lines.append("For each method, the report computes tail metrics over the final 10 episodes:")
    lines.append("")
    lines.append("$$ \\mathrm{RMSE}_j = \\sqrt{\\frac{1}{N}\\sum_{k\\in\\mathcal{T}} e_{k,j}^2}, \\qquad R_{\\mathrm{tail}} = \\frac{1}{10}\\sum_{q=191}^{200}\\bar R_q. $$")
    lines.append("")
    lines.append(
        "For Markov methods, the executed correction `z` is analyzed together with the active safety cap. The current polymer Markov profile uses `z_bound = 0.05`, dynamic coordinate caps `0.025 -> 0.05`, and vector norm cap `0.075`."
    )
    lines.append("")
    lines.append("## Main Quantitative Results")
    lines.append("")
    lines.append(summary_table)
    lines.append("")
    lines.append(f"Best tail reward: **{best_reward['method']}** with `{fmt(best_reward['reward_tail10_mean'], 2)}`.")
    lines.append(
        f"Best normalized tail tracking: **{best_scaled['method']}** with scaled RMSE `{fmt(best_scaled['scaled_rmse_tail'], 4)}`."
    )
    if best_eta["method"] == best_temp["method"]:
        lines.append(
            f"Best output-specific tracking: **{best_eta['method']}** has the lowest eta RMSE and the lowest T RMSE."
        )
    else:
        lines.append(
            f"Best output-specific tracking is split: eta RMSE is lowest for **{best_eta['method']}**, while T RMSE is lowest for **{best_temp['method']}**."
        )
    lines.append("")
    lines.append("![Episode reward trends](figures/polymer_latest_family_runs_20260521/fig_reward_learning_curves.png)")
    lines.append("")
    lines.append("![Tail reward comparison](figures/polymer_latest_family_runs_20260521/fig_tail_reward_comparison.png)")
    lines.append("")
    lines.append("![Tail tracking RMSE physical](figures/polymer_latest_family_runs_20260521/fig_tail_tracking_rmse_physical.png)")
    lines.append("")
    lines.append("![Tail tracking RMSE scaled](figures/polymer_latest_family_runs_20260521/fig_tail_tracking_rmse_scaled.png)")
    lines.append("")
    lines.append("## Ranking And Interpretation")
    lines.append("")
    lines.append(rank_table)
    lines.append("")
    lines.append(
        f"Relative to OF-MPC tail reward `{fmt(mpc_tail, 2)}`, the best RL method changes the tail reward by `{fmt(best_reward['reward_tail10_mean'] - mpc_tail, 2)}`. Because the reward combines tracking and input movement in scaled coordinates, the reward ranking should be read together with the RMSE and input-movement plots rather than alone."
    )
    lines.append("")
    lines.append("![Final tail tracking overlay](figures/polymer_latest_family_runs_20260521/fig_final_tail_tracking_overlay.png)")
    lines.append("")
    lines.append("![Tail input movement](figures/polymer_latest_family_runs_20260521/fig_tail_input_movement.png)")
    lines.append("")
    lines.append("## Controller Mechanism Diagnostics")
    lines.append("")
    lines.append(mechanism_table)
    lines.append("")
    lines.append("![Controller mechanism dashboard](figures/polymer_latest_family_runs_20260521/fig_controller_mechanism_dashboard.png)")
    lines.append("")
    lines.append("![Markov z safety tail](figures/polymer_latest_family_runs_20260521/fig_markov_z_safety_tail.png)")
    lines.append("")
    lines.append("![Markov source and projection fractions](figures/polymer_latest_family_runs_20260521/fig_markov_source_projection_fractions.png)")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    combined = by_method["Combined"]
    markov = by_method["TD3 Markov"]
    residual = by_method["TD3 Residual"]
    weights = by_method["TD3 Weights"]
    horizon = by_method["Horizon DQN"]
    dueling = by_method["Dueling Horizon"]
    lines.append(
        f"- The combined controller is the strongest run in this latest batch. Its tail reward `{fmt(combined['reward_tail10_mean'], 2)}` improves over OF-MPC by `{fmt(combined['reward_tail10_mean'] - mpc_tail, 2)}` and improves over Markov-only by `{fmt(combined['reward_tail10_mean'] - markov['reward_tail10_mean'], 2)}`."
    )
    lines.append(
        f"- Markov-only is safe but conservative in performance. Its post-warm TD3 source fraction is `{fmt(mech['TD3 Markov']['markov_td3_source_post_warm'], 3)}`, so TD3 is usually the executed source, but projection activity is `{fmt(mech['TD3 Markov']['markov_projection_active_fraction'], 3)}`. That high projection rate means the raw TD3 request often pushes outside the dynamic safety envelope and is then pulled back before execution."
    )
    lines.append(
        f"- The combined Markov block has a similar q95 absolute coordinate size `{fmt(mech['Combined']['markov_z_q95_abs'], 4)}` to Markov-only `{fmt(mech['TD3 Markov']['markov_z_q95_abs'], 4)}`, but lower projection activity `{fmt(mech['Combined']['markov_projection_active_fraction'], 3)}`. This suggests the other agents are helping the Markov action operate in a less frequently clipped regime."
    )
    lines.append(
        f"- Residual-only has tail scaled RMSE `{fmt(residual['scaled_rmse_tail'], 4)}` and residual correction norm `{fmt(mech['TD3 Residual']['residual_tail_norm_mean'], 5)}`. It is the cleanest way to add direct input authority, but it can also increase input movement if the learned correction is noisy."
    )
    lines.append(
        f"- Weight-only changes the optimizer objective rather than the plant model or input directly. Its tail Q multipliers are Q1 `{fmt(mech['TD3 Weights']['weight_tail_Q1_mean'], 3)}` and Q2 `{fmt(mech['TD3 Weights']['weight_tail_Q2_mean'], 3)}`, which should be read as a learned preference for viscosity versus temperature tracking."
    )
    lines.append(
        f"- Horizon-only and dueling-horizon mainly alter prediction/control horizon selection. Their tail scaled RMSE values are `{fmt(horizon['scaled_rmse_tail'], 4)}` and `{fmt(dueling['scaled_rmse_tail'], 4)}`, so their benefit is limited if the fixed-horizon OF-MPC baseline is already near the best reachable horizon tradeoff."
    )
    lines.append(
        f"- Both horizon agents and the combined horizon block used all `{fmt(mech['Horizon DQN']['horizon_distinct_actions'], 0)}` horizon recipes after warm start. This is a warning that the training rollout is still exploratory rather than a clean frozen-policy horizon schedule."
    )
    lines.append("")
    lines.append("## Bugs, Inconsistencies, And Risks")
    lines.append("")
    lines.append(
        "- The current comparison is not a held-out evaluation because the saved test cycle is all false. This is useful for learning-progress diagnosis, but the next scientific claim should use a fixed test schedule or frozen-policy replay."
    )
    lines.append(
        "- Single-seed results are not enough to separate true controller improvement from exploration noise, especially after increasing network size to `[512, 512, 512, 512, 512]`."
    )
    lines.append(
        "- The combined runner now executes multiple high-authority mechanisms together. Even when each individual layer is safe, interactions between changed horizon, changed Q/R, Markov model correction, and residual input correction can create non-additive behavior."
    )
    lines.append(
        "- Reward and tracking can disagree. If a method improves one output but increases input movement or another output's error, the scalar reward may hide the mechanism."
    )
    lines.append("")
    lines.append("## Figure Audit")
    lines.append("")
    lines.append(
        "The generated figures use the saved `input_data.pkl` bundles listed in the provenance table and were visually checked after generation. Output overlays include setpoints in physical units. Tracking RMSE figures separate viscosity and temperature because they have different physical scales. Markov safety figures show both executed `z` norm and active cap/projection fractions so the safety mechanism is visible rather than inferred."
    )
    lines.append("")
    lines.append("## Literature Context")
    lines.append("")
    lines.append(
        "No new literature citations were added in this report. The goal here is an internal empirical audit of the latest saved polymer runs. If this report is later moved into a paper section, the likely literature links are safe RL/MPC filtering, residual RL for process control, and adaptive/value-augmented MPC."
    )
    lines.append("")
    lines.append("## Recommended Next Experiments")
    lines.append("")
    lines.append(
        "1. Run a frozen-policy test pass for OF-MPC and each trained RL family using the same disturbance and setpoint schedule. Metric to watch: tail scaled RMSE and tail reward without exploration."
    )
    lines.append(
        "2. Run three seeds for Markov-only and combined with the current `[512]*5` network. Metric to watch: seed spread in tail reward and Markov source/projection fractions."
    )
    lines.append(
        "3. Add an ablation of combined without residual and combined without weights. Metric to watch: whether Markov plus horizon is already sufficient, or whether residual/weights add measurable benefit."
    )
    lines.append(
        "4. For combined, log component-wise reward terms if possible. Metric to watch: whether reward loss comes from tracking, input movement, or constraint/saturation behavior."
    )
    lines.append("")
    lines.append("## Remaining Uncertainty")
    lines.append("")
    lines.append(
        "The report is based on completed saved bundles, not rerun simulations. It does not prove generalization because the current runs are training rollouts and appear to use one seed. The strongest next step is a frozen-policy evaluation pass with the same disturbance schedule across all controllers."
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    summary_rows = []
    mechanism_rows = []
    for label, spec in RUNS.items():
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing run bundle for {label}: {spec['path']}")
        result = load_run(label, spec)
        results[label] = result
        summary_rows.append(result["metrics"])
        mechanism_rows.append(result["mechanism"])

    mpc_tail = summary_rows[0]["reward_tail10_mean"]
    mpc_scaled = summary_rows[0]["scaled_rmse_tail"]
    for row in summary_rows:
        row["tail_reward_delta_vs_mpc"] = float(row["reward_tail10_mean"] - mpc_tail)
        row["tail_scaled_rmse_delta_vs_mpc"] = float(row["scaled_rmse_tail"] - mpc_scaled)

    summary_fields = [
        "method",
        "family",
        "path",
        "nFE",
        "episodes",
        "time_in_sub_episodes",
        "warm_start_episode_count",
        "reward_mean",
        "reward_post_warm_mean",
        "reward_tail10_mean",
        "reward_final_episode",
        "tail_reward_delta_vs_mpc",
        "scaled_rmse_tail",
        "tail_scaled_rmse_delta_vs_mpc",
        "eta_rmse_tail",
        "T_rmse_tail",
        "eta_mae_tail",
        "T_mae_tail",
        "eta_max_abs_tail",
        "T_max_abs_tail",
        "mean_abs_delta_u_scaled_tail",
        "input_move_mean_tail",
    ]
    mechanism_fields = list(mechanism_rows[0].keys())
    save_csv(FIG_DIR / "polymer_latest_summary_metrics.csv", summary_rows, summary_fields)
    save_csv(FIG_DIR / "polymer_latest_mechanism_metrics.csv", mechanism_rows, mechanism_fields)
    (FIG_DIR / "summary.json").write_text(
        json.dumps(
            {
                "summary_rows": summary_rows,
                "mechanism_rows": mechanism_rows,
                "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    make_figures(results, summary_rows, mechanism_rows)
    write_report(summary_rows, mechanism_rows)
    print(f"Wrote report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote figures: {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
