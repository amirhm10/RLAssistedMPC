"""Analyze the latest distillation Markov SG-TD3 run.

This script reads saved result bundles only. It does not launch Aspen or rerun
any controller. The outputs are report-specific CSV summaries, a manifest, and
figures under report/figures/distillation_markov_sg_td3_20260603/.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "figures" / "distillation_markov_sg_td3_20260603"

SG_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_unified"
    / "20260602_192543"
    / "input_data.pkl"
)
SG_COMPARE_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation"
    / "20260602_192558"
    / "input_data.pkl"
)
TD3_FULL_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_current_reward_unified"
    / "20260601_214256"
    / "input_data.pkl"
)
TD3_FULL_COMPARE_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_markov_td3_disturb_fluctuation_td3_only_no_safeguard_current_reward"
    / "20260601_214316"
    / "input_data.pkl"
)
OFMPC_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"

METHOD_COLORS = {
    "SG-TD3": "#1b9e77",
    "TD3-full": "#d95f02",
    "OF-MPC": "#4d4d4d",
}
OUTPUT_NAMES = ["x24 ethane", "T85"]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_quantile(values, q: float) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def apply_min_max(data, min_val, max_val):
    data = np.asarray(data, float)
    min_val = np.asarray(min_val, float)
    max_val = np.asarray(max_val, float)
    return (data - min_val) / np.maximum(max_val - min_val, 1.0e-12)


def reverse_min_max(data, min_val, max_val):
    data = np.asarray(data, float)
    min_val = np.asarray(min_val, float)
    max_val = np.asarray(max_val, float)
    return data * (max_val - min_val) + min_val


def pick_array(bundle: dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in bundle and bundle[key] is not None:
            return np.asarray(bundle[key], float)
    raise KeyError(f"None of these keys are present: {keys}")


def n_steps(bundle: dict) -> int:
    if "nFE" in bundle:
        return int(bundle["nFE"])
    if "y_sp" in bundle:
        return int(np.asarray(bundle["y_sp"]).shape[0])
    return int(pick_array(bundle, "u_mpc", "u_rl", "u").shape[0])


def episode_len(bundle: dict) -> int:
    return int(bundle.get("time_in_sub_episodes", 400))


def warm_step(bundle: dict) -> int:
    if "warm_start_step" in bundle:
        return int(bundle["warm_start_step"])
    return int(bundle.get("warm_start", 10)) * episode_len(bundle)


def y_line(bundle: dict, n: int | None = None) -> np.ndarray:
    n = n_steps(bundle) if n is None else int(n)
    y = pick_array(bundle, "y_line_full", "y_rl", "y_mpc", "y")
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] >= n + 1:
        return y[: n + 1, :]
    if y.shape[0] == n:
        return np.vstack([y, y[-1:, :]])
    pad = np.repeat(y[-1:, :], n + 1 - y.shape[0], axis=0)
    return np.vstack([y, pad])


def u_step(bundle: dict, n: int | None = None) -> np.ndarray:
    n = n_steps(bundle) if n is None else int(n)
    u = pick_array(bundle, "u_mpc", "u_rl", "u")
    if u.ndim == 1:
        u = u[:, None]
    if u.shape[0] >= n:
        return u[:n, :]
    pad = np.repeat(u[-1:, :], n - u.shape[0], axis=0)
    return np.vstack([u, pad])


def y_sp_phys(bundle: dict, n: int | None = None) -> np.ndarray:
    n = n_steps(bundle) if n is None else int(n)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = apply_min_max(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    ysp = np.asarray(bundle["y_sp"], float)[:n, :]
    return reverse_min_max(ysp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def delta_u_scaled(bundle: dict, start: int, end: int) -> np.ndarray:
    if "delta_u_storage" in bundle and bundle["delta_u_storage"] is not None:
        return np.asarray(bundle["delta_u_storage"], float)[start:end, :]
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    u = u_step(bundle)
    u_scaled = apply_min_max(u, data_min[:n_inputs], data_max[:n_inputs])
    ss_inputs = np.asarray(bundle["steady_states"]["ss_inputs"], float)
    ss_scaled = apply_min_max(ss_inputs, data_min[:n_inputs], data_max[:n_inputs])
    du = np.zeros_like(u_scaled)
    du[0, :] = u_scaled[0, :] - ss_scaled
    du[1:, :] = u_scaled[1:, :] - u_scaled[:-1, :]
    return du[start:end, :]


def reward_summary(method: str, avg_rewards: np.ndarray, warm_ep: int) -> dict:
    avg = np.asarray(avg_rewards, float)
    return {
        "method": method,
        "mean_reward": finite_mean(avg),
        "worst_post_warm_reward": float(np.nanmin(avg[warm_ep:])),
        "worst_first20_post_warm_reward": float(np.nanmin(avg[warm_ep : warm_ep + 20])),
        "tail20_reward": finite_mean(avg[-20:]),
        "final_reward": float(avg[-1]),
    }


def tracking_rows(method: str, bundle: dict, window: str, start: int, end: int) -> list[dict]:
    n = n_steps(bundle)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    y = y_line(bundle, n)
    ysp = y_sp_phys(bundle, n)
    err = y[start + 1 : end + 1, :] - ysp[start:end, :]
    du = delta_u_scaled(bundle, start, end)
    move = float(np.mean(np.abs(du))) if du.size else float("nan")
    rows = []
    for idx, output in enumerate(OUTPUT_NAMES):
        e = err[:, idx]
        rows.append(
            {
                "method": method,
                "window": window,
                "output": output,
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "max_abs": float(np.max(np.abs(e))),
                "mean_signed": float(np.mean(e)),
                "final_abs": float(abs(e[-1])),
                "mean_abs_du_scaled": move,
            }
        )
    return rows


def episode_segments(bundle: dict) -> list[tuple[int, int, str]]:
    ep_len = episode_len(bundle)
    ysp = np.asarray(bundle["y_sp"], float)[:ep_len, :]
    changes = np.where(np.any(np.abs(np.diff(ysp, axis=0)) > 1.0e-12, axis=1))[0] + 1
    cuts = [0] + [int(v) for v in changes] + [ep_len]
    return [(cuts[i], cuts[i + 1], f"SP{i + 1}") for i in range(len(cuts) - 1)]


def blockwise_rows(method: str, bundle: dict, episode_ids: list[int], window: str) -> list[dict]:
    ep_len = episode_len(bundle)
    y = y_line(bundle)
    ysp = y_sp_phys(bundle)
    rows = []
    for seg_start, seg_end, block in episode_segments(bundle):
        idx_parts = [np.arange(ep * ep_len + seg_start, ep * ep_len + seg_end) for ep in episode_ids]
        idx = np.concatenate(idx_parts)
        idx = idx[(idx >= 0) & (idx < ysp.shape[0]) & (idx + 1 < y.shape[0])]
        err = y[idx + 1, :] - ysp[idx, :]
        for out_idx, output in enumerate(OUTPUT_NAMES):
            e = err[:, out_idx]
            rows.append(
                {
                    "method": method,
                    "window": window,
                    "block": block,
                    "output": output,
                    "steps": int(e.size),
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "max_abs": float(np.max(np.abs(e))),
                    "mean_signed": float(np.mean(e)),
                }
            )
    return rows


def source_fraction_rows(bundle: dict, window: str, start: int, end: int) -> list[dict]:
    source = np.asarray(bundle.get("rl_action_source_log", []), int).reshape(-1)
    names = {int(k): str(v) for k, v in dict(bundle.get("rl_action_source_names", {})).items()}
    end = min(end, source.size)
    sub = source[start:end]
    rows = []
    for code, name in sorted(names.items()):
        rows.append(
            {
                "method": "SG-TD3",
                "window": window,
                "source": name,
                "fraction": float(np.mean(sub == code)) if sub.size else float("nan"),
            }
        )
    supervisor = np.isin(sub, [6, 7, 8])
    rows.append(
        {
            "method": "SG-TD3",
            "window": window,
            "source": "all_supervisor",
            "fraction": float(np.mean(supervisor)) if sub.size else float("nan"),
        }
    )
    return rows


def sg_gate_summary_rows(bundle: dict, window: str, start: int, end: int) -> list[dict]:
    rows = []
    keys = {
        "sg_advantage_log": "advantage",
        "sg_score_policy_log": "policy_score",
        "sg_score_supervisor_log": "supervisor_score",
        "sg_q_gap_policy_log": "policy_q_gap",
        "sg_q_gap_supervisor_log": "supervisor_q_gap",
    }
    for key, label in keys.items():
        arr = np.asarray(bundle.get(key, []), float)[start:end]
        rows.extend(
            [
                {"window": window, "metric": f"{label}_mean", "value": finite_mean(arr)},
                {"window": window, "metric": f"{label}_q10", "value": finite_quantile(arr, 0.10)},
                {"window": window, "metric": f"{label}_q90", "value": finite_quantile(arr, 0.90)},
            ]
        )
    policy_score = np.asarray(bundle.get("sg_score_policy_log", []), float)[start:end]
    supervisor_score = np.asarray(bundle.get("sg_score_supervisor_log", []), float)[start:end]
    rows.append(
        {
            "window": window,
            "metric": "policy_score_gt_supervisor_fraction",
            "value": float(np.mean(policy_score > supervisor_score)) if policy_score.size else float("nan"),
        }
    )
    return rows


def episode_mean(bundle: dict, key: str, reducer="mean") -> np.ndarray:
    arr = np.asarray(bundle.get(key, []), float)
    ep_len = episode_len(bundle)
    n_ep = n_steps(bundle) // ep_len
    out = np.full(n_ep, np.nan, dtype=float)
    if arr.size == 0:
        return out
    arr = arr[: n_ep * ep_len]
    if arr.ndim == 1:
        reshaped = arr.reshape(n_ep, ep_len)
    else:
        reshaped = np.linalg.norm(arr.reshape(n_ep, ep_len, arr.shape[1]), axis=2)
    for idx in range(n_ep):
        values = reshaped[idx]
        if reducer == "fraction_positive":
            out[idx] = float(np.mean(values > 0.0))
        else:
            out[idx] = finite_mean(values)
    return out


def episode_diagnostics_rows(method: str, bundle: dict, avg_rewards: np.ndarray) -> list[dict]:
    ep_len = episode_len(bundle)
    n_ep = n_steps(bundle) // ep_len
    src = np.asarray(bundle.get("rl_action_source_log", []), int)
    z_norm = episode_mean(bundle, "z_executed_log")
    du_norm = episode_mean(bundle, "delta_u_storage")
    adv = episode_mean(bundle, "sg_advantage_log")
    policy_score = episode_mean(bundle, "sg_score_policy_log")
    supervisor_score = episode_mean(bundle, "sg_score_supervisor_log")
    rows = []
    for ep in range(n_ep):
        start = ep * ep_len
        end = min(start + ep_len, src.size)
        sub = src[start:end]
        rows.append(
            {
                "method": method,
                "episode": ep + 1,
                "avg_reward": float(avg_rewards[ep]) if ep < len(avg_rewards) else float("nan"),
                "td3_fraction": float(np.mean(sub == 2)) if sub.size else float("nan"),
                "supervisor_fraction": float(np.mean(np.isin(sub, [6, 7, 8]))) if sub.size else float("nan"),
                "z_executed_norm_mean": float(z_norm[ep]) if ep < len(z_norm) else float("nan"),
                "delta_u_norm_mean": float(du_norm[ep]) if ep < len(du_norm) else float("nan"),
                "sg_advantage_mean": float(adv[ep]) if ep < len(adv) else float("nan"),
                "sg_policy_score_mean": float(policy_score[ep]) if ep < len(policy_score) else float("nan"),
                "sg_supervisor_score_mean": float(supervisor_score[ep]) if ep < len(supervisor_score) else float("nan"),
            }
        )
    return rows


def plot_reward_curves(reward_data: dict[str, np.ndarray], warm_ep: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for method, values in reward_data.items():
        x = np.arange(1, len(values) + 1)
        style = "--" if method == "OF-MPC" else "-"
        ax.plot(x, values, label=method, color=METHOD_COLORS[method], linewidth=2.0, linestyle=style)
    ax.axvspan(1, warm_ep, color="#bdbdbd", alpha=0.22, label="warm start")
    ax.axvspan(len(next(iter(reward_data.values()))) - 19, len(next(iter(reward_data.values()))), color="#80cdc1", alpha=0.20, label="tail 20")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_title("Distillation Markov reward comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = OUT_DIR / "fig_reward_curves_sg_vs_td3full_vs_ofmpc.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def plot_reward_bars(summary_rows: list[dict]) -> str:
    metrics = [
        ("tail20_reward", "Tail-20"),
        ("final_reward", "Final"),
        ("worst_first20_post_warm_reward", "Worst first 20 post-warm"),
        ("worst_post_warm_reward", "Worst post-warm"),
    ]
    methods = ["SG-TD3", "TD3-full", "OF-MPC"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    by_method = {row["method"]: row for row in summary_rows}
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = [by_method[m][key] for m in methods]
        ax.bar(methods, vals, color=[METHOD_COLORS[m] for m in methods])
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = OUT_DIR / "fig_reward_summary_bars.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def plot_tracking_window(method_bundles: dict[str, dict], window: str, start: int, end: int, filename: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    base = method_bundles["SG-TD3"]
    x = np.arange(start, end) / episode_len(base)
    ysp = y_sp_phys(base)[start:end, :]
    for out_idx, ax in enumerate(axes):
        ax.plot(x, ysp[:, out_idx], color="black", linestyle=":", linewidth=1.8, label="setpoint")
        for method, bundle in method_bundles.items():
            y = y_line(bundle)
            yy = y[start + 1 : end + 1, out_idx]
            ax.plot(x[: yy.size], yy, label=method, color=METHOD_COLORS[method], linewidth=1.5)
        ax.set_ylabel(OUTPUT_NAMES[out_idx])
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{window} tracking - {OUTPUT_NAMES[out_idx]}")
    axes[-1].set_xlabel("Episode index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def plot_blockwise_mae(block_rows: list[dict]) -> str:
    methods = ["SG-TD3", "TD3-full", "OF-MPC"]
    blocks = sorted({row["block"] for row in block_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, output in zip(axes, OUTPUT_NAMES):
        labels = blocks
        x = np.arange(len(labels))
        width = 0.25
        for idx, method in enumerate(methods):
            vals = [
                next(row["mae"] for row in block_rows if row["method"] == method and row["block"] == block and row["output"] == output)
                for block in labels
            ]
            ax.bar(x + (idx - 1) * width, vals, width, label=method, color=METHOD_COLORS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(output)
        ax.set_ylabel("Tail-20 MAE")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    path = OUT_DIR / "fig_tail_blockwise_mae.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def plot_source_fractions(source_rows: list[dict]) -> str:
    windows = ["first20_postwarm", "postwarm", "tail20"]
    sources = ["td3_accepted", "sg_supervisor_ls", "sg_supervisor_mpc"]
    colors = ["#1b9e77", "#7570b3", "#e7298a"]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bottoms = np.zeros(len(windows))
    for source, color in zip(sources, colors):
        vals = [
            next((row["fraction"] for row in source_rows if row["window"] == window and row["source"] == source), 0.0)
            for window in windows
        ]
        ax.bar(windows, vals, bottom=bottoms, label=source, color=color)
        bottoms += np.asarray(vals, float)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of steps")
    ax.set_title("SG-TD3 action source fractions")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "fig_action_source_fractions.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def plot_sg_advantage_and_scores(bundle: dict, avg_rewards: np.ndarray) -> str:
    ep = np.arange(1, len(avg_rewards) + 1)
    adv = episode_mean(bundle, "sg_advantage_log")
    policy = episode_mean(bundle, "sg_score_policy_log")
    supervisor = episode_mean(bundle, "sg_score_supervisor_log")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    axes[0].plot(ep, avg_rewards, color=METHOD_COLORS["SG-TD3"], linewidth=2.0, label="reward")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("SG-TD3 reward and critic-gate diagnostics")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(ep, adv[: ep.size], color="#1b9e77", linewidth=1.8, label="policy - supervisor score")
    axes[1].plot(ep, policy[: ep.size], color="#d95f02", alpha=0.75, linewidth=1.2, label="policy score")
    axes[1].plot(ep, supervisor[: ep.size], color="#7570b3", alpha=0.75, linewidth=1.2, label="supervisor score")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3)
    fig.tight_layout()
    path = OUT_DIR / "fig_sg_advantage_and_scores.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def plot_markov_norm_and_move(method_bundles: dict[str, dict], reward_data: dict[str, np.ndarray]) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for method in ["SG-TD3", "TD3-full"]:
        bundle = method_bundles[method]
        ep = np.arange(1, len(reward_data[method]) + 1)
        axes[0].plot(ep, episode_mean(bundle, "z_executed_log")[: ep.size], label=method, color=METHOD_COLORS[method], linewidth=1.8)
        axes[1].plot(ep, episode_mean(bundle, "delta_u_storage")[: ep.size], label=method, color=METHOD_COLORS[method], linewidth=1.8)
    axes[0].set_ylabel("Mean ||z||2")
    axes[0].set_title("Markov correction norm and input movement")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_ylabel("Mean ||delta u||2")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    path = OUT_DIR / "fig_markov_correction_norm_and_input_move.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rel(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sg = load_pickle(SG_PATH)
    sg_compare = load_pickle(SG_COMPARE_PATH)
    td3_full = load_pickle(TD3_FULL_PATH)
    td3_compare = load_pickle(TD3_FULL_COMPARE_PATH)
    ofmpc = load_pickle(OFMPC_PATH)

    ep_len = episode_len(sg)
    n = n_steps(sg)
    warm = warm_step(sg)
    warm_ep = warm // ep_len
    first20_end = warm + 20 * ep_len
    tail_start = n - 20 * ep_len
    final_start = n - ep_len

    reward_data = {
        "SG-TD3": np.asarray(sg_compare["avg_rewards_rl"], float),
        "TD3-full": np.asarray(td3_compare["avg_rewards_rl"], float),
        "OF-MPC": np.asarray(sg_compare["avg_rewards_mpc"], float),
    }
    method_bundles = {"SG-TD3": sg, "TD3-full": td3_full, "OF-MPC": ofmpc}

    reward_rows = [reward_summary(method, rewards, warm_ep) for method, rewards in reward_data.items()]

    tracking = []
    for method, bundle in method_bundles.items():
        tracking.extend(tracking_rows(method, bundle, "first20_postwarm", warm, first20_end))
        tracking.extend(tracking_rows(method, bundle, "tail20", tail_start, n))
        tracking.extend(tracking_rows(method, bundle, "final_episode", final_start, n))

    tail_episode_ids = list(range(n // ep_len - 20, n // ep_len))
    block_rows = []
    for method, bundle in method_bundles.items():
        block_rows.extend(blockwise_rows(method, bundle, tail_episode_ids, "tail20"))

    windows = {
        "first20_postwarm": (warm, first20_end),
        "postwarm": (warm, n),
        "tail20": (tail_start, n),
    }
    source_rows = []
    gate_rows = []
    for window, (start, end) in windows.items():
        source_rows.extend(source_fraction_rows(sg, window, start, end))
        gate_rows.extend(sg_gate_summary_rows(sg, window, start, end))

    episode_rows = episode_diagnostics_rows("SG-TD3", sg, reward_data["SG-TD3"])

    figure_paths = {
        "reward_curves": plot_reward_curves(reward_data, warm_ep),
        "reward_summary": plot_reward_bars(reward_rows),
        "early_release_tracking": plot_tracking_window(method_bundles, "First 20 post-warm episodes", warm, first20_end, "fig_early_release_tracking_zoom.png"),
        "tail_tracking": plot_tracking_window(method_bundles, "Final tail episode", final_start, n, "fig_tail_tracking_overlay.png"),
        "tail_blockwise_mae": plot_blockwise_mae(block_rows),
        "action_source_fractions": plot_source_fractions(source_rows),
        "sg_advantage_and_scores": plot_sg_advantage_and_scores(sg, reward_data["SG-TD3"]),
        "markov_correction_norm_and_input_move": plot_markov_norm_and_move(method_bundles, reward_data),
    }

    csv_paths = {
        "reward_summary": OUT_DIR / "reward_summary.csv",
        "tracking_summary": OUT_DIR / "tracking_summary.csv",
        "tail_blockwise_tracking": OUT_DIR / "tail_blockwise_tracking_summary.csv",
        "action_source_summary": OUT_DIR / "action_source_summary.csv",
        "sg_gate_summary": OUT_DIR / "sg_gate_summary.csv",
        "episode_diagnostics": OUT_DIR / "episode_diagnostics.csv",
    }
    write_csv(csv_paths["reward_summary"], reward_rows)
    write_csv(csv_paths["tracking_summary"], tracking)
    write_csv(csv_paths["tail_blockwise_tracking"], block_rows)
    write_csv(csv_paths["action_source_summary"], source_rows)
    write_csv(csv_paths["sg_gate_summary"], gate_rows)
    write_csv(csv_paths["episode_diagnostics"], episode_rows)

    manifest = {
        "sources": {
            "sg_td3": rel(SG_PATH),
            "sg_td3_compare": rel(SG_COMPARE_PATH),
            "td3_full": rel(TD3_FULL_PATH),
            "td3_full_compare": rel(TD3_FULL_COMPARE_PATH),
            "ofmpc": rel(OFMPC_PATH),
        },
        "windows": {
            "warm_start_step": int(warm),
            "episode_length": int(ep_len),
            "first20_postwarm": [int(warm), int(first20_end)],
            "tail20": [int(tail_start), int(n)],
            "final_episode": [int(final_start), int(n)],
        },
        "figures": figure_paths,
        "csv": {name: rel(path) for name, path in csv_paths.items()},
        "headline": {
            "sg_tail20_reward": next(row["tail20_reward"] for row in reward_rows if row["method"] == "SG-TD3"),
            "ofmpc_tail20_reward": next(row["tail20_reward"] for row in reward_rows if row["method"] == "OF-MPC"),
            "sg_final_reward": next(row["final_reward"] for row in reward_rows if row["method"] == "SG-TD3"),
            "ofmpc_final_reward": next(row["final_reward"] for row in reward_rows if row["method"] == "OF-MPC"),
            "sg_worst_first20_postwarm": next(row["worst_first20_post_warm_reward"] for row in reward_rows if row["method"] == "SG-TD3"),
            "td3_full_worst_first20_postwarm": next(row["worst_first20_post_warm_reward"] for row in reward_rows if row["method"] == "TD3-full"),
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
