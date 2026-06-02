"""Analyze the 2026-06-02 distillation residual SG-TD3 critic-warm run.

The script reads saved bundles only. It compares the new zero-residual
supervisor-gated TD3 run against the current OF-MPC reward recalculation and
the 2026-06-01 residual TD3/TD7 runs that showed collapse after safety release.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "figures" / "distillation_residual_sg_td3_20260602"

K_REL = np.array([0.3, 0.01], dtype=float)
BAND_FLOOR_PHYS = np.array([0.003, 0.2], dtype=float)
SOURCE_NAMES = {
    0: "warm_start",
    1: "supervisor",
    2: "policy",
    3: "held",
    4: "fallback",
}

RUNS = {
    "baseline": {
        "label": "OF-MPC",
        "method": "baseline",
        "path": ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    },
    "td3": {
        "label": "TD3 residual",
        "method": "residual_td3",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260601_170240"
        / "input_data.pkl",
    },
    "td7": {
        "label": "TD7 residual",
        "method": "residual_td7",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td7_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260601_172611"
        / "input_data.pkl",
    },
    "sg_td3": {
        "label": "SG-TD3 residual",
        "method": "residual_sg_td3",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho"
        / "20260602_125954"
        / "input_data.pkl",
    },
}

COMPARE_PATHS = {
    "baseline_current_reward": ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation"
    / "20260602_130007"
    / "input_data.pkl",
    "td3_compare": ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_residual_td3_disturb_fluctuation"
    / "20260601_170255"
    / "input_data.pkl",
    "td7_compare": ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_residual_td7_disturb_fluctuation"
    / "20260601_172624"
    / "input_data.pkl",
}

WINDOWS = {
    "warm_1_10": (1, 10),
    "critic_warm_11_13": (11, 13),
    "sg_live_or_old_guard_14_30": (14, 30),
    "old_collapse_31_40": (31, 40),
    "recovery_41_80": (41, 80),
    "tail_181_200": (181, 200),
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def finite_mean(values) -> float:
    if values is None:
        return math.nan
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def finite_quantile(values, q: float) -> float:
    if values is None:
        return math.nan
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else math.nan


def minmax_scale(data, min_val, max_val):
    return (np.asarray(data, float) - np.asarray(min_val, float)) / (
        np.asarray(max_val, float) - np.asarray(min_val, float)
    )


def reverse_minmax(data, min_val, max_val):
    return np.asarray(data, float) * (np.asarray(max_val, float) - np.asarray(min_val, float)) + np.asarray(
        min_val, float
    )


def episode_len(bundle: dict) -> int:
    return int(bundle.get("time_in_sub_episodes", 400))


def n_episodes(bundle: dict) -> int:
    avg = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
    if avg.size:
        return int(avg.size)
    return int(bundle.get("nFE", 0)) // max(1, episode_len(bundle))


def episode_slice(bundle: dict, ep_start: int, ep_end: int) -> slice:
    ep_len = episode_len(bundle)
    return slice((ep_start - 1) * ep_len, ep_end * ep_len)


def episode_mean(values: np.ndarray, ep_len: int) -> np.ndarray:
    arr = np.asarray(values, float)
    n_ep = arr.shape[0] // ep_len
    if n_ep <= 0:
        return np.asarray([], float)
    reshaped = arr[: n_ep * ep_len].reshape(n_ep, ep_len, *arr.shape[1:])
    return np.nanmean(reshaped, axis=1)


def episode_fraction(values: np.ndarray, ep_len: int, predicate) -> np.ndarray:
    arr = np.asarray(values)
    n_ep = arr.shape[0] // ep_len
    if n_ep <= 0:
        return np.asarray([], float)
    mask = predicate(arr[: n_ep * ep_len]).reshape(n_ep, ep_len)
    return mask.mean(axis=1)


def y_sp_phys(bundle: dict) -> np.ndarray:
    y_sp = np.asarray(bundle["y_sp"], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    n_inputs = 2
    y_ss_scaled = minmax_scale(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    return reverse_minmax(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def y_phys_line(bundle: dict) -> np.ndarray:
    for key in ("y_line_full", "y", "y_rl", "y_mpc"):
        value = bundle.get(key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError("No output trajectory found.")


def u_phys_step(bundle: dict) -> np.ndarray:
    for key in ("u_step_full", "u", "u_rl", "u_mpc"):
        value = bundle.get(key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError("No input trajectory found.")


def step_error_phys(bundle: dict):
    y_line = y_phys_line(bundle)
    ysp = y_sp_phys(bundle)
    n = min(y_line.shape[0] - 1, ysp.shape[0])
    y_step = y_line[1 : n + 1, :]
    ysp_step = ysp[:n, :]
    return y_step - ysp_step, y_step, ysp_step


def avg_rewards(bundle: dict) -> np.ndarray:
    return np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)


def reward_with_baseline_current(bundles: dict[str, dict]) -> None:
    compare = load_pickle(COMPARE_PATHS["baseline_current_reward"])
    baseline = dict(bundles["baseline"])
    baseline["avg_rewards"] = np.asarray(compare["avg_rewards_mpc"], float)
    baseline["avg_rewards_mpc"] = np.asarray(compare["avg_rewards_mpc"], float)
    bundles["baseline"] = baseline


def first_recovery_episode(avg: np.ndarray, baseline: np.ndarray, start_idx: int, window: int = 5) -> int | None:
    if avg.size == 0 or baseline.size == 0:
        return None
    limit = min(avg.size, baseline.size) - window + 1
    for idx in range(max(0, start_idx + 1), limit):
        if float(np.nanmean(avg[idx : idx + window])) >= float(np.nanmean(baseline[idx : idx + window])):
            return int(idx + 1)
    return None


def window_tracking_metrics(bundle: dict, ep_start: int, ep_end: int) -> dict[str, float]:
    err, _y, ysp = step_error_phys(bundle)
    sl = episode_slice(bundle, ep_start, ep_end)
    e = err[sl, :]
    ysp_w = ysp[sl, :]
    band = np.maximum(K_REL * np.abs(ysp_w), BAND_FLOOR_PHYS)
    abs_e = np.abs(e)
    return {
        "comp_mae": finite_mean(abs_e[:, 0]),
        "temp_mae": finite_mean(abs_e[:, 1]),
        "comp_rmse": float(np.sqrt(finite_mean(e[:, 0] ** 2))),
        "temp_rmse": float(np.sqrt(finite_mean(e[:, 1] ** 2))),
        "band_norm_mae": finite_mean(abs_e / np.maximum(band, 1.0e-12)),
        "outside_band_frac": finite_mean(abs_e > band),
        "comp_max_abs": finite_quantile(abs_e[:, 0], 1.0),
        "temp_max_abs": finite_quantile(abs_e[:, 1], 1.0),
    }


def residual_metrics(bundle: dict, ep_start: int, ep_end: int) -> dict[str, float]:
    if bundle.get("method_family") != "residual" and "delta_u_res_exec_log" not in bundle:
        return {}
    sl = episode_slice(bundle, ep_start, ep_end)
    requested = np.linalg.norm(np.asarray(bundle["delta_u_res_requested_log"], float), axis=1)
    executed = np.linalg.norm(np.asarray(bundle["delta_u_res_exec_log"], float), axis=1)
    post_cap = np.linalg.norm(np.asarray(bundle.get("delta_u_res_post_cap_log", bundle["delta_u_res_exec_log"]), float), axis=1)
    gap = np.asarray(bundle.get("policy_executed_gap_norm_log", np.full_like(executed, np.nan)), float)
    out = {
        "requested_residual_norm_mean": finite_mean(requested[sl]),
        "requested_residual_norm_q95": finite_quantile(requested[sl], 0.95),
        "post_cap_residual_norm_mean": finite_mean(post_cap[sl]),
        "executed_residual_norm_mean": finite_mean(executed[sl]),
        "executed_residual_norm_q95": finite_quantile(executed[sl], 0.95),
        "policy_executed_gap_norm_mean": finite_mean(gap[sl]),
        "cap_projection_frac": finite_mean(np.asarray(bundle.get("residual_cap_projection_active_log", []))[sl]),
        "guard_active_frac": finite_mean(np.asarray(bundle.get("residual_guard_active_log", []))[sl]),
        "guard_trigger_frac": finite_mean(np.asarray(bundle.get("residual_guard_triggered_log", []))[sl]),
        "headroom_projection_frac": finite_mean(np.asarray(bundle.get("projection_due_to_headroom_log", []))[sl]),
        "nonfinite_zero_fallback_frac": finite_mean(
            np.asarray(bundle.get("residual_zero_fallback_reason_log", []))[sl] > 0
        ),
    }
    if "sg_selected_source_log" in bundle and bundle["sg_selected_source_log"] is not None:
        src = np.asarray(bundle["sg_selected_source_log"], int)
        for code, name in SOURCE_NAMES.items():
            out[f"sg_{name}_frac"] = finite_mean(src[sl] == code)
        out["sg_policy_score_mean"] = finite_mean(np.asarray(bundle["sg_score_policy_log"], float)[sl])
        out["sg_supervisor_score_mean"] = finite_mean(np.asarray(bundle["sg_score_supervisor_log"], float)[sl])
        out["sg_advantage_mean"] = finite_mean(np.asarray(bundle["sg_advantage_log"], float)[sl])
        out["sg_policy_q_gap_mean"] = finite_mean(np.asarray(bundle["sg_q_gap_policy_log"], float)[sl])
        out["sg_supervisor_q_gap_mean"] = finite_mean(np.asarray(bundle["sg_q_gap_supervisor_log"], float)[sl])
    return out


def summary_rows(bundles: dict[str, dict]) -> pd.DataFrame:
    baseline_avg = avg_rewards(bundles["baseline"])
    rows = []
    for key, meta in RUNS.items():
        bundle = bundles[key]
        avg = avg_rewards(bundle)
        ep_len = episode_len(bundle)
        warm_episodes = int(bundle.get("warm_start_step", 0) // max(1, ep_len))
        postwarm = avg[warm_episodes:]
        postwarm_min_local = int(np.nanargmin(postwarm)) if postwarm.size else 0
        postwarm_min_idx = warm_episodes + postwarm_min_local
        action_freeze_episodes = int(
            bundle.get("post_warm_start_action_freeze_subepisodes", bundle.get("phase1_action_freeze_subepisodes", 0))
        )
        actor_freeze_episodes = int(
            bundle.get("post_warm_start_actor_freeze_subepisodes", bundle.get("phase1_actor_freeze_subepisodes", 0))
        )
        release_start_ep = warm_episodes + action_freeze_episodes + 1
        release_start_idx = max(0, release_start_ep - 1)
        release_avg = avg[release_start_idx:]
        release_min_local = int(np.nanargmin(release_avg)) if release_avg.size else 0
        release_min_idx = release_start_idx + release_min_local

        row = {
            "key": key,
            "label": meta["label"],
            "method": meta["method"],
            "path": str(meta["path"].relative_to(ROOT)),
            "agent_kind": bundle.get("agent_kind", "of_mpc"),
            "notebook_source": bundle.get("notebook_source", "baseline"),
            "warm_episodes": warm_episodes,
            "action_freeze_episodes": action_freeze_episodes,
            "actor_freeze_episodes": actor_freeze_episodes,
            "reward_mean_all": finite_mean(avg),
            "reward_tail20": finite_mean(avg[-20:]),
            "reward_tail10": finite_mean(avg[-10:]),
            "reward_final": float(avg[-1]) if avg.size else math.nan,
            "reward_best": float(np.nanmax(avg)) if avg.size else math.nan,
            "postwarm_min_reward": float(avg[postwarm_min_idx]) if avg.size else math.nan,
            "postwarm_min_episode": int(postwarm_min_idx + 1) if avg.size else -1,
            "release_min_reward": float(avg[release_min_idx]) if avg.size else math.nan,
            "release_min_episode": int(release_min_idx + 1) if avg.size else -1,
            "recovery_5ep_ge_baseline_after_postwarm_min": first_recovery_episode(
                avg, baseline_avg, postwarm_min_idx, window=5
            ),
            "recovery_5ep_ge_baseline_after_release_min": first_recovery_episode(
                avg, baseline_avg, release_min_idx, window=5
            ),
            "tail20_reward_delta_vs_baseline": finite_mean(avg[-20:]) - finite_mean(baseline_avg[-20:]),
            "final_reward_delta_vs_baseline": (float(avg[-1]) - float(baseline_avg[-1])) if avg.size else math.nan,
        }
        row.update({f"tail20_{k}": v for k, v in window_tracking_metrics(bundle, 181, 200).items()})
        row.update({f"old_collapse_{k}": v for k, v in window_tracking_metrics(bundle, 31, 40).items()})
        row.update({f"tail20_{k}": v for k, v in residual_metrics(bundle, 181, 200).items()})
        row.update({f"old_collapse_{k}": v for k, v in residual_metrics(bundle, 31, 40).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def window_rows(bundles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, meta in RUNS.items():
        bundle = bundles[key]
        avg = avg_rewards(bundle)
        for window_name, (ep_start, ep_end) in WINDOWS.items():
            ep_sl = slice(ep_start - 1, ep_end)
            row = {
                "key": key,
                "label": meta["label"],
                "window": window_name,
                "episode_start": ep_start,
                "episode_end": ep_end,
                "reward_mean": finite_mean(avg[ep_sl]),
                "reward_min": float(np.nanmin(avg[ep_sl])) if avg.size else math.nan,
                "reward_min_episode": int(ep_start + np.nanargmin(avg[ep_sl])) if avg.size else -1,
            }
            row.update(window_tracking_metrics(bundle, ep_start, ep_end))
            row.update(residual_metrics(bundle, ep_start, ep_end))
            rows.append(row)
    return pd.DataFrame(rows)


def sg_episode_rows(bundle: dict) -> pd.DataFrame:
    ep_len = episode_len(bundle)
    src = np.asarray(bundle["sg_selected_source_log"], int)
    avg = avg_rewards(bundle)
    executed_norm = np.linalg.norm(np.asarray(bundle["delta_u_res_exec_log"], float), axis=1)
    requested_norm = np.linalg.norm(np.asarray(bundle["delta_u_res_requested_log"], float), axis=1)
    gap = np.asarray(bundle["policy_executed_gap_norm_log"], float)
    adv = np.asarray(bundle["sg_advantage_log"], float)
    score_policy = np.asarray(bundle["sg_score_policy_log"], float)
    score_supervisor = np.asarray(bundle["sg_score_supervisor_log"], float)
    n_ep = min(n_episodes(bundle), src.size // ep_len)
    rows = []
    for idx in range(n_ep):
        sl = slice(idx * ep_len, (idx + 1) * ep_len)
        row = {
            "episode": idx + 1,
            "reward": float(avg[idx]) if idx < avg.size else math.nan,
            "requested_residual_norm_mean": finite_mean(requested_norm[sl]),
            "executed_residual_norm_mean": finite_mean(executed_norm[sl]),
            "executed_residual_norm_q95": finite_quantile(executed_norm[sl], 0.95),
            "policy_executed_gap_norm_mean": finite_mean(gap[sl]),
            "sg_advantage_mean": finite_mean(adv[sl]),
            "sg_score_policy_mean": finite_mean(score_policy[sl]),
            "sg_score_supervisor_mean": finite_mean(score_supervisor[sl]),
        }
        for code, name in SOURCE_NAMES.items():
            row[f"source_{name}_frac"] = finite_mean(src[sl] == code)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reward(summary: pd.DataFrame, bundles: dict[str, dict]) -> None:
    colors = {
        "baseline": "#333333",
        "td3": "#b23a48",
        "td7": "#4b6cb7",
        "sg_td3": "#15803d",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, meta in RUNS.items():
        avg = avg_rewards(bundles[key])
        episodes = np.arange(1, avg.size + 1)
        series = pd.Series(avg).rolling(5, center=True, min_periods=1).mean().to_numpy()
        ax.plot(episodes, avg, color=colors[key], alpha=0.18, linewidth=0.9)
        ax.plot(episodes, series, color=colors[key], linewidth=2.2, label=meta["label"])
    ax.axvline(10, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(13, color="#15803d", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(30, color="#b23a48", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title("Distillation residual reward: SG-TD3 versus prior residual runs")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average episode reward")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    ax.text(10.5, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "warm end", fontsize=8)
    ax.text(13.5, ax.get_ylim()[0] + 0.13 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "SG live", fontsize=8)
    ax.text(30.5, ax.get_ylim()[0] + 0.21 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "old guard end", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_reward_comparison.png", dpi=180)
    plt.close(fig)


def plot_window_metrics(window_df: pd.DataFrame) -> None:
    order = list(WINDOWS)
    labels = [w.replace("_", "\n") for w in order]
    colors = {
        "OF-MPC": "#333333",
        "TD3 residual": "#b23a48",
        "TD7 residual": "#4b6cb7",
        "SG-TD3 residual": "#15803d",
    }
    metrics = [
        ("reward_mean", "Mean reward"),
        ("band_norm_mae", "Mean abs error / tolerance"),
        ("temp_mae", "Tray-85 temp MAE"),
        ("outside_band_frac", "Outside-band fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    width = 0.18
    x = np.arange(len(order))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for offset_idx, label in enumerate(colors):
            vals = []
            for window in order:
                row = window_df[(window_df["label"] == label) & (window_df["window"] == window)]
                vals.append(float(row.iloc[0][metric]) if not row.empty else math.nan)
            ax.bar(x + (offset_idx - 1.5) * width, vals, width=width, color=colors[label], label=label)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
    axes.flat[0].legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_window_metrics.png", dpi=180)
    plt.close(fig)


def plot_sg_gate_sources(ep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[1.1, 0.9])
    episodes = ep_df["episode"].to_numpy()
    stack_keys = [
        ("source_warm_start_frac", "warm", "#9ca3af"),
        ("source_held_frac", "held", "#60a5fa"),
        ("source_supervisor_frac", "supervisor", "#f59e0b"),
        ("source_policy_frac", "policy", "#15803d"),
        ("source_fallback_frac", "fallback", "#dc2626"),
    ]
    values = [ep_df[k].to_numpy() for k, _label, _color in stack_keys]
    axes[0].stackplot(
        episodes,
        values,
        labels=[label for _k, label, _color in stack_keys],
        colors=[color for _k, _label, color in stack_keys],
        alpha=0.9,
    )
    axes[0].set_ylabel("Step fraction")
    axes[0].set_title("SG-TD3 residual gate source fractions by episode")
    axes[0].legend(ncol=5, frameon=False, loc="upper center", fontsize=8)
    axes[0].grid(alpha=0.25)
    reward_axis = axes[1]
    advantage_axis = reward_axis.twinx()
    reward_line = reward_axis.plot(episodes, ep_df["reward"], color="#15803d", linewidth=1.8, label="reward")
    advantage_line = advantage_axis.plot(
        episodes,
        pd.Series(ep_df["sg_advantage_mean"]).rolling(5, center=True, min_periods=1).mean(),
        color="#334155",
        linewidth=1.4,
        label="mean SG advantage, 5-ep smooth",
    )
    advantage_axis.axhline(0.0, color="#111827", linewidth=0.8, linestyle="--", alpha=0.8)
    reward_axis.set_xlabel("Episode")
    reward_axis.set_ylabel("Reward", color="#15803d")
    advantage_axis.set_ylabel("SG advantage", color="#334155")
    lines = reward_line + advantage_line
    reward_axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="lower right")
    reward_axis.grid(alpha=0.25)
    for ax in axes:
        ax.axvline(10, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(13, color="#15803d", linestyle="--", linewidth=1.0, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sg_gate_sources.png", dpi=180)
    plt.close(fig)


def plot_residual_norms(bundles: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {
        "td3": "#b23a48",
        "td7": "#4b6cb7",
        "sg_td3": "#15803d",
    }
    for key in ("td3", "td7", "sg_td3"):
        bundle = bundles[key]
        ep_len = episode_len(bundle)
        exec_norm = np.linalg.norm(np.asarray(bundle["delta_u_res_exec_log"], float), axis=1)
        req_norm = np.linalg.norm(np.asarray(bundle["delta_u_res_requested_log"], float), axis=1)
        exec_ep = episode_mean(exec_norm, ep_len)
        req_ep = episode_mean(req_norm, ep_len)
        episodes = np.arange(1, exec_ep.size + 1)
        axes[0].plot(episodes, req_ep, color=colors[key], alpha=0.35, linewidth=1.0)
        axes[0].plot(episodes, exec_ep, color=colors[key], linewidth=1.8, label=RUNS[key]["label"])
    axes[0].set_title("Executed residual norm, with requested residual shown faintly")
    axes[0].set_ylabel("Scaled delta-u residual norm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)

    sg = bundles["sg_td3"]
    ep_len = episode_len(sg)
    gap_ep = episode_mean(np.asarray(sg["policy_executed_gap_norm_log"], float), ep_len)
    policy_ep = episode_fraction(np.asarray(sg["sg_selected_source_log"], int), ep_len, lambda src: src == 2)
    supervisor_ep = episode_fraction(np.asarray(sg["sg_selected_source_log"], int), ep_len, lambda src: src == 1)
    episodes = np.arange(1, gap_ep.size + 1)
    axes[1].plot(episodes, gap_ep, color="#334155", linewidth=1.7, label="policy-executed gap norm")
    axes[1].plot(episodes, policy_ep, color="#15803d", linewidth=1.7, label="policy fraction")
    axes[1].plot(episodes, supervisor_ep, color="#f59e0b", linewidth=1.7, label="supervisor fraction")
    axes[1].set_title("SG-TD3 gate intervention and actor authority")
    axes[1].set_ylabel("Fraction / norm")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    for ax in axes:
        ax.axvline(10, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(13, color="#15803d", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(30, color="#b23a48", linestyle=":", linewidth=1.0, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_norms_and_gate.png", dpi=180)
    plt.close(fig)


def plot_tail_tracking(bundles: dict[str, dict]) -> None:
    labels = ["OF-MPC", "TD3 residual", "TD7 residual", "SG-TD3 residual"]
    keys = ["baseline", "td3", "td7", "sg_td3"]
    colors = ["#333333", "#b23a48", "#4b6cb7", "#15803d"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    tail_eps = 3
    for key, label, color in zip(keys, labels, colors):
        bundle = bundles[key]
        ep_len = episode_len(bundle)
        sl = episode_slice(bundle, n_episodes(bundle) - tail_eps + 1, n_episodes(bundle))
        _err, y, ysp = step_error_phys(bundle)
        t = np.arange(sl.stop - sl.start)
        axes[0].plot(t, y[sl, 0], color=color, linewidth=1.5, label=label)
        axes[1].plot(t, y[sl, 1], color=color, linewidth=1.5, label=label)
        if key == "baseline":
            axes[0].plot(t, ysp[sl, 0], color="#111827", linestyle="--", linewidth=1.0, label="setpoint")
            axes[1].plot(t, ysp[sl, 1], color="#111827", linestyle="--", linewidth=1.0, label="setpoint")
    axes[0].set_title("Final three episodes: tray-24 ethane composition")
    axes[1].set_title("Final three episodes: tray-85 temperature")
    axes[1].set_xlabel("Step within final three episodes")
    axes[0].set_ylabel("Composition")
    axes[1].set_ylabel("Temperature")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_tracking_comparison.png", dpi=180)
    plt.close(fig)


def config_summary(bundles: dict[str, dict]) -> dict:
    sg = bundles["sg_td3"]
    bc = sg.get("behavioral_cloning", {}) or {}
    safety = sg.get("residual_safety", {}) or {}
    gate = sg.get("supervisor_gate", sg.get("supervisor_gate_config", {})) or {}
    return {
        "analysis_date": "2026-06-02",
        "runs": {key: str(meta["path"].relative_to(ROOT)) for key, meta in RUNS.items()},
        "compare_paths": {key: str(path.relative_to(ROOT)) for key, path in COMPARE_PATHS.items()},
        "sg_config": {
            "agent_kind": sg.get("agent_kind"),
            "notebook_source": sg.get("notebook_source"),
            "run_mode": sg.get("run_mode"),
            "state_mode": sg.get("state_mode"),
            "warm_start_step": int(sg.get("warm_start_step", -1)),
            "time_in_sub_episodes": int(sg.get("time_in_sub_episodes", -1)),
            "post_warm_start_action_freeze_subepisodes": int(
                sg.get("post_warm_start_action_freeze_subepisodes", sg.get("phase1_action_freeze_subepisodes", -1))
            ),
            "post_warm_start_actor_freeze_subepisodes": int(
                sg.get("post_warm_start_actor_freeze_subepisodes", sg.get("phase1_actor_freeze_subepisodes", -1))
            ),
            "behavioral_cloning_enabled": bool(bc.get("enabled", False)),
            "bc_handoff_enabled": bool((bc.get("handoff", {}) or {}).get("enabled", False)),
            "bc_release_gate_enabled": bool((bc.get("release_gate", {}) or {}).get("enabled", False)),
            "td3_authority_ramp_enabled": bool((sg.get("td3_authority_ramp", {}) or {}).get("enabled", False)),
            "residual_authority_enabled": bool(sg.get("residual_authority_enabled", False)),
            "use_rho_authority": bool(sg.get("use_rho_authority", False)),
            "residual_zero_deadband_enabled": bool(sg.get("residual_zero_deadband_enabled", False)),
            "early_release_guard_enabled": bool((safety.get("early_release_guard", {}) or {}).get("enabled", False)),
            "fallback_to_zero_on_nonfinite": bool(safety.get("fallback_to_zero_on_nonfinite", False)),
            "supervisor_gate": gate,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {key: load_pickle(meta["path"]) for key, meta in RUNS.items()}
    reward_with_baseline_current(bundles)

    summary = summary_rows(bundles)
    windows = window_rows(bundles)
    sg_episodes = sg_episode_rows(bundles["sg_td3"])
    cfg = config_summary(bundles)

    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    windows.to_csv(OUT_DIR / "window_metrics.csv", index=False)
    sg_episodes.to_csv(OUT_DIR / "sg_episode_metrics.csv", index=False)
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    plot_reward(summary, bundles)
    plot_window_metrics(windows)
    plot_sg_gate_sources(sg_episodes)
    plot_residual_norms(bundles)
    plot_tail_tracking(bundles)

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print(summary[["label", "reward_tail20", "reward_final", "postwarm_min_reward", "postwarm_min_episode"]])
    print(windows[(windows["window"] == "old_collapse_31_40")][["label", "reward_mean", "temp_mae", "outside_band_frac"]])


if __name__ == "__main__":
    main()
