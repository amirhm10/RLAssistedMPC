"""Analyze the 2026-06-02 distillation weight SG-TD3 result.

The script reads saved bundles only. It focuses on why the SG-TD3 weight run is
safe but less compelling than the residual SG-TD3 result: gate conservatism,
multiplier diversity, common-scaling behavior, and exploration diagnostics.
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
OUT_DIR = ROOT / "report" / "figures" / "distillation_weights_sg_td3_20260602"

K_REL = np.array([0.3, 0.01], dtype=float)
BAND_FLOOR_PHYS = np.array([0.003, 0.2], dtype=float)
WEIGHT_LABELS = ["Q1", "Q2", "R1", "R2"]
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
    "td3_weights": {
        "label": "TD3 weights",
        "method": "weights_td3",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260601_155305"
        / "input_data.pkl",
    },
    "sg_weights": {
        "label": "SG-TD3 weights",
        "method": "weights_sg_td3",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch"
        / "20260602_140102"
        / "input_data.pkl",
    },
    "sg_residual": {
        "label": "SG-TD3 residual",
        "method": "residual_sg_td3_reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho"
        / "20260602_125954"
        / "input_data.pkl",
    },
}

COMPARE_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_weights_sg_td3_critic_warm3_manual_off_disturb_fluctuation"
    / "20260602_140116"
    / "input_data.pkl"
)

WINDOWS = {
    "warm_1_10": (1, 10),
    "critic_warm_11_13": (11, 13),
    "early_live_14_40": (14, 40),
    "middle_41_120": (41, 120),
    "late_121_180": (121, 180),
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


def finite_std(values) -> float:
    if values is None:
        return math.nan
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else math.nan


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


def inject_current_baseline_reward(bundles: dict[str, dict]) -> None:
    compare = load_pickle(COMPARE_PATH)
    baseline = dict(bundles["baseline"])
    baseline["avg_rewards"] = np.asarray(compare["avg_rewards_mpc"], float)
    baseline["avg_rewards_mpc"] = np.asarray(compare["avg_rewards_mpc"], float)
    bundles["baseline"] = baseline


def tracking_metrics(bundle: dict, ep_start: int, ep_end: int) -> dict[str, float]:
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
    }


def weight_array(bundle: dict, key: str = "weight_log"):
    value = bundle.get(key)
    if value is None:
        return None
    arr = np.asarray(value, float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return None
    return arr


def raw_to_multiplier(bundle: dict, raw: np.ndarray) -> np.ndarray:
    low = np.asarray(bundle["low_coef"], float)
    high = np.asarray(bundle["high_coef"], float)
    return low + ((np.asarray(raw, float) + 1.0) / 2.0) * (high - low)


def identity_raw(bundle: dict) -> np.ndarray:
    low = np.asarray(bundle["low_coef"], float)
    high = np.asarray(bundle["high_coef"], float)
    return 2.0 * (np.ones(4, dtype=float) - low) / (high - low) - 1.0


def multiplier_diagnostics(m: np.ndarray, prefix: str = "") -> dict[str, float]:
    if m is None:
        return {}
    m = np.asarray(m, float)
    common = np.nanmean(m, axis=1)
    rel_centered = m - common[:, None]
    log_m = np.log(np.maximum(m, 1.0e-12))
    log_disp = np.nanstd(log_m, axis=1)
    dist_identity = np.linalg.norm(m - 1.0, axis=1)
    same_within_005 = np.max(np.abs(rel_centered), axis=1) <= 0.05
    return {
        f"{prefix}multiplier_mean_all": finite_mean(m),
        f"{prefix}common_multiplier_mean": finite_mean(common),
        f"{prefix}common_multiplier_std_time": finite_std(common),
        f"{prefix}coord_log_dispersion_mean": finite_mean(log_disp),
        f"{prefix}coord_log_dispersion_q95": finite_quantile(log_disp, 0.95),
        f"{prefix}max_relative_coord_abs_mean": finite_mean(np.max(np.abs(rel_centered), axis=1)),
        f"{prefix}same_within_005_frac": finite_mean(same_within_005),
        f"{prefix}dist_to_identity_mean": finite_mean(dist_identity),
        f"{prefix}dist_to_identity_q95": finite_quantile(dist_identity, 0.95),
        f"{prefix}q1_mean": finite_mean(m[:, 0]),
        f"{prefix}q2_mean": finite_mean(m[:, 1]),
        f"{prefix}r1_mean": finite_mean(m[:, 2]),
        f"{prefix}r2_mean": finite_mean(m[:, 3]),
        f"{prefix}q1_std": finite_std(m[:, 0]),
        f"{prefix}q2_std": finite_std(m[:, 1]),
        f"{prefix}r1_std": finite_std(m[:, 2]),
        f"{prefix}r2_std": finite_std(m[:, 3]),
    }


def weight_metrics(bundle: dict, ep_start: int, ep_end: int) -> dict[str, float]:
    m = weight_array(bundle, "weight_log")
    if m is None:
        return {}
    sl = episode_slice(bundle, ep_start, ep_end)
    low = np.asarray(bundle["low_coef"], float)
    high = np.asarray(bundle["high_coef"], float)
    out = multiplier_diagnostics(m[sl, :])
    out["multiplier_boundary_frac"] = finite_mean(
        np.any(np.isclose(m[sl, :], low, atol=1.0e-9) | np.isclose(m[sl, :], high, atol=1.0e-9), axis=1)
    )
    out["multiplier_boundary_coord_frac"] = finite_mean(
        np.isclose(m[sl, :], low, atol=1.0e-9) | np.isclose(m[sl, :], high, atol=1.0e-9)
    )
    out["weight_cap_projection_frac"] = finite_mean(np.asarray(bundle.get("weight_cap_projection_active_log", []))[sl])
    out["weight_fallback_frac"] = finite_mean(np.asarray(bundle.get("weight_fallback_reason_log", []))[sl] > 0)
    out["weight_identity_source_frac"] = finite_mean(np.asarray(bundle.get("weight_action_source_log", []))[sl] == 3)
    if bundle.get("sg_selected_source_log") is not None:
        src = np.asarray(bundle["sg_selected_source_log"], int)
        for code, name in SOURCE_NAMES.items():
            out[f"sg_{name}_frac"] = finite_mean(src[sl] == code)
        out["sg_advantage_mean"] = finite_mean(np.asarray(bundle["sg_advantage_log"], float)[sl])
        out["sg_policy_score_mean"] = finite_mean(np.asarray(bundle["sg_score_policy_log"], float)[sl])
        out["sg_supervisor_score_mean"] = finite_mean(np.asarray(bundle["sg_score_supervisor_log"], float)[sl])
        out["sg_policy_q_gap_mean"] = finite_mean(np.asarray(bundle["sg_q_gap_policy_log"], float)[sl])
        out["sg_supervisor_q_gap_mean"] = finite_mean(np.asarray(bundle["sg_q_gap_supervisor_log"], float)[sl])
        policy_raw = np.asarray(bundle["sg_policy_action_raw_log"], float)
        supervisor_raw = np.asarray(bundle["sg_supervisor_action_raw_log"], float)
        executed_raw = np.asarray(bundle["sg_executed_action_raw_log"], float)
        policy_m = raw_to_multiplier(bundle, policy_raw)
        supervisor_m = raw_to_multiplier(bundle, supervisor_raw)
        executed_m = raw_to_multiplier(bundle, executed_raw)
        out.update(multiplier_diagnostics(policy_m[sl, :], prefix="policy_"))
        out["policy_raw_distance_to_identity_mean"] = finite_mean(
            np.linalg.norm(policy_raw[sl, :] - identity_raw(bundle), axis=1)
        )
        out["policy_multiplier_distance_to_identity_mean"] = finite_mean(
            np.linalg.norm(policy_m[sl, :] - np.ones(4), axis=1)
        )
        out["policy_executed_multiplier_gap_mean"] = finite_mean(np.linalg.norm(policy_m[sl, :] - executed_m[sl, :], axis=1))
        out["policy_supervisor_multiplier_gap_mean"] = finite_mean(
            np.linalg.norm(policy_m[sl, :] - supervisor_m[sl, :], axis=1)
        )
    return out


def residual_metrics(bundle: dict, ep_start: int, ep_end: int) -> dict[str, float]:
    if "delta_u_res_exec_log" not in bundle:
        return {}
    res = np.asarray(bundle.get("delta_u_res_exec_log"), float)
    if res.ndim != 2 or res.shape[1] == 0:
        return {}
    sl = episode_slice(bundle, ep_start, ep_end)
    exec_norm = np.linalg.norm(res, axis=1)
    out = {
        "residual_exec_norm_mean": finite_mean(exec_norm[sl]),
        "residual_exec_norm_q95": finite_quantile(exec_norm[sl], 0.95),
    }
    if bundle.get("sg_selected_source_log") is not None:
        src = np.asarray(bundle["sg_selected_source_log"], int)
        for code, name in SOURCE_NAMES.items():
            out[f"sg_{name}_frac"] = finite_mean(src[sl] == code)
    return out


def first_recovery_episode(avg: np.ndarray, baseline: np.ndarray, start_idx: int, window: int = 5) -> int | None:
    if avg.size == 0 or baseline.size == 0:
        return None
    limit = min(avg.size, baseline.size) - window + 1
    for idx in range(max(0, start_idx + 1), limit):
        if float(np.nanmean(avg[idx : idx + window])) >= float(np.nanmean(baseline[idx : idx + window])):
            return int(idx + 1)
    return None


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
        row = {
            "key": key,
            "label": meta["label"],
            "method": meta["method"],
            "path": str(meta["path"].relative_to(ROOT)),
            "agent_kind": bundle.get("agent_kind", "of_mpc"),
            "notebook_source": bundle.get("notebook_source", "baseline"),
            "warm_episodes": warm_episodes,
            "action_freeze_episodes": int(bundle.get("phase1_action_freeze_subepisodes", 0)),
            "actor_freeze_episodes": int(bundle.get("phase1_actor_freeze_subepisodes", 0)),
            "reward_mean_all": finite_mean(avg),
            "reward_tail20": finite_mean(avg[-20:]),
            "reward_final": float(avg[-1]) if avg.size else math.nan,
            "reward_best": float(np.nanmax(avg)) if avg.size else math.nan,
            "postwarm_min_reward": float(avg[postwarm_min_idx]) if avg.size else math.nan,
            "postwarm_min_episode": int(postwarm_min_idx + 1) if avg.size else -1,
            "recovery_5ep_ge_baseline_after_postwarm_min": first_recovery_episode(
                avg, baseline_avg, postwarm_min_idx, window=5
            ),
            "tail20_reward_delta_vs_baseline": finite_mean(avg[-20:]) - finite_mean(baseline_avg[-20:]),
            "final_reward_delta_vs_baseline": (float(avg[-1]) - float(baseline_avg[-1])) if avg.size else math.nan,
        }
        row.update({f"tail20_{k}": v for k, v in tracking_metrics(bundle, 181, 200).items()})
        row.update({f"early_live_{k}": v for k, v in tracking_metrics(bundle, 14, 40).items()})
        row.update({f"tail20_{k}": v for k, v in weight_metrics(bundle, 181, 200).items()})
        row.update({f"middle_{k}": v for k, v in weight_metrics(bundle, 41, 120).items()})
        row.update({f"early_live_{k}": v for k, v in weight_metrics(bundle, 14, 40).items()})
        row.update({f"tail20_{k}": v for k, v in residual_metrics(bundle, 181, 200).items()})
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
            row.update(tracking_metrics(bundle, ep_start, ep_end))
            row.update(weight_metrics(bundle, ep_start, ep_end))
            row.update(residual_metrics(bundle, ep_start, ep_end))
            rows.append(row)
    return pd.DataFrame(rows)


def sg_episode_rows(bundle: dict) -> pd.DataFrame:
    ep_len = episode_len(bundle)
    avg = avg_rewards(bundle)
    m = np.asarray(bundle["weight_log"], float)
    policy_m = raw_to_multiplier(bundle, np.asarray(bundle["sg_policy_action_raw_log"], float))
    executed_m = raw_to_multiplier(bundle, np.asarray(bundle["sg_executed_action_raw_log"], float))
    src = np.asarray(bundle["sg_selected_source_log"], int)
    adv = np.asarray(bundle["sg_advantage_log"], float)
    n_ep = min(n_episodes(bundle), m.shape[0] // ep_len)
    rows = []
    for idx in range(n_ep):
        sl = slice(idx * ep_len, (idx + 1) * ep_len)
        common = np.nanmean(m[sl, :], axis=1)
        policy_common = np.nanmean(policy_m[sl, :], axis=1)
        row = {
            "episode": idx + 1,
            "reward": float(avg[idx]) if idx < avg.size else math.nan,
            "executed_common_multiplier_mean": finite_mean(common),
            "executed_coord_log_dispersion_mean": finite_mean(np.nanstd(np.log(np.maximum(m[sl, :], 1.0e-12)), axis=1)),
            "executed_dist_identity_mean": finite_mean(np.linalg.norm(m[sl, :] - 1.0, axis=1)),
            "policy_common_multiplier_mean": finite_mean(policy_common),
            "policy_coord_log_dispersion_mean": finite_mean(
                np.nanstd(np.log(np.maximum(policy_m[sl, :], 1.0e-12)), axis=1)
            ),
            "policy_dist_identity_mean": finite_mean(np.linalg.norm(policy_m[sl, :] - 1.0, axis=1)),
            "policy_executed_gap_mean": finite_mean(np.linalg.norm(policy_m[sl, :] - executed_m[sl, :], axis=1)),
            "sg_advantage_mean": finite_mean(adv[sl]),
        }
        for j, label in enumerate(WEIGHT_LABELS):
            row[f"executed_{label}_mean"] = finite_mean(m[sl, j])
            row[f"policy_{label}_mean"] = finite_mean(policy_m[sl, j])
        for code, name in SOURCE_NAMES.items():
            row[f"source_{name}_frac"] = finite_mean(src[sl] == code)
        rows.append(row)
    return pd.DataFrame(rows)


def exploration_summary(bundle: dict) -> dict:
    out = {}
    for key in ("exploration_magnitude_trace", "param_noise_scale_trace", "action_saturation_trace"):
        arr = np.asarray(bundle.get(key, []), float)
        out[f"{key}_mean"] = finite_mean(arr)
        out[f"{key}_tail_last20pct_mean"] = finite_mean(arr[int(0.8 * arr.size) :]) if arr.size else math.nan
        out[f"{key}_q95"] = finite_quantile(arr, 0.95)
    return out


def plot_reward(bundles: dict[str, dict]) -> None:
    colors = {
        "baseline": "#333333",
        "td3_weights": "#4b6cb7",
        "sg_weights": "#b45309",
        "sg_residual": "#15803d",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, meta in RUNS.items():
        avg = avg_rewards(bundles[key])
        episodes = np.arange(1, avg.size + 1)
        smooth = pd.Series(avg).rolling(5, center=True, min_periods=1).mean().to_numpy()
        ax.plot(episodes, avg, color=colors[key], alpha=0.18, linewidth=0.9)
        ax.plot(episodes, smooth, color=colors[key], linewidth=2.2, label=meta["label"])
    ax.axvline(10, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(13, color="#b45309", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Distillation weights SG-TD3: safe but limited improvement")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average episode reward")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_reward_comparison.png", dpi=180)
    plt.close(fig)


def plot_weight_multipliers(bundles: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    colors = ["#2563eb", "#dc2626", "#7c3aed", "#059669"]
    for ax, key in zip(axes.flat, ["td3_weights", "sg_weights"]):
        bundle = bundles[key]
        ep_len = episode_len(bundle)
        m_ep = episode_mean(np.asarray(bundle["weight_log"], float), ep_len)
        episodes = np.arange(1, m_ep.shape[0] + 1)
        for j, label in enumerate(WEIGHT_LABELS):
            ax.plot(episodes, m_ep[:, j], color=colors[j], linewidth=1.6, label=label)
        ax.axhline(1.0, color="#111827", linestyle="--", linewidth=0.9)
        ax.set_title(f"Executed multipliers: {RUNS[key]['label']}")
        ax.grid(alpha=0.25)
        ax.set_ylabel("Multiplier")
        ax.legend(ncol=4, frameon=False, fontsize=8)

    sg = bundles["sg_weights"]
    ep_len = episode_len(sg)
    policy_m = raw_to_multiplier(sg, np.asarray(sg["sg_policy_action_raw_log"], float))
    policy_ep = episode_mean(policy_m, ep_len)
    episodes = np.arange(1, policy_ep.shape[0] + 1)
    for j, label in enumerate(WEIGHT_LABELS):
        axes.flat[2].plot(episodes, policy_ep[:, j], color=colors[j], linewidth=1.6, label=label)
    axes.flat[2].axhline(1.0, color="#111827", linestyle="--", linewidth=0.9)
    axes.flat[2].set_title("SG-TD3 policy-proposed multipliers")
    axes.flat[2].set_xlabel("Episode")
    axes.flat[2].set_ylabel("Multiplier")
    axes.flat[2].grid(alpha=0.25)

    exec_ep = episode_mean(np.asarray(sg["weight_log"], float), ep_len)
    exec_disp = np.nanstd(np.log(np.maximum(exec_ep, 1.0e-12)), axis=1)
    policy_disp = np.nanstd(np.log(np.maximum(policy_ep, 1.0e-12)), axis=1)
    axes.flat[3].plot(episodes, policy_disp, color="#334155", linewidth=1.8, label="policy dispersion")
    axes.flat[3].plot(episodes, exec_disp, color="#b45309", linewidth=1.8, label="executed dispersion")
    axes.flat[3].set_title("Relative penalty diversity")
    axes.flat[3].set_xlabel("Episode")
    axes.flat[3].set_ylabel("Std of log multipliers")
    axes.flat[3].grid(alpha=0.25)
    axes.flat[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_weight_multipliers.png", dpi=180)
    plt.close(fig)


def plot_gate_and_exploration(bundles: dict[str, dict], sg_ep: pd.DataFrame) -> None:
    sg = bundles["sg_weights"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    episodes = sg_ep["episode"].to_numpy()
    stack_keys = [
        ("source_warm_start_frac", "warm", "#9ca3af"),
        ("source_supervisor_frac", "supervisor", "#f59e0b"),
        ("source_policy_frac", "policy", "#15803d"),
        ("source_fallback_frac", "fallback", "#dc2626"),
    ]
    axes[0].stackplot(
        episodes,
        [sg_ep[k].to_numpy() for k, _label, _color in stack_keys],
        labels=[label for _k, label, _color in stack_keys],
        colors=[color for _k, _label, color in stack_keys],
        alpha=0.9,
    )
    axes[0].set_title("SG-TD3 weights gate source fractions")
    axes[0].set_ylabel("Step fraction")
    axes[0].legend(ncol=4, frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].axvline(10, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].axvline(13, color="#b45309", linestyle="--", linewidth=1.0, alpha=0.7)

    advantage_axis = axes[1].twinx()
    reward_line = axes[1].plot(episodes, sg_ep["reward"], color="#b45309", linewidth=1.8, label="reward")
    gap_line = axes[1].plot(
        episodes,
        sg_ep["policy_executed_gap_mean"],
        color="#334155",
        linewidth=1.5,
        label="policy-executed multiplier gap",
    )
    advantage_line = advantage_axis.plot(
        episodes,
        sg_ep["sg_advantage_mean"],
        color="#7c3aed",
        linewidth=1.1,
        alpha=0.8,
        label="SG advantage",
    )
    advantage_axis.axhline(0.0, color="#111827", linestyle="--", linewidth=0.8)
    axes[1].set_title("Reward, policy-executed gap, and SG advantage")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward / multiplier gap")
    advantage_axis.set_ylabel("SG advantage", color="#7c3aed")
    lines = reward_line + gap_line + advantage_line
    axes[1].legend(lines, [line.get_label() for line in lines], ncol=3, frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)

    for key, color, label in [
        ("exploration_magnitude_trace", "#2563eb", "exploration magnitude"),
        ("param_noise_scale_trace", "#dc2626", "param noise scale"),
        ("action_saturation_trace", "#059669", "action saturation"),
    ]:
        arr = np.asarray(sg.get(key, []), float)
        if arr.size:
            x = np.arange(arr.size)
            smooth = pd.Series(arr).rolling(1000, min_periods=1).mean().to_numpy()
            axes[2].plot(x, smooth, color=color, linewidth=1.5, label=label)
    axes[2].set_title("Training-step exploration diagnostics, 1000-step smooth")
    axes[2].set_xlabel("Training selection index")
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_gate_and_exploration.png", dpi=180)
    plt.close(fig)


def plot_tail_tracking(bundles: dict[str, dict]) -> None:
    keys = ["baseline", "td3_weights", "sg_weights", "sg_residual"]
    colors = ["#333333", "#4b6cb7", "#b45309", "#15803d"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    tail_eps = 3
    for key, color in zip(keys, colors):
        bundle = bundles[key]
        ep_len = episode_len(bundle)
        sl = episode_slice(bundle, n_episodes(bundle) - tail_eps + 1, n_episodes(bundle))
        _err, y, ysp = step_error_phys(bundle)
        t = np.arange(sl.stop - sl.start)
        axes[0].plot(t, y[sl, 0], color=color, linewidth=1.5, label=RUNS[key]["label"])
        axes[1].plot(t, y[sl, 1], color=color, linewidth=1.5, label=RUNS[key]["label"])
        if key == "baseline":
            axes[0].plot(t, ysp[sl, 0], color="#111827", linestyle="--", linewidth=1.0, label="setpoint")
            axes[1].plot(t, ysp[sl, 1], color="#111827", linestyle="--", linewidth=1.0, label="setpoint")
    axes[0].set_title("Final three episodes: tray-24 ethane composition")
    axes[1].set_title("Final three episodes: tray-85 temperature")
    axes[0].set_ylabel("Composition")
    axes[1].set_ylabel("Temperature")
    axes[1].set_xlabel("Step within final three episodes")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_tracking.png", dpi=180)
    plt.close(fig)


def config_summary(bundles: dict[str, dict], summary: pd.DataFrame) -> dict:
    sg = bundles["sg_weights"]
    cfg = sg.get("config_snapshot", {}) or {}
    return {
        "analysis_date": "2026-06-02",
        "runs": {key: str(meta["path"].relative_to(ROOT)) for key, meta in RUNS.items()},
        "baseline_compare_path": str(COMPARE_PATH.relative_to(ROOT)),
        "sg_weights_config": {
            "agent_kind": sg.get("agent_kind"),
            "notebook_source": sg.get("notebook_source"),
            "run_mode": sg.get("run_mode"),
            "state_mode": sg.get("state_mode"),
            "warm_start_step": int(sg.get("warm_start_step", -1)),
            "time_in_sub_episodes": int(sg.get("time_in_sub_episodes", -1)),
            "phase1_action_freeze_subepisodes": int(sg.get("phase1_action_freeze_subepisodes", -1)),
            "phase1_actor_freeze_subepisodes": int(sg.get("phase1_actor_freeze_subepisodes", -1)),
            "low_coef": np.asarray(sg.get("low_coef"), float).tolist(),
            "high_coef": np.asarray(sg.get("high_coef"), float).tolist(),
            "identity_raw_action": identity_raw(sg).tolist(),
            "supervisor_gate": cfg.get("supervisor_gate"),
            "weight_safety": cfg.get("weight_safety"),
            "td3_authority_ramp": cfg.get("td3_authority_ramp"),
        },
        "exploration_summary": {
            "td3_weights": exploration_summary(bundles["td3_weights"]),
            "sg_weights": exploration_summary(sg),
        },
        "tail_summary": summary.set_index("key")[
            [
                "reward_tail20",
                "reward_final",
                "postwarm_min_reward",
                "tail20_comp_mae",
                "tail20_temp_mae",
                "tail20_outside_band_frac",
            ]
        ].to_dict(orient="index"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {key: load_pickle(meta["path"]) for key, meta in RUNS.items()}
    inject_current_baseline_reward(bundles)

    summary = summary_rows(bundles)
    windows = window_rows(bundles)
    sg_episodes = sg_episode_rows(bundles["sg_weights"])
    cfg = config_summary(bundles, summary)

    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    windows.to_csv(OUT_DIR / "window_metrics.csv", index=False)
    sg_episodes.to_csv(OUT_DIR / "sg_episode_metrics.csv", index=False)
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    plot_reward(bundles)
    plot_weight_multipliers(bundles)
    plot_gate_and_exploration(bundles, sg_episodes)
    plot_tail_tracking(bundles)

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print(summary[["label", "reward_tail20", "reward_final", "postwarm_min_reward", "tail20_temp_mae"]])
    print(
        windows[windows["label"].eq("SG-TD3 weights")][
            [
                "window",
                "reward_mean",
                "sg_policy_frac",
                "sg_supervisor_frac",
                "coord_log_dispersion_mean",
                "policy_coord_log_dispersion_mean",
                "policy_executed_multiplier_gap_mean",
            ]
        ]
    )


if __name__ == "__main__":
    main()
