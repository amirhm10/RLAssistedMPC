"""Analyze the wider-search distillation safety-restoration batch.

This reads saved bundles only. It does not modify raw experiment directories.
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
OUT_DIR = ROOT / "report" / "figures" / "distillation_wider_safety_20260530"

INPUT_BOUNDS = {
    "u_min": np.array([300000.0, 100.0], dtype=float),
    "u_max": np.array([460000.0, 150.0], dtype=float),
}
K_REL = np.array([0.3, 0.01], dtype=float)
BAND_FLOOR_PHYS = np.array([0.003, 0.2], dtype=float)


CURRENT_RUNS = {
    "baseline": {
        "label": "OF-MPC",
        "method": "baseline",
        "path": ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    },
    "weights": {
        "label": "TD3 weights",
        "method": "weights",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260529_211201"
        / "input_data.pkl",
    },
    "horizon": {
        "label": "Horizon DDQN",
        "method": "horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260529_213627"
        / "input_data.pkl",
    },
    "dueling": {
        "label": "Dueling horizon",
        "method": "dueling_horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260529_213847"
        / "input_data.pkl",
    },
    "residual": {
        "label": "TD3 residual",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260529_213534"
        / "input_data.pkl",
    },
    "markov": {
        "label": "TD3 Markov",
        "method": "markov",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260529_220507"
        / "input_data.pkl",
    },
}

PREVIOUS_RUNS = {
    "weights": ROOT
    / "Distillation"
    / "Results"
    / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
    / "20260528_194904"
    / "input_data.pkl",
    "horizon": ROOT
    / "Distillation"
    / "Results"
    / "distillation_horizon_disturb_fluctuation_mismatch_unified"
    / "20260528_201659"
    / "input_data.pkl",
    "dueling": ROOT
    / "Distillation"
    / "Results"
    / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
    / "20260528_202519"
    / "input_data.pkl",
    "residual": ROOT
    / "Distillation"
    / "Results"
    / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
    / "20260528_213902"
    / "input_data.pkl",
    "markov": ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_unified"
    / "20260528_230949"
    / "input_data.pkl",
}

BASELINE_COMPARE_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_markov_td3_disturb_fluctuation"
    / "20260529_220519"
    / "input_data.pkl"
)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def arr(bundle: dict, key: str, default=None):
    value = bundle.get(key, default)
    if value is None:
        return None
    return np.asarray(value)


def finite_mean(values) -> float:
    if values is None:
        return math.nan
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def finite_quantile(values, q: float) -> float:
    if values is None:
        return math.nan
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else math.nan


def finite_sum(values) -> float:
    if values is None:
        return math.nan
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.sum(values)) if values.size else math.nan


def sliced(bundle: dict, key: str, sl: slice):
    value = arr(bundle, key)
    if value is None:
        return None
    return value[sl]


def sliced_delta(bundle: dict, lhs: str, rhs: str, sl: slice):
    left = sliced(bundle, lhs, sl)
    right = sliced(bundle, rhs, sl)
    if left is None or right is None:
        return None
    return left - right


def positive_sliced(bundle: dict, key: str, sl: slice):
    value = sliced(bundle, key, sl)
    if value is None:
        return None
    return value > 0


def minmax_scale(data, min_val, max_val):
    return (np.asarray(data, float) - np.asarray(min_val, float)) / (
        np.asarray(max_val, float) - np.asarray(min_val, float)
    )


def reverse_minmax(data, min_val, max_val):
    return np.asarray(data, float) * (np.asarray(max_val, float) - np.asarray(min_val, float)) + np.asarray(
        min_val, float
    )


def y_sp_phys(bundle: dict) -> np.ndarray:
    y_sp = np.asarray(bundle["y_sp"], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = 2
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = minmax_scale(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    return reverse_minmax(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def y_phys_line(bundle: dict) -> np.ndarray:
    for key in ("y_line_full", "y", "y_rl", "y_mpc"):
        value = arr(bundle, key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError("No output trajectory found.")


def u_phys_step(bundle: dict) -> np.ndarray:
    for key in ("u_step_full", "u", "u_rl", "u_mpc"):
        value = arr(bundle, key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError("No input trajectory found.")


def step_slice(bundle: dict, tail_episodes: int | None = None) -> slice:
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    if tail_episodes is None:
        return slice(0, nfe)
    ep_len = int(bundle.get("time_in_sub_episodes", 400))
    return slice(max(0, nfe - int(tail_episodes) * ep_len), nfe)


def early_slice(bundle: dict, episodes: int = 20) -> slice:
    ep_len = int(bundle.get("time_in_sub_episodes", 400))
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    return slice(0, min(nfe, int(episodes) * ep_len))


def step_error_phys(bundle: dict):
    y_line = y_phys_line(bundle)
    ysp = y_sp_phys(bundle)
    n = min(y_line.shape[0] - 1, ysp.shape[0])
    y_step = y_line[1 : n + 1, :]
    ysp_step = ysp[:n, :]
    return y_step - ysp_step, y_step, ysp_step


def reward_summary(bundle: dict) -> dict[str, float]:
    avg = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
    if avg.size == 0:
        return {}
    return {
        "reward_mean": float(np.nanmean(avg)),
        "reward_tail20": float(np.nanmean(avg[-20:])),
        "reward_tail10": float(np.nanmean(avg[-10:])),
        "reward_final": float(avg[-1]),
        "reward_first20_min": float(np.nanmin(avg[:20])),
        "reward_first20_mean": float(np.nanmean(avg[:20])),
        "reward_best": float(np.nanmax(avg)),
        "reward_best_episode": float(np.nanargmax(avg) + 1),
    }


def tracking_summary(bundle: dict, tail_episodes: int = 20) -> dict[str, float]:
    err, _y, ysp = step_error_phys(bundle)
    sl = step_slice(bundle, tail_episodes)
    sl = slice(sl.start, min(sl.stop, err.shape[0]))
    e = err[sl, :]
    ysp_tail = ysp[sl, :]
    band = np.maximum(K_REL * np.abs(ysp_tail), BAND_FLOOR_PHYS)
    abs_e = np.abs(e)
    return {
        "tail20_comp_mae": float(np.nanmean(abs_e[:, 0])),
        "tail20_temp_mae": float(np.nanmean(abs_e[:, 1])),
        "tail20_comp_rmse": float(np.sqrt(np.nanmean(e[:, 0] ** 2))),
        "tail20_temp_rmse": float(np.sqrt(np.nanmean(e[:, 1] ** 2))),
        "tail20_band_norm_mae": float(np.nanmean(abs_e / np.maximum(band, 1e-12))),
        "tail20_outside_band_frac": float(np.nanmean(abs_e > band)),
        "tail20_final_comp_abs": float(abs_e[-1, 0]),
        "tail20_final_temp_abs": float(abs_e[-1, 1]),
    }


def input_summary(bundle: dict, tail_episodes: int = 20) -> dict[str, float]:
    u = u_phys_step(bundle)
    n = min(int(bundle.get("nFE", len(u))), len(u))
    u = u[:n, :]
    du = np.diff(u, axis=0, prepend=u[[0], :])
    sl = step_slice(bundle, tail_episodes)
    sl = slice(sl.start, min(sl.stop, n))
    u_tail = u[sl, :]
    du_tail = du[sl, :]
    tol = np.array([100.0, 0.05], dtype=float)
    sat = (u_tail <= INPUT_BOUNDS["u_min"] + tol) | (u_tail >= INPUT_BOUNDS["u_max"] - tol)
    return {
        "tail20_du_abs_mean": float(np.nanmean(np.abs(du_tail))),
        "tail20_du_l2_mean": float(np.nanmean(np.linalg.norm(du_tail, axis=1))),
        "tail20_input_saturation_frac": float(np.nanmean(sat)),
        "tail20_reflux_min": float(np.nanmin(u_tail[:, 0])),
        "tail20_reflux_max": float(np.nanmax(u_tail[:, 0])),
        "tail20_reboiler_min": float(np.nanmin(u_tail[:, 1])),
        "tail20_reboiler_max": float(np.nanmax(u_tail[:, 1])),
    }


def frac_by_code(values, sl: slice, codes: dict) -> dict[str, float]:
    out = {}
    if values is None:
        return out
    vals = np.asarray(values)[sl].astype(int)
    normalized: dict[int, str] = {}
    for key, value in (codes or {}).items():
        try:
            normalized[int(value)] = str(key)
        except (TypeError, ValueError):
            normalized[int(key)] = str(value)
    for code, name in normalized.items():
        out[f"tail20_{name}_frac"] = float(np.mean(vals == code)) if vals.size else math.nan
    return out


def safety_summary(bundle: dict, method: str) -> dict[str, float | bool]:
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    tail = step_slice(bundle, 20)
    early = early_slice(bundle, 20)
    out: dict[str, float | bool] = {}

    if method == "weights":
        weights = arr(bundle, "weight_log")
        if weights is not None:
            out.update(
                {
                    "tail20_weight_q1_mean": finite_mean(weights[tail, 0]),
                    "tail20_weight_q2_mean": finite_mean(weights[tail, 1]),
                    "tail20_weight_r1_mean": finite_mean(weights[tail, 2]),
                    "tail20_weight_r2_mean": finite_mean(weights[tail, 3]),
                    "tail20_weight_std_mean": finite_mean(np.std(weights[tail, :], axis=0)),
                    "tail20_weight_boundary_frac": finite_mean(
                        sliced(bundle, "weight_multiplier_saturation_log", tail)
                    ),
                    "final_weight_q1": float(weights[-1, 0]),
                    "final_weight_q2": float(weights[-1, 1]),
                    "final_weight_r1": float(weights[-1, 2]),
                    "final_weight_r2": float(weights[-1, 3]),
                }
            )
        out.update(frac_by_code(arr(bundle, "weight_action_source_log"), tail, bundle.get("weight_action_source_codes", {})))
        out["tail20_weight_cap_projection_frac"] = finite_mean(sliced(bundle, "weight_cap_projection_active_log", tail))
        out["early_weight_cap_projection_frac"] = finite_mean(sliced(bundle, "weight_cap_projection_active_log", early))
        out["weight_probation_trigger_count"] = finite_sum(arr(bundle, "weight_probation_trigger_log"))
        out["tail20_weight_probation_frac"] = finite_mean(sliced(bundle, "weight_probation_active_log", tail))
        out["tail20_weight_fallback_frac"] = finite_mean(positive_sliced(bundle, "weight_fallback_reason_log", tail))
        out["tail20_shadow_identity_objective_delta_mean"] = finite_mean(
            sliced_delta(bundle, "weight_shadow_selected_objective_log", "weight_shadow_identity_objective_log", tail)
        )
        out["tail20_shadow_first_move_delta_norm_mean"] = finite_mean(
            sliced(bundle, "weight_shadow_first_move_delta_norm_log", tail)
        )

    if method in {"horizon", "dueling_horizon"}:
        horizon = arr(bundle, "horizon_trace")
        if horizon is not None:
            h = horizon[tail, :].astype(int)
            pairs, counts = np.unique(h, axis=0, return_counts=True)
            probs = counts / max(1, counts.sum())
            out.update(
                {
                    "tail20_predict_h_mean": finite_mean(h[:, 0]),
                    "tail20_control_h_mean": finite_mean(h[:, 1]),
                    "tail20_predict_h_std": float(np.std(h[:, 0])),
                    "tail20_control_h_std": float(np.std(h[:, 1])),
                    "tail20_unique_horizon_pairs": float(len(pairs)),
                    "tail20_horizon_entropy": float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))),
                    "tail20_top_pair_fraction": float(np.max(probs)) if probs.size else math.nan,
                }
            )
        out.update(frac_by_code(arr(bundle, "horizon_safety_reason_log"), tail, bundle.get("horizon_safety_reason_codes", {})))
        out["tail20_horizon_projection_frac"] = finite_mean(sliced(bundle, "horizon_projection_active_log", tail))
        out["early_horizon_projection_frac"] = finite_mean(sliced(bundle, "horizon_projection_active_log", early))
        out["horizon_probation_trigger_count"] = finite_sum(arr(bundle, "horizon_reward_probation_trigger_log"))
        out["tail20_horizon_cooldown_frac"] = finite_mean(sliced(bundle, "horizon_cooldown_active_log", tail))
        out["tail20_switch_step_frac"] = finite_mean(sliced(bundle, "horizon_change_log", tail))
        out["tail20_shadow_objective_delta_mean"] = finite_mean(
            sliced_delta(bundle, "horizon_shadow_selected_objective_log", "horizon_shadow_default_objective_log", tail)
        )
        out["tail20_shadow_first_move_delta_norm_mean"] = finite_mean(
            sliced(bundle, "horizon_shadow_first_move_delta_norm_log", tail)
        )

    if method == "residual":
        raw = arr(bundle, "delta_u_res_raw_log")
        exe = arr(bundle, "delta_u_res_exec_log")
        if raw is not None and exe is not None:
            raw_norm = np.linalg.norm(raw, axis=1)
            exe_norm = np.linalg.norm(exe, axis=1)
            diff = np.linalg.norm(exe - raw, axis=1)
            out.update(
                {
                    "tail20_raw_residual_norm_mean": finite_mean(raw_norm[tail]),
                    "tail20_exec_residual_norm_mean": finite_mean(exe_norm[tail]),
                    "tail20_exec_residual_norm_q95": finite_quantile(exe_norm[tail], 0.95),
                    "tail20_raw_exec_diff_norm_mean": finite_mean(diff[tail]),
                    "tail20_material_projection_frac": finite_mean(diff[tail] > 1.0e-6),
                }
            )
        out.update(frac_by_code(arr(bundle, "residual_action_source_log"), tail, bundle.get("residual_action_source_codes", {})))
        out["tail20_residual_cap_projection_frac"] = finite_mean(sliced(bundle, "residual_cap_projection_active_log", tail))
        out["early_residual_cap_projection_frac"] = finite_mean(sliced(bundle, "residual_cap_projection_active_log", early))
        out["residual_probation_trigger_count"] = finite_sum(arr(bundle, "residual_probation_trigger_log"))
        out["tail20_residual_probation_frac"] = finite_mean(sliced(bundle, "residual_probation_active_log", tail))
        out["tail20_residual_zero_fallback_frac"] = finite_mean(
            positive_sliced(bundle, "residual_zero_fallback_reason_log", tail)
        )
        out["tail20_shadow_rho_projection_frac"] = finite_mean(sliced(bundle, "shadow_rho_projection_active_log", tail))
        out["tail20_shadow_rho_diff_norm_mean"] = finite_mean(sliced(bundle, "shadow_rho_exec_diff_norm_log", tail))
        out["tail20_shadow_rho_eff_mean"] = finite_mean(sliced(bundle, "shadow_rho_eff_log", tail))

    if method == "markov":
        out.update(frac_by_code(arr(bundle, "rl_action_source_log"), tail, bundle.get("rl_action_source_names", {})))
        out["tail20_fallback_frac"] = finite_mean(sliced(bundle, "fallback_log", tail))
        out["early_fallback_frac"] = finite_mean(sliced(bundle, "fallback_log", early))
        out["markov_probation_trigger_count"] = float(bundle.get("td3_probation_trigger_count", math.nan))
        out["tail20_markov_probation_frac"] = finite_mean(sliced(bundle, "td3_probation_active_log", tail))
        z = arr(bundle, "z_executed_log")
        if z is not None:
            z_norm = np.linalg.norm(z, axis=1)
            out["tail20_z_norm_mean"] = finite_mean(z_norm[tail])
            out["tail20_z_norm_q95"] = finite_quantile(z_norm[tail], 0.95)
            out["tail20_abs_z_q95"] = finite_quantile(np.abs(z[tail, :]).reshape(-1), 0.95)
        z_projection = sliced(bundle, "z_safety_projection_active_log", tail)
        if z_projection is None:
            z_projection = sliced(bundle, "z_safety_requested_projection_active_log", tail)
        out["tail20_z_projection_frac"] = finite_mean(z_projection)
        z_vector_projection = sliced(bundle, "z_safety_vector_projection_active_log", tail)
        if z_vector_projection is None:
            z_vector_projection = sliced(bundle, "z_safety_requested_vector_projection_active_log", tail)
        out["tail20_z_vector_projection_frac"] = finite_mean(z_vector_projection)
        out["tail20_requested_cost_guard_pass_frac"] = finite_mean(sliced(bundle, "requested_cost_guard_pass_log", tail))
        out["tail20_requested_prediction_score_mean"] = finite_mean(sliced(bundle, "requested_prediction_score_log", tail))
        out["tail20_requested_cost_margin_mean"] = finite_mean(sliced(bundle, "requested_cost_margin_log", tail))

    return out


def row_for(name: str, meta: dict, bundle: dict, batch: str) -> dict:
    row = {
        "key": name,
        "label": meta["label"],
        "method": meta["method"],
        "batch": batch,
        "path": str(meta["path"].relative_to(ROOT)),
    }
    row.update(reward_summary(bundle))
    row.update(tracking_summary(bundle))
    row.update(input_summary(bundle))
    row.update(safety_summary(bundle, meta["method"]))
    return row


def plot_reward(current_bundles: dict[str, dict], summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    colors = {
        "baseline": "black",
        "weights": "#4c78a8",
        "horizon": "#f58518",
        "dueling": "#54a24b",
        "residual": "#e45756",
        "markov": "#b279a2",
    }
    for key, bundle in current_bundles.items():
        avg = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc")), float)
        ax.plot(np.arange(1, len(avg) + 1), avg, label=CURRENT_RUNS[key]["label"], color=colors[key], linewidth=1.55)
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.set_title("Wider-search safety-restored distillation runs")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_reward_trajectories.png", dpi=180)
    plt.close(fig)

    latest = summary.sort_values("reward_tail20", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.barh(latest["label"], latest["reward_tail20"], color="#4c78a8")
    for y, value in enumerate(latest["reward_tail20"]):
        ax.text(value + 0.2, y, f"{value:.2f}", va="center")
    ax.set_xlabel("Tail-20 average reward")
    ax.set_title("Tail-20 reward ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_reward_ranking.png", dpi=180)
    plt.close(fig)


def plot_current_vs_previous(compare: pd.DataFrame) -> None:
    methods = ["weights", "horizon", "dueling", "residual", "markov"]
    labels = ["Weights", "Horizon", "Dueling", "Residual", "Markov"]
    df = compare.set_index("key").loc[methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    width = 0.34
    ax.bar(x - width / 2, df["previous_tail20"], width, label="Previous May 28", color="#bab0ac")
    ax.bar(x + width / 2, df["current_tail20"], width, label="Current May 29", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tail-20 average reward")
    ax.set_title("Effect of latest safety and search changes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_current_vs_previous_tail20.png", dpi=180)
    plt.close(fig)


def plot_tracking(summary: pd.DataFrame) -> None:
    df = summary.sort_values("reward_tail20", ascending=False)
    x = np.arange(len(df))
    width = 0.32
    fig, ax1 = plt.subplots(figsize=(10.0, 4.8))
    ax1.bar(x - width / 2, df["tail20_comp_mae"], width, label="Composition MAE", color="#4c78a8")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, df["tail20_temp_mae"], width, label="Temperature MAE", color="#f58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=20, ha="right")
    ax1.set_ylabel("Tray-24 composition MAE")
    ax2.set_ylabel("Tray-85 temperature MAE")
    ax1.set_title("Tail-20 physical tracking errors")
    ax1.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_tracking_errors.png", dpi=180)
    plt.close(fig)


def plot_safety(summary: pd.DataFrame) -> None:
    df = summary.set_index("key")
    labels = ["Weights", "Horizon", "Dueling", "Residual", "Markov"]
    values = [
        df.loc["weights", "tail20_weight_cap_projection_frac"],
        df.loc["horizon", "tail20_horizon_projection_frac"],
        df.loc["dueling", "tail20_horizon_projection_frac"],
        df.loc["residual", "tail20_residual_cap_projection_frac"],
        df.loc["markov", "tail20_fallback_frac"],
    ]
    early = [
        df.loc["weights", "early_weight_cap_projection_frac"],
        df.loc["horizon", "early_horizon_projection_frac"],
        df.loc["dueling", "early_horizon_projection_frac"],
        df.loc["residual", "early_residual_cap_projection_frac"],
        df.loc["markov", "early_fallback_frac"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    width = 0.34
    ax.bar(x - width / 2, early, width, label="First 20 episodes", color="#f58518")
    ax.bar(x + width / 2, values, width, label="Tail 20 episodes", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Safety intervention fraction")
    ax.set_title("Safety intervention rates by runner")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_safety_intervention_rates.png", dpi=180)
    plt.close(fig)


def plot_horizon_usage(current_bundles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key in ("horizon", "dueling"):
        h = np.asarray(current_bundles[key]["horizon_trace"], int)
        sl = step_slice(current_bundles[key], 20)
        h_tail = h[sl, :]
        pairs, counts = np.unique(h_tail, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        for rank, idx in enumerate(order[:20], start=1):
            rows.append(
                {
                    "runner": key,
                    "label": CURRENT_RUNS[key]["label"],
                    "rank": rank,
                    "predict_h": int(pairs[idx, 0]),
                    "control_h": int(pairs[idx, 1]),
                    "tail20_count": int(counts[idx]),
                    "tail20_fraction": float(counts[idx] / max(1, counts.sum())),
                }
            )
    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(OUT_DIR / "current_horizon_pair_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, key in zip(axes, ("horizon", "dueling")):
        top = counts_df[counts_df["runner"] == key].head(10).copy()
        names = [f"({int(r.predict_h)},{int(r.control_h)})" for r in top.itertuples()]
        ax.barh(names[::-1], top["tail20_fraction"].to_numpy()[::-1], color="#4c78a8")
        ax.set_title(CURRENT_RUNS[key]["label"])
        ax.set_xlabel("Tail-20 fraction")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Horizon pair")
    fig.suptitle("Wide horizon search: top tail pairs")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_horizon_tail_pair_usage.png", dpi=180)
    plt.close(fig)
    return counts_df


def plot_weight_residual_markov(summary: pd.DataFrame, current_bundles: dict[str, dict]) -> None:
    weights = np.asarray(current_bundles["weights"]["weight_log"], float)
    residual = np.asarray(current_bundles["residual"]["delta_u_res_exec_log"], float)
    z = np.asarray(current_bundles["markov"]["z_executed_log"], float)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=False)
    axes[0].plot(weights[:, 0], label="Q1")
    axes[0].plot(weights[:, 1], label="Q2")
    axes[0].plot(weights[:, 2], label="R1")
    axes[0].plot(weights[:, 3], label="R2")
    axes[0].set_title("Weights multiplier trajectory")
    axes[0].set_ylabel("Multiplier")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(np.linalg.norm(residual, axis=1), color="#e45756")
    axes[1].set_title("Residual executed correction norm")
    axes[1].set_ylabel("Scaled norm")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(np.linalg.norm(z, axis=1), color="#b279a2")
    axes[2].set_title("Markov executed z norm")
    axes[2].set_ylabel("z 2-norm")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_continuous_action_norms.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = {key: load_pickle(meta["path"]) for key, meta in CURRENT_RUNS.items()}
    baseline_compare = load_pickle(BASELINE_COMPARE_PATH)
    current["baseline"] = dict(current["baseline"])
    current["baseline"]["avg_rewards"] = np.asarray(baseline_compare["avg_rewards_mpc"], float)
    previous = {key: load_pickle(path) for key, path in PREVIOUS_RUNS.items()}

    current_rows = [row_for(key, meta, current[key], "2026-05-29 wider safety") for key, meta in CURRENT_RUNS.items()]
    current_summary = pd.DataFrame(current_rows)
    current_summary.to_csv(OUT_DIR / "current_summary_metrics.csv", index=False)

    previous_rows = []
    for key, path in PREVIOUS_RUNS.items():
        meta = dict(CURRENT_RUNS[key])
        meta["path"] = path
        previous_rows.append(row_for(key, meta, previous[key], "2026-05-28 previous"))
    previous_summary = pd.DataFrame(previous_rows)
    previous_summary.to_csv(OUT_DIR / "previous_summary_metrics.csv", index=False)

    baseline = current_summary.set_index("key").loc["baseline"]
    baseline_rows = []
    for key in PREVIOUS_RUNS:
        row = current_summary.set_index("key").loc[key]
        baseline_rows.append(
            {
                "key": key,
                "label": CURRENT_RUNS[key]["label"],
                "tail20_reward_delta_vs_baseline": float(row["reward_tail20"] - baseline["reward_tail20"]),
                "final_reward_delta_vs_baseline": float(row["reward_final"] - baseline["reward_final"]),
                "first20_min_delta_vs_baseline": float(row["reward_first20_min"] - baseline["reward_first20_min"]),
                "band_norm_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_band_norm_mae"] / baseline["tail20_band_norm_mae"] - 1.0)
                ),
                "outside_band_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_outside_band_frac"] / baseline["tail20_outside_band_frac"] - 1.0)
                ),
                "composition_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_comp_mae"] / baseline["tail20_comp_mae"] - 1.0)
                ),
                "temperature_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_temp_mae"] / baseline["tail20_temp_mae"] - 1.0)
                ),
                "input_saturation_delta_pct_vs_baseline": float(
                    100.0
                    * (
                        row["tail20_input_saturation_frac"]
                        / max(float(baseline["tail20_input_saturation_frac"]), 1.0e-12)
                        - 1.0
                    )
                ),
            }
        )
    baseline_compare = pd.DataFrame(baseline_rows)
    baseline_compare.to_csv(OUT_DIR / "current_vs_baseline_metrics.csv", index=False)

    compare_rows = []
    current_idx = current_summary.set_index("key")
    previous_idx = previous_summary.set_index("key")
    for key in PREVIOUS_RUNS:
        compare_rows.append(
            {
                "key": key,
                "label": CURRENT_RUNS[key]["label"],
                "current_tail20": float(current_idx.loc[key, "reward_tail20"]),
                "previous_tail20": float(previous_idx.loc[key, "reward_tail20"]),
                "tail20_delta": float(current_idx.loc[key, "reward_tail20"] - previous_idx.loc[key, "reward_tail20"]),
                "current_final": float(current_idx.loc[key, "reward_final"]),
                "previous_final": float(previous_idx.loc[key, "reward_final"]),
                "final_delta": float(current_idx.loc[key, "reward_final"] - previous_idx.loc[key, "reward_final"]),
                "current_first20_min": float(current_idx.loc[key, "reward_first20_min"]),
                "previous_first20_min": float(previous_idx.loc[key, "reward_first20_min"]),
                "first20_min_delta": float(
                    current_idx.loc[key, "reward_first20_min"] - previous_idx.loc[key, "reward_first20_min"]
                ),
                "band_norm_mae_delta": float(
                    current_idx.loc[key, "tail20_band_norm_mae"] - previous_idx.loc[key, "tail20_band_norm_mae"]
                ),
                "outside_band_frac_delta": float(
                    current_idx.loc[key, "tail20_outside_band_frac"] - previous_idx.loc[key, "tail20_outside_band_frac"]
                ),
                "composition_mae_delta": float(current_idx.loc[key, "tail20_comp_mae"] - previous_idx.loc[key, "tail20_comp_mae"]),
                "temperature_mae_delta": float(current_idx.loc[key, "tail20_temp_mae"] - previous_idx.loc[key, "tail20_temp_mae"]),
            }
        )
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(OUT_DIR / "current_vs_previous_metrics.csv", index=False)

    plot_reward(current, current_summary)
    plot_current_vs_previous(compare)
    plot_tracking(current_summary)
    plot_safety(current_summary)
    horizon_counts = plot_horizon_usage(current)
    plot_weight_residual_markov(current_summary, current)

    json_summary = {
        "current_rank_tail20": current_summary.sort_values("reward_tail20", ascending=False)[
            [
                "label",
                "reward_tail20",
                "reward_tail10",
                "reward_final",
                "reward_first20_min",
                "tail20_band_norm_mae",
                "tail20_outside_band_frac",
            ]
        ].to_dict(orient="records"),
        "current_vs_previous": compare.to_dict(orient="records"),
        "current_vs_baseline": baseline_compare.to_dict(orient="records"),
        "safety_highlights": current_summary[
            [
                "key",
                "label",
                "tail20_weight_cap_projection_frac",
                "tail20_horizon_projection_frac",
                "tail20_residual_cap_projection_frac",
                "tail20_fallback_frac",
                "weight_probation_trigger_count",
                "horizon_probation_trigger_count",
                "residual_probation_trigger_count",
                "markov_probation_trigger_count",
            ]
        ].to_dict(orient="records"),
        "artifacts": {
            "figure_dir": str(OUT_DIR.relative_to(ROOT)),
            "csvs": sorted(p.name for p in OUT_DIR.glob("*.csv")),
            "figures": sorted(p.name for p in OUT_DIR.glob("fig_*.png")),
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    print(json.dumps(json_summary, indent=2))


if __name__ == "__main__":
    main()
