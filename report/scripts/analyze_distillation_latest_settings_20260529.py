"""Analyze latest distillation RL-assisted MPC runs.

The script compares the May 28 BC-handoff/no-release-gate batch with the
previous protected and controlled-authority batches. It intentionally reads
saved bundles only and does not modify any raw experiment directory.
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
OUT_DIR = ROOT / "report" / "figures" / "distillation_latest_settings_20260529"

INPUT_BOUNDS = {
    "u_min": np.array([300000.0, 100.0], dtype=float),
    "u_max": np.array([460000.0, 150.0], dtype=float),
}
K_REL = np.array([0.3, 0.01], dtype=float)
BAND_FLOOR_PHYS = np.array([0.003, 0.2], dtype=float)


RUNS = {
    "baseline": {
        "label": "OF-MPC",
        "batch": "baseline",
        "method": "baseline",
        "path": ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    },
    "weights_20260522": {
        "label": "Weights TD3",
        "batch": "2026-05-22 protected gate",
        "method": "weights",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260522_181031"
        / "input_data.pkl",
    },
    "residual_20260522": {
        "label": "Residual TD3",
        "batch": "2026-05-22 protected gate",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260522_180223"
        / "input_data.pkl",
    },
    "horizon_20260522": {
        "label": "Horizon DDQN",
        "batch": "2026-05-22 protected gate",
        "method": "horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260522_183427"
        / "input_data.pkl",
    },
    "dueling_20260522": {
        "label": "Dueling horizon",
        "batch": "2026-05-22 protected gate",
        "method": "dueling_horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260522_183355"
        / "input_data.pkl",
    },
    "markov_20260522": {
        "label": "Markov TD3",
        "batch": "2026-05-22 protected gate",
        "method": "markov",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260522_190448"
        / "input_data.pkl",
    },
    "weights_20260523": {
        "label": "Weights TD3",
        "batch": "2026-05-23 controlled authority",
        "method": "weights",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260523_204146"
        / "input_data.pkl",
    },
    "residual_20260523": {
        "label": "Residual TD3",
        "batch": "2026-05-23 controlled authority",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260523_203649"
        / "input_data.pkl",
    },
    "horizon_20260523": {
        "label": "Horizon DDQN",
        "batch": "2026-05-23 controlled authority",
        "method": "horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260523_210449"
        / "input_data.pkl",
    },
    "dueling_20260523": {
        "label": "Dueling horizon",
        "batch": "2026-05-23 controlled authority",
        "method": "dueling_horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260523_210709"
        / "input_data.pkl",
    },
    "markov_20260523": {
        "label": "Markov TD3",
        "batch": "2026-05-23 controlled authority",
        "method": "markov",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260523_224108"
        / "input_data.pkl",
    },
    "weights_20260528": {
        "label": "Weights TD3",
        "batch": "2026-05-28 BC handoff",
        "method": "weights",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260528_194904"
        / "input_data.pkl",
    },
    "horizon_20260528": {
        "label": "Horizon DDQN",
        "batch": "2026-05-28 BC handoff",
        "method": "horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260528_201659"
        / "input_data.pkl",
    },
    "dueling_20260528": {
        "label": "Dueling horizon",
        "batch": "2026-05-28 BC handoff",
        "method": "dueling_horizon",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260528_202519"
        / "input_data.pkl",
    },
    "residual_20260528": {
        "label": "Residual TD3 no-rho",
        "batch": "2026-05-28 BC handoff",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260528_213902"
        / "input_data.pkl",
    },
    "markov_20260528": {
        "label": "Markov TD3",
        "batch": "2026-05-28 BC handoff",
        "method": "markov",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260528_230949"
        / "input_data.pkl",
    },
    "residual_best_rho": {
        "label": "Residual TD3 best rho-history",
        "batch": "2026-05-07 rho reference",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260507_212833"
        / "input_data.pkl",
    },
    "markov_20260518_soft": {
        "label": "Markov TD3 soft handoff",
        "batch": "2026-05-18 soft handoff",
        "method": "markov",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260518_184548"
        / "input_data.pkl",
    },
    "markov_20260518_no_safeguard": {
        "label": "Markov TD3 no safeguard",
        "batch": "2026-05-18 no safeguard",
        "method": "markov",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified"
        / "20260518_091937"
        / "input_data.pkl",
    },
}

LATEST_COMPARE_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_residual_td3_disturb_fluctuation"
    / "20260528_213916"
    / "input_data.pkl"
)


LATEST_KEYS = [
    "baseline",
    "weights_20260528",
    "horizon_20260528",
    "dueling_20260528",
    "residual_20260528",
    "markov_20260528",
]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def arr(bundle: dict, key: str, default=None) -> np.ndarray | None:
    value = bundle.get(key, default)
    if value is None:
        return None
    return np.asarray(value)


def finite_mean(values) -> float:
    if values is None:
        return math.nan
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def finite_quantile(values, q: float) -> float:
    if values is None:
        return math.nan
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else math.nan


def finite_frac(mask) -> float:
    if mask is None:
        return math.nan
    values = np.asarray(mask)
    values = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values
    return float(np.mean(values.astype(bool))) if values.size else math.nan


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
    n_inputs = int(bundle.get("n_inputs", 2))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = minmax_scale(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    return reverse_minmax(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def y_phys_line(bundle: dict) -> np.ndarray:
    for key in ("y_line_full", "y", "y_rl", "y_mpc"):
        value = arr(bundle, key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError("No output trajectory found in bundle.")


def u_phys_step(bundle: dict) -> np.ndarray:
    for key in ("u_step_full", "u", "u_rl", "u_mpc"):
        value = arr(bundle, key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError("No input trajectory found in bundle.")


def step_error_phys(bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_line = y_phys_line(bundle)
    ysp = y_sp_phys(bundle)
    n = min(y_line.shape[0] - 1, ysp.shape[0])
    y_step = y_line[1 : n + 1, :]
    ysp_step = ysp[:n, :]
    return y_step - ysp_step, y_step, ysp_step


def step_slice(bundle: dict, tail_episodes: int | None = None) -> slice:
    nfe = int(bundle.get("nFE", len(bundle["y_sp"])))
    if tail_episodes is None:
        return slice(0, nfe)
    ep_len = int(bundle.get("time_in_sub_episodes", 400))
    start = max(0, nfe - int(tail_episodes) * ep_len)
    return slice(start, nfe)


def reward_summary(avg_rewards: np.ndarray | None) -> dict[str, float]:
    if avg_rewards is None or len(avg_rewards) == 0:
        return {
            "reward_mean": math.nan,
            "reward_tail20": math.nan,
            "reward_tail10": math.nan,
            "reward_final": math.nan,
            "reward_first20_min": math.nan,
            "reward_first20_mean": math.nan,
            "reward_best": math.nan,
            "reward_best_episode": math.nan,
        }
    avg_rewards = np.asarray(avg_rewards, float)
    return {
        "reward_mean": float(np.nanmean(avg_rewards)),
        "reward_tail20": float(np.nanmean(avg_rewards[-20:])),
        "reward_tail10": float(np.nanmean(avg_rewards[-10:])),
        "reward_final": float(avg_rewards[-1]),
        "reward_first20_min": float(np.nanmin(avg_rewards[:20])),
        "reward_first20_mean": float(np.nanmean(avg_rewards[:20])),
        "reward_best": float(np.nanmax(avg_rewards)),
        "reward_best_episode": float(np.nanargmax(avg_rewards) + 1),
    }


def tracking_summary(bundle: dict, tail_episodes: int = 20) -> dict[str, float]:
    err, _y, ysp = step_error_phys(bundle)
    sl = step_slice(bundle, tail_episodes)
    sl = slice(sl.start, min(sl.stop, err.shape[0]))
    e = err[sl, :]
    ysp_tail = ysp[sl, :]
    band = np.maximum(K_REL * np.abs(ysp_tail), BAND_FLOOR_PHYS)
    abs_e = np.abs(e)
    result = {
        "tail20_comp_rmse": float(np.sqrt(np.nanmean(e[:, 0] ** 2))),
        "tail20_temp_rmse": float(np.sqrt(np.nanmean(e[:, 1] ** 2))),
        "tail20_comp_mae": float(np.nanmean(abs_e[:, 0])),
        "tail20_temp_mae": float(np.nanmean(abs_e[:, 1])),
        "tail20_band_norm_mae": float(np.nanmean(abs_e / np.maximum(band, 1e-12))),
        "tail20_outside_band_frac": float(np.nanmean(abs_e > band)),
        "tail20_comp_final_abs": float(abs_e[-1, 0]),
        "tail20_temp_final_abs": float(abs_e[-1, 1]),
    }

    ep_len = int(bundle.get("time_in_sub_episodes", 400))
    half = max(1, ep_len // 2)
    idx = np.arange(sl.start, sl.stop)
    phase1 = (idx % ep_len) < half
    for name, mask in (("sp1", phase1), ("sp2", ~phase1)):
        if np.any(mask):
            result[f"tail20_{name}_comp_mae"] = float(np.nanmean(abs_e[mask, 0]))
            result[f"tail20_{name}_temp_mae"] = float(np.nanmean(abs_e[mask, 1]))
        else:
            result[f"tail20_{name}_comp_mae"] = math.nan
            result[f"tail20_{name}_temp_mae"] = math.nan
    return result


def input_summary(bundle: dict, tail_episodes: int = 20) -> dict[str, float]:
    u = u_phys_step(bundle)
    n = min(int(bundle.get("nFE", len(u))), len(u))
    u = u[:n, :]
    du = np.diff(u, axis=0, prepend=u[[0], :])
    sl = step_slice(bundle, tail_episodes)
    sl = slice(sl.start, min(sl.stop, n))
    u_tail = u[sl, :]
    du_tail = du[sl, :]
    lower = INPUT_BOUNDS["u_min"]
    upper = INPUT_BOUNDS["u_max"]
    tol = np.array([100.0, 0.05], dtype=float)
    sat = (u_tail <= lower + tol) | (u_tail >= upper - tol)
    return {
        "tail20_du_l1_mean": float(np.nanmean(np.abs(du_tail))),
        "tail20_du_l2_mean": float(np.nanmean(np.linalg.norm(du_tail, axis=1))),
        "tail20_input_saturation_frac": float(np.nanmean(sat)),
        "tail20_reflux_min": float(np.nanmin(u_tail[:, 0])),
        "tail20_reflux_max": float(np.nanmax(u_tail[:, 0])),
        "tail20_reboiler_min": float(np.nanmin(u_tail[:, 1])),
        "tail20_reboiler_max": float(np.nanmax(u_tail[:, 1])),
    }


def mechanism_summary(bundle: dict, method: str, tail_episodes: int = 20) -> dict[str, float | str | bool]:
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    sl = step_slice(bundle, tail_episodes)
    sl = slice(sl.start, min(sl.stop, nfe))
    rho_eff_for_inference = arr(bundle, "rho_eff_log")
    residual_authority_inferred = bool(
        bundle.get(
            "residual_authority_enabled",
            rho_eff_for_inference is not None and np.any(np.isfinite(np.asarray(rho_eff_for_inference, float))),
        )
    )
    out: dict[str, float | str | bool] = {
        "bc_handoff_enabled": bool(bundle.get("bc_handoff_enabled", False)),
        "protected_release_gate_enabled": bool(bundle.get("protected_bc_release_gate_enabled", False)),
        "td3_authority_ramp_enabled": bool(bundle.get("td3_authority_ramp_enabled", False)),
        "authority_use_rho": bool(bundle.get("authority_use_rho", bundle.get("use_rho_authority", False))),
        "append_rho_to_state": bool(bundle.get("append_rho_to_state", False)),
        "residual_authority_enabled": residual_authority_inferred,
        "markov_priority_enabled": bool((bundle.get("td3_priority_fallback") or {}).get("enabled", False)),
        "force_td3_execute": bool(bundle.get("force_td3_execute", False)),
    }

    auth = arr(bundle, "bc_handoff_authority_log")
    out["handoff_authority_first10ep_mean"] = finite_mean(auth[: min(len(auth), 4000)] if auth is not None else None)
    out["handoff_authority_tail20_mean"] = finite_mean(auth[sl] if auth is not None else None)

    release = arr(bundle, "release_gate_released_log")
    blocked = arr(bundle, "release_gate_blocked_log")
    if release is None:
        release = arr(bundle, "rl_release_gate_released_log")
        blocked = arr(bundle, "rl_release_gate_blocked_log")
    out["release_released_tail20_frac"] = finite_mean(release[sl] if release is not None else None)
    out["release_blocked_tail20_frac"] = finite_mean(blocked[sl] if blocked is not None else None)

    sat = arr(bundle, "action_saturation_trace")
    out["action_saturation_tail20_mean"] = finite_mean(sat[-20 * int(bundle.get("time_in_sub_episodes", 400)) :] if sat is not None else None)

    if method == "weights":
        weight_log = arr(bundle, "weight_log")
        if weight_log is not None:
            w = weight_log[sl, :]
            out["tail20_weight_mean_q1"] = finite_mean(w[:, 0])
            out["tail20_weight_mean_q2"] = finite_mean(w[:, 1])
            out["tail20_weight_mean_r1"] = finite_mean(w[:, 2])
            out["tail20_weight_mean_r2"] = finite_mean(w[:, 3])
            out["tail20_weight_std_mean"] = finite_mean(np.std(w, axis=0))
            out["final_weight_q1"] = float(weight_log[-1, 0])
            out["final_weight_q2"] = float(weight_log[-1, 1])
            out["final_weight_r1"] = float(weight_log[-1, 2])
            out["final_weight_r2"] = float(weight_log[-1, 3])

    if method == "residual":
        raw = arr(bundle, "delta_u_res_raw_log")
        exe = arr(bundle, "delta_u_res_exec_log")
        if raw is not None:
            raw_norm = np.linalg.norm(raw, axis=1)
            out["tail20_raw_residual_norm_mean"] = finite_mean(raw_norm[sl])
            out["tail20_raw_residual_norm_q95"] = finite_quantile(raw_norm[sl], 0.95)
        if exe is not None:
            exe_norm = np.linalg.norm(exe, axis=1)
            out["tail20_exec_residual_norm_mean"] = finite_mean(exe_norm[sl])
            out["tail20_exec_residual_norm_q95"] = finite_quantile(exe_norm[sl], 0.95)
        if raw is not None and exe is not None:
            raw_norm = np.linalg.norm(raw, axis=1)
            exe_norm = np.linalg.norm(exe, axis=1)
            diff_norm = np.linalg.norm(exe - raw, axis=1)
            out["tail20_exec_raw_norm_ratio"] = finite_mean(exe_norm[sl] / np.maximum(raw_norm[sl], 1e-12))
            out["tail20_exec_raw_diff_norm_mean"] = finite_mean(diff_norm[sl])
            out["tail20_exec_raw_diff_norm_q95"] = finite_quantile(diff_norm[sl], 0.95)
            out["tail20_material_projection_frac"] = float(np.mean(diff_norm[sl] > 1.0e-6))
        for key, label in (
            ("projection_active_log", "projection_active_tail20_frac"),
            ("projection_due_to_authority_log", "projection_authority_tail20_frac"),
            ("projection_due_to_deadband_log", "projection_deadband_tail20_frac"),
            ("projection_due_to_headroom_log", "projection_headroom_tail20_frac"),
            ("deadband_active_log", "deadband_active_tail20_frac"),
        ):
            value = arr(bundle, key)
            out[label] = finite_mean(value[sl] if value is not None else None)
        rho_eff = arr(bundle, "rho_eff_log")
        out["tail20_rho_eff_mean"] = finite_mean(rho_eff[sl] if rho_eff is not None else None)

    if method in {"horizon", "dueling_horizon"}:
        horizon = arr(bundle, "horizon_trace")
        if horizon is not None:
            h = horizon[sl, :]
            out["tail20_predict_h_mean"] = finite_mean(h[:, 0])
            out["tail20_control_h_mean"] = finite_mean(h[:, 1])
            out["tail20_unique_horizon_pairs"] = float(len({tuple(map(int, row)) for row in h.astype(int)}))
            out["tail20_predict_h_std"] = finite_mean(np.std(h[:, 0]))
            out["tail20_control_h_std"] = finite_mean(np.std(h[:, 1]))

    if method == "markov":
        src = arr(bundle, "rl_action_source_log")
        names = bundle.get("rl_action_source_names", {}) or {}
        if src is not None:
            src_tail = src[sl].astype(int)
            for code, name in names.items():
                out[f"tail20_source_{name}_frac"] = float(np.mean(src_tail == int(code)))
        accepted = arr(bundle, "accepted_log")
        fallback = arr(bundle, "fallback_log")
        out["tail20_accepted_frac"] = finite_mean(accepted[sl] if accepted is not None else None)
        out["tail20_fallback_frac"] = finite_mean(fallback[sl] if fallback is not None else None)
        z_exec = arr(bundle, "z_executed_log")
        if z_exec is not None:
            z_norm = np.linalg.norm(z_exec, axis=1)
            out["tail20_z_norm_mean"] = finite_mean(z_norm[sl])
            out["tail20_abs_z_q95"] = finite_quantile(np.abs(z_exec[sl, :]).reshape(-1), 0.95)
        for key, label in (
            ("z_safety_requested_projection_active_log", "z_requested_projection_tail20_frac"),
            ("z_safety_requested_coord_clip_active_log", "z_requested_coord_clip_tail20_frac"),
            ("z_safety_requested_vector_projection_active_log", "z_requested_vector_projection_tail20_frac"),
            ("requested_cost_guard_pass_log", "requested_cost_guard_pass_tail20_frac"),
            ("executed_cost_guard_pass_log", "executed_cost_guard_pass_tail20_frac"),
        ):
            value = arr(bundle, key)
            out[label] = finite_mean(value[sl] if value is not None else None)
        pred = arr(bundle, "requested_prediction_score_log")
        cost = arr(bundle, "requested_cost_margin_log")
        gain = arr(bundle, "requested_gain_drift_log")
        out["tail20_requested_prediction_score_mean"] = finite_mean(pred[sl] if pred is not None else None)
        out["tail20_requested_cost_margin_mean"] = finite_mean(cost[sl] if cost is not None else None)
        out["tail20_requested_gain_drift_mean"] = finite_mean(gain[sl] if gain is not None else None)

    return out


def method_row(key: str, bundle: dict, baseline_rewards: np.ndarray | None = None) -> dict:
    meta = RUNS[key]
    avg = np.asarray(bundle.get("avg_rewards"), float) if bundle.get("avg_rewards") is not None else None
    if key == "baseline" and baseline_rewards is not None:
        avg = baseline_rewards
    row = {
        "key": key,
        "label": meta["label"],
        "batch": meta["batch"],
        "method": meta["method"],
        "path": str(meta["path"].relative_to(ROOT)),
    }
    row.update(reward_summary(avg))
    row.update(tracking_summary(bundle))
    row.update(input_summary(bundle))
    row.update(mechanism_summary(bundle, meta["method"]))
    return row


def save_latest_reward_plot(summary_df: pd.DataFrame, bundles: dict[str, dict], baseline_rewards: np.ndarray) -> None:
    plt.figure(figsize=(9.5, 4.8))
    base_rewards = np.asarray(baseline_rewards, float)
    episodes = np.arange(1, len(base_rewards) + 1)
    plt.plot(episodes, base_rewards, color="black", linewidth=2.0, label="OF-MPC recomputed")
    colors = {
        "weights_20260528": "#4878d0",
        "horizon_20260528": "#ee854a",
        "dueling_20260528": "#6acc64",
        "residual_20260528": "#d65f5f",
        "markov_20260528": "#956cb4",
    }
    for key, color in colors.items():
        rewards = np.asarray(bundles[key]["avg_rewards"], float)
        plt.plot(episodes, rewards, linewidth=1.6, label=RUNS[key]["label"], color=color)
    plt.axhline(float(np.mean(base_rewards[-20:])), color="black", linewidth=1.0, linestyle=":")
    plt.xlabel("Subepisode")
    plt.ylabel("Average reward")
    plt.title("Latest May 28 distillation batch: reward trajectories")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_latest_reward_trajectories.png", dpi=180)
    plt.close()

    latest = summary_df[summary_df["key"].isin(LATEST_KEYS)].copy()
    latest = latest.sort_values("reward_tail20", ascending=True)
    plt.figure(figsize=(8.8, 4.4))
    bars = plt.barh(latest["label"], latest["reward_tail20"], color="#4c78a8")
    for bar, value in zip(bars, latest["reward_tail20"]):
        plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    plt.xlabel("Tail-20 average reward")
    plt.title("Latest batch tail reward ranking")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_latest_tail_reward_ranking.png", dpi=180)
    plt.close()


def save_latest_tracking_plot(summary_df: pd.DataFrame) -> None:
    latest = summary_df[summary_df["key"].isin(LATEST_KEYS)].copy()
    latest = latest.sort_values("reward_tail20", ascending=False)
    x = np.arange(len(latest))
    width = 0.32
    fig, ax1 = plt.subplots(figsize=(9.5, 4.7))
    ax1.bar(x - width / 2, latest["tail20_comp_mae"], width, label="Composition MAE", color="#4c78a8")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, latest["tail20_temp_mae"], width, label="Temperature MAE", color="#f58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(latest["label"], rotation=22, ha="right")
    ax1.set_ylabel("Tray-24 composition MAE")
    ax2.set_ylabel("Tray-85 temperature MAE")
    ax1.set_title("Tail-20 tracking errors in physical coordinates")
    ax1.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_latest_tail_tracking_metrics.png", dpi=180)
    plt.close(fig)


def save_batch_comparison_plot(summary_df: pd.DataFrame) -> None:
    methods = ["weights", "horizon", "dueling_horizon", "residual", "markov"]
    batches = [
        "2026-05-22 protected gate",
        "2026-05-23 controlled authority",
        "2026-05-28 BC handoff",
    ]
    pivot = (
        summary_df[summary_df["method"].isin(methods) & summary_df["batch"].isin(batches)]
        .pivot_table(index="method", columns="batch", values="reward_tail20", aggfunc="first")
        .reindex(methods)
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.24
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, batch in enumerate(batches):
        values = pivot[batch].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, values, width, label=batch, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(["Weights", "Horizon", "Dueling", "Residual", "Markov"], rotation=0)
    ax.set_ylabel("Tail-20 average reward")
    ax.set_title("Effect of safety/authority changes across comparable batches")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_batch_tail_reward_comparison.png", dpi=180)
    plt.close(fig)


def save_residual_mechanism_plot(mechanism_df: pd.DataFrame) -> None:
    keys = ["residual_best_rho", "residual_20260522", "residual_20260523", "residual_20260528"]
    labels = {
        "residual_best_rho": "May 7 rho reference",
        "residual_20260522": "May 22 gated",
        "residual_20260523": "May 23 rho authority",
        "residual_20260528": "May 28 no rho",
    }
    df = mechanism_df[mechanism_df["key"].isin(keys)].set_index("key").loc[keys].reset_index()
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.bar(x - width / 2, df["tail20_raw_residual_norm_mean"], width, label="Raw residual norm", color="#4c78a8")
    ax.bar(x + width / 2, df["tail20_exec_residual_norm_mean"], width, label="Executed residual norm", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in df["key"]], rotation=20, ha="right")
    ax.set_ylabel("Tail-20 mean scaled-input residual norm")
    ax.set_title("Residual authority changed the raw/executed correction relationship")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_raw_vs_executed_norm.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.8, 4.8))
    ax1.bar(x - width / 2, df["reward_tail20"], width, color="#72b7b2", label="Tail-20 reward")
    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2,
        df["projection_authority_tail20_frac"].fillna(0.0),
        width,
        color="#b279a2",
        label="Authority projection fraction",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[k] for k in df["key"]], rotation=20, ha="right")
    ax1.set_ylabel("Tail-20 reward")
    ax2.set_ylabel("Tail-20 authority projection fraction")
    ax1.set_title("Residual performance versus rho/headroom authority")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_reward_vs_authority.png", dpi=180)
    plt.close(fig)


def save_markov_mechanism_plot(summary_df: pd.DataFrame, mechanism_df: pd.DataFrame) -> None:
    keys = ["markov_20260518_soft", "markov_20260518_no_safeguard", "markov_20260522", "markov_20260523", "markov_20260528"]
    labels = {
        "markov_20260518_soft": "May 18 soft",
        "markov_20260518_no_safeguard": "May 18 no safety",
        "markov_20260522": "May 22 gated",
        "markov_20260523": "May 23 override",
        "markov_20260528": "May 28 force TD3",
    }
    df = summary_df[summary_df["key"].isin(keys)].set_index("key").loc[keys].reset_index()
    mech = mechanism_df[mechanism_df["key"].isin(keys)].set_index("key").loc[keys].reset_index()
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(keys))
    ax.plot(x, df["reward_tail20"], marker="o", linewidth=2.0, color="#4c78a8", label="Tail-20 reward")
    ax.plot(x, df["reward_first20_min"], marker="s", linewidth=1.7, color="#e45756", label="Worst first-20 reward")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("Average reward")
    ax.set_title("Markov TD3 shows high upside but needs candidate safety")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_markov_safety_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    td3_cols = [c for c in mech.columns if c.endswith("source_td3_accepted_frac")]
    td3_col = td3_cols[0] if td3_cols else None
    td3_frac = mech[td3_col].fillna(0.0) if td3_col else np.zeros(len(mech))
    guard = mech.get("requested_cost_guard_pass_tail20_frac", pd.Series(np.nan, index=mech.index))
    pred = mech.get("tail20_requested_prediction_score_mean", pd.Series(np.nan, index=mech.index))
    ax.bar(x - 0.2, td3_frac, 0.38, label="TD3 source fraction", color="#54a24b")
    ax.bar(x + 0.2, guard.fillna(0.0), 0.38, label="Cost-guard pass fraction", color="#f58518")
    ax2 = ax.twinx()
    ax2.plot(x, pred, color="#b279a2", marker="o", label="Prediction score")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tail-20 fraction")
    ax2.set_ylabel("Tail-20 requested prediction score")
    ax.set_title("Markov authority without screening admits bad candidates")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_markov_source_and_guard.png", dpi=180)
    plt.close(fig)


def save_tail_overlay(bundles: dict[str, dict]) -> None:
    keys = ["baseline", "weights_20260528", "horizon_20260528", "dueling_20260528", "residual_20260528", "markov_20260528"]
    labels = {key: RUNS[key]["label"] for key in keys}
    colors = {
        "baseline": "black",
        "weights_20260528": "#4878d0",
        "horizon_20260528": "#ee854a",
        "dueling_20260528": "#6acc64",
        "residual_20260528": "#d65f5f",
        "markov_20260528": "#956cb4",
    }
    base = bundles["baseline"]
    ep_len = int(base.get("time_in_sub_episodes", 400))
    start = int(base.get("nFE", 80000)) - 2 * ep_len
    stop = int(base.get("nFE", 80000))
    t = np.arange(stop - start) * float(base.get("delta_t", 1.0 / 6.0))
    ysp = y_sp_phys(base)[start:stop, :]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True)
    output_names = ["Tray-24 C2H6 composition", "Tray-85 temperature"]
    for j, ax in enumerate(axes):
        ax.step(t, ysp[:, j], where="post", color="black", linestyle=":", linewidth=1.8, label="Setpoint")
        for key in keys:
            y_line = y_phys_line(bundles[key])
            y_seg = y_line[start + 1 : stop + 1, j]
            ax.plot(t[: len(y_seg)], y_seg, label=labels[key], color=colors[key], linewidth=1.15, alpha=0.95)
        ax.set_ylabel(output_names[j])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Tail time (h), last 2 subepisodes")
    axes[0].set_title("Latest May 28 output tracking tail overlay")
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_latest_tail_output_overlay.png", dpi=180)
    plt.close(fig)


def save_horizon_pair_counts(bundles: dict[str, dict]) -> None:
    rows = []
    for key in ("horizon_20260528", "dueling_20260528"):
        bundle = bundles[key]
        horizon = arr(bundle, "horizon_trace")
        if horizon is None:
            continue
        sl = step_slice(bundle, 20)
        h = horizon[sl, :].astype(int)
        unique, counts = np.unique(h, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        for rank, idx in enumerate(order[:20], start=1):
            rows.append(
                {
                    "runner": key,
                    "label": RUNS[key]["label"],
                    "rank": rank,
                    "predict_h": int(unique[idx, 0]),
                    "control_h": int(unique[idx, 1]),
                    "tail20_count": int(counts[idx]),
                    "tail20_fraction": float(counts[idx] / max(1, h.shape[0])),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "latest_horizon_pair_counts.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {}
    missing = []
    for key, meta in RUNS.items():
        path = meta["path"]
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        bundles[key] = load_pickle(path)

    if missing:
        raise FileNotFoundError("Missing expected bundles:\n" + "\n".join(missing))

    compare_bundle = load_pickle(LATEST_COMPARE_PATH)
    baseline_rewards = np.asarray(compare_bundle.get("avg_rewards_mpc"), float)
    if baseline_rewards.size == 0:
        raise ValueError(f"No avg_rewards_mpc found in {LATEST_COMPARE_PATH.relative_to(ROOT)}")

    rows = [method_row(key, bundles[key], baseline_rewards=baseline_rewards) for key in bundles]
    summary_df = pd.DataFrame(rows)
    mechanism_cols = [
        c
        for c in summary_df.columns
        if c
        in {
            "key",
            "label",
            "batch",
            "method",
            "path",
            "reward_tail20",
            "reward_tail10",
            "reward_final",
            "bc_handoff_enabled",
            "protected_release_gate_enabled",
            "td3_authority_ramp_enabled",
            "authority_use_rho",
            "append_rho_to_state",
            "residual_authority_enabled",
            "markov_priority_enabled",
            "force_td3_execute",
        }
        or "tail20_" in c
        or c.endswith("_frac")
        or c.startswith("handoff_")
        or c.startswith("release_")
    ]
    mechanism_df = summary_df[mechanism_cols].copy()

    summary_df.to_csv(OUT_DIR / "latest_settings_summary_metrics.csv", index=False)
    mechanism_df.to_csv(OUT_DIR / "latest_settings_mechanism_diagnostics.csv", index=False)

    save_latest_reward_plot(summary_df, bundles, baseline_rewards)
    save_latest_tracking_plot(summary_df)
    save_batch_comparison_plot(summary_df)
    save_residual_mechanism_plot(mechanism_df)
    save_markov_mechanism_plot(summary_df, mechanism_df)
    save_tail_overlay(bundles)
    save_horizon_pair_counts(bundles)

    latest = summary_df[summary_df["key"].isin(LATEST_KEYS)].copy()
    latest_rank = latest.sort_values("reward_tail20", ascending=False)[
        ["label", "reward_tail20", "reward_tail10", "reward_final", "tail20_band_norm_mae", "tail20_outside_band_frac"]
    ].to_dict(orient="records")
    residual_rows = mechanism_df[mechanism_df["method"] == "residual"][
        [
            "key",
            "label",
            "batch",
            "reward_tail20",
            "reward_final",
            "authority_use_rho",
            "residual_authority_enabled",
            "tail20_raw_residual_norm_mean",
            "tail20_exec_residual_norm_mean",
            "projection_authority_tail20_frac",
            "projection_deadband_tail20_frac",
            "projection_headroom_tail20_frac",
            "tail20_rho_eff_mean",
        ]
    ].to_dict(orient="records")
    markov_latest = mechanism_df[mechanism_df["key"] == "markov_20260528"].to_dict(orient="records")[0]

    summary = {
        "latest_rank_tail20": latest_rank,
        "residual_rows": residual_rows,
        "markov_20260528": markov_latest,
        "artifacts": {
            "summary_csv": str((OUT_DIR / "latest_settings_summary_metrics.csv").relative_to(ROOT)),
            "mechanism_csv": str((OUT_DIR / "latest_settings_mechanism_diagnostics.csv").relative_to(ROOT)),
            "horizon_pair_counts_csv": str((OUT_DIR / "latest_horizon_pair_counts.csv").relative_to(ROOT)),
            "figures": sorted(p.name for p in OUT_DIR.glob("fig_*.png")),
        },
    }
    (OUT_DIR / "latest_settings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
