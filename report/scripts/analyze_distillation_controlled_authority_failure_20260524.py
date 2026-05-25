from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS


RESULTS = REPO_ROOT / "Distillation" / "Results"
FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_controlled_authority_failure_20260524"
SUMMARY_CSV = FIG_DIR / "summary_metrics.csv"
AUTHORITY_CSV = FIG_DIR / "authority_diagnostics.csv"
MECHANISM_JSON = FIG_DIR / "mechanism_summary.json"
N_INPUTS = 2
TAIL_EPISODES = 10


RUNS = [
    {
        "method": "OF-MPC",
        "family": "baseline",
        "batch": "baseline",
        "path": REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
        "color": "#111827",
        "linestyle": "-",
    },
    {
        "method": "TD3 Weights",
        "family": "weights",
        "batch": "previous_blocked",
        "path": RESULTS
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260522_181031"
        / "input_data.pkl",
        "color": "#a78bfa",
        "linestyle": "--",
    },
    {
        "method": "TD3 Residual",
        "family": "residual",
        "batch": "previous_blocked",
        "path": RESULTS
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260522_180223"
        / "input_data.pkl",
        "color": "#fca5a5",
        "linestyle": "--",
    },
    {
        "method": "TD3 Markov",
        "family": "markov",
        "batch": "previous_blocked",
        "path": RESULTS / "distillation_markov_td3_disturb_fluctuation_unified" / "20260522_190448" / "input_data.pkl",
        "color": "#86efac",
        "linestyle": "--",
    },
    {
        "method": "TD3 Weights",
        "family": "weights",
        "batch": "latest_controlled_authority",
        "path": RESULTS
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260523_204146"
        / "input_data.pkl",
        "color": "#7c3aed",
        "linestyle": "-",
    },
    {
        "method": "TD3 Residual",
        "family": "residual",
        "batch": "latest_controlled_authority",
        "path": RESULTS
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260523_203649"
        / "input_data.pkl",
        "color": "#dc2626",
        "linestyle": "-",
    },
    {
        "method": "Horizon DDQN",
        "family": "horizon",
        "batch": "latest_controlled_authority",
        "path": RESULTS / "distillation_horizon_disturb_fluctuation_mismatch_unified" / "20260523_210449" / "input_data.pkl",
        "color": "#2563eb",
        "linestyle": "-",
    },
    {
        "method": "Dueling Horizon",
        "family": "dueling",
        "batch": "latest_controlled_authority",
        "path": RESULTS
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260523_210709"
        / "input_data.pkl",
        "color": "#0891b2",
        "linestyle": "-",
    },
    {
        "method": "TD3 Markov",
        "family": "markov",
        "batch": "latest_controlled_authority",
        "path": RESULTS / "distillation_markov_td3_disturb_fluctuation_unified" / "20260523_224108" / "input_data.pkl",
        "color": "#16a34a",
        "linestyle": "-",
    },
]


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def arr(value: Any, dtype=float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    try:
        data = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return np.asarray([], dtype=dtype)
    if data.ndim == 0 and data.dtype == object:
        return np.asarray([], dtype=dtype)
    return data


def nanmean(value: Any, default=np.nan) -> float:
    data = arr(value, float).reshape(-1)
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else float(default)


def tail_mean(value: Any, tail: slice) -> float:
    data = arr(value, float)
    if data.ndim == 0 or data.size == 0:
        return float("nan")
    return nanmean(data[tail])


def reverse_min_max(value: Any, data_min: Any, data_max: Any) -> np.ndarray:
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return np.asarray(value, float) * np.maximum(data_max - data_min, 1.0e-12) + data_min


def min_max_scale(value: Any, data_min: Any, data_max: Any) -> np.ndarray:
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return (np.asarray(value, float) - data_min) / np.maximum(data_max - data_min, 1.0e-12)


def y_ss_scaled(bundle: dict[str, Any]) -> np.ndarray:
    steady_states = bundle.get("steady_states", {})
    y_ss = arr(steady_states.get("y_ss") if isinstance(steady_states, dict) else None)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if y_ss.size != 2 or data_min.size < 4 or data_max.size < 4:
        return np.asarray([], float)
    return min_max_scale(y_ss, data_min[N_INPUTS:], data_max[N_INPUTS:])


def physical_setpoints(bundle: dict[str, Any]) -> np.ndarray:
    y_sp = arr(bundle.get("y_sp"))
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    ss_scaled = y_ss_scaled(bundle)
    if y_sp.ndim != 2 or ss_scaled.size != y_sp.shape[1]:
        return np.asarray([], float)
    return reverse_min_max(y_sp + ss_scaled.reshape(1, -1), data_min[N_INPUTS:], data_max[N_INPUTS:])


def y_steps(bundle: dict[str, Any]) -> np.ndarray:
    y = arr(bundle.get("y_rl", bundle.get("y")))
    y_sp = arr(bundle.get("y_sp"))
    if y.ndim != 2 or y_sp.ndim != 2:
        return np.asarray([], float)
    n = min(y.shape[0] - 1, y_sp.shape[0])
    return y[1 : n + 1] if n > 0 else np.asarray([], float)


def tail_slice(bundle: dict[str, Any], episodes: int = TAIL_EPISODES) -> slice:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    width = min(nfe, steps * episodes)
    return slice(max(0, nfe - width), nfe)


def post_warm_slice(bundle: dict[str, Any]) -> slice:
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    return slice(min(nfe, int(bundle.get("warm_start_step") or 0) + 1), nfe)


def episode_mean(series: Any, bundle: dict[str, Any]) -> np.ndarray:
    data = arr(series, float)
    if data.ndim == 0 or data.size == 0:
        return np.asarray([], float)
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    n = (data.shape[0] // steps) * steps
    if n <= 0:
        return np.asarray([], float)
    shaped = data[:n].reshape((-1, steps) + data.shape[1:])
    return np.nanmean(shaped, axis=1)


def recompute_rewards(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    delta_y = arr(bundle.get("delta_y_storage"))
    delta_u = arr(bundle.get("delta_u_storage"))
    y_sp_phys = physical_setpoints(bundle)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if delta_y.ndim != 2 or delta_u.ndim != 2 or y_sp_phys.ndim != 2 or data_min.size < 4:
        return np.asarray([], float), arr(bundle.get("avg_rewards"))
    n = min(delta_y.shape[0], delta_u.shape[0], y_sp_phys.shape[0])
    delta_y = delta_y[:n]
    delta_u = delta_u[:n]
    y_sp_phys = y_sp_phys[:n]

    params = RL_REWARD_DEFAULTS
    dy_scale = np.maximum(data_max[N_INPUTS:] - data_min[N_INPUTS:], 1.0e-12)
    q_diag = arr(params["Q_diag"])
    r_diag = arr(params["R_diag"])
    band_scaled = np.maximum(
        arr(params["k_rel"]).reshape(1, -1) * np.abs(y_sp_phys),
        arr(params["band_floor_phys"]).reshape(1, -1),
    ) / dy_scale.reshape(1, -1)
    tau_scaled = float(params.get("tau_frac", 0.7)) * band_scaled
    abs_e = np.abs(delta_y)
    s_i = 1.0 / (1.0 + np.exp(-np.clip((band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12), -60.0, 60.0)))
    w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])
    err_quad = np.sum(q_diag.reshape(1, -1) * delta_y**2, axis=1)
    move = np.sum(r_diag.reshape(1, -1) * delta_u**2, axis=1)
    slope_at_edge = 2.0 * q_diag.reshape(1, -1) * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    lin_out = (1.0 - w_in) * np.sum(float(params.get("gamma_out", 0.5)) * slope_at_edge * overflow, axis=1)
    lin_in = w_in * np.sum(float(params.get("gamma_in", 0.5)) * slope_at_edge * inside_mag, axis=1)
    z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
    bonus_k = float(params.get("bonus_k", 12.0))
    phi = (np.exp(-bonus_k * z) - np.exp(-bonus_k)) / (1.0 - np.exp(-bonus_k))
    qb2 = q_diag.reshape(1, -1) * band_scaled**2
    bonus = w_in * float(params.get("beta", 7.0)) * np.sum(qb2 * phi, axis=1)
    rewards = (-(err_quad + move + lin_out + lin_in) + bonus) * float(params.get("reward_scale", 1.0))
    avg = episode_mean(rewards, bundle)
    return rewards, avg


def tracking_metrics(bundle: dict[str, Any], tail: slice) -> dict[str, float]:
    y = y_steps(bundle)
    y_sp = physical_setpoints(bundle)
    if y.size == 0 or y_sp.size == 0:
        return {
            "x24_rmse_tail": np.nan,
            "T85_rmse_tail": np.nan,
            "band_norm_mae_tail": np.nan,
            "outside_band_tail_frac": np.nan,
        }
    n = min(y.shape[0], y_sp.shape[0])
    err = y[:n] - y_sp[:n]
    err_tail = err[tail]
    band = np.maximum(
        arr(RL_REWARD_DEFAULTS["k_rel"]).reshape(1, -1) * np.abs(y_sp[:n]),
        arr(RL_REWARD_DEFAULTS["band_floor_phys"]).reshape(1, -1),
    )
    band_tail = band[tail]
    norm_tail = np.abs(err_tail) / np.maximum(band_tail, 1.0e-12)
    return {
        "x24_rmse_tail": float(np.sqrt(np.mean(err_tail[:, 0] ** 2))),
        "T85_rmse_tail": float(np.sqrt(np.mean(err_tail[:, 1] ** 2))),
        "band_norm_mae_tail": float(np.mean(norm_tail)),
        "outside_band_tail_frac": float(np.mean(np.any(norm_tail > 1.0, axis=1))),
    }


def source_fraction(bundle: dict[str, Any], key: str, code: int, sl: slice) -> float:
    data = arr(bundle.get(key), int)
    if data.ndim == 0 or data.size == 0:
        return float("nan")
    return float(np.mean(data[sl] == int(code)))


def family_latest(rows: list[dict[str, Any]], family: str) -> dict[str, Any] | None:
    for row in rows:
        if row["family"] == family and row["batch"] == "latest_controlled_authority":
            return row
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(run: dict[str, Any], bundle: dict[str, Any], avg_rewards: np.ndarray) -> dict[str, Any]:
    tail = tail_slice(bundle)
    metrics = tracking_metrics(bundle, tail)
    delta_u = arr(bundle.get("delta_u_storage"))
    return {
        "method": run["method"],
        "family": run["family"],
        "batch": run["batch"],
        "bundle": str(run["path"].relative_to(REPO_ROOT)),
        "episodes": int(avg_rewards.size),
        "warm_episodes": float((bundle.get("warm_start_step") or 0) / max(1, int(bundle.get("time_in_sub_episodes") or 400))),
        "current_tail_reward": float(np.mean(avg_rewards[-min(TAIL_EPISODES, avg_rewards.size) :])) if avg_rewards.size else np.nan,
        "current_final_reward": float(avg_rewards[-1]) if avg_rewards.size else np.nan,
        **metrics,
        "mean_abs_du_scaled_tail": float(np.mean(np.abs(delta_u[tail]))) if delta_u.ndim == 2 else np.nan,
    }


def summarize_authority(run: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    post = post_warm_slice(bundle)
    tail = tail_slice(bundle)
    release_prefix = "rl_" if "rl_release_gate_blocked_log" in bundle else ""
    ramp_prefix = "rl_" if "rl_td3_authority_ramp_live_log" in bundle else ""
    row = {
        "method": run["method"],
        "family": run["family"],
        "batch": run["batch"],
        "release_step": int(bundle.get(f"{release_prefix}protected_bc_release_gate_release_step", -1)),
        "post_release_blocked_frac": tail_mean(bundle.get(f"{release_prefix}release_gate_blocked_log"), post),
        "post_release_released_frac": tail_mean(bundle.get(f"{release_prefix}release_gate_released_log"), post),
        "post_ramp_live_frac": tail_mean(bundle.get(f"{ramp_prefix}td3_authority_ramp_live_log"), post),
        "post_ramp_gate_override_frac": tail_mean(bundle.get(f"{ramp_prefix}td3_authority_ramp_gate_override_log"), post),
        "tail_ramp_projection_frac": tail_mean(bundle.get(f"{ramp_prefix}td3_authority_ramp_projection_active_log"), tail),
        "tail_action_saturation_frac": tail_mean(bundle.get("action_saturation_trace"), slice(-4000, None)),
        "post_bc_gap_mean": tail_mean(bundle.get(f"{release_prefix}bc_policy_target_distance_log", bundle.get("bc_policy_target_distance_log")), post),
        "tail_bc_gap_mean": tail_mean(bundle.get(f"{release_prefix}bc_policy_target_distance_log", bundle.get("bc_policy_target_distance_log")), tail),
        "post_td3_source_frac": source_fraction(bundle, "rl_action_source_log", 2, post),
        "tail_td3_source_frac": source_fraction(bundle, "rl_action_source_log", 2, tail),
        "tail_accepted_frac": tail_mean(bundle.get("accepted_log"), tail),
        "tail_probation_frac": tail_mean(bundle.get("td3_probation_active_log"), tail),
        "tail_authority_scale": tail_mean(bundle.get("td3_authority_scale_log"), tail),
        "tail_requested_prediction_score": tail_mean(bundle.get("requested_prediction_score_log"), tail),
        "tail_requested_cost_guard_pass": tail_mean(bundle.get("requested_cost_guard_pass_log"), tail),
        "tail_requested_cost_margin": tail_mean(bundle.get("requested_cost_margin_log"), tail),
        "actor_update_trace_count": int(arr(bundle.get("actor_losses")).reshape(-1).size),
        "critic_update_trace_count": int(arr(bundle.get("critic_losses")).reshape(-1).size),
        "replay_push_count": int(np.nansum(arr(bundle.get("rl_replay_pushed_log"), float)))
        if "rl_replay_pushed_log" in bundle
        else np.nan,
        "train_update_count": int(np.nansum(arr(bundle.get("rl_train_updated_log"), float)))
        if "rl_train_updated_log" in bundle
        else np.nan,
        "bc_active_step_count": int(np.nansum(arr(bundle.get(f"{release_prefix}bc_active_log", bundle.get("bc_active_log")), float)))
        if (f"{release_prefix}bc_active_log" in bundle or "bc_active_log" in bundle)
        else np.nan,
    }

    if run["family"] == "weights":
        weights = arr(bundle.get("weight_log"))
        if weights.ndim == 2 and weights.shape[1] == 4:
            wt = weights[tail]
            row.update(
                {
                    "tail_Q1_mult": float(np.mean(wt[:, 0])),
                    "tail_Q2_mult": float(np.mean(wt[:, 1])),
                    "tail_R1_mult": float(np.mean(wt[:, 2])),
                    "tail_R2_mult": float(np.mean(wt[:, 3])),
                    "tail_weight_std_sum": float(np.sum(np.std(wt, axis=0))),
                    "tail_weight_at_final_ramp_extreme_frac": float(np.mean((np.isclose(wt, 0.75)) | (np.isclose(wt, 1.25)))),
                }
            )
    if run["family"] == "residual":
        raw = arr(bundle.get("delta_u_res_raw_log"))
        exe = arr(bundle.get("delta_u_res_exec_log"))
        if raw.ndim == 2 and exe.ndim == 2:
            row.update(
                {
                    "tail_raw_residual_norm": float(np.mean(np.linalg.norm(raw[tail], axis=1))),
                    "tail_exec_residual_norm": float(np.mean(np.linalg.norm(exe[tail], axis=1))),
                    "tail_projection_active_frac": tail_mean(bundle.get("projection_active_log"), tail),
                    "tail_projection_authority_frac": tail_mean(bundle.get("projection_due_to_authority_log"), tail),
                }
            )
    if run["family"] == "markov":
        z = arr(bundle.get("z_executed_log"))
        if z.ndim == 2:
            row.update(
                {
                    "tail_z_norm_mean": float(np.mean(np.linalg.norm(z[tail], axis=1))),
                    "tail_q95_abs_z": float(np.quantile(np.abs(z[tail]), 0.95)),
                    "tail_z_pattern": json.dumps(np.mean(z[tail], axis=0).round(6).tolist()),
                    "tail_z_safety_cap": tail_mean(bundle.get("z_safety_effective_cap_log"), tail),
                    "tail_z_projection_frac": tail_mean(bundle.get("z_safety_requested_projection_active_log"), tail),
                }
            )
    if run["family"] in {"horizon", "dueling"}:
        h = arr(bundle.get("horizon_trace"))
        if h.ndim == 2 and h.shape[1] == 2:
            row.update(
                {
                    "tail_Hp_mean": float(np.mean(h[tail, 0])),
                    "tail_Hc_mean": float(np.mean(h[tail, 1])),
                    "tail_distinct_horizons": int(np.unique(h[tail], axis=0).shape[0]),
                    "post_distinct_horizons": int(np.unique(h[post], axis=0).shape[0]),
                }
            )
    return row


def plot_reward_curves(records: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for rec in records:
        run = rec["run"]
        if run["batch"] == "previous_blocked" and run["family"] not in {"weights", "residual", "markov"}:
            continue
        avg = rec["avg_rewards"]
        if avg.size == 0:
            continue
        label = run["method"] if run["batch"] in {"baseline", "latest_controlled_authority"} else f"{run['method']} prev"
        ax.plot(np.arange(1, avg.size + 1), avg, label=label, color=run["color"], linestyle=run["linestyle"], linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Distillation reward curves: latest controlled authority vs previous blocked-safe TD3")
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Current-rescored average reward")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reward_curves_latest_failure.png", dpi=170)
    plt.close(fig)


def plot_tail_bars(summary_rows: list[dict[str, Any]]) -> None:
    latest = [r for r in summary_rows if r["batch"] in {"baseline", "latest_controlled_authority"}]
    labels = [r["method"] for r in latest]
    rewards = [r["current_tail_reward"] for r in latest]
    band = [r["band_norm_mae_tail"] for r in latest]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(labels, rewards, color=["#111827", "#7c3aed", "#dc2626", "#2563eb", "#0891b2", "#16a34a"])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Tail reward")
    axes[0].set_title("Latest tail performance")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(labels, band, color=["#111827", "#7c3aed", "#dc2626", "#2563eb", "#0891b2", "#16a34a"])
    axes[1].set_ylabel("Tail band-normalized MAE")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tail_reward_and_band_error_latest.png", dpi=170)
    plt.close(fig)


def plot_authority_bars(authority_rows: list[dict[str, Any]]) -> None:
    latest = [r for r in authority_rows if r["batch"] == "latest_controlled_authority" and r["family"] in {"weights", "residual", "markov"}]
    metrics = [
        ("post_release_blocked_frac", "BC gate blocked"),
        ("post_ramp_gate_override_frac", "Ramp override"),
        ("tail_action_saturation_frac", "Actor saturation"),
        ("tail_ramp_projection_frac", "Ramp clipped"),
        ("tail_td3_source_frac", "TD3 source"),
        ("tail_accepted_frac", "Accepted"),
        ("tail_probation_frac", "Probation"),
    ]
    x = np.arange(len(latest))
    width = 0.11
    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, (key, label) in enumerate(metrics):
        vals = [0.0 if not np.isfinite(float(r.get(key, np.nan))) else float(r.get(key, 0.0)) for r in latest]
        ax.bar(x + (idx - 3) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([r["method"] for r in latest])
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Latest TD3 authority diagnostics")
    ax.set_ylabel("Fraction")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_td3_authority_failure_bars.png", dpi=170)
    plt.close(fig)


def plot_action_collapse(records: list[dict[str, Any]]) -> None:
    latest_by_family = {rec["run"]["family"]: rec for rec in records if rec["run"]["batch"] == "latest_controlled_authority"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)

    weights = latest_by_family["weights"]["bundle"]
    w_episode = episode_mean(weights.get("weight_log"), weights)
    if w_episode.size:
        for idx, label in enumerate(["Q1", "Q2", "R1", "R2"]):
            axes[0].plot(np.arange(1, w_episode.shape[0] + 1), w_episode[:, idx], label=label, linewidth=2)
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].axhline(0.75, color="tab:red", linestyle=":", linewidth=1)
        axes[0].axhline(1.25, color="tab:red", linestyle=":", linewidth=1)
    axes[0].set_title("Weights TD3 collapsed to authority-ramp extremes")
    axes[0].set_ylabel("Multiplier")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=4)

    residual = latest_by_family["residual"]["bundle"]
    raw = episode_mean(np.linalg.norm(arr(residual.get("delta_u_res_raw_log")), axis=1), residual)
    exe = episode_mean(np.linalg.norm(arr(residual.get("delta_u_res_exec_log")), axis=1), residual)
    if raw.size and exe.size:
        axes[1].plot(np.arange(1, raw.size + 1), raw, label="raw residual norm", color="#f97316")
        axes[1].plot(np.arange(1, exe.size + 1), exe, label="executed residual norm", color="#dc2626")
    axes[1].set_title("Residual TD3 is active but strongly projected by rho/headroom")
    axes[1].set_ylabel("Scaled residual norm")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    markov = latest_by_family["markov"]["bundle"]
    z_episode = episode_mean(markov.get("z_executed_log"), markov)
    if z_episode.size:
        for idx, label in enumerate(["y1/u1", "y1/u2", "y2/u1", "y2/u2"]):
            axes[2].plot(np.arange(1, z_episode.shape[0] + 1), z_episode[:, idx], label=label, linewidth=2)
    axes[2].set_title("Markov TD3 converged to a fixed z pattern")
    axes[2].set_xlabel("Subepisode")
    axes[2].set_ylabel("Executed z")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_td3_action_collapse.png", dpi=170)
    plt.close(fig)


def plot_markov_guard_failure(records: list[dict[str, Any]]) -> None:
    rec = next(r for r in records if r["run"]["family"] == "markov" and r["run"]["batch"] == "latest_controlled_authority")
    bundle = rec["bundle"]
    avg = rec["avg_rewards"]
    accepted = episode_mean(bundle.get("accepted_log"), bundle)
    td3_source = episode_mean(arr(bundle.get("rl_action_source_log")) == 2, bundle)
    probation = episode_mean(bundle.get("td3_probation_active_log"), bundle)
    pred = episode_mean(bundle.get("requested_prediction_score_log"), bundle)
    cost_pass = episode_mean(bundle.get("requested_cost_guard_pass_log"), bundle)
    scale = episode_mean(bundle.get("td3_authority_scale_log"), bundle)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(np.arange(1, avg.size + 1), avg, color="#16a34a", linewidth=2)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Markov failure: reward collapses while TD3 remains accepted")
    axes[0].grid(True, alpha=0.25)

    for series, label in [(accepted, "accepted"), (td3_source, "TD3 source"), (probation, "probation active"), (cost_pass, "strict cost guard pass")]:
        if series.size:
            axes[1].plot(np.arange(1, series.size + 1), series, label=label, linewidth=2)
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].set_ylabel("Fraction")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=2)

    if pred.size:
        axes[2].plot(np.arange(1, pred.size + 1), pred, label="prediction score", color="#ef4444", linewidth=2)
    if scale.size:
        axes[2].plot(np.arange(1, scale.size + 1), scale, label="authority scale", color="#6366f1", linewidth=2)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_xlabel("Subepisode")
    axes[2].set_ylabel("Score / scale")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_markov_guard_failure.png", dpi=170)
    plt.close(fig)


def plot_tail_tracking(records: list[dict[str, Any]]) -> None:
    selected = [
        rec
        for rec in records
        if rec["run"]["batch"] in {"baseline", "latest_controlled_authority"}
        and rec["run"]["family"] in {"baseline", "weights", "residual", "dueling", "markov"}
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for rec in selected:
        bundle = rec["bundle"]
        tail = tail_slice(bundle)
        y = y_steps(bundle)
        y_sp = physical_setpoints(bundle)
        if y.size == 0 or y_sp.size == 0:
            continue
        n = min(y.shape[0], y_sp.shape[0])
        sl = slice(max(0, n - 4000), n)
        t = np.arange(sl.start, sl.stop)
        label = rec["run"]["method"]
        for out_idx, ax in enumerate(axes):
            ax.plot(t, y[sl, out_idx], label=label, color=rec["run"]["color"], linestyle=rec["run"]["linestyle"], alpha=0.9)
            if rec["run"]["family"] == "baseline":
                ax.plot(t, y_sp[sl, out_idx], color="black", linestyle="--", linewidth=1.5, label="setpoint")
    axes[0].set_title("Tail output tracking, latest batch")
    axes[0].set_ylabel("Tray-24 ethane composition")
    axes[1].set_ylabel("Tray-85 temperature")
    axes[1].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tail_tracking_latest.png", dpi=170)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []

    for run in RUNS:
        bundle = load_pickle(run["path"])
        _, avg_rewards = recompute_rewards(bundle)
        records.append({"run": run, "bundle": bundle, "avg_rewards": avg_rewards})
        summary_rows.append(summarize_run(run, bundle, avg_rewards))
        authority_rows.append(summarize_authority(run, bundle))

    previous_by_family = {
        row["family"]: row for row in summary_rows if row["batch"] == "previous_blocked"
    }
    baseline = next(row for row in summary_rows if row["batch"] == "baseline")
    for row in summary_rows:
        row["tail_reward_delta_vs_ofmpc"] = float(row["current_tail_reward"] - baseline["current_tail_reward"])
        prev = previous_by_family.get(row["family"])
        row["tail_reward_delta_vs_previous_same_family"] = (
            float(row["current_tail_reward"] - prev["current_tail_reward"])
            if prev is not None and row["batch"] == "latest_controlled_authority"
            else np.nan
        )

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(AUTHORITY_CSV, authority_rows)

    mechanism = {
        "summary": summary_rows,
        "authority": authority_rows,
        "claim": (
            "The latest controlled-authority batch did not fail because training stopped. "
            "It failed because the diagnostic BC gate was overridden while the actors were still far from safe labels; "
            "weights and Markov then saturated into fixed boundary actions, and Markov's candidate guard accepted the "
            "scaled saturated z despite reward-probation being active."
        ),
    }
    MECHANISM_JSON.write_text(json.dumps(mechanism, indent=2), encoding="utf-8")

    plot_reward_curves(records)
    plot_tail_bars(summary_rows)
    plot_authority_bars(authority_rows)
    plot_action_collapse(records)
    plot_markov_guard_failure(records)
    plot_tail_tracking(records)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {AUTHORITY_CSV}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
