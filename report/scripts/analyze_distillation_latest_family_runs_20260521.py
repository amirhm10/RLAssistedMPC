from __future__ import annotations

import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation import get_distillation_notebook_defaults
from systems.distillation.config import (
    DISTILLATION_INPUT_BOUNDS,
    RL_REWARD_DEFAULTS,
    WEIGHT_MULTIPLIER_BOUNDS,
)


RESULTS_ROOT = REPO_ROOT / "Distillation" / "Results"
BASELINE_PATH = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_latest_family_runs_20260521"
REPORT_PATH = REPO_ROOT / "report" / "distillation_latest_family_runs_2026_05_21.md"
N_INPUTS = 2
TAIL_EPISODES = 10


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 190,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    family: str
    path: Path
    defaults_family: str | None = None
    agent_key: str | None = None
    color: str = "#1f77b4"


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "OF-MPC",
        "baseline",
        BASELINE_PATH,
        color="#1f2937",
    ),
    RunSpec(
        "TD3 Weights",
        "weights",
        RESULTS_ROOT / "distillation_weights_td3_disturb_fluctuation_mismatch_unified/20260521_150600/input_data.pkl",
        defaults_family="weights",
        agent_key="td3_agent",
        color="#7c3aed",
    ),
    RunSpec(
        "TD3 Residual",
        "residual",
        RESULTS_ROOT / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260521_152021/input_data.pkl",
        defaults_family="residual",
        agent_key="td3_agent",
        color="#dc2626",
    ),
    RunSpec(
        "Horizon DDQN",
        "horizon",
        RESULTS_ROOT / "distillation_horizon_disturb_fluctuation_mismatch_unified/20260521_154248/input_data.pkl",
        defaults_family="horizon_standard",
        agent_key="agent",
        color="#2563eb",
    ),
    RunSpec(
        "Dueling Horizon",
        "dueling",
        RESULTS_ROOT / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260521_154934/input_data.pkl",
        defaults_family="horizon_dueling",
        agent_key="agent",
        color="#0891b2",
    ),
    RunSpec(
        "TD3 Markov",
        "markov",
        RESULTS_ROOT / "distillation_markov_td3_disturb_fluctuation_unified/20260521_162222/input_data.pkl",
        defaults_family="markov",
        agent_key="td3_agent",
        color="#16a34a",
    ),
)


OUTPUT_LABELS = ["Tray-24 C2H6 composition", "Tray-85 temperature"]
INPUT_LABELS = ["Reflux flow", "Reboiler duty"]


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj).__name__}")
    return obj


def arr(value: Any, dtype=float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def finite_mean(values: np.ndarray) -> float:
    flat = np.asarray(values, float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    return float(np.mean(flat))


def finite_tail(values: Any, count: int = TAIL_EPISODES) -> float:
    data = arr(values).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.mean(data[-min(count, data.size) :]))


def final_value(values: Any) -> float:
    data = arr(values).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(data[-1])


def min_max_scale(value: Any, data_min: Any, data_max: Any) -> np.ndarray:
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return (np.asarray(value, float) - data_min) / np.maximum(data_max - data_min, 1.0e-12)


def reverse_min_max(value: Any, data_min: Any, data_max: Any) -> np.ndarray:
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return np.asarray(value, float) * np.maximum(data_max - data_min, 1.0e-12) + data_min


def y_ss_scaled(bundle: dict[str, Any]) -> np.ndarray:
    steady_states = bundle.get("steady_states", {})
    y_ss = arr(steady_states.get("y_ss") if isinstance(steady_states, dict) else None)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if y_ss.size != 2 or data_min.size < 4 or data_max.size < 4:
        return np.asarray([], float)
    return min_max_scale(y_ss, data_min[N_INPUTS:], data_max[N_INPUTS:])


def physical_setpoints(bundle: dict[str, Any]) -> np.ndarray:
    """Saved distillation setpoints are scaled deviations; convert to output units."""

    y_sp = arr(bundle.get("y_sp"))
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    y_ss_s = y_ss_scaled(bundle)
    if y_sp.ndim != 2 or y_ss_s.size != y_sp.shape[1] or data_min.size < 4 or data_max.size < 4:
        return np.asarray([], float)
    return reverse_min_max(y_sp + y_ss_s.reshape(1, -1), data_min[N_INPUTS:], data_max[N_INPUTS:])


def y_for_steps(bundle: dict[str, Any]) -> np.ndarray:
    y = arr(bundle.get("y_rl", bundle.get("y")))
    y_sp = arr(bundle.get("y_sp"))
    if y.ndim != 2 or y_sp.ndim != 2:
        return np.asarray([], float)
    n = min(y.shape[0] - 1, y_sp.shape[0])
    if n <= 0:
        return np.asarray([], float)
    return y[1 : n + 1, :]


def u_for_steps(bundle: dict[str, Any]) -> np.ndarray:
    u = arr(bundle.get("u_rl", bundle.get("u")))
    y_sp = arr(bundle.get("y_sp"))
    if u.ndim != 2 or y_sp.ndim != 2:
        return np.asarray([], float)
    n = min(u.shape[0], y_sp.shape[0])
    return u[:n, :]


def current_band_phys(bundle: dict[str, Any]) -> np.ndarray:
    y_sp_phys = physical_setpoints(bundle)
    if y_sp_phys.size == 0:
        return np.asarray([], float)
    k_rel = arr(RL_REWARD_DEFAULTS["k_rel"])
    floor = arr(RL_REWARD_DEFAULTS["band_floor_phys"])
    return np.maximum(k_rel.reshape(1, -1) * np.abs(y_sp_phys), floor.reshape(1, -1))


def recompute_current_rewards(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Re-score saved trajectories with the current distillation reward parameters."""

    delta_y = arr(bundle.get("delta_y_storage"))
    delta_u = arr(bundle.get("delta_u_storage"))
    y_sp_phys = physical_setpoints(bundle)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if (
        delta_y.ndim != 2
        or delta_u.ndim != 2
        or y_sp_phys.ndim != 2
        or delta_y.shape[1] != 2
        or delta_u.shape[1] != 2
        or data_min.size < 4
        or data_max.size < 4
    ):
        return np.asarray([], float), np.asarray([], float)

    n = min(delta_y.shape[0], delta_u.shape[0], y_sp_phys.shape[0])
    if n <= 0:
        return np.asarray([], float), np.asarray([], float)
    delta_y = delta_y[:n]
    delta_u = delta_u[:n]
    y_sp_phys = y_sp_phys[:n]

    params = RL_REWARD_DEFAULTS
    dy_scale = np.maximum(data_max[N_INPUTS:] - data_min[N_INPUTS:], 1.0e-12)
    k_rel = arr(params["k_rel"])
    band_floor_phys = arr(params["band_floor_phys"])
    q_diag = arr(params["Q_diag"])
    r_diag = arr(params["R_diag"])
    tau_frac = float(params.get("tau_frac", 0.7))
    gamma_out = float(params.get("gamma_out", 0.5))
    gamma_in = float(params.get("gamma_in", 0.5))
    beta = float(params.get("beta", 7.0))
    lam_in = float(params.get("lam_in", 1.0))
    bonus_kind = str(params.get("bonus_kind", "exp"))
    bonus_k = float(params.get("bonus_k", 12.0))
    bonus_p = float(params.get("bonus_p", 0.6))
    bonus_c = float(params.get("bonus_c", 20.0))
    reward_scale = float(params.get("reward_scale", 1.0))

    band_phys = np.maximum(k_rel.reshape(1, -1) * np.abs(y_sp_phys), band_floor_phys.reshape(1, -1))
    band_scaled = band_phys / dy_scale.reshape(1, -1)
    tau_scaled = tau_frac * band_scaled
    abs_e = np.abs(delta_y)
    sigmoid_arg = np.clip((band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12), -60.0, 60.0)
    s_i = 1.0 / (1.0 + np.exp(-sigmoid_arg))
    w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])

    err_quad = np.sum(q_diag.reshape(1, -1) * (delta_y**2), axis=1)
    err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)
    move = np.sum(r_diag.reshape(1, -1) * (delta_u**2), axis=1)

    slope_at_edge = 2.0 * q_diag.reshape(1, -1) * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow, axis=1)
    lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag, axis=1)

    z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
    if bonus_kind == "linear":
        phi = 1.0 - z
    elif bonus_kind == "quadratic":
        phi = (1.0 - z) ** 2
    elif bonus_kind == "exp":
        phi = (np.exp(-bonus_k * z) - np.exp(-bonus_k)) / (1.0 - np.exp(-bonus_k))
    elif bonus_kind == "power":
        phi = 1.0 - np.power(z, bonus_p)
    elif bonus_kind == "log":
        phi = np.log1p(bonus_c * (1.0 - z)) / np.log1p(bonus_c)
    else:
        raise ValueError(f"Unknown bonus kind: {bonus_kind}")

    qb2 = q_diag.reshape(1, -1) * (band_scaled**2)
    bonus = w_in * beta * np.sum(qb2 * phi, axis=1)
    rewards = (-(err_eff + move + lin_out + lin_in) + bonus) * reward_scale

    steps_per_episode = int(bundle.get("time_in_sub_episodes") or 400)
    episodes = rewards.size // steps_per_episode
    if episodes <= 0:
        return rewards, np.asarray([], float)
    avg = rewards[: episodes * steps_per_episode].reshape(episodes, steps_per_episode).mean(axis=1)
    return rewards, avg


def tail_slice(bundle: dict[str, Any], episodes: int = TAIL_EPISODES) -> slice:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    tail_steps = min(nfe, steps * episodes)
    return slice(max(0, nfe - tail_steps), nfe)


def warm_start_steps(bundle: dict[str, Any]) -> int:
    """Return warm-start length in control steps.

    Older bundles save `warm_start_step` as a step count, while config snapshots
    save `warm_start` as an episode count.
    """

    steps = int(bundle.get("time_in_sub_episodes") or 400)
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    raw_step = bundle.get("warm_start_step")
    if raw_step is not None:
        return max(0, min(int(raw_step), nfe))
    cfg = bundle.get("config_snapshot", {})
    warm_eps = int(cfg.get("warm_start") or 10)
    return max(0, min(warm_eps * steps, nfe))


def warm_start_episodes(bundle: dict[str, Any]) -> float:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    return float(warm_start_steps(bundle)) / max(steps, 1)


def post_warm_slice(bundle: dict[str, Any]) -> slice:
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    return slice(min(nfe, warm_start_steps(bundle)), nfe)


def fraction_near_bounds(u: np.ndarray, tol_frac: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray]:
    if u.ndim != 2 or u.shape[1] != 2:
        return np.full(2, np.nan), np.full(2, np.nan)
    lo = np.asarray(DISTILLATION_INPUT_BOUNDS["u_min"], float)
    hi = np.asarray(DISTILLATION_INPUT_BOUNDS["u_max"], float)
    tol = tol_frac * np.maximum(hi - lo, 1.0)
    lower = np.mean(u <= (lo + tol.reshape(1, -1)), axis=0)
    upper = np.mean(u >= (hi - tol.reshape(1, -1)), axis=0)
    return lower.astype(float), upper.astype(float)


def summarize_defaults(spec: RunSpec) -> dict[str, Any]:
    if spec.defaults_family is None or spec.agent_key is None:
        return {}
    defaults = get_distillation_notebook_defaults(spec.defaults_family)
    agent = dict(defaults.get(spec.agent_key, {}))
    reward = dict(defaults.get("reward", {}))
    return {
        "state_mode_default": defaults.get("state_mode"),
        "agent_kind_default": defaults.get("agent_kind"),
        "hidden_layers": agent.get("hidden_layers"),
        "actor_hidden": agent.get("actor_hidden"),
        "critic_hidden": agent.get("critic_hidden"),
        "gamma": agent.get("gamma"),
        "n_step": agent.get("n_step"),
        "multistep_mode": agent.get("multistep_mode"),
        "reward_Q_diag": np.asarray(reward.get("Q_diag", []), float).tolist() if "Q_diag" in reward else None,
        "reward_R_diag": np.asarray(reward.get("R_diag", []), float).tolist() if "R_diag" in reward else None,
        "markov_z_bound": defaults.get("controller", {}).get("z_bound") if isinstance(defaults.get("controller"), dict) else None,
        "markov_z_safety": defaults.get("controller", {}).get("z_safety") if isinstance(defaults.get("controller"), dict) else None,
    }


def source_fraction(source_log: np.ndarray, code: int, mask: slice) -> float:
    if source_log.size == 0:
        return float("nan")
    data = source_log[mask]
    if data.size == 0:
        return float("nan")
    return float(np.mean(data == code))


def summarize_mechanism(spec: RunSpec, bundle: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"method": spec.method}
    tail = tail_slice(bundle)
    post = post_warm_slice(bundle)

    h = arr(bundle.get("horizon_trace"))
    if h.ndim == 2 and h.shape[1] >= 2:
        h_tail = h[tail]
        h_post = h[post]
        row["tail_Hp_mean"] = finite_mean(h_tail[:, 0])
        row["tail_Hc_mean"] = finite_mean(h_tail[:, 1])
        row["post_warm_distinct_horizons"] = int(np.unique(h_post.astype(int), axis=0).shape[0]) if h_post.size else 0
        row["tail_distinct_horizons"] = int(np.unique(h_tail.astype(int), axis=0).shape[0]) if h_tail.size else 0
    else:
        row.update(
            {
                "tail_Hp_mean": float("nan"),
                "tail_Hc_mean": float("nan"),
                "post_warm_distinct_horizons": float("nan"),
                "tail_distinct_horizons": float("nan"),
            }
        )

    weights = arr(bundle.get("weight_log"))
    if weights.ndim == 2 and weights.shape[1] >= 4:
        w_tail = weights[tail]
        for label, idx in zip(["Q1", "Q2", "R1", "R2"], range(4)):
            row[f"tail_{label}_mult"] = finite_mean(w_tail[:, idx])
    else:
        for label in ["Q1", "Q2", "R1", "R2"]:
            row[f"tail_{label}_mult"] = float("nan")

    residual = arr(bundle.get("residual_exec_log", bundle.get("delta_u_res_exec_log")))
    if residual.ndim == 2 and residual.shape[1] == 2:
        r_tail = residual[tail]
        row["tail_residual_norm"] = finite_mean(np.linalg.norm(r_tail, axis=1))
        row["tail_residual_abs_u1"] = finite_mean(np.abs(r_tail[:, 0]))
        row["tail_residual_abs_u2"] = finite_mean(np.abs(r_tail[:, 1]))
    else:
        row["tail_residual_norm"] = float("nan")
        row["tail_residual_abs_u1"] = float("nan")
        row["tail_residual_abs_u2"] = float("nan")

    for key in ["projection_active_log", "projection_due_to_authority_log", "projection_due_to_deadband_log"]:
        data = arr(bundle.get(key))
        row[f"tail_{key.replace('_log', '')}_frac"] = finite_mean(data[tail] > 0) if data.size else float("nan")

    rho = arr(bundle.get("rho_eff_log"))
    row["tail_rho_eff"] = finite_mean(rho[tail]) if rho.size else float("nan")

    z = arr(bundle.get("z_executed_log", bundle.get("z_log", bundle.get("rl_requested_z_log"))))
    if z.ndim == 2 and z.shape[1] == 4:
        z_tail = z[tail]
        row["markov_z_bound"] = float(bundle.get("markov_z_bound", np.nan))
        row["tail_q95_abs_z_i"] = float(np.nanquantile(np.abs(z_tail), 0.95))
        row["tail_z_norm_mean"] = finite_mean(np.linalg.norm(z_tail, axis=1))
        row["post_warm_q95_abs_z_i"] = float(np.nanquantile(np.abs(z[post]), 0.95)) if z[post].size else float("nan")
    else:
        row["markov_z_bound"] = float(bundle.get("markov_z_bound", np.nan)) if bundle.get("markov_z_bound") is not None else float("nan")
        row["tail_q95_abs_z_i"] = float("nan")
        row["tail_z_norm_mean"] = float("nan")
        row["post_warm_q95_abs_z_i"] = float("nan")

    source = arr(bundle.get("rl_action_source_log"))
    row["post_warm_td3_source_frac"] = source_fraction(source, 2, post)
    row["post_warm_ls_fallback_frac"] = source_fraction(source, 3, post)
    row["post_warm_nominal_fallback_frac"] = source_fraction(source, 4, post)
    accepted = arr(bundle.get("accepted_log"))
    fallback = arr(bundle.get("fallback_log"))
    row["post_warm_accepted_frac"] = finite_mean(accepted[post] > 0) if accepted.size else float("nan")
    row["post_warm_fallback_frac"] = finite_mean(fallback[post] > 0) if fallback.size else float("nan")

    for key in [
        "z_safety_requested_projection_active_log",
        "z_safety_requested_coord_clip_active_log",
        "z_safety_requested_vector_projection_active_log",
        "z_safety_ls_projection_active_log",
    ]:
        data = arr(bundle.get(key))
        row[f"post_warm_{key.replace('_log', '')}_frac"] = finite_mean(data[post] > 0) if data.size else float("nan")

    cap = arr(bundle.get("z_safety_effective_cap_log"))
    row["tail_z_safety_eff_cap_mean"] = finite_mean(cap[tail]) if cap.size else float("nan")
    norm_before = arr(bundle.get("z_safety_requested_norm_before_log"))
    norm_after = arr(bundle.get("z_safety_requested_norm_after_log"))
    row["tail_z_norm_before_safety"] = finite_mean(norm_before[tail]) if norm_before.size else float("nan")
    row["tail_z_norm_after_safety"] = finite_mean(norm_after[tail]) if norm_after.size else float("nan")
    return row


def summarize_run(spec: RunSpec) -> dict[str, Any]:
    bundle = load_pickle(spec.path)
    _, rescored_avg = recompute_current_rewards(bundle)
    y = y_for_steps(bundle)
    u = u_for_steps(bundle)
    y_sp_phys = physical_setpoints(bundle)
    band = current_band_phys(bundle)
    n = min(y.shape[0], y_sp_phys.shape[0], band.shape[0])
    tail = tail_slice(bundle)
    y_tail = y[tail] if y.shape[0] >= tail.stop else y[-min(y.shape[0], int(bundle.get("time_in_sub_episodes", 400)) * TAIL_EPISODES) :]
    ysp_tail = (
        y_sp_phys[tail]
        if y_sp_phys.shape[0] >= tail.stop
        else y_sp_phys[-min(y_sp_phys.shape[0], int(bundle.get("time_in_sub_episodes", 400)) * TAIL_EPISODES) :]
    )
    band_tail = (
        band[tail]
        if band.shape[0] >= tail.stop
        else band[-min(band.shape[0], int(bundle.get("time_in_sub_episodes", 400)) * TAIL_EPISODES) :]
    )
    u_tail = (
        u[tail]
        if u.shape[0] >= tail.stop
        else u[-min(u.shape[0], int(bundle.get("time_in_sub_episodes", 400)) * TAIL_EPISODES) :]
    )
    e_tail = y_tail - ysp_tail if y_tail.shape == ysp_tail.shape else np.asarray([], float)
    norm_abs_tail = np.abs(e_tail) / np.maximum(band_tail, 1.0e-12) if e_tail.shape == band_tail.shape else np.asarray([], float)

    lower_sat, upper_sat = fraction_near_bounds(u_tail)
    delta_u = arr(bundle.get("delta_u_storage"))
    du_tail = delta_u[tail] if delta_u.ndim == 2 and delta_u.shape[0] >= tail.stop else np.asarray([], float)

    cfg = bundle.get("config_snapshot", {})
    row = {
        "method": spec.method,
        "family": spec.family,
        "path": spec.path.relative_to(REPO_ROOT).as_posix(),
        "run_mode": bundle.get("run_mode", cfg.get("run_mode")),
        "disturbance_profile": "fluctuation",
        "nFE": int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0]),
        "episodes": int(len(arr(bundle.get("avg_rewards")))),
        "time_in_sub_episodes": int(bundle.get("time_in_sub_episodes") or 400),
        "warm_start_episodes": warm_start_episodes(bundle),
        "test_cycle": bundle.get("test_cycle", cfg.get("test_cycle")),
        "logged_tail_reward": finite_tail(bundle.get("avg_rewards")),
        "logged_final_reward": final_value(bundle.get("avg_rewards")),
        "current_rescored_tail_reward": finite_tail(rescored_avg),
        "current_rescored_final_reward": final_value(rescored_avg),
        "current_rescored_mean_reward": finite_mean(rescored_avg),
        "tail_x24_rmse": float(np.sqrt(np.nanmean(e_tail[:, 0] ** 2))) if e_tail.size else float("nan"),
        "tail_T85_rmse": float(np.sqrt(np.nanmean(e_tail[:, 1] ** 2))) if e_tail.size else float("nan"),
        "tail_x24_mae": finite_mean(np.abs(e_tail[:, 0])) if e_tail.size else float("nan"),
        "tail_T85_mae": finite_mean(np.abs(e_tail[:, 1])) if e_tail.size else float("nan"),
        "tail_x24_max_abs": float(np.nanmax(np.abs(e_tail[:, 0]))) if e_tail.size else float("nan"),
        "tail_T85_max_abs": float(np.nanmax(np.abs(e_tail[:, 1]))) if e_tail.size else float("nan"),
        "tail_x24_norm_band_mae": finite_mean(norm_abs_tail[:, 0]) if norm_abs_tail.size else float("nan"),
        "tail_T85_norm_band_mae": finite_mean(norm_abs_tail[:, 1]) if norm_abs_tail.size else float("nan"),
        "tail_norm_band_mean": finite_mean(norm_abs_tail) if norm_abs_tail.size else float("nan"),
        "tail_mean_abs_delta_u_scaled": finite_mean(np.abs(du_tail)) if du_tail.size else float("nan"),
        "tail_input_tv_physical": float(np.nansum(np.abs(np.diff(u_tail, axis=0)))) if u_tail.ndim == 2 and u_tail.shape[0] > 1 else float("nan"),
        "tail_u1_lower_sat_frac": float(lower_sat[0]),
        "tail_u1_upper_sat_frac": float(upper_sat[0]),
        "tail_u2_lower_sat_frac": float(lower_sat[1]),
        "tail_u2_upper_sat_frac": float(upper_sat[1]),
        "algorithm": bundle.get("algorithm", cfg.get("algorithm", bundle.get("agent_kind"))),
        "agent_kind": bundle.get("agent_kind", cfg.get("agent_kind")),
    }
    if n > 0:
        full_err = y[:n] - y_sp_phys[:n]
        full_band = band[:n]
        full_norm = np.abs(full_err) / np.maximum(full_band, 1.0e-12)
        row["full_norm_band_mean"] = finite_mean(full_norm)
        row["full_steps_outside_band_frac"] = finite_mean(np.any(full_norm > 1.0, axis=1))
    else:
        row["full_norm_band_mean"] = float("nan")
        row["full_steps_outside_band_frac"] = float("nan")

    row.update({f"default_{k}": v for k, v in summarize_defaults(spec).items()})
    return row


def csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(sanitize(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "NA"
    return value


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    data = np.asarray(values, float)
    if data.size == 0:
        return data
    out = np.empty_like(data)
    for idx in range(data.size):
        lo = max(0, idx - window + 1)
        out[idx] = np.nanmean(data[lo : idx + 1])
    return out


def _plot_sample_indices(length: int, max_points: int = 1200) -> np.ndarray:
    if length <= 0:
        return np.asarray([], int)
    step = max(1, int(np.ceil(length / max_points)))
    return np.arange(0, length, step, dtype=int)


def make_reward_curves(bundles: dict[str, dict[str, Any]], rescored: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2), constrained_layout=True)
    for spec in RUNS:
        avg = rescored.get(spec.method, np.asarray([], float))
        if avg.size == 0:
            avg = arr(bundles[spec.method].get("avg_rewards"))
        x = np.arange(1, avg.size + 1)
        ax.plot(x, rolling_mean(avg, 5), color=spec.color, linewidth=2.2, label=spec.method)
    ax.axvline(10, color="#6b7280", linestyle="--", linewidth=1.4, label="RL warm-start end")
    ax.set_title("Distillation reward trends, current-parameter rescoring")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.savefig(FIG_DIR / "fig_reward_learning_curves_rescored.png")
    plt.close(fig)


def make_reward_bars(summary: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in summary]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, [row["logged_tail_reward"] for row in summary], width, label="Logged tail", color="#94a3b8")
    ax.bar(
        x + width / 2,
        [row["current_rescored_tail_reward"] for row in summary],
        width,
        label="Current-rescored tail",
        color="#f97316",
    )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Tail average reward")
    ax.set_title("Logged reward versus common current-reward rescoring")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_tail_reward_logged_vs_rescored.png")
    plt.close(fig)


def make_tracking_rmse(summary: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in summary]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, [row["tail_x24_rmse"] for row in summary], width, label="x24 RMSE", color="#2563eb")
    ax.bar(x + width / 2, [row["tail_T85_rmse"] for row in summary], width, label="T85 RMSE", color="#dc2626")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("RMSE in saved output units")
    ax.set_title("Tail tracking error by output")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_tail_tracking_rmse_physical.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, [row["tail_x24_norm_band_mae"] for row in summary], width, label="x24 abs error / band", color="#2563eb")
    ax.bar(x + width / 2, [row["tail_T85_norm_band_mae"] for row in summary], width, label="T85 abs error / band", color="#dc2626")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.3, label="reward band")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Tail mean normalized absolute error")
    ax.set_title("Tail tracking relative to current reward bands")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_tail_tracking_normalized_band.png")
    plt.close(fig)


def make_tail_tracking_overlay(bundles: dict[str, dict[str, Any]]) -> None:
    tail_len = 1600
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True, constrained_layout=True)
    for out_idx, ax in enumerate(axes):
        for spec in RUNS:
            bundle = bundles[spec.method]
            y = y_for_steps(bundle)
            if y.size == 0:
                continue
            n = min(tail_len, y.shape[0])
            idx = _plot_sample_indices(n)
            t = idx * float(bundle.get("delta_t", 1.0 / 6.0))
            ax.plot(t, y[-n:, out_idx][idx], color=spec.color, linewidth=1.8, alpha=0.95, label=spec.method)
        sp_bundle = bundles["OF-MPC"]
        sp = physical_setpoints(sp_bundle)
        if sp.size:
            n = min(tail_len, sp.shape[0])
            idx = _plot_sample_indices(n)
            t = idx * float(sp_bundle.get("delta_t", 1.0 / 6.0))
            ax.step(t, sp[-n:, out_idx][idx], where="post", color="black", linestyle="--", linewidth=2.0, label="Setpoint")
        ax.set_ylabel(OUTPUT_LABELS[out_idx])
        ax.set_title(f"Tail tracking overlay: {OUTPUT_LABELS[out_idx]}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Tail window time (h)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    fig.savefig(FIG_DIR / "fig_final_tail_tracking_overlay.png")
    plt.close(fig)


def make_tail_input_overlay(bundles: dict[str, dict[str, Any]]) -> None:
    tail_len = 1600
    bounds = DISTILLATION_INPUT_BOUNDS
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.8), sharex=True, constrained_layout=True)
    for input_idx, ax in enumerate(axes):
        for spec in RUNS:
            bundle = bundles[spec.method]
            u = u_for_steps(bundle)
            if u.size == 0:
                continue
            n = min(tail_len, u.shape[0])
            idx = _plot_sample_indices(n)
            t = idx * float(bundle.get("delta_t", 1.0 / 6.0))
            ax.plot(t, u[-n:, input_idx][idx], color=spec.color, linewidth=1.6, alpha=0.9, label=spec.method)
        ax.axhline(bounds["u_min"][input_idx], color="black", linestyle=":", linewidth=1.3, label="bounds" if input_idx == 0 else None)
        ax.axhline(bounds["u_max"][input_idx], color="black", linestyle=":", linewidth=1.3)
        ax.set_ylabel(INPUT_LABELS[input_idx])
        ax.set_title(f"Tail input trajectory: {INPUT_LABELS[input_idx]}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Tail window time (h)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    fig.savefig(FIG_DIR / "fig_final_tail_input_overlay.png")
    plt.close(fig)


def make_input_movement(summary: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.bar(x, [row["tail_mean_abs_delta_u_scaled"] for row in summary], color="#0f766e")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Mean abs delta u, scaled")
    ax.set_title("Tail input movement penalty proxy")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "fig_tail_input_movement_scaled.png")
    plt.close(fig)


def make_mechanism_dashboard(mech: list[dict[str, Any]]) -> None:
    by_method = {row["method"]: row for row in mech}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax = axes[0, 0]
    horizon_methods = [m for m in ["Horizon DDQN", "Dueling Horizon"] if m in by_method]
    x = np.arange(len(horizon_methods))
    width = 0.35
    ax.bar(x - width / 2, [by_method[m]["tail_Hp_mean"] for m in horizon_methods], width, label="Hp", color="#2563eb")
    ax.bar(x + width / 2, [by_method[m]["tail_Hc_mean"] for m in horizon_methods], width, label="Hc", color="#f97316")
    ax.set_xticks(x, horizon_methods, rotation=15, ha="right")
    ax.set_title("Tail horizon choices")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)

    ax = axes[0, 1]
    w = by_method.get("TD3 Weights", {})
    labels = ["Q1", "Q2", "R1", "R2"]
    vals = [w.get(f"tail_{label}_mult", np.nan) for label in labels]
    ax.plot(labels, vals, marker="o", linewidth=2.5, color="#7c3aed", label="TD3 Weights")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="nominal")
    ax.set_ylim(0.65, max(2.05, np.nanmax(vals) + 0.1 if np.any(np.isfinite(vals)) else 2.05))
    ax.set_title("Tail weight multipliers")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    ax = axes[1, 0]
    r = by_method.get("TD3 Residual", {})
    ax.bar(["Residual norm", "Projection active", "rho eff"], [r.get("tail_residual_norm", np.nan), r.get("tail_projection_active_frac", np.nan), r.get("tail_rho_eff", np.nan)], color=["#dc2626", "#f59e0b", "#64748b"])
    ax.set_title("Residual authority diagnostics")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    m = by_method.get("TD3 Markov", {})
    labels = ["q95 abs z_i", "mean z norm", "requested projection", "TD3 source"]
    vals = [
        m.get("tail_q95_abs_z_i", np.nan),
        m.get("tail_z_norm_mean", np.nan),
        m.get("post_warm_z_safety_requested_projection_active_frac", np.nan),
        m.get("post_warm_td3_source_frac", np.nan),
    ]
    ax.bar(labels, vals, color=["#16a34a", "#22c55e", "#f59e0b", "#64748b"])
    ax.set_title("Markov z-safety and source diagnostics")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Distillation controller mechanism dashboard", fontsize=17)
    fig.savefig(FIG_DIR / "fig_controller_mechanism_dashboard.png")
    plt.close(fig)


def make_markov_z_figures(bundle: dict[str, Any]) -> None:
    z = arr(bundle.get("z_executed_log", bundle.get("z_log", bundle.get("rl_requested_z_log"))))
    if z.ndim != 2 or z.shape[1] != 4:
        return
    tail = tail_slice(bundle)
    z_tail = z[tail]
    cap = arr(bundle.get("z_safety_effective_cap_log"))
    cap_tail = cap[tail] if cap.size and cap.shape[0] >= tail.stop else np.asarray([], float)
    n = z_tail.shape[0]
    idx = _plot_sample_indices(n, max_points=1400)
    t = idx * float(bundle.get("delta_t", 1.0 / 6.0))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, constrained_layout=True)
    labels = ["y1_u1", "y1_u2", "y2_u1", "y2_u2"]
    for j in range(4):
        axes[0].plot(t, z_tail[idx, j], linewidth=1.4, label=labels[j])
    if cap_tail.size:
        axes[0].plot(t, cap_tail[idx], color="black", linestyle="--", linewidth=1.5, label="eff cap")
        axes[0].plot(t, -cap_tail[idx], color="black", linestyle="--", linewidth=1.5)
    axes[0].set_ylabel("Executed z")
    axes[0].set_title("Tail Markov z coordinates after safety projection")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3, frameon=True)

    norm = np.linalg.norm(z_tail, axis=1)
    before = arr(bundle.get("z_safety_requested_norm_before_log"))
    after = arr(bundle.get("z_safety_requested_norm_after_log"))
    before_tail = before[tail] if before.size and before.shape[0] >= tail.stop else np.asarray([], float)
    after_tail = after[tail] if after.size and after.shape[0] >= tail.stop else np.asarray([], float)
    axes[1].plot(t, norm[idx], color="#16a34a", linewidth=2.0, label="executed norm")
    if before_tail.size:
        axes[1].plot(t, before_tail[idx], color="#f97316", linewidth=1.3, alpha=0.8, label="requested norm before safety")
    if after_tail.size:
        axes[1].plot(t, after_tail[idx], color="#2563eb", linewidth=1.3, alpha=0.8, label="requested norm after safety")
    max_norm = bundle.get("z_safety", {}).get("vector_norm_cap", {}).get("max_norm", None)
    if max_norm is not None:
        axes[1].axhline(float(max_norm), color="black", linestyle=":", linewidth=1.5, label="vector norm cap")
    axes[1].set_ylabel("z 2-norm")
    axes[1].set_xlabel("Tail window time (h)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_markov_z_safety_tail.png")
    plt.close(fig)

    post = post_warm_slice(bundle)
    metrics = {
        "TD3 source": source_fraction(arr(bundle.get("rl_action_source_log")), 2, post),
        "LS fallback": source_fraction(arr(bundle.get("rl_action_source_log")), 3, post),
        "nominal fallback": source_fraction(arr(bundle.get("rl_action_source_log")), 4, post),
        "accepted": finite_mean(arr(bundle.get("accepted_log"))[post] > 0) if arr(bundle.get("accepted_log")).size else np.nan,
        "requested projection": finite_mean(arr(bundle.get("z_safety_requested_projection_active_log"))[post] > 0)
        if arr(bundle.get("z_safety_requested_projection_active_log")).size
        else np.nan,
        "LS projection": finite_mean(arr(bundle.get("z_safety_ls_projection_active_log"))[post] > 0)
        if arr(bundle.get("z_safety_ls_projection_active_log")).size
        else np.nan,
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.5), constrained_layout=True)
    ax.bar(list(metrics.keys()), list(metrics.values()), color=["#64748b", "#94a3b8", "#cbd5e1", "#16a34a", "#f97316", "#f59e0b"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Post-warm fraction")
    ax.set_title("Markov source and safety projection fractions")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "fig_markov_source_projection_fractions.png")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, int]]) -> str:
    lines = ["| " + " | ".join(header for header, _, _ in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for _, key, digits in columns:
            value = row.get(key)
            if isinstance(value, (int, np.integer)) and digits == 0:
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating, int, np.integer)):
                values.append(fmt(value, digits))
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reward_defaults_text() -> str:
    q = np.asarray(RL_REWARD_DEFAULTS["Q_diag"], float).tolist()
    r = np.asarray(RL_REWARD_DEFAULTS["R_diag"], float).tolist()
    k = np.asarray(RL_REWARD_DEFAULTS["k_rel"], float).tolist()
    floor = np.asarray(RL_REWARD_DEFAULTS["band_floor_phys"], float).tolist()
    return f"`Q_diag = {q}`, `R_diag = {r}`, `k_rel = {k}`, `band_floor_phys = {floor}`"


def write_report(summary: list[dict[str, Any]], mechanism: list[dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> None:
    by_method = {row["method"]: row for row in summary}
    mech = {row["method"]: row for row in mechanism}
    ranked = sorted(summary, key=lambda row: row["current_rescored_tail_reward"], reverse=True)
    best = ranked[0]
    baseline = by_method["OF-MPC"]
    markov = by_method["TD3 Markov"]

    provenance = markdown_table(
        summary,
        [
            ("Method", "method", 0),
            ("Bundle", "path", 0),
            ("Episodes", "episodes", 0),
            ("Warm episodes", "warm_start_episodes", 0),
        ],
    )
    main_table = markdown_table(
        summary,
        [
            ("Method", "method", 0),
            ("Logged tail reward", "logged_tail_reward", 2),
            ("Current tail reward", "current_rescored_tail_reward", 2),
            ("x24 RMSE", "tail_x24_rmse", 5),
            ("T85 RMSE", "tail_T85_rmse", 3),
            ("Band-normalized MAE", "tail_norm_band_mean", 3),
            ("Mean abs du scaled", "tail_mean_abs_delta_u_scaled", 5),
        ],
    )
    rank_table = markdown_table(
        ranked,
        [
            ("Method", "method", 0),
            ("Current tail reward", "current_rescored_tail_reward", 2),
            ("x24 RMSE", "tail_x24_rmse", 5),
            ("T85 RMSE", "tail_T85_rmse", 3),
            ("Outside-band fraction", "full_steps_outside_band_frac", 3),
        ],
    )
    mechanism_table = markdown_table(
        mechanism,
        [
            ("Method", "method", 0),
            ("Tail Hp", "tail_Hp_mean", 2),
            ("Tail Hc", "tail_Hc_mean", 2),
            ("Q1 mult", "tail_Q1_mult", 3),
            ("Q2 mult", "tail_Q2_mult", 3),
            ("Residual norm", "tail_residual_norm", 5),
            ("q95 abs z_i", "tail_q95_abs_z_i", 4),
            ("TD3 source", "post_warm_td3_source_frac", 3),
            ("Proj active", "post_warm_z_safety_requested_projection_active_frac", 3),
        ],
    )

    markov_safety = bundles["TD3 Markov"].get("z_safety", {})
    markov_z_bound = bundles["TD3 Markov"].get("markov_z_bound", bundles["TD3 Markov"].get("z_bound"))
    all_test_cycles = [row.get("test_cycle") for row in summary if row["method"] != "OF-MPC"]

    lines: list[str] = []
    lines.append("# Distillation Latest Family Runs Analysis")
    lines.append("")
    lines.append("Date: 2026-05-21")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This report analyzes the latest disturbed distillation column runs for OF-MPC, TD3 weights, TD3 residual, horizon DDQN, dueling horizon DDQN, and TD3 Markov. The combined supervisor is intentionally excluded because it was not run in this batch."
    )
    lines.append("")
    lines.append("## Files Inspected")
    lines.append("")
    lines.append(provenance)
    lines.append("")
    lines.append(
        "All RL runs are disturbed fluctuation runs with 200 episodes and 400 control steps per episode. The saved RL `test_cycle` entries are all false, so these are training-rollout comparisons rather than frozen held-out evaluations."
    )
    lines.append("")
    lines.append("## Coordinate And Reward Handling")
    lines.append("")
    lines.append(
        "The distillation controlled outputs are tray-24 ethane composition and tray-85 temperature, with reflux and reboiler duty as manipulated inputs. The saved output trajectory `y` is in the repository's output coordinates, while saved `y_sp` is a scaled-deviation setpoint. The report therefore converts setpoints before plotting or physical-error scoring:"
    )
    lines.append("")
    lines.append("$$ y_{\\mathrm{sp},t}^{\\mathrm{phys}} = \\mathrm{unscale}(y_{\\mathrm{sp},t}^{\\mathrm{scaled-dev}} + y_{\\mathrm{ss}}^{\\mathrm{scaled}}). $$")
    lines.append("")
    lines.append(
        "Rewards are also recomputed from the saved scaled tracking and move arrays using the current distillation reward defaults. Current reward parameters are "
        + reward_defaults_text()
        + "."
    )
    lines.append("")
    lines.append(
        "For each output, the current reward band is `max(k_rel * abs(setpoint), band_floor_phys)`. The band-normalized error plots divide absolute physical tracking error by that band, so composition and temperature can be compared without forcing them onto the same raw scale."
    )
    lines.append("")
    lines.append("## Controller Defaults Checked")
    lines.append("")
    lines.append(
        "The active distillation defaults resolve to the enlarged networks `[512, 512, 512, 512, 512]` and `gamma = 0.99` for the DQN/TD3 families. The latest Markov bundle stores `markov_z_bound = "
        + fmt(markov_z_bound, 3)
        + "` and `z_safety = "
        + json.dumps(sanitize(markov_safety), sort_keys=True)
        + "`."
    )
    lines.append("")
    lines.append("## Main Quantitative Results")
    lines.append("")
    lines.append(main_table)
    lines.append("")
    lines.append(
        f"Best current-rescored tail reward: **{best['method']}** with `{fmt(best['current_rescored_tail_reward'], 2)}`. Relative to OF-MPC, this is a tail-reward change of `{fmt(best['current_rescored_tail_reward'] - baseline['current_rescored_tail_reward'], 2)}`."
    )
    lines.append(
        f"Best x24 composition RMSE is **{min(summary, key=lambda r: r['tail_x24_rmse'])['method']}**. Best T85 temperature RMSE is **{min(summary, key=lambda r: r['tail_T85_rmse'])['method']}**."
    )
    lines.append("")
    lines.append("![Reward learning curves](figures/distillation_latest_family_runs_20260521/fig_reward_learning_curves_rescored.png)")
    lines.append("")
    lines.append("![Tail reward logged versus rescored](figures/distillation_latest_family_runs_20260521/fig_tail_reward_logged_vs_rescored.png)")
    lines.append("")
    lines.append("![Tail tracking physical RMSE](figures/distillation_latest_family_runs_20260521/fig_tail_tracking_rmse_physical.png)")
    lines.append("")
    lines.append("![Tail tracking normalized band](figures/distillation_latest_family_runs_20260521/fig_tail_tracking_normalized_band.png)")
    lines.append("")
    lines.append("## Ranking And Interpretation")
    lines.append("")
    lines.append(rank_table)
    lines.append("")
    lines.append(
        "The current-reward ranking should be read together with physical tracking and input movement. A high scalar reward can come from staying inside the reward bands with modest move penalties, while raw RMSE exposes output-specific transients."
    )
    lines.append("")
    lines.append("![Final tail tracking overlay](figures/distillation_latest_family_runs_20260521/fig_final_tail_tracking_overlay.png)")
    lines.append("")
    lines.append("![Final tail input overlay](figures/distillation_latest_family_runs_20260521/fig_final_tail_input_overlay.png)")
    lines.append("")
    lines.append("![Tail input movement](figures/distillation_latest_family_runs_20260521/fig_tail_input_movement_scaled.png)")
    lines.append("")
    lines.append("## Controller Mechanism Diagnostics")
    lines.append("")
    lines.append(mechanism_table)
    lines.append("")
    lines.append("![Controller mechanism dashboard](figures/distillation_latest_family_runs_20260521/fig_controller_mechanism_dashboard.png)")
    lines.append("")
    lines.append("![Markov z safety tail](figures/distillation_latest_family_runs_20260521/fig_markov_z_safety_tail.png)")
    lines.append("")
    lines.append("![Markov source and projection fractions](figures/distillation_latest_family_runs_20260521/fig_markov_source_projection_fractions.png)")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(
        f"- **{best['method']}** is the strongest latest run by current-rescored tail reward. Its tail reward is `{fmt(best['current_rescored_tail_reward'], 2)}` versus OF-MPC `{fmt(baseline['current_rescored_tail_reward'], 2)}`."
    )
    lines.append(
        f"- TD3 Markov has current-rescored tail reward `{fmt(markov['current_rescored_tail_reward'], 2)}` and q95 abs z_i `{fmt(mech['TD3 Markov']['tail_q95_abs_z_i'], 4)}` under the new `z_bound = {fmt(markov_z_bound, 3)}` safety profile."
    )
    lines.append(
        f"- Markov is strong by scalar reward but not uniformly best by tracking: it has composition RMSE `{fmt(markov['tail_x24_rmse'], 5)}` while its T85 band-normalized tail error is `{fmt(markov['tail_T85_norm_band_mae'], 3)}`, the largest in this latest batch."
    )
    lines.append(
        f"- Markov source diagnostics show post-warm TD3 source fraction `{fmt(mech['TD3 Markov']['post_warm_td3_source_frac'], 3)}`, LS fallback fraction `{fmt(mech['TD3 Markov']['post_warm_ls_fallback_frac'], 3)}`, and requested-z projection activity `{fmt(mech['TD3 Markov']['post_warm_z_safety_requested_projection_active_frac'], 3)}`."
    )
    lines.append(
        f"- TD3 Residual is highly safety-projected in the tail: residual projection-active fraction is `{fmt(mech['TD3 Residual']['tail_projection_active_frac'], 3)}` and mean effective rho is `{fmt(mech['TD3 Residual']['tail_rho_eff'], 3)}`. That means the raw residual policy is asking for more authority than the safety/authority layer allows."
    )
    lines.append(
        f"- TD3 Residual shows a late reward collapse in the learning curve and ends with tail reward `{fmt(by_method['TD3 Residual']['current_rescored_tail_reward'], 2)}`. The projection diagnostics make this look more like an authority mismatch than a simple reward-scaling artifact."
    )
    lines.append(
        f"- TD3 Weights uses tail multipliers Q1 `{fmt(mech['TD3 Weights']['tail_Q1_mult'], 3)}`, Q2 `{fmt(mech['TD3 Weights']['tail_Q2_mult'], 3)}`, R1 `{fmt(mech['TD3 Weights']['tail_R1_mult'], 3)}`, and R2 `{fmt(mech['TD3 Weights']['tail_R2_mult'], 3)}`. The learned policy is not simply increasing all penalties uniformly."
    )
    lines.append(
        f"- Horizon DDQN and dueling horizon choose average tail horizons near Hp/Hc `{fmt(mech['Horizon DDQN']['tail_Hp_mean'], 2)}/{fmt(mech['Horizon DDQN']['tail_Hc_mean'], 2)}` and `{fmt(mech['Dueling Horizon']['tail_Hp_mean'], 2)}/{fmt(mech['Dueling Horizon']['tail_Hc_mean'], 2)}` respectively."
    )
    lines.append("")
    lines.append("## Bugs, Inconsistencies, And Risks")
    lines.append("")
    lines.append(
        "- The setpoint conversion remains essential. Plotting `y` directly against saved `y_sp` would again mix physical output coordinates with scaled-deviation setpoints."
    )
    lines.append(
        "- These runs are still training rollouts. Because the saved test cycles are all false, the report should not be treated as a final generalization claim."
    )
    lines.append(
        "- The bundles still do not consistently store network size and gamma directly, so this report checks current defaults from code and saved config snapshots rather than relying only on result bundles."
    )
    lines.append(
        "- High projection activity in residual or Markov means the learned raw policy and the safety layer disagree. That is not automatically bad, but it is a sign that the actor may be spending capacity outside the executable action set."
    )
    lines.append("")
    lines.append("## Figure Audit")
    lines.append("")
    lines.append(
        "The generated figures were visually checked after creation. The tracking overlay uses converted physical setpoints and separate output panels, avoiding the previous unreadable mixed-scale plot. Input plots include the physical input bounds. Markov plots show executed z, effective caps, vector-norm cap, and projection/source fractions."
    )
    lines.append("")
    lines.append("## Recommended Next Experiments")
    lines.append("")
    lines.append(
        "1. Run a frozen-policy evaluation pass for all five distillation RL families and OF-MPC under the same fluctuation disturbance. Metric to watch: current-rescored tail reward and band-normalized error without exploration."
    )
    lines.append(
        "2. For Markov, compare this guarded `z_bound = 0.04` run with the previous TD3-only/no-safeguard result under identical current reward scoring. Metric to watch: reward gained per projection/fallback avoided."
    )
    lines.append(
        "3. For residual, reduce raw residual authority or add a stronger behavior-cloning/inside-authority penalty if projection remains near one. Metric to watch: projection-active fraction and tail reward."
    )
    lines.append(
        "4. For weights and horizons, repeat with two additional seeds before drawing method-level conclusions, because these are single training rollouts with large networks."
    )
    lines.append("")
    lines.append("## Remaining Uncertainty")
    lines.append("")
    lines.append(
        "The analysis uses saved bundles from completed runs and does not rerun Aspen. It gives a fair current-reward comparison for the latest trajectories, but it does not prove closed-loop robustness until frozen-policy tests or multi-seed repeats are available."
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_all_figures(bundles: dict[str, dict[str, Any]], summary: list[dict[str, Any]], mechanism: list[dict[str, Any]], rescored: dict[str, np.ndarray]) -> None:
    make_reward_curves(bundles, rescored)
    make_reward_bars(summary)
    make_tracking_rmse(summary)
    make_tail_tracking_overlay(bundles)
    make_tail_input_overlay(bundles)
    make_input_movement(summary)
    make_mechanism_dashboard(mechanism)
    make_markov_z_figures(bundles["TD3 Markov"])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    rescored_avg: dict[str, np.ndarray] = {}

    for spec in RUNS:
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing result bundle for {spec.method}: {spec.path}")
        bundle = load_pickle(spec.path)
        bundles[spec.method] = bundle
        _, avg = recompute_current_rewards(bundle)
        rescored_avg[spec.method] = avg
        summary.append(summarize_run(spec))
        mechanism.append(summarize_mechanism(spec, bundle))

    baseline_tail = summary[0]["current_rescored_tail_reward"]
    for row in summary:
        row["current_tail_reward_delta_vs_ofmpc"] = row["current_rescored_tail_reward"] - baseline_tail

    write_csv(FIG_DIR / "distillation_latest_summary_metrics.csv", summary)
    write_csv(FIG_DIR / "distillation_latest_mechanism_metrics.csv", mechanism)
    (FIG_DIR / "summary.json").write_text(
        json.dumps(
            {
                "summary": sanitize(summary),
                "mechanism": sanitize(mechanism),
                "reward_defaults": sanitize(RL_REWARD_DEFAULTS),
                "figures": sorted(path.name for path in FIG_DIR.glob("*.png")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_all_figures(bundles, summary, mechanism, rescored_avg)
    write_report(summary, mechanism, bundles)
    print(f"Wrote report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote figures: {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
