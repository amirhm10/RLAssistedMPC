from __future__ import annotations

import csv
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.config import DISTILLATION_INPUT_BOUNDS, RL_REWARD_DEFAULTS as DIST_REWARD_DEFAULTS


FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_residual_markov_reward_focus_20260521"
REPORT_PATH = REPO_ROOT / "report" / "distillation_residual_markov_reward_focus_2026_05_21.md"
N_INPUTS = 2
TAIL_EPISODES = 10


DIST_RUNS = {
    "OF-MPC": REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    "TD3 Weights": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
    / "20260521_150600"
    / "input_data.pkl",
    "TD3 Residual": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
    / "20260521_152021"
    / "input_data.pkl",
    "Horizon DDQN": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_horizon_disturb_fluctuation_mismatch_unified"
    / "20260521_154248"
    / "input_data.pkl",
    "Dueling Horizon": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
    / "20260521_154934"
    / "input_data.pkl",
    "TD3 Markov": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_unified"
    / "20260521_162222"
    / "input_data.pkl",
}


POLY_RUNS = {
    "OF-MPC": REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
    "TD3 Residual": REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260520_214325" / "input_data.pkl",
}


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


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, found {type(obj).__name__}")
    return obj


def arr(value: Any, dtype=float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def finite_mean(value: Any) -> float:
    data = arr(value).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.mean(data))


def finite_tail(value: Any, count: int = TAIL_EPISODES) -> float:
    data = arr(value).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.mean(data[-min(count, data.size) :]))


def final_value(value: Any) -> float:
    data = arr(value).reshape(-1)
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


def dist_y_ss_scaled(bundle: dict[str, Any]) -> np.ndarray:
    steady = bundle.get("steady_states", {})
    y_ss = arr(steady.get("y_ss") if isinstance(steady, dict) else None)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if y_ss.size != 2 or data_min.size < 4 or data_max.size < 4:
        return np.asarray([], float)
    return min_max_scale(y_ss, data_min[N_INPUTS:], data_max[N_INPUTS:])


def dist_setpoint_phys(bundle: dict[str, Any]) -> np.ndarray:
    y_sp = arr(bundle.get("y_sp"))
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    y_ss_s = dist_y_ss_scaled(bundle)
    if y_sp.ndim != 2 or y_ss_s.size != 2 or data_min.size < 4 or data_max.size < 4:
        return np.asarray([], float)
    return reverse_min_max(y_sp + y_ss_s.reshape(1, -1), data_min[N_INPUTS:], data_max[N_INPUTS:])


def dist_y_steps(bundle: dict[str, Any]) -> np.ndarray:
    y = arr(bundle.get("y_rl", bundle.get("y")))
    y_sp = arr(bundle.get("y_sp"))
    if y.ndim != 2 or y_sp.ndim != 2:
        return np.asarray([], float)
    n = min(y.shape[0] - 1, y_sp.shape[0])
    return y[1 : n + 1, :] if n > 0 else np.asarray([], float)


def episode_view(values: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    data = np.asarray(values)
    if data.shape[0] < steps:
        return np.asarray([], dtype=data.dtype)
    n_ep = data.shape[0] // steps
    return data[: n_ep * steps].reshape(n_ep, steps, *data.shape[1:])


def tail_slice(bundle: dict[str, Any], episodes: int = TAIL_EPISODES) -> slice:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    tail_steps = min(nfe, steps * episodes)
    return slice(max(0, nfe - tail_steps), nfe)


def warm_steps(bundle: dict[str, Any]) -> int:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    if bundle.get("warm_start_step") is not None:
        return min(nfe, int(bundle["warm_start_step"]))
    return min(nfe, int(bundle.get("config_snapshot", {}).get("warm_start", 10)) * steps)


def post_warm_slice(bundle: dict[str, Any]) -> slice:
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    return slice(warm_steps(bundle), nfe)


def dist_band_phys(bundle: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    sp = dist_setpoint_phys(bundle)
    if sp.size == 0:
        return np.asarray([], float)
    k_rel = arr(params["k_rel"])
    floor = arr(params["band_floor_phys"])
    return np.maximum(k_rel.reshape(1, -1) * np.abs(sp), floor.reshape(1, -1))


def dist_reward_breakdown(bundle: dict[str, Any], params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return current-reward-style step rewards and component terms."""

    delta_y = arr(bundle.get("delta_y_storage"))
    delta_u = arr(bundle.get("delta_u_storage"))
    y_sp_phys = dist_setpoint_phys(bundle)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if (
        delta_y.ndim != 2
        or delta_u.ndim != 2
        or y_sp_phys.ndim != 2
        or data_min.size < 4
        or data_max.size < 4
    ):
        return {}
    n = min(delta_y.shape[0], delta_u.shape[0], y_sp_phys.shape[0])
    delta_y = delta_y[:n]
    delta_u = delta_u[:n]
    y_sp_phys = y_sp_phys[:n]

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

    err_quad_by_output = q_diag.reshape(1, -1) * (delta_y**2)
    err_quad = np.sum(err_quad_by_output, axis=1)
    err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)
    move_by_input = r_diag.reshape(1, -1) * (delta_u**2)
    move = np.sum(move_by_input, axis=1)

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
    bonus_by_output = w_in.reshape(-1, 1) * beta * qb2 * phi
    bonus = np.sum(bonus_by_output, axis=1)
    reward = (-(err_eff + move + lin_out + lin_in) + bonus) * reward_scale
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    episodes = reward.size // steps
    avg = reward[: episodes * steps].reshape(episodes, steps).mean(axis=1) if episodes else np.asarray([], float)
    return {
        "reward": reward,
        "avg_reward": avg,
        "err_eff": err_eff,
        "move": move,
        "lin_out": lin_out,
        "lin_in": lin_in,
        "bonus": bonus,
        "err_quad_by_output": err_quad_by_output,
        "move_by_input": move_by_input,
        "bonus_by_output": bonus_by_output,
        "band_phys": band_phys,
        "norm_abs_error": abs_e / np.maximum(band_scaled, 1.0e-12),
        "w_in": w_in,
    }


def reward_variants() -> dict[str, dict[str, Any]]:
    current = deepcopy(DIST_REWARD_DEFAULTS)
    q5000 = deepcopy(DIST_REWARD_DEFAULTS)
    q5000["Q_diag"] = np.array([3.7e4, 5.0e3], float)
    q10000 = deepcopy(DIST_REWARD_DEFAULTS)
    q10000["Q_diag"] = np.array([3.7e4, 1.0e4], float)
    wide = deepcopy(DIST_REWARD_DEFAULTS)
    wide["k_rel"] = np.array([0.3, 0.02], float)
    wide["band_floor_phys"] = np.array([0.003, 0.3], float)
    qt1000 = deepcopy(DIST_REWARD_DEFAULTS)
    qt1000["Q_diag"] = np.array([3.7e4, 1.0e3], float)
    return {
        "current_tight_QT1500": current,
        "tight_QT1000": qt1000,
        "tight_QT5000": q5000,
        "tight_QT10000": q10000,
        "wide_band_QT1500": wide,
    }


def tail_mean_series(series: np.ndarray, bundle: dict[str, Any]) -> float:
    if series.size == 0:
        return float("nan")
    slc = tail_slice(bundle)
    return finite_mean(series[slc])


def residual_summary(case: str, baseline: dict[str, Any], residual: dict[str, Any]) -> dict[str, Any]:
    slc = tail_slice(residual)
    base_tail = finite_tail(baseline.get("avg_rewards"))
    res_tail = finite_tail(residual.get("avg_rewards"))
    raw = arr(residual.get("residual_raw_log"))
    exe = arr(residual.get("residual_exec_log"))
    raw_tail = raw[slc] if raw.ndim == 2 and raw.shape[0] >= slc.stop else np.asarray([], float)
    exe_tail = exe[slc] if exe.ndim == 2 and exe.shape[0] >= slc.stop else np.asarray([], float)
    bound = float(np.nanmax(np.abs(raw))) if raw.size else float("nan")
    raw_norm = np.linalg.norm(raw_tail, axis=1) if raw_tail.size else np.asarray([], float)
    exe_norm = np.linalg.norm(exe_tail, axis=1) if exe_tail.size else np.asarray([], float)
    raw_saturation = float(np.mean(np.any(np.abs(raw_tail) >= 0.98 * bound, axis=1))) if raw_tail.size and np.isfinite(bound) else float("nan")
    ratio = finite_mean(exe_norm / np.maximum(raw_norm, 1.0e-12)) if raw_norm.size else float("nan")
    return {
        "case": case,
        "reward_basis": "native saved reward",
        "native_baseline_tail_reward": base_tail,
        "native_residual_tail_reward": res_tail,
        "native_tail_reward_delta_vs_baseline": res_tail - base_tail,
        "baseline_tail_reward": base_tail,
        "residual_tail_reward": res_tail,
        "tail_reward_delta_vs_baseline": res_tail - base_tail,
        "residual_final_reward": final_value(residual.get("avg_rewards")),
        "raw_tail_norm": finite_mean(raw_norm),
        "exec_tail_norm": finite_mean(exe_norm),
        "exec_to_raw_norm_ratio": ratio,
        "raw_tail_saturation_fraction": raw_saturation,
        "projection_active_tail_fraction": tail_mean_series(arr(residual.get("projection_active_log")) > 0, residual),
        "authority_projection_tail_fraction": tail_mean_series(arr(residual.get("projection_due_to_authority_log")) > 0, residual),
        "deadband_projection_tail_fraction": tail_mean_series(arr(residual.get("projection_due_to_deadband_log")) > 0, residual),
        "rho_eff_tail_mean": tail_mean_series(arr(residual.get("rho_eff_log")), residual),
        "raw_tail_mean_u1": finite_mean(raw_tail[:, 0]) if raw_tail.size else float("nan"),
        "raw_tail_mean_u2": finite_mean(raw_tail[:, 1]) if raw_tail.size else float("nan"),
        "exec_tail_mean_u1": finite_mean(exe_tail[:, 0]) if exe_tail.size else float("nan"),
        "exec_tail_mean_u2": finite_mean(exe_tail[:, 1]) if exe_tail.size else float("nan"),
    }


def apply_distillation_recomputed_residual_rewards(
    residual_rows: list[dict[str, Any]], reward_rows: list[dict[str, Any]]
) -> None:
    """Use one consistent current-reward basis for the distillation residual comparison.

    The saved OF-MPC baseline reward in the historical pickle is not directly
    comparable to the latest distillation reward function. The recomputed
    current-reward table is the fair basis for the residual-vs-baseline claim.
    """

    current = [r for r in reward_rows if r["variant"] == "current_tight_QT1500"]
    by_method = {r["method"]: r for r in current}
    if "OF-MPC" not in by_method or "TD3 Residual" not in by_method:
        return
    for row in residual_rows:
        if row.get("case") != "Distillation":
            continue
        base_tail = float(by_method["OF-MPC"]["tail_reward"])
        res_tail = float(by_method["TD3 Residual"]["tail_reward"])
        row["reward_basis"] = "recomputed current distillation reward"
        row["baseline_tail_reward"] = base_tail
        row["residual_tail_reward"] = res_tail
        row["tail_reward_delta_vs_baseline"] = res_tail - base_tail


def dist_tracking_summary(bundle: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
    y = dist_y_steps(bundle)
    sp = dist_setpoint_phys(bundle)
    band = dist_band_phys(bundle, params)
    n = min(y.shape[0], sp.shape[0], band.shape[0])
    if n <= 0:
        return {}
    slc = tail_slice(bundle)
    e = y[:n] - sp[:n]
    e_tail = e[slc]
    norm = np.abs(e[:n]) / np.maximum(band[:n], 1.0e-12)
    norm_tail = norm[slc]
    return {
        "tail_x24_rmse": float(np.sqrt(np.mean(e_tail[:, 0] ** 2))),
        "tail_T85_rmse": float(np.sqrt(np.mean(e_tail[:, 1] ** 2))),
        "tail_x24_norm_mae": finite_mean(norm_tail[:, 0]),
        "tail_T85_norm_mae": finite_mean(norm_tail[:, 1]),
        "tail_norm_mae": finite_mean(norm_tail),
        "tail_outside_band_fraction": finite_mean(np.any(norm_tail > 1.0, axis=1)),
    }


def reward_sensitivity_rows(dist_bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_name, params in reward_variants().items():
        for method, bundle in dist_bundles.items():
            breakdown = dist_reward_breakdown(bundle, params)
            row = {
                "variant": variant_name,
                "method": method,
                "tail_reward": finite_tail(breakdown.get("avg_reward", np.asarray([], float))),
                "final_reward": final_value(breakdown.get("avg_reward", np.asarray([], float))),
            }
            row.update(dist_tracking_summary(bundle, params))
            rows.append(row)
    return rows


def reward_component_rows(dist_bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = deepcopy(DIST_REWARD_DEFAULTS)
    for method in ["OF-MPC", "TD3 Weights", "TD3 Residual", "TD3 Markov"]:
        bundle = dist_bundles[method]
        b = dist_reward_breakdown(bundle, params)
        if not b:
            continue
        slc = tail_slice(bundle)
        row = {
            "method": method,
            "tail_reward": finite_tail(b["avg_reward"]),
            "tail_error_penalty": tail_mean_series(b["err_eff"], bundle),
            "tail_move_penalty": tail_mean_series(b["move"], bundle),
            "tail_outside_linear_penalty": tail_mean_series(b["lin_out"], bundle),
            "tail_inside_linear_penalty": tail_mean_series(b["lin_in"], bundle),
            "tail_bonus": tail_mean_series(b["bonus"], bundle),
            "tail_x24_error_quad": finite_mean(b["err_quad_by_output"][slc, 0]),
            "tail_T85_error_quad": finite_mean(b["err_quad_by_output"][slc, 1]),
            "tail_w_in": finite_mean(b["w_in"][slc]),
        }
        rows.append(row)
    return rows


def markov_episode_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    params = deepcopy(DIST_REWARD_DEFAULTS)
    breakdown = dist_reward_breakdown(bundle, params)
    rewards = breakdown["reward"]
    z = arr(bundle.get("z_executed_log", bundle.get("z_log")))
    req_proj = arr(bundle.get("z_safety_requested_projection_active_log"))
    ls_proj = arr(bundle.get("z_safety_ls_projection_active_log"))
    source = arr(bundle.get("rl_action_source_log"))
    score = arr(bundle.get("requested_prediction_score_log"))
    markov_pred = arr(bundle.get("prediction_error_markov_log"))
    nominal_pred = arr(bundle.get("prediction_error_nominal_log"))
    cost_pass = arr(bundle.get("requested_cost_guard_pass_log"))
    norm_error = breakdown["norm_abs_error"]
    views = {
        "reward": episode_view(rewards, bundle),
        "z_norm": episode_view(np.linalg.norm(z, axis=1), bundle) if z.ndim == 2 else np.asarray([]),
        "req_proj": episode_view(req_proj > 0, bundle),
        "ls_proj": episode_view(ls_proj > 0, bundle),
        "source": episode_view(source, bundle),
        "score": episode_view(score, bundle),
        "markov_pred": episode_view(markov_pred, bundle),
        "nominal_pred": episode_view(nominal_pred, bundle),
        "cost_pass": episode_view(cost_pass > 0, bundle),
        "norm_error": episode_view(norm_error, bundle),
    }
    n_ep = views["reward"].shape[0]
    rows: list[dict[str, Any]] = []
    for ep in range(n_ep):
        row = {
            "episode": ep + 1,
            "reward": finite_mean(views["reward"][ep]),
            "z_norm": finite_mean(views["z_norm"][ep]) if views["z_norm"].size else float("nan"),
            "requested_projection_fraction": finite_mean(views["req_proj"][ep]) if views["req_proj"].size else float("nan"),
            "ls_projection_fraction": finite_mean(views["ls_proj"][ep]) if views["ls_proj"].size else float("nan"),
            "td3_source_fraction": finite_mean(views["source"][ep] == 2) if views["source"].size else float("nan"),
            "prediction_score": finite_mean(views["score"][ep]) if views["score"].size else float("nan"),
            "markov_prediction_error": finite_mean(views["markov_pred"][ep]) if views["markov_pred"].size else float("nan"),
            "nominal_prediction_error": finite_mean(views["nominal_pred"][ep]) if views["nominal_pred"].size else float("nan"),
            "requested_cost_guard_pass_fraction": finite_mean(views["cost_pass"][ep]) if views["cost_pass"].size else float("nan"),
            "x24_norm_error": finite_mean(views["norm_error"][ep, :, 0]) if views["norm_error"].size else float("nan"),
            "T85_norm_error": finite_mean(views["norm_error"][ep, :, 1]) if views["norm_error"].size else float("nan"),
        }
        row["prediction_error_ratio"] = row["markov_prediction_error"] / max(row["nominal_prediction_error"], 1.0e-12)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], cols: list[tuple[str, str, int]]) -> str:
    lines = ["| " + " | ".join(c[0] for c in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        vals = []
        for _, key, digits in cols:
            val = row.get(key)
            if isinstance(val, str):
                vals.append(val)
            else:
                vals.append(fmt(val, digits))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_residual_figures(res_rows: list[dict[str, Any]], poly_res: dict[str, Any], dist_res: dict[str, Any]) -> None:
    labels = [row["case"] for row in res_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    axes[0].bar(x, [row["tail_reward_delta_vs_baseline"] for row in res_rows], color=["#2563eb", "#dc2626"])
    axes[0].axhline(0.0, color="black", linewidth=1.2)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Residual tail reward change vs OF-MPC")
    axes[0].set_ylabel("Tail reward delta")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.18
    metrics = [
        ("raw bound frac", "raw_tail_saturation_fraction"),
        ("projection", "projection_active_tail_fraction"),
        ("authority", "authority_projection_tail_fraction"),
        ("exec/raw norm", "exec_to_raw_norm_ratio"),
        ("rho eff", "rho_eff_tail_mean"),
    ]
    for idx, (label, key) in enumerate(metrics):
        axes[1].bar(x + (idx - 2) * width, [row[key] for row in res_rows], width, label=label)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.12)
    axes[1].set_title("Residual policy versus authority layer")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_residual_polymer_vs_distillation_authority.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.2), sharex=False, constrained_layout=True)
    for ax, label, bundle in [(axes[0], "Polymer", poly_res), (axes[1], "Distillation", dist_res)]:
        slc = tail_slice(bundle, episodes=4)
        raw = arr(bundle.get("residual_raw_log"))[slc]
        exe = arr(bundle.get("residual_exec_log"))[slc]
        rho = arr(bundle.get("rho_eff_log"))[slc]
        n = raw.shape[0]
        step = max(1, n // 1200)
        idx = np.arange(0, n, step)
        ax.plot(idx, np.linalg.norm(raw[idx], axis=1), color="#f97316", linewidth=1.4, label="raw residual norm")
        ax.plot(idx, np.linalg.norm(exe[idx], axis=1), color="#2563eb", linewidth=1.4, label="executed residual norm")
        if rho.size:
            ax.plot(idx, rho[idx], color="#64748b", linewidth=1.1, alpha=0.85, label="rho_eff")
        ax.set_title(f"{label} residual tail authority trace")
        ax.set_ylabel("Scaled residual / rho")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, ncol=3)
    axes[-1].set_xlabel("Tail sample index")
    fig.savefig(FIG_DIR / "fig_residual_tail_authority_traces.png")
    plt.close(fig)


def make_markov_figures(rows: list[dict[str, Any]]) -> None:
    ep = np.asarray([r["episode"] for r in rows])
    reward = np.asarray([r["reward"] for r in rows])
    z_norm = np.asarray([r["z_norm"] for r in rows])
    proj = np.asarray([r["requested_projection_fraction"] for r in rows])
    t_norm = np.asarray([r["T85_norm_error"] for r in rows])
    pred_ratio = np.asarray([r["prediction_error_ratio"] for r in rows])
    cost_pass = np.asarray([r["requested_cost_guard_pass_fraction"] for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    axes[0].plot(ep, reward, color="#16a34a", linewidth=1.7)
    axes[0].axhline(0.0, color="black", linewidth=1.1)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Distillation Markov episode diagnostics")
    axes[0].grid(alpha=0.25)
    axes[1].plot(ep, z_norm, color="#2563eb", linewidth=1.4, label="mean z norm")
    axes[1].plot(ep, proj, color="#f97316", linewidth=1.4, label="requested projection fraction")
    axes[1].set_ylabel("z / projection")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=True)
    axes[2].plot(ep, t_norm, color="#dc2626", linewidth=1.4, label="T85 abs error / band")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="reward band")
    axes[2].set_ylabel("T85 norm err")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=True)
    axes[3].plot(ep, pred_ratio, color="#7c3aed", linewidth=1.4, label="Markov pred error / nominal")
    axes[3].plot(ep, cost_pass, color="#64748b", linewidth=1.4, label="requested cost guard pass frac")
    axes[3].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[3].set_ylabel("Prediction diagnostics")
    axes[3].set_xlabel("Episode")
    axes[3].grid(alpha=0.25)
    axes[3].legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_markov_episode_diagnostics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    scatter = ax.scatter(z_norm, reward, c=t_norm, cmap="magma", s=55, edgecolor="black", linewidth=0.3)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Episode mean z norm")
    ax.set_ylabel("Episode reward")
    ax.set_title("Markov reward versus z norm, colored by T85 band error")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("T85 abs error / reward band")
    fig.savefig(FIG_DIR / "fig_markov_reward_vs_z_norm.png")
    plt.close(fig)


def make_reward_sensitivity_figures(rows: list[dict[str, Any]], component_rows: list[dict[str, Any]]) -> None:
    variants = list(reward_variants().keys())
    variant_labels = ["current\nQT=1500", "tight\nQT=1000", "tight\nQT=5000", "tight\nQT=10000", "wide band\nQT=1500"]
    methods = list(DIST_RUNS.keys())
    x_pos = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for method in methods:
        vals = [next(r["tail_reward"] for r in rows if r["variant"] == variant and r["method"] == method) for variant in variants]
        ax.plot(x_pos, vals, marker="o", linewidth=2.0, label=method)
    ax.set_xticks(x_pos, variant_labels)
    ax.set_ylabel("Tail reward under variant")
    ax.set_title("Distillation reward-parameter sensitivity on saved trajectories")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.savefig(FIG_DIR / "fig_reward_parameter_sensitivity.png")
    plt.close(fig)

    subset = ["OF-MPC", "TD3 Weights", "TD3 Residual", "TD3 Markov"]
    by_method = {r["method"]: r for r in component_rows}
    x = np.arange(len(subset))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    comps = [
        ("error", "tail_error_penalty", "#dc2626"),
        ("move", "tail_move_penalty", "#f97316"),
        ("outside", "tail_outside_linear_penalty", "#7c3aed"),
        ("bonus", "tail_bonus", "#16a34a"),
    ]
    for idx, (label, key, color) in enumerate(comps):
        ax.bar(x + (idx - 1.5) * width, [by_method[m][key] for m in subset], width, label=label, color=color)
    ax.set_xticks(x, subset, rotation=15, ha="right")
    ax.set_ylabel("Tail mean component magnitude")
    ax.set_title("Current reward component breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(FIG_DIR / "fig_reward_component_breakdown.png")
    plt.close(fig)


def report_text(
    residual_rows: list[dict[str, Any]],
    markov_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    dist_bundles: dict[str, dict[str, Any]],
) -> str:
    res_by_case = {r["case"]: r for r in residual_rows}
    markov_tail = markov_rows[-TAIL_EPISODES:]
    markov_summary = {
        "tail_reward": finite_mean([r["reward"] for r in markov_tail]),
        "tail_z_norm": finite_mean([r["z_norm"] for r in markov_tail]),
        "tail_requested_projection": finite_mean([r["requested_projection_fraction"] for r in markov_tail]),
        "tail_T85_norm_error": finite_mean([r["T85_norm_error"] for r in markov_tail]),
        "tail_prediction_ratio": finite_mean([r["prediction_error_ratio"] for r in markov_tail]),
        "tail_cost_guard_pass": finite_mean([r["requested_cost_guard_pass_fraction"] for r in markov_tail]),
    }
    reward_current = [r for r in reward_rows if r["variant"] == "current_tight_QT1500"]
    reward_q5000 = [r for r in reward_rows if r["variant"] == "tight_QT5000"]
    reward_q10000 = [r for r in reward_rows if r["variant"] == "tight_QT10000"]
    reward_wide = [r for r in reward_rows if r["variant"] == "wide_band_QT1500"]
    current_rank = sorted(reward_current, key=lambda r: r["tail_reward"], reverse=True)
    q5000_rank = sorted(reward_q5000, key=lambda r: r["tail_reward"], reverse=True)
    q10000_rank = sorted(reward_q10000, key=lambda r: r["tail_reward"], reverse=True)

    residual_table = markdown_table(
        residual_rows,
        [
            ("Case", "case", 0),
            ("Reward basis", "reward_basis", 0),
            ("Residual tail reward", "residual_tail_reward", 2),
            ("Delta vs OF-MPC", "tail_reward_delta_vs_baseline", 2),
            ("Raw bound frac", "raw_tail_saturation_fraction", 3),
            ("Projection frac", "projection_active_tail_fraction", 3),
            ("Authority frac", "authority_projection_tail_fraction", 3),
            ("Exec/raw norm", "exec_to_raw_norm_ratio", 3),
            ("rho eff", "rho_eff_tail_mean", 3),
        ],
    )
    reward_table_rows = []
    for method in DIST_RUNS:
        row = {"method": method}
        for variant in ["current_tight_QT1500", "tight_QT5000", "tight_QT10000", "wide_band_QT1500"]:
            hit = next(r for r in reward_rows if r["variant"] == variant and r["method"] == method)
            row[variant] = hit["tail_reward"]
        reward_table_rows.append(row)
    reward_table = markdown_table(
        reward_table_rows,
        [
            ("Method", "method", 0),
            ("Current QT1500", "current_tight_QT1500", 2),
            ("Tight QT5000", "tight_QT5000", 2),
            ("Tight QT10000", "tight_QT10000", 2),
            ("Wide band QT1500", "wide_band_QT1500", 2),
        ],
    )
    component_table = markdown_table(
        component_rows,
        [
            ("Method", "method", 0),
            ("Tail reward", "tail_reward", 2),
            ("Error penalty", "tail_error_penalty", 2),
            ("Move penalty", "tail_move_penalty", 2),
            ("Outside penalty", "tail_outside_linear_penalty", 2),
            ("Inside penalty", "tail_inside_linear_penalty", 2),
            ("Bonus", "tail_bonus", 2),
            ("T85 quad", "tail_T85_error_quad", 3),
            ("w_in", "tail_w_in", 3),
        ],
    )

    markov_bundle = dist_bundles["TD3 Markov"]
    z_safety = json.dumps(sanitize(markov_bundle.get("z_safety", {})), sort_keys=True)

    lines: list[str] = []
    lines.append("# Distillation Residual, Markov, And Reward-Parameter Focus Report")
    lines.append("")
    lines.append("Date: 2026-05-21")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This focused report answers three questions: why the residual policy helps in the polymer case but fails in the latest distillation run, why the latest distillation Markov run looks strange, and what reward-parameter changes are worth testing next."
    )
    lines.append("")
    lines.append(
        f"- Residual is not failing because the residual layer is inactive. It is failing because the distillation residual actor saturates the raw correction at the action bounds while the authority layer projects every tail step. Tail projection is `{fmt(res_by_case['Distillation']['projection_active_tail_fraction'], 3)}` and raw-bound fraction is `{fmt(res_by_case['Distillation']['raw_tail_saturation_fraction'], 3)}`."
    )
    lines.append(
        f"- Polymer residual also gets projected, but it still improves tail reward over OF-MPC by `{fmt(res_by_case['Polymer']['tail_reward_delta_vs_baseline'], 2)}` on the saved polymer reward basis. Distillation residual changes tail reward by `{fmt(res_by_case['Distillation']['tail_reward_delta_vs_baseline'], 2)}` versus OF-MPC after recomputing all distillation trajectories with the current reward, which is the opposite sign."
    )
    lines.append(
        f"- The Markov run is weird because TD3 is the executed source in the tail, but the Markov prediction/cost diagnostics are not reassuring: tail requested projection fraction is `{fmt(markov_summary['tail_requested_projection'], 3)}`, T85 normalized error is `{fmt(markov_summary['tail_T85_norm_error'], 3)}`, and requested cost-guard pass fraction is only `{fmt(markov_summary['tail_cost_guard_pass'], 3)}`."
    )
    lines.append(
        f"- Reward sensitivity says increasing the temperature weight from `Q_T = 1500` to `5000` or `10000` mostly punishes the Markov and residual trajectories more. It does not rescue residual. The ranking remains led by `{current_rank[0]['method']}` under current reward and `{q5000_rank[0]['method']}` under `Q_T = 5000`."
    )
    lines.append("")
    lines.append("## Files Inspected")
    lines.append("")
    for label, path in DIST_RUNS.items():
        lines.append(f"- Distillation {label}: `{path.relative_to(REPO_ROOT).as_posix()}`")
    for label, path in POLY_RUNS.items():
        lines.append(f"- Polymer {label}: `{path.relative_to(REPO_ROOT).as_posix()}`")
    lines.append("- `utils/residual_runner.py`")
    lines.append("- `utils/markov_runner.py`")
    lines.append("- `systems/distillation/config.py`")
    lines.append("- `systems/distillation/notebook_params.py`")
    lines.append("- `systems/polymer/config.py`")
    lines.append("- `systems/polymer/notebook_params.py`")
    lines.append("")
    lines.append("## Method Reconstruction")
    lines.append("")
    lines.append("For residual RL, the MPC proposes a nominal first move `u_MPC`, and TD3 proposes a scaled residual `a_res`. The executable correction is not the raw actor output; it is projected through the authority layer:")
    lines.append("")
    lines.append("$$ \\Delta u_{\\mathrm{res,exec}} = \\Pi_{\\rho,\\mathrm{bounds},\\mathrm{deadband}}(\\Delta u_{\\mathrm{res,raw}}), \\qquad u_0 = u_{\\mathrm{MPC}} + \\Delta u_{\\mathrm{res,exec}}. $$")
    lines.append("")
    lines.append("For Markov RL, the TD3 action selects a lifted response correction:")
    lines.append("")
    lines.append("$$ M_z = M_0 + \\sum_i z_i B_i, \\qquad |z_i| \\leq z_{\\mathrm{cap,eff}}, \\qquad ||z||_2 \\leq z_{\\mathrm{norm,max}}. $$")
    lines.append("")
    lines.append(f"The latest distillation Markov bundle uses `markov_z_bound = {fmt(markov_bundle.get('markov_z_bound'), 3)}` and `z_safety = {z_safety}`.")
    lines.append("")
    lines.append("For reward sensitivity, each saved distillation trajectory is rescored without rerunning Aspen using:")
    lines.append("")
    lines.append("$$ r_t = -\\ell_{e,t} - \\ell_{\\Delta u,t} - \\ell_{\\mathrm{outside},t} - \\ell_{\\mathrm{inside},t} + b_t. $$")
    lines.append("")
    lines.append("The alternatives change only the reward parameters used to evaluate the same saved trajectories; they are not counterfactual closed-loop simulations.")
    lines.append("")
    lines.append("## 1. Why Residual Works In Polymer But Not Here")
    lines.append("")
    lines.append(residual_table)
    lines.append("")
    lines.append(
        "Note: the distillation row uses recomputed current reward because the historical OF-MPC pickle stores an older native reward that is not comparable to the latest reward function. The polymer row uses its native saved reward because the comparison is within the same saved polymer reward basis."
    )
    lines.append("")
    lines.append("![Residual polymer versus distillation authority](figures/distillation_residual_markov_reward_focus_20260521/fig_residual_polymer_vs_distillation_authority.png)")
    lines.append("")
    lines.append("![Residual tail authority traces](figures/distillation_residual_markov_reward_focus_20260521/fig_residual_tail_authority_traces.png)")
    lines.append("")
    lines.append(
        "The mechanism is different across plants. In polymer, residual authority is also constrained, but the executed correction still gives a useful local input nudge and the tail reward improves relative to OF-MPC. In distillation, the raw residual actor is almost a bang-bang controller at `[+0.05, -0.05]`, while the authority layer scales it back every tail step. That means the critic is learning around a raw action that is mostly not executable."
    )
    lines.append("")
    lines.append(
        "The distillation column is also more sensitive to direct input residuals: reflux and reboiler duty are tightly coupled through slow tray/composition dynamics, and the reward band on T85 is tight. A residual correction that looks small in scaled input coordinates can create long thermal/composition transients. The residual layer has no prediction model of that long tail; it only adds a direct move correction after the MPC solve."
    )
    lines.append("")
    lines.append("My interpretation: residual is not a good first standalone authority mechanism for this distillation setup unless we either reduce residual bounds, train the actor inside the projected action set, or add a candidate-cost/safety gate analogous to the Markov gate.")
    lines.append("")
    lines.append("## 2. Why The Latest Markov Run Looks Weird")
    lines.append("")
    lines.append(
        f"The latest Markov run is strong by scalar reward but strange by diagnostics. Tail TD3 source is effectively always on, requested z is projected in `{fmt(markov_summary['tail_requested_projection'], 3)}` of tail steps, and the T85 normalized error stays around `{fmt(markov_summary['tail_T85_norm_error'], 3)}`. The output plot shows good composition behavior but poor/oscillatory T85 behavior."
    )
    lines.append("")
    lines.append("![Markov episode diagnostics](figures/distillation_residual_markov_reward_focus_20260521/fig_markov_episode_diagnostics.png)")
    lines.append("")
    lines.append("![Markov reward versus z norm](figures/distillation_residual_markov_reward_focus_20260521/fig_markov_reward_vs_z_norm.png)")
    lines.append("")
    lines.append(
        "A key red flag is that Markov prediction error is often worse than nominal prediction error, while TD3 remains the executed source. That does not mean the run is unusable; it means the scalar reward is rewarding some parts of the behavior even when the Markov model correction is not a uniformly better local predictor."
    )
    lines.append("")
    lines.append(
        "My interpretation: the z-safety layer is doing its job as a magnitude shield, but the acceptance metric is still too permissive for this plant. The next Markov step should not be only a smaller z bound; it should also require prediction/cost usefulness, especially on T85-sensitive transients."
    )
    lines.append("")
    lines.append("## 3. Reward-Parameter Sensitivity")
    lines.append("")
    lines.append(reward_table)
    lines.append("")
    lines.append("![Reward parameter sensitivity](figures/distillation_residual_markov_reward_focus_20260521/fig_reward_parameter_sensitivity.png)")
    lines.append("")
    lines.append(component_table)
    lines.append("")
    lines.append("![Reward component breakdown](figures/distillation_residual_markov_reward_focus_20260521/fig_reward_component_breakdown.png)")
    lines.append("")
    lines.append(
        f"Under the current tight reward with `Q_T = 1500`, the best saved trajectory is `{current_rank[0]['method']}`. Under `Q_T = 5000`, the best saved trajectory is `{q5000_rank[0]['method']}`. Under `Q_T = 10000`, the best saved trajectory is `{q10000_rank[0]['method']}`."
    )
    lines.append("")
    lines.append(
        "Increasing `Q_T` makes T85 error more visible and penalizes Markov's weird temperature behavior. That is scientifically useful if we care about tray-85 temperature, but it will not by itself fix residual. Residual is already failing dynamically and through projection, so a stronger temperature penalty may simply make the failure more obvious."
    )
    lines.append("")
    lines.append("The wide-band variant raises rewards by making the temperature target easier. That is helpful for diagnosing whether a reward is too harsh, but it should not be used to claim better control unless we explicitly accept a looser T85 tolerance.")
    lines.append("")
    lines.append("## Recommended Next Steps")
    lines.append("")
    lines.append("1. **Residual next run:** shrink residual bounds for distillation from `[-0.05, 0.05]` to about `[-0.02, 0.02]`, keep rho authority, and store executed actions in replay. Metric: projection fraction should drop and raw-bound fraction should no longer be near one.")
    lines.append("2. **Residual safety gate:** add an MPC candidate-cost usefulness gate or behavior-cloning penalty to keep raw residual actions close to the executable region. Metric: raw/executed residual norm ratio should increase without tail reward collapse.")
    lines.append("3. **Markov next run:** keep `z_bound = 0.04`, but add a usefulness gate that rejects/probates TD3 when Markov prediction error is worse than nominal or when T85 band error is growing. Metric: T85 normalized error should fall below one without losing composition tracking.")
    lines.append("4. **Reward experiment:** do not jump to `Q_T = 10000` first. Run one controlled `Q_T = 5000` tight-band experiment and compare to `Q_T = 1500`. Metric: T85 band-normalized error, not only scalar reward.")
    lines.append("5. **Evaluation protocol:** freeze policies and run a common fluctuation schedule. Current reports are training-rollout diagnostics, not final generalization evidence.")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("For the next distillation work, I would prioritize **TD3 Weights as the current best performer**, **Markov with a stronger usefulness/T85 gate**, and **a much smaller residual-authority experiment**. Reward changes are worth testing, but they should be treated as diagnostics unless the closed-loop behavior also improves.")
    return "\n".join(lines) + "\n"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dist_bundles = {label: load_pickle(path) for label, path in DIST_RUNS.items()}
    poly_bundles = {label: load_pickle(path) for label, path in POLY_RUNS.items()}

    residual_rows = [
        residual_summary("Polymer", poly_bundles["OF-MPC"], poly_bundles["TD3 Residual"]),
        residual_summary("Distillation", dist_bundles["OF-MPC"], dist_bundles["TD3 Residual"]),
    ]
    reward_rows = reward_sensitivity_rows(dist_bundles)
    apply_distillation_recomputed_residual_rewards(residual_rows, reward_rows)
    component_rows = reward_component_rows(dist_bundles)
    markov_rows = markov_episode_rows(dist_bundles["TD3 Markov"])

    write_csv(FIG_DIR / "residual_polymer_distillation_summary.csv", residual_rows)
    write_csv(FIG_DIR / "distillation_reward_parameter_sensitivity.csv", reward_rows)
    write_csv(FIG_DIR / "distillation_reward_component_breakdown.csv", component_rows)
    write_csv(FIG_DIR / "distillation_markov_episode_diagnostics.csv", markov_rows)
    (FIG_DIR / "summary.json").write_text(
        json.dumps(
            {
                "residual": sanitize(residual_rows),
                "reward_sensitivity": sanitize(reward_rows),
                "reward_components": sanitize(component_rows),
                "markov_episode_diagnostics": sanitize(markov_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    make_residual_figures(residual_rows, poly_bundles["TD3 Residual"], dist_bundles["TD3 Residual"])
    make_markov_figures(markov_rows)
    make_reward_sensitivity_figures(reward_rows, component_rows)
    REPORT_PATH.write_text(report_text(residual_rows, markov_rows, reward_rows, component_rows, dist_bundles), encoding="utf-8")
    print(f"Wrote report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote figures: {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
