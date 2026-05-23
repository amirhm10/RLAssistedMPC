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

from systems.distillation.config import DISTILLATION_INPUT_BOUNDS, RL_REWARD_DEFAULTS


RESULTS = REPO_ROOT / "Distillation" / "Results"
FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_protected_bc_latest_20260523"
SUMMARY_CSV = FIG_DIR / "summary_metrics.csv"
MECH_CSV = FIG_DIR / "td3_authority_diagnostics.csv"
SUMMARY_JSON = FIG_DIR / "summary.json"
N_INPUTS = 2
TAIL_EPISODES = 10


RUNS = [
    {
        "method": "OF-MPC",
        "family": "baseline",
        "path": REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
        "color": "#111827",
    },
    {
        "method": "TD3 Weights",
        "family": "weights",
        "path": RESULTS / "distillation_weights_td3_disturb_fluctuation_mismatch_unified" / "20260522_181031" / "input_data.pkl",
        "color": "#7c3aed",
    },
    {
        "method": "TD3 Residual",
        "family": "residual",
        "path": RESULTS / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified" / "20260522_180223" / "input_data.pkl",
        "color": "#dc2626",
    },
    {
        "method": "Horizon DDQN",
        "family": "horizon",
        "path": RESULTS / "distillation_horizon_disturb_fluctuation_mismatch_unified" / "20260522_183427" / "input_data.pkl",
        "color": "#2563eb",
    },
    {
        "method": "Dueling Horizon",
        "family": "dueling",
        "path": RESULTS / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified" / "20260522_183355" / "input_data.pkl",
        "color": "#0891b2",
    },
    {
        "method": "TD3 Markov",
        "family": "markov",
        "path": RESULTS / "distillation_markov_td3_disturb_fluctuation_unified" / "20260522_190448" / "input_data.pkl",
        "color": "#16a34a",
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
    return np.asarray(value, dtype=dtype)


def finite_mean(value: Any) -> float:
    data = np.asarray(value, float).reshape(-1)
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else float("nan")


def tail_mean(value: Any, count: int = TAIL_EPISODES) -> float:
    data = np.asarray(value, float).reshape(-1)
    data = data[np.isfinite(data)]
    if not data.size:
        return float("nan")
    return float(np.mean(data[-min(count, data.size) :]))


def final_value(value: Any) -> float:
    data = np.asarray(value, float).reshape(-1)
    data = data[np.isfinite(data)]
    return float(data[-1]) if data.size else float("nan")


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


def u_steps(bundle: dict[str, Any]) -> np.ndarray:
    u = arr(bundle.get("u_rl", bundle.get("u")))
    y_sp = arr(bundle.get("y_sp"))
    if u.ndim != 2 or y_sp.ndim != 2:
        return np.asarray([], float)
    return u[: min(u.shape[0], y_sp.shape[0])]


def band_phys(bundle: dict[str, Any]) -> np.ndarray:
    y_sp = physical_setpoints(bundle)
    if y_sp.size == 0:
        return np.asarray([], float)
    k_rel = arr(RL_REWARD_DEFAULTS["k_rel"])
    floor = arr(RL_REWARD_DEFAULTS["band_floor_phys"])
    return np.maximum(k_rel.reshape(1, -1) * np.abs(y_sp), floor.reshape(1, -1))


def recompute_rewards(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    delta_y = arr(bundle.get("delta_y_storage"))
    delta_u = arr(bundle.get("delta_u_storage"))
    y_sp_phys = physical_setpoints(bundle)
    data_min = arr(bundle.get("data_min"))
    data_max = arr(bundle.get("data_max"))
    if delta_y.ndim != 2 or delta_u.ndim != 2 or y_sp_phys.ndim != 2 or data_min.size < 4:
        return np.asarray([], float), np.asarray([], float)
    n = min(delta_y.shape[0], delta_u.shape[0], y_sp_phys.shape[0])
    delta_y = delta_y[:n]
    delta_u = delta_u[:n]
    y_sp_phys = y_sp_phys[:n]

    params = RL_REWARD_DEFAULTS
    dy_scale = np.maximum(data_max[N_INPUTS:] - data_min[N_INPUTS:], 1.0e-12)
    q_diag = arr(params["Q_diag"])
    r_diag = arr(params["R_diag"])
    band_scaled = np.maximum(arr(params["k_rel"]).reshape(1, -1) * np.abs(y_sp_phys), arr(params["band_floor_phys"]).reshape(1, -1)) / dy_scale.reshape(1, -1)
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
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    episodes = rewards.size // steps
    avg = rewards[: episodes * steps].reshape(episodes, steps).mean(axis=1) if episodes else np.asarray([], float)
    return rewards, avg


def tail_slice(bundle: dict[str, Any], episodes: int = TAIL_EPISODES) -> slice:
    steps = int(bundle.get("time_in_sub_episodes") or 400)
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    width = min(nfe, steps * episodes)
    return slice(max(0, nfe - width), nfe)


def post_warm_slice(bundle: dict[str, Any]) -> slice:
    nfe = int(bundle.get("nFE") or arr(bundle.get("y_sp")).shape[0])
    return slice(min(nfe, int(bundle.get("warm_start_step") or 0)), nfe)


def release_key(bundle: dict[str, Any], suffix: str) -> str | None:
    if suffix in bundle:
        return suffix
    prefixed = "rl_" + suffix
    return prefixed if prefixed in bundle else None


def fraction(bundle: dict[str, Any], key: str | None, sl: slice) -> float:
    if key is None:
        return float("nan")
    data = arr(bundle.get(key))
    return finite_mean(data[sl] > 0) if data.size else float("nan")


def summarize(method: str, family: str, path: Path, bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _, rescored_avg = recompute_rewards(bundle)
    y = y_steps(bundle)
    sp = physical_setpoints(bundle)
    band = band_phys(bundle)
    tail = tail_slice(bundle)
    post = post_warm_slice(bundle)
    e_tail = y[tail] - sp[tail]
    norm_tail = np.abs(e_tail) / np.maximum(band[tail], 1.0e-12)
    delta_u = arr(bundle.get("delta_u_storage"))
    du_tail = delta_u[tail] if delta_u.ndim == 2 else np.asarray([], float)

    summary = {
        "method": method,
        "family": family,
        "bundle": path.relative_to(REPO_ROOT).as_posix(),
        "episodes": int(len(arr(bundle.get("avg_rewards")))),
        "warm_episodes": float(int(bundle.get("warm_start_step") or 0) / max(int(bundle.get("time_in_sub_episodes") or 400), 1)),
        "logged_tail_reward": tail_mean(bundle.get("avg_rewards")),
        "current_tail_reward": tail_mean(rescored_avg),
        "current_final_reward": final_value(rescored_avg),
        "x24_rmse_tail": float(np.sqrt(np.mean(e_tail[:, 0] ** 2))),
        "T85_rmse_tail": float(np.sqrt(np.mean(e_tail[:, 1] ** 2))),
        "band_norm_mae_tail": finite_mean(norm_tail),
        "outside_band_tail_frac": finite_mean(np.any(norm_tail > 1.0, axis=1)),
        "mean_abs_du_scaled_tail": finite_mean(np.abs(du_tail)),
    }

    blocked_key = release_key(bundle, "release_gate_blocked_log")
    released_key = release_key(bundle, "release_gate_released_log")
    gap_key = release_key(bundle, "release_gate_action_gap_norm_log")
    max_gap_key = release_key(bundle, "release_gate_max_coordinate_gap_log")
    release_step_key = "protected_bc_release_gate_release_step" if "protected_bc_release_gate_release_step" in bundle else "rl_protected_bc_release_gate_release_step"
    mechanism = {
        "method": method,
        "post_warm_release_blocked_frac": fraction(bundle, blocked_key, post),
        "post_warm_release_released_frac": fraction(bundle, released_key, post),
        "tail_release_blocked_frac": fraction(bundle, blocked_key, tail),
        "release_step": int(bundle.get(release_step_key, -1)) if release_step_key in bundle else -1,
        "bc_active_frac": fraction(bundle, "bc_active_log", slice(0, int(bundle.get("nFE") or 0))),
        "bc_gap_post_warm_mean": finite_mean(arr(bundle.get("bc_policy_target_distance_log"))[post]) if "bc_policy_target_distance_log" in bundle else float("nan"),
        "release_gap_post_warm_mean": finite_mean(arr(bundle.get(gap_key))[post]) if gap_key else float("nan"),
        "release_max_coord_gap_post_warm_mean": finite_mean(arr(bundle.get(max_gap_key))[post]) if max_gap_key else float("nan"),
    }

    weight_log = arr(bundle.get("weight_log"))
    if weight_log.ndim == 2:
        for idx, label in enumerate(["Q1", "Q2", "R1", "R2"]):
            mechanism[f"tail_{label}_mult"] = finite_mean(weight_log[tail, idx])
        mechanism["tail_weight_distinct_rows"] = int(np.unique(weight_log[tail], axis=0).shape[0])

    raw_res = arr(bundle.get("a_res_raw_log"))
    exec_res = arr(bundle.get("a_res_exec_log"))
    if raw_res.ndim == 2 and exec_res.ndim == 2:
        mechanism["tail_raw_residual_norm"] = finite_mean(np.linalg.norm(raw_res[tail], axis=1))
        mechanism["tail_exec_residual_norm"] = finite_mean(np.linalg.norm(exec_res[tail], axis=1))
        mechanism["tail_projection_active_frac"] = fraction(bundle, "projection_active_log", tail)
        mechanism["tail_raw_exec_norm_ratio"] = finite_mean(arr(bundle.get("residual_raw_executed_norm_ratio_log"))[tail])

    h = arr(bundle.get("horizon_trace"))
    if h.ndim == 2:
        mechanism["tail_Hp_mean"] = finite_mean(h[tail, 0])
        mechanism["tail_Hc_mean"] = finite_mean(h[tail, 1])
        mechanism["post_warm_distinct_horizons"] = int(np.unique(h[post], axis=0).shape[0])
        mechanism["tail_distinct_horizons"] = int(np.unique(h[tail], axis=0).shape[0])

    source = arr(bundle.get("rl_action_source_log"))
    if source.ndim == 1 and source.size:
        mechanism["post_warm_td3_source_frac"] = finite_mean(source[post] == 2)
        mechanism["post_warm_ls_source_frac"] = finite_mean(source[post] == 3)
        mechanism["post_warm_nominal_source_frac"] = finite_mean(source[post] == 4)
        mechanism["post_warm_warm_ls_source_frac"] = finite_mean(source[post] == 1)

    z = arr(bundle.get("z_executed_log", bundle.get("z_log")))
    if z.ndim == 2:
        mechanism["tail_q95_abs_z"] = float(np.nanquantile(np.abs(z[tail]), 0.95))
        mechanism["tail_z_norm_mean"] = finite_mean(np.linalg.norm(z[tail], axis=1))
        mechanism["post_warm_z_safety_projection_frac"] = fraction(bundle, "z_safety_requested_projection_active_log", post)

    return summary, mechanism


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
            writer.writerow({key: row.get(key, "") for key in fields})


def rolling_mean(data: Any, window: int = 5) -> np.ndarray:
    values = arr(data)
    out = np.zeros_like(values)
    for idx in range(values.size):
        out[idx] = np.nanmean(values[max(0, idx - window + 1) : idx + 1])
    return out


def plot_reward(bundles: dict[str, dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2), constrained_layout=True)
    for spec in specs:
        _, avg = recompute_rewards(bundles[spec["method"]])
        ax.plot(np.arange(1, avg.size + 1), rolling_mean(avg), color=spec["color"], linewidth=2.0, label=spec["method"])
    ax.axvline(10, color="#6b7280", linestyle="--", linewidth=1.4, label="warm-start end")
    ax.set_title("Latest distillation runs: current-reward episode trends")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.savefig(FIG_DIR / "fig_reward_curves.png")
    plt.close(fig)


def plot_reward_error(summary: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in summary]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].bar(x, [row["current_tail_reward"] for row in summary], color="#f97316")
    axes[0].set_ylabel("Tail reward")
    axes[0].set_title("Current-rescored tail reward")
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.36
    axes[1].bar(x - width / 2, [row["x24_rmse_tail"] for row in summary], width, label="x24 RMSE", color="#2563eb")
    axes[1].bar(x + width / 2, [row["T85_rmse_tail"] for row in summary], width, label="T85 RMSE", color="#dc2626")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Tail physical tracking RMSE")
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=17, ha="right")
    fig.savefig(FIG_DIR / "fig_tail_reward_and_error.png")
    plt.close(fig)


def plot_td3_release(mech: list[dict[str, Any]]) -> None:
    rows = [row for row in mech if row["method"] in {"TD3 Weights", "TD3 Residual", "TD3 Markov"}]
    labels = [row["method"] for row in rows]
    x = np.arange(len(labels))
    width = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), constrained_layout=True)
    axes[0].bar(x - width, [row["post_warm_release_blocked_frac"] for row in rows], width, label="blocked", color="#ef4444")
    axes[0].bar(x, [row["post_warm_release_released_frac"] for row in rows], width, label="released", color="#22c55e")
    axes[0].bar(x + width, [row.get("bc_active_frac", np.nan) for row in rows], width, label="BC active", color="#64748b")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Protected-BC release status")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - width / 2, [row["release_gap_post_warm_mean"] for row in rows], width, label="mean raw action gap", color="#f97316")
    axes[1].bar(x + width / 2, [row["release_max_coord_gap_post_warm_mean"] for row in rows], width, label="max coordinate gap", color="#7c3aed")
    axes[1].axhline(0.25, color="#f97316", linestyle="--", linewidth=1.3, label="mean threshold")
    axes[1].axhline(0.20, color="#7c3aed", linestyle=":", linewidth=1.5, label="coordinate threshold")
    axes[1].set_ylabel("Raw-action gap")
    axes[1].set_title("Why continuous TD3 was not released")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "fig_td3_release_gate_diagnostics.png")
    plt.close(fig)


def sample_idx(n: int, max_points: int = 1400) -> np.ndarray:
    return np.arange(0, n, max(1, int(np.ceil(n / max_points))), dtype=int)


def plot_tail_tracking(bundles: dict[str, dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    tail_len = 1600
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.4), sharex=True, constrained_layout=True)
    labels = ["Tray-24 ethane composition", "Tray-85 temperature"]
    for out_idx, ax in enumerate(axes):
        for spec in specs:
            bundle = bundles[spec["method"]]
            y = y_steps(bundle)
            n = min(tail_len, y.shape[0])
            idx = sample_idx(n)
            t = idx * float(bundle.get("delta_t", 1.0 / 6.0))
            ax.plot(t, y[-n:, out_idx][idx], color=spec["color"], linewidth=1.6, label=spec["method"])
        sp = physical_setpoints(bundles["OF-MPC"])
        n = min(tail_len, sp.shape[0])
        idx = sample_idx(n)
        t = idx * float(bundles["OF-MPC"].get("delta_t", 1.0 / 6.0))
        ax.step(t, sp[-n:, out_idx][idx], color="black", linestyle="--", linewidth=2.0, where="post", label="setpoint")
        ax.set_ylabel(labels[out_idx])
        ax.grid(alpha=0.25)
        ax.set_title(f"Tail tracking overlay: {labels[out_idx]}")
    axes[-1].set_xlabel("Tail window time (h)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=True)
    fig.savefig(FIG_DIR / "fig_tail_tracking_overlay.png")
    plt.close(fig)


def plot_horizon_markov(bundles: dict[str, dict[str, Any]], mech: list[dict[str, Any]]) -> None:
    by_method = {row["method"]: row for row in mech}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    labels = ["Horizon DDQN", "Dueling Horizon"]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, [by_method[m].get("tail_Hp_mean", np.nan) for m in labels], width, label="Hp", color="#2563eb")
    axes[0].bar(x + width / 2, [by_method[m].get("tail_Hc_mean", np.nan) for m in labels], width, label="Hc", color="#f97316")
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_title("Tail selected horizons")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=True)

    markov = by_method["TD3 Markov"]
    names = ["warm LS", "TD3", "LS fallback", "nominal", "z projection"]
    vals = [
        markov.get("post_warm_warm_ls_source_frac", np.nan),
        markov.get("post_warm_td3_source_frac", np.nan),
        markov.get("post_warm_ls_source_frac", np.nan),
        markov.get("post_warm_nominal_source_frac", np.nan),
        markov.get("post_warm_z_safety_projection_frac", np.nan),
    ]
    axes[1].bar(names, vals, color=["#94a3b8", "#16a34a", "#0ea5e9", "#ef4444", "#f97316"])
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_title("Markov post-warm source/safety fractions")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "fig_horizon_and_markov_mechanisms.png")
    plt.close(fig)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    for spec in RUNS:
        bundle = load_pickle(spec["path"])
        bundles[spec["method"]] = bundle
        row, mech = summarize(spec["method"], spec["family"], spec["path"], bundle)
        summary.append(row)
        mechanism.append(mech)

    write_csv(SUMMARY_CSV, summary)
    write_csv(MECH_CSV, mechanism)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump({"summary": sanitize(summary), "mechanism": sanitize(mechanism)}, handle, indent=2)

    plot_reward(bundles, RUNS)
    plot_reward_error(summary)
    plot_td3_release(mechanism)
    plot_tail_tracking(bundles, RUNS)
    plot_horizon_markov(bundles, mechanism)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {MECH_CSV}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
