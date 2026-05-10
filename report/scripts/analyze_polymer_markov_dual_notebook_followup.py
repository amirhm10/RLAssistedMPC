from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.rewards import make_reward_fn_prototype_legacy


OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_dual_notebook_followup_20260510"

UNIFIED_LATEST = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260510_134814" / "input_data.pkl"
LEGACY_LATEST = (
    REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260510_123506" / "input_data.pkl"
)
LEGACY_OLD = REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260508_123902" / "input_data.pkl"
CANONICAL_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
SHARED_REWARD_REFERENCE = (
    REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260509_023119" / "input_data.pkl"
)

N_EPISODES = 200
STEPS_PER_EPISODE = 800
WINDOW = 20


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def physical_to_scaled_abs(values: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    return (np.asarray(values, float) - np.asarray(data_min, float)) / np.maximum(
        np.asarray(data_max, float) - np.asarray(data_min, float), 1.0e-12
    )


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def bonus_phi(z, kind="exp", k=12.0, p=0.6, c=20.0):
    z = np.clip(z, 0.0, 1.0)
    if kind == "linear":
        return 1.0 - z
    if kind == "quadratic":
        return (1.0 - z) ** 2
    if kind == "exp":
        return (np.exp(-k * z) - np.exp(-k)) / (1.0 - np.exp(-k))
    if kind == "power":
        return 1.0 - np.power(z, p)
    if kind == "log":
        return np.log1p(c * (1.0 - z)) / np.log1p(c)
    raise ValueError(f"unknown bonus kind: {kind}")


def vector_shared_reward(
    e_scaled: np.ndarray,
    du_scaled: np.ndarray,
    y_sp_phys: np.ndarray,
    reward_params: dict,
    dy_phys: np.ndarray,
) -> np.ndarray:
    e_scaled = np.asarray(e_scaled, float)
    du_scaled = np.asarray(du_scaled, float)
    y_sp_phys = np.asarray(y_sp_phys, float)

    k_rel = np.asarray(reward_params["k_rel"], float).reshape(1, -1)
    band_floor_phys = np.asarray(reward_params["band_floor_phys"], float).reshape(1, -1)
    q_diag = np.asarray(reward_params["Q_diag"], float).reshape(1, -1)
    r_diag = np.asarray(reward_params["R_diag"], float).reshape(1, -1)

    band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor_phys)
    band_scaled = band_phys / np.maximum(np.asarray(dy_phys, float).reshape(1, -1), 1.0e-12)
    tau_scaled = float(reward_params["tau_frac"]) * band_scaled

    abs_e = np.abs(e_scaled)
    s_i = sigmoid((band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12))
    gate = str(reward_params["gate"])
    if gate == "prod":
        w_in = np.prod(s_i, axis=1)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=1)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])
    else:
        raise ValueError(f"unknown gate: {gate}")

    err_quad = np.sum(q_diag * e_scaled**2, axis=1)
    lam_in = float(reward_params["lam_in"])
    err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)
    move = np.sum(r_diag * du_scaled**2, axis=1)

    slope_at_edge = 2.0 * q_diag * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    gamma_out = float(reward_params["gamma_out"])
    gamma_in = float(reward_params["gamma_in"])
    lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow, axis=1)
    lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag, axis=1)

    qb2 = q_diag * band_scaled**2
    z = abs_e / np.maximum(band_scaled, 1.0e-12)
    phi = bonus_phi(
        z,
        kind=str(reward_params["bonus_kind"]),
        k=float(reward_params["bonus_k"]),
        p=float(reward_params["bonus_p"]),
        c=float(reward_params["bonus_c"]),
    )
    beta = float(reward_params["beta"])
    bonus = w_in * beta * np.sum(qb2 * phi, axis=1)
    reward_scale = float(reward_params["reward_scale"])
    return (-(err_eff + move + lin_out + lin_in) + bonus) * reward_scale


def make_prototype_reward(bundle: dict, steady_states: dict):
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_ss_scaled = physical_to_scaled_abs(np.asarray(steady_states["y_ss"], float), data_min[2:], data_max[2:])
    params, reward_fn = make_reward_fn_prototype_legacy(
        data_min=data_min,
        data_max=data_max,
        n_inputs=2,
        y_ss_scaled=y_ss_scaled,
        Q_diag=np.asarray(bundle["reward_params"]["Q_diag"], float),
        R_diag=np.asarray(bundle["reward_params"]["R_diag"], float),
        bonus_A=float(bundle["reward_params"]["bonus_A"]),
        bonus_B=float(bundle["reward_params"]["bonus_B"]),
    )
    return params, reward_fn


def rewards_from_trajectories_prototype(
    y_phys: np.ndarray,
    u_phys: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
    reward_fn,
) -> np.ndarray:
    y_phys = np.asarray(y_phys, float)
    u_phys = np.asarray(u_phys, float)
    y_sp_scaled = np.asarray(y_sp_scaled, float)

    ss_u_scaled = physical_to_scaled_abs(np.asarray(steady_states["ss_inputs"], float), data_min[:2], data_max[:2])
    ss_y_scaled = physical_to_scaled_abs(np.asarray(steady_states["y_ss"], float), data_min[2:], data_max[2:])

    u_scaled = physical_to_scaled_abs(u_phys, data_min[:2], data_max[:2])
    u_dev = u_scaled - ss_u_scaled.reshape(1, -1)
    du = np.vstack([np.zeros((1, 2), dtype=float), np.diff(u_dev, axis=0)])

    y_scaled = physical_to_scaled_abs(y_phys[1:], data_min[2:], data_max[2:])
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    e_scaled = y_dev - y_sp_scaled

    y_sp_phys = (y_sp_scaled + ss_y_scaled.reshape(1, -1)) * (data_max[2:] - data_min[2:]).reshape(
        1, -1
    ) + data_min[2:].reshape(1, -1)
    return np.asarray([reward_fn(e_scaled[i], du[i], y_sp_phys=y_sp_phys[i]) for i in range(len(du))], float)


def rewards_from_trajectories_shared(
    y_phys: np.ndarray,
    u_phys: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
    reward_params: dict,
) -> np.ndarray:
    y_phys = np.asarray(y_phys, float)
    u_phys = np.asarray(u_phys, float)
    y_sp_scaled = np.asarray(y_sp_scaled, float)

    n_inputs = u_phys.shape[1]
    ss_u_phys = np.asarray(steady_states["ss_inputs"], float)
    ss_y_phys = np.asarray(steady_states["y_ss"], float)
    ss_u_scaled = physical_to_scaled_abs(ss_u_phys, data_min[:n_inputs], data_max[:n_inputs])
    ss_y_scaled = physical_to_scaled_abs(ss_y_phys, data_min[n_inputs:], data_max[n_inputs:])

    u_scaled = physical_to_scaled_abs(u_phys, data_min[:n_inputs], data_max[:n_inputs])
    u_dev = u_scaled - ss_u_scaled.reshape(1, -1)
    du = np.vstack([np.zeros((1, n_inputs), dtype=float), np.diff(u_dev, axis=0)])

    y_scaled = physical_to_scaled_abs(y_phys[1:], data_min[n_inputs:], data_max[n_inputs:])
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    e_scaled = y_dev - y_sp_scaled
    y_sp_phys = (y_sp_scaled + ss_y_scaled.reshape(1, -1)) * (data_max[n_inputs:] - data_min[n_inputs:]).reshape(
        1, -1
    ) + data_min[n_inputs:].reshape(1, -1)
    dy_phys = data_max[n_inputs:] - data_min[n_inputs:]
    return vector_shared_reward(e_scaled, du, y_sp_phys, reward_params, dy_phys)


def episode_mean(series: np.ndarray) -> np.ndarray:
    return np.asarray(series, float).reshape(N_EPISODES, STEPS_PER_EPISODE).mean(axis=1)


def window_mean(series: np.ndarray, width: int = WINDOW) -> np.ndarray:
    series = np.asarray(series, float)
    out = []
    for start in range(0, len(series), width):
        out.append(float(series[start : min(start + width, len(series))].mean()))
    return np.asarray(out, float)


def window_labels(width: int = WINDOW) -> list[str]:
    labels = []
    for start in range(0, N_EPISODES, width):
        stop = min(start + width, N_EPISODES)
        labels.append(f"{start + 1}-{stop}")
    return labels


def compute_episode_mae(y_phys: np.ndarray, y_sp_scaled: np.ndarray, steady_states: dict, data_min, data_max):
    ss_y_scaled = physical_to_scaled_abs(
        np.asarray(steady_states["y_ss"], float), np.asarray(data_min, float)[2:], np.asarray(data_max, float)[2:]
    )
    y_scaled = physical_to_scaled_abs(
        np.asarray(y_phys, float)[1:], np.asarray(data_min, float)[2:], np.asarray(data_max, float)[2:]
    )
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    e = (y_dev - np.asarray(y_sp_scaled, float)).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)
    return np.mean(np.abs(e), axis=1)


def episode_input_movement(u_phys: np.ndarray) -> np.ndarray:
    u_ep = np.asarray(u_phys, float).reshape(N_EPISODES, STEPS_PER_EPISODE, -1)
    du = np.diff(u_ep, axis=1, prepend=u_ep[:, :1, :])
    return np.mean(np.linalg.norm(du, axis=2), axis=1)


def summarize_comparison(
    label: str,
    y_markov: np.ndarray,
    u_markov: np.ndarray,
    y_ref: np.ndarray,
    u_ref: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
    prototype_reward_fn,
    shared_reward_params: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    proto_markov = rewards_from_trajectories_prototype(
        y_markov, u_markov, y_sp_scaled, steady_states, data_min, data_max, prototype_reward_fn
    )
    proto_ref = rewards_from_trajectories_prototype(
        y_ref, u_ref, y_sp_scaled, steady_states, data_min, data_max, prototype_reward_fn
    )
    shared_markov = rewards_from_trajectories_shared(
        y_markov, u_markov, y_sp_scaled, steady_states, data_min, data_max, shared_reward_params
    )
    shared_ref = rewards_from_trajectories_shared(
        y_ref, u_ref, y_sp_scaled, steady_states, data_min, data_max, shared_reward_params
    )

    proto_delta = episode_mean(proto_markov - proto_ref)
    shared_delta = episode_mean(shared_markov - shared_ref)
    mae_markov = compute_episode_mae(y_markov, y_sp_scaled, steady_states, data_min, data_max)
    mae_ref = compute_episode_mae(y_ref, y_sp_scaled, steady_states, data_min, data_max)
    move_markov = episode_input_movement(u_markov)
    move_ref = episode_input_movement(u_ref)

    row = {
        "label": label,
        "proto_reward_delta_mean": float(proto_delta.mean()),
        "proto_reward_delta_last20": float(proto_delta[-20:].mean()),
        "proto_reward_better_frac": float((proto_delta > 0.0).mean()),
        "shared_reward_delta_mean": float(shared_delta.mean()),
        "shared_reward_delta_last20": float(shared_delta[-20:].mean()),
        "shared_reward_better_frac": float((shared_delta > 0.0).mean()),
        "output1_mae_delta_mean": float((mae_markov - mae_ref).mean(axis=0)[0]),
        "output2_mae_delta_mean": float((mae_markov - mae_ref).mean(axis=0)[1]),
        "output1_mae_delta_last20": float((mae_markov[-20:] - mae_ref[-20:]).mean(axis=0)[0]),
        "output2_mae_delta_last20": float((mae_markov[-20:] - mae_ref[-20:]).mean(axis=0)[1]),
        "input_move_delta_mean": float((move_markov - move_ref).mean()),
        "input_move_delta_last20": float((move_markov[-20:] - move_ref[-20:]).mean()),
    }
    return row, proto_delta, shared_delta


def source_summary(bundle: dict, label: str) -> dict:
    source = np.asarray(bundle["rl_action_source_log"], int)
    z_exec = np.asarray(bundle["z_executed_log"], float)
    z_abs = np.abs(z_exec)
    bound = float(np.nanmax(z_abs))
    return {
        "label": label,
        "td3_fraction": float((source == 2).mean()),
        "ls_fraction": float((source == 3).mean()),
        "nominal_fraction": float((source == 4).mean()),
        "warm_fraction": float((source == 1).mean()),
        "z_bound_inferred": bound,
        "z_sat_any98_fraction": float((z_abs.max(axis=1) >= 0.98 * max(bound, 1.0e-12)).mean()),
        "z_norm_mean": float(np.linalg.norm(z_exec, axis=1).mean()),
        "prediction_score_mean": float(np.nanmean(np.asarray(bundle["s_pred_log"], float))),
        "gain_drift_mean": float(np.nanmean(np.asarray(bundle["gain_drift_log"], float))),
    }


def direct_gap_summary(label: str, y_a: np.ndarray, y_b: np.ndarray, u_a: np.ndarray, u_b: np.ndarray) -> dict:
    y_diff = np.asarray(y_a, float) - np.asarray(y_b, float)
    u_diff = np.asarray(u_a, float) - np.asarray(u_b, float)
    return {
        "label": label,
        "output_rmse_1": float(np.sqrt(np.mean(y_diff[:, 0] ** 2))),
        "output_rmse_2": float(np.sqrt(np.mean(y_diff[:, 1] ** 2))),
        "input_rmse_1": float(np.sqrt(np.mean(u_diff[:, 0] ** 2))),
        "input_rmse_2": float(np.sqrt(np.mean(u_diff[:, 1] ** 2))),
        "max_output_abs_diff": float(np.max(np.abs(y_diff))),
        "max_input_abs_diff": float(np.max(np.abs(u_diff))),
    }


def plot_reward_window_compare(proto_deltas: dict[str, np.ndarray], out_path: Path):
    xs = np.arange(len(window_labels()))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    series_meta = [
        ("Unified latest vs canonical MPC", proto_deltas["unified_vs_canonical"], "#1f77b4"),
        ("Legacy latest vs its own nominal", proto_deltas["legacy_vs_nominal"], "#ff7f0e"),
        ("Original legacy run vs its own nominal", proto_deltas["legacy_old_vs_nominal"], "#2ca02c"),
    ]
    for label, values, color in series_meta:
        ax.plot(xs, window_mean(values), marker="o", linewidth=2.2, color=color, label=label)
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(window_labels(), rotation=30)
    ax.set_ylabel("Prototype reward delta")
    ax.set_xlabel("Episode window")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_legacy_baseline_choice(proto_deltas: dict[str, np.ndarray], shared_deltas: dict[str, np.ndarray], out_path: Path):
    xs = np.arange(len(window_labels()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    left = axes[0]
    left.plot(xs, window_mean(proto_deltas["legacy_vs_nominal"]), marker="o", linewidth=2.2, color="#ff7f0e", label="Legacy latest vs own nominal")
    left.plot(xs, window_mean(proto_deltas["legacy_vs_canonical"]), marker="s", linewidth=2.0, color="#d62728", label="Legacy latest vs canonical MPC")
    left.plot(xs, window_mean(proto_deltas["unified_vs_canonical"]), marker="^", linewidth=2.0, color="#1f77b4", label="Unified latest vs canonical MPC")
    left.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    left.set_title("Prototype reward")
    left.set_xticks(xs)
    left.set_xticklabels(window_labels(), rotation=30)
    left.set_ylabel("Reward delta")
    left.grid(alpha=0.25)
    left.legend(loc="best", fontsize=8)

    right = axes[1]
    right.plot(xs, window_mean(shared_deltas["legacy_vs_nominal"]), marker="o", linewidth=2.2, color="#ff7f0e", label="Legacy latest vs own nominal")
    right.plot(xs, window_mean(shared_deltas["legacy_vs_canonical"]), marker="s", linewidth=2.0, color="#d62728", label="Legacy latest vs canonical MPC")
    right.plot(xs, window_mean(shared_deltas["unified_vs_canonical"]), marker="^", linewidth=2.0, color="#1f77b4", label="Unified latest vs canonical MPC")
    right.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    right.set_title("Shared reward")
    right.set_xticks(xs)
    right.set_xticklabels(window_labels(), rotation=30)
    right.set_ylabel("Reward delta")
    right.grid(alpha=0.25)
    right.legend(loc="best", fontsize=8)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tail_overlay(unified: dict, legacy: dict, legacy_old: dict, baseline: dict, out_path: Path):
    t = np.arange(STEPS_PER_EPISODE) * float(baseline["delta_t"])
    last = -1

    unified_y = np.asarray(unified["y"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[last]
    baseline_y = np.asarray(baseline["y_mpc"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[last]
    legacy_y = np.asarray(legacy["y_markov"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[last]
    legacy_nom = np.asarray(legacy["y_nominal"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[last]
    legacy_old_y = np.asarray(legacy_old["y_markov"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[last]
    y_sp_phys = (
        np.asarray(legacy["y_sp"], float).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[last]
        + physical_to_scaled_abs(np.asarray(baseline["steady_states"]["y_ss"], float), baseline["data_min"][2:], baseline["data_max"][2:]).reshape(1, -1)
    ) * (np.asarray(baseline["data_max"], float)[2:] - np.asarray(baseline["data_min"], float)[2:]).reshape(1, -1) + np.asarray(
        baseline["data_min"], float
    )[2:].reshape(1, -1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    for row in range(2):
        axes[row, 0].plot(t, y_sp_phys[:, row], color="0.2", linestyle="--", linewidth=1.5, label="Setpoint")
        axes[row, 0].plot(t, baseline_y[:, row], color="#7f7f7f", linewidth=1.8, label="Canonical MPC")
        axes[row, 0].plot(t, unified_y[:, row], color="#1f77b4", linewidth=2.1, label="Unified latest")
        axes[row, 0].set_ylabel(f"Output {row + 1}")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(t, y_sp_phys[:, row], color="0.2", linestyle="--", linewidth=1.5, label="Setpoint")
        axes[row, 1].plot(t, legacy_nom[:, row], color="#7f7f7f", linewidth=1.8, label="Legacy nominal")
        axes[row, 1].plot(t, legacy_old_y[:, row], color="#2ca02c", linewidth=1.6, label="Original legacy")
        axes[row, 1].plot(t, legacy_y[:, row], color="#ff7f0e", linewidth=2.1, label="Legacy latest")
        axes[row, 1].grid(alpha=0.25)

    axes[0, 0].set_title("Unified latest vs canonical baseline")
    axes[0, 1].set_title("Restored legacy vs original legacy path")
    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_action_and_z_summary(rows: list[dict], out_path: Path):
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    bottom = np.zeros(len(labels), dtype=float)
    for key, color, display in [
        ("warm_fraction", "#c7c7c7", "Warm-start LS"),
        ("td3_fraction", "#1f77b4", "TD3 accepted"),
        ("ls_fraction", "#ff7f0e", "LS fallback"),
        ("nominal_fraction", "#d62728", "Nominal fallback"),
    ]:
        values = np.asarray([row[key] for row in rows], float)
        axes[0].bar(x, values, bottom=bottom, color=color, label=display)
        bottom += values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Fraction of control steps")
    axes[0].set_title("Action-source mix")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x - 0.18, [row["z_sat_any98_fraction"] for row in rows], width=0.36, color="#9467bd", label="Any-z saturation")
    axes[1].bar(x + 0.18, [row["z_norm_mean"] for row in rows], width=0.36, color="#8c564b", label="Mean ||z||")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_title("Correction magnitude")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].bar(x - 0.18, [row["prediction_score_mean"] for row in rows], width=0.36, color="#2ca02c", label="Mean prediction score")
    axes[2].bar(x + 0.18, [row["gain_drift_mean"] for row in rows], width=0.36, color="#d62728", label="Mean gain drift")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20)
    axes[2].set_title("Filter diagnostics")
    axes[2].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.2, axis="y")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_behavioral_distance(rows: list[dict], out_path: Path):
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(x - 0.18, [row["output_rmse_1"] for row in rows], width=0.36, color="#1f77b4", label="Output 1 RMSE")
    axes[0].bar(x + 0.18, [row["output_rmse_2"] for row in rows], width=0.36, color="#ff7f0e", label="Output 2 RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_title("Run-to-run output distance")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x - 0.18, [row["input_rmse_1"] for row in rows], width=0.36, color="#2ca02c", label="Input 1 RMSE")
    axes[1].bar(x + 0.18, [row["input_rmse_2"] for row in rows], width=0.36, color="#d62728", label="Input 2 RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_title("Run-to-run input distance")
    axes[1].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.2, axis="y")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    unified = load_pickle(UNIFIED_LATEST)
    legacy = load_pickle(LEGACY_LATEST)
    legacy_old = load_pickle(LEGACY_OLD)
    canonical = load_pickle(CANONICAL_BASELINE)
    shared_ref = load_pickle(SHARED_REWARD_REFERENCE)

    steady_states = canonical["steady_states"]
    data_min = np.asarray(canonical["data_min"], float)
    data_max = np.asarray(canonical["data_max"], float)
    shared_reward_params = shared_ref["reward_params"]
    _proto_params, proto_reward_fn = make_prototype_reward(unified, steady_states)

    comparison_rows = []
    proto_deltas = {}
    shared_deltas = {}

    for key, args in [
        (
            "unified_vs_canonical",
            (
                "Unified latest vs canonical MPC",
                unified["y"],
                unified["u"],
                canonical["y_mpc"],
                canonical["u_mpc"],
                unified["y_sp"],
            ),
        ),
        (
            "legacy_vs_nominal",
            (
                "Legacy latest vs own nominal",
                legacy["y_markov"],
                legacy["u_markov"],
                legacy["y_nominal"],
                legacy["u_nominal"],
                legacy["y_sp"],
            ),
        ),
        (
            "legacy_vs_canonical",
            (
                "Legacy latest vs canonical MPC",
                legacy["y_markov"],
                legacy["u_markov"],
                canonical["y_mpc"],
                canonical["u_mpc"],
                legacy["y_sp"],
            ),
        ),
        (
            "legacy_old_vs_nominal",
            (
                "Original legacy vs own nominal",
                legacy_old["y_markov"],
                legacy_old["u_markov"],
                legacy_old["y_nominal"],
                legacy_old["u_nominal"],
                legacy_old["y_sp"],
            ),
        ),
    ]:
        row, proto_delta, shared_delta = summarize_comparison(
            *args,
            steady_states=steady_states,
            data_min=data_min,
            data_max=data_max,
            prototype_reward_fn=proto_reward_fn,
            shared_reward_params=shared_reward_params,
        )
        comparison_rows.append(row)
        proto_deltas[key] = proto_delta
        shared_deltas[key] = shared_delta

    source_rows = [
        source_summary(unified, "Unified latest"),
        source_summary(legacy, "Legacy latest"),
        source_summary(legacy_old, "Original legacy"),
    ]

    gap_rows = [
        direct_gap_summary(
            "Legacy latest vs original legacy",
            legacy["y_markov"],
            legacy_old["y_markov"],
            legacy["u_markov"],
            legacy_old["u_markov"],
        ),
        direct_gap_summary(
            "Unified latest vs legacy latest",
            unified["y"],
            legacy["y_markov"],
            unified["u"],
            legacy["u_markov"],
        ),
    ]

    write_csv(OUT_DIR / "comparison_summary.csv", comparison_rows)
    write_csv(OUT_DIR / "source_summary.csv", source_rows)
    write_csv(OUT_DIR / "behavioral_distance_summary.csv", gap_rows)

    plot_reward_window_compare(proto_deltas, OUT_DIR / "prototype_reward_window_compare.png")
    plot_legacy_baseline_choice(proto_deltas, shared_deltas, OUT_DIR / "baseline_choice_effect.png")
    plot_tail_overlay(unified, legacy, legacy_old, canonical, OUT_DIR / "tail_output_overlay.png")
    plot_action_and_z_summary(source_rows, OUT_DIR / "action_source_and_z_summary.png")
    plot_behavioral_distance(gap_rows, OUT_DIR / "behavioral_distance_summary.png")

    print("Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()
