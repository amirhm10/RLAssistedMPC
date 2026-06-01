"""Analyze latest distillation weight-multiplier failure mode.

This script reads saved result bundles only. It recomputes rewards for all
saved distillation weight runs under the current reward definition so older
TD3/SAC runs can be compared fairly after the May 30 temperature-weight change.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS

OUT_DIR = ROOT / "report" / "figures" / "distillation_weights_latest_20260601"

TD3_ROOT = ROOT / "Distillation" / "Results" / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
SAC_ROOT = ROOT / "Distillation" / "Results" / "distillation_weights_sac_disturb_fluctuation_mismatch_unified"
BASELINE_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"

INPUT_BOUNDS = {
    "u_min": np.array([300000.0, 100.0], dtype=float),
    "u_max": np.array([460000.0, 150.0], dtype=float),
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def minmax_scale(data, min_val, max_val):
    return (np.asarray(data, float) - np.asarray(min_val, float)) / np.maximum(
        np.asarray(max_val, float) - np.asarray(min_val, float),
        1.0e-12,
    )


def reverse_minmax(data, min_val, max_val):
    return np.asarray(data, float) * (
        np.asarray(max_val, float) - np.asarray(min_val, float)
    ) + np.asarray(min_val, float)


def y_sp_phys(bundle: dict) -> np.ndarray:
    y_sp = np.asarray(bundle["y_sp"], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = minmax_scale(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    return reverse_minmax(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def step_error_phys(bundle: dict):
    y_line = np.asarray(bundle.get("y_line_full", bundle.get("y")), float)
    ysp = y_sp_phys(bundle)
    n = min(y_line.shape[0] - 1, ysp.shape[0])
    y_step = y_line[1 : n + 1, :]
    ysp_step = ysp[:n, :]
    return y_step - ysp_step, y_step, ysp_step


def vectorized_current_reward(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    dy_out = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1.0e-12)

    delta_y = np.asarray(bundle["delta_y_storage"], float)
    delta_u = np.asarray(bundle["delta_u_storage"], float)
    ysp = y_sp_phys(bundle)
    n = min(delta_y.shape[0], delta_u.shape[0], ysp.shape[0])
    delta_y = delta_y[:n, :]
    delta_u = delta_u[:n, :]
    ysp = ysp[:n, :]

    q_diag = np.asarray(RL_REWARD_DEFAULTS["Q_diag"], float)
    r_diag = np.asarray(RL_REWARD_DEFAULTS["R_diag"], float)
    k_rel = np.asarray(RL_REWARD_DEFAULTS["k_rel"], float)
    floor_phys = np.asarray(RL_REWARD_DEFAULTS["band_floor_phys"], float)
    band_phys = np.maximum(k_rel.reshape(1, -1) * np.abs(ysp), floor_phys.reshape(1, -1))
    band_scaled = band_phys / dy_out.reshape(1, -1)
    tau_scaled = float(RL_REWARD_DEFAULTS["tau_frac"]) * band_scaled

    abs_e = np.abs(delta_y)
    sigmoid_arg = np.clip((band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12), -60.0, 60.0)
    s_i = 1.0 / (1.0 + np.exp(-sigmoid_arg))
    gate = str(RL_REWARD_DEFAULTS["gate"]).lower()
    if gate == "prod":
        w_in = np.prod(s_i, axis=1)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=1)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])
    else:
        raise ValueError(f"Unsupported reward gate: {gate}")

    err_quad = np.sum(q_diag.reshape(1, -1) * delta_y * delta_y, axis=1)
    move = np.sum(r_diag.reshape(1, -1) * delta_u * delta_u, axis=1)
    err_eff = ((1.0 - w_in) + w_in * float(RL_REWARD_DEFAULTS["lam_in"])) * err_quad

    slope_at_edge = 2.0 * q_diag.reshape(1, -1) * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    lin_out = (1.0 - w_in) * np.sum(float(RL_REWARD_DEFAULTS["gamma_out"]) * slope_at_edge * overflow, axis=1)
    lin_in = w_in * np.sum(float(RL_REWARD_DEFAULTS["gamma_in"]) * slope_at_edge * inside_mag, axis=1)

    z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
    bonus_kind = str(RL_REWARD_DEFAULTS["bonus_kind"]).lower()
    if bonus_kind != "exp":
        raise ValueError(f"This analysis script currently supports exp bonus only, got {bonus_kind}.")
    k_bonus = float(RL_REWARD_DEFAULTS["bonus_k"])
    phi = (np.exp(-k_bonus * z) - np.exp(-k_bonus)) / (1.0 - np.exp(-k_bonus))
    qb2 = q_diag.reshape(1, -1) * band_scaled * band_scaled
    bonus = w_in * float(RL_REWARD_DEFAULTS["beta"]) * np.sum(qb2 * phi, axis=1)

    rewards = (-(err_eff + move + lin_out + lin_in) + bonus) * float(RL_REWARD_DEFAULTS["reward_scale"])
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    n_episodes = rewards.size // episode_len
    avg = rewards[: n_episodes * episode_len].reshape(n_episodes, episode_len).mean(axis=1)
    return rewards, avg


def step_slice(bundle: dict, episodes: int = 20) -> slice:
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    return slice(max(0, nfe - episodes * episode_len), nfe)


def window_slice(bundle: dict, start_episode: int, end_episode: int) -> slice:
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    return slice((start_episode - 1) * episode_len, end_episode * episode_len)


def finite_mean(values) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def action_stats(bundle: dict, sl: slice, prefix: str) -> dict:
    out: dict[str, object] = {}
    for key in (
        "policy_action_raw_log",
        "weight_requested_action_raw_log",
        "weight_post_handoff_action_raw_log",
        "weight_post_cap_action_raw_log",
        "weight_executed_action_raw_log",
    ):
        value = bundle.get(key)
        if value is None:
            continue
        arr = np.asarray(value, float)[sl]
        if arr.size == 0:
            continue
        finite = np.all(np.isfinite(arr), axis=1)
        if not np.any(finite):
            continue
        arr = arr[finite]
        out[f"{prefix}_{key}_mean_abs"] = float(np.mean(np.abs(arr)))
        out[f"{prefix}_{key}_sat98"] = float(np.mean(np.abs(arr) >= 0.98))
        out[f"{prefix}_{key}_mean"] = np.mean(arr, axis=0).tolist()
    return out


def weight_stats(bundle: dict, sl: slice, prefix: str) -> dict:
    out: dict[str, object] = {}
    weights = bundle.get("weight_log")
    if weights is None:
        return out
    weights = np.asarray(weights, float)[sl]
    if weights.size == 0:
        return out
    out[f"{prefix}_weight_q1_mean"] = float(np.mean(weights[:, 0]))
    out[f"{prefix}_weight_q2_mean"] = float(np.mean(weights[:, 1]))
    out[f"{prefix}_weight_r1_mean"] = float(np.mean(weights[:, 2]))
    out[f"{prefix}_weight_r2_mean"] = float(np.mean(weights[:, 3]))
    out[f"{prefix}_weight_mean"] = np.mean(weights, axis=0).tolist()
    out[f"{prefix}_weight_q05"] = np.quantile(weights, 0.05, axis=0).tolist()
    out[f"{prefix}_weight_q95"] = np.quantile(weights, 0.95, axis=0).tolist()
    sat = bundle.get("weight_multiplier_saturation_log")
    if sat is not None:
        out[f"{prefix}_weight_boundary_frac"] = finite_mean(np.asarray(sat, float)[sl])
    return out


def tracking_stats(bundle: dict, sl: slice, prefix: str) -> dict:
    err, _y, ysp = step_error_phys(bundle)
    stop = min(sl.stop, err.shape[0])
    use = slice(sl.start, stop)
    e = err[use, :]
    ysp_use = ysp[use, :]
    band = np.maximum(
        np.asarray(RL_REWARD_DEFAULTS["k_rel"], float).reshape(1, -1) * np.abs(ysp_use),
        np.asarray(RL_REWARD_DEFAULTS["band_floor_phys"], float).reshape(1, -1),
    )
    abs_e = np.abs(e)
    return {
        f"{prefix}_comp_mae": float(np.mean(abs_e[:, 0])),
        f"{prefix}_temp_mae": float(np.mean(abs_e[:, 1])),
        f"{prefix}_comp_rmse": float(np.sqrt(np.mean(e[:, 0] ** 2))),
        f"{prefix}_temp_rmse": float(np.sqrt(np.mean(e[:, 1] ** 2))),
        f"{prefix}_band_norm_mae": float(np.mean(abs_e / np.maximum(band, 1.0e-12))),
        f"{prefix}_outside_band_frac": float(np.mean(abs_e > band)),
    }


def input_stats(bundle: dict, sl: slice, prefix: str) -> dict:
    u = np.asarray(bundle.get("u_step_full", bundle.get("u")), float)
    stop = min(sl.stop, u.shape[0])
    use = slice(sl.start, stop)
    u_use = u[use, :]
    du = np.diff(u, axis=0, prepend=u[[0], :])[use, :]
    sat_tol = np.array([100.0, 0.05], dtype=float)
    sat = (u_use <= INPUT_BOUNDS["u_min"] + sat_tol) | (u_use >= INPUT_BOUNDS["u_max"] - sat_tol)
    return {
        f"{prefix}_du_abs_mean": float(np.mean(np.abs(du))),
        f"{prefix}_du_l2_mean": float(np.mean(np.linalg.norm(du, axis=1))),
        f"{prefix}_input_saturation_frac": float(np.mean(sat)),
        f"{prefix}_reflux_min": float(np.min(u_use[:, 0])),
        f"{prefix}_reflux_max": float(np.max(u_use[:, 0])),
        f"{prefix}_reboiler_min": float(np.min(u_use[:, 1])),
        f"{prefix}_reboiler_max": float(np.max(u_use[:, 1])),
    }


def summarize_bundle(kind: str, run: str, path: Path, bundle: dict) -> dict:
    _step_rewards, avg_current = vectorized_current_reward(bundle)
    stored = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
    tail = step_slice(bundle, 20)
    warm = window_slice(bundle, 1, 10)
    handoff = window_slice(bundle, 11, 20)
    early_live = window_slice(bundle, 21, 40)

    row: dict[str, object] = {
        "kind": kind,
        "run": run,
        "path": str(path.relative_to(ROOT)),
        "stored_tail20_reward": float(np.mean(stored[-20:])) if stored.size else float("nan"),
        "current_reward_mean": float(np.mean(avg_current)),
        "current_tail20_reward": float(np.mean(avg_current[-20:])),
        "current_tail10_reward": float(np.mean(avg_current[-10:])),
        "current_final_reward": float(avg_current[-1]),
        "current_first20_min_reward": float(np.min(avg_current[:20])),
        "current_warm10_mean_reward": float(np.mean(avg_current[:10])),
        "current_handoff10_mean_reward": float(np.mean(avg_current[10:20])),
        "current_early_live20_mean_reward": float(np.mean(avg_current[20:40])),
    }
    row.update(tracking_stats(bundle, tail, "tail20"))
    row.update(input_stats(bundle, tail, "tail20"))
    row.update(weight_stats(bundle, tail, "tail20"))
    row.update(action_stats(bundle, tail, "tail20"))

    for name, sl in (("warm10", warm), ("handoff10", handoff), ("early_live20", early_live)):
        row.update(weight_stats(bundle, sl, name))
        row.update(action_stats(bundle, sl, name))

    low = np.asarray(bundle.get("low_coef", []), float)
    high = np.asarray(bundle.get("high_coef", []), float)
    if low.size and high.size:
        row["low_coef"] = low.tolist()
        row["high_coef"] = high.tolist()
        row["identity_raw_action"] = (2.0 * (1.0 - low) / np.maximum(high - low, 1.0e-12) - 1.0).tolist()

    for key in (
        "weight_cap_projection_active_log",
        "weight_fallback_reason_log",
        "weight_probation_active_log",
        "release_gate_blocked_log",
        "release_gate_pass_log",
        "bc_active_log",
        "bc_weight_log",
        "bc_policy_nominal_distance_log",
        "bc_handoff_authority_log",
        "td3_authority_ramp_projection_active_log",
        "td3_authority_ramp_cap_log",
    ):
        value = bundle.get(key)
        if value is None:
            continue
        arr = np.asarray(value, float)
        if arr.size >= tail.stop:
            tail_arr = arr[tail]
            if key == "weight_fallback_reason_log":
                row[f"tail20_{key}_frac"] = float(np.mean(tail_arr != 0))
            else:
                row[f"tail20_{key}_mean"] = finite_mean(tail_arr)

    selected = bundle.get("weight_shadow_selected_objective_log")
    identity = bundle.get("weight_shadow_identity_objective_log")
    if selected is not None and identity is not None:
        diff = np.asarray(selected, float)[tail] - np.asarray(identity, float)[tail]
        row["tail20_shadow_selected_minus_identity_objective"] = finite_mean(diff)
    first_move_delta = bundle.get("weight_shadow_first_move_delta_norm_log")
    if first_move_delta is not None:
        row["tail20_shadow_first_move_delta_norm_mean"] = finite_mean(np.asarray(first_move_delta, float)[tail])

    row["_avg_current"] = avg_current
    return row


def episode_profile(kind: str, run: str, bundle: dict) -> pd.DataFrame:
    _step_rewards, avg_current = vectorized_current_reward(bundle)
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    n_episodes = avg_current.size
    rows = []
    weight_log = bundle.get("weight_log")
    req = bundle.get("weight_requested_action_raw_log")
    policy = bundle.get("policy_action_raw_log")
    cap = bundle.get("weight_cap_projection_active_log")
    fallback = bundle.get("weight_fallback_reason_log")
    for ep in range(n_episodes):
        sl = slice(ep * episode_len, (ep + 1) * episode_len)
        row = {
            "kind": kind,
            "run": run,
            "episode": ep + 1,
            "current_reward": float(avg_current[ep]),
        }
        if weight_log is not None:
            w = np.asarray(weight_log, float)[sl, :]
            row.update(
                {
                    "q1": float(np.mean(w[:, 0])),
                    "q2": float(np.mean(w[:, 1])),
                    "r1": float(np.mean(w[:, 2])),
                    "r2": float(np.mean(w[:, 3])),
                }
            )
        if req is not None:
            a = np.asarray(req, float)[sl, :]
            row["requested_action_sat98"] = float(np.mean(np.abs(a) >= 0.98))
        if policy is not None:
            a = np.asarray(policy, float)[sl, :]
            row["policy_action_sat98"] = float(np.mean(np.abs(a) >= 0.98))
        if cap is not None:
            row["cap_projection_frac"] = finite_mean(np.asarray(cap, float)[sl])
        if fallback is not None:
            row["fallback_frac"] = float(np.mean(np.asarray(fallback, int)[sl] != 0))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_runs() -> tuple[pd.DataFrame, dict[str, dict], pd.DataFrame]:
    bundles: dict[str, dict] = {}
    rows: list[dict] = []
    profiles: list[pd.DataFrame] = []

    baseline = load_pickle(BASELINE_PATH)
    base_row = summarize_bundle("baseline", "OF-MPC", BASELINE_PATH, baseline)
    base_row["label"] = "OF-MPC"
    rows.append(base_row)
    bundles["baseline:OF-MPC"] = baseline
    profiles.append(episode_profile("baseline", "OF-MPC", baseline))

    for kind, root in (("td3", TD3_ROOT), ("sac", SAC_ROOT)):
        if not root.exists():
            continue
        for path in sorted(root.glob("*/input_data.pkl")):
            run = path.parent.name
            bundle = load_pickle(path)
            row = summarize_bundle(kind, run, path, bundle)
            row["label"] = f"{kind.upper()} {run}"
            rows.append(row)
            bundles[f"{kind}:{run}"] = bundle
            profiles.append(episode_profile(kind, run, bundle))

    summary = pd.DataFrame(rows)
    profile = pd.concat(profiles, ignore_index=True)
    avg_map = {f"{row['kind']}:{row['run']}": row.pop("_avg_current") for row in rows}
    for key, avg in avg_map.items():
        bundles.setdefault(key, {})["_avg_current"] = avg
    return summary, bundles, profile


def plot_reward_trajectories(summary: pd.DataFrame, bundles: dict[str, dict], profile: pd.DataFrame) -> None:
    selected = [
        ("baseline", "OF-MPC", "OF-MPC", "black"),
        ("td3", "20260601_155305", "TD3 latest 20260601", "#e45756"),
        ("td3", "20260530_220604", "TD3 May 30", "#4c78a8"),
        ("td3", "20260528_194904", "TD3 May 28 best", "#54a24b"),
        ("td3", "20260521_150600", "TD3 May 21", "#72b7b2"),
        ("sac", "20260518_142138", "SAC May 18", "#f58518"),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    for kind, run, label, color in selected:
        df = profile[(profile["kind"] == kind) & (profile["run"] == run)]
        if df.empty:
            continue
        ax.plot(df["episode"], df["current_reward"], label=label, linewidth=1.5, color=color)
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward recomputed with current reward")
    ax.set_title("Distillation weight runs under the current reward definition")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_current_reward_trajectories.png", dpi=180)
    plt.close(fig)


def plot_tail_ranking(summary: pd.DataFrame) -> None:
    ranking = summary.sort_values("current_tail20_reward", ascending=True)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    colors = ["#999999" if kind == "baseline" else "#4c78a8" for kind in ranking["kind"]]
    ax.barh(ranking["label"], ranking["current_tail20_reward"], color=colors)
    for y, value in enumerate(ranking["current_tail20_reward"]):
        ax.text(value + 0.25, y, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Tail-20 reward recomputed with current reward")
    ax.set_title("All saved distillation weight runs")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail20_reward_ranking.png", dpi=180)
    plt.close(fig)


def plot_tracking(summary: pd.DataFrame) -> None:
    keep = summary[
        summary["run"].isin(
            ["OF-MPC", "20260601_155305", "20260530_220604", "20260528_194904", "20260521_150600", "20260518_142138"]
        )
    ].copy()
    keep["sort_key"] = keep["current_tail20_reward"]
    keep = keep.sort_values("sort_key", ascending=False)
    labels = keep["label"].tolist()
    x = np.arange(len(keep))
    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    width = 0.34
    ax1.bar(x - width / 2, keep["tail20_comp_mae"], width, color="#4c78a8", label="Composition MAE")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, keep["tail20_temp_mae"], width, color="#f58518", label="Temperature MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Tray-24 composition MAE")
    ax2.set_ylabel("Tray-85 temperature MAE")
    ax1.set_title("Tail-20 physical tracking errors")
    ax1.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_tracking_errors.png", dpi=180)
    plt.close(fig)


def plot_latest_action_layers(profile: pd.DataFrame) -> None:
    latest = profile[(profile["kind"] == "td3") & (profile["run"] == "20260601_155305")].copy()
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    axes[0].plot(latest["episode"], latest["current_reward"], color="#e45756", linewidth=1.4)
    axes[0].axhline(0.0, color="0.45", linestyle=":", linewidth=0.9)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Latest TD3 weights: reward and action collapse")
    for col, color in zip(("q1", "q2", "r1", "r2"), ("#4c78a8", "#f58518", "#54a24b", "#b279a2")):
        axes[1].plot(latest["episode"], latest[col], label=col.upper(), linewidth=1.2, color=color)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="identity")
    axes[1].axhline(0.75, color="0.55", linestyle=":", linewidth=0.9, label="lower bound")
    axes[1].set_ylabel("Episode mean multiplier")
    axes[1].legend(ncol=5, fontsize=8)
    axes[2].plot(latest["episode"], latest["policy_action_sat98"], label="clean policy abs action >= 0.98", color="#e45756")
    axes[2].plot(latest["episode"], latest["requested_action_sat98"], label="requested abs action >= 0.98", color="#4c78a8")
    axes[2].plot(latest["episode"], latest["cap_projection_frac"], label="cap projection", color="#54a24b")
    axes[2].plot(latest["episode"], latest["fallback_frac"], label="fallback", color="black")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("Fraction")
    axes[2].set_xlabel("Subepisode")
    axes[2].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.axvline(10, color="0.5", linestyle="--", linewidth=0.9)
        ax.axvline(20, color="0.5", linestyle=":", linewidth=0.9)
        ax.axvline(40, color="0.5", linestyle="-.", linewidth=0.9)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_latest_action_layers.png", dpi=180)
    plt.close(fig)


def plot_latest_vs_may30_multipliers(profile: pd.DataFrame) -> None:
    selected = [
        ("20260601_155305", "Latest TD3", "#e45756"),
        ("20260530_220604", "May 30 TD3", "#4c78a8"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    for run, label, color in selected:
        df = profile[(profile["kind"] == "td3") & (profile["run"] == run)].copy()
        if df.empty:
            continue
        axes[0].plot(df["episode"], df["current_reward"], label=label, color=color, linewidth=1.4)
        axes[1].plot(df["episode"], df["q2"], label=f"{label} Q2", color=color, linewidth=1.2)
        axes[1].plot(df["episode"], df["q1"], label=f"{label} Q1", color=color, linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("Current reward")
    axes[0].set_title("Latest TD3 lost the May 30 temperature-weight behavior")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=0.9)
    axes[1].set_ylabel("Episode mean Q multiplier")
    axes[1].set_xlabel("Subepisode")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_latest_vs_may30_multipliers.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, bundles, profile = collect_runs()

    public_summary = summary.drop(columns=[c for c in summary.columns if c.startswith("_")], errors="ignore")
    public_summary = public_summary.sort_values("current_tail20_reward", ascending=False)
    public_summary.to_csv(OUT_DIR / "weight_run_summary.csv", index=False)

    selected_runs = {
        ("baseline", "OF-MPC"),
        ("td3", "20260601_155305"),
        ("td3", "20260530_220604"),
        ("td3", "20260528_194904"),
        ("td3", "20260521_150600"),
        ("sac", "20260518_142138"),
    }
    selected_profile = profile[
        profile.apply(lambda row: (row["kind"], row["run"]) in selected_runs, axis=1)
    ].copy()
    selected_profile.to_csv(OUT_DIR / "selected_episode_profiles.csv", index=False)

    latest = summary[(summary["kind"] == "td3") & (summary["run"] == "20260601_155305")].iloc[0]
    baseline = summary[(summary["kind"] == "baseline") & (summary["run"] == "OF-MPC")].iloc[0]
    may30 = summary[(summary["kind"] == "td3") & (summary["run"] == "20260530_220604")].iloc[0]
    best = summary[summary["kind"].isin(["td3", "sac"])].sort_values("current_tail20_reward", ascending=False).iloc[0]
    sac_best = summary[summary["kind"] == "sac"].sort_values("current_tail20_reward", ascending=False).iloc[0]

    latest_window = public_summary[
        public_summary["run"].isin(["OF-MPC", "20260601_155305", "20260530_220604", str(best["run"]), str(sac_best["run"])])
    ].copy()
    latest_window.to_csv(OUT_DIR / "selected_tail_metrics.csv", index=False)

    plot_reward_trajectories(summary, bundles, profile)
    plot_tail_ranking(summary)
    plot_tracking(summary)
    plot_latest_action_layers(profile)
    plot_latest_vs_may30_multipliers(profile)

    result = {
        "latest_vs_baseline": {
            "tail20_reward_delta": float(latest["current_tail20_reward"] - baseline["current_tail20_reward"]),
            "final_reward_delta": float(latest["current_final_reward"] - baseline["current_final_reward"]),
            "tail20_comp_mae_ratio": float(latest["tail20_comp_mae"] / baseline["tail20_comp_mae"]),
            "tail20_temp_mae_ratio": float(latest["tail20_temp_mae"] / baseline["tail20_temp_mae"]),
            "tail20_band_mae_ratio": float(latest["tail20_band_norm_mae"] / baseline["tail20_band_norm_mae"]),
        },
        "latest_vs_may30": {
            "tail20_reward_delta": float(latest["current_tail20_reward"] - may30["current_tail20_reward"]),
            "tail20_comp_mae_delta": float(latest["tail20_comp_mae"] - may30["tail20_comp_mae"]),
            "tail20_temp_mae_delta": float(latest["tail20_temp_mae"] - may30["tail20_temp_mae"]),
            "latest_tail20_weight_mean": latest["tail20_weight_mean"],
            "may30_tail20_weight_mean": may30["tail20_weight_mean"],
        },
        "best_historical_current_reward": {
            "kind": str(best["kind"]),
            "run": str(best["run"]),
            "tail20_reward": float(best["current_tail20_reward"]),
            "tail20_comp_mae": float(best["tail20_comp_mae"]),
            "tail20_temp_mae": float(best["tail20_temp_mae"]),
            "tail20_weight_mean": best["tail20_weight_mean"],
        },
        "best_sac_current_reward": {
            "run": str(sac_best["run"]),
            "tail20_reward": float(sac_best["current_tail20_reward"]),
            "tail20_comp_mae": float(sac_best["tail20_comp_mae"]),
            "tail20_temp_mae": float(sac_best["tail20_temp_mae"]),
            "tail20_weight_mean": sac_best["tail20_weight_mean"],
        },
        "reward_defaults": {
            key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
            for key, value in RL_REWARD_DEFAULTS.items()
        },
        "artifacts": {
            "figure_dir": str(OUT_DIR.relative_to(ROOT)),
            "csvs": sorted(p.name for p in OUT_DIR.glob("*.csv")),
            "figures": sorted(p.name for p in OUT_DIR.glob("fig_*.png")),
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
