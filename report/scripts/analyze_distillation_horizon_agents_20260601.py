"""Analyze distillation horizon-agent runs under the current reward.

The saved horizon and dueling horizon bundles span several reward and safety
configurations. This script reads only saved result bundles and rescoring data;
it does not launch Aspen or rerun any controller.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
    }
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS


OUT_DIR = ROOT / "report" / "figures" / "distillation_horizon_agents_20260601"
BASELINE_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
HORIZON_ROOT = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_horizon_disturb_fluctuation_mismatch_unified"
)
DUELING_ROOT = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
)


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


def current_reward_series(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
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
    sigmoid_arg = np.clip(
        (band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12),
        -60.0,
        60.0,
    )
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
    lin_out = (1.0 - w_in) * np.sum(
        float(RL_REWARD_DEFAULTS["gamma_out"]) * slope_at_edge * overflow,
        axis=1,
    )
    lin_in = w_in * np.sum(
        float(RL_REWARD_DEFAULTS["gamma_in"]) * slope_at_edge * inside_mag,
        axis=1,
    )

    z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
    bonus_kind = str(RL_REWARD_DEFAULTS["bonus_kind"]).lower()
    if bonus_kind != "exp":
        raise ValueError(f"Unsupported bonus kind for this analysis: {bonus_kind}")
    k_bonus = float(RL_REWARD_DEFAULTS["bonus_k"])
    phi = (np.exp(-k_bonus * z) - np.exp(-k_bonus)) / (1.0 - np.exp(-k_bonus))
    qb2 = q_diag.reshape(1, -1) * band_scaled * band_scaled
    bonus = w_in * float(RL_REWARD_DEFAULTS["beta"]) * np.sum(qb2 * phi, axis=1)

    rewards = (-(err_eff + move + lin_out + lin_in) + bonus) * float(
        RL_REWARD_DEFAULTS["reward_scale"]
    )
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    n_episodes = rewards.size // episode_len
    avg = rewards[: n_episodes * episode_len].reshape(n_episodes, episode_len).mean(axis=1)
    return rewards, avg


def episode_slice(bundle: dict, start_episode: int, end_episode: int) -> slice:
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    return slice((start_episode - 1) * episode_len, end_episode * episode_len)


def tail_step_slice(bundle: dict, episodes: int = 20) -> slice:
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    episode_len = int(bundle.get("time_in_sub_episodes", 400))
    return slice(max(0, nfe - episodes * episode_len), nfe)


def finite_mean(values) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_fraction(values) -> float:
    arr = np.asarray(values, float)
    return float(np.mean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else float("nan")


def physical_error_metrics(bundle: dict, sl: slice) -> dict:
    y = np.asarray(bundle.get("y_line_full", bundle.get("y")), float)
    ysp = y_sp_phys(bundle)
    n = min(y.shape[0] - 1, ysp.shape[0])
    y_step = y[1 : n + 1, :]
    ysp_step = ysp[:n, :]
    sl = slice(max(0, sl.start or 0), min(n, sl.stop or n))
    err = y_step[sl, :] - ysp_step[sl, :]

    delta_y = np.asarray(bundle["delta_y_storage"], float)[:n, :][sl, :]
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    dy_out = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1.0e-12)
    k_rel = np.asarray(RL_REWARD_DEFAULTS["k_rel"], float)
    floor_phys = np.asarray(RL_REWARD_DEFAULTS["band_floor_phys"], float)
    band_phys = np.maximum(
        k_rel.reshape(1, -1) * np.abs(ysp_step[sl, :]),
        floor_phys.reshape(1, -1),
    )
    band_scaled = band_phys / dy_out.reshape(1, -1)
    z_abs = np.abs(delta_y) / np.maximum(band_scaled, 1.0e-12)

    delta_u = np.asarray(bundle["delta_u_storage"], float)[:n, :][sl, :]
    return {
        "comp_mae": float(np.mean(np.abs(err[:, 0]))),
        "temp_mae": float(np.mean(np.abs(err[:, 1]))),
        "comp_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "temp_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "band_norm_mae": float(np.mean(z_abs)),
        "outside_band_frac": float(np.mean(np.any(z_abs > 1.0, axis=1))),
        "mean_abs_du_scaled": float(np.mean(np.abs(delta_u))),
    }


def entropy_from_counts(counts: Counter, total: int) -> float:
    if total <= 0:
        return float("nan")
    probs = np.array([count / total for count in counts.values()], dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def horizon_stats(bundle: dict, sl: slice) -> dict:
    trace = np.asarray(
        bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace")),
        float,
    )
    if trace.ndim != 2 or trace.shape[1] != 2:
        return {}
    trace = trace[sl, :]
    finite = np.all(np.isfinite(trace), axis=1)
    trace = trace[finite, :]
    pairs = [tuple(map(int, row)) for row in trace]
    counts = Counter(pairs)
    total = len(pairs)
    top_pair, top_count = counts.most_common(1)[0] if counts else ((None, None), 0)
    default_count = counts.get((6, 3), 0)

    requested = np.asarray(bundle.get("horizon_requested_trace_log", trace), float)[sl, :]
    requested = requested[np.all(np.isfinite(requested), axis=1), :]
    req_pairs = [tuple(map(int, row)) for row in requested]
    req_counts = Counter(req_pairs)
    req_top_pair, req_top_count = req_counts.most_common(1)[0] if req_counts else ((None, None), 0)

    change = np.asarray(bundle.get("horizon_change_log", []), float)
    projection = np.asarray(bundle.get("horizon_projection_active_log", []), float)
    cooldown = np.asarray(bundle.get("horizon_cooldown_active_log", []), float)
    reason_log = np.asarray(bundle.get("horizon_safety_reason_log", []), float)
    codes = bundle.get("horizon_safety_reason_codes", {}) or {}
    accepted_code = codes.get("accepted", 0)
    warm_code = codes.get("warm_default", 1)
    proj_code = codes.get("release_projection", 2)
    cooldown_code = codes.get("cooldown_default", 3)

    out = {
        "recipe_count": int(len(bundle.get("horizon_recipes", []))),
        "tail_mean_predict": float(np.mean(trace[:, 0])) if total else float("nan"),
        "tail_mean_control": float(np.mean(trace[:, 1])) if total else float("nan"),
        "tail_unique_pairs": int(len(counts)),
        "tail_entropy": entropy_from_counts(counts, total),
        "tail_top_pair": str(top_pair),
        "tail_top_pair_frac": float(top_count / total) if total else float("nan"),
        "tail_default_pair_frac": float(default_count / total) if total else float("nan"),
        "tail_requested_top_pair": str(req_top_pair),
        "tail_requested_top_pair_frac": float(req_top_count / len(req_pairs)) if req_pairs else float("nan"),
        "tail_switch_frac": finite_fraction(change[sl]) if change.size else float("nan"),
        "tail_projection_frac": finite_fraction(projection[sl]) if projection.size else float("nan"),
        "tail_cooldown_frac": finite_fraction(cooldown[sl]) if cooldown.size else float("nan"),
    }
    if reason_log.size:
        reason = reason_log[sl]
        finite_reason = reason[np.isfinite(reason)]
        denom = max(1, finite_reason.size)
        out.update(
            {
                "tail_reason_accepted_frac": float(np.mean(finite_reason == accepted_code)),
                "tail_reason_warm_default_frac": float(np.mean(finite_reason == warm_code)),
                "tail_reason_projection_frac": float(np.mean(finite_reason == proj_code)),
                "tail_reason_cooldown_frac": float(np.mean(finite_reason == cooldown_code)),
            }
        )
    else:
        out.update(
            {
                "tail_reason_accepted_frac": float("nan"),
                "tail_reason_warm_default_frac": float("nan"),
                "tail_reason_projection_frac": float("nan"),
                "tail_reason_cooldown_frac": float("nan"),
            }
        )
    return out


def learning_stats(bundle: dict) -> dict:
    eps = np.asarray(bundle.get("epsilon_trace", []), float)
    loss = np.asarray(bundle.get("dqn_loss_trace", []), float)
    noisy_sigma = np.asarray(bundle.get("noisy_sigma_trace", []), float)
    exploration = np.asarray(bundle.get("exploration_trace", []), float)
    snap = bundle.get("replay_buffer_snapshot", {}) or {}
    reward_params = bundle.get("reward_params", {}) or {}
    q_diag = np.asarray(reward_params.get("Q_diag", []), float)
    return {
        "buffer_capacity": snap.get("capacity"),
        "buffer_size": snap.get("size"),
        "epsilon_first": finite_mean(eps[:100]) if eps.size else float("nan"),
        "epsilon_tail": finite_mean(eps[-1000:]) if eps.size else float("nan"),
        "noisy_sigma_first": finite_mean(noisy_sigma[:100]) if noisy_sigma.size else float("nan"),
        "noisy_sigma_tail": finite_mean(noisy_sigma[-1000:]) if noisy_sigma.size else float("nan"),
        "exploration_trace_tail": finite_mean(exploration[-1000:]) if exploration.size else float("nan"),
        "dqn_loss_tail": finite_mean(loss[-1000:]) if loss.size else float("nan"),
        "saved_q1": float(q_diag[0]) if q_diag.size >= 1 else float("nan"),
        "saved_q2": float(q_diag[1]) if q_diag.size >= 2 else float("nan"),
        "saved_beta": reward_params.get("beta"),
        "reward_probation_enabled": bundle.get("horizon_reward_probation_enabled"),
    }


def summarize_run(method: str, path: Path, baseline_avg_current: np.ndarray) -> dict:
    bundle = load_pickle(path)
    _, avg_current = current_reward_series(bundle)
    tail_sl = tail_step_slice(bundle, 20)
    early_sl = episode_slice(bundle, 11, 30)
    logged = np.asarray(bundle.get("avg_rewards", []), float)
    row = {
        "method": method,
        "timestamp": path.parent.name,
        "path": str(path.relative_to(ROOT)),
        "current_tail20_reward": finite_mean(avg_current[-20:]),
        "current_final_reward": float(avg_current[-1]) if avg_current.size else float("nan"),
        "current_first20_live_min": float(np.nanmin(avg_current[10:30])) if avg_current.size >= 30 else float("nan"),
        "current_vs_ofmpc_tail_delta": finite_mean(avg_current[-20:])
        - finite_mean(baseline_avg_current[-20:]),
        "logged_tail20_reward": finite_mean(logged[-20:]) if logged.size else float("nan"),
        "logged_final_reward": float(logged[-1]) if logged.size else float("nan"),
        "n_episodes": int(avg_current.size),
    }
    row.update({f"tail_{k}": v for k, v in physical_error_metrics(bundle, tail_sl).items()})
    row.update({f"early_{k}": v for k, v in physical_error_metrics(bundle, early_sl).items()})
    row.update(horizon_stats(bundle, tail_sl))
    row.update(learning_stats(bundle))
    return row


def collect_runs(root: Path, method: str) -> list[Path]:
    return sorted(root.glob("*/input_data.pkl"))


def plot_reward_history(summary: pd.DataFrame, baseline_tail: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for method, group in summary.groupby("method"):
        group = group.sort_values("timestamp")
        ax.plot(
            group["timestamp"],
            group["current_tail20_reward"],
            marker="o",
            linewidth=1.8,
            label=method,
        )
    ax.axhline(baseline_tail, color="black", linestyle="--", linewidth=1.4, label="OF-MPC current")
    ax.set_ylabel("Tail-20 reward rescored with current reward")
    ax.set_xlabel("Run timestamp")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_horizon_current_reward_history.png", dpi=180)
    plt.close(fig)


def plot_latest_metrics(latest: pd.DataFrame, baseline: dict) -> None:
    metrics = [
        ("current_tail20_reward", "Tail reward"),
        ("tail_temp_mae", "Temp MAE"),
        ("tail_comp_mae", "Comp MAE"),
        ("tail_outside_band_frac", "Outside band"),
        ("tail_unique_pairs", "Unique pairs"),
        ("tail_switch_frac", "Switch frac"),
    ]
    labels = ["OF-MPC"] + [
        "Dueling" if name == "Dueling Horizon" else "Standard"
        for name in latest["method"].tolist()
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [baseline.get(key, np.nan)] + latest[key].tolist()
        ax.bar(labels, values, color=["#555555", "#2a7fbb", "#cc7a00"][: len(labels)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_latest_horizon_vs_ofmpc_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_pickle(BASELINE_PATH)
    _, baseline_avg_current = current_reward_series(baseline)
    baseline_tail_sl = tail_step_slice(baseline, 20)
    baseline_row = {
        "method": "OF-MPC",
        "timestamp": "baseline",
        "path": str(BASELINE_PATH.relative_to(ROOT)),
        "current_tail20_reward": finite_mean(baseline_avg_current[-20:]),
        "current_final_reward": float(baseline_avg_current[-1]),
        "current_first20_live_min": float(np.nanmin(baseline_avg_current[10:30])),
        "current_vs_ofmpc_tail_delta": 0.0,
        "logged_tail20_reward": finite_mean(np.asarray(baseline.get("avg_rewards", []), float)[-20:]),
        "logged_final_reward": float(np.asarray(baseline.get("avg_rewards", []), float)[-1]),
        "n_episodes": int(baseline_avg_current.size),
    }
    baseline_row.update(
        {f"tail_{k}": v for k, v in physical_error_metrics(baseline, baseline_tail_sl).items()}
    )

    rows = []
    for method, root in (("Horizon DDQN", HORIZON_ROOT), ("Dueling Horizon", DUELING_ROOT)):
        for path in collect_runs(root, method):
            rows.append(summarize_run(method, path, baseline_avg_current))

    summary = pd.DataFrame(rows).sort_values(["method", "timestamp"])
    latest = summary.sort_values("timestamp").groupby("method").tail(1).sort_values("method")

    focus_timestamps = {
        "20260518_141636",
        "20260518_140746",
        "20260521_154248",
        "20260521_154934",
        "20260528_201659",
        "20260528_202519",
        "20260529_213627",
        "20260529_213847",
        "20260530_225843",
        "20260530_223346",
        "20260601_160538",
        "20260601_160947",
    }
    focus = summary[summary["timestamp"].isin(focus_timestamps)].copy()

    baseline_df = pd.DataFrame([baseline_row])
    summary.to_csv(OUT_DIR / "all_horizon_runs_current_reward_summary.csv", index=False)
    focus.to_csv(OUT_DIR / "focus_horizon_runs_current_reward_summary.csv", index=False)
    latest.to_csv(OUT_DIR / "latest_horizon_pair_summary.csv", index=False)
    baseline_df.to_csv(OUT_DIR / "ofmpc_current_reward_summary.csv", index=False)

    plot_reward_history(summary, baseline_row["current_tail20_reward"])
    plot_latest_metrics(latest, baseline_row)

    manifest = {
        "baseline": baseline_row,
        "latest": latest.to_dict(orient="records"),
        "current_reward_defaults": {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in RL_REWARD_DEFAULTS.items()
        },
        "files": {
            "all_summary": str((OUT_DIR / "all_horizon_runs_current_reward_summary.csv").relative_to(ROOT)),
            "focus_summary": str((OUT_DIR / "focus_horizon_runs_current_reward_summary.csv").relative_to(ROOT)),
            "latest_summary": str((OUT_DIR / "latest_horizon_pair_summary.csv").relative_to(ROOT)),
            "baseline_summary": str((OUT_DIR / "ofmpc_current_reward_summary.csv").relative_to(ROOT)),
            "reward_history_fig": str((OUT_DIR / "fig_horizon_current_reward_history.png").relative_to(ROOT)),
            "latest_metrics_fig": str((OUT_DIR / "fig_latest_horizon_vs_ofmpc_metrics.png").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
