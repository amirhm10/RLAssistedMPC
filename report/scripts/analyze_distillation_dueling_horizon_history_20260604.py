"""Analyze only distillation dueling-horizon history.

This script reads saved result bundles and the disturbed OF-MPC baseline only.
It does not launch Aspen or rerun the controller. The output is a dueling-only
history table, summary JSON, and figures for reward provenance, physical
tracking, and horizon-policy stability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS


OUT_DIR = ROOT / "report" / "figures" / "distillation_dueling_horizon_history_20260604"
OFMPC_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
RUN_FOLDERS = [
    ROOT / "Distillation" / "Results" / "distillation_dueling_horizon_disturb_fluctuation_standard_unified",
    ROOT / "Distillation" / "Results" / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified",
]
EPISODE_LEN_FALLBACK = 400
TAIL_EPISODES = 20
POST_WARM_EPISODES = 20
DEFAULT_PAIR = (6, 3)

CURRENT_REWARD = deepcopy(RL_REWARD_DEFAULTS)
LEGACY_HORIZON_REWARD = {
    "k_rel": np.asarray([0.3, 0.02], dtype=float),
    "band_floor_phys": np.asarray([0.003, 0.3], dtype=float),
    "Q_diag": np.asarray([3.7e4, 1.5e3], dtype=float),
    "R_diag": np.asarray([2.5e3, 2.5e3], dtype=float),
    "tau_frac": 0.7,
    "gamma_out": 0.5,
    "gamma_in": 0.5,
    "beta": 7.0,
    "gate": "geom",
    "lam_in": 1.0,
    "bonus_kind": "exp",
    "bonus_k": 12.0,
    "bonus_p": 0.6,
    "bonus_c": 20.0,
    "reward_scale": 1.0,
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite(values) -> np.ndarray:
    arr = np.asarray(values, float).reshape(-1)
    return arr[np.isfinite(arr)]


def finite_mean(values) -> float:
    arr = finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_minmax(data, min_val, max_val):
    return (np.asarray(data, float) - np.asarray(min_val, float)) / np.maximum(
        np.asarray(max_val, float) - np.asarray(min_val, float),
        1.0e-12,
    )


def reverse_minmax(data, min_val, max_val):
    return np.asarray(data, float) * (
        np.asarray(max_val, float) - np.asarray(min_val, float)
    ) + np.asarray(min_val, float)


def n_steps(bundle: dict) -> int:
    if bundle.get("nFE") is not None:
        return int(bundle["nFE"])
    return int(np.asarray(bundle["y_sp"]).shape[0])


def episode_len(bundle: dict) -> int:
    return int(bundle.get("time_in_sub_episodes", EPISODE_LEN_FALLBACK))


def y_sp_phys(bundle: dict) -> np.ndarray:
    y_sp = np.asarray(bundle["y_sp"], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = apply_minmax(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    return reverse_minmax(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def _reward_phi(z: np.ndarray, cfg: dict) -> np.ndarray:
    kind = str(cfg.get("bonus_kind", "exp")).lower()
    z = np.clip(z, 0.0, 1.0)
    if kind == "exp":
        k = float(cfg.get("bonus_k", 12.0))
        return (np.exp(-k * z) - np.exp(-k)) / (1.0 - np.exp(-k))
    if kind == "linear":
        return 1.0 - z
    if kind == "quadratic":
        return (1.0 - z) ** 2
    if kind == "power":
        return 1.0 - np.power(z, float(cfg.get("bonus_p", 0.6)))
    if kind == "log":
        c = float(cfg.get("bonus_c", 20.0))
        return np.log1p(c * (1.0 - z)) / np.log1p(c)
    raise ValueError(f"Unsupported bonus kind: {kind}")


def reward_step_series(bundle: dict, cfg: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    dy_out = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1.0e-12)

    delta_y = np.asarray(bundle["delta_y_storage"], float)
    delta_u = np.asarray(bundle["delta_u_storage"], float)
    ysp = y_sp_phys(bundle)
    n = min(delta_y.shape[0], delta_u.shape[0], ysp.shape[0], n_steps(bundle))
    delta_y = delta_y[:n, :]
    delta_u = delta_u[:n, :]
    ysp = ysp[:n, :]

    q_diag = np.asarray(cfg["Q_diag"], float).reshape(1, -1)
    r_diag = np.asarray(cfg["R_diag"], float).reshape(1, -1)
    k_rel = np.asarray(cfg["k_rel"], float).reshape(1, -1)
    floor_phys = np.asarray(cfg["band_floor_phys"], float).reshape(1, -1)
    band_phys = np.maximum(k_rel * np.abs(ysp), floor_phys)
    band_scaled = band_phys / dy_out.reshape(1, -1)
    tau_scaled = float(cfg.get("tau_frac", 0.7)) * band_scaled

    abs_e = np.abs(delta_y)
    sigmoid_arg = np.clip((band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12), -60.0, 60.0)
    s_i = 1.0 / (1.0 + np.exp(-sigmoid_arg))
    gate = str(cfg.get("gate", "geom")).lower()
    if gate == "prod":
        w_in = np.prod(s_i, axis=1)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=1)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])
    else:
        raise ValueError(f"Unsupported gate: {gate}")

    err_quad = np.sum(q_diag * delta_y * delta_y, axis=1)
    move = np.sum(r_diag * delta_u * delta_u, axis=1)
    err_eff = ((1.0 - w_in) + w_in * float(cfg.get("lam_in", 1.0))) * err_quad

    slope_at_edge = 2.0 * q_diag * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    lin_out = (1.0 - w_in) * np.sum(float(cfg.get("gamma_out", 0.5)) * slope_at_edge * overflow, axis=1)
    lin_in = w_in * np.sum(float(cfg.get("gamma_in", 0.5)) * slope_at_edge * inside_mag, axis=1)

    z = abs_e / np.maximum(band_scaled, 1.0e-12)
    qb2 = q_diag * band_scaled * band_scaled
    bonus = w_in * float(cfg.get("beta", 7.0)) * np.sum(qb2 * _reward_phi(z, cfg), axis=1)
    return (-(err_eff + move + lin_out + lin_in) + bonus) * float(cfg.get("reward_scale", 1.0))


def episode_average(series: np.ndarray, bundle: dict) -> np.ndarray:
    ep_len = episode_len(bundle)
    n_ep = int(series.size // ep_len)
    return series[: n_ep * ep_len].reshape(n_ep, ep_len).mean(axis=1)


def step_slice(bundle: dict, start_episode: int, end_episode: int | None = None) -> slice:
    ep_len = episode_len(bundle)
    start = max(0, int(start_episode) * ep_len)
    stop = n_steps(bundle) if end_episode is None else min(n_steps(bundle), int(end_episode) * ep_len)
    return slice(start, stop)


def tail_slice(bundle: dict, episodes: int = TAIL_EPISODES) -> slice:
    ep_len = episode_len(bundle)
    return slice(max(0, n_steps(bundle) - int(episodes) * ep_len), n_steps(bundle))


def tracking_metrics(bundle: dict, cfg: dict, sl: slice) -> dict:
    n = n_steps(bundle)
    y = np.asarray(bundle.get("y_line_full", bundle.get("y")), float)
    if y.shape[0] >= n + 1:
        y_step = y[1 : n + 1, :]
    else:
        y_step = y[:n, :]
    ysp = y_sp_phys(bundle)[:n, :]
    sl = slice(max(0, sl.start or 0), min(n, sl.stop or n))
    err = y_step[sl, :] - ysp[sl, :]

    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    dy_out = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1.0e-12)
    k_rel = np.asarray(cfg["k_rel"], float).reshape(1, -1)
    floor_phys = np.asarray(cfg["band_floor_phys"], float).reshape(1, -1)
    band_phys = np.maximum(k_rel * np.abs(ysp[sl, :]), floor_phys)
    band_scaled = band_phys / dy_out.reshape(1, -1)
    delta_y = np.asarray(bundle["delta_y_storage"], float)[:n, :][sl, :]
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


def horizon_stats(bundle: dict, sl: slice) -> dict:
    trace = np.asarray(bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace", [])), float)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return {
            "recipe_count": float("nan"),
            "unique_pairs": float("nan"),
            "top_pair": "",
            "top_pair_frac": float("nan"),
            "default_pair_frac": float("nan"),
            "switch_frac": float("nan"),
            "mean_predict": float("nan"),
            "mean_control": float("nan"),
        }
    n = min(trace.shape[0], n_steps(bundle))
    trace = trace[:n, :][slice(max(0, sl.start or 0), min(n, sl.stop or n)), :]
    trace = trace[np.all(np.isfinite(trace), axis=1), :]
    pairs = [tuple(map(int, row)) for row in trace]
    counts = Counter(pairs)
    total = len(pairs)
    top_pair, top_count = counts.most_common(1)[0] if counts else ("", 0)
    switch_frac = float(np.mean(np.any(np.diff(trace, axis=0) != 0, axis=1))) if trace.shape[0] > 1 else float("nan")
    return {
        "recipe_count": int(len(bundle.get("horizon_recipes", []))),
        "unique_pairs": int(len(counts)),
        "top_pair": str(top_pair),
        "top_pair_frac": float(top_count / total) if total else float("nan"),
        "default_pair_frac": float(counts.get(DEFAULT_PAIR, 0) / total) if total else float("nan"),
        "switch_frac": switch_frac,
        "mean_predict": float(np.mean(trace[:, 0])) if total else float("nan"),
        "mean_control": float(np.mean(trace[:, 1])) if total else float("nan"),
    }


def trace_tail_mean(bundle: dict, key: str) -> float:
    arr = np.asarray(bundle.get(key, []), float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return float("nan")
    return float(np.mean(arr[-min(arr.size, 1000) :]))


def trace_first(bundle: dict, key: str) -> float:
    arr = np.asarray(bundle.get(key, []), float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr[0]) if arr.size else float("nan")


def trajectory_hash(bundle: dict) -> str:
    h = hashlib.sha1()
    for key in ("delta_y_storage", "delta_u_storage", "horizon_trace"):
        arr = np.asarray(bundle.get(key, []), float)
        h.update(np.ascontiguousarray(np.round(arr, 10)).view(np.uint8))
    return h.hexdigest()[:12]


def infer_exploration(bundle: dict) -> str:
    eps_tail = trace_tail_mean(bundle, "epsilon_trace")
    noisy_tail = trace_tail_mean(bundle, "noisy_sigma_trace")
    if np.isfinite(eps_tail) and eps_tail > 0.05:
        return "epsilon-greedy"
    if np.isfinite(noisy_tail) and noisy_tail > 1.0e-4:
        return "NoisyNet"
    if np.isfinite(eps_tail):
        return "low-epsilon"
    return "unknown"


def analyze_run(path: Path, ofmpc_current_tail: float, ofmpc_legacy_tail: float) -> dict:
    bundle = load_pickle(path)
    current_avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    legacy_avg = episode_average(reward_step_series(bundle, LEGACY_HORIZON_REWARD), bundle)
    logged_avg = np.asarray(bundle.get("avg_rewards", []), float)
    n_ep = min(current_avg.size, legacy_avg.size, logged_avg.size if logged_avg.size else current_avg.size)

    warm_start = int(bundle.get("warm_start_step", 10 * episode_len(bundle)) // episode_len(bundle))
    tail_ep = min(TAIL_EPISODES, n_ep)
    tail = slice(max(0, n_ep - tail_ep), n_ep)
    first_post = slice(warm_start, min(n_ep, warm_start + POST_WARM_EPISODES))
    tail_steps = tail_slice(bundle, TAIL_EPISODES)
    first_post_steps = step_slice(bundle, warm_start, warm_start + POST_WARM_EPISODES)

    saved_reward = bundle.get("reward_params", {}) or {}
    replay = bundle.get("replay_buffer_snapshot", {}) or {}
    folder = path.parent.parent.name
    state_mode = str(bundle.get("state_mode", "standard"))
    row = {
        "timestamp": path.parent.name,
        "state_mode": state_mode,
        "folder": folder,
        "path": rel(path),
        "notebook_source": bundle.get("notebook_source", ""),
        "n_episodes": int(n_ep),
        "warm_start": int(warm_start),
        "tail_logged_reward": finite_mean(logged_avg[tail]) if logged_avg.size else float("nan"),
        "tail_current_reward": finite_mean(current_avg[tail]),
        "tail_legacy_reward": finite_mean(legacy_avg[tail]),
        "current_delta_vs_ofmpc": finite_mean(current_avg[tail]) - ofmpc_current_tail,
        "legacy_delta_vs_ofmpc": finite_mean(legacy_avg[tail]) - ofmpc_legacy_tail,
        "final_current_reward": float(current_avg[n_ep - 1]),
        "final_legacy_reward": float(legacy_avg[n_ep - 1]),
        "first20_postwarm_current_min": float(np.min(current_avg[first_post])) if first_post.stop > first_post.start else float("nan"),
        "first20_postwarm_legacy_min": float(np.min(legacy_avg[first_post])) if first_post.stop > first_post.start else float("nan"),
        "saved_q1": float(np.asarray(saved_reward.get("Q_diag", [np.nan, np.nan]), float)[0]),
        "saved_q2": float(np.asarray(saved_reward.get("Q_diag", [np.nan, np.nan]), float)[1]),
        "saved_krel2": float(np.asarray(saved_reward.get("k_rel", [np.nan, np.nan]), float)[1]),
        "saved_band_floor2": float(np.asarray(saved_reward.get("band_floor_phys", [np.nan, np.nan]), float)[1]),
        "saved_beta": float(saved_reward.get("beta", np.nan)),
        "saved_reward_scale": float(saved_reward.get("reward_scale", np.nan)),
        "buffer_capacity": int(replay.get("capacity", 0)) if replay else float("nan"),
        "buffer_size": int(replay.get("size", 0)) if replay else float("nan"),
        "epsilon_first": trace_first(bundle, "epsilon_trace"),
        "epsilon_tail": trace_tail_mean(bundle, "epsilon_trace"),
        "noisy_sigma_first": trace_first(bundle, "noisy_sigma_trace"),
        "noisy_sigma_tail": trace_tail_mean(bundle, "noisy_sigma_trace"),
        "exploration_inferred": infer_exploration(bundle),
        "loss_tail": trace_tail_mean(bundle, "dqn_loss_trace"),
        "duplicate_hash": trajectory_hash(bundle),
    }
    tail_current_tracking = tracking_metrics(bundle, CURRENT_REWARD, tail_steps)
    tail_legacy_tracking = tracking_metrics(bundle, LEGACY_HORIZON_REWARD, tail_steps)
    early_current_tracking = tracking_metrics(bundle, CURRENT_REWARD, first_post_steps)
    row.update({f"tail_current_{k}": v for k, v in tail_current_tracking.items()})
    row.update({f"tail_legacy_{k}": v for k, v in tail_legacy_tracking.items()})
    row.update({f"first20_current_{k}": v for k, v in early_current_tracking.items()})
    row.update({f"tail_{k}": v for k, v in horizon_stats(bundle, tail_steps).items()})
    row.update({f"first20_{k}": v for k, v in horizon_stats(bundle, first_post_steps).items()})
    return row


def baseline_summary(bundle: dict) -> dict:
    current_avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    legacy_avg = episode_average(reward_step_series(bundle, LEGACY_HORIZON_REWARD), bundle)
    tail = slice(max(0, current_avg.size - TAIL_EPISODES), current_avg.size)
    tail_steps = tail_slice(bundle, TAIL_EPISODES)
    row = {
        "current_tail20_reward": finite_mean(current_avg[tail]),
        "legacy_tail20_reward": finite_mean(legacy_avg[tail]),
        "current_final_reward": float(current_avg[-1]),
        "legacy_final_reward": float(legacy_avg[-1]),
    }
    row.update({f"current_{k}": v for k, v in tracking_metrics(bundle, CURRENT_REWARD, tail_steps).items()})
    row.update({f"legacy_{k}": v for k, v in tracking_metrics(bundle, LEGACY_HORIZON_REWARD, tail_steps).items()})
    return row


def find_runs() -> list[Path]:
    paths: list[Path] = []
    for folder in RUN_FOLDERS:
        if folder.exists():
            paths.extend(sorted(folder.glob("*/input_data.pkl")))
    return sorted(paths, key=lambda p: p.parent.name)


def add_duplicate_counts(rows: list[dict]) -> None:
    counts = Counter(row["duplicate_hash"] for row in rows)
    first_seen: dict[str, str] = {}
    for row in rows:
        h = row["duplicate_hash"]
        first_seen.setdefault(h, row["timestamp"])
        row["duplicate_count"] = counts[h]
        row["duplicate_first_timestamp"] = first_seen[h]


def plot_reward_history(rows: list[dict], ofmpc: dict) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(rows))
    labels = [row["timestamp"] for row in rows]
    modes = [row["state_mode"] for row in rows]
    colors = ["#4C78A8" if mode == "standard" else "#F58518" for mode in modes]
    ax.scatter(x, [row["tail_current_reward"] for row in rows], c=colors, marker="o", s=54, label="Current reward rescoring")
    ax.scatter(x, [row["tail_legacy_reward"] for row in rows], c=colors, marker="^", s=48, alpha=0.65, label="Legacy reward rescoring")
    ax.plot(x, [row["tail_current_reward"] for row in rows], color="#4d4d4d", linewidth=0.8, alpha=0.45)
    ax.axhline(ofmpc["current_tail20_reward"], color="#111111", linestyle="--", linewidth=1.2, label="OF-MPC current")
    ax.axhline(ofmpc["legacy_tail20_reward"], color="#777777", linestyle=":", linewidth=1.4, label="OF-MPC legacy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Tail-20 average reward")
    ax.set_title("Dueling horizon history under common current and legacy reward scoring")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="best")
    fig.text(0.13, 0.01, "Blue: standard state; orange: mismatch state. Circles use current reward, triangles use legacy horizon reward.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / "fig_dueling_reward_history_common_scoring.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_logged_vs_current(rows: list[dict], ofmpc: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    q2 = np.asarray([row["saved_q2"] for row in rows], float)
    sc = ax.scatter(
        [row["tail_logged_reward"] for row in rows],
        [row["tail_current_reward"] for row in rows],
        c=q2,
        cmap="viridis",
        s=[45 + 90 * max(0.0, row["tail_top_pair_frac"]) for row in rows],
        edgecolor="#222222",
        linewidth=0.4,
    )
    lo = min(np.nanmin([row["tail_logged_reward"] for row in rows]), np.nanmin([row["tail_current_reward"] for row in rows]))
    hi = max(np.nanmax([row["tail_logged_reward"] for row in rows]), np.nanmax([row["tail_current_reward"] for row in rows]))
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle=":", linewidth=1.0)
    ax.axhline(ofmpc["current_tail20_reward"], color="#111111", linestyle="--", linewidth=1.1, label="OF-MPC current")
    ax.set_xlabel("Saved/logged tail-20 reward")
    ax.set_ylabel("Same trajectory rescored with current reward")
    ax.set_title("Logged reward is not comparable across reward revisions")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Saved temperature weight Q2")
    ax.legend(loc="best")
    fig.tight_layout()
    out = OUT_DIR / "fig_logged_vs_current_reward.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_tracking_tradeoff(rows: list[dict], ofmpc: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    delta = np.asarray([row["current_delta_vs_ofmpc"] for row in rows], float)
    sc = ax.scatter(
        [row["tail_current_comp_mae"] for row in rows],
        [row["tail_current_temp_mae"] for row in rows],
        c=delta,
        cmap="coolwarm",
        s=[45 + 160 * max(0.0, row["tail_top_pair_frac"]) for row in rows],
        edgecolor="#222222",
        linewidth=0.4,
    )
    ax.scatter(
        [ofmpc["current_comp_mae"]],
        [ofmpc["current_temp_mae"]],
        marker="*",
        s=220,
        color="#111111",
        label="OF-MPC",
    )
    ax.set_xlabel("Tail x24 composition MAE")
    ax.set_ylabel("Tail T85 MAE")
    ax.set_title("Successful dueling runs protect temperature while improving composition")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Current tail reward delta vs OF-MPC")
    ax.legend(loc="best")
    fig.tight_layout()
    out = OUT_DIR / "fig_tracking_tradeoff_current_reward.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_policy_stability(rows: list[dict]) -> Path:
    fig, ax1 = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(rows))
    labels = [row["timestamp"] for row in rows]
    bars = ax1.bar(x, [row["tail_unique_pairs"] for row in rows], color="#9ecae1", label="Unique tail pairs")
    ax1.set_ylabel("Unique horizon pairs in tail")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=55, ha="right")
    ax1.grid(True, axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["tail_top_pair_frac"] for row in rows], color="#d95f0e", marker="o", label="Top pair fraction")
    ax2.plot(x, [row["tail_default_pair_frac"] for row in rows], color="#238b45", marker="s", label="Default (6,3) fraction")
    ax2.set_ylabel("Tail fraction")
    ax1.set_title("Dueling horizon success correlates with a concentrated tail schedule")
    handles = [bars, *ax2.get_lines()]
    labels_legend = [h.get_label() for h in handles]
    ax1.legend(handles, labels_legend, loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "fig_policy_stability_history.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_config_timeline(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.0), sharex=True)
    x = np.arange(len(rows))
    labels = [row["timestamp"] for row in rows]
    axes[0].plot(x, [row["saved_q2"] for row in rows], marker="o", color="#756bb1")
    axes[0].set_ylabel("Saved Q2")
    axes[1].plot(x, [row["buffer_capacity"] for row in rows], marker="o", color="#2b8cbe")
    axes[1].set_ylabel("Replay cap")
    axes[2].plot(x, [row["tail_recipe_count"] for row in rows], marker="o", color="#31a354")
    axes[2].set_ylabel("Action count")
    axes[3].plot(x, [row["epsilon_tail"] for row in rows], marker="o", color="#e6550d", label="epsilon")
    axes[3].plot(x, [row["noisy_sigma_tail"] for row in rows], marker="s", color="#636363", label="noisy sigma")
    axes[3].set_ylabel("Tail exploration")
    axes[3].legend(loc="best")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=55, ha="right")
    axes[0].set_title("Algorithm/configuration drift across dueling-horizon runs")
    fig.tight_layout()
    out = OUT_DIR / "fig_config_timeline.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def tail_pair_distribution_for_row(row: dict) -> dict[tuple[int, int], float]:
    bundle = load_pickle(ROOT / row["path"])
    trace = np.asarray(bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace", [])), float)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return {}
    sl = tail_slice(bundle, TAIL_EPISODES)
    n = min(trace.shape[0], n_steps(bundle))
    trace = trace[:n, :][slice(max(0, sl.start or 0), min(n, sl.stop or n)), :]
    trace = trace[np.all(np.isfinite(trace), axis=1), :]
    pairs = [tuple(map(int, row_trace)) for row_trace in trace]
    counts = Counter(pairs)
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {pair: count / total for pair, count in counts.items()}


def build_tail_distributions(unique_rows: list[dict]) -> dict[str, dict[tuple[int, int], float]]:
    return {row["duplicate_hash"]: tail_pair_distribution_for_row(row) for row in unique_rows}


def is_stable_success(row: dict) -> bool:
    return (
        float(row["current_delta_vs_ofmpc"]) > 1.0
        and float(row["tail_current_temp_mae"]) <= 0.205
        and int(row["tail_recipe_count"]) == 87
        and (
            float(row["tail_top_pair_frac"]) >= 0.25
            or float(row["tail_default_pair_frac"]) >= 0.20
        )
    )


def is_failure_or_overwide(row: dict) -> bool:
    return (
        float(row["current_delta_vs_ofmpc"]) < 0.0
        or float(row["tail_current_temp_mae"]) > 0.215
        or int(row["tail_recipe_count"]) > 87
    )


def _success_weight(row: dict) -> float:
    delta = max(float(row["current_delta_vs_ofmpc"]), 0.25)
    concentration = max(float(row["tail_top_pair_frac"]), float(row["tail_default_pair_frac"]), 0.05)
    return delta * (0.5 + concentration)


def _failure_weight(row: dict) -> float:
    delta_loss = max(-float(row["current_delta_vs_ofmpc"]), 0.0)
    temp_excess = max(float(row["tail_current_temp_mae"]) - 0.205, 0.0) * 30.0
    wide_penalty = 1.0 if int(row["tail_recipe_count"]) > 87 else 0.0
    return max(1.0, delta_loss + temp_excess + wide_penalty)


def recommend_horizon_pairs(unique_rows: list[dict], distributions: dict[str, dict[tuple[int, int], float]]) -> list[dict]:
    success_rows = [row for row in unique_rows if is_stable_success(row)]
    failure_rows = [row for row in unique_rows if is_failure_or_overwide(row)]
    pairs = sorted({pair for dist in distributions.values() for pair in dist})
    success_total = sum(_success_weight(row) for row in success_rows)
    failure_total = sum(_failure_weight(row) for row in failure_rows)

    rows: list[dict] = []
    for pair in pairs:
        success_frac = 0.0
        failure_frac = 0.0
        support_count = 0
        failure_support_count = 0
        best_single_frac = 0.0
        for row in success_rows:
            frac = float(distributions.get(row["duplicate_hash"], {}).get(pair, 0.0))
            success_frac += _success_weight(row) * frac
            support_count += int(frac >= 0.05)
            best_single_frac = max(best_single_frac, frac)
        for row in failure_rows:
            frac = float(distributions.get(row["duplicate_hash"], {}).get(pair, 0.0))
            failure_frac += _failure_weight(row) * frac
            failure_support_count += int(frac >= 0.05)
        success_frac = success_frac / success_total if success_total else 0.0
        failure_frac = failure_frac / failure_total if failure_total else 0.0
        score = success_frac - 0.5 * failure_frac + 0.015 * support_count
        rows.append(
            {
                "pair": str(pair),
                "predict_h": int(pair[0]),
                "control_h": int(pair[1]),
                "weighted_success_tail_frac": success_frac,
                "weighted_failure_tail_frac": failure_frac,
                "success_support_count": support_count,
                "failure_support_count": failure_support_count,
                "best_single_success_frac": best_single_frac,
                "recommendation_score": score,
            }
        )
    rows.sort(key=lambda row: row["recommendation_score"], reverse=True)
    return rows


def valid_recipe_set(predict_min: int, predict_max: int, control_min: int, control_max: int) -> set[tuple[int, int]]:
    return {
        (predict_h, control_h)
        for predict_h in range(int(predict_min), int(predict_max) + 1)
        for control_h in range(int(control_min), int(control_max) + 1)
        if control_h <= predict_h
    }


def coverage_for_rows(
    rows: list[dict],
    distributions: dict[str, dict[tuple[int, int], float]],
    recipe_set: set[tuple[int, int]],
    weight_fn,
) -> float:
    total_weight = sum(weight_fn(row) for row in rows)
    if total_weight <= 0:
        return float("nan")
    covered = 0.0
    for row in rows:
        dist = distributions.get(row["duplicate_hash"], {})
        covered += weight_fn(row) * sum(frac for pair, frac in dist.items() if pair in recipe_set)
    return covered / total_weight


def recommend_horizon_ranges(unique_rows: list[dict], distributions: dict[str, dict[tuple[int, int], float]]) -> list[dict]:
    success_rows = [row for row in unique_rows if is_stable_success(row)]
    failure_rows = [row for row in unique_rows if is_failure_or_overwide(row)]
    candidates: list[dict] = []
    seen_ranges: set[tuple[int, int, int, int]] = set()

    for predict_min in range(4, 12):
        for predict_max in range(predict_min + 1, 15):
            for control_min in range(2, 9):
                for control_max in range(control_min, 14):
                    recipe_set = valid_recipe_set(predict_min, predict_max, control_min, control_max)
                    n_actions = len(recipe_set)
                    if n_actions < 12 or n_actions > 55:
                        continue
                    success_cov = coverage_for_rows(success_rows, distributions, recipe_set, _success_weight)
                    if not np.isfinite(success_cov) or success_cov < 0.60:
                        continue
                    failure_cov = coverage_for_rows(failure_rows, distributions, recipe_set, _failure_weight)
                    score = success_cov - 0.30 * failure_cov - 0.10 * (n_actions / 87.0)
                    key = (predict_min, predict_max, control_min, control_max)
                    seen_ranges.add(key)
                    candidates.append(
                        {
                            "name": "searched_rectangle",
                            "predict_min": predict_min,
                            "predict_max": predict_max,
                            "control_min": control_min,
                            "control_max": control_max,
                            "n_actions": n_actions,
                            "stable_success_coverage": success_cov,
                            "failure_or_overwide_coverage": failure_cov,
                            "range_score": score,
                        }
                    )

    manual_ranges = {
        "current_87": (4, 14, 2, 13),
        "medium_4_12_2_8": (4, 12, 2, 8),
        "compact_5_11_2_6": (5, 11, 2, 6),
        "default_neighborhood_4_8_2_5": (4, 8, 2, 5),
        "long_tail_8_12_5_11": (8, 12, 5, 11),
    }
    for name, (predict_min, predict_max, control_min, control_max) in manual_ranges.items():
        recipe_set = valid_recipe_set(predict_min, predict_max, control_min, control_max)
        success_cov = coverage_for_rows(success_rows, distributions, recipe_set, _success_weight)
        failure_cov = coverage_for_rows(failure_rows, distributions, recipe_set, _failure_weight)
        score = success_cov - 0.30 * failure_cov - 0.10 * (len(recipe_set) / 87.0)
        candidates.append(
            {
                "name": name,
                "predict_min": predict_min,
                "predict_max": predict_max,
                "control_min": control_min,
                "control_max": control_max,
                "n_actions": len(recipe_set),
                "stable_success_coverage": success_cov,
                "failure_or_overwide_coverage": failure_cov,
                "range_score": score,
            }
        )

    candidates.sort(key=lambda row: row["range_score"], reverse=True)
    return candidates


def plot_pair_recommendations(pair_rows: list[dict]) -> Path:
    top = pair_rows[:14]
    labels = [row["pair"] for row in top][::-1]
    success = [row["weighted_success_tail_frac"] for row in top][::-1]
    failure = [-row["weighted_failure_tail_frac"] for row in top][::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.barh(y, success, color="#238b45", label="Success tail fraction")
    ax.barh(y, failure, color="#cb181d", alpha=0.72, label="Failure tail fraction")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Weighted tail usage fraction")
    ax.set_title("Candidate horizon pairs from stable successful dueling runs")
    ax.axvline(0.0, color="#111111", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "fig_recommended_horizon_pairs.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_range_candidates(range_rows: list[dict]) -> Path:
    rows = range_rows[:40]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(
        [row["n_actions"] for row in rows],
        [row["stable_success_coverage"] for row in rows],
        c=[row["failure_or_overwide_coverage"] for row in rows],
        s=[45 + 120 * max(float(row["range_score"]), 0.0) for row in rows],
        cmap="magma_r",
        edgecolor="#222222",
        linewidth=0.4,
    )
    for row in rows[:8]:
        label = row["name"] if row["name"] != "searched_rectangle" else f'{row["predict_min"]}-{row["predict_max"]}/{row["control_min"]}-{row["control_max"]}'
        ax.annotate(label, (row["n_actions"], row["stable_success_coverage"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Number of valid horizon recipes")
    ax.set_ylabel("Coverage of stable-success tail actions")
    ax.set_title("Reduced horizon-range candidates")
    x_values = [row["n_actions"] for row in rows]
    ax.set_xlim(min(x_values) - 3, max(x_values) + 8)
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Coverage of failure / overwide tails")
    fig.tight_layout()
    out = OUT_DIR / "fig_recommended_horizon_ranges.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ofmpc_bundle = load_pickle(OFMPC_PATH)
    ofmpc = baseline_summary(ofmpc_bundle)

    rows = [analyze_run(path, ofmpc["current_tail20_reward"], ofmpc["legacy_tail20_reward"]) for path in find_runs()]
    rows.sort(key=lambda row: row["timestamp"])
    add_duplicate_counts(rows)

    unique_rows = []
    seen = set()
    for row in rows:
        if row["duplicate_hash"] not in seen:
            unique_rows.append(row)
            seen.add(row["duplicate_hash"])

    distributions = build_tail_distributions(unique_rows)
    pair_recommendations = recommend_horizon_pairs(unique_rows, distributions)
    range_recommendations = recommend_horizon_ranges(unique_rows, distributions)

    figure_paths = [
        plot_reward_history(rows, ofmpc),
        plot_logged_vs_current(rows, ofmpc),
        plot_tracking_tradeoff(rows, ofmpc),
        plot_policy_stability(rows),
        plot_config_timeline(rows),
        plot_pair_recommendations(pair_recommendations),
        plot_range_candidates(range_recommendations),
    ]

    write_csv(OUT_DIR / "dueling_horizon_history_summary.csv", rows)
    write_csv(OUT_DIR / "dueling_horizon_unique_trajectory_summary.csv", unique_rows)
    write_csv(OUT_DIR / "ofmpc_reward_reference.csv", [ofmpc])
    write_csv(OUT_DIR / "recommended_horizon_pairs.csv", pair_recommendations)
    write_csv(OUT_DIR / "recommended_horizon_ranges.csv", range_recommendations)

    sorted_current = sorted(rows, key=lambda row: row["tail_current_reward"], reverse=True)
    sorted_legacy = sorted(rows, key=lambda row: row["tail_legacy_reward"], reverse=True)
    sorted_stable = sorted(rows, key=lambda row: row["tail_top_pair_frac"], reverse=True)
    payload = {
        "n_runs": len(rows),
        "n_unique_trajectories": len(unique_rows),
        "ofmpc": ofmpc,
        "best_current_reward": sorted_current[0] if sorted_current else {},
        "best_legacy_reward": sorted_legacy[0] if sorted_legacy else {},
        "most_concentrated_tail": sorted_stable[0] if sorted_stable else {},
        "latest": rows[-1] if rows else {},
        "figures": [rel(path) for path in figure_paths],
        "outputs": {
            "history_csv": rel(OUT_DIR / "dueling_horizon_history_summary.csv"),
            "unique_csv": rel(OUT_DIR / "dueling_horizon_unique_trajectory_summary.csv"),
            "ofmpc_csv": rel(OUT_DIR / "ofmpc_reward_reference.csv"),
            "recommended_pairs_csv": rel(OUT_DIR / "recommended_horizon_pairs.csv"),
            "recommended_ranges_csv": rel(OUT_DIR / "recommended_horizon_ranges.csv"),
        },
        "n_stable_success_trajectories": sum(is_stable_success(row) for row in unique_rows),
        "n_failure_or_overwide_trajectories": sum(is_failure_or_overwide(row) for row in unique_rows),
        "top_recommended_pairs": pair_recommendations[:12],
        "top_recommended_ranges": range_recommendations[:12],
        "current_reward": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in CURRENT_REWARD.items()},
        "legacy_reward": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in LEGACY_HORIZON_REWARD.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("n_runs", "n_unique_trajectories", "figures", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
