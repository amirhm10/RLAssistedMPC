"""Analyze distillation SG-TD3 and limited-grid SG-DQN runs.

The script reads saved result bundles only. It rescales trajectories with the
current distillation reward, compares the latest mismatch/limited-horizon runs
against the closest previous mismatch references, and writes a Markdown report,
CSV/JSON metrics, and figures.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report.scripts.analyze_distillation_dueling_horizon_history_20260604 import (  # noqa: E402
    CURRENT_REWARD,
    DEFAULT_PAIR,
    OFMPC_PATH,
    TAIL_EPISODES,
    episode_average,
    episode_len,
    finite_mean,
    horizon_stats,
    load_pickle,
    rel,
    reward_step_series,
    step_slice,
    tail_slice,
    tracking_metrics,
)


OUT_DIR = ROOT / "report" / "figures" / "distillation_sg_limited_horizons_20260606"
REPORT_PATH = ROOT / "report" / "distillation_sg_td3_sg_dqn_limited_horizon_results_2026_06_06.md"
SUMMARY_CSV = OUT_DIR / "distillation_sg_limited_summary.csv"
SUMMARY_JSON = OUT_DIR / "distillation_sg_limited_summary.json"
EPISODE_CSV = OUT_DIR / "distillation_sg_limited_episode_diagnostics.csv"
HORIZON_PAIR_CSV = OUT_DIR / "distillation_sg_limited_horizon_pairs.csv"

SOURCE_WARM_START = 0
SOURCE_SUPERVISOR = 1
SOURCE_POLICY = 2
SOURCE_HELD = 3
SOURCE_FALLBACK = 4
WARM_START_SUBEPISODES = 10
HANDOVER_SUBEPISODES = 3
LIVE_RELEASE_SUBEPISODE = WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES + 1


RUNS = [
    {
        "key": "ofmpc",
        "label": "OF-MPC",
        "family": "baseline",
        "role": "baseline",
        "path": OFMPC_PATH,
    },
    {
        "key": "weights_current",
        "label": "Weights SG-TD3 current",
        "family": "weights",
        "role": "current",
        "reference_key": "weights_previous",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch"
        / "20260606_075201"
        / "input_data.pkl",
    },
    {
        "key": "weights_previous",
        "label": "Weights SG-TD3 previous",
        "family": "weights",
        "role": "reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch"
        / "20260603_124214"
        / "input_data.pkl",
    },
    {
        "key": "residual_current",
        "label": "Residual SG-TD3 current",
        "family": "residual",
        "role": "current",
        "reference_key": "residual_previous",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho"
        / "20260606_074932"
        / "input_data.pkl",
    },
    {
        "key": "residual_previous",
        "label": "Residual SG-TD3 previous",
        "family": "residual",
        "role": "reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho"
        / "20260602_125954"
        / "input_data.pkl",
    },
    {
        "key": "markov_current",
        "label": "Markov SG-TD3 current param-noise",
        "family": "markov",
        "role": "current",
        "reference_key": "markov_previous",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_mismatch_paramnoise"
        / "20260606_105404"
        / "input_data.pkl",
    },
    {
        "key": "markov_previous",
        "label": "Markov SG-TD3 previous",
        "family": "markov",
        "role": "reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_unified"
        / "20260602_192543"
        / "input_data.pkl",
    },
    {
        "key": "horizon_limited",
        "label": "Horizon SG-DQN limited 39",
        "family": "horizon",
        "role": "current",
        "reference_key": "horizon_wide",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11"
        / "20260606_071451"
        / "input_data.pkl",
    },
    {
        "key": "horizon_wide",
        "label": "Horizon SG-DQN wide 87",
        "family": "horizon",
        "role": "reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch"
        / "20260604_103231"
        / "input_data.pkl",
    },
    {
        "key": "dueling_wide",
        "label": "Dueling SG-DQN wide 87",
        "family": "dueling horizon",
        "role": "context",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch"
        / "20260604_105140"
        / "input_data.pkl",
    },
]


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


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, float).reshape(-1)
    return arr[np.isfinite(arr)]


def _mean(values) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _q(values, quantile: float) -> float:
    arr = _finite(values)
    return float(np.quantile(arr, quantile)) if arr.size else float("nan")


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _write_csv(path: Path, rows: list[dict]) -> None:
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


def _safe_slice(arr: np.ndarray, sl: slice) -> np.ndarray:
    start = max(0, sl.start or 0)
    stop = min(arr.shape[0], sl.stop or arr.shape[0])
    if stop <= start:
        return arr[:0]
    return arr[start:stop]


def _sg_metrics(bundle: dict, sl: slice) -> dict:
    selected = np.asarray(bundle.get("sg_selected_source_log", []), int).reshape(-1)
    if selected.size == 0:
        return {
            "sg_policy_step_frac": float("nan"),
            "sg_supervisor_step_frac": float("nan"),
            "sg_held_step_frac": float("nan"),
            "sg_fallback_step_frac": float("nan"),
            "sg_policy_decision_frac": float("nan"),
            "sg_adv_median": float("nan"),
            "sg_adv_q10": float("nan"),
            "sg_adv_q90": float("nan"),
        }
    selected_sl = _safe_slice(selected, sl)
    decision = (selected_sl == SOURCE_POLICY) | (selected_sl == SOURCE_SUPERVISOR)
    score_policy = np.asarray(bundle.get("sg_score_policy_log", []), float).reshape(-1)
    score_supervisor = np.asarray(bundle.get("sg_score_supervisor_log", []), float).reshape(-1)
    advantage = np.asarray(bundle.get("sg_advantage_log", []), float).reshape(-1)
    if advantage.size == 0 and score_policy.size and score_supervisor.size:
        n = min(score_policy.size, score_supervisor.size)
        advantage = score_policy[:n] - score_supervisor[:n]
    advantage_sl = _safe_slice(advantage, sl) if advantage.size else np.asarray([])
    if advantage_sl.size == selected_sl.size:
        advantage_decision = advantage_sl[decision]
    else:
        advantage_decision = advantage_sl
    decision_count = int(np.sum(decision))
    return {
        "sg_policy_step_frac": float(np.mean(selected_sl == SOURCE_POLICY)) if selected_sl.size else float("nan"),
        "sg_supervisor_step_frac": (
            float(np.mean(selected_sl == SOURCE_SUPERVISOR)) if selected_sl.size else float("nan")
        ),
        "sg_held_step_frac": float(np.mean(selected_sl == SOURCE_HELD)) if selected_sl.size else float("nan"),
        "sg_fallback_step_frac": float(np.mean(selected_sl == SOURCE_FALLBACK)) if selected_sl.size else float("nan"),
        "sg_policy_decision_frac": (
            float(np.sum(selected_sl == SOURCE_POLICY) / decision_count) if decision_count else float("nan")
        ),
        "sg_adv_median": _q(advantage_decision, 0.50),
        "sg_adv_q10": _q(advantage_decision, 0.10),
        "sg_adv_q90": _q(advantage_decision, 0.90),
    }


def _source_counts(bundle: dict) -> dict:
    selected = np.asarray(bundle.get("sg_selected_source_log", []), int).reshape(-1)
    if selected.size == 0:
        return {}
    counts = Counter(selected.tolist())
    total = selected.size
    return {f"source_{code}_count": int(count) for code, count in sorted(counts.items())} | {
        f"source_{code}_frac": float(count / total) for code, count in sorted(counts.items())
    }


def _horizon_pair_rows(key: str, label: str, bundle: dict, sl: slice, window: str, limit: int = 12) -> list[dict]:
    trace = np.asarray(bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace", [])), float)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return []
    n = min(trace.shape[0], int(bundle.get("nFE", trace.shape[0])))
    view = trace[:n, :][slice(max(0, sl.start or 0), min(n, sl.stop or n)), :]
    view = view[np.all(np.isfinite(view), axis=1), :]
    pairs = [tuple(map(int, row)) for row in view]
    counts = Counter(pairs)
    total = sum(counts.values())
    rows = []
    for rank, ((np_h, nc_h), count) in enumerate(counts.most_common(limit), start=1):
        rows.append(
            {
                "key": key,
                "label": label,
                "window": window,
                "rank": rank,
                "Np": np_h,
                "Nc": nc_h,
                "count": int(count),
                "fraction": float(count / total) if total else float("nan"),
            }
        )
    return rows


def _horizon_recipe_summary(bundle: dict) -> dict:
    recipes = bundle.get("horizon_recipes")
    if recipes is None:
        return {
            "recipe_count": "",
            "recipe_np_min": "",
            "recipe_np_max": "",
            "recipe_nc_min": "",
            "recipe_nc_max": "",
        }
    arr = np.asarray(recipes, int)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
        return {
            "recipe_count": 0,
            "recipe_np_min": "",
            "recipe_np_max": "",
            "recipe_nc_min": "",
            "recipe_nc_max": "",
        }
    return {
        "recipe_count": int(arr.shape[0]),
        "recipe_np_min": int(np.min(arr[:, 0])),
        "recipe_np_max": int(np.max(arr[:, 0])),
        "recipe_nc_min": int(np.min(arr[:, 1])),
        "recipe_nc_max": int(np.max(arr[:, 1])),
    }


def _row_for_run(cfg: dict, ofmpc_tail_reward: float) -> tuple[dict, dict, np.ndarray]:
    bundle = load_pickle(cfg["path"])
    rewards = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    n_ep = int(rewards.size)
    warm_start = int(bundle.get("warm_start_step", WARM_START_SUBEPISODES * episode_len(bundle)) // episode_len(bundle))
    tail_ep = slice(max(0, n_ep - TAIL_EPISODES), n_ep)
    post_ep = slice(warm_start, n_ep)
    tail_steps = tail_slice(bundle, TAIL_EPISODES)
    first_live_steps = step_slice(bundle, WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES, WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES + 20)
    tracking_tail = tracking_metrics(bundle, CURRENT_REWARD, tail_steps)
    tracking_first_live = tracking_metrics(bundle, CURRENT_REWARD, first_live_steps)

    row = {
        "key": cfg["key"],
        "label": cfg["label"],
        "family": cfg["family"],
        "role": cfg["role"],
        "path": rel(cfg["path"]),
        "timestamp": cfg["path"].parent.name,
        "agent_kind": bundle.get("agent_kind", "ofmpc"),
        "algorithm": bundle.get("algorithm", ""),
        "notebook_source": bundle.get("notebook_source", ""),
        "state_mode": bundle.get("state_mode", ""),
        "markov_state_mode": bundle.get("markov_state_mode", ""),
        "markov_agent_state_features": bundle.get("markov_agent_state_features", ""),
        "mpc_horizons": str(tuple(map(int, bundle.get("mpc_horizons", [])))) if bundle.get("mpc_horizons") is not None else "",
        "episodes": n_ep,
        "warm_start_subepisodes": warm_start,
        "handover_subepisodes": int(bundle.get("post_warm_start_action_freeze_subepisodes", HANDOVER_SUBEPISODES) or HANDOVER_SUBEPISODES),
        "tail_reward": finite_mean(rewards[tail_ep]),
        "delta_vs_ofmpc": finite_mean(rewards[tail_ep]) - ofmpc_tail_reward,
        "final_reward": float(rewards[-1]) if rewards.size else float("nan"),
        "worst_postwarm_reward": float(np.min(rewards[post_ep])) if rewards.size > warm_start else float("nan"),
        "negative_postwarm_episodes": int(np.sum(rewards[post_ep] < 0.0)) if rewards.size > warm_start else 0,
        "first_live_reward_mean": finite_mean(
            rewards[
                WARM_START_SUBEPISODES
                + HANDOVER_SUBEPISODES : min(n_ep, WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES + 20)
            ]
        ),
        "tail_comp_mae": tracking_tail["comp_mae"],
        "tail_temp_mae": tracking_tail["temp_mae"],
        "tail_outside_band_frac": tracking_tail["outside_band_frac"],
        "tail_mean_abs_du_scaled": tracking_tail["mean_abs_du_scaled"],
        "first_live_comp_mae": tracking_first_live["comp_mae"],
        "first_live_temp_mae": tracking_first_live["temp_mae"],
        "first_live_outside_band_frac": tracking_first_live["outside_band_frac"],
    }
    row.update(_horizon_recipe_summary(bundle))
    row.update({f"tail_{k}": v for k, v in _sg_metrics(bundle, tail_steps).items()})
    row.update({f"first_live_{k}": v for k, v in _sg_metrics(bundle, first_live_steps).items()})
    row.update(_source_counts(bundle))
    if bundle.get("horizon_recipes") is not None:
        row.update({f"tail_horizon_{k}": v for k, v in horizon_stats(bundle, tail_steps).items()})
        row.update({f"first_live_horizon_{k}": v for k, v in horizon_stats(bundle, first_live_steps).items()})
    return row, bundle, rewards


def _episode_rows_for_run(cfg: dict, bundle: dict, rewards: np.ndarray) -> list[dict]:
    selected = np.asarray(bundle.get("sg_selected_source_log", []), int).reshape(-1)
    rows = []
    for ep, reward in enumerate(rewards):
        sl = step_slice(bundle, ep, ep + 1)
        tracking = tracking_metrics(bundle, CURRENT_REWARD, sl)
        sg = _sg_metrics(bundle, sl)
        row = {
            "key": cfg["key"],
            "label": cfg["label"],
            "family": cfg["family"],
            "role": cfg["role"],
            "subepisode": ep + 1,
            "reward": float(reward),
            "temp_mae": tracking["temp_mae"],
            "comp_mae": tracking["comp_mae"],
            "outside_band_frac": tracking["outside_band_frac"],
        }
        row.update(sg)
        if selected.size == 0:
            row["sg_decision_count"] = float("nan")
        else:
            source_sl = _safe_slice(selected, sl)
            decision = (source_sl == SOURCE_POLICY) | (source_sl == SOURCE_SUPERVISOR)
            row["sg_decision_count"] = int(np.sum(decision))
        rows.append(row)
    return rows


def _comparison_rows(summary_rows: list[dict]) -> list[dict]:
    by_key = {row["key"]: row for row in summary_rows}
    rows = []
    for row in summary_rows:
        ref_key = next((cfg.get("reference_key") for cfg in RUNS if cfg["key"] == row["key"]), None)
        if not ref_key or ref_key not in by_key:
            continue
        ref = by_key[ref_key]
        rows.append(
            {
                "current_key": row["key"],
                "reference_key": ref_key,
                "family": row["family"],
                "tail_reward_delta": row["tail_reward"] - ref["tail_reward"],
                "tail_temp_mae_delta": row["tail_temp_mae"] - ref["tail_temp_mae"],
                "tail_comp_mae_delta": row["tail_comp_mae"] - ref["tail_comp_mae"],
                "negative_postwarm_delta": row["negative_postwarm_episodes"] - ref["negative_postwarm_episodes"],
                "policy_step_frac_delta": row.get("tail_sg_policy_step_frac", float("nan"))
                - ref.get("tail_sg_policy_step_frac", float("nan")),
                "first_live_policy_step_frac_delta": row.get("first_live_sg_policy_step_frac", float("nan"))
                - ref.get("first_live_sg_policy_step_frac", float("nan")),
            }
        )
    return rows


def _plot_current_tail(summary_rows: list[dict]) -> Path:
    current = [row for row in summary_rows if row["role"] == "current"]
    baseline = next(row for row in summary_rows if row["key"] == "ofmpc")
    x = np.arange(len(current))
    fig, ax1 = plt.subplots(figsize=(9.8, 5.2))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"]
    ax1.bar(x, [row["tail_reward"] for row in current], color=colors, alpha=0.88)
    ax1.axhline(baseline["tail_reward"], color="#333333", linestyle="--", linewidth=1.0, label="OF-MPC reward")
    ax1.set_ylabel("tail-20 reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["family"] for row in current])
    ax1.grid(axis="y", alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["tail_temp_mae"] for row in current], color="#222222", marker="o", label="T85 MAE")
    ax2.axhline(baseline["tail_temp_mae"], color="#777777", linestyle=":", linewidth=1.0, label="OF-MPC T85 MAE")
    ax2.set_ylabel("tail T85 MAE")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Current limited-horizon SG runs: reward and temperature tracking")
    fig.tight_layout()
    out = OUT_DIR / "fig_current_tail_reward_tracking.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_current_vs_reference(comparison_rows: list[dict]) -> Path:
    labels = [row["family"] for row in comparison_rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9.4, 4.9))
    ax1.bar(x, [row["tail_reward_delta"] for row in comparison_rows], color="#4E79A7", alpha=0.86)
    ax1.axhline(0.0, color="#333333", linewidth=0.9)
    ax1.set_ylabel("tail reward change")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["tail_temp_mae_delta"] for row in comparison_rows], color="#E15759", marker="o")
    ax2.axhline(0.0, color="#E15759", linestyle=":", linewidth=0.9)
    ax2.set_ylabel("tail T85 MAE change")
    ax1.set_title("Current run change relative to the closest previous mismatch reference")
    fig.tight_layout()
    out = OUT_DIR / "fig_current_vs_previous_delta.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_episode_rewards(episode_rows: list[dict]) -> Path:
    current_keys = {"weights_current", "residual_current", "markov_current", "horizon_limited", "ofmpc"}
    rows = [row for row in episode_rows if row["key"] in current_keys]
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    colors = {
        "ofmpc": "#333333",
        "weights_current": "#4C78A8",
        "residual_current": "#59A14F",
        "markov_current": "#F28E2B",
        "horizon_limited": "#B07AA1",
    }
    for key in ["ofmpc", "weights_current", "residual_current", "markov_current", "horizon_limited"]:
        series = [row for row in rows if row["key"] == key]
        if not series:
            continue
        ax.plot(
            [row["subepisode"] for row in series],
            [row["reward"] for row in series],
            label=series[0]["label"],
            color=colors.get(key),
            linewidth=1.15 if key != "ofmpc" else 1.0,
            alpha=0.95 if key != "ofmpc" else 0.7,
        )
    ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.25, label="warm start")
    ax.axvspan(WARM_START_SUBEPISODES + 1, LIVE_RELEASE_SUBEPISODE - 1, color="#f3c567", alpha=0.20, label="handover")
    ax.axvline(LIVE_RELEASE_SUBEPISODE, color="#8E3B46", linestyle=":", linewidth=1.0, label="live release")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.7)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("current reward")
    ax.set_title("Episode reward histories for current SG runs")
    ax.grid(alpha=0.22)
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "fig_current_episode_rewards.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_gate_fractions(episode_rows: list[dict]) -> Path:
    keys = ["weights_current", "residual_current", "markov_current", "horizon_limited"]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    colors = {
        "weights_current": "#4C78A8",
        "residual_current": "#59A14F",
        "markov_current": "#F28E2B",
        "horizon_limited": "#B07AA1",
    }
    for key in keys:
        rows = [row for row in episode_rows if row["key"] == key]
        if not rows:
            continue
        x = np.asarray([row["subepisode"] for row in rows], float)
        y = np.asarray([row["sg_policy_step_frac"] for row in rows], float)
        if y.size >= 5:
            kernel = np.ones(5) / 5.0
            y_smooth = np.convolve(np.nan_to_num(y, nan=0.0), kernel, mode="same")
        else:
            y_smooth = y
        ax.plot(x, y_smooth, label=rows[0]["label"], color=colors[key], linewidth=1.25)
    ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.25)
    ax.axvspan(WARM_START_SUBEPISODES + 1, LIVE_RELEASE_SUBEPISODE - 1, color="#f3c567", alpha=0.20)
    ax.axvline(LIVE_RELEASE_SUBEPISODE, color="#8E3B46", linestyle=":", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("policy-selected step fraction, 5-episode moving average")
    ax.set_title("Supervisor-gate release behavior")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    out = OUT_DIR / "fig_current_gate_policy_fraction.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _heatmap_data(bundle: dict, sl: slice) -> tuple[np.ndarray, list[int], list[int]]:
    trace = np.asarray(bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace", [])), float)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return np.zeros((0, 0)), [], []
    n = min(trace.shape[0], int(bundle.get("nFE", trace.shape[0])))
    view = trace[:n, :][slice(max(0, sl.start or 0), min(n, sl.stop or n)), :]
    view = view[np.all(np.isfinite(view), axis=1), :]
    if view.size == 0:
        return np.zeros((0, 0)), [], []
    pairs = [tuple(map(int, row)) for row in view]
    counts = Counter(pairs)
    nps = sorted({p[0] for p in pairs})
    ncs = sorted({p[1] for p in pairs})
    heat = np.zeros((len(ncs), len(nps)), dtype=float)
    total = sum(counts.values())
    for j, np_h in enumerate(nps):
        for i, nc_h in enumerate(ncs):
            heat[i, j] = counts.get((np_h, nc_h), 0) / total if total else 0.0
    return heat, nps, ncs


def _plot_horizon_heatmaps(bundles: dict[str, dict]) -> Path:
    keys = ["horizon_limited", "horizon_wide", "dueling_wide"]
    titles = ["SG-DQN limited 39", "SG-DQN wide 87", "Dueling SG-DQN wide 87"]
    fig, axes = plt.subplots(1, len(keys), figsize=(13.2, 4.4), constrained_layout=True)
    vmax = 0.0
    data = []
    for key in keys:
        bundle = bundles[key]
        heat, nps, ncs = _heatmap_data(bundle, tail_slice(bundle, TAIL_EPISODES))
        data.append((heat, nps, ncs))
        if heat.size:
            vmax = max(vmax, float(np.max(heat)))
    for ax, key, title, (heat, nps, ncs) in zip(axes, keys, titles, data):
        if not heat.size:
            ax.set_axis_off()
            continue
        im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=max(vmax, 1.0e-9))
        ax.set_xticks(np.arange(len(nps)))
        ax.set_xticklabels(nps)
        ax.set_yticks(np.arange(len(ncs)))
        ax.set_yticklabels(ncs)
        ax.set_xlabel("Np")
        ax.set_ylabel("Nc")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.82, label="tail frequency")
    out = OUT_DIR / "fig_horizon_tail_heatmaps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _worst_current_episode_rows(episode_rows: list[dict]) -> list[dict]:
    out = []
    for key in ["weights_current", "residual_current", "markov_current", "horizon_limited"]:
        candidates = [
            row
            for row in episode_rows
            if row["key"] == key and int(row["subepisode"]) > WARM_START_SUBEPISODES
        ]
        if not candidates:
            continue
        worst = min(candidates, key=lambda row: float(row["reward"]))
        out.append(worst)
    return out


def _top_pair_strings(horizon_pair_rows: list[dict], key: str, window: str = "tail", limit: int = 5) -> str:
    rows = [
        row
        for row in horizon_pair_rows
        if row["key"] == key and row["window"] == window and int(row["rank"]) <= limit
    ]
    if not rows:
        return ""
    return "; ".join(
        f"({int(row['Np'])}, {int(row['Nc'])}) at {_fmt(row['fraction'] * 100.0, 1)} percent"
        for row in rows
    )


def _build_report(
    summary_rows: list[dict],
    comparison_rows: list[dict],
    episode_rows: list[dict],
    horizon_pair_rows: list[dict],
    figures: list[Path],
) -> str:
    by_key = {row["key"]: row for row in summary_rows}
    current = [row for row in summary_rows if row["role"] == "current"]
    ofmpc = by_key["ofmpc"]
    best_tail = max(current, key=lambda row: row["tail_reward"])
    safest = min(current, key=lambda row: (row["negative_postwarm_episodes"], -row["tail_reward"]))
    horizon = by_key["horizon_limited"]
    horizon_ref = by_key["horizon_wide"]
    dueling_note = (
        "No saved limited-grid dueling-DQN mismatch bundle was found under `Distillation/Results`; "
        "the current dueling row is therefore the older wide-grid June 4 context run only."
    )

    lines: list[str] = []
    lines.append("# Distillation SG-TD3 and Limited-Horizon SG-DQN Results")
    lines.append("")
    lines.append("Generated on 2026-06-06 from saved result bundles only; Aspen was not relaunched.")
    lines.append("")
    lines.append("## Executive Takeaways")
    lines.append("")
    lines.append(
        f"- Best tail performance is **{best_tail['label']}** with tail reward {_fmt(best_tail['tail_reward'])}, "
        f"which is {_fmt(best_tail['delta_vs_ofmpc'])} above OF-MPC."
    )
    lines.append(
        f"- The most stable current run by post-warm failures is **{safest['label']}**: "
        f"{safest['negative_postwarm_episodes']} negative post-warm episodes and tail reward {_fmt(safest['tail_reward'])}."
    )
    lines.append(
        f"- Limited-grid SG-DQN improved over the old wide-grid SG-DQN by "
        f"{_fmt(horizon['tail_reward'] - horizon_ref['tail_reward'])} reward points, reduced T85 MAE by "
        f"{_fmt(horizon_ref['tail_temp_mae'] - horizon['tail_temp_mae'])}, and removed the previous negative post-warm episode."
    )
    lines.append(
        "- The limited horizon grid is not just smaller; the saved recipes are triangular: "
        "`Np = 6..11` and `Nc = 3..Np`, giving 39 valid actions rather than a full Cartesian 54-action grid."
    )
    lines.append(f"- {dueling_note}")
    lines.append("")
    lines.append("## Method Snapshot")
    lines.append("")
    lines.append(
        "All reported rewards are recomputed from the saved trajectories with the current distillation reward, "
        "so runs with different logged reward revisions are compared on the same scale."
    )
    lines.append("")
    lines.append(
        "$$ r_t = -e_t^\\top Q e_t - \\Delta u_t^\\top R \\Delta u_t + b(e_t, y_{\\mathrm{sp},t}), $$"
    )
    lines.append("")
    lines.append(
        "where the report uses the saved scaled-deviation output errors, scaled input moves, and physical setpoints "
        "to reconstruct the same band-gated bonus term used by the current distillation reward code."
    )
    lines.append("")
    lines.append(
        "SG-TD3 weights, residual, and Markov runs use continuous actors with twin critics and a supervisor gate. "
        "The current runs keep fixed MPC horizons `(6, 3)`. Horizon SG-DQN is discrete: the agent selects an `(Np, Nc)` "
        "recipe and the supervisor gate compares the learned Q score for the candidate action against the OF-MPC/default supervisor action. "
        "The dueling DQN variant changes the Q-network decomposition, not the fact that the gate is still single-Q discrete."
    )
    lines.append("")
    lines.append("## Current Run Summary")
    lines.append("")
    lines.append("| Run | Tail reward | Delta vs OF-MPC | Final reward | Worst post-warm | Neg post-warm | T85 MAE | x24 MAE | Policy step frac tail | Policy step frac first live |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in [ofmpc, *current]:
        lines.append(
            f"| {row['label']} | {_fmt(row['tail_reward'])} | {_fmt(row['delta_vs_ofmpc'])} | "
            f"{_fmt(row['final_reward'])} | {_fmt(row['worst_postwarm_reward'])} | "
            f"{row['negative_postwarm_episodes']} | {_fmt(row['tail_temp_mae'])} | {_fmt(row['tail_comp_mae'], 5)} | "
            f"{_fmt(row.get('tail_sg_policy_step_frac', float('nan')))} | "
            f"{_fmt(row.get('first_live_sg_policy_step_frac', float('nan')))} |"
        )
    lines.append("")
    lines.append("## Change Against Previous Mismatch References")
    lines.append("")
    lines.append("| Family | Tail reward change | T85 MAE change | x24 MAE change | Neg post-warm change | Tail policy step change | First-live policy step change |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in comparison_rows:
        lines.append(
            f"| {row['family']} | {_fmt(row['tail_reward_delta'])} | {_fmt(row['tail_temp_mae_delta'])} | "
            f"{_fmt(row['tail_comp_mae_delta'], 5)} | {row['negative_postwarm_delta']} | "
            f"{_fmt(row['policy_step_frac_delta'])} | {_fmt(row['first_live_policy_step_frac_delta'])} |"
        )
    lines.append("")
    lines.append("Positive reward change is good; negative T85 or x24 MAE change is good.")
    lines.append("")
    lines.append("## Family Interpretation")
    lines.append("")
    lines.append("### Weights SG-TD3")
    weights = by_key["weights_current"]
    weights_ref = by_key["weights_previous"]
    lines.append("")
    lines.append(
        f"The weights supervisor is the safest successful continuous run. Tail reward rose from "
        f"{_fmt(weights_ref['tail_reward'])} to {_fmt(weights['tail_reward'])}, with zero negative post-warm episodes in both runs. "
        f"The tail gate became more policy-forward, increasing from {_fmt(weights_ref['tail_sg_policy_step_frac'])} to "
        f"{_fmt(weights['tail_sg_policy_step_frac'])}, while the first-live gate stayed conservative "
        f"({_fmt(weights['first_live_sg_policy_step_frac'])})."
    )
    lines.append("")
    lines.append("### Residual SG-TD3")
    residual = by_key["residual_current"]
    residual_ref = by_key["residual_previous"]
    lines.append("")
    lines.append(
        f"Residual is the strongest tail performer but not the cleanest handover. Tail reward improved from "
        f"{_fmt(residual_ref['tail_reward'])} to {_fmt(residual['tail_reward'])}, and T85 MAE improved to "
        f"{_fmt(residual['tail_temp_mae'])}. The cost is fragility: negative post-warm episodes increased from "
        f"{residual_ref['negative_postwarm_episodes']} to {residual['negative_postwarm_episodes']}, with the worst collapse at "
        f"{_fmt(residual['worst_postwarm_reward'])}. This matches the residual-action risk we expected: once accepted, "
        "a residual move can directly perturb the plant input rather than only reshaping the MPC objective."
    )
    lines.append("")
    lines.append("### Markov SG-TD3")
    markov = by_key["markov_current"]
    markov_ref = by_key["markov_previous"]
    lines.append("")
    lines.append(
        f"Markov improved modestly over the previous Markov reference: tail reward {_fmt(markov_ref['tail_reward'])} to "
        f"{_fmt(markov['tail_reward'])}, T85 MAE {_fmt(markov_ref['tail_temp_mae'])} to {_fmt(markov['tail_temp_mae'])}. "
        f"It remains conservative at release, with first-live policy step fraction only {_fmt(markov['first_live_sg_policy_step_frac'])}. "
        "The saved bundle reports top-level `state_mode='standard'`, but the Markov-specific fields are "
        f"`markov_state_mode='{markov['markov_state_mode']}'` and `markov_agent_state_features='{markov['markov_agent_state_features']}'`; "
        "therefore the run should be interpreted as the Markov/mismatch-conditioned param-noise run, with a generic top-level logging ambiguity."
    )
    lines.append("")
    lines.append("### Horizon SG-DQN")
    lines.append("")
    lines.append(
        f"The limited 39-action horizon DQN is a real improvement over the wide 87-action SG-DQN: tail reward "
        f"{_fmt(horizon_ref['tail_reward'])} to {_fmt(horizon['tail_reward'])}, T85 MAE "
        f"{_fmt(horizon_ref['tail_temp_mae'])} to {_fmt(horizon['tail_temp_mae'])}, and post-warm negative episodes "
        f"{horizon_ref['negative_postwarm_episodes']} to {horizon['negative_postwarm_episodes']}. "
        "The heatmap still shows a diffuse policy: all 39 recipes appear in the tail, and the most common pair only occupies "
        f"{_fmt(horizon['tail_horizon_top_pair_frac'])} of tail steps. So the narrowed range helped stability, but did not yet create a confident horizon policy."
    )
    lines.append("")
    lines.append("## Failure Diagnostics")
    lines.append("")
    lines.append(
        "The worst current post-warm episodes show two different failure modes. Residual has a real handover shock; "
        "Markov has a later disturbance-region dip while the gate is still mostly supervisor-led; weights and limited SG-DQN do not collapse below zero."
    )
    lines.append("")
    lines.append("| Run | Worst subepisode | Reward | T85 MAE | x24 MAE | Policy step frac | Median gate advantage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in _worst_current_episode_rows(episode_rows):
        lines.append(
            f"| {row['label']} | {int(row['subepisode'])} | {_fmt(row['reward'])} | "
            f"{_fmt(row['temp_mae'])} | {_fmt(row['comp_mae'], 5)} | "
            f"{_fmt(row.get('sg_policy_step_frac', float('nan')))} | {_fmt(row.get('sg_adv_median', float('nan')))} |"
        )
    lines.append("")
    lines.append(
        "This is why the residual tail result should not be read as universally safer than weights: residual eventually learns the best tail behavior, "
        "but its first live window accepts enough direct input residuals to create the worst short-term collapse. Markov is more conservative at release, "
        "so its weak episodes look more like insufficient correction around a harder disturbance region than a gate handover failure."
    )
    lines.append("")
    lines.append("## Horizon Candidate Diagnostics")
    lines.append("")
    lines.append("| Run | Tail top pairs |")
    lines.append("|---|---|")
    for key in ["horizon_limited", "horizon_wide", "dueling_wide"]:
        row = by_key[key]
        lines.append(f"| {row['label']} | {_top_pair_strings(horizon_pair_rows, key)} |")
    lines.append("")
    lines.append(
        "The limited run shifts mass toward short prediction horizons, especially `(6, 4)`, `(6, 6)`, and `(6, 3)`. "
        "The old wide SG-DQN spent tail probability on both very short and very long pairs, while the old dueling run concentrated on `(6, 3)` "
        "and `(11, 11)` but still had poor tracking. That argues for a reduced candidate set, not for returning to the full wide grid."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig_path in figures:
        lines.append(f"- [{fig_path.name}]({rel(fig_path)})")
    lines.append("")
    lines.append("## Recommended Next Recipes")
    lines.append("")
    lines.append(
        "For the next horizon-only distillation pass, keep the reduced lower bound and avoid returning to the full 87-action grid. "
        "The current data support `Np = 6..11` and `Nc = 3..Np` as a better candidate set than the old wide grid. "
        "A slightly more focused follow-up is worth testing: `Np = 6..10`, `Nc = 3..min(Np, 7)`, while keeping `(6, 3)` as the supervisor/default action. "
        "That keeps the pairs used most often by the successful limited run and removes some high-control-horizon actions that still appear exploratory rather than decisively useful."
    )
    lines.append("")
    lines.append("For continuous supervisors, the current ranking is:")
    lines.append("")
    lines.append("- Residual SG-TD3 is best for final/tail tracking but needs acceptance judged with the negative-episode risk visible.")
    lines.append("- Weights SG-TD3 is the cleanest stable candidate and the easiest to defend as a robust improvement.")
    lines.append("- Markov SG-TD3 is useful and conservative, but its state-mode logging should be cleaned before using the bundle as final paper evidence.")
    lines.append("- Horizon SG-DQN limited is improved but still behind the continuous supervisors.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("Files inspected:")
    lines.append("")
    inspected = [
        "report/distillation_standard_mode_sg_analysis_2026_06_05.md",
        "report/scripts/analyze_distillation_standard_mode_sg_20260605.py",
        "report/scripts/analyze_distillation_dueling_horizon_history_20260604.py",
        "report/scripts/analyze_distillation_sg_dqn_horizon_followup_20260604.py",
        "distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py",
        "distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py",
        "distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py",
        "tests/test_supervisor_gated_horizon_runners.py",
    ]
    for item in inspected:
        lines.append(f"- `{item}`")
    for row in summary_rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    lines.append("Generated outputs:")
    lines.append("")
    for item in [SUMMARY_CSV, SUMMARY_JSON, EPISODE_CSV, HORIZON_PAIR_CSV, *figures]:
        lines.append(f"- `{rel(item)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first_bundle = load_pickle(RUNS[0]["path"])
    ofmpc_tail = finite_mean(episode_average(reward_step_series(first_bundle, CURRENT_REWARD), first_bundle)[-TAIL_EPISODES:])
    summary_rows: list[dict] = []
    episode_rows: list[dict] = []
    horizon_pair_rows: list[dict] = []
    bundles: dict[str, dict] = {}
    rewards_by_key: dict[str, np.ndarray] = {}

    for cfg in RUNS:
        if not cfg["path"].exists():
            print(f"missing: {cfg['path']}")
            continue
        row, bundle, rewards = _row_for_run(cfg, ofmpc_tail)
        summary_rows.append(row)
        bundles[cfg["key"]] = bundle
        rewards_by_key[cfg["key"]] = rewards
        episode_rows.extend(_episode_rows_for_run(cfg, bundle, rewards))
        if bundle.get("horizon_recipes") is not None:
            horizon_pair_rows.extend(
                _horizon_pair_rows(cfg["key"], cfg["label"], bundle, tail_slice(bundle, TAIL_EPISODES), "tail")
            )
            horizon_pair_rows.extend(
                _horizon_pair_rows(
                    cfg["key"],
                    cfg["label"],
                    bundle,
                    step_slice(bundle, WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES, WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES + 20),
                    "first_live",
                )
            )

    comparison_rows = _comparison_rows(summary_rows)
    figures = [
        _plot_current_tail(summary_rows),
        _plot_current_vs_reference(comparison_rows),
        _plot_episode_rewards(episode_rows),
        _plot_gate_fractions(episode_rows),
        _plot_horizon_heatmaps(bundles),
    ]

    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(EPISODE_CSV, episode_rows)
    _write_csv(HORIZON_PAIR_CSV, horizon_pair_rows)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "comparisons": comparison_rows,
                "figures": [rel(path) for path in figures],
                "tail_episodes": TAIL_EPISODES,
                "warm_start_subepisodes": WARM_START_SUBEPISODES,
                "handover_subepisodes": HANDOVER_SUBEPISODES,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        _build_report(summary_rows, comparison_rows, episode_rows, horizon_pair_rows, figures),
        encoding="utf-8",
    )
    print(f"wrote {rel(REPORT_PATH)}")
    print(f"wrote {rel(SUMMARY_CSV)}")
    print(f"wrote {len(figures)} figures under {rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
