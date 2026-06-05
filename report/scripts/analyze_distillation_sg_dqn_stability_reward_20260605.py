"""Analyze June 5 distillation SG-DQN horizon stability versus reward.

This script reads saved result bundles only. It compares the new reduced-grid
SG-DQN run and the Aspen-6 legacy-reward SG-dueling check against the June 4
SG-DQN/SG-dueling runs, OF-MPC, and one stable historical dueling reference.
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
    LEGACY_HORIZON_REWARD,
    OFMPC_PATH,
    TAIL_EPISODES,
    analyze_run,
    baseline_summary,
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
    write_csv,
)
from report.scripts.analyze_distillation_sg_dqn_horizon_followup_20260604 import (  # noqa: E402
    _compare_rewards,
    _release_episode,
    _sg_window_metrics,
)


OUT_DIR = ROOT / "report" / "figures" / "distillation_sg_dqn_stability_reward_20260605"

RUNS = {
    "ofmpc": {
        "label": "OF-MPC",
        "kind": "baseline",
        "path": OFMPC_PATH,
        "compare_path": None,
    },
    "june4_sg_dqn_87_mismatch": {
        "label": "June4 SG-DQN 87 mismatch",
        "kind": "sg_dqn_reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch"
        / "20260604_103231"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation"
        / "20260604_103242"
        / "input_data.pkl",
    },
    "june5_sg_dqn_39_standard": {
        "label": "June5 SG-DQN 39 standard",
        "kind": "sg_dqn_reduced_grid",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard_np6_11_nc3_11"
        / "20260605_105331"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard_np6_11_nc3_11"
        / "20260605_105340"
        / "input_data.pkl",
    },
    "june4_sg_dueling_87_mismatch": {
        "label": "June4 SG-dueling 87 mismatch",
        "kind": "sg_dueling_reference",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch"
        / "20260604_105140"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation"
        / "20260604_105151"
        / "input_data.pkl",
    },
    "june5_sg_dueling_aspen6_legacy_87_standard": {
        "label": "June5 SG-dueling Aspen6 legacy 87 standard",
        "kind": "sg_dueling_legacy_reward",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_sg_dqn_aspen6_legacyreward_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard"
        / "20260605_105758"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_dueling_horizon_sg_dqn_aspen6_legacyreward_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard"
        / "20260605_105807"
        / "input_data.pkl",
    },
    "stable_dueling_20260511": {
        "label": "Stable dueling 20260511",
        "kind": "historical_dueling",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260511_131656"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_dueling_horizon_disturb_fluctuation_mismatch"
        / "20260511_131708"
        / "input_data.pkl",
    },
}


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, float).reshape(-1)
    return arr[np.isfinite(arr)]


def _episode_window(bundle: dict, start_episode: int, length_episodes: int) -> slice:
    return step_slice(bundle, start_episode, start_episode + length_episodes)


def _top_pairs(bundle: dict, sl: slice, key: str, window_name: str, limit: int = 8) -> list[dict]:
    trace = np.asarray(bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace", [])), float)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return []
    n = min(trace.shape[0], int(bundle.get("nFE", trace.shape[0])))
    start = max(0, sl.start or 0)
    stop = min(n, sl.stop or n)
    view = trace[:n, :][start:stop, :]
    view = view[np.all(np.isfinite(view), axis=1), :]
    pairs = [tuple(map(int, row)) for row in view]
    total = len(pairs)
    rows = []
    for rank, (pair, count) in enumerate(Counter(pairs).most_common(limit), start=1):
        rows.append(
            {
                "run": key,
                "window": window_name,
                "rank": rank,
                "pair": str(pair),
                "count": int(count),
                "fraction": float(count / total) if total else float("nan"),
            }
        )
    return rows


def _baseline_row() -> dict:
    bundle = load_pickle(OFMPC_PATH)
    current_avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    legacy_avg = episode_average(reward_step_series(bundle, LEGACY_HORIZON_REWARD), bundle)
    post = slice(10, current_avg.size)
    tail_steps = tail_slice(bundle, TAIL_EPISODES)
    tracking = tracking_metrics(bundle, CURRENT_REWARD, tail_steps)
    row = {
        "key": "ofmpc",
        "label": "OF-MPC",
        "kind": "baseline",
        "path": rel(OFMPC_PATH),
        "compare_path": "",
        "agent_kind": "ofmpc",
        "state_mode": "",
        "notebook_source": "",
        "saved_reward_family": "current",
        "recipe_count": float("nan"),
        "tail_current_reward": finite_mean(current_avg[-TAIL_EPISODES:]),
        "tail_legacy_reward": finite_mean(legacy_avg[-TAIL_EPISODES:]),
        "delta_vs_ofmpc": 0.0,
        "final_current_reward": float(current_avg[-1]),
        "worst_postwarm_current_reward": float(np.min(current_avg[post])),
        "negative_postwarm_episodes": int(np.sum(current_avg[post] < 0.0)),
        "tail_comp_mae": tracking["comp_mae"],
        "tail_temp_mae": tracking["temp_mae"],
        "tail_outside_band_frac": tracking["outside_band_frac"],
        "tail_mean_abs_du_scaled": tracking["mean_abs_du_scaled"],
        "tail_unique_pairs": float("nan"),
        "tail_top_pair": "",
        "tail_top_pair_frac": float("nan"),
        "tail_default_pair_frac": float("nan"),
        "tail_switch_frac": float("nan"),
        "tail_mean_predict": float("nan"),
        "tail_mean_control": float("nan"),
        "tail_sg_policy_decision_frac": float("nan"),
        "tail_sg_adv_median": float("nan"),
        "first_live_current_reward": float("nan"),
        "first_live_temp_mae": float("nan"),
        "first_live_default_pair_frac": float("nan"),
        "compare_tail_reward": float("nan"),
        "compare_mpc_tail_reward": float("nan"),
    }
    return row


def _row_for_run(key: str, cfg: dict, ofmpc_current_tail: float, ofmpc_legacy_tail: float) -> tuple[dict, list[dict]]:
    bundle = load_pickle(cfg["path"])
    analyzed = analyze_run(cfg["path"], ofmpc_current_tail, ofmpc_legacy_tail)
    current_avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    post = slice(10, current_avg.size)
    release_ep = _release_episode(bundle)
    first_live_ep = slice(release_ep, min(current_avg.size, release_ep + 20))
    first_live_steps = _episode_window(bundle, release_ep, 20)
    tail_steps = tail_slice(bundle, TAIL_EPISODES)
    first_tracking = tracking_metrics(bundle, CURRENT_REWARD, first_live_steps)
    tail_horizon = horizon_stats(bundle, tail_steps)
    first_horizon = horizon_stats(bundle, first_live_steps)
    compare = _compare_rewards(cfg["compare_path"])
    saved_q2 = float(analyzed.get("saved_q2", np.nan))
    saved_reward_family = "legacy" if saved_q2 < 5000.0 else "current"

    row = {
        "key": key,
        "label": cfg["label"],
        "kind": cfg["kind"],
        "path": rel(cfg["path"]),
        "compare_path": rel(cfg["compare_path"]) if cfg["compare_path"] is not None else "",
        "agent_kind": bundle.get("agent_kind", ""),
        "state_mode": analyzed["state_mode"],
        "notebook_source": analyzed["notebook_source"],
        "saved_reward_family": saved_reward_family,
        "recipe_count": analyzed["tail_recipe_count"],
        "tail_current_reward": analyzed["tail_current_reward"],
        "tail_legacy_reward": analyzed["tail_legacy_reward"],
        "delta_vs_ofmpc": analyzed["current_delta_vs_ofmpc"],
        "final_current_reward": analyzed["final_current_reward"],
        "worst_postwarm_current_reward": float(np.min(current_avg[post])) if current_avg.size > 10 else float("nan"),
        "negative_postwarm_episodes": int(np.sum(current_avg[post] < 0.0)) if current_avg.size > 10 else float("nan"),
        "tail_comp_mae": analyzed["tail_current_comp_mae"],
        "tail_temp_mae": analyzed["tail_current_temp_mae"],
        "tail_outside_band_frac": analyzed["tail_current_outside_band_frac"],
        "tail_mean_abs_du_scaled": analyzed["tail_current_mean_abs_du_scaled"],
        "tail_unique_pairs": tail_horizon["unique_pairs"],
        "tail_top_pair": tail_horizon["top_pair"],
        "tail_top_pair_frac": tail_horizon["top_pair_frac"],
        "tail_default_pair_frac": tail_horizon["default_pair_frac"],
        "tail_switch_frac": tail_horizon["switch_frac"],
        "tail_mean_predict": tail_horizon["mean_predict"],
        "tail_mean_control": tail_horizon["mean_control"],
        "first_live_current_reward": finite_mean(current_avg[first_live_ep]),
        "first_live_temp_mae": first_tracking["temp_mae"],
        "first_live_default_pair_frac": first_horizon["default_pair_frac"],
        "epsilon_tail": analyzed["epsilon_tail"],
        "epsilon_final": float(_finite(bundle.get("epsilon_trace", [np.nan]))[-1])
        if _finite(bundle.get("epsilon_trace", [])).size
        else float("nan"),
    }
    row.update(compare)
    row.update({f"tail_{metric}": value for metric, value in _sg_window_metrics(bundle, tail_steps).items()})
    row.update({f"first_live_{metric}": value for metric, value in _sg_window_metrics(bundle, first_live_steps).items()})

    top_rows = []
    top_rows.extend(_top_pairs(bundle, first_live_steps, key, "first_live_20"))
    top_rows.extend(_top_pairs(bundle, tail_steps, key, "tail_20"))
    return row, top_rows


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    x = np.asarray(x_values, float)
    y = np.asarray(y_values, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _correlation_rows(rows: list[dict]) -> list[dict]:
    method_rows = [row for row in rows if row["key"] != "ofmpc"]
    targets = {
        "tail_unique_pairs": "unique tail pairs",
        "tail_top_pair_frac": "top-pair fraction",
        "tail_default_pair_frac": "default-pair fraction",
        "tail_switch_frac": "switch fraction",
        "tail_temp_mae": "tail T85 MAE",
        "negative_postwarm_episodes": "negative post-warm episodes",
        "worst_postwarm_current_reward": "worst post-warm reward",
        "tail_sg_policy_decision_frac": "policy decision fraction",
        "tail_sg_adv_median": "tail SG advantage median",
    }
    return [
        {
            "metric": label,
            "key": key,
            "pearson_vs_tail_current_reward": _pearson(
                [float(row.get(key, np.nan)) for row in method_rows],
                [float(row["tail_current_reward"]) for row in method_rows],
            ),
            "n": int(
                np.sum(
                    np.isfinite([float(row.get(key, np.nan)) for row in method_rows])
                    & np.isfinite([float(row["tail_current_reward"]) for row in method_rows])
                )
            ),
        }
        for key, label in targets.items()
    ]


def _plot_reward_curves(rows: list[dict]) -> Path:
    keys = [
        "ofmpc",
        "june4_sg_dqn_87_mismatch",
        "june5_sg_dqn_39_standard",
        "june4_sg_dueling_87_mismatch",
        "june5_sg_dueling_aspen6_legacy_87_standard",
    ]
    by_key = {row["key"]: row for row in rows}
    colors = {
        "ofmpc": "#555555",
        "june4_sg_dqn_87_mismatch": "#1f78b4",
        "june5_sg_dqn_39_standard": "#33a02c",
        "june4_sg_dueling_87_mismatch": "#7b3294",
        "june5_sg_dueling_aspen6_legacy_87_standard": "#f28e2b",
    }
    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    for key in keys:
        bundle = load_pickle(RUNS[key]["path"])
        avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
        ax.plot(np.arange(1, avg.size + 1), avg, label=by_key[key]["label"], color=colors[key], linewidth=1.2)
    ax.axvline(10, color="#222222", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.axvline(13, color="#222222", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("current-reward episode average")
    ax.set_title("June 5 distillation SG-DQN horizon reward curves")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    out = OUT_DIR / "fig_june5_reward_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_stability_reward(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] != "ofmpc"]
    colors = {
        "sg_dqn_reference": "#1f78b4",
        "sg_dqn_reduced_grid": "#33a02c",
        "sg_dueling_reference": "#7b3294",
        "sg_dueling_legacy_reward": "#f28e2b",
        "historical_dueling": "#4daf4a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    ax = axes[0]
    for row in plot_rows:
        ax.scatter(
            row["tail_temp_mae"],
            row["tail_current_reward"],
            s=85 + 120 * max(float(row["tail_default_pair_frac"]), 0.0),
            color=colors.get(row["kind"], "#999999"),
            edgecolor="white",
            linewidth=0.8,
            label=row["label"],
            alpha=0.88,
        )
        ax.annotate(row["label"].replace("June", "J"), (row["tail_temp_mae"], row["tail_current_reward"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("tail T85 MAE")
    ax.set_ylabel("tail current reward")
    ax.set_title("Reward mainly follows T85 stability")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for row in plot_rows:
        ax.scatter(
            row["tail_switch_frac"],
            row["tail_current_reward"],
            s=85 + 140 * max(float(row["tail_default_pair_frac"]), 0.0),
            color=colors.get(row["kind"], "#999999"),
            edgecolor="white",
            linewidth=0.8,
            label=row["label"],
            alpha=0.88,
        )
        ax.annotate(row["tail_top_pair"], (row["tail_switch_frac"], row["tail_current_reward"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("tail horizon switch fraction")
    ax.set_ylabel("tail current reward")
    ax.set_title("Horizon concentration helps, but is not sufficient")
    ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out = OUT_DIR / "fig_june5_reward_vs_stability.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_tail_bars(rows: list[dict]) -> Path:
    keys = [
        "ofmpc",
        "june4_sg_dqn_87_mismatch",
        "june5_sg_dqn_39_standard",
        "june4_sg_dueling_87_mismatch",
        "june5_sg_dueling_aspen6_legacy_87_standard",
    ]
    plot_rows = [next(row for row in rows if row["key"] == key) for key in keys]
    labels = [row["label"].replace(" ", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    colors = ["#555555", "#1f78b4", "#33a02c", "#7b3294", "#f28e2b"]
    fig, ax1 = plt.subplots(figsize=(10.0, 4.8))
    ax1.bar(x, [row["tail_current_reward"] for row in plot_rows], color=colors, alpha=0.86, label="tail reward")
    ax1.axhline(plot_rows[0]["tail_current_reward"], color="#333333", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("tail-20 current reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["negative_postwarm_episodes"] for row in plot_rows], color="#cb181d", marker="o", label="negative post-warm episodes")
    ax2.set_ylabel("negative post-warm episodes")
    ax1.set_title("Reward improves when post-warm instability disappears")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "fig_june5_tail_reward_negative_episodes.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_top_pairs(top_pairs: list[dict]) -> Path:
    target_runs = {
        "june5_sg_dqn_39_standard": "SG-DQN 39",
        "june5_sg_dueling_aspen6_legacy_87_standard": "Legacy dueling 87",
    }
    target = [
        row
        for row in top_pairs
        if row["window"] == "tail_20" and row["run"] in target_runs and int(row["rank"]) <= 8
    ]
    labels = sorted({row["pair"] for row in target})
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, run, color in [
        (-width / 2, "june5_sg_dqn_39_standard", "#33a02c"),
        (width / 2, "june5_sg_dueling_aspen6_legacy_87_standard", "#f28e2b"),
    ]:
        values = []
        for label in labels:
            match = [row for row in target if row["run"] == run and row["pair"] == label]
            values.append(float(match[0]["fraction"]) if match else 0.0)
        ax.bar(x + offset, values, width, label=target_runs[run], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("tail fraction")
    ax.set_title("June 5 top executed tail horizon pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig_june5_top_tail_pairs.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_outputs() -> tuple[list[dict], list[dict], list[dict]]:
    ofmpc = baseline_summary(load_pickle(OFMPC_PATH))
    rows = [_baseline_row()]
    top_pairs: list[dict] = []
    for key in (
        "june4_sg_dqn_87_mismatch",
        "june5_sg_dqn_39_standard",
        "june4_sg_dueling_87_mismatch",
        "june5_sg_dueling_aspen6_legacy_87_standard",
        "stable_dueling_20260511",
    ):
        row, pair_rows = _row_for_run(
            key,
            RUNS[key],
            ofmpc["current_tail20_reward"],
            ofmpc["legacy_tail20_reward"],
        )
        rows.append(row)
        top_pairs.extend(pair_rows)
    return rows, _correlation_rows(rows), top_pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, correlations, top_pairs = build_outputs()
    write_csv(OUT_DIR / "june5_sg_dqn_stability_summary.csv", rows)
    write_csv(OUT_DIR / "june5_sg_dqn_stability_correlations.csv", correlations)
    write_csv(OUT_DIR / "june5_sg_dqn_stability_top_pairs.csv", top_pairs)
    figures = [
        _plot_reward_curves(rows),
        _plot_stability_reward(rows),
        _plot_tail_bars(rows),
        _plot_top_pairs(top_pairs),
    ]
    payload = {
        "rows": rows,
        "correlations": correlations,
        "top_pairs": top_pairs,
        "figures": [rel(path) for path in figures],
        "outputs": {
            "summary_csv": rel(OUT_DIR / "june5_sg_dqn_stability_summary.csv"),
            "correlations_csv": rel(OUT_DIR / "june5_sg_dqn_stability_correlations.csv"),
            "top_pairs_csv": rel(OUT_DIR / "june5_sg_dqn_stability_top_pairs.csv"),
        },
    }
    (OUT_DIR / "june5_sg_dqn_stability_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"figures": payload["figures"], "outputs": payload["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
