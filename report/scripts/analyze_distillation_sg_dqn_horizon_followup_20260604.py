"""Extend the distillation dueling-horizon report with SG-DQN follow-up metrics.

The script reads saved distillation horizon result bundles only. It reuses the
current-reward rescoring and physical tracking definitions from the dueling
horizon history report, then adds supervisor-gate and horizon-concentration
diagnostics for the latest SG-DQN and SG-dueling-DQN runs.
"""

from __future__ import annotations

import csv
import json
import pickle
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
    LEGACY_HORIZON_REWARD,
    OFMPC_PATH,
    OUT_DIR,
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


SOURCE_SUPERVISOR = 1
SOURCE_POLICY = 2
SOURCE_HELD = 3

RUNS = {
    "ofmpc": {
        "label": "OF-MPC",
        "kind": "baseline",
        "path": OFMPC_PATH,
        "compare_path": None,
    },
    "horizon_epsilon": {
        "label": "DDQN epsilon",
        "kind": "standard",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260603_130635"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_horizon_disturb_fluctuation_mismatch"
        / "20260603_130644"
        / "input_data.pkl",
    },
    "horizon_sg": {
        "label": "SG-DQN",
        "kind": "standard_sg",
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
    "dueling_epsilon": {
        "label": "Dueling epsilon",
        "kind": "dueling",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260603_130106"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_dueling_horizon_disturb_fluctuation_mismatch"
        / "20260603_130117"
        / "input_data.pkl",
    },
    "dueling_sg": {
        "label": "SG-dueling-DQN",
        "kind": "dueling_sg",
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
    "dueling_stable_history": {
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


def _window_from_episodes(bundle: dict, start_episode: int, length_episodes: int) -> slice:
    return step_slice(bundle, start_episode, start_episode + length_episodes)


def _release_episode(bundle: dict) -> int:
    warm_episode = int(bundle.get("warm_start_step", 10 * episode_len(bundle)) // episode_len(bundle))
    freeze = int(bundle.get("post_warm_start_action_freeze_subepisodes", 0) or 0)
    return warm_episode + freeze


def _compare_rewards(compare_path: Path | None) -> dict:
    if compare_path is None or not compare_path.exists():
        return {
            "compare_tail_reward": float("nan"),
            "compare_final_reward": float("nan"),
            "compare_worst_postwarm_reward": float("nan"),
            "compare_negative_postwarm_episodes": float("nan"),
            "compare_mpc_tail_reward": float("nan"),
        }
    bundle = load_pickle(compare_path)
    avg_rl = np.asarray(bundle.get("avg_rewards_rl", []), float)
    avg_mpc = np.asarray(bundle.get("avg_rewards_mpc", []), float)
    tail = slice(max(0, avg_rl.size - TAIL_EPISODES), avg_rl.size)
    post = slice(10, avg_rl.size)
    return {
        "compare_tail_reward": finite_mean(avg_rl[tail]) if avg_rl.size else float("nan"),
        "compare_final_reward": float(avg_rl[-1]) if avg_rl.size else float("nan"),
        "compare_worst_postwarm_reward": float(np.min(avg_rl[post])) if avg_rl.size > 10 else float("nan"),
        "compare_negative_postwarm_episodes": int(np.sum(avg_rl[post] < 0.0)) if avg_rl.size > 10 else float("nan"),
        "compare_mpc_tail_reward": finite_mean(avg_mpc[-TAIL_EPISODES:]) if avg_mpc.size else float("nan"),
    }


def _sg_window_metrics(bundle: dict, sl: slice) -> dict:
    selected = np.asarray(bundle.get("sg_selected_source_log", []), int)
    if selected.size == 0:
        return {
            "sg_policy_step_frac": float("nan"),
            "sg_supervisor_step_frac": float("nan"),
            "sg_held_step_frac": float("nan"),
            "sg_policy_decision_frac": float("nan"),
            "sg_decision_count": float("nan"),
            "sg_adv_mean": float("nan"),
            "sg_adv_median": float("nan"),
            "sg_adv_q10": float("nan"),
            "sg_adv_q90": float("nan"),
            "sg_adv_positive_frac": float("nan"),
        }
    selected = selected[sl]
    decision = (selected == SOURCE_POLICY) | (selected == SOURCE_SUPERVISOR)
    score_policy = np.asarray(bundle.get("sg_score_policy_log", []), float)[sl]
    score_supervisor = np.asarray(bundle.get("sg_score_supervisor_log", []), float)[sl]
    adv = score_policy - score_supervisor
    adv_decision = adv[decision & np.isfinite(adv)]
    denom = selected.size
    decision_count = int(np.sum(decision))
    return {
        "sg_policy_step_frac": float(np.sum(selected == SOURCE_POLICY) / denom) if denom else float("nan"),
        "sg_supervisor_step_frac": float(np.sum(selected == SOURCE_SUPERVISOR) / denom) if denom else float("nan"),
        "sg_held_step_frac": float(np.sum(selected == SOURCE_HELD) / denom) if denom else float("nan"),
        "sg_policy_decision_frac": (
            float(np.sum(selected == SOURCE_POLICY) / decision_count) if decision_count else float("nan")
        ),
        "sg_decision_count": decision_count,
        "sg_adv_mean": finite_mean(adv_decision),
        "sg_adv_median": float(np.median(adv_decision)) if adv_decision.size else float("nan"),
        "sg_adv_q10": float(np.quantile(adv_decision, 0.10)) if adv_decision.size else float("nan"),
        "sg_adv_q90": float(np.quantile(adv_decision, 0.90)) if adv_decision.size else float("nan"),
        "sg_adv_positive_frac": float(np.mean(adv_decision > 0.0)) if adv_decision.size else float("nan"),
    }


def _top_pairs(bundle: dict, sl: slice, run_key: str, window_name: str, limit: int = 8) -> list[dict]:
    trace = np.asarray(bundle.get("horizon_executed_trace_log", bundle.get("horizon_trace", [])), float)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return []
    n = min(trace.shape[0], int(bundle.get("nFE", trace.shape[0])))
    view = trace[:n, :][slice(max(0, sl.start or 0), min(n, sl.stop or n)), :]
    view = view[np.all(np.isfinite(view), axis=1), :]
    pairs = [tuple(map(int, row)) for row in view]
    total = len(pairs)
    rows = []
    for rank, (pair, count) in enumerate(Counter(pairs).most_common(limit), start=1):
        rows.append(
            {
                "run": run_key,
                "window": window_name,
                "rank": rank,
                "pair": str(pair),
                "count": int(count),
                "fraction": float(count / total) if total else float("nan"),
            }
        )
    return rows


def _row_for_baseline(ofmpc_bundle: dict) -> dict:
    avg = episode_average(reward_step_series(ofmpc_bundle, CURRENT_REWARD), ofmpc_bundle)
    tail_ep = slice(max(0, avg.size - TAIL_EPISODES), avg.size)
    post = slice(10, avg.size)
    tail_steps = tail_slice(ofmpc_bundle, TAIL_EPISODES)
    tracking = tracking_metrics(ofmpc_bundle, CURRENT_REWARD, tail_steps)
    row = {
        "key": "ofmpc",
        "label": "OF-MPC",
        "kind": "baseline",
        "path": rel(OFMPC_PATH),
        "tail_current_reward": finite_mean(avg[tail_ep]),
        "final_current_reward": float(avg[-1]),
        "worst_postwarm_current_reward": float(np.min(avg[post])),
        "negative_postwarm_episodes": int(np.sum(avg[post] < 0.0)),
        "tail_comp_mae": tracking["comp_mae"],
        "tail_temp_mae": tracking["temp_mae"],
        "tail_band_norm_mae": tracking["band_norm_mae"],
        "tail_outside_band_frac": tracking["outside_band_frac"],
        "tail_mean_abs_du_scaled": tracking["mean_abs_du_scaled"],
        "tail_unique_pairs": float("nan"),
        "tail_top_pair": "",
        "tail_top_pair_frac": float("nan"),
        "tail_default_pair_frac": float("nan"),
        "tail_switch_frac": float("nan"),
        "first_live_current_reward": float("nan"),
        "first_live_temp_mae": float("nan"),
    }
    row.update(_compare_rewards(None))
    return row


def _row_for_run(key: str, cfg: dict, ofmpc_current_tail: float, ofmpc_legacy_tail: float) -> tuple[dict, list[dict]]:
    bundle = load_pickle(cfg["path"])
    analyzed = analyze_run(cfg["path"], ofmpc_current_tail, ofmpc_legacy_tail)
    avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    post = slice(10, avg.size)
    release_ep = _release_episode(bundle)
    first_live_ep = slice(release_ep, min(avg.size, release_ep + 20))
    first_live_steps = _window_from_episodes(bundle, release_ep, 20)
    tail_steps = tail_slice(bundle, TAIL_EPISODES)

    first_tracking = tracking_metrics(bundle, CURRENT_REWARD, first_live_steps)
    tail_horizon = horizon_stats(bundle, tail_steps)
    first_horizon = horizon_stats(bundle, first_live_steps)
    compare = _compare_rewards(cfg["compare_path"])

    row = {
        "key": key,
        "label": cfg["label"],
        "kind": cfg["kind"],
        "path": rel(cfg["path"]),
        "compare_path": rel(cfg["compare_path"]) if cfg["compare_path"] is not None else "",
        "state_mode": analyzed["state_mode"],
        "notebook_source": analyzed["notebook_source"],
        "tail_current_reward": analyzed["tail_current_reward"],
        "tail_legacy_reward": analyzed["tail_legacy_reward"],
        "delta_vs_ofmpc": analyzed["current_delta_vs_ofmpc"],
        "final_current_reward": analyzed["final_current_reward"],
        "worst_postwarm_current_reward": float(np.min(avg[post])) if avg.size > 10 else float("nan"),
        "negative_postwarm_episodes": int(np.sum(avg[post] < 0.0)) if avg.size > 10 else float("nan"),
        "first_live_current_reward": finite_mean(avg[first_live_ep]),
        "first_live_temp_mae": first_tracking["temp_mae"],
        "tail_comp_mae": analyzed["tail_current_comp_mae"],
        "tail_temp_mae": analyzed["tail_current_temp_mae"],
        "tail_band_norm_mae": analyzed["tail_current_band_norm_mae"],
        "tail_outside_band_frac": analyzed["tail_current_outside_band_frac"],
        "tail_mean_abs_du_scaled": analyzed["tail_current_mean_abs_du_scaled"],
        "tail_unique_pairs": tail_horizon["unique_pairs"],
        "tail_top_pair": tail_horizon["top_pair"],
        "tail_top_pair_frac": tail_horizon["top_pair_frac"],
        "tail_default_pair_frac": tail_horizon["default_pair_frac"],
        "tail_switch_frac": tail_horizon["switch_frac"],
        "tail_mean_predict": tail_horizon["mean_predict"],
        "tail_mean_control": tail_horizon["mean_control"],
        "first_live_unique_pairs": first_horizon["unique_pairs"],
        "first_live_top_pair": first_horizon["top_pair"],
        "first_live_top_pair_frac": first_horizon["top_pair_frac"],
        "first_live_default_pair_frac": first_horizon["default_pair_frac"],
        "first_live_switch_frac": first_horizon["switch_frac"],
        "epsilon_tail": analyzed["epsilon_tail"],
        "epsilon_final": float(_finite(bundle.get("epsilon_trace", [np.nan]))[-1])
        if _finite(bundle.get("epsilon_trace", [])).size
        else float("nan"),
        "loss_tail": analyzed["loss_tail"],
        "recipe_count": analyzed["tail_recipe_count"],
    }
    row.update(compare)
    row.update({f"tail_{k}": v for k, v in _sg_window_metrics(bundle, tail_steps).items()})
    row.update({f"first_live_{k}": v for k, v in _sg_window_metrics(bundle, first_live_steps).items()})

    top_rows = []
    top_rows.extend(_top_pairs(bundle, first_live_steps, key, "first_live_20"))
    top_rows.extend(_top_pairs(bundle, tail_steps, key, "tail_20"))
    return row, top_rows


def build_outputs() -> tuple[list[dict], list[dict], list[dict]]:
    ofmpc_bundle = load_pickle(OFMPC_PATH)
    ofmpc = baseline_summary(ofmpc_bundle)
    rows = [_row_for_baseline(ofmpc_bundle)]
    top_pairs: list[dict] = []
    for key in ("horizon_epsilon", "horizon_sg", "dueling_epsilon", "dueling_sg", "dueling_stable_history"):
        row, pair_rows = _row_for_run(
            key,
            RUNS[key],
            ofmpc["current_tail20_reward"],
            ofmpc["legacy_tail20_reward"],
        )
        rows.append(row)
        top_pairs.extend(pair_rows)

    comparisons = []
    by_key = {row["key"]: row for row in rows}
    for new_key, old_key in (("horizon_sg", "horizon_epsilon"), ("dueling_sg", "dueling_epsilon")):
        new = by_key[new_key]
        old = by_key[old_key]
        comparisons.append(
            {
                "comparison": f"{new['label']} vs {old['label']}",
                "tail_reward_delta": new["tail_current_reward"] - old["tail_current_reward"],
                "tail_temp_mae_delta": new["tail_temp_mae"] - old["tail_temp_mae"],
                "tail_comp_mae_delta": new["tail_comp_mae"] - old["tail_comp_mae"],
                "negative_postwarm_delta": new["negative_postwarm_episodes"] - old["negative_postwarm_episodes"],
                "tail_unique_pairs_delta": new["tail_unique_pairs"] - old["tail_unique_pairs"],
                "tail_default_frac_delta": new["tail_default_pair_frac"] - old["tail_default_pair_frac"],
                "tail_policy_decision_frac": new.get("tail_sg_policy_decision_frac", float("nan")),
            }
        )
    return rows, comparisons, top_pairs


def plot_reward_curves(rows: list[dict]) -> Path:
    keys = ["ofmpc", "horizon_epsilon", "horizon_sg", "dueling_epsilon", "dueling_sg"]
    labels = {row["key"]: row["label"] for row in rows}
    colors = {
        "ofmpc": "#555555",
        "horizon_epsilon": "#8da0cb",
        "horizon_sg": "#1f78b4",
        "dueling_epsilon": "#c2a5cf",
        "dueling_sg": "#7b3294",
    }
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for key in keys:
        bundle = load_pickle(RUNS[key]["path"])
        avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
        ax.plot(np.arange(1, avg.size + 1), avg, label=labels[key], color=colors[key], linewidth=1.25)
    ax.axvline(10, color="#222222", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.axvline(13, color="#222222", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("current-reward episode average")
    ax.set_title("Distillation SG-DQN horizon follow-up reward curves")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_dqn_followup_reward_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_tail_bars(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] in {"ofmpc", "horizon_epsilon", "horizon_sg", "dueling_epsilon", "dueling_sg"}]
    labels = [row["label"].replace(" ", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.bar(x, [row["tail_current_reward"] for row in plot_rows], color=["#555555", "#8da0cb", "#1f78b4", "#c2a5cf", "#7b3294"])
    ax.axhline(plot_rows[0]["tail_current_reward"], color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("tail-20 current reward")
    ax.set_title("Tail reward: SG-DQN helps standard, not dueling")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_dqn_followup_tail_reward.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_tracking_tradeoff(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] in {"ofmpc", "horizon_epsilon", "horizon_sg", "dueling_epsilon", "dueling_sg", "dueling_stable_history"}]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colors = {
        "ofmpc": "#555555",
        "horizon_epsilon": "#8da0cb",
        "horizon_sg": "#1f78b4",
        "dueling_epsilon": "#c2a5cf",
        "dueling_sg": "#7b3294",
        "dueling_stable_history": "#4daf4a",
    }
    for row in plot_rows:
        ax.scatter(
            row["tail_temp_mae"],
            row["tail_comp_mae"],
            s=60 + 24 * max(row["tail_current_reward"], 0.0),
            color=colors[row["key"]],
            alpha=0.82,
            label=row["label"],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.annotate(row["label"], (row["tail_temp_mae"], row["tail_comp_mae"]), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("tail T85 MAE")
    ax.set_ylabel("tail x24 ethane MAE")
    ax.set_title("Tracking tradeoff under current reward")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_dqn_followup_tracking_tradeoff.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_gate_diagnostics(rows: list[dict]) -> Path:
    sg_rows = [row for row in rows if row["key"] in {"horizon_sg", "dueling_sg"}]
    labels = [row["label"].replace("-", "\n") for row in sg_rows]
    x = np.arange(len(sg_rows))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.bar(x - width, [row["first_live_sg_policy_decision_frac"] for row in sg_rows], width, label="first-live decisions")
    ax.bar(x, [row["tail_sg_policy_decision_frac"] for row in sg_rows], width, label="tail decisions")
    ax.bar(x + width, [row["tail_default_pair_frac"] for row in sg_rows], width, label="tail default pair")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("SG-DQN gate authority and default-horizon use")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_dqn_followup_gate_diagnostics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_horizon_stability(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] in {"horizon_epsilon", "horizon_sg", "dueling_epsilon", "dueling_sg", "dueling_stable_history"}]
    labels = [row["label"].replace(" ", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    width = 0.28
    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    ax1.bar(x - width / 2, [row["tail_unique_pairs"] for row in plot_rows], width, label="unique tail pairs", color="#80b1d3")
    ax1.set_ylabel("unique tail pairs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x + width / 2, [row["tail_top_pair_frac"] for row in plot_rows], marker="o", color="#d95f02", label="top-pair fraction")
    ax2.plot(x + width / 2, [row["tail_switch_frac"] for row in plot_rows], marker="s", color="#1b9e77", label="switch fraction")
    ax2.set_ylim(0.0, 0.75)
    ax2.set_ylabel("fraction")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Horizon-policy stability")
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_dqn_followup_horizon_stability.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_top_pairs(top_pairs: list[dict]) -> Path:
    target = [row for row in top_pairs if row["window"] == "tail_20" and row["run"] in {"horizon_sg", "dueling_sg"} and row["rank"] <= 6]
    labels = sorted({row["pair"] for row in target})
    runs = ["horizon_sg", "dueling_sg"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    width = 0.36
    for offset, run, color in [(-width / 2, "horizon_sg", "#1f78b4"), (width / 2, "dueling_sg", "#7b3294")]:
        vals = []
        for label in labels:
            match = [row for row in target if row["run"] == run and row["pair"] == label]
            vals.append(match[0]["fraction"] if match else 0.0)
        ax.bar(x + offset, vals, width, label=RUNS[run]["label"], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("tail fraction")
    ax.set_title("Top executed tail horizon pairs in SG-DQN runs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_dqn_followup_top_pairs.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, comparisons, top_pairs = build_outputs()
    write_csv(OUT_DIR / "sg_dqn_followup_summary.csv", rows)
    write_csv(OUT_DIR / "sg_dqn_followup_comparisons.csv", comparisons)
    write_csv(OUT_DIR / "sg_dqn_followup_top_pairs.csv", top_pairs)
    figures = [
        plot_reward_curves(rows),
        plot_tail_bars(rows),
        plot_tracking_tradeoff(rows),
        plot_gate_diagnostics(rows),
        plot_horizon_stability(rows),
        plot_top_pairs(top_pairs),
    ]
    payload = {
        "rows": rows,
        "comparisons": comparisons,
        "top_pairs": top_pairs,
        "figures": [rel(path) for path in figures],
        "outputs": {
            "summary_csv": rel(OUT_DIR / "sg_dqn_followup_summary.csv"),
            "comparisons_csv": rel(OUT_DIR / "sg_dqn_followup_comparisons.csv"),
            "top_pairs_csv": rel(OUT_DIR / "sg_dqn_followup_top_pairs.csv"),
        },
    }
    (OUT_DIR / "sg_dqn_followup_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"figures": payload["figures"], "outputs": payload["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
