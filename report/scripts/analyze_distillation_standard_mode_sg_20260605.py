"""Compare distillation standard-mode SG runs against prior SG references.

The script reads saved result bundles only. It rescales all trajectories with
the current reward definition so standard-mode TD3/DQN runs can be compared
against earlier mismatch or mismatch-conditioned runs on common tracking terms.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report.scripts.analyze_distillation_dueling_horizon_history_20260604 import (  # noqa: E402
    CURRENT_REWARD,
    OFMPC_PATH,
    TAIL_EPISODES,
    baseline_summary,
    episode_average,
    episode_len,
    finite_mean,
    load_pickle,
    rel,
    reward_step_series,
    step_slice,
    tail_slice,
    tracking_metrics,
    write_csv,
    y_sp_phys,
)
from report.scripts.analyze_distillation_sg_dqn_horizon_followup_20260604 import _sg_window_metrics  # noqa: E402


OUT_DIR = ROOT / "report" / "figures" / "distillation_standard_mode_sg_20260605"

SOURCE_WARM_START = 0
SOURCE_SUPERVISOR = 1
SOURCE_POLICY = 2
SOURCE_FALLBACK = 4
SG_TD3_KEYS = [
    "weights_mismatch",
    "weights_standard",
    "residual_mismatch",
    "residual_standard",
    "markov_mismatch_conditioned",
    "markov_standard",
]
SG_TD3_PAIRS = [
    ("weights_standard", "weights_mismatch"),
    ("residual_standard", "residual_mismatch"),
    ("markov_standard", "markov_mismatch_conditioned"),
]
WARM_START_SUBEPISODES = 10
POST_WARM_ACTION_FREEZE_SUBEPISODES = 3
LIVE_RELEASE_SUBEPISODE = WARM_START_SUBEPISODES + POST_WARM_ACTION_FREEZE_SUBEPISODES + 1

RUNS = {
    "ofmpc": {
        "family": "baseline",
        "label": "OF-MPC",
        "variant": "baseline",
        "path": OFMPC_PATH,
    },
    "weights_mismatch": {
        "family": "weights",
        "label": "Weights mismatch",
        "variant": "mismatch",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch"
        / "20260603_124214"
        / "input_data.pkl",
    },
    "weights_standard": {
        "family": "weights",
        "label": "Weights standard",
        "variant": "standard",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_standard"
        / "20260605_105053"
        / "input_data.pkl",
    },
    "residual_mismatch": {
        "family": "residual",
        "label": "Residual mismatch",
        "variant": "mismatch",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho"
        / "20260602_125954"
        / "input_data.pkl",
    },
    "residual_standard": {
        "family": "residual",
        "label": "Residual standard",
        "variant": "standard",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_standard_no_rho"
        / "20260605_110345"
        / "input_data.pkl",
    },
    "markov_mismatch_conditioned": {
        "family": "markov",
        "label": "Markov mismatch-conditioned",
        "variant": "mismatch-conditioned",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_unified"
        / "20260602_192543"
        / "input_data.pkl",
    },
    "markov_standard": {
        "family": "markov",
        "label": "Markov standard",
        "variant": "standard",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_standard"
        / "20260605_114125"
        / "input_data.pkl",
    },
    "horizon_mismatch_87": {
        "family": "horizon",
        "label": "SG-DQN mismatch 87",
        "variant": "mismatch 87",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch"
        / "20260604_103231"
        / "input_data.pkl",
    },
    "horizon_standard_39": {
        "family": "horizon",
        "label": "SG-DQN standard 39",
        "variant": "standard 39",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard_np6_11_nc3_11"
        / "20260605_105331"
        / "input_data.pkl",
    },
    "dueling_mismatch_87": {
        "family": "dueling horizon",
        "label": "SG-dueling mismatch 87",
        "variant": "mismatch 87",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch"
        / "20260604_105140"
        / "input_data.pkl",
    },
    "dueling_standard_legacy": {
        "family": "dueling horizon",
        "label": "SG-dueling standard legacy",
        "variant": "standard legacy",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_sg_dqn_aspen6_legacyreward_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard"
        / "20260605_105758"
        / "input_data.pkl",
    },
}


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, float).reshape(-1)
    return arr[np.isfinite(arr)]


def _nan_stat(values, reducer, default=float("nan")) -> float:
    arr = _finite(values)
    return float(reducer(arr)) if arr.size else float(default)


def _q(values, quantile: float) -> float:
    return _nan_stat(values, lambda arr: np.quantile(arr, quantile))


def _array(bundle: dict, key: str) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        return np.asarray([])
    return np.asarray(value)


def _episode_phase(subepisode: int) -> str:
    if int(subepisode) <= WARM_START_SUBEPISODES:
        return "warm_start"
    if int(subepisode) < LIVE_RELEASE_SUBEPISODE:
        return "protected_handover"
    return "live"


def _advantage_log(bundle: dict) -> np.ndarray:
    advantage = _array(bundle, "sg_advantage_log")
    if advantage.size:
        return np.asarray(advantage, float).reshape(-1)
    policy = _array(bundle, "sg_score_policy_log")
    supervisor = _array(bundle, "sg_score_supervisor_log")
    if policy.size and supervisor.size:
        n = min(policy.size, supervisor.size)
        return np.asarray(policy[:n], float).reshape(-1) - np.asarray(supervisor[:n], float).reshape(-1)
    return np.asarray([])


def _norm_log(bundle: dict, key: str) -> np.ndarray:
    arr = _array(bundle, key)
    if not arr.size:
        return np.asarray([])
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.linalg.norm(arr, axis=1)


def _mean_action_distance(bundle: dict, first_key: str, second_key: str, sl: slice) -> float:
    first = _array(bundle, first_key)
    second = _array(bundle, second_key)
    if not first.size or not second.size:
        return float("nan")
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    if first.ndim == 1:
        first = first.reshape(-1, 1)
    if second.ndim == 1:
        second = second.reshape(-1, 1)
    n = min(first.shape[0], second.shape[0])
    first = first[:n, :]
    second = second[:n, :]
    start = max(0, sl.start or 0)
    stop = min(n, sl.stop or n)
    if stop <= start:
        return float("nan")
    return _nan_stat(np.linalg.norm(first[start:stop, :] - second[start:stop, :], axis=1), np.mean)


def _row_for_bundle(key: str, cfg: dict, ofmpc_tail: float) -> dict:
    bundle = load_pickle(cfg["path"])
    avg = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    tail_steps = tail_slice(bundle, TAIL_EPISODES)
    tracking = tracking_metrics(bundle, CURRENT_REWARD, tail_steps)
    post = avg[10:] if avg.size > 10 else avg
    sg = _sg_window_metrics(bundle, tail_steps) if "sg_selected_source_log" in bundle else {}
    episode_tail = avg[-TAIL_EPISODES:] if avg.size >= TAIL_EPISODES else avg
    return {
        "key": key,
        "family": cfg["family"],
        "label": cfg["label"],
        "variant": cfg["variant"],
        "path": rel(cfg["path"]),
        "agent_kind": bundle.get("agent_kind", "ofmpc"),
        "state_mode": bundle.get("state_mode", ""),
        "markov_state_mode": bundle.get("markov_state_mode", ""),
        "markov_agent_state_features": bundle.get("markov_agent_state_features", ""),
        "notebook_source": bundle.get("notebook_source", ""),
        "recipe_count": len(bundle.get("horizon_recipes", [])) if "horizon_recipes" in bundle else "",
        "tail_current_reward": finite_mean(episode_tail),
        "delta_vs_ofmpc": finite_mean(episode_tail) - ofmpc_tail,
        "final_current_reward": float(avg[-1]) if avg.size else float("nan"),
        "worst_postwarm_current_reward": float(np.min(post)) if post.size else float("nan"),
        "negative_postwarm_episodes": int(np.sum(post < 0.0)) if post.size else 0,
        "tail_temp_mae": tracking["temp_mae"],
        "tail_comp_mae": tracking["comp_mae"],
        "tail_outside_band_frac": tracking["outside_band_frac"],
        "tail_mean_abs_du_scaled": tracking["mean_abs_du_scaled"],
        "tail_sg_policy_decision_frac": sg.get("sg_policy_decision_frac", float("nan")),
        "tail_sg_policy_step_frac": sg.get("sg_policy_step_frac", float("nan")),
        "tail_sg_adv_median": sg.get("sg_adv_median", float("nan")),
        "tail_sg_adv_q10": sg.get("sg_adv_q10", float("nan")),
        "tail_sg_adv_q90": sg.get("sg_adv_q90", float("nan")),
    }


def _episode_rows_for_bundle(key: str, bundle: dict | None = None) -> list[dict]:
    cfg = RUNS[key]
    bundle = load_pickle(cfg["path"]) if bundle is None else bundle
    rewards = episode_average(reward_step_series(bundle, CURRENT_REWARD), bundle)
    selected = _array(bundle, "sg_selected_source_log").reshape(-1)
    advantage = _advantage_log(bundle)
    q_gap_policy = _array(bundle, "sg_q_gap_policy_log").reshape(-1)
    q_gap_supervisor = _array(bundle, "sg_q_gap_supervisor_log").reshape(-1)
    q1_policy = _array(bundle, "sg_q1_policy_log").reshape(-1)
    q2_policy = _array(bundle, "sg_q2_policy_log").reshape(-1)
    q1_supervisor = _array(bundle, "sg_q1_supervisor_log").reshape(-1)
    q2_supervisor = _array(bundle, "sg_q2_supervisor_log").reshape(-1)
    executed_norm = _norm_log(bundle, "sg_executed_action_raw_log")
    policy_norm = _norm_log(bundle, "sg_policy_action_raw_log")
    supervisor_norm = _norm_log(bundle, "sg_supervisor_action_raw_log")
    rows = []
    for ep, reward in enumerate(rewards):
        sl = step_slice(bundle, ep, ep + 1)
        tracking = tracking_metrics(bundle, CURRENT_REWARD, sl)
        source_sl = selected[sl] if selected.size else np.asarray([])
        adv_sl = advantage[sl] if advantage.size else np.asarray([])
        qgap_policy_sl = q_gap_policy[sl] if q_gap_policy.size else np.asarray([])
        qgap_supervisor_sl = q_gap_supervisor[sl] if q_gap_supervisor.size else np.asarray([])
        q1_policy_sl = q1_policy[sl] if q1_policy.size else np.asarray([])
        q2_policy_sl = q2_policy[sl] if q2_policy.size else np.asarray([])
        q1_supervisor_sl = q1_supervisor[sl] if q1_supervisor.size else np.asarray([])
        q2_supervisor_sl = q2_supervisor[sl] if q2_supervisor.size else np.asarray([])
        executed_norm_sl = executed_norm[sl] if executed_norm.size else np.asarray([])
        policy_norm_sl = policy_norm[sl] if policy_norm.size else np.asarray([])
        supervisor_norm_sl = supervisor_norm[sl] if supervisor_norm.size else np.asarray([])
        subepisode = ep + 1
        rows.append(
            {
                "key": key,
                "family": cfg["family"],
                "label": cfg["label"],
                "variant": cfg["variant"],
                "subepisode": subepisode,
                "phase": _episode_phase(subepisode),
                "current_reward": float(reward),
                "temp_mae": tracking["temp_mae"],
                "comp_mae": tracking["comp_mae"],
                "outside_band_frac": tracking["outside_band_frac"],
                "mean_abs_du_scaled": tracking["mean_abs_du_scaled"],
                "policy_fraction": float(np.mean(source_sl == SOURCE_POLICY)) if source_sl.size else float("nan"),
                "supervisor_fraction": float(np.mean(source_sl == SOURCE_SUPERVISOR)) if source_sl.size else float("nan"),
                "warm_start_fraction": float(np.mean(source_sl == SOURCE_WARM_START)) if source_sl.size else float("nan"),
                "fallback_fraction": float(np.mean(source_sl == SOURCE_FALLBACK)) if source_sl.size else float("nan"),
                "adv_mean": _nan_stat(adv_sl, np.mean),
                "adv_median": _nan_stat(adv_sl, np.median),
                "adv_q10": _q(adv_sl, 0.10),
                "adv_q90": _q(adv_sl, 0.90),
                "q_gap_policy_mean": _nan_stat(qgap_policy_sl, np.mean),
                "q_gap_supervisor_mean": _nan_stat(qgap_supervisor_sl, np.mean),
                "q1_policy_mean": _nan_stat(q1_policy_sl, np.mean),
                "q2_policy_mean": _nan_stat(q2_policy_sl, np.mean),
                "q1_supervisor_mean": _nan_stat(q1_supervisor_sl, np.mean),
                "q2_supervisor_mean": _nan_stat(q2_supervisor_sl, np.mean),
                "executed_action_norm_mean": _nan_stat(executed_norm_sl, np.mean),
                "policy_action_norm_mean": _nan_stat(policy_norm_sl, np.mean),
                "supervisor_action_norm_mean": _nan_stat(supervisor_norm_sl, np.mean),
                "policy_minus_supervisor_norm_mean": _mean_action_distance(
                    bundle,
                    "sg_policy_action_raw_log",
                    "sg_supervisor_action_raw_log",
                    sl,
                ),
                "executed_minus_supervisor_norm_mean": _mean_action_distance(
                    bundle,
                    "sg_executed_action_raw_log",
                    "sg_supervisor_action_raw_log",
                    sl,
                ),
            }
        )
    return rows


def _episode_diagnostic_rows() -> list[dict]:
    rows = []
    for key in SG_TD3_KEYS:
        rows.extend(_episode_rows_for_bundle(key))
    return rows


def _worst_episode_rows(episode_rows: list[dict], n_worst: int = 5) -> list[dict]:
    by_key_episode = {
        (str(row["key"]), int(row["subepisode"])): row
        for row in episode_rows
    }
    out = []
    for standard_key, reference_key in SG_TD3_PAIRS:
        candidates = [
            row
            for row in episode_rows
            if row["key"] == standard_key and int(row["subepisode"]) > WARM_START_SUBEPISODES
        ]
        candidates = sorted(candidates, key=lambda row: float(row["current_reward"]))[:n_worst]
        for rank, row in enumerate(candidates, start=1):
            reference = by_key_episode.get((reference_key, int(row["subepisode"])), {})
            out.append(
                {
                    "rank": rank,
                    "standard_key": standard_key,
                    "reference_key": reference_key,
                    "family": row["family"],
                    "subepisode": row["subepisode"],
                    "phase": row["phase"],
                    "standard_reward": row["current_reward"],
                    "reference_reward_same_subepisode": reference.get("current_reward", float("nan")),
                    "standard_temp_mae": row["temp_mae"],
                    "reference_temp_mae_same_subepisode": reference.get("temp_mae", float("nan")),
                    "standard_comp_mae": row["comp_mae"],
                    "reference_comp_mae_same_subepisode": reference.get("comp_mae", float("nan")),
                    "standard_outside_band_frac": row["outside_band_frac"],
                    "reference_outside_band_frac_same_subepisode": reference.get("outside_band_frac", float("nan")),
                    "standard_policy_fraction": row["policy_fraction"],
                    "reference_policy_fraction_same_subepisode": reference.get("policy_fraction", float("nan")),
                    "standard_adv_median": row["adv_median"],
                    "reference_adv_median_same_subepisode": reference.get("adv_median", float("nan")),
                    "standard_policy_minus_supervisor_norm": row["policy_minus_supervisor_norm_mean"],
                    "reference_policy_minus_supervisor_norm_same_subepisode": reference.get(
                        "policy_minus_supervisor_norm_mean",
                        float("nan"),
                    ),
                    "standard_executed_minus_supervisor_norm": row["executed_minus_supervisor_norm_mean"],
                    "reference_executed_minus_supervisor_norm_same_subepisode": reference.get(
                        "executed_minus_supervisor_norm_mean",
                        float("nan"),
                    ),
                    "standard_q_gap_policy_mean": row["q_gap_policy_mean"],
                    "reference_q_gap_policy_mean_same_subepisode": reference.get(
                        "q_gap_policy_mean",
                        float("nan"),
                    ),
                }
            )
    return out


def _pairwise_rows(rows: list[dict]) -> list[dict]:
    by_key = {row["key"]: row for row in rows}
    pairs = [
        ("weights_standard", "weights_mismatch"),
        ("residual_standard", "residual_mismatch"),
        ("markov_standard", "markov_mismatch_conditioned"),
        ("horizon_standard_39", "horizon_mismatch_87"),
        ("dueling_standard_legacy", "dueling_mismatch_87"),
    ]
    out = []
    for standard_key, reference_key in pairs:
        standard = by_key[standard_key]
        reference = by_key[reference_key]
        out.append(
            {
                "comparison": f"{standard['label']} vs {reference['label']}",
                "family": standard["family"],
                "tail_reward_delta": standard["tail_current_reward"] - reference["tail_current_reward"],
                "tail_temp_mae_delta": standard["tail_temp_mae"] - reference["tail_temp_mae"],
                "tail_comp_mae_delta": standard["tail_comp_mae"] - reference["tail_comp_mae"],
                "negative_postwarm_delta": standard["negative_postwarm_episodes"]
                - reference["negative_postwarm_episodes"],
                "policy_decision_frac_delta": standard["tail_sg_policy_decision_frac"]
                - reference["tail_sg_policy_decision_frac"],
                "adv_median_delta": standard["tail_sg_adv_median"] - reference["tail_sg_adv_median"],
                "confounded": standard["family"] in {"horizon", "dueling horizon"},
            }
        )
    return out


def _plot_tail_reward(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] != "ofmpc"]
    labels = [row["label"].replace(" ", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    colors = ["#1f78b4" if "standard" not in row["variant"] else "#f28e2b" for row in plot_rows]
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.bar(x, [row["tail_current_reward"] for row in plot_rows], color=colors, alpha=0.86)
    ofmpc = next(row for row in rows if row["key"] == "ofmpc")
    ax.axhline(ofmpc["tail_current_reward"], color="#333333", linestyle="--", linewidth=0.9, label="OF-MPC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("tail-20 current reward")
    ax.set_title("Distillation SG standard-mode runs against earlier references")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig_standard_mode_tail_reward.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_temp_reward(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] != "ofmpc"]
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    markers = {
        "weights": "o",
        "residual": "s",
        "markov": "^",
        "horizon": "D",
        "dueling horizon": "P",
    }
    for row in plot_rows:
        is_standard = "standard" in row["variant"]
        ax.scatter(
            row["tail_temp_mae"],
            row["tail_current_reward"],
            marker=markers.get(row["family"], "o"),
            s=90,
            color="#f28e2b" if is_standard else "#1f78b4",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(row["label"].replace("SG-", ""), (row["tail_temp_mae"], row["tail_current_reward"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("tail T85 MAE")
    ax.set_ylabel("tail current reward")
    ax.set_title("Reward loss is mostly a T85/stability loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_standard_mode_reward_vs_t85.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_negative_episodes(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] != "ofmpc"]
    labels = [row["label"].replace(" ", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    colors = ["#1f78b4" if "standard" not in row["variant"] else "#f28e2b" for row in plot_rows]
    fig, ax = plt.subplots(figsize=(12.2, 4.9))
    ax.bar(x, [row["negative_postwarm_episodes"] for row in plot_rows], color=colors, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("negative post-warm episodes")
    ax.set_title("Standard mode often increases post-warm fragility")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_standard_mode_negative_postwarm.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_gate(rows: list[dict]) -> Path:
    plot_rows = [row for row in rows if row["key"] != "ofmpc"]
    labels = [row["label"].replace(" ", "\n") for row in plot_rows]
    x = np.arange(len(plot_rows))
    fig, ax1 = plt.subplots(figsize=(12.2, 5.0))
    colors = ["#1f78b4" if "standard" not in row["variant"] else "#f28e2b" for row in plot_rows]
    ax1.bar(x, [row["tail_sg_policy_decision_frac"] for row in plot_rows], color=colors, alpha=0.78, label="policy decision frac")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("tail policy decision fraction")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["tail_sg_adv_median"] for row in plot_rows], color="#333333", marker="o", label="median SG advantage")
    ax2.axhline(0.0, color="#333333", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("median Q advantage")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Standard mode weakens the learned gate ranking in several families")
    fig.tight_layout()
    out = OUT_DIR / "fig_standard_mode_gate_diagnostics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_sg_td3_episode_rewards(episode_rows: list[dict]) -> Path:
    family_pairs = [
        ("weights", "weights_standard", "weights_mismatch"),
        ("residual", "residual_standard", "residual_mismatch"),
        ("markov", "markov_standard", "markov_mismatch_conditioned"),
    ]
    fig, axes = plt.subplots(len(family_pairs), 1, figsize=(11.2, 8.8), sharex=True)
    by_key = {}
    for row in episode_rows:
        by_key.setdefault(row["key"], []).append(row)
    for ax, (family, standard_key, reference_key) in zip(axes, family_pairs):
        standard = sorted(by_key[standard_key], key=lambda row: int(row["subepisode"]))
        reference = sorted(by_key[reference_key], key=lambda row: int(row["subepisode"]))
        ep = np.asarray([row["subepisode"] for row in standard], dtype=float)
        ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.28, label="warm start")
        ax.axvspan(
            WARM_START_SUBEPISODES + 1,
            LIVE_RELEASE_SUBEPISODE - 1,
            color="#f3c567",
            alpha=0.24,
            label="protected handover",
        )
        ax.axvline(LIVE_RELEASE_SUBEPISODE, color="#a23b72", linestyle=":", linewidth=1.1)
        ax.plot(
            [row["subepisode"] for row in reference],
            [row["current_reward"] for row in reference],
            color="#1f78b4",
            linewidth=1.25,
            label="reference",
        )
        ax.plot(
            ep,
            [row["current_reward"] for row in standard],
            color="#f28e2b",
            linewidth=1.35,
            label="standard",
        )
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.75)
        ax.set_ylabel(f"{family}\nreward")
        ax.grid(alpha=0.22)
        ax2 = ax.twinx()
        ax2.plot(
            ep,
            [row["policy_fraction"] for row in standard],
            color="#e15759",
            alpha=0.48,
            linewidth=0.9,
            label="standard policy frac",
        )
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("policy frac")
        if ax is axes[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=4)
    axes[-1].set_xlabel("subepisode")
    fig.suptitle("SG-TD3 standard-mode bad episodes align with live policy release", y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_td3_episode_reward_sources.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_sg_td3_gate_failure_scatter(episode_rows: list[dict], worst_rows: list[dict]) -> Path:
    standard_keys = {"weights_standard", "residual_standard", "markov_standard"}
    rows = [
        row
        for row in episode_rows
        if row["key"] in standard_keys and int(row["subepisode"]) > WARM_START_SUBEPISODES
    ]
    colors = {
        "weights": "#4e79a7",
        "residual": "#f28e2b",
        "markov": "#59a14f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharey=True)
    for family, color in colors.items():
        fam_rows = [row for row in rows if row["family"] == family]
        axes[0].scatter(
            [row["policy_fraction"] for row in fam_rows],
            [row["current_reward"] for row in fam_rows],
            s=28,
            alpha=0.58,
            color=color,
            edgecolor="none",
            label=family,
        )
        axes[1].scatter(
            [row["adv_median"] for row in fam_rows],
            [row["current_reward"] for row in fam_rows],
            s=28,
            alpha=0.58,
            color=color,
            edgecolor="none",
            label=family,
    )
    for item in worst_rows:
        if int(item["rank"]) > 3:
            continue
        if item["family"] == "weights" and int(item["rank"]) > 1:
            continue
        label = f"{item['family']} {int(item['subepisode'])}"
        axes[0].annotate(
            label,
            (float(item["standard_policy_fraction"]), float(item["standard_reward"])),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
        axes[1].annotate(
            label,
            (float(item["standard_adv_median"]), float(item["standard_reward"])),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=0.75)
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=0.75)
    axes[1].axvline(0.0, color="#333333", linestyle=":", linewidth=0.75)
    axes[0].set_xlabel("policy-selected fraction per subepisode")
    axes[0].set_ylabel("current reward")
    axes[1].set_xlabel("median SG advantage")
    axes[0].set_title("Bad episodes often have high policy authority")
    axes[1].set_title("Critic optimism can be wrong in standard mode")
    for ax in axes:
        ax.grid(alpha=0.24)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_td3_episode_gate_failure_modes.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _y_and_setpoint(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    ysp = y_sp_phys(bundle)
    y = np.asarray(bundle.get("y_line_full", bundle.get("y")), float)
    n = min(ysp.shape[0], _array(bundle, "sg_selected_source_log").size or ysp.shape[0])
    if y.shape[0] >= n + 1:
        y_step = y[1 : n + 1, :]
    else:
        y_step = y[:n, :]
    n = min(n, y_step.shape[0], ysp.shape[0])
    return y_step[:n, :], ysp[:n, :]


def _episode_row_lookup(episode_rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {
        (str(row["key"]), int(row["subepisode"])): row
        for row in episode_rows
    }


def _plot_sg_td3_worst_tracking(episode_rows: list[dict]) -> Path:
    lookup = _episode_row_lookup(episode_rows)
    cases = [
        ("residual_standard", 18),
        ("markov_standard", 86),
    ]
    fig, axes = plt.subplots(len(cases), 2, figsize=(11.5, 6.6), sharex=True)
    for row_axes, (key, subepisode) in zip(axes, cases):
        bundle = load_pickle(RUNS[key]["path"])
        y, ysp = _y_and_setpoint(bundle)
        sl = step_slice(bundle, subepisode - 1, subepisode)
        start = max(0, sl.start or 0)
        stop = min(y.shape[0], sl.stop or y.shape[0])
        steps = np.arange(stop - start)
        selected = _array(bundle, "sg_selected_source_log").reshape(-1)
        policy = selected[start:stop] == SOURCE_POLICY if selected.size >= stop else np.zeros(stop - start, dtype=bool)
        data = lookup[(key, subepisode)]
        signals = [
            (y[start:stop, 1] - ysp[start:stop, 1], "T85 error"),
            (y[start:stop, 0] - ysp[start:stop, 0], "x24 error"),
        ]
        for ax, (err, ylabel) in zip(row_axes, signals):
            ax.plot(steps, err, color="#2f4b7c", linewidth=1.25)
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.75)
            ymin, ymax = ax.get_ylim()
            ytick = ymin + 0.04 * max(1.0e-12, ymax - ymin)
            ax.scatter(
                steps[policy],
                np.full(int(np.sum(policy)), ytick),
                marker="|",
                s=10,
                color="#e15759",
                alpha=0.55,
                label="policy-selected step",
            )
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
        row_axes[0].set_title(
            f"{RUNS[key]['label']} subepisode {subepisode}: "
            f"reward {data['current_reward']:.1f}, policy frac {data['policy_fraction']:.2f}"
        )
        row_axes[1].set_title(f"median advantage {data['adv_median']:.1f}")
    for ax in axes[-1, :]:
        ax.set_xlabel("steps inside subepisode")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "fig_sg_td3_worst_episode_tracking.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ofmpc = baseline_summary(load_pickle(OFMPC_PATH))
    rows = [_row_for_bundle("ofmpc", RUNS["ofmpc"], ofmpc["current_tail20_reward"])]
    for key in RUNS:
        if key == "ofmpc":
            continue
        rows.append(_row_for_bundle(key, RUNS[key], ofmpc["current_tail20_reward"]))
    pairwise = _pairwise_rows(rows)
    episode_rows = _episode_diagnostic_rows()
    worst_episode_rows = _worst_episode_rows(episode_rows)
    write_csv(OUT_DIR / "distillation_standard_mode_sg_summary.csv", rows)
    write_csv(OUT_DIR / "distillation_standard_mode_sg_pairwise.csv", pairwise)
    write_csv(OUT_DIR / "distillation_standard_mode_sg_td3_episode_diagnostics.csv", episode_rows)
    write_csv(OUT_DIR / "distillation_standard_mode_sg_td3_worst_episodes.csv", worst_episode_rows)
    figures = [
        _plot_tail_reward(rows),
        _plot_temp_reward(rows),
        _plot_negative_episodes(rows),
        _plot_gate(rows),
        _plot_sg_td3_episode_rewards(episode_rows),
        _plot_sg_td3_gate_failure_scatter(episode_rows, worst_episode_rows),
        _plot_sg_td3_worst_tracking(episode_rows),
    ]
    payload = {
        "rows": rows,
        "pairwise": pairwise,
        "sg_td3_worst_episodes": worst_episode_rows,
        "figures": [rel(path) for path in figures],
        "outputs": {
            "summary_csv": rel(OUT_DIR / "distillation_standard_mode_sg_summary.csv"),
            "pairwise_csv": rel(OUT_DIR / "distillation_standard_mode_sg_pairwise.csv"),
            "sg_td3_episode_diagnostics_csv": rel(
                OUT_DIR / "distillation_standard_mode_sg_td3_episode_diagnostics.csv"
            ),
            "sg_td3_worst_episodes_csv": rel(
                OUT_DIR / "distillation_standard_mode_sg_td3_worst_episodes.csv"
            ),
        },
    }
    (OUT_DIR / "distillation_standard_mode_sg_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"figures": payload["figures"], "outputs": payload["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
