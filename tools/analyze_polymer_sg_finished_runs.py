"""Reproduce the June 2026 polymer SG-SAC/SG-DQN comparison metrics.

The script reads timestamped result bundles, writes a metric CSV, and regenerates
the compact comparison figures used in
``report/polymer_sg_sac_sg_dqn_finished_runs_2026-06-04.md``.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "Polymer" / "Results"
OUT_DIR = REPO_ROOT / "report" / "figures" / "2026-06-04_polymer_sg_runs"

RUNS = [
    (
        "SG-SAC residual",
        "sg_sac_residual_critic_warm3_zero_shadow_disturb/20260603_205207/input_data.pkl",
        "prev SG-TD3 residual",
        "sg_td3_residual_critic_warm3_conservative_disturb/20260601_182754/input_data.pkl",
        "disturb_compare_sg_sac_residual_critic_warm3_zero_shadow/20260603_205225/input_data.pkl",
    ),
    (
        "SG-SAC weights",
        "sg_sac_weights_critic_warm3_identity_shadow_disturb/20260603_205218/input_data.pkl",
        "prev SG-TD3 weights",
        "sg_td3_weights_critic_warm3_conservative_disturb/20260601_184718/input_data.pkl",
        "disturb_compare_sg_sac_weights_critic_warm3_identity_shadow/20260603_205232/input_data.pkl",
    ),
    (
        "SG-SAC Markov",
        "sg_sac_markov_critic_warm3_ls_else_mpc_shadow_disturb/20260603_212647/input_data.pkl",
        "prev SG-TD3 Markov",
        "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb/20260601_215126/input_data.pkl",
        "disturb_compare_sg_sac_markov_critic_warm3_ls_else_mpc_shadow/20260603_212712/input_data.pkl",
    ),
    (
        "SG-DQN horizon",
        "horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/20260603_210127/input_data.pkl",
        "prev DQN horizon",
        "horizon_disturb_unified/20260520_222308/input_data.pkl",
        "disturb_compare_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002/20260603_210142/input_data.pkl",
    ),
    (
        "SG-dueling DQN",
        "dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/20260603_210913/input_data.pkl",
        "prev dueling DQN",
        "dueling_horizon_disturb_unified/20260520_225756/input_data.pkl",
        "disturb_compare_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002/20260603_210928/input_data.pkl",
    ),
]


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def load_bundle(rel_path: str) -> dict:
    path = RESULT_ROOT / rel_path
    with path.open("rb") as fh:
        return pickle.load(fh)


def pct_lower_is_better(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
        return float("nan")
    return 100.0 * (previous - current) / abs(previous)


def pct_higher_is_better(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
        return float("nan")
    return 100.0 * (current - previous) / abs(previous)


def run_metrics(bundle: dict) -> dict:
    n_steps = int(bundle.get("nFE", len(bundle.get("rewards_step", []))))
    warm_start = int(bundle.get("warm_start_step", 0) or 0)
    steps_per_subepisode = int(bundle.get("time_in_sub_episodes", 800) or 800)
    tail_start = max(0, n_steps - 20 * steps_per_subepisode)
    post = slice(warm_start, n_steps)
    tail = slice(tail_start, n_steps)

    tracking_scaled = bundle.get("tracking_error_log")
    tracking_raw = bundle.get("tracking_error_raw_log")
    rewards = bundle.get("rewards_step")
    delta_u = bundle.get("delta_u_storage")
    selected = bundle.get("sg_selected_source_log")
    summary = bundle.get("summary_metrics", {}) or {}

    scaled = np.asarray(tracking_scaled, dtype=float) if tracking_scaled is not None else None
    raw = np.asarray(tracking_raw, dtype=float) if tracking_raw is not None else None
    reward_arr = np.asarray(rewards, dtype=float) if rewards is not None else None
    delta_u_arr = np.asarray(delta_u, dtype=float) if delta_u is not None else None
    selected_arr = np.asarray(selected, dtype=float) if selected is not None else None

    out = {
        "algorithm": bundle.get("algorithm") or bundle.get("agent_kind"),
        "tail_reward": finite_mean(reward_arr[tail]) if reward_arr is not None else float("nan"),
        "post_scaled_mae": finite_mean(np.abs(scaled[post])) if scaled is not None else float("nan"),
        "tail_scaled_mae": finite_mean(np.abs(scaled[tail])) if scaled is not None else float("nan"),
        "tail_eta_mae": finite_mean(np.abs(raw[tail, 0])) if raw is not None else float("nan"),
        "tail_T_mae": finite_mean(np.abs(raw[tail, 1])) if raw is not None else float("nan"),
        "post_du_abs": finite_mean(np.abs(delta_u_arr[post])) if delta_u_arr is not None else float("nan"),
        "prediction_score_mean": summary.get("prediction_score_mean", float("nan")),
        "gain_drift_mean": summary.get("gain_drift_mean", float("nan")),
        "accepted_fraction_post_warm": summary.get("accepted_fraction_post_warm", float("nan")),
    }

    if selected_arr is None:
        for key in (
            "sg_policy_post",
            "sg_policy_tail",
            "sg_supervisor_post",
            "sg_supervisor_tail",
            "sg_fallback_post",
        ):
            out[key] = float("nan")
    else:
        out["sg_policy_post"] = float(np.mean(selected_arr[post] == 2))
        out["sg_policy_tail"] = float(np.mean(selected_arr[tail] == 2))
        out["sg_supervisor_post"] = float(np.mean(selected_arr[post] == 1))
        out["sg_supervisor_tail"] = float(np.mean(selected_arr[tail] == 1))
        out["sg_fallback_post"] = float(np.mean(selected_arr[post] == 4))
    return out


def build_rows() -> list[dict]:
    rows = []
    for current_name, current_rel, previous_name, previous_rel, compare_rel in RUNS:
        current = run_metrics(load_bundle(current_rel))
        previous = run_metrics(load_bundle(previous_rel))
        compare_bundle = load_bundle(compare_rel)
        mpc_tail = finite_mean(np.asarray(compare_bundle["avg_rewards_mpc"], dtype=float)[-20:])
        rows.append(
            {
                "runner": current_name,
                "previous": previous_name,
                "current_algorithm": current["algorithm"],
                "previous_algorithm": previous["algorithm"],
                "current_path": str(RESULT_ROOT / current_rel),
                "previous_path": str(RESULT_ROOT / previous_rel),
                "tail_reward_current": current["tail_reward"],
                "tail_reward_previous": previous["tail_reward"],
                "tail_reward_ofmpc": mpc_tail,
                "tail_reward_vs_prev_pct": pct_higher_is_better(
                    current["tail_reward"], previous["tail_reward"]
                ),
                "tail_reward_vs_ofmpc_pct": pct_higher_is_better(current["tail_reward"], mpc_tail),
                "tail_scaled_mae_current": current["tail_scaled_mae"],
                "tail_scaled_mae_previous": previous["tail_scaled_mae"],
                "tail_scaled_mae_vs_prev_pct": pct_lower_is_better(
                    current["tail_scaled_mae"], previous["tail_scaled_mae"]
                ),
                "post_scaled_mae_current": current["post_scaled_mae"],
                "post_scaled_mae_previous": previous["post_scaled_mae"],
                "post_scaled_mae_vs_prev_pct": pct_lower_is_better(
                    current["post_scaled_mae"], previous["post_scaled_mae"]
                ),
                "tail_eta_mae_current": current["tail_eta_mae"],
                "tail_eta_mae_previous": previous["tail_eta_mae"],
                "tail_eta_mae_vs_prev_pct": pct_lower_is_better(
                    current["tail_eta_mae"], previous["tail_eta_mae"]
                ),
                "tail_T_mae_current": current["tail_T_mae"],
                "tail_T_mae_previous": previous["tail_T_mae"],
                "tail_T_mae_vs_prev_pct": pct_lower_is_better(
                    current["tail_T_mae"], previous["tail_T_mae"]
                ),
                "post_du_abs_current": current["post_du_abs"],
                "post_du_abs_previous": previous["post_du_abs"],
                "post_du_abs_vs_prev_pct": pct_lower_is_better(
                    current["post_du_abs"], previous["post_du_abs"]
                ),
                "sg_policy_post_current": current["sg_policy_post"],
                "sg_policy_post_previous": previous["sg_policy_post"],
                "sg_policy_tail_current": current["sg_policy_tail"],
                "sg_policy_tail_previous": previous["sg_policy_tail"],
                "sg_supervisor_tail_current": current["sg_supervisor_tail"],
                "sg_supervisor_tail_previous": previous["sg_supervisor_tail"],
                "sg_fallback_post_current": current["sg_fallback_post"],
                "prediction_score_current": current["prediction_score_mean"],
                "prediction_score_previous": previous["prediction_score_mean"],
                "gain_drift_current": current["gain_drift_mean"],
                "gain_drift_previous": previous["gain_drift_mean"],
                "accepted_post_current": current["accepted_fraction_post_warm"],
                "accepted_post_previous": previous["accepted_fraction_post_warm"],
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "polymer_sg_run_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _short_labels(rows: list[dict]) -> list[str]:
    return [
        row["runner"]
        .replace("SG-SAC ", "SAC\n")
        .replace("SG-DQN ", "DQN\n")
        .replace("SG-dueling DQN", "Duel DQN")
        for row in rows
    ]


def write_figures(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = _short_labels(rows)
    x_all = np.arange(len(rows))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x_all - width, [row["tail_reward_previous"] for row in rows], width, label="previous")
    ax.bar(x_all, [row["tail_reward_current"] for row in rows], width, label="current SG")
    ax.bar(x_all + width, [row["tail_reward_ofmpc"] for row in rows], width, label="OF-MPC reward")
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels)
    ax.set_ylabel("tail-20 mean reward (higher is better)")
    ax.set_title("Polymer disturbed tail reward comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tail_reward_comparison.png", dpi=180)
    plt.close(fig)

    tracking_rows = [row for row in rows if np.isfinite(row["tail_scaled_mae_current"])]
    tracking_labels = _short_labels(tracking_rows)
    x = np.arange(len(tracking_rows))
    pair_width = 0.35

    for filename, current_key, previous_key, ylabel, title in [
        (
            "tail_scaled_tracking_mae.png",
            "tail_scaled_mae_current",
            "tail_scaled_mae_previous",
            "tail-20 mean abs scaled tracking error",
            "Tail tracking error for runs with saved tracking-error logs",
        ),
        (
            "tail_eta_mae.png",
            "tail_eta_mae_current",
            "tail_eta_mae_previous",
            "tail-20 eta MAE, physical units",
            "Tail viscosity tracking error",
        ),
        (
            "tail_T_mae.png",
            "tail_T_mae_current",
            "tail_T_mae_previous",
            "tail-20 T MAE, K",
            "Tail temperature tracking error",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(
            x - pair_width / 2,
            [row[previous_key] for row in tracking_rows],
            pair_width,
            label="previous",
        )
        ax.bar(
            x + pair_width / 2,
            [row[current_key] for row in tracking_rows],
            pair_width,
            label="current SG",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tracking_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        x_all - pair_width / 2,
        [0.0 if not np.isfinite(row["sg_policy_post_previous"]) else row["sg_policy_post_previous"] for row in rows],
        pair_width,
        label="previous",
    )
    ax.bar(
        x_all + pair_width / 2,
        [0.0 if not np.isfinite(row["sg_policy_post_current"]) else row["sg_policy_post_current"] for row in rows],
        pair_width,
        label="current SG",
    )
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels)
    ax.set_ylabel("post-warm policy-selected fraction")
    ax.set_title("Policy authority admitted by supervisor gate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sg_policy_fraction.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_figures(rows)
    print(f"Wrote metrics and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
