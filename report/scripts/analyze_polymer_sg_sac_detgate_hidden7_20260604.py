"""Analyze polymer SG-SAC detgate hidden-7 reruns against pre-detgate SG-SAC.

The script reads saved polymer result bundles only. It writes CSV/JSON metrics
and regenerates local figures for the extension of
``report/polymer_sg_sac_sg_dqn_finished_runs_2026-06-04.md``.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "Polymer" / "Results"
OUT_DIR = REPO_ROOT / "report" / "figures" / "2026-06-04_polymer_sg_sac_detgate_hidden7"

SOURCE_WARM_START = 0
SOURCE_SUPERVISOR = 1
SOURCE_POLICY = 2
SOURCE_HELD = 3
SOURCE_FALLBACK = 4

PAIRS = [
    {
        "family": "Residual",
        "previous_label": "critic_warm3",
        "current_label": "detgate_hidden7",
        "previous": "sg_sac_residual_critic_warm3_zero_shadow_disturb/20260603_205207/input_data.pkl",
        "current": "sg_sac_residual_detgate_hidden7_zero_shadow_disturb/20260603_230146/input_data.pkl",
        "compare_previous": "disturb_compare_sg_sac_residual_critic_warm3_zero_shadow/20260603_205225/input_data.pkl",
        "compare_current": "disturb_compare_sg_sac_residual_detgate_hidden7_zero_shadow/20260603_230203/input_data.pkl",
        "dominance_margin": 0.5,
        "previous_action_freeze_subepisodes": 3,
        "previous_actor_freeze_subepisodes": 3,
        "current_action_freeze_subepisodes": 10,
        "current_actor_freeze_subepisodes": 3,
    },
    {
        "family": "Weights",
        "previous_label": "critic_warm3",
        "current_label": "detgate_hidden7",
        "previous": "sg_sac_weights_critic_warm3_identity_shadow_disturb/20260603_205218/input_data.pkl",
        "current": "sg_sac_weights_detgate_hidden7_identity_shadow_disturb/20260603_230130/input_data.pkl",
        "compare_previous": "disturb_compare_sg_sac_weights_critic_warm3_identity_shadow/20260603_205232/input_data.pkl",
        "compare_current": "disturb_compare_sg_sac_weights_detgate_hidden7_identity_shadow/20260603_230144/input_data.pkl",
        "dominance_margin": 0.5,
        "previous_action_freeze_subepisodes": 3,
        "previous_actor_freeze_subepisodes": 3,
        "current_action_freeze_subepisodes": 10,
        "current_actor_freeze_subepisodes": 3,
    },
    {
        "family": "Markov",
        "previous_label": "critic_warm3",
        "current_label": "detgate_hidden7",
        "previous": "sg_sac_markov_critic_warm3_ls_else_mpc_shadow_disturb/20260603_212647/input_data.pkl",
        "current": "sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_disturb/20260603_234148/input_data.pkl",
        "compare_previous": "disturb_compare_sg_sac_markov_critic_warm3_ls_else_mpc_shadow/20260603_212712/input_data.pkl",
        "compare_current": "disturb_compare_sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow/20260603_234211/input_data.pkl",
        "dominance_margin": 0.0,
        "previous_action_freeze_subepisodes": 3,
        "previous_actor_freeze_subepisodes": 3,
        "current_action_freeze_subepisodes": 10,
        "current_actor_freeze_subepisodes": 3,
    },
]


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_fraction(mask) -> float:
    arr = np.asarray(mask)
    if arr.size == 0:
        return float("nan")
    finite = np.isfinite(arr.astype(float, copy=False)) if arr.dtype != bool else np.ones(arr.shape, dtype=bool)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(arr[finite]))


def pct_higher(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
        return float("nan")
    return 100.0 * (current - previous) / abs(previous)


def pct_lower(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
        return float("nan")
    return 100.0 * (previous - current) / abs(previous)


def load_bundle(rel_path: str) -> dict:
    with (RESULT_ROOT / rel_path).open("rb") as fh:
        return pickle.load(fh)


def output_spans(bundle: dict) -> np.ndarray:
    n_inputs = int(np.asarray(bundle.get("u", np.empty((0, 2)))).shape[1] or 2)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    return data_max[n_inputs:] - data_min[n_inputs:]


def physical_output_error(bundle: dict) -> np.ndarray:
    y = np.asarray(bundle["y"], dtype=float)
    if y.shape[0] == int(bundle.get("nFE", y.shape[0] - 1)) + 1:
        y = y[1:, :]
    y_sp = np.asarray(bundle["y_sp"], dtype=float)
    n_inputs = int(np.asarray(bundle.get("u", np.empty((0, 2)))).shape[1] or 2)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp_phys = (y_sp + y_ss_scaled) * (data_max[n_inputs:] - data_min[n_inputs:]) + data_min[n_inputs:]
    n = min(y.shape[0], y_sp_phys.shape[0])
    return y[:n, :] - y_sp_phys[:n, :]


def slice_for(bundle: dict, start_step: int, subepisodes: int | None = None) -> slice:
    n_steps = int(bundle.get("nFE", len(bundle.get("rewards_step", []))))
    steps_per_sub = int(bundle.get("time_in_sub_episodes", 800) or 800)
    if subepisodes is None:
        return slice(max(0, start_step), n_steps)
    return slice(max(0, start_step), min(n_steps, start_step + int(subepisodes) * steps_per_sub))


def bundle_metrics(
    bundle: dict,
    dominance_margin: float,
    *,
    fallback_action_freeze_subepisodes: int = 0,
    fallback_actor_freeze_subepisodes: int = 0,
) -> dict:
    n_steps = int(bundle.get("nFE", len(bundle.get("rewards_step", []))))
    warm_start = int(bundle.get("warm_start_step", 0) or 0)
    steps_per_sub = int(bundle.get("time_in_sub_episodes", 800) or 800)
    tail = slice(max(0, n_steps - 20 * steps_per_sub), n_steps)
    post = slice(warm_start, n_steps)

    action_freeze_sub = int(
        bundle.get(
            "phase1_action_freeze_subepisodes",
            bundle.get("config_snapshot", {}).get(
                "post_warm_start_action_freeze_subepisodes",
                fallback_action_freeze_subepisodes,
            )
            or fallback_action_freeze_subepisodes,
        )
    )
    actor_freeze_sub = int(
        bundle.get(
            "phase1_actor_freeze_subepisodes",
            bundle.get("config_snapshot", {}).get(
                "post_warm_start_actor_freeze_subepisodes",
                fallback_actor_freeze_subepisodes,
            )
            or fallback_actor_freeze_subepisodes,
        )
    )
    hidden_actor_sub = max(0, action_freeze_sub - actor_freeze_sub)
    action_release_step = warm_start + action_freeze_sub * steps_per_sub
    live_first10 = slice_for(bundle, action_release_step, 10)
    hidden_window = slice_for(bundle, warm_start + actor_freeze_sub * steps_per_sub, hidden_actor_sub)

    rewards = np.asarray(bundle.get("rewards_step", []), dtype=float)
    avg_rewards = np.asarray(bundle.get("avg_rewards", []), dtype=float)
    delta_y = np.asarray(bundle.get("delta_y_storage", []), dtype=float)
    delta_u = np.asarray(bundle.get("delta_u_storage", []), dtype=float)
    selected = np.asarray(bundle.get("sg_selected_source_log", []), dtype=float)
    phys_error = physical_output_error(bundle)

    cfg = dict(bundle.get("config_snapshot", {}) or {})
    gate = dict(bundle.get("supervisor_gate", {}) or cfg.get("supervisor_gate", {}) or {})
    summary = dict(bundle.get("summary_metrics", {}) or {})

    q1_policy = np.asarray(bundle.get("sg_q1_policy_log", []), dtype=float)
    q2_policy = np.asarray(bundle.get("sg_q2_policy_log", []), dtype=float)
    q1_sup = np.asarray(bundle.get("sg_q1_supervisor_log", []), dtype=float)
    q2_sup = np.asarray(bundle.get("sg_q2_supervisor_log", []), dtype=float)
    score_policy = np.asarray(bundle.get("sg_score_policy_log", []), dtype=float)
    score_sup = np.asarray(bundle.get("sg_score_supervisor_log", []), dtype=float)
    bc_loss = np.asarray(bundle.get("sg_bc_loss_trace", []), dtype=float)
    sampled_bc_loss = np.asarray(bundle.get("sg_sampled_bc_loss_trace", []), dtype=float)

    dominance = np.full(n_steps, np.nan, dtype=float)
    n_q = min(n_steps, q1_policy.size, q2_policy.size, q1_sup.size, q2_sup.size)
    if n_q:
        q_mask = (
            np.isfinite(q1_policy[:n_q])
            & np.isfinite(q2_policy[:n_q])
            & np.isfinite(q1_sup[:n_q])
            & np.isfinite(q2_sup[:n_q])
        )
        dom = (
            (q1_policy[:n_q] >= q1_sup[:n_q] + dominance_margin)
            & (q2_policy[:n_q] >= q2_sup[:n_q] + dominance_margin)
        )
        dominance[:n_q] = np.where(q_mask, dom.astype(float), np.nan)

    score_gap = np.full(n_steps, np.nan, dtype=float)
    n_score = min(n_steps, score_policy.size, score_sup.size)
    if n_score:
        score_gap[:n_score] = score_policy[:n_score] - score_sup[:n_score]

    def selected_fraction(value: int, window: slice) -> float:
        if selected.size == 0:
            return float("nan")
        return float(np.mean(selected[window] == value))

    metrics = {
        "state_mode": bundle.get("state_mode"),
        "markov_state_mode": bundle.get("markov_state_mode"),
        "action_freeze_subepisodes": action_freeze_sub,
        "actor_freeze_subepisodes": actor_freeze_sub,
        "hidden_actor_train_subepisodes": hidden_actor_sub,
        "candidate_mode": gate.get("candidate_mode", "sampled_or_default"),
        "advantage_margin": gate.get("advantage_margin"),
        "critic_dominance_gate_enabled": bool(gate.get("critic_dominance_gate_enabled", False)),
        "critic_dominance_margin": gate.get("critic_dominance_margin"),
        "sampled_supervisor_bc_weight": gate.get("sampled_supervisor_bc_weight", 0.0),
        "tail_reward": finite_mean(rewards[tail]) if rewards.size else float("nan"),
        "post_reward": finite_mean(rewards[post]) if rewards.size else float("nan"),
        "live_first10_reward": finite_mean(rewards[live_first10]) if rewards.size else float("nan"),
        "hidden_window_reward": finite_mean(rewards[hidden_window]) if rewards.size and hidden_actor_sub else float("nan"),
        "final_episode_reward": float(avg_rewards[-1]) if avg_rewards.size else float("nan"),
        "tail_scaled_mae": finite_mean(np.abs(delta_y[tail])) if delta_y.size else float("nan"),
        "tail_eta_phys_mae": finite_mean(np.abs(phys_error[tail, 0])) if phys_error.size else float("nan"),
        "tail_T_phys_mae": finite_mean(np.abs(phys_error[tail, 1])) if phys_error.size else float("nan"),
        "post_du_abs": finite_mean(np.abs(delta_u[post])) if delta_u.size else float("nan"),
        "tail_du_abs": finite_mean(np.abs(delta_u[tail])) if delta_u.size else float("nan"),
        "sg_policy_post": selected_fraction(SOURCE_POLICY, post),
        "sg_policy_tail": selected_fraction(SOURCE_POLICY, tail),
        "sg_supervisor_tail": selected_fraction(SOURCE_SUPERVISOR, tail),
        "sg_held_hidden": selected_fraction(SOURCE_HELD, hidden_window) if hidden_actor_sub else float("nan"),
        "sg_fallback_post": selected_fraction(SOURCE_FALLBACK, post),
        "score_gap_tail_mean": finite_mean(score_gap[tail]),
        "score_gap_post_mean": finite_mean(score_gap[post]),
        "critic_dominance_tail_fraction": finite_mean(dominance[tail]),
        "critic_dominance_post_fraction": finite_mean(dominance[post]),
        "sg_bc_loss_mean": finite_mean(bc_loss),
        "sg_sampled_bc_loss_mean": finite_mean(sampled_bc_loss),
        "prediction_score_mean": summary.get("prediction_score_mean", float("nan")),
        "gain_drift_mean": summary.get("gain_drift_mean", float("nan")),
        "markov_policy_fraction_summary": summary.get("sg_policy_fraction", float("nan")),
        "ofmpc_tail_reward": float("nan"),
    }
    return metrics


def compare_bundle_metrics(rel_path: str) -> dict:
    bundle = load_bundle(rel_path)
    out = {}
    for key in ("avg_rewards_mpc", "avg_rewards_rl"):
        if key in bundle:
            out[key] = finite_mean(np.asarray(bundle[key], dtype=float)[-20:])
    return out


def build_rows() -> list[dict]:
    rows = []
    for pair in PAIRS:
        current_bundle = load_bundle(pair["current"])
        previous_bundle = load_bundle(pair["previous"])
        current = bundle_metrics(
            current_bundle,
            float(pair["dominance_margin"]),
            fallback_action_freeze_subepisodes=int(pair["current_action_freeze_subepisodes"]),
            fallback_actor_freeze_subepisodes=int(pair["current_actor_freeze_subepisodes"]),
        )
        previous = bundle_metrics(
            previous_bundle,
            float(pair["dominance_margin"]),
            fallback_action_freeze_subepisodes=int(pair["previous_action_freeze_subepisodes"]),
            fallback_actor_freeze_subepisodes=int(pair["previous_actor_freeze_subepisodes"]),
        )
        compare_current = compare_bundle_metrics(pair["compare_current"])
        compare_previous = compare_bundle_metrics(pair["compare_previous"])
        current["ofmpc_tail_reward"] = compare_current.get("avg_rewards_mpc", float("nan"))
        previous["ofmpc_tail_reward"] = compare_previous.get("avg_rewards_mpc", float("nan"))

        base = {
            "family": pair["family"],
            "previous_label": pair["previous_label"],
            "current_label": pair["current_label"],
            "previous_path": pair["previous"],
            "current_path": pair["current"],
        }
        for prefix, metrics in (("previous", previous), ("current", current)):
            for key, value in metrics.items():
                base[f"{prefix}_{key}"] = value
        for key, higher_is_better in (
            ("tail_reward", True),
            ("post_reward", True),
            ("live_first10_reward", True),
            ("tail_scaled_mae", False),
            ("tail_eta_phys_mae", False),
            ("tail_T_phys_mae", False),
            ("post_du_abs", False),
            ("tail_du_abs", False),
            ("sg_policy_tail", False),
            ("score_gap_tail_mean", True),
            ("critic_dominance_tail_fraction", True),
            ("prediction_score_mean", False),
            ("gain_drift_mean", False),
        ):
            now = current.get(key, float("nan"))
            old = previous.get(key, float("nan"))
            base[f"delta_{key}"] = now - old if np.isfinite(now) and np.isfinite(old) else float("nan")
            base[f"pct_{key}"] = pct_higher(now, old) if higher_is_better else pct_lower(now, old)
        rows.append(base)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(rows: list[dict], filename: str, metric: str, ylabel: str, title: str) -> None:
    labels = [row["family"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - width / 2, [row[f"previous_{metric}"] for row in rows], width, label="critic_warm3")
    ax.bar(x + width / 2, [row[f"current_{metric}"] for row in rows], width, label="detgate_hidden7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def plot_reward_traces(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for ax, row in zip(axes, rows):
        prev = load_bundle(row["previous_path"])
        curr = load_bundle(row["current_path"])
        for label, bundle, color in (
            ("critic_warm3", prev, "#777777"),
            ("detgate_hidden7", curr, "#1f77b4"),
        ):
            rewards = np.asarray(bundle.get("avg_rewards", []), dtype=float)
            if rewards.size:
                ax.plot(np.arange(1, rewards.size + 1), rewards, label=label, color=color, linewidth=1.4)
        ax.axvline(10, color="#111111", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.axvline(row["previous_action_freeze_subepisodes"] + 10, color="#777777", linestyle=":", linewidth=0.9)
        ax.axvline(row["current_action_freeze_subepisodes"] + 10, color="#1f77b4", linestyle=":", linewidth=0.9)
        ax.set_title(row["family"])
        ax.set_xlabel("subepisode")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rolling subepisode reward")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Polymer SG-SAC reward traces: warm start and hidden release context", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "sg_sac_reward_traces.png", dpi=180)
    plt.close(fig)


def write_figures(rows: list[dict]) -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_grouped_bars(
        rows,
        "tail_reward_detgate_vs_critic_warm3.png",
        "tail_reward",
        "tail-20 mean reward (higher is better)",
        "SG-SAC tail reward: safer detgate hidden-7 versus pre-detgate",
    )
    plot_grouped_bars(
        rows,
        "tail_scaled_mae_detgate_vs_critic_warm3.png",
        "tail_scaled_mae",
        "tail-20 mean abs scaled output error",
        "SG-SAC scaled tracking error",
    )
    plot_grouped_bars(
        rows,
        "tail_eta_phys_mae_detgate_vs_critic_warm3.png",
        "tail_eta_phys_mae",
        "tail-20 eta MAE, physical units",
        "SG-SAC viscosity tracking error",
    )
    plot_grouped_bars(
        rows,
        "tail_T_phys_mae_detgate_vs_critic_warm3.png",
        "tail_T_phys_mae",
        "tail-20 T MAE, physical units",
        "SG-SAC temperature tracking error",
    )
    plot_grouped_bars(
        rows,
        "tail_policy_fraction_detgate_vs_critic_warm3.png",
        "sg_policy_tail",
        "tail policy-selection fraction",
        "SG-SAC policy authority admitted by the gate",
    )
    plot_grouped_bars(
        rows,
        "tail_critic_dominance_detgate_vs_critic_warm3.png",
        "critic_dominance_tail_fraction",
        "tail twin-critic dominance fraction",
        "Twin-critic dominance diagnostic",
    )
    plot_reward_traces(rows)
    return [
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in sorted(OUT_DIR.glob("*.png"))
    ]


def main() -> None:
    rows = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "polymer_sg_sac_detgate_hidden7_metrics.csv", rows)
    figures = write_figures(rows)
    payload = {
        "rows": rows,
        "figures": figures,
        "metrics_csv": str((OUT_DIR / "polymer_sg_sac_detgate_hidden7_metrics.csv").relative_to(REPO_ROOT)).replace(
            "\\", "/"
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"n_pairs": len(rows), "figures": figures, "metrics_csv": payload["metrics_csv"]}, indent=2))


if __name__ == "__main__":
    main()
