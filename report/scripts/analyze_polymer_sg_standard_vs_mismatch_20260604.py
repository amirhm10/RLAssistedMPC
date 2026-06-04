"""Compare polymer supervisor-gated standard-state runs against mismatch-state runs.

The script reads saved polymer result bundles only. It writes CSV/JSON metrics
and figures for the standard-versus-mismatch extension of
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
OUT_DIR = REPO_ROOT / "report" / "figures" / "2026-06-04_polymer_sg_standard_vs_mismatch"

SOURCE_SUPERVISOR = 1
SOURCE_POLICY = 2
SOURCE_HELD = 3
SOURCE_FALLBACK = 4

PAIRS = [
    {
        "family": "TD3 residual",
        "class": "TD3",
        "mismatch": "sg_td3_residual_critic_warm3_conservative_disturb/20260601_182754/input_data.pkl",
        "standard": "sg_td3_residual_critic_warm3_conservative_disturb_standard/20260604_023110/input_data.pkl",
        "compare_mismatch": (
            "disturb_compare_sg_td3_residual_critic_warm3_conservative/20260601_182813/input_data.pkl"
        ),
        "compare_standard": (
            "disturb_compare_sg_td3_residual_critic_warm3_conservative_standard/20260604_023126/input_data.pkl"
        ),
    },
    {
        "family": "TD3 weights",
        "class": "TD3",
        "mismatch": "sg_td3_weights_critic_warm3_conservative_disturb/20260601_184718/input_data.pkl",
        "standard": "sg_td3_weights_critic_warm3_conservative_disturb_standard/20260604_023040/input_data.pkl",
        "compare_mismatch": (
            "disturb_compare_sg_td3_weights_critic_warm3_conservative/20260601_184731/input_data.pkl"
        ),
        "compare_standard": (
            "disturb_compare_sg_td3_weights_critic_warm3_conservative_standard/20260604_023054/input_data.pkl"
        ),
    },
    {
        "family": "TD3 Markov",
        "class": "TD3",
        "mismatch": "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb/20260601_215126/input_data.pkl",
        "standard": (
            "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_standard/20260604_031237/input_data.pkl"
        ),
        "compare_mismatch": (
            "disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow/20260601_215148/input_data.pkl"
        ),
        "compare_standard": (
            "disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow_standard/20260604_031301/input_data.pkl"
        ),
    },
    {
        "family": "SAC residual",
        "class": "SAC",
        "mismatch": "sg_sac_residual_detgate_hidden7_zero_shadow_disturb/20260603_230146/input_data.pkl",
        "standard": "sg_sac_residual_detgate_hidden7_zero_shadow_disturb_standard/20260604_025548/input_data.pkl",
        "compare_mismatch": "disturb_compare_sg_sac_residual_detgate_hidden7_zero_shadow/20260603_230203/input_data.pkl",
        "compare_standard": (
            "disturb_compare_sg_sac_residual_detgate_hidden7_zero_shadow_standard/20260604_025602/input_data.pkl"
        ),
    },
    {
        "family": "SAC weights",
        "class": "SAC",
        "mismatch": "sg_sac_weights_detgate_hidden7_identity_shadow_disturb/20260603_230130/input_data.pkl",
        "standard": "sg_sac_weights_detgate_hidden7_identity_shadow_disturb_standard/20260604_025515/input_data.pkl",
        "compare_mismatch": "disturb_compare_sg_sac_weights_detgate_hidden7_identity_shadow/20260603_230144/input_data.pkl",
        "compare_standard": (
            "disturb_compare_sg_sac_weights_detgate_hidden7_identity_shadow_standard/20260604_025527/input_data.pkl"
        ),
    },
    {
        "family": "SAC Markov",
        "class": "SAC",
        "mismatch": "sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_disturb/20260603_234148/input_data.pkl",
        "standard": (
            "sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_disturb_standard/20260604_032618/input_data.pkl"
        ),
        "compare_mismatch": "disturb_compare_sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow/20260603_234211/input_data.pkl",
        "compare_standard": (
            "disturb_compare_sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_standard/20260604_032640/input_data.pkl"
        ),
    },
    {
        "family": "DQN horizon",
        "class": "DQN",
        "mismatch": "horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/20260603_210127/input_data.pkl",
        "standard": "horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_standard/20260604_014818/input_data.pkl",
        "compare_mismatch": (
            "disturb_compare_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002/20260603_210142/input_data.pkl"
        ),
        "compare_standard": (
            "disturb_compare_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_standard/20260604_014832/input_data.pkl"
        ),
    },
    {
        "family": "Dueling DQN",
        "class": "DQN",
        "mismatch": (
            "dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/"
            "20260603_210913/input_data.pkl"
        ),
        "standard": (
            "dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_standard/"
            "20260604_022558/input_data.pkl"
        ),
        "compare_mismatch": (
            "disturb_compare_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002/"
            "20260603_210928/input_data.pkl"
        ),
        "compare_standard": (
            "disturb_compare_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_standard/"
            "20260604_022614/input_data.pkl"
        ),
    },
]


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def pct_higher(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
        return float("nan")
    return 100.0 * (current - previous) / abs(previous)


def pct_lower(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
        return float("nan")
    return 100.0 * (previous - current) / abs(previous)


def load_bundle(rel_path: str) -> dict:
    path = RESULT_ROOT / rel_path
    with path.open("rb") as fh:
        return pickle.load(fh)


def physical_output_error(bundle: dict) -> np.ndarray:
    y = np.asarray(bundle["y"], dtype=float)
    if y.shape[0] == int(bundle.get("nFE", y.shape[0] - 1)) + 1:
        y = y[1:, :]
    y_sp = np.asarray(bundle["y_sp"], dtype=float)
    n_inputs = int(np.asarray(bundle.get("u", np.empty((0, 2)))).shape[1] or 2)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    y_span = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / y_span
    y_sp_phys = (y_sp + y_ss_scaled) * y_span + data_min[n_inputs:]
    n = min(y.shape[0], y_sp_phys.shape[0])
    return y[:n, :] - y_sp_phys[:n, :]


def infer_state_dim(bundle: dict) -> int:
    xhat = np.asarray(bundle.get("xhatdhat", []), dtype=float)
    y_sp = np.asarray(bundle.get("y_sp", []), dtype=float)
    u = np.asarray(bundle.get("u", []), dtype=float)
    base_aug_dim = int(xhat.shape[0]) if xhat.ndim == 2 else int(bundle.get("xhatdhat_dim", 0) or 0)
    n_outputs = int(y_sp.shape[1]) if y_sp.ndim == 2 else 2
    n_inputs = int(u.shape[1]) if u.ndim == 2 else 2
    state_mode = str(bundle.get("state_mode") or "standard")
    markov_mode = bundle.get("markov_state_mode")

    base_dim = base_aug_dim + n_outputs + n_inputs
    mismatch_extra = 0 if state_mode == "standard" else 2 * n_outputs
    if bool(bundle.get("append_rho_to_state", False)) and state_mode != "standard":
        mismatch_extra += 1
    dim = base_dim + mismatch_extra
    if markov_mode is not None:
        z_dim = int(np.asarray(bundle.get("z_log", np.empty((0, 4))), dtype=float).shape[1] or 4)
        markov_base_mode = "standard" if str(markov_mode).lower() == "standard" else "mismatch"
        dim = base_dim + (0 if markov_base_mode == "standard" else 2 * n_outputs) + 2 * z_dim + 2
    return int(dim)


def window_slices(bundle: dict) -> dict[str, slice]:
    n_steps = int(bundle.get("nFE", len(bundle.get("rewards_step", []))))
    warm = int(bundle.get("warm_start_step", 0) or 0)
    steps_per_sub = int(bundle.get("time_in_sub_episodes", 800) or 800)
    action_freeze = int(bundle.get("post_warm_start_action_freeze_subepisodes", 0) or 0)
    release = warm + action_freeze * steps_per_sub
    tail_start = max(0, n_steps - 20 * steps_per_sub)
    return {
        "post": slice(warm, n_steps),
        "tail": slice(tail_start, n_steps),
        "first_live_10": slice(release, min(n_steps, release + 10 * steps_per_sub)),
    }


def compare_tail_reward(rel_path: str) -> float:
    bundle = load_bundle(rel_path)
    if "avg_rewards_mpc" not in bundle:
        return float("nan")
    return finite_mean(np.asarray(bundle["avg_rewards_mpc"], dtype=float)[-20:])


def selected_fraction(selected: np.ndarray, value: int, window: slice) -> float:
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected[window] == value))


def bundle_metrics(bundle: dict, compare_rel: str) -> dict:
    windows = window_slices(bundle)
    rewards = np.asarray(bundle.get("rewards_step", []), dtype=float)
    delta_y = np.asarray(bundle.get("delta_y_storage", []), dtype=float)
    delta_u = np.asarray(bundle.get("delta_u_storage", []), dtype=float)
    selected = np.asarray(bundle.get("sg_selected_source_log", []), dtype=float)
    phys_error = physical_output_error(bundle)
    summary = dict(bundle.get("summary_metrics", {}) or {})

    return {
        "agent_kind": bundle.get("agent_kind") or bundle.get("algorithm"),
        "state_mode": bundle.get("state_mode"),
        "markov_state_mode": bundle.get("markov_state_mode"),
        "state_dim_inferred": infer_state_dim(bundle),
        "tail_reward": finite_mean(rewards[windows["tail"]]),
        "post_reward": finite_mean(rewards[windows["post"]]),
        "first_live_10_reward": finite_mean(rewards[windows["first_live_10"]]),
        "tail_scaled_mae": finite_mean(np.abs(delta_y[windows["tail"]])),
        "tail_eta_phys_mae": finite_mean(np.abs(phys_error[windows["tail"], 0])),
        "tail_T_phys_mae": finite_mean(np.abs(phys_error[windows["tail"], 1])),
        "post_du_abs": finite_mean(np.abs(delta_u[windows["post"]])),
        "tail_du_abs": finite_mean(np.abs(delta_u[windows["tail"]])),
        "sg_policy_post": selected_fraction(selected, SOURCE_POLICY, windows["post"]),
        "sg_policy_tail": selected_fraction(selected, SOURCE_POLICY, windows["tail"]),
        "sg_supervisor_tail": selected_fraction(selected, SOURCE_SUPERVISOR, windows["tail"]),
        "sg_held_post": selected_fraction(selected, SOURCE_HELD, windows["post"]),
        "sg_fallback_post": selected_fraction(selected, SOURCE_FALLBACK, windows["post"]),
        "prediction_score_mean": summary.get("prediction_score_mean", float("nan")),
        "gain_drift_mean": summary.get("gain_drift_mean", float("nan")),
        "ofmpc_tail_reward": compare_tail_reward(compare_rel),
    }


def build_rows() -> list[dict]:
    rows = []
    for pair in PAIRS:
        mismatch = bundle_metrics(load_bundle(pair["mismatch"]), pair["compare_mismatch"])
        standard = bundle_metrics(load_bundle(pair["standard"]), pair["compare_standard"])
        row = {
            "family": pair["family"],
            "class": pair["class"],
            "mismatch_path": pair["mismatch"],
            "standard_path": pair["standard"],
        }
        for prefix, metrics in (("mismatch", mismatch), ("standard", standard)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        for key, higher_is_better in (
            ("tail_reward", True),
            ("post_reward", True),
            ("first_live_10_reward", True),
            ("tail_scaled_mae", False),
            ("tail_eta_phys_mae", False),
            ("tail_T_phys_mae", False),
            ("post_du_abs", False),
            ("tail_du_abs", False),
            ("sg_policy_tail", False),
            ("prediction_score_mean", False),
            ("gain_drift_mean", False),
        ):
            now = standard.get(key, float("nan"))
            old = mismatch.get(key, float("nan"))
            row[f"delta_{key}"] = now - old if np.isfinite(now) and np.isfinite(old) else float("nan")
            row[f"pct_{key}"] = pct_higher(now, old) if higher_is_better else pct_lower(now, old)
        row["delta_state_dim"] = standard["state_dim_inferred"] - mismatch["state_dim_inferred"]
        row["pct_state_dim_reduction"] = pct_lower(
            standard["state_dim_inferred"],
            mismatch["state_dim_inferred"],
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_label(label: str) -> str:
    return (
        label.replace("TD3 ", "TD3\n")
        .replace("SAC ", "SAC\n")
        .replace("DQN ", "DQN\n")
        .replace("Dueling DQN", "Dueling\nDQN")
    )


def plot_grouped(rows: list[dict], filename: str, metric: str, ylabel: str, title: str) -> None:
    labels = [short_label(row["family"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(x - width / 2, [row[f"mismatch_{metric}"] for row in rows], width, label="mismatch")
    ax.bar(x + width / 2, [row[f"standard_{metric}"] for row in rows], width, label="standard")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def plot_pct_change(rows: list[dict]) -> None:
    labels = [short_label(row["family"]) for row in rows]
    metrics = [
        ("pct_tail_reward", "reward", "#2f6f9f"),
        ("pct_tail_scaled_mae", "scaled MAE", "#9f5f2f"),
        ("pct_tail_T_phys_mae", "T MAE", "#6f8f3f"),
    ]
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for offset, (key, label, color) in zip((-width, 0.0, width), metrics):
        ax.bar(x + offset, [row[key] for row in rows], width, label=label, color=color)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("standard vs mismatch change, percent")
    ax.set_title("Standard-state change versus mismatch-state runs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "standard_vs_mismatch_percent_change.png", dpi=180)
    plt.close(fig)


def plot_reward_traces(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, rows):
        for label, rel, color in (
            ("mismatch", row["mismatch_path"], "#777777"),
            ("standard", row["standard_path"], "#1f77b4"),
        ):
            bundle = load_bundle(rel)
            rewards = np.asarray(bundle.get("avg_rewards", []), dtype=float)
            if rewards.size:
                ax.plot(np.arange(1, rewards.size + 1), rewards, label=label, color=color, linewidth=1.2)
        ax.axvline(10, color="#111111", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(row["family"], fontsize=10)
        ax.grid(alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel("subepisode")
    for ax in axes[:, 0]:
        ax.set_ylabel("subepisode reward")
    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Polymer SG standard versus mismatch reward traces", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "standard_vs_mismatch_reward_traces.png", dpi=180)
    plt.close(fig)


def write_figures(rows: list[dict]) -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_grouped(
        rows,
        "tail_reward_standard_vs_mismatch.png",
        "tail_reward",
        "tail-20 mean reward (higher is better)",
        "Polymer SG tail reward: standard versus mismatch state",
    )
    plot_grouped(
        rows,
        "tail_scaled_mae_standard_vs_mismatch.png",
        "tail_scaled_mae",
        "tail-20 mean abs scaled output error",
        "Polymer SG scaled tracking error",
    )
    plot_grouped(
        rows,
        "tail_T_mae_standard_vs_mismatch.png",
        "tail_T_phys_mae",
        "tail-20 T MAE, physical units",
        "Polymer SG temperature tracking error",
    )
    plot_grouped(
        rows,
        "tail_policy_fraction_standard_vs_mismatch.png",
        "sg_policy_tail",
        "tail policy-selection fraction",
        "Policy authority admitted by the supervisor gate",
    )
    plot_grouped(
        rows,
        "state_dim_standard_vs_mismatch.png",
        "state_dim_inferred",
        "inferred RL state dimension",
        "RL state dimensionality reduction from standard mode",
    )
    plot_pct_change(rows)
    plot_reward_traces(rows)
    return [
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in sorted(OUT_DIR.glob("*.png"))
    ]


def main() -> None:
    rows = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv = OUT_DIR / "polymer_sg_standard_vs_mismatch_metrics.csv"
    write_csv(metrics_csv, rows)
    figures = write_figures(rows)
    payload = {
        "rows": rows,
        "figures": figures,
        "metrics_csv": str(metrics_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"n_pairs": len(rows), "figures": figures, "metrics_csv": payload["metrics_csv"]}, indent=2))


if __name__ == "__main__":
    main()
