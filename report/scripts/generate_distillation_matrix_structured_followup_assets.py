from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.config import (
    DISTILLATION_RL_SETPOINTS_PHYS,
    RL_REWARD_DEFAULTS as DISTILLATION_REWARD_DEFAULTS,
)
from systems.polymer.config import (
    POLYMER_RL_SETPOINTS_PHYS,
    RL_REWARD_DEFAULTS as POLYMER_REWARD_DEFAULTS,
)


FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_matrix_structured_followup_20260508"
DISTILLATION_SUMMARY_CSV = FIG_DIR / "distillation_decision_interval_followup_summary.csv"
CROSS_SYSTEM_SUMMARY_CSV = FIG_DIR / "cross_system_latest_summary.csv"
STATS_CSV = FIG_DIR / "exploratory_stats_summary.csv"
REWARD_GEOMETRY_CSV = FIG_DIR / "reward_geometry_summary.csv"
SUMMARY_JSON = FIG_DIR / "summary.json"

N_EPISODES = 200
DISTILLATION_BASELINE_PATH = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
POLYMER_BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    system: str
    family: str
    phase: str
    rl_path: Path
    compare_path: Path
    baseline_path: Path


RUNS = [
    RunSpec(
        key="distillation_matrix_old",
        label="Distillation scalar matrix (May 3, decision interval 1)",
        system="distillation",
        family="scalar_matrix",
        phase="previous",
        rl_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified"
        / "20260503_062605"
        / "input_data.pkl",
        compare_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_matrix_td3_disturb_fluctuation_mismatch"
        / "20260503_062617"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_PATH,
    ),
    RunSpec(
        key="distillation_matrix_latest",
        label="Distillation scalar matrix (May 8, decision interval 20)",
        system="distillation",
        family="scalar_matrix",
        phase="latest",
        rl_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified"
        / "20260508_015834"
        / "input_data.pkl",
        compare_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_matrix_td3_disturb_fluctuation_mismatch"
        / "20260508_015844"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_PATH,
    ),
    RunSpec(
        key="distillation_structured_old",
        label="Distillation structured matrix (May 3, decision interval 1)",
        system="distillation",
        family="structured_matrix",
        phase="previous",
        rl_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_structured_matrix_td3_disturb_fluctuation_mismatch_unified"
        / "20260503_083936"
        / "input_data.pkl",
        compare_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_structured_matrix_td3_disturb_fluctuation_mismatch"
        / "20260503_083950"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_PATH,
    ),
    RunSpec(
        key="distillation_structured_latest",
        label="Distillation structured matrix (May 8, decision interval 20)",
        system="distillation",
        family="structured_matrix",
        phase="latest",
        rl_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_structured_matrix_td3_disturb_fluctuation_mismatch_unified"
        / "20260508_005027"
        / "input_data.pkl",
        compare_path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_structured_matrix_td3_disturb_fluctuation_mismatch"
        / "20260508_005037"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_PATH,
    ),
    RunSpec(
        key="polymer_matrix_latest",
        label="Polymer scalar matrix (latest)",
        system="polymer",
        family="scalar_matrix",
        phase="latest",
        rl_path=REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260501_214340" / "input_data.pkl",
        compare_path=REPO_ROOT
        / "Polymer"
        / "Results"
        / "disturb_compare_td3_multipliers"
        / "20260501_214354"
        / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_PATH,
    ),
    RunSpec(
        key="polymer_structured_latest",
        label="Polymer structured matrix (latest)",
        system="polymer",
        family="structured_matrix",
        phase="latest",
        rl_path=REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_structured_matrices_disturb"
        / "20260501_212829"
        / "input_data.pkl",
        compare_path=REPO_ROOT
        / "Polymer"
        / "Results"
        / "disturb_compare_td3_structured_matrices"
        / "20260501_212846"
        / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_PATH,
    ),
]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def reward_delta(compare_bundle: dict) -> np.ndarray:
    rl = np.asarray(compare_bundle["avg_rewards_rl"], float).reshape(-1)
    mpc = np.asarray(compare_bundle["avg_rewards_mpc"], float).reshape(-1)
    if mpc.size != rl.size:
        mpc = np.full(rl.shape, float(mpc[-1]), dtype=float)
    return rl - mpc


def infer_steps_per_episode(bundle: dict) -> int:
    delta_y = np.asarray(bundle["delta_y_storage"], float)
    return int(delta_y.shape[0] // N_EPISODES)


def infer_first_live_episode(bundle: dict) -> int:
    cfg = dict(bundle.get("config_snapshot", {}))
    warm_start = int(cfg.get("warm_start", 0) or 0)
    freeze = int(cfg.get("post_warm_start_action_freeze_subepisodes", 0) or 0)
    return warm_start + freeze + 1


def per_episode_metrics(bundle: dict) -> dict[str, np.ndarray | int]:
    delta_y = np.asarray(bundle["delta_y_storage"], float)
    delta_u = np.asarray(bundle["delta_u_storage"], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(delta_u.shape[1])
    output_ranges = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    input_ranges = np.maximum(data_max[:n_inputs] - data_min[:n_inputs], 1e-12)
    steps_per_episode = infer_steps_per_episode(bundle)

    err_scaled = delta_y.reshape(N_EPISODES, steps_per_episode, -1)
    du_scaled = delta_u.reshape(N_EPISODES, steps_per_episode, -1)
    err_phys = np.abs(err_scaled) * output_ranges.reshape(1, 1, -1)
    du_phys = np.abs(du_scaled) * input_ranges.reshape(1, 1, -1)

    metrics = {
        "steps_per_episode": steps_per_episode,
        "err_scaled": err_scaled,
        "du_scaled": du_scaled,
        "mae_out_phys": err_phys.mean(axis=1),
        "mae_mean_phys": err_phys.mean(axis=(1, 2)),
        "rmse_scaled": np.sqrt(np.mean(err_scaled**2, axis=(1, 2))),
        "iae_scaled": np.sum(np.abs(err_scaled), axis=(1, 2)),
        "move_mean_phys": du_phys.mean(axis=(1, 2)),
        "move_in_phys": du_phys.mean(axis=1),
        "max_abs_err_scaled": np.max(np.abs(err_scaled), axis=(1, 2)),
        "final_episode_err_phys": err_phys[-1],
    }
    return metrics


def reshape_log(bundle: dict, key: str, steps_per_episode: int) -> np.ndarray | None:
    if key not in bundle or bundle[key] is None:
        return None
    arr = np.asarray(bundle[key], float)
    if arr.size == 0:
        return None
    try:
        return arr.reshape(N_EPISODES, steps_per_episode, *arr.shape[1:])
    except ValueError:
        return None


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, float).reshape(-1)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def exploratory_sign_test(values: np.ndarray) -> tuple[int, int, float]:
    arr = np.asarray(values, float).reshape(-1)
    pos = int(np.sum(arr > 0.0))
    neg = int(np.sum(arr < 0.0))
    trials = pos + neg
    pvalue = float("nan") if trials == 0 else float(binomtest(pos, trials, p=0.5).pvalue)
    return pos, neg, pvalue


def compute_reward_geometry(system: str) -> list[dict]:
    if system == "distillation":
        reward_cfg = DISTILLATION_REWARD_DEFAULTS
        setpoints = np.asarray(DISTILLATION_RL_SETPOINTS_PHYS, float)
        baseline_bundle = load_pickle(DISTILLATION_BASELINE_PATH)
    else:
        reward_cfg = POLYMER_REWARD_DEFAULTS
        setpoints = np.asarray(POLYMER_RL_SETPOINTS_PHYS, float)
        baseline_bundle = load_pickle(POLYMER_BASELINE_PATH)

    q_diag = np.asarray(reward_cfg["Q_diag"], float)
    k_rel = np.asarray(reward_cfg["k_rel"], float)
    band_floor = np.asarray(reward_cfg["band_floor_phys"], float)
    beta = float(reward_cfg["beta"])
    data_min = np.asarray(baseline_bundle["data_min"], float)
    data_max = np.asarray(baseline_bundle["data_max"], float)
    n_inputs = int(np.asarray(baseline_bundle["delta_u_storage"], float).shape[1])
    output_ranges = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    rows = []
    for idx, y_sp in enumerate(setpoints, start=1):
        band_phys = np.maximum(k_rel * np.abs(y_sp), band_floor)
        band_scaled = band_phys / output_ranges
        slope_at_edge = 2.0 * q_diag * band_scaled
        bonus_prefactor = beta * q_diag * (band_scaled**2)
        rows.append(
            {
                "system": system,
                "setpoint": f"SP{idx}",
                "output1_band_phys": float(band_phys[0]),
                "output2_band_phys": float(band_phys[1]),
                "output1_band_scaled": float(band_scaled[0]),
                "output2_band_scaled": float(band_scaled[1]),
                "output1_edge_slope": float(slope_at_edge[0]),
                "output2_edge_slope": float(slope_at_edge[1]),
                "output1_bonus_prefactor": float(bonus_prefactor[0]),
                "output2_bonus_prefactor": float(bonus_prefactor[1]),
                "edge_ratio_out1_to_out2": float(slope_at_edge[0] / max(slope_at_edge[1], 1e-12)),
                "bonus_ratio_out1_to_out2": float(bonus_prefactor[0] / max(bonus_prefactor[1], 1e-12)),
                "q1_edge_equalized_target": float(q_diag[1] * band_scaled[1] / max(band_scaled[0], 1e-12)),
                "q1_bonus_equalized_target": float(q_diag[1] * (band_scaled[1] / max(band_scaled[0], 1e-12)) ** 2),
                "current_q1": float(q_diag[0]),
                "current_q2": float(q_diag[1]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(spec: RunSpec) -> tuple[dict, dict]:
    rl_bundle = load_pickle(spec.rl_path)
    compare_bundle = load_pickle(spec.compare_path)
    baseline_bundle = load_pickle(spec.baseline_path)
    rl_metrics = per_episode_metrics(rl_bundle)
    baseline_metrics = per_episode_metrics(baseline_bundle)
    delta = reward_delta(compare_bundle)
    first_live_episode = infer_first_live_episode(rl_bundle)
    post_live_start = max(first_live_episode - 1, 0)
    post_live_slice = slice(post_live_start, None)
    tail_slice = slice(-10, None)

    row = {
        "key": spec.key,
        "label": spec.label,
        "system": spec.system,
        "family": spec.family,
        "phase": spec.phase,
        "decision_interval": int(
            rl_bundle.get(
                "decision_interval",
                dict(rl_bundle.get("config_snapshot", {})).get("decision_interval", 1) or 1,
            )
        ),
        "steps_per_episode": int(rl_metrics["steps_per_episode"]),
        "warm_start_episodes": int(dict(rl_bundle.get("config_snapshot", {})).get("warm_start", 0) or 0),
        "first_live_episode": int(first_live_episode),
        "reward_delta_last_episode": float(delta[-1]),
        "reward_delta_tail10_mean": float(delta[tail_slice].mean()),
        "reward_delta_post_live_mean": float(delta[post_live_slice].mean()),
        "reward_delta_post_live_median": float(np.median(delta[post_live_slice])),
        "post_live_win_rate": float(np.mean(delta[post_live_slice] > 0.0)),
        "tail10_mae_phys_mean": float(np.asarray(rl_metrics["mae_mean_phys"])[tail_slice].mean()),
        "tail10_mae_phys_mean_mpc": float(np.asarray(baseline_metrics["mae_mean_phys"])[tail_slice].mean()),
        "tail10_rmse_scaled": float(np.asarray(rl_metrics["rmse_scaled"])[tail_slice].mean()),
        "tail10_rmse_scaled_mpc": float(np.asarray(baseline_metrics["rmse_scaled"])[tail_slice].mean()),
        "tail10_move_phys_mean": float(np.asarray(rl_metrics["move_mean_phys"])[tail_slice].mean()),
        "tail10_move_phys_mean_mpc": float(np.asarray(baseline_metrics["move_mean_phys"])[tail_slice].mean()),
        "tail10_out1_mae_phys": float(np.asarray(rl_metrics["mae_out_phys"])[tail_slice, 0].mean()),
        "tail10_out1_mae_phys_mpc": float(np.asarray(baseline_metrics["mae_out_phys"])[tail_slice, 0].mean()),
        "tail10_out2_mae_phys": float(np.asarray(rl_metrics["mae_out_phys"])[tail_slice, 1].mean()),
        "tail10_out2_mae_phys_mpc": float(np.asarray(baseline_metrics["mae_out_phys"])[tail_slice, 1].mean()),
        "final_episode_out1_mae_phys": float(np.asarray(rl_metrics["mae_out_phys"])[-1, 0]),
        "final_episode_out2_mae_phys": float(np.asarray(rl_metrics["mae_out_phys"])[-1, 1]),
        "final_episode_out1_mae_phys_mpc": float(np.asarray(baseline_metrics["mae_out_phys"])[-1, 0]),
        "final_episode_out2_mae_phys_mpc": float(np.asarray(baseline_metrics["mae_out_phys"])[-1, 1]),
        "rl_path": str(spec.rl_path.relative_to(REPO_ROOT)),
        "compare_path": str(spec.compare_path.relative_to(REPO_ROOT)),
        "baseline_path": str(spec.baseline_path.relative_to(REPO_ROOT)),
    }

    auxiliary_means = {}
    for key in (
        "release_guard_active_log",
        "release_clip_fraction_log",
        "action_saturation_fraction_log",
        "near_bound_fraction_log",
        "observer_recalc_event_log",
        "observer_recalc_success_log",
        "A_model_delta_ratio_log",
        "B_model_delta_ratio_log",
        "spectral_radius_log",
    ):
        reshaped = reshape_log(rl_bundle, key, rl_metrics["steps_per_episode"])
        if reshaped is None:
            continue
        collapsed = reshaped.mean(axis=1)
        if collapsed.ndim > 1:
            collapsed = collapsed.mean(axis=tuple(range(1, collapsed.ndim)))
        auxiliary_means[key] = np.asarray(collapsed, float)
        row[f"{key}_mean"] = float(np.mean(collapsed))
        row[f"{key}_p95"] = float(np.percentile(collapsed, 95))

    payload = {
        "spec": spec,
        "row": row,
        "rl_bundle": rl_bundle,
        "baseline_bundle": baseline_bundle,
        "compare_bundle": compare_bundle,
        "reward_delta": delta,
        "rl_metrics": rl_metrics,
        "baseline_metrics": baseline_metrics,
        "post_live_slice": post_live_slice,
        "auxiliary_means": auxiliary_means,
    }
    return row, payload


def build_exploratory_stats(run_payloads: dict[str, dict]) -> list[dict]:
    rows = []
    for key in (
        "distillation_matrix_latest",
        "distillation_structured_latest",
        "polymer_matrix_latest",
        "polymer_structured_latest",
    ):
        payload = run_payloads[key]
        reward_delta_arr = np.asarray(payload["reward_delta"], float)[payload["post_live_slice"]]
        rl_mae = np.asarray(payload["rl_metrics"]["mae_mean_phys"], float)[payload["post_live_slice"]]
        mpc_mae = np.asarray(payload["baseline_metrics"]["mae_mean_phys"], float)[payload["post_live_slice"]]
        rl_out2 = np.asarray(payload["rl_metrics"]["mae_out_phys"], float)[payload["post_live_slice"], 1]
        mpc_out2 = np.asarray(payload["baseline_metrics"]["mae_out_phys"], float)[payload["post_live_slice"], 1]
        rl_move = np.asarray(payload["rl_metrics"]["move_mean_phys"], float)[payload["post_live_slice"]]
        mpc_move = np.asarray(payload["baseline_metrics"]["move_mean_phys"], float)[payload["post_live_slice"]]

        reward_mean, reward_lo, reward_hi = bootstrap_mean_ci(reward_delta_arr)
        mae_mean, mae_lo, mae_hi = bootstrap_mean_ci(rl_mae - mpc_mae)
        out2_mean, out2_lo, out2_hi = bootstrap_mean_ci(rl_out2 - mpc_out2)
        move_mean, move_lo, move_hi = bootstrap_mean_ci(rl_move - mpc_move)
        wins, losses, pvalue = exploratory_sign_test(reward_delta_arr)

        row = {
            "key": key,
            "label": payload["row"]["label"],
            "reward_delta_post_live_mean": reward_mean,
            "reward_delta_post_live_ci_low": reward_lo,
            "reward_delta_post_live_ci_high": reward_hi,
            "mae_delta_post_live_mean": mae_mean,
            "mae_delta_post_live_ci_low": mae_lo,
            "mae_delta_post_live_ci_high": mae_hi,
            "out2_mae_delta_post_live_mean": out2_mean,
            "out2_mae_delta_post_live_ci_low": out2_lo,
            "out2_mae_delta_post_live_ci_high": out2_hi,
            "move_delta_post_live_mean": move_mean,
            "move_delta_post_live_ci_low": move_lo,
            "move_delta_post_live_ci_high": move_hi,
            "reward_sign_wins": wins,
            "reward_sign_losses": losses,
            "reward_sign_test_pvalue": pvalue,
        }

        reward_delta_full = np.asarray(payload["reward_delta"], float)
        out2_full = np.asarray(payload["rl_metrics"]["mae_out_phys"], float)[:, 1]
        row["corr_reward_vs_out2_mae"] = float(np.corrcoef(reward_delta_full, out2_full)[0, 1])
        aux = payload["auxiliary_means"]
        if "A_model_delta_ratio_log" in aux:
            row["corr_reward_vs_A_drift"] = float(np.corrcoef(reward_delta_full, aux["A_model_delta_ratio_log"])[0, 1])
            row["corr_out2_mae_vs_A_drift"] = float(np.corrcoef(out2_full, aux["A_model_delta_ratio_log"])[0, 1])
        if "B_model_delta_ratio_log" in aux:
            row["corr_reward_vs_B_drift"] = float(np.corrcoef(reward_delta_full, aux["B_model_delta_ratio_log"])[0, 1])
            row["corr_out2_mae_vs_B_drift"] = float(np.corrcoef(out2_full, aux["B_model_delta_ratio_log"])[0, 1])
        if "action_saturation_fraction_log" in aux:
            row["corr_reward_vs_saturation"] = float(
                np.corrcoef(reward_delta_full, aux["action_saturation_fraction_log"])[0, 1]
            )
        if "observer_recalc_event_log" in aux:
            row["corr_reward_vs_observer_refresh"] = float(
                np.corrcoef(reward_delta_full, aux["observer_recalc_event_log"])[0, 1]
            )
            row["corr_out2_mae_vs_observer_refresh"] = float(
                np.corrcoef(out2_full, aux["observer_recalc_event_log"])[0, 1]
            )
        rows.append(row)
    return rows


def plot_distillation_reward_followup(run_payloads: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharey=True)
    episodes = np.arange(1, N_EPISODES + 1)
    family_cfg = [
        (
            axes[0],
            "Scalar matrix",
            run_payloads["distillation_matrix_old"],
            run_payloads["distillation_matrix_latest"],
            "#1f77b4",
            "#d62728",
        ),
        (
            axes[1],
            "Structured matrix",
            run_payloads["distillation_structured_old"],
            run_payloads["distillation_structured_latest"],
            "#2ca02c",
            "#9467bd",
        ),
    ]
    for ax, title, old_payload, new_payload, old_color, new_color in family_cfg:
        ax.plot(episodes, old_payload["reward_delta"], lw=2.0, color=old_color, label="May 3, interval 1")
        ax.plot(episodes, new_payload["reward_delta"], lw=2.0, color=new_color, label="May 8, interval 20")
        ax.axhline(0.0, color="0.3", lw=1.0, linestyle=":")
        ax.axvline(old_payload["row"]["first_live_episode"], color="0.45", lw=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Reward delta vs MPC (RL - MPC)")
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Distillation follow-up: slowing decisions to 20 steps did not rescue the matrix family", y=1.03)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_distillation_reward_delta_old_vs_new.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distillation_latest_dashboard(run_payloads: dict[str, dict]) -> Path:
    scalar = run_payloads["distillation_matrix_latest"]
    structured = run_payloads["distillation_structured_latest"]
    baseline = scalar["baseline_bundle"]
    baseline_metrics = scalar["baseline_metrics"]
    baseline_outputs = np.asarray(baseline["y"], float)
    baseline_setpoints = np.asarray(baseline["y_sp"], float)
    steps = infer_steps_per_episode(baseline)
    time = np.arange(steps)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))

    labels = ["Scalar latest", "Structured latest", "MPC baseline"]
    reward_tail = [
        scalar["row"]["reward_delta_tail10_mean"],
        structured["row"]["reward_delta_tail10_mean"],
        0.0,
    ]
    move_tail = [
        scalar["row"]["tail10_move_phys_mean"],
        structured["row"]["tail10_move_phys_mean"],
        scalar["row"]["tail10_move_phys_mean_mpc"],
    ]
    out2_tail = [
        scalar["row"]["tail10_out2_mae_phys"],
        structured["row"]["tail10_out2_mae_phys"],
        scalar["row"]["tail10_out2_mae_phys_mpc"],
    ]
    colors = ["#d62728", "#9467bd", "0.55"]

    axes[0, 0].bar(labels, reward_tail, color=colors)
    axes[0, 0].axhline(0.0, color="0.3", lw=1.0)
    axes[0, 0].set_title("Tail-10 reward delta vs MPC")
    axes[0, 0].set_ylabel("Reward delta")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(labels, out2_tail, color=colors)
    axes[0, 1].set_title("Tail-10 output-2 MAE")
    axes[0, 1].set_ylabel("Physical MAE")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(labels, move_tail, color=colors)
    axes[1, 0].set_title("Tail-10 mean input movement")
    axes[1, 0].set_ylabel("Mean |delta u| in physical units")
    axes[1, 0].grid(axis="y", alpha=0.25)

    latest_runs = [
        ("Scalar latest", scalar, "#d62728"),
        ("Structured latest", structured, "#9467bd"),
    ]
    for label, payload, color in latest_runs:
        y = np.asarray(payload["rl_bundle"]["y"], float)
        axes[1, 1].plot(time, y[-steps:, 1], lw=1.9, color=color, label=label)
    axes[1, 1].plot(time, baseline_outputs[-steps:, 1], lw=2.0, color="0.25", label="MPC baseline")
    axes[1, 1].plot(time, baseline_setpoints[-steps:, 1], lw=1.2, color="0.55", linestyle="--", label="Setpoint")
    axes[1, 1].set_title("Final episode output 2 trajectory")
    axes[1, 1].set_xlabel("Step in final episode")
    axes[1, 1].set_ylabel("Physical output 2")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Latest distillation matrix-family outcome: slower updates changed the failure shape, but not the sign", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_distillation_latest_dashboard.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cross_system_effects(stats_rows: list[dict]) -> Path:
    keep = [
        row
        for row in stats_rows
        if row["key"]
        in {
            "distillation_matrix_latest",
            "distillation_structured_latest",
            "polymer_matrix_latest",
            "polymer_structured_latest",
        }
    ]
    order = [
        "distillation_matrix_latest",
        "distillation_structured_latest",
        "polymer_matrix_latest",
        "polymer_structured_latest",
    ]
    keep = [next(row for row in keep if row["key"] == key) for key in order]
    labels = [
        "Distill scalar",
        "Distill structured",
        "Polymer scalar",
        "Polymer structured",
    ]
    y = np.arange(len(keep))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2), sharey=True)
    metrics = [
        (
            "reward_delta_post_live_mean",
            "reward_delta_post_live_ci_low",
            "reward_delta_post_live_ci_high",
            "Post-live reward delta",
        ),
        (
            "out2_mae_delta_post_live_mean",
            "out2_mae_delta_post_live_ci_low",
            "out2_mae_delta_post_live_ci_high",
            "Post-live output-2 MAE delta",
        ),
        (
            "move_delta_post_live_mean",
            "move_delta_post_live_ci_low",
            "move_delta_post_live_ci_high",
            "Post-live move delta",
        ),
    ]
    colors = ["#d62728", "#9467bd", "#1f77b4", "#2ca02c"]

    for ax, (mean_key, lo_key, hi_key, title) in zip(axes, metrics):
        for idx, row in enumerate(keep):
            mean = row[mean_key]
            lo = row[lo_key]
            hi = row[hi_key]
            ax.errorbar(
                mean,
                y[idx],
                xerr=np.array([[mean - lo], [hi - mean]]),
                fmt="o",
                color=colors[idx],
                markersize=7,
                capsize=4,
                lw=1.8,
            )
        ax.axvline(0.0, color="0.35", lw=1.0, linestyle=":")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Cross-system latest effect direction: polymer gains, distillation losses", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_cross_system_effects.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reward_geometry(geometry_rows: list[dict]) -> Path:
    systems = ["polymer", "distillation"]
    setpoints = ["SP1", "SP2"]
    edge_ratios = []
    bonus_ratios = []
    xlabels = []
    for system in systems:
        for sp in setpoints:
            row = next(item for item in geometry_rows if item["system"] == system and item["setpoint"] == sp)
            edge_ratios.append(row["edge_ratio_out1_to_out2"])
            bonus_ratios.append(row["bonus_ratio_out1_to_out2"])
            xlabels.append(f"{system[:4].title()} {sp}")

    x = np.arange(len(xlabels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.bar(x - width / 2, edge_ratios, width, color="#4c72b0", label="Edge-slope ratio out1/out2")
    ax.bar(x + width / 2, bonus_ratios, width, color="#dd8452", label="Bonus ratio out1/out2")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Ratio")
    ax.set_title("Reward geometry asymmetry is much stronger in distillation than in polymer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_cross_system_reward_geometry.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_adaptation_diagnostics(run_payloads: dict[str, dict], stats_rows: list[dict]) -> Path:
    latest_keys = [
        "distillation_matrix_latest",
        "distillation_structured_latest",
        "polymer_matrix_latest",
        "polymer_structured_latest",
    ]
    colors = {
        "distillation_matrix_latest": "#d62728",
        "distillation_structured_latest": "#9467bd",
        "polymer_matrix_latest": "#1f77b4",
        "polymer_structured_latest": "#2ca02c",
    }
    labels = {
        "distillation_matrix_latest": "Distill scalar",
        "distillation_structured_latest": "Distill structured",
        "polymer_matrix_latest": "Polymer scalar",
        "polymer_structured_latest": "Polymer structured",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    for key in latest_keys:
        payload = run_payloads[key]
        reward = np.asarray(payload["reward_delta"], float)
        out2 = np.asarray(payload["rl_metrics"]["mae_out_phys"], float)[:, 1]
        b_drift = payload["auxiliary_means"].get("B_model_delta_ratio_log")
        if b_drift is not None:
            axes[0].scatter(
                b_drift,
                out2,
                s=18,
                alpha=0.45,
                color=colors[key],
                label=labels[key],
            )
        observer_refresh = payload["auxiliary_means"].get("observer_recalc_event_log")
        if observer_refresh is not None and key.startswith("distillation"):
            axes[1].scatter(
                observer_refresh,
                out2,
                s=20,
                alpha=0.55,
                color=colors[key],
                label=labels[key],
            )

    axes[0].set_title("Episode mean B-drift vs output-2 MAE")
    axes[0].set_xlabel("Episode mean B-model delta ratio")
    axes[0].set_ylabel("Episode mean output-2 MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title("Distillation observer refresh rate vs output-2 MAE")
    axes[1].set_xlabel("Episode mean observer refresh event rate")
    axes[1].set_ylabel("Episode mean output-2 MAE")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Adaptation diagnostics: in distillation, more model motion is not translating into better temperature control", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_adaptation_diagnostics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    run_payloads: dict[str, dict] = {}
    for spec in RUNS:
        row, payload = summarize_run(spec)
        summary_rows.append(row)
        run_payloads[spec.key] = payload

    geometry_rows = compute_reward_geometry("polymer") + compute_reward_geometry("distillation")
    stats_rows = build_exploratory_stats(run_payloads)

    write_csv(DISTILLATION_SUMMARY_CSV, [row for row in summary_rows if row["system"] == "distillation"])
    write_csv(CROSS_SYSTEM_SUMMARY_CSV, [row for row in summary_rows if row["phase"] == "latest"])
    write_csv(STATS_CSV, stats_rows)
    write_csv(REWARD_GEOMETRY_CSV, geometry_rows)

    figures = {
        "distillation_reward_followup": str(plot_distillation_reward_followup(run_payloads).relative_to(REPO_ROOT)),
        "distillation_latest_dashboard": str(plot_distillation_latest_dashboard(run_payloads).relative_to(REPO_ROOT)),
        "cross_system_effects": str(plot_cross_system_effects(stats_rows).relative_to(REPO_ROOT)),
        "cross_system_reward_geometry": str(plot_reward_geometry(geometry_rows).relative_to(REPO_ROOT)),
        "adaptation_diagnostics": str(plot_adaptation_diagnostics(run_payloads, stats_rows).relative_to(REPO_ROOT)),
    }

    summary_json = {
        "figures": figures,
        "distillation_latest": {
            row["key"]: row
            for row in summary_rows
            if row["key"] in {"distillation_matrix_latest", "distillation_structured_latest"}
        },
        "cross_system_latest": {row["key"]: row for row in summary_rows if row["phase"] == "latest"},
        "exploratory_stats": {row["key"]: row for row in stats_rows},
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)

    print(f"Wrote figure directory: {FIG_DIR}")
    print(f"Wrote summary CSV: {DISTILLATION_SUMMARY_CSV}")
    print(f"Wrote summary CSV: {CROSS_SYSTEM_SUMMARY_CSV}")
    print(f"Wrote stats CSV: {STATS_CSV}")
    print(f"Wrote reward geometry CSV: {REWARD_GEOMETRY_CSV}")
    print(f"Wrote summary JSON: {SUMMARY_JSON}")
    for label, rel_path in figures.items():
        print(f"Wrote figure [{label}]: {rel_path}")


if __name__ == "__main__":
    main()
