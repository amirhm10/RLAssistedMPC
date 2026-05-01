from __future__ import annotations

import pickle
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "matrix_multiplier_step3d_relaxed_gate_20260501"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"

from utils.multiplier_release_schedule import PHASE_FULL, PHASE_NOMINAL, PHASE_PROTECTED, PHASE_RAMP
from utils.plotting_core import normalize_external_bundle, normalize_result_bundle, ysp_scaled_dev_to_phys


RUNS = {
    "scalar_relaxed": {
        "family": "Scalar matrix",
        "variant": "Step 3D relaxed gate",
        "style": "relaxed",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260501_124838" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260501_124847" / "input_data.pkl",
    },
    "scalar_hard": {
        "family": "Scalar matrix",
        "variant": "Step 3D hard gate",
        "style": "hard",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260430_160646" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260430_160657" / "input_data.pkl",
    },
    "scalar_step4g": {
        "family": "Scalar matrix",
        "variant": "Step 4G reference",
        "style": "step4g",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260428_162645" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260428_162700" / "input_data.pkl",
    },
    "structured_relaxed": {
        "family": "Structured matrix",
        "variant": "Step 3D relaxed gate",
        "style": "relaxed",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260501_125051" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260501_125104" / "input_data.pkl",
    },
    "structured_hard": {
        "family": "Structured matrix",
        "variant": "Step 3D hard gate",
        "style": "hard",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260430_134358" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260430_134411" / "input_data.pkl",
    },
    "structured_step4g": {
        "family": "Structured matrix",
        "variant": "Step 4G reference",
        "style": "step4g",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260428_162948" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260428_163005" / "input_data.pkl",
    },
}


STYLE_COLORS = {
    "hard": "tab:red",
    "relaxed": "tab:orange",
    "step4g": "tab:green",
    "mpc": "0.35",
}


PHASE_LABELS = {
    PHASE_NOMINAL: "nominal",
    PHASE_PROTECTED: "protected",
    PHASE_RAMP: "ramp",
    PHASE_FULL: "full",
}


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, float)
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def episode_slice(bundle: dict, episode_index: int) -> slice:
    steps = int(bundle["time_in_sub_episodes"])
    return slice(episode_index * steps, (episode_index + 1) * steps)


def executed_multiplier_log(bundle: dict) -> np.ndarray:
    if bundle.get("executed_multiplier_log") is not None:
        return np.asarray(bundle["executed_multiplier_log"], float)
    if bundle.get("effective_multiplier_log") is not None:
        return np.asarray(bundle["effective_multiplier_log"], float)
    return np.asarray(bundle["candidate_multiplier_log"], float)


def candidate_multiplier_log(bundle: dict) -> np.ndarray:
    if bundle.get("candidate_multiplier_log") is not None:
        return np.asarray(bundle["candidate_multiplier_log"], float)
    if bundle.get("theta_candidate_log") is not None:
        return np.asarray(bundle["theta_candidate_log"], float)
    return executed_multiplier_log(bundle)


def build_run_payload(spec: dict) -> dict:
    rl = normalize_result_bundle(load_pickle(spec["rl_path"]))
    compare = load_pickle(spec["compare_path"])
    baseline = normalize_external_bundle(load_pickle(BASELINE_PATH), rl)
    return {**spec, "rl": rl, "compare": compare, "baseline": baseline}


def summarize_run(spec: dict) -> dict:
    rl = spec["rl"]
    baseline = spec["baseline"]
    compare = spec["compare"]

    n_ep = int(rl["nFE"] // rl["time_in_sub_episodes"])
    final_slice = episode_slice(rl, n_ep - 1)
    tail_slice = slice(100 * rl["time_in_sub_episodes"], n_ep * rl["time_in_sub_episodes"])

    y_sp_phys = ysp_scaled_dev_to_phys(
        rl["y_sp"],
        rl["steady_states"],
        rl["data_min"],
        rl["data_max"],
        rl["n_inputs"],
    )
    y_rl = np.asarray(rl["y_line_full"][1:], float)
    y_mpc = np.asarray(baseline["y_line_full"][1:], float)
    err_rl = np.abs(y_rl - y_sp_phys)
    err_mpc = np.abs(y_mpc - y_sp_phys)
    delta_reward = np.asarray(compare["avg_rewards_rl"], float) - np.asarray(compare["avg_rewards_mpc"], float)

    executed = executed_multiplier_log(rl)
    candidate = candidate_multiplier_log(rl)
    executed_dist = np.linalg.norm(executed - 1.0, axis=1)
    candidate_dist = np.linalg.norm(candidate - 1.0, axis=1)

    out = {
        "family": spec["family"],
        "variant": spec["variant"],
        "style": spec["style"],
        "rl_path": spec["rl_path"].relative_to(REPO_ROOT).as_posix(),
        "compare_path": spec["compare_path"].relative_to(REPO_ROOT).as_posix(),
        "reward_delta_mean_1_200": float(np.mean(delta_reward)),
        "reward_delta_mean_11_20": float(np.mean(delta_reward[10:20])),
        "reward_delta_mean_21_40": float(np.mean(delta_reward[20:40])),
        "reward_delta_mean_41_100": float(np.mean(delta_reward[40:100])),
        "reward_delta_mean_101_200": float(np.mean(delta_reward[100:200])),
        "reward_delta_mean_last10": float(np.mean(delta_reward[-10:])),
        "final_phys_mae_mean": float(np.mean(err_rl[final_slice])),
        "final_phys_mae_out1": float(np.mean(err_rl[final_slice][:, 0])),
        "final_phys_mae_out2": float(np.mean(err_rl[final_slice][:, 1])),
        "mpc_final_phys_mae_mean": float(np.mean(err_mpc[final_slice])),
        "tail_phys_mae_mean": float(np.mean(err_rl[tail_slice])),
        "candidate_multiplier_distance_final": float(np.mean(candidate_dist[final_slice])),
        "executed_multiplier_distance_final": float(np.mean(executed_dist[final_slice])),
        "gate_pass_full": safe_fraction(rl.get("mpc_usefulness_gate_gate_pass_log", [])),
        "gate_pass_final": safe_fraction(np.asarray(rl.get("mpc_usefulness_gate_gate_pass_log", []), float)[final_slice]),
        "safe_pass_full": safe_fraction(rl.get("mpc_usefulness_gate_safe_pass_log", [])),
        "benefit_pass_full": safe_fraction(rl.get("mpc_usefulness_gate_benefit_pass_log", [])),
        "gain_pass_full": safe_fraction(rl.get("mpc_usefulness_gate_gain_pass_log", [])),
        "safe_pass_final": safe_fraction(np.asarray(rl.get("mpc_usefulness_gate_safe_pass_log", []), float)[final_slice]),
        "benefit_pass_final": safe_fraction(np.asarray(rl.get("mpc_usefulness_gate_benefit_pass_log", []), float)[final_slice]),
        "gain_pass_final": safe_fraction(np.asarray(rl.get("mpc_usefulness_gate_gain_pass_log", []), float)[final_slice]),
        "nominal_penalty_mean": safe_mean(rl.get("mpc_usefulness_gate_nominal_penalty_log", [])),
        "safe_threshold_mean": safe_mean(rl.get("mpc_usefulness_gate_safe_threshold_log", [])),
        "candidate_advantage_mean": safe_mean(rl.get("mpc_usefulness_gate_candidate_advantage_log", [])),
        "benefit_threshold_mean": safe_mean(rl.get("mpc_usefulness_gate_benefit_threshold_log", [])),
        "gain_drift_mean": safe_mean(rl.get("mpc_usefulness_gate_gain_drift_log", [])),
        "gain_threshold_mean": safe_mean(rl.get("mpc_usefulness_gate_gain_drift_threshold_log", [])),
        "gain_drift_final": safe_mean(np.asarray(rl.get("mpc_usefulness_gate_gain_drift_log", []), float)[final_slice]),
        "gain_threshold_final": safe_mean(np.asarray(rl.get("mpc_usefulness_gate_gain_drift_threshold_log", []), float)[final_slice]),
    }
    return out


def summarize_reason_breakdown(spec: dict) -> list[dict]:
    rl = spec["rl"]
    reasons = np.asarray(rl.get("mpc_usefulness_gate_reason_code_log", []), int)
    if reasons.size == 0:
        return []

    rows = []
    total = int(reasons.size)
    unique, counts = np.unique(reasons, return_counts=True)
    count_map = {int(code): int(count) for code, count in zip(unique, counts, strict=True)}
    rows.append(
        {
            "family": spec["family"],
            "variant": spec["variant"],
            "accepted_fraction": float(count_map.get(1, 0) / total),
            "reject_safe_fraction": float(count_map.get(3, 0) / total),
            "reject_useful_fraction": float(count_map.get(4, 0) / total),
            "reject_gain_fraction": float(count_map.get(5, 0) / total),
        }
    )
    return rows


def summarize_phase_acceptance(spec: dict) -> list[dict]:
    rl = spec["rl"]
    if "mpc_usefulness_gate_gate_pass_log" not in rl:
        return []
    phases = np.asarray(rl["release_phase_log"], int)
    gate = np.asarray(rl["mpc_usefulness_gate_gate_pass_log"], float)
    rows = []
    for phase_code, phase_label in PHASE_LABELS.items():
        mask = phases == int(phase_code)
        if not np.any(mask):
            continue
        rows.append(
            {
                "family": spec["family"],
                "variant": spec["variant"],
                "style": spec["style"],
                "phase_code": int(phase_code),
                "phase_label": phase_label,
                "steps": int(np.sum(mask)),
                "gate_pass_fraction": float(np.mean(gate[mask])),
            }
        )
    return rows


def summarize_window_acceptance(spec: dict) -> list[dict]:
    rl = spec["rl"]
    if "mpc_usefulness_gate_gate_pass_log" not in rl:
        return []
    gate = np.asarray(rl["mpc_usefulness_gate_gate_pass_log"], float)
    executed = executed_multiplier_log(rl)
    executed_dist = np.linalg.norm(executed - 1.0, axis=1)
    steps = int(rl["time_in_sub_episodes"])
    rows = []
    for start_ep, end_ep in [(11, 20), (21, 40), (41, 100), (101, 200)]:
        data_slice = slice((start_ep - 1) * steps, end_ep * steps)
        rows.append(
            {
                "family": spec["family"],
                "variant": spec["variant"],
                "style": spec["style"],
                "window": f"{start_ep}-{end_ep}",
                "gate_pass_fraction": float(np.mean(gate[data_slice])),
                "executed_distance_mean": float(np.mean(executed_dist[data_slice])),
            }
        )
    return rows


def plot_reward_delta(run_payloads: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, family in zip(axes, ["Scalar matrix", "Structured matrix"], strict=True):
        family_payloads = [payload for payload in run_payloads.values() if payload["family"] == family]
        for payload in family_payloads:
            delta = np.asarray(payload["compare"]["avg_rewards_rl"], float) - np.asarray(payload["compare"]["avg_rewards_mpc"], float)
            x = np.arange(1, delta.size + 1)
            ax.plot(x, delta, linewidth=2.0, color=STYLE_COLORS[payload["style"]], label=payload["variant"])
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        for boundary in [10, 20, 40, 100]:
            ax.axvline(boundary, color="0.8", linestyle="--", linewidth=1.0)
        ax.set_title(f"{family}: reward delta vs MPC")
        ax.set_xlabel("Episode")
        ax.set_ylabel("RL - MPC average reward")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mae_and_authority(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    families = ["Scalar matrix", "Structured matrix"]
    order = ["step4g", "hard", "relaxed"]
    labels = {"step4g": "Step 4G", "hard": "Hard gate", "relaxed": "Relaxed gate"}

    for row_idx, family in enumerate(families):
        subset = summary_df[summary_df["family"] == family].set_index("style")
        ax = axes[row_idx, 0]
        bar_labels = ["MPC"] + [labels[key] for key in order]
        values = [float(subset.iloc[0]["mpc_final_phys_mae_mean"])] + [float(subset.loc[key, "final_phys_mae_mean"]) for key in order]
        colors = [STYLE_COLORS["mpc"]] + [STYLE_COLORS[key] for key in order]
        ax.bar(bar_labels, values, color=colors)
        ax.set_title(f"{family}: final-test physical MAE")
        ax.set_ylabel("Mean absolute error")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)

        ax = axes[row_idx, 1]
        metric_labels = ["Hard exec", "Relaxed exec", "Step 4G exec"]
        metric_values = [
            float(subset.loc["hard", "executed_multiplier_distance_final"]),
            float(subset.loc["relaxed", "executed_multiplier_distance_final"]),
            float(subset.loc["step4g", "executed_multiplier_distance_final"]),
        ]
        metric_colors = [STYLE_COLORS["hard"], STYLE_COLORS["relaxed"], STYLE_COLORS["step4g"]]
        ax.bar(metric_labels, metric_values, color=metric_colors)
        ax.set_title(f"{family}: final-test executed multiplier distance")
        ax.set_ylabel("L2 distance")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_diagnostics(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    families = ["Scalar matrix", "Structured matrix"]
    compare_styles = ["hard", "relaxed"]
    labels = {"hard": "Hard gate", "relaxed": "Relaxed gate"}

    for row_idx, family in enumerate(families):
        subset = summary_df[summary_df["family"] == family].set_index("style")

        ax = axes[row_idx, 0]
        x = np.arange(3)
        width = 0.22
        for idx, style in enumerate(compare_styles):
            row = subset.loc[style]
            ax.bar(
                x + (idx - 0.5) * width,
                [row["nominal_penalty_mean"], row["candidate_advantage_mean"], row["gain_drift_mean"]],
                width=width,
                color=STYLE_COLORS[style],
                label=f"{labels[style]} statistic",
            )
        threshold_row = subset.loc["relaxed"]
        ax.plot(x, [threshold_row["safe_threshold_mean"], threshold_row["benefit_threshold_mean"], threshold_row["gain_threshold_mean"]], color="0.25", marker="o", linewidth=2.0, label="Relaxed thresholds")
        hard_row = subset.loc["hard"]
        ax.plot(x, [hard_row["safe_threshold_mean"], hard_row["benefit_threshold_mean"], hard_row["gain_threshold_mean"]], color="0.55", marker="s", linewidth=1.5, linestyle="--", label="Hard thresholds")
        ax.set_xticks(x)
        ax.set_xticklabels(["Safety", "Usefulness", "Gain drift"])
        ax.set_title(f"{family}: gate statistics vs thresholds")
        ax.grid(True, axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[row_idx, 1]
        x = np.arange(4)
        width = 0.32
        hard_vals = [subset.loc["hard", "safe_pass_final"], subset.loc["hard", "benefit_pass_final"], subset.loc["hard", "gain_pass_final"], subset.loc["hard", "gate_pass_final"]]
        relaxed_vals = [subset.loc["relaxed", "safe_pass_final"], subset.loc["relaxed", "benefit_pass_final"], subset.loc["relaxed", "gain_pass_final"], subset.loc["relaxed", "gate_pass_final"]]
        ax.bar(x - width / 2, hard_vals, width=width, color=STYLE_COLORS["hard"], label="Hard gate")
        ax.bar(x + width / 2, relaxed_vals, width=width, color=STYLE_COLORS["relaxed"], label="Relaxed gate")
        ax.set_xticks(x)
        ax.set_xticklabels(["Safe", "Useful", "Gain", "Gate"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{family}: final-test pass fractions")
        ax.grid(True, axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(frameon=False)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_and_window_diagnostics(phase_df: pd.DataFrame, window_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    families = ["Scalar matrix", "Structured matrix"]
    phase_order = ["nominal", "protected", "ramp", "full"]

    for row_idx, family in enumerate(families):
        ax = axes[row_idx, 0]
        subset = phase_df[(phase_df["family"] == family) & (phase_df["style"].isin(["hard", "relaxed"]))].copy()
        x = np.arange(len(phase_order))
        width = 0.32
        for idx, style in enumerate(["hard", "relaxed"]):
            style_rows = subset[subset["style"] == style].set_index("phase_label")
            vals = [float(style_rows.loc[label, "gate_pass_fraction"]) if label in style_rows.index else 0.0 for label in phase_order]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, color=STYLE_COLORS[style], label="Hard gate" if style == "hard" else "Relaxed gate")
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order)
        ax.set_title(f"{family}: gate acceptance by release phase")
        ax.set_ylabel("Accepted-step fraction")
        ax.grid(True, axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(frameon=False)

        ax = axes[row_idx, 1]
        subset = window_df[(window_df["family"] == family) & (window_df["style"].isin(["hard", "relaxed"]))].copy()
        x = np.arange(4)
        width = 0.32
        for idx, style in enumerate(["hard", "relaxed"]):
            style_rows = subset[subset["style"] == style].set_index("window")
            vals = [float(style_rows.loc[label, "executed_distance_mean"]) for label in ["11-20", "21-40", "41-100", "101-200"]]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, color=STYLE_COLORS[style], label="Hard gate" if style == "hard" else "Relaxed gate")
        ax.set_xticks(x)
        ax.set_xticklabels(["11-20", "21-40", "41-100", "101-200"])
        ax.set_title(f"{family}: executed multiplier distance by window")
        ax.set_ylabel("Mean executed L2 distance")
        ax.grid(True, axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(frameon=False)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_payloads = {name: build_run_payload(spec) for name, spec in RUNS.items()}

    summary_df = pd.DataFrame([summarize_run(payload) for payload in run_payloads.values()])
    reason_df = pd.DataFrame([row for payload in run_payloads.values() for row in summarize_reason_breakdown(payload)])
    phase_df = pd.DataFrame([row for payload in run_payloads.values() for row in summarize_phase_acceptance(payload)])
    window_df = pd.DataFrame([row for payload in run_payloads.values() for row in summarize_window_acceptance(payload)])

    summary_df.to_csv(OUTPUT_DIR / "polymer_step3d_relaxed_gate_summary.csv", index=False)
    reason_df.to_csv(OUTPUT_DIR / "polymer_step3d_relaxed_gate_reason_breakdown.csv", index=False)
    phase_df.to_csv(OUTPUT_DIR / "polymer_step3d_relaxed_gate_phase_acceptance.csv", index=False)
    window_df.to_csv(OUTPUT_DIR / "polymer_step3d_relaxed_gate_window_diagnostics.csv", index=False)

    plot_reward_delta(run_payloads, OUTPUT_DIR / "polymer_step3d_relaxed_gate_reward_delta.png")
    plot_mae_and_authority(summary_df, OUTPUT_DIR / "polymer_step3d_relaxed_gate_mae_and_authority.png")
    plot_gate_diagnostics(summary_df, OUTPUT_DIR / "polymer_step3d_relaxed_gate_criteria.png")
    plot_phase_and_window_diagnostics(phase_df, window_df, OUTPUT_DIR / "polymer_step3d_relaxed_gate_phase_and_window.png")


if __name__ == "__main__":
    main()
