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
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "matrix_multiplier_step3d_20260430"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"

from utils.plotting_core import normalize_external_bundle, normalize_result_bundle, ysp_scaled_dev_to_phys

RUNS = {
    "matrix_step3d_latest": {
        "family": "Scalar matrix",
        "variant": "Step 3D latest",
        "style": "step3d",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260430_160646" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260430_160657" / "input_data.pkl",
    },
    "matrix_step4g": {
        "family": "Scalar matrix",
        "variant": "Step 4G reference",
        "style": "step4g",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260428_162645" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260428_162700" / "input_data.pkl",
    },
    "structured_step3d_latest": {
        "family": "Structured matrix",
        "variant": "Step 3D latest",
        "style": "step3d",
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
    "step3d": "tab:red",
    "step4g": "tab:green",
    "mpc": "0.35",
}

REASON_LABELS = {
    1: "accepted",
    2: "candidate_solve_failed",
    3: "rejected_nominal_safety",
    4: "rejected_candidate_usefulness",
    5: "rejected_gain_drift",
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


def policy_action_raw_log(bundle: dict) -> np.ndarray:
    if bundle.get("policy_action_raw_log") is not None:
        return np.asarray(bundle["policy_action_raw_log"], float)
    if bundle.get("candidate_action_raw_log") is not None:
        return np.asarray(bundle["candidate_action_raw_log"], float)
    if bundle.get("candidate_action_log") is not None:
        return np.asarray(bundle["candidate_action_log"], float)
    return np.zeros((0, 0), float)


def build_run_payload(spec: dict) -> dict:
    rl = normalize_result_bundle(load_pickle(spec["rl_path"]))
    compare = load_pickle(spec["compare_path"])
    baseline = normalize_external_bundle(load_pickle(BASELINE_PATH), rl)
    return {
        **spec,
        "rl": rl,
        "compare": compare,
        "baseline": baseline,
    }


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
    u_rl = np.asarray(rl["u_step_full"], float)
    u_mpc = np.asarray(baseline["u_step_full"], float)

    err_rl = np.abs(y_rl - y_sp_phys)
    err_mpc = np.abs(y_mpc - y_sp_phys)
    delta_reward = np.asarray(compare["avg_rewards_rl"], float) - np.asarray(compare["avg_rewards_mpc"], float)

    executed = executed_multiplier_log(rl)
    candidate = candidate_multiplier_log(rl)
    executed_dist = np.linalg.norm(executed - 1.0, axis=1)
    candidate_dist = np.linalg.norm(candidate - 1.0, axis=1)

    executed_raw = np.asarray(rl["executed_action_raw_log"], float)
    policy_raw = policy_action_raw_log(rl)
    policy_executed_gap = (
        np.linalg.norm(policy_raw - executed_raw, axis=1)
        if policy_raw.size and executed_raw.size and policy_raw.shape == executed_raw.shape
        else np.asarray([], float)
    )

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
        "reward_delta_last_episode": float(delta_reward[-1]),
        "final_phys_mae_mean": float(np.mean(err_rl[final_slice])),
        "final_phys_mae_out1": float(np.mean(err_rl[final_slice][:, 0])),
        "final_phys_mae_out2": float(np.mean(err_rl[final_slice][:, 1])),
        "tail_phys_mae_mean": float(np.mean(err_rl[tail_slice])),
        "tail_phys_mae_out1": float(np.mean(err_rl[tail_slice][:, 0])),
        "tail_phys_mae_out2": float(np.mean(err_rl[tail_slice][:, 1])),
        "mpc_final_phys_mae_mean": float(np.mean(err_mpc[final_slice])),
        "mpc_tail_phys_mae_mean": float(np.mean(err_mpc[tail_slice])),
        "final_input_move_mean": float(np.mean(np.linalg.norm(np.diff(u_rl[final_slice], axis=0), axis=1))),
        "mpc_final_input_move_mean": float(np.mean(np.linalg.norm(np.diff(u_mpc[final_slice], axis=0), axis=1))),
        "mean_abs_output_diff_vs_mpc_out1": float(np.mean(np.abs(y_rl - y_mpc)[:, 0])),
        "mean_abs_output_diff_vs_mpc_out2": float(np.mean(np.abs(y_rl - y_mpc)[:, 1])),
        "max_abs_output_diff_vs_mpc_out1": float(np.max(np.abs(y_rl - y_mpc)[:, 0])),
        "max_abs_output_diff_vs_mpc_out2": float(np.max(np.abs(y_rl - y_mpc)[:, 1])),
        "mean_abs_input_diff_vs_mpc_u1": float(np.mean(np.abs(u_rl - u_mpc)[:, 0])),
        "mean_abs_input_diff_vs_mpc_u2": float(np.mean(np.abs(u_rl - u_mpc)[:, 1])),
        "max_abs_input_diff_vs_mpc_u1": float(np.max(np.abs(u_rl - u_mpc)[:, 0])),
        "max_abs_input_diff_vs_mpc_u2": float(np.max(np.abs(u_rl - u_mpc)[:, 1])),
        "candidate_multiplier_distance_mean": float(np.mean(candidate_dist)),
        "candidate_multiplier_distance_final": float(np.mean(candidate_dist[final_slice])),
        "executed_multiplier_distance_mean": float(np.mean(executed_dist)),
        "executed_multiplier_distance_final": float(np.mean(executed_dist[final_slice])),
        "policy_executed_raw_gap_mean": safe_mean(policy_executed_gap),
        "policy_executed_raw_gap_final": safe_mean(policy_executed_gap[final_slice]) if policy_executed_gap.size else float("nan"),
        "behavioral_cloning_enabled": bool(rl.get("behavioral_cloning_enabled", False)),
        "release_guard_enabled": bool(rl.get("release_guard_enabled", False)),
        "mpc_usefulness_gate_enabled": bool(rl.get("mpc_usefulness_gate_enabled", False)),
    }

    if out["mpc_usefulness_gate_enabled"]:
        gate = {
            "gate_pass_full": safe_fraction(rl["mpc_usefulness_gate_gate_pass_log"]),
            "gate_pass_final": safe_fraction(np.asarray(rl["mpc_usefulness_gate_gate_pass_log"], float)[final_slice]),
            "safe_pass_full": safe_fraction(rl["mpc_usefulness_gate_safe_pass_log"]),
            "safe_pass_final": safe_fraction(np.asarray(rl["mpc_usefulness_gate_safe_pass_log"], float)[final_slice]),
            "benefit_pass_full": safe_fraction(rl["mpc_usefulness_gate_benefit_pass_log"]),
            "benefit_pass_final": safe_fraction(np.asarray(rl["mpc_usefulness_gate_benefit_pass_log"], float)[final_slice]),
            "gain_pass_full": safe_fraction(rl["mpc_usefulness_gate_gain_pass_log"]),
            "gain_pass_final": safe_fraction(np.asarray(rl["mpc_usefulness_gate_gain_pass_log"], float)[final_slice]),
            "nominal_penalty_mean": safe_mean(rl["mpc_usefulness_gate_nominal_penalty_log"]),
            "nominal_penalty_final": safe_mean(np.asarray(rl["mpc_usefulness_gate_nominal_penalty_log"], float)[final_slice]),
            "safe_threshold_mean": safe_mean(rl["mpc_usefulness_gate_safe_threshold_log"]),
            "safe_threshold_final": safe_mean(np.asarray(rl["mpc_usefulness_gate_safe_threshold_log"], float)[final_slice]),
            "candidate_advantage_mean": safe_mean(rl["mpc_usefulness_gate_candidate_advantage_log"]),
            "candidate_advantage_final": safe_mean(np.asarray(rl["mpc_usefulness_gate_candidate_advantage_log"], float)[final_slice]),
            "benefit_threshold_mean": safe_mean(rl["mpc_usefulness_gate_benefit_threshold_log"]),
            "benefit_threshold_final": safe_mean(np.asarray(rl["mpc_usefulness_gate_benefit_threshold_log"], float)[final_slice]),
            "gain_drift_mean": safe_mean(rl["mpc_usefulness_gate_gain_drift_log"]),
            "gain_drift_final": safe_mean(np.asarray(rl["mpc_usefulness_gate_gain_drift_log"], float)[final_slice]),
            "gain_threshold_mean": safe_mean(rl["mpc_usefulness_gate_gain_drift_threshold_log"]),
            "gain_threshold_final": safe_mean(np.asarray(rl["mpc_usefulness_gate_gain_drift_threshold_log"], float)[final_slice]),
        }
        out.update(gate)

    return out


def summarize_reason_breakdown(spec: dict) -> list[dict]:
    rl = spec["rl"]
    if not bool(rl.get("mpc_usefulness_gate_enabled", False)):
        return []

    n_ep = int(rl["nFE"] // rl["time_in_sub_episodes"])
    final_slice = episode_slice(rl, n_ep - 1)
    rows = []
    for window_name, data_slice in {
        "full_run": slice(0, rl["nFE"]),
        "final_test": final_slice,
    }.items():
        reason_codes = np.asarray(rl["mpc_usefulness_gate_reason_code_log"], int)[data_slice]
        total = max(1, int(reason_codes.size))
        unique, counts = np.unique(reason_codes, return_counts=True)
        count_map = {int(code): int(count) for code, count in zip(unique, counts, strict=True)}
        row = {
            "family": spec["family"],
            "variant": spec["variant"],
            "window": window_name,
        }
        for code, label in REASON_LABELS.items():
            row[f"{label}_count"] = int(count_map.get(code, 0))
            row[f"{label}_fraction"] = float(count_map.get(code, 0) / total)
        rows.append(row)
    return rows


def plot_reward_delta(run_payloads: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, family in zip(axes, ["Scalar matrix", "Structured matrix"], strict=True):
        family_payloads = [payload for payload in run_payloads.values() if payload["family"] == family]
        for payload in family_payloads:
            delta = np.asarray(payload["compare"]["avg_rewards_rl"], float) - np.asarray(payload["compare"]["avg_rewards_mpc"], float)
            x = np.arange(1, delta.size + 1)
            ax.plot(
                x,
                delta,
                linewidth=2.0,
                color=STYLE_COLORS[payload["style"]],
                label=payload["variant"],
            )
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
    for row_idx, family in enumerate(families):
        subset = summary_df[summary_df["family"] == family].copy()
        subset = subset.set_index("style")

        ax = axes[row_idx, 0]
        labels = ["MPC", "Step 4G", "Step 3D"]
        values = [
            float(subset.iloc[0]["mpc_final_phys_mae_mean"]),
            float(subset.loc["step4g", "final_phys_mae_mean"]),
            float(subset.loc["step3d", "final_phys_mae_mean"]),
        ]
        colors = [STYLE_COLORS["mpc"], STYLE_COLORS["step4g"], STYLE_COLORS["step3d"]]
        ax.bar(labels, values, color=colors)
        ax.set_title(f"{family}: final-test physical MAE")
        ax.set_ylabel("Mean absolute error")
        ax.grid(True, axis="y", alpha=0.25)

        ax = axes[row_idx, 1]
        metric_labels = [
            "Step 4G cand",
            "Step 4G exec",
            "Step 3D cand",
            "Step 3D exec",
        ]
        metric_values = [
            float(subset.loc["step4g", "candidate_multiplier_distance_final"]),
            float(subset.loc["step4g", "executed_multiplier_distance_final"]),
            float(subset.loc["step3d", "candidate_multiplier_distance_final"]),
            float(subset.loc["step3d", "executed_multiplier_distance_final"]),
        ]
        metric_colors = [
            STYLE_COLORS["step4g"],
            STYLE_COLORS["step4g"],
            STYLE_COLORS["step3d"],
            STYLE_COLORS["step3d"],
        ]
        ax.bar(metric_labels, metric_values, color=metric_colors)
        ax.set_title(f"{family}: final-test multiplier distance from nominal")
        ax.set_ylabel("L2 distance")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_criteria(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    families = ["Scalar matrix", "Structured matrix"]
    for row_idx, family in enumerate(families):
        row = summary_df[(summary_df["family"] == family) & (summary_df["style"] == "step3d")].iloc[0]

        ax = axes[row_idx, 0]
        x = np.arange(3)
        ax.bar(x - 0.18, [row["nominal_penalty_mean"], row["candidate_advantage_mean"], row["gain_drift_mean"]], width=0.36, color="tab:red", label="Candidate statistic")
        ax.bar(x + 0.18, [row["safe_threshold_mean"], row["benefit_threshold_mean"], row["gain_threshold_mean"]], width=0.36, color="0.4", label="Gate threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Safety", "Usefulness", "Gain drift"])
        ax.set_title(f"{family}: full-run gate statistics vs thresholds")
        ax.grid(True, axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(frameon=False)

        ax = axes[row_idx, 1]
        x = np.arange(4)
        full_vals = [row["safe_pass_full"], row["benefit_pass_full"], row["gain_pass_full"], row["gate_pass_full"]]
        final_vals = [row["safe_pass_final"], row["benefit_pass_final"], row["gain_pass_final"], row["gate_pass_final"]]
        ax.bar(x - 0.18, full_vals, width=0.36, color="tab:blue", label="Full run")
        ax.bar(x + 0.18, final_vals, width=0.36, color="tab:orange", label="Final test")
        ax.set_xticks(x)
        ax.set_xticklabels(["Safe", "Useful", "Gain", "Gate"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{family}: gate pass rates")
        ax.grid(True, axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(frameon=False)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_payloads = {name: build_run_payload(spec) for name, spec in RUNS.items()}

    summary_rows = [summarize_run(payload) for payload in run_payloads.values()]
    reason_rows: list[dict] = []
    for payload in run_payloads.values():
        reason_rows.extend(summarize_reason_breakdown(payload))

    summary_df = pd.DataFrame(summary_rows)
    reason_df = pd.DataFrame(reason_rows)

    summary_df.to_csv(OUTPUT_DIR / "polymer_step3d_latest_summary.csv", index=False)
    reason_df.to_csv(OUTPUT_DIR / "polymer_step3d_gate_reason_breakdown.csv", index=False)

    plot_reward_delta(run_payloads, OUTPUT_DIR / "polymer_step3d_reward_delta_vs_step4g.png")
    plot_mae_and_authority(summary_df, OUTPUT_DIR / "polymer_step3d_mae_and_authority.png")
    plot_gate_criteria(summary_df, OUTPUT_DIR / "polymer_step3d_gate_criteria.png")


if __name__ == "__main__":
    main()
