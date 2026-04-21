from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "report" / "polymer_wide_range_matrix_structured"
FIG_ROOT = REPORT_ROOT / "figures"
DATA_ROOT = REPORT_ROOT / "data"

BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
POLYMER_SYS_DICT = REPO_ROOT / "Polymer" / "Data" / "system_dict.pickle"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    family: str
    variant: str
    label: str
    path: Path


RUN_SPECS = [
    RunSpec("baseline", "baseline", "baseline", "Baseline MPC", BASELINE_PATH),
    RunSpec("matrix_legacy", "matrix", "legacy", "Matrix legacy", REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260411_011134" / "input_data.pkl"),
    RunSpec("matrix_refresh", "matrix", "narrow_refresh", "Matrix narrow refresh", REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260420_234944" / "input_data.pkl"),
    RunSpec("matrix_wide", "matrix", "wide_refresh", "Matrix wide", REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260421_011145" / "input_data.pkl"),
    RunSpec("structured_legacy", "structured", "legacy", "Structured legacy", REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260409_193654" / "input_data.pkl"),
    RunSpec("structured_refresh", "structured", "narrow_refresh", "Structured narrow refresh", REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260420_235100" / "input_data.pkl"),
    RunSpec("structured_wide", "structured", "wide_refresh", "Structured wide", REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260421_013208" / "input_data.pkl"),
    RunSpec("reid_refresh", "reidentification", "refresh", "Reidentification refresh", REPO_ROOT / "Polymer" / "Results" / "td3_reidentification_disturb" / "20260420_234346" / "input_data.pkl"),
]


COLORS = {
    "baseline": "#3A3A3A",
    "legacy": "#9C755F",
    "narrow_refresh": "#4E79A7",
    "wide_refresh": "#F28E2B",
    "refresh": "#E15759",
}


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def y_array(bundle: dict) -> np.ndarray:
    return np.asarray(bundle.get("y", bundle.get("y_mpc")), float)[:-1]


def setpoint_phys(bundle: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = len(np.asarray(bundle["steady_states"]["ss_inputs"], float))
    y_ss_phys = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss_phys - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp_scaled_abs = np.asarray(bundle["y_sp"], float) + y_ss_scaled
    return data_min[n_inputs:] + y_sp_scaled_abs * (data_max[n_inputs:] - data_min[n_inputs:])


def test_slice(bundle: dict) -> slice:
    episode_len = int(bundle["time_in_sub_episodes"])
    return slice(-episode_len, None)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def image_lines(filename: str, alt: str) -> list[str]:
    return [f"![{alt}](./polymer_wide_range_matrix_structured/figures/{filename})", ""]


def fmt(value: float, digits: int = 4) -> str:
    arr = np.asarray(value)
    if arr.ndim > 0:
        raise ValueError("fmt expects a scalar")
    if np.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def pct_change(new_value: float, old_value: float) -> str:
    if not np.isfinite(new_value) or not np.isfinite(old_value) or abs(old_value) < 1e-12:
        return "n/a"
    return f"{100.0 * (new_value - old_value) / old_value:+.1f}%"


def disturbance_consistent(bundle: dict, baseline_bundle: dict) -> bool:
    d0 = baseline_bundle.get("disturbance_profile")
    d1 = bundle.get("disturbance_profile")
    if not isinstance(d0, dict) or not isinstance(d1, dict):
        return False
    if set(d0.keys()) != set(d1.keys()):
        return False
    return all(np.allclose(np.asarray(d0[key], float), np.asarray(d1[key], float)) for key in d0.keys())


def reward_components_for_test(bundle: dict) -> dict:
    dy = np.asarray(bundle["delta_y_storage"], float)[test_slice(bundle)]
    du = np.asarray(bundle["delta_u_storage"], float)[test_slice(bundle)]
    q = np.array([5.0, 1.0], float)
    scaled_mae_out = np.mean(np.abs(dy), axis=0)
    weighted_quad = np.mean((dy ** 2) * q.reshape(1, -1), axis=0)
    return {
        "scaled_mae_out1": float(scaled_mae_out[0]),
        "scaled_mae_out2": float(scaled_mae_out[1]),
        "weighted_quad_out1": float(weighted_quad[0]),
        "weighted_quad_out2": float(weighted_quad[1]),
        "u_move_test": float(np.mean(np.linalg.norm(du, axis=1))),
        "reward_test": float(np.asarray(bundle["avg_rewards"], float)[-1]),
    }


def compute_summary(spec: RunSpec, bundle: dict, baseline_bundle: dict, nominal_A_radius: float) -> dict:
    y = y_array(bundle)
    sp = setpoint_phys(bundle)
    err_phys = y - sp
    dy = np.asarray(bundle["delta_y_storage"], float)
    rewards = np.asarray(bundle["avg_rewards"], float)
    ts = test_slice(bundle)
    tail = slice(int(0.9 * len(err_phys)), len(err_phys))

    row = {
        "run_id": spec.run_id,
        "family": spec.family,
        "variant": spec.variant,
        "label": spec.label,
        "run_path": repo_rel(spec.path),
        "run_mode": str(bundle.get("run_mode", "n/a")),
        "state_mode": str(bundle.get("state_mode", "n/a")),
        "observer_update_alignment": str(bundle.get("observer_update_alignment", "n/a")),
        "base_state_norm_mode": str(bundle.get("base_state_norm_mode", "n/a")),
        "mismatch_feature_transform_mode": str(bundle.get("mismatch_feature_transform_mode", "n/a")),
        "range_profile": str(bundle.get("range_profile", "n/a")),
        "update_family": str(bundle.get("update_family", "n/a")),
        "disturbance_consistent": disturbance_consistent(bundle, baseline_bundle),
        "tail_phys_mae_mean": float(np.mean(np.abs(err_phys[tail]))),
        "test_phys_mae_mean": float(np.mean(np.abs(err_phys[ts]))),
        "test_phys_mae_out1": float(np.mean(np.abs(err_phys[ts, 0]))),
        "test_phys_mae_out2": float(np.mean(np.abs(err_phys[ts, 1]))),
        "test_scaled_mae_out1": float(np.mean(np.abs(dy[ts, 0]))),
        "test_scaled_mae_out2": float(np.mean(np.abs(dy[ts, 1]))),
        "test_u_move": float(np.mean(np.linalg.norm(np.asarray(bundle["delta_u_storage"], float)[ts], axis=1))),
        "reward_last": float(rewards[-1]),
        "reward_final10_mean": float(np.mean(rewards[-10:])),
        "reward_best": float(np.max(rewards)),
        "reward_trough": float(np.min(rewards)),
        "reward_trough_episode": int(np.argmin(rewards) + 1),
    }

    target_end10 = row["reward_final10_mean"]
    trough_idx = int(np.argmin(rewards))
    recover_idx = np.where(rewards[trough_idx:] >= target_end10)[0]
    row["reward_recover_episode"] = int(trough_idx + 1 + recover_idx[0]) if recover_idx.size else np.nan

    row.update(reward_components_for_test(bundle))

    if bundle.get("alpha_log") is not None:
        alpha = np.asarray(bundle["alpha_log"], float)[ts]
        delta = np.asarray(bundle["delta_log"], float)[ts]
        derived_sr = nominal_A_radius * alpha
        row.update(
            {
                "alpha_mean_test": float(np.mean(alpha)),
                "alpha_p95_test": float(np.percentile(alpha, 95)),
                "alpha_max_test": float(np.max(alpha)),
                "delta1_mean_test": float(np.mean(delta[:, 0])),
                "delta2_mean_test": float(np.mean(delta[:, 1])),
                "delta1_p95_test": float(np.percentile(delta[:, 0], 95)),
                "delta2_p95_test": float(np.percentile(delta[:, 1], 95)),
                "A_model_delta_ratio_mean_test": float(np.mean(np.asarray(bundle["A_model_delta_ratio_log"], float)[ts])),
                "B_model_delta_ratio_mean_test": float(np.mean(np.asarray(bundle["B_model_delta_ratio_log"], float)[ts])),
                "derived_spectral_mean_test": float(np.mean(derived_sr)),
                "derived_spectral_p95_test": float(np.percentile(derived_sr, 95)),
                "derived_spectral_max_test": float(np.max(derived_sr)),
            }
        )

    if bundle.get("mapped_multiplier_log") is not None:
        mapped = np.asarray(bundle["mapped_multiplier_log"], float)[ts]
        row.update(
            {
                "mapped_multiplier_abs_mean_test": float(np.mean(np.abs(mapped))),
                "mapped_multiplier_abs_p95_test": float(np.percentile(np.abs(mapped), 95)),
            }
        )

    if bundle.get("spectral_radius_log") is not None:
        sr = np.asarray(bundle["spectral_radius_log"], float)[ts]
        row.update(
            {
                "spectral_mean_test": float(np.mean(sr)),
                "spectral_p95_test": float(np.percentile(sr, 95)),
                "spectral_max_test": float(np.max(sr)),
                "near_bound_mean_test": float(np.mean(np.asarray(bundle["near_bound_fraction_log"], float)[ts])),
                "action_saturation_mean_test": float(np.mean(np.asarray(bundle["action_saturation_fraction_log"], float)[ts])),
                "prediction_fallback_frac_test": float(np.mean(np.asarray(bundle.get("prediction_fallback_active_log", np.zeros(len(sr))), float)[ts])),
            }
        )

    if "id_candidate_valid_log" in bundle:
        residual_ratio_log = bundle.get("id_residual_ratio_full_log")
        row.update(
            {
                "candidate_valid_frac": float(np.mean(np.asarray(bundle["id_candidate_valid_log"], float))),
                "update_success_frac": float(np.mean(np.asarray(bundle["id_update_success_log"], float))),
                "condition_median": float(np.percentile(np.asarray(bundle["id_condition_number_log"], float), 50)),
                "condition_p95": float(np.percentile(np.asarray(bundle["id_condition_number_log"], float), 95)),
                "residual_ratio_median": float(np.percentile(np.asarray(residual_ratio_log, float), 50)) if residual_ratio_log is not None else np.nan,
                "residual_ratio_p95": float(np.percentile(np.asarray(residual_ratio_log, float), 95)) if residual_ratio_log is not None else np.nan,
            }
        )

    return row


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_reward_curves(runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    families = [
        ("matrix", ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide"], "Matrix reward curves"),
        ("structured", ["baseline", "structured_legacy", "structured_refresh", "structured_wide"], "Structured reward curves"),
    ]
    for ax, (_, run_ids, title) in zip(axs, families):
        for run_id in run_ids:
            spec = runs[run_id]["spec"]
            rewards = np.asarray(runs[run_id]["bundle"]["avg_rewards"], float)
            variant_key = spec.variant
            color = COLORS["baseline"] if run_id == "baseline" else COLORS[variant_key]
            lw = 2.4 if "wide" in run_id else 1.8
            ls = "--" if run_id == "baseline" else "-"
            ax.plot(np.arange(1, len(rewards) + 1), rewards, color=color, lw=lw, ls=ls, label=spec.label)
            trough_idx = int(np.argmin(rewards))
            ax.scatter(trough_idx + 1, rewards[trough_idx], color=color, s=20)
        ax.set_title(title)
        ax.set_xlabel("Sub-episode")
        ax.set_ylabel("Average reward")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "wide_range_reward_curves.png", dpi=220)
    plt.close(fig)


def plot_final_test_outputs(runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    panels = [
        ("matrix", ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide"], 0, "Matrix family: final test episode"),
        ("structured", ["baseline", "structured_legacy", "structured_refresh", "structured_wide"], 1, "Structured family: final test episode"),
    ]
    for _, run_ids, col, title in panels:
        for run_id in run_ids:
            spec = runs[run_id]["spec"]
            bundle = runs[run_id]["bundle"]
            ts = test_slice(bundle)
            y = y_array(bundle)[ts]
            sp = setpoint_phys(bundle)[ts]
            color = COLORS["baseline"] if run_id == "baseline" else COLORS[spec.variant]
            lw = 2.4 if "wide" in run_id else 1.8
            ls = "--" if run_id == "baseline" else "-"
            axs[0, col].plot(y[:, 0], color=color, lw=lw, ls=ls, label=spec.label)
            axs[1, col].plot(y[:, 1], color=color, lw=lw, ls=ls, label=spec.label)
        axs[0, col].plot(sp[:, 0], color="#000000", lw=1.2, ls=":", label="setpoint")
        axs[1, col].plot(sp[:, 1], color="#000000", lw=1.2, ls=":", label="setpoint")
        axs[0, col].set_title(title)
        axs[1, col].set_xlabel("Final test step")
        axs[0, col].grid(alpha=0.25)
        axs[1, col].grid(alpha=0.25)
        axs[0, col].legend(fontsize=8)
    axs[0, 0].set_ylabel("eta")
    axs[1, 0].set_ylabel("T")
    fig.savefig(FIG_ROOT / "wide_range_final_test_outputs.png", dpi=220)
    plt.close(fig)


def plot_model_usage(runs: dict[str, dict], nominal_A_radius: float) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    matrix_refresh = runs["matrix_refresh"]["bundle"]
    matrix_wide = runs["matrix_wide"]["bundle"]
    ts_m = test_slice(matrix_wide)
    alpha_refresh = np.asarray(matrix_refresh["alpha_log"], float)[test_slice(matrix_refresh)]
    alpha_wide = np.asarray(matrix_wide["alpha_log"], float)[ts_m]
    delta_refresh = np.asarray(matrix_refresh["delta_log"], float)[test_slice(matrix_refresh)]
    delta_wide = np.asarray(matrix_wide["delta_log"], float)[ts_m]

    labels = ["alpha", "delta1", "delta2"]
    x = np.arange(len(labels))
    w = 0.35
    mean_refresh = np.array([np.mean(alpha_refresh), np.mean(delta_refresh[:, 0]), np.mean(delta_refresh[:, 1])], float)
    mean_wide = np.array([np.mean(alpha_wide), np.mean(delta_wide[:, 0]), np.mean(delta_wide[:, 1])], float)
    p95_refresh = np.array([np.percentile(alpha_refresh, 95), np.percentile(delta_refresh[:, 0], 95), np.percentile(delta_refresh[:, 1], 95)], float)
    p95_wide = np.array([np.percentile(alpha_wide, 95), np.percentile(delta_wide[:, 0], 95), np.percentile(delta_wide[:, 1], 95)], float)

    axs[0, 0].bar(x - w / 2, mean_refresh, width=w, color=COLORS["narrow_refresh"], label="narrow mean")
    axs[0, 0].bar(x + w / 2, mean_wide, width=w, color=COLORS["wide_refresh"], label="wide mean")
    axs[0, 0].scatter(x - w / 2, p95_refresh, color="#173F5F", marker="D", s=30, label="narrow p95")
    axs[0, 0].scatter(x + w / 2, p95_wide, color="#A23E02", marker="D", s=30, label="wide p95")
    axs[0, 0].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[0, 0].set_title("Matrix final-test multiplier usage")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].legend(fontsize=8)

    sr_refresh = nominal_A_radius * alpha_refresh
    sr_wide = nominal_A_radius * alpha_wide
    axs[0, 1].hist(sr_refresh, bins=30, alpha=0.6, color=COLORS["narrow_refresh"], label="matrix narrow")
    axs[0, 1].hist(sr_wide, bins=30, alpha=0.6, color=COLORS["wide_refresh"], label="matrix wide")
    axs[0, 1].axvline(1.0, color="black", lw=1.0, ls=":")
    axs[0, 1].set_title("Matrix derived spectral radius in final test")
    axs[0, 1].set_xlabel("spectral radius")
    axs[0, 1].legend(fontsize=8)

    structured_refresh = runs["structured_refresh"]["bundle"]
    structured_wide = runs["structured_wide"]["bundle"]
    mm_refresh = np.asarray(structured_refresh["mapped_multiplier_log"], float)[test_slice(structured_refresh)]
    mm_wide = np.asarray(structured_wide["mapped_multiplier_log"], float)[test_slice(structured_wide)]
    dim_labels = [f"m{i+1}" for i in range(mm_wide.shape[1])]
    x2 = np.arange(mm_wide.shape[1])
    axs[1, 0].bar(x2 - w / 2, np.mean(mm_refresh, axis=0), width=w, color=COLORS["narrow_refresh"], label="narrow mean")
    axs[1, 0].bar(x2 + w / 2, np.mean(mm_wide, axis=0), width=w, color=COLORS["wide_refresh"], label="wide mean")
    axs[1, 0].scatter(x2 - w / 2, np.percentile(mm_refresh, 95, axis=0), color="#173F5F", marker="D", s=25, label="narrow p95")
    axs[1, 0].scatter(x2 + w / 2, np.percentile(mm_wide, 95, axis=0), color="#A23E02", marker="D", s=25, label="wide p95")
    axs[1, 0].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[1, 0].set_title("Structured final-test multiplier usage")
    axs[1, 0].set_xticks(x2)
    axs[1, 0].set_xticklabels(dim_labels)
    axs[1, 0].legend(fontsize=8)

    sr_refresh_s = np.asarray(structured_refresh["spectral_radius_log"], float)[test_slice(structured_refresh)]
    sr_wide_s = np.asarray(structured_wide["spectral_radius_log"], float)[test_slice(structured_wide)]
    axs[1, 1].hist(sr_refresh_s, bins=30, alpha=0.6, color=COLORS["narrow_refresh"], label="structured narrow")
    axs[1, 1].hist(sr_wide_s, bins=30, alpha=0.6, color=COLORS["wide_refresh"], label="structured wide")
    axs[1, 1].axvline(1.0, color="black", lw=1.0, ls=":")
    axs[1, 1].set_title("Structured spectral radius in final test")
    axs[1, 1].set_xlabel("spectral radius")
    axs[1, 1].legend(fontsize=8)

    for ax in axs.flat:
        ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "wide_range_model_usage.png", dpi=220)
    plt.close(fig)


def plot_tradeoff(summary_map: dict[str, dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    run_ids = ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide", "structured_legacy", "structured_refresh", "structured_wide", "reid_refresh"]
    for run_id in run_ids:
        row = summary_map[run_id]
        color = COLORS["baseline"] if run_id == "baseline" else COLORS.get(row["variant"], "#888888")
        marker = "X" if "wide" in run_id else "o"
        axs[0].scatter(row["reward_test"], row["test_phys_mae_mean"], color=color, marker=marker, s=90)
        axs[0].text(row["reward_test"], row["test_phys_mae_mean"] + 0.002, row["label"], fontsize=8)
    axs[0].set_title("Final test reward vs final test physical MAE")
    axs[0].set_xlabel("Final test average reward")
    axs[0].set_ylabel("Final test physical MAE mean")
    axs[0].grid(alpha=0.25)

    labels = ["Matrix narrow", "Matrix wide", "Structured narrow", "Structured wide"]
    rows = [summary_map["matrix_refresh"], summary_map["matrix_wide"], summary_map["structured_refresh"], summary_map["structured_wide"]]
    x = np.arange(len(labels))
    w = 0.35
    out1 = [row["test_scaled_mae_out1"] for row in rows]
    out2 = [row["test_scaled_mae_out2"] for row in rows]
    axs[1].bar(x - w / 2, out1, width=w, color="#59A14F", label="output 1 scaled MAE")
    axs[1].bar(x + w / 2, out2, width=w, color="#E15759", label="output 2 scaled MAE")
    axs[1].set_title("Final test scaled MAE by output")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=15, ha="right")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_reward_tradeoff.png", dpi=220)
    plt.close(fig)


def write_markdown(rows: list[dict], nominal_A_radius: float) -> None:
    summary = {row["run_id"]: row for row in rows}
    matrix_legacy = summary["matrix_legacy"]
    matrix_refresh = summary["matrix_refresh"]
    matrix_wide = summary["matrix_wide"]
    structured_legacy = summary["structured_legacy"]
    structured_refresh = summary["structured_refresh"]
    structured_wide = summary["structured_wide"]
    reid_refresh = summary["reid_refresh"]
    baseline = summary["baseline"]

    lines: list[str] = [
        "# Polymer Wide-Range Matrix and Structured Report",
        "",
        "Date: 2026-04-21",
        "",
        "This report focuses on the latest widened-range polymer matrix and structured-matrix runs and compares them against the previous legacy/narrow runs, the baseline MPC, and the latest polymer reidentification run.",
        "",
        "The goal is to answer four questions with data from the saved runs:",
        "",
        "1. Why did the widened matrix/structured methods improve, and why did reidentification still fail?",
        "2. Why does structured wide achieve better reward while not improving the held-out evaluation episode?",
        "3. Why do the wide runs first degrade and then recover, and what is a practical fix?",
        "4. Can the range be widened further, and what should limit that when the controller uses a state-space model?",
        "",
        "## Run Set",
        "",
    ]

    config_rows = []
    for run_id in ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide", "structured_legacy", "structured_refresh", "structured_wide", "reid_refresh"]:
        row = summary[run_id]
        config_rows.append(
            [
                row["label"],
                f"`{row['run_path']}`",
                row["run_mode"],
                row["state_mode"],
                row["observer_update_alignment"],
                row["base_state_norm_mode"],
                row["mismatch_feature_transform_mode"],
                row["range_profile"],
                row["update_family"],
                "yes" if row["disturbance_consistent"] else "no",
            ]
        )
    lines += markdown_table(
        [
            "Run",
            "Saved bundle",
            "run_mode",
            "state_mode",
            "observer",
            "base_state_norm",
            "mismatch_transform",
            "range_profile",
            "update_family",
            "disturbance consistent",
        ],
        config_rows,
    )
    lines += [
        "",
        "All compared runs use the same polymer disturbance schedule. The saved `qi`, `qs`, and `ha` arrays in the result bundles are numerically identical to the baseline schedule, so the widened-range comparison is not confounded by different disturbances.",
        "",
        "## Main Comparison",
        "",
    ]

    perf_rows = []
    for run_id in ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide", "structured_legacy", "structured_refresh", "structured_wide", "reid_refresh"]:
        row = summary[run_id]
        perf_rows.append(
            [
                row["label"],
                fmt(row["tail_phys_mae_mean"]),
                fmt(row["test_phys_mae_mean"]),
                fmt(row["test_phys_mae_out1"]),
                fmt(row["test_phys_mae_out2"]),
                fmt(row["reward_test"]),
                fmt(row["reward_final10_mean"]),
                fmt(row["test_u_move"]),
            ]
        )
    lines += markdown_table(
        [
            "Run",
            "Tail phys MAE",
            "Final test phys MAE",
            "Final test out1 MAE",
            "Final test out2 MAE",
            "Final test reward",
            "Final-10 reward",
            "Final test input move",
        ],
        perf_rows,
    )
    lines += [
        "",
        f"- Matrix wide is the strongest run in this study. Final test physical MAE drops from `{fmt(matrix_refresh['test_phys_mae_mean'])}` in the previous narrow refresh to `{fmt(matrix_wide['test_phys_mae_mean'])}` ({pct_change(matrix_wide['test_phys_mae_mean'], matrix_refresh['test_phys_mae_mean'])}), beating both the legacy matrix run `{fmt(matrix_legacy['test_phys_mae_mean'])}` and baseline MPC `{fmt(baseline['test_phys_mae_mean'])}`.",
        f"- Structured wide is mixed. Its overall late-run tail MAE improves from `{fmt(structured_refresh['tail_phys_mae_mean'])}` to `{fmt(structured_wide['tail_phys_mae_mean'])}`, but the held-out final test gets worse: `{fmt(structured_refresh['test_phys_mae_mean'])}` to `{fmt(structured_wide['test_phys_mae_mean'])}` ({pct_change(structured_wide['test_phys_mae_mean'], structured_refresh['test_phys_mae_mean'])}).",
        f"- Reidentification still fails to convert the richer RL state into better control. Its final test MAE remains `{fmt(reid_refresh['test_phys_mae_mean'])}`, worse than matrix wide and worse than baseline MPC.",
        "",
    ]
    lines += image_lines("wide_range_reward_curves.png", "Reward curves for matrix and structured narrow versus wide runs")
    lines += [
        "The reward curves show the same pattern in both wide runs: a severe early deterioration followed by recovery to a better late-stage reward than the narrow runs. The matrix wide trough is deeper (`{}` at episode `{}`) than the structured wide trough (`{}` at episode `{}`), but both recover only much later, around episodes `{}` and `{}` respectively.".format(
            fmt(matrix_wide["reward_trough"]), matrix_wide["reward_trough_episode"], fmt(structured_wide["reward_trough"]), structured_wide["reward_trough_episode"], int(matrix_wide["reward_recover_episode"]), int(structured_wide["reward_recover_episode"])
        ),
        "",
        "## Final Test Episode",
        "",
    ]
    lines += image_lines("wide_range_final_test_outputs.png", "Final held-out test episode outputs for matrix and structured families")
    lines += [
        f"Matrix wide improves both outputs in the held-out test episode: output 1 MAE falls from `{fmt(matrix_refresh['test_phys_mae_out1'])}` to `{fmt(matrix_wide['test_phys_mae_out1'])}`, and output 2 MAE falls from `{fmt(matrix_refresh['test_phys_mae_out2'])}` to `{fmt(matrix_wide['test_phys_mae_out2'])}`.",
        f"Structured wide does not. Output 1 improves from `{fmt(structured_refresh['test_phys_mae_out1'])}` to `{fmt(structured_wide['test_phys_mae_out1'])}`, but output 2 gets substantially worse: `{fmt(structured_refresh['test_phys_mae_out2'])}` to `{fmt(structured_wide['test_phys_mae_out2'])}`. This is why the final test mean degrades even though the overall late-run tail and reward look better.",
        "",
        "## Why Matrix Wide Worked And Reidentification Did Not",
        "",
        "The latest matrix result is successful because the controller is allowed to choose from a small, direct, always-realized model family. There is no identification gate between the action and the prediction model used by MPC. The action changes three smooth multipliers and the model correction is immediately available to the optimizer on every step.",
        "",
        f"In the final test episode, matrix wide mainly uses a damped `A` correction and a stronger first-input `B` correction: `alpha` mean is `{fmt(matrix_wide['alpha_mean_test'])}`, while `delta1` mean is `{fmt(matrix_wide['delta1_mean_test'])}` and its p95 reaches `{fmt(matrix_wide['delta1_p95_test'])}`. That is a coherent low-dimensional correction, not a noisy online identification problem.",
        "",
        f"Reidentification is fundamentally different. The policy may request strong blend authority, but the online identification engine almost never produces an admissible new model. In the latest polymer reidentification run, candidate-valid fraction is only `{fmt(reid_refresh['candidate_valid_frac'])}` and update-success fraction is only `{fmt(reid_refresh['update_success_frac'])}`. Median condition number is `{fmt(reid_refresh['condition_median'], digits=1)}` and p95 is `{fmt(reid_refresh['condition_p95'], digits=1)}`. So reidentification is bottlenecked by data informativity and numerical conditioning, not by the policy alone.",
        "",
        "This difference matches the literature. The direct multiplier methods behave like bounded parametric adaptation inside a fixed low-dimensional family. Reidentification, by contrast, needs persistently informative data and a numerically healthy regression problem. The adaptive MPC and identification literature repeatedly stresses persistence of excitation, targeted excitation, and dual control/experiment design as prerequisites for successful online model maintenance [Berberich2022] [Heirung2015] [Heirung2017] [Parsi2020] [Oshima2024].",
        "",
        "## Why Structured Wide Gets Better Reward But Worse Evaluation",
        "",
        "The answer is in the reward itself. The reward is built from scaled output errors and input moves, with output weights `Q = [5, 1]`. That means output 1 is five times more expensive than output 2 in the quadratic term. So a policy can improve reward by helping output 1 a lot even if output 2 gets worse.",
        "",
    ]

    reward_trade_rows = []
    for row in [matrix_refresh, matrix_wide, structured_refresh, structured_wide]:
        reward_trade_rows.append(
            [
                row["label"],
                fmt(row["test_scaled_mae_out1"]),
                fmt(row["test_scaled_mae_out2"]),
                fmt(row["weighted_quad_out1"]),
                fmt(row["weighted_quad_out2"]),
                fmt(row["test_u_move"]),
                fmt(row["reward_test"]),
            ]
        )
    lines += markdown_table(
        [
            "Run",
            "Final test scaled MAE out1",
            "Final test scaled MAE out2",
            "Weighted quad out1",
            "Weighted quad out2",
            "Final test input move",
            "Final test reward",
        ],
        reward_trade_rows,
    )
    lines += image_lines("wide_range_reward_tradeoff.png", "Reward versus evaluation tradeoff for the wide matrix and structured runs")
    lines += [
        f"Structured wide improves output 1 sharply: final test scaled MAE drops from `{fmt(structured_refresh['test_scaled_mae_out1'])}` to `{fmt(structured_wide['test_scaled_mae_out1'])}`. But output 2 worsens from `{fmt(structured_refresh['test_scaled_mae_out2'])}` to `{fmt(structured_wide['test_scaled_mae_out2'])}`. Because output 1 has the larger weight, the weighted output-1 term falls from `{fmt(structured_refresh['weighted_quad_out1'])}` to `{fmt(structured_wide['weighted_quad_out1'])}`, while the output-2 term rises only modestly from `{fmt(structured_refresh['weighted_quad_out2'])}` to `{fmt(structured_wide['weighted_quad_out2'])}`. The result is a better reward even though the held-out evaluation episode is worse on mean physical tracking.",
        "",
        f"This strongly suggests that if the evaluation objective values both outputs more evenly, then yes, the reward parameters should be revisited. The run is not benefiting from a changed disturbance profile; the disturbance schedules are identical. It is benefiting from an objective mismatch between the training reward and the evaluation metric.",
        "",
        "## Why The Wide Runs First Degrade And Then Recover",
        "",
        "The early deterioration is consistent with a larger action/model-search space. Widening the multiplier ranges expands the set of prediction models the agent can induce. Early in training, the replay buffer is dominated by low-quality exploratory transitions from this larger space, so the policy gets worse before it learns which stronger corrections are actually useful. Once enough informative transitions accumulate, the policy recovers and starts exploiting the wider authority.",
        "",
        "The control literature and RL literature both suggest practical fixes:",
        "",
        "- Progressive widening / continuation: instead of jumping directly from narrow to wide bounds, increase the bounds in stages once reward, solver success, and held-out tracking have stabilized. This is conceptually similar to coarse-to-fine action selection and continuation-style training [Seo2025].",
        "- Smoother exploration: use temporally coherent exploration rather than step-to-step jagged action noise. Autoregressive exploration is a direct literature-backed way to reduce violent exploratory swings in continuous control [Korenkevych2019].",
        "- Data-aware excitation only when needed: if model learning is part of the method, dual/adaptive MPC papers recommend adding excitation only when uncertainty is high or data are not informative enough, rather than exciting continuously [Heirung2015] [Heirung2017] [Parsi2020].",
        "",
        "In this repository, the most practical fix is staged widening. Start from the narrow run, continue training with an intermediate range, and widen again only after the held-out test episode improves and the model-usage statistics are not saturating.",
        "",
        "## How Far Can The Range Be Widened Safely?",
        "",
        f"The nominal polymer physical `A` has spectral radius `{fmt(nominal_A_radius)}`. In the final test episode, matrix wide already reaches a derived p95 spectral radius of `{fmt(matrix_wide['derived_spectral_p95_test'])}` and max `{fmt(matrix_wide['derived_spectral_max_test'])}`. Structured wide is more aggressive still: mean spectral radius is `{fmt(structured_wide['spectral_mean_test'])}`, p95 is `{fmt(structured_wide['spectral_p95_test'])}`, max is `{fmt(structured_wide['spectral_max_test'])}`, and several structured multipliers hit the hard upper bound `1.2` at p95.",
        "",
        "So the answer is different for the two methods:",
        "",
        f"- Matrix: a slightly wider range may still be worth testing, but not symmetrically. The successful wide matrix policy mainly uses stronger `B` correction and slightly smaller `A`, so the safer next ablation is to widen `B` more than `A`, not to raise every bound uniformly.",
        f"- Structured: the current wide run already looks too aggressive for held-out evaluation. It pushes multiple grouped multipliers to `1.2`, keeps the test spectral radius above `1.0` on average, and doubles the final-test input movement from `{fmt(structured_refresh['test_u_move'])}` to `{fmt(structured_wide['test_u_move'])}`. Widening further before adding guards is not justified by these results.",
        "",
        "With a state-space model in the loop, the practical limit is not a single scalar bound. It is the largest uncertainty set for which the prediction model remains numerically admissible for MPC: stabilizable/detectable enough for the observer-controller pair, solver-feasible, and not so aggressive that held-out performance collapses. Robust MPC literature frames this as keeping the model family inside an uncertainty set where recursive feasibility and robust performance can still be guaranteed or approximated [Chen2024] [Limon2013] [Kothare2010].",
        "",
        "For this project, a practical safe-widening recipe is:",
        "",
        "1. Keep the current solve fallback for structured wide.",
        "2. Add an explicit spectral-radius cap or smooth fade-back to nominal when the prediction model gets too aggressive.",
        "3. Widen bounds asymmetrically, favoring `B` before `A`.",
        "4. Use a staged widening schedule tied to held-out test MAE and solver/fallback statistics.",
        "5. Treat the empirical limit as reached when p95 multiplier use is already on the hard bound and held-out test performance no longer improves.",
        "",
        "## Model-Usage Diagnostics",
        "",
    ]
    lines += image_lines("wide_range_model_usage.png", "Wide-range multiplier usage and spectral-radius diagnostics")
    lines += [
        f"Matrix wide uses the expanded range in a targeted way. It does not simply saturate everything upward. Instead, it tends to push the first input gain higher while keeping `A` on average below nominal. Structured wide behaves differently: several grouped multipliers have p95 equal to the hard upper bound `1.2`, and the test spectral radius stays above `1.1` on average. That explains why structured wide can still improve reward while giving an over-aggressive held-out trajectory.",
        "",
        "## Conclusions",
        "",
        f"- The latest matrix wide run is genuinely more successful than the previous matrix runs. Its final test MAE `{fmt(matrix_wide['test_phys_mae_mean'])}` beats the previous narrow refresh `{fmt(matrix_refresh['test_phys_mae_mean'])}`, the legacy matrix run `{fmt(matrix_legacy['test_phys_mae_mean'])}`, and baseline MPC `{fmt(baseline['test_phys_mae_mean'])}`.",
        f"- Structured wide is not a clean success. It improves late-run reward and aggregate tail MAE, but its held-out final test MAE degrades from `{fmt(structured_refresh['test_phys_mae_mean'])}` to `{fmt(structured_wide['test_phys_mae_mean'])}` because it over-optimizes the heavily weighted first output and becomes much more aggressive on the prediction model and control effort.",
        f"- Direct multiplier methods work better than reidentification here because they operate in a small always-realized model family. Reidentification still fails due to poor candidate validity (`{fmt(reid_refresh['candidate_valid_frac'])}`) and poor conditioning, not because the new state conditioning failed.",
        "- The degradation-then-recovery pattern is expected when the authority range is widened abruptly. The most practical fix is staged widening with smoother exploration and explicit held-out performance checks.",
        "- Matrix may be widened a bit further, but only asymmetrically and with monitoring. Structured should not be widened further until a stability/admissibility guard is added.",
        "",
        "## Sources",
        "",
        "- [Berberich2022] Forward-looking persistent excitation in model predictive control.",
        "- [Heirung2015] MPC-based dual control with online experiment design.",
        "- [Heirung2017] Dual adaptive model predictive control.",
        "- [Parsi2020] Active exploration in adaptive model predictive control.",
        "- [Oshima2024] Targeted excitation and re-identification methods for multivariate process and model predictive control.",
        "- [Korenkevych2019] Autoregressive Policies for Continuous Control Deep Reinforcement Learning.",
        "- [Seo2025] Continuous Control with Coarse-to-fine Reinforcement Learning.",
        "- [Chen2024] Robust model predictive control with polytopic model uncertainty through System Level Synthesis.",
        "- [Limon2013] Robust feedback model predictive control of constrained uncertain systems.",
        "- [Kothare2010] Robust model predictive control design with input constraints.",
        "",
        "[Berberich2022]: https://doi.org/10.1016/j.automatica.2021.110033",
        "[Heirung2015]: https://doi.org/10.1016/j.jprocont.2015.04.012",
        "[Heirung2017]: https://doi.org/10.1016/j.automatica.2017.01.030",
        "[Parsi2020]: https://www.research-collection.ethz.ch/handle/20.500.11850/461407",
        "[Oshima2024]: https://doi.org/10.1016/j.jprocont.2024.103190",
        "[Korenkevych2019]: https://doi.org/10.24963/ijcai.2019/382",
        "[Seo2025]: https://proceedings.mlr.press/v270/seo25a.html",
        "[Chen2024]: https://doi.org/10.1016/j.automatica.2023.111431",
        "[Limon2013]: https://doi.org/10.1016/j.jprocont.2012.08.003",
        "[Kothare2010]: https://doi.org/10.1016/j.isatra.2009.10.003",
    ]

    (REPO_ROOT / "report" / "polymer_wide_range_matrix_structured_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    with POLYMER_SYS_DICT.open("rb") as fh:
        sys_dict = pickle.load(fh)
    nominal_A_radius = float(np.max(np.abs(np.linalg.eigvals(np.asarray(sys_dict["A"], float)))))

    runs: dict[str, dict] = {}
    for spec in RUN_SPECS:
        runs[spec.run_id] = {"spec": spec, "bundle": load_pickle(spec.path)}

    baseline_bundle = runs["baseline"]["bundle"]
    rows = [compute_summary(spec=spec, bundle=runs[spec.run_id]["bundle"], baseline_bundle=baseline_bundle, nominal_A_radius=nominal_A_radius) for spec in RUN_SPECS]
    summary_map = {row["run_id"]: row for row in rows}

    save_csv(
        DATA_ROOT / "wide_range_summary.csv",
        rows,
        [
            "run_id",
            "family",
            "variant",
            "label",
            "run_path",
            "run_mode",
            "state_mode",
            "observer_update_alignment",
            "base_state_norm_mode",
            "mismatch_feature_transform_mode",
            "range_profile",
            "update_family",
            "disturbance_consistent",
            "tail_phys_mae_mean",
            "test_phys_mae_mean",
            "test_phys_mae_out1",
            "test_phys_mae_out2",
            "test_scaled_mae_out1",
            "test_scaled_mae_out2",
            "reward_test",
            "reward_final10_mean",
            "reward_trough",
            "reward_trough_episode",
            "reward_recover_episode",
            "test_u_move",
            "weighted_quad_out1",
            "weighted_quad_out2",
            "alpha_mean_test",
            "alpha_p95_test",
            "alpha_max_test",
            "delta1_mean_test",
            "delta2_mean_test",
            "delta1_p95_test",
            "delta2_p95_test",
            "A_model_delta_ratio_mean_test",
            "B_model_delta_ratio_mean_test",
            "derived_spectral_mean_test",
            "derived_spectral_p95_test",
            "derived_spectral_max_test",
            "spectral_mean_test",
            "spectral_p95_test",
            "spectral_max_test",
            "near_bound_mean_test",
            "action_saturation_mean_test",
            "prediction_fallback_frac_test",
            "candidate_valid_frac",
            "update_success_frac",
            "condition_median",
            "condition_p95",
            "residual_ratio_median",
            "residual_ratio_p95",
        ],
    )

    plot_reward_curves(runs)
    plot_final_test_outputs(runs)
    plot_model_usage(runs, nominal_A_radius=nominal_A_radius)
    plot_tradeoff(summary_map)
    write_markdown(rows, nominal_A_radius=nominal_A_radius)


if __name__ == "__main__":
    main()
