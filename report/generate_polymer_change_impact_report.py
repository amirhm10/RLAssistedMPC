from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "report" / "polymer_change_impact"
FIG_ROOT = REPORT_ROOT / "figures"
DATA_ROOT = REPORT_ROOT / "data"

BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
POLYMER_MINMAX_PATH = REPO_ROOT / "Polymer" / "Data" / "min_max_states.pickle"


@dataclass(frozen=True)
class RunSpec:
    family: str
    run_label: str
    path: Path
    is_refreshed: bool


RUN_SPECS = [
    RunSpec(
        family="residual",
        run_label="Residual legacy",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260413_004620" / "input_data.pkl",
        is_refreshed=False,
    ),
    RunSpec(
        family="residual",
        run_label="Residual refreshed",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260420_225631" / "input_data.pkl",
        is_refreshed=True,
    ),
    RunSpec(
        family="matrix",
        run_label="Matrix legacy",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260411_011134" / "input_data.pkl",
        is_refreshed=False,
    ),
    RunSpec(
        family="matrix",
        run_label="Matrix refreshed",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260420_215528" / "input_data.pkl",
        is_refreshed=True,
    ),
]


OUTPUT_LABELS = ["eta", "T"]


class RunningFeatureNormalizer:
    def __init__(self, feature_dim: int, clip_obs: float = 10.0, epsilon: float = 1e-8):
        self.feature_dim = int(feature_dim)
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self.count = 0.0
        self.mean = np.zeros(self.feature_dim, dtype=float)
        self.m2 = np.zeros(self.feature_dim, dtype=float)

    @property
    def var(self) -> np.ndarray:
        return self.m2 / max(self.count, 1.0)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, float).reshape(-1)
        self.count += 1.0
        delta = values - self.mean
        self.mean += delta / self.count
        delta2 = values - self.mean
        self.m2 += delta * delta2
        z = (values - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip_obs, self.clip_obs)


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


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


def detect_setpoint_change_segments(y_sp: np.ndarray) -> list[tuple[int, int]]:
    change_idx = np.flatnonzero(np.any(np.abs(np.diff(y_sp, axis=0)) > 1e-12, axis=1)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(y_sp)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s > 5]


def late_segment_mask(y_sp: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(y_sp), dtype=bool)
    for start, end in detect_setpoint_change_segments(y_sp):
        window = max(10, min(30, (end - start) // 4))
        mask[max(start, end - window) : end] = True
    return mask


def quantile_span(arr: np.ndarray) -> np.ndarray:
    return np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)


def physical_xhat_policy_block(bundle: dict, min_s: np.ndarray, max_s: np.ndarray, mode_override: str | None = None) -> np.ndarray:
    xhatdhat = np.asarray(bundle["xhatdhat"][:, :-1], float).T
    n_outputs = int(np.asarray(bundle["y_sp"]).shape[1])
    n_phys = xhatdhat.shape[1] - n_outputs
    xhat_phys = xhatdhat[:, :n_phys]
    mode = str(mode_override or bundle.get("base_state_norm_mode", "fixed_minmax")).strip().lower()

    if mode == "fixed_minmax":
        span = np.maximum(max_s[:n_phys] - min_s[:n_phys], 1e-12)
        return 2.0 * ((xhat_phys - min_s[:n_phys]) / span) - 1.0

    if mode != "running_zscore_physical_xhat":
        raise ValueError(f"Unsupported mode: {mode}")

    normalizer = RunningFeatureNormalizer(
        feature_dim=n_phys,
        clip_obs=float(bundle.get("base_state_running_norm_clip", 10.0)),
        epsilon=float(bundle.get("base_state_running_norm_eps", 1e-8)),
    )
    normed = np.zeros_like(xhat_phys, dtype=float)
    for idx, row in enumerate(xhat_phys):
        normed[idx] = normalizer.normalize(row)
    return normed


def state_span_summary(bundle: dict, min_s: np.ndarray, max_s: np.ndarray, mode_override: str | None = None) -> dict:
    y_sp = np.asarray(bundle["y_sp"], float)
    late_mask = late_segment_mask(y_sp)
    states = physical_xhat_policy_block(bundle, min_s, max_s, mode_override=mode_override)
    full_span = quantile_span(states)
    late_span = quantile_span(states[late_mask])
    return {
        "full_span_med": float(np.median(full_span)),
        "late_span_med": float(np.median(late_span)),
        "late_full_ratio_med": float(np.median(late_span / np.maximum(full_span, 1e-12))),
    }


def exact_clip_fraction(arr: np.ndarray, clip_value: float = 3.0) -> np.ndarray:
    return np.mean(np.isclose(np.abs(arr), clip_value, atol=1e-8), axis=0)


def compute_metrics(bundle: dict, baseline_bundle: dict | None = None) -> dict:
    y_err_scaled = np.asarray(bundle["delta_y_storage"], float)
    u_err_scaled = np.asarray(bundle["delta_u_storage"], float)
    rewards = np.asarray(bundle.get("avg_rewards", []), float)
    y_phys = y_array(bundle)
    y_sp_phys = setpoint_phys(bundle)
    y_err_phys = y_phys - y_sp_phys
    tail = slice(int(0.9 * y_err_scaled.shape[0]), y_err_scaled.shape[0])

    metrics = {
        "scaled_mae_out1": float(np.mean(np.abs(y_err_scaled[:, 0]))),
        "scaled_mae_out2": float(np.mean(np.abs(y_err_scaled[:, 1]))),
        "scaled_mae_mean": float(np.mean(np.mean(np.abs(y_err_scaled), axis=0))),
        "scaled_mae_tail_mean": float(np.mean(np.mean(np.abs(y_err_scaled[tail]), axis=0))),
        "phys_mae_out1": float(np.mean(np.abs(y_err_phys[:, 0]))),
        "phys_mae_out2": float(np.mean(np.abs(y_err_phys[:, 1]))),
        "phys_mae_mean": float(np.mean(np.mean(np.abs(y_err_phys), axis=0))),
        "phys_mae_tail_out1": float(np.mean(np.abs(y_err_phys[tail, 0]))),
        "phys_mae_tail_out2": float(np.mean(np.abs(y_err_phys[tail, 1]))),
        "phys_mae_tail_mean": float(np.mean(np.mean(np.abs(y_err_phys[tail]), axis=0))),
        "u_move_mean": float(np.mean(np.linalg.norm(u_err_scaled, axis=1))),
        "u_move_tail": float(np.mean(np.linalg.norm(u_err_scaled[tail], axis=1))),
        "reward_final20_mean": float(np.mean(rewards[-20:])) if rewards.size else np.nan,
        "reward_best": float(np.max(rewards)) if rewards.size else np.nan,
        "reward_last": float(rewards[-1]) if rewards.size else np.nan,
    }

    for key in ("innovation_log", "tracking_error_log"):
        if bundle.get(key) is not None:
            arr = np.asarray(bundle[key], float)
            prefix = key.replace("_log", "")
            metrics[f"{prefix}_p99_out1"] = float(np.percentile(np.abs(arr[:, 0]), 99))
            metrics[f"{prefix}_p99_out2"] = float(np.percentile(np.abs(arr[:, 1]), 99))
            metrics[f"{prefix}_exact3_out1"] = float(exact_clip_fraction(arr)[0])
            metrics[f"{prefix}_exact3_out2"] = float(exact_clip_fraction(arr)[1])
        else:
            prefix = key.replace("_log", "")
            metrics[f"{prefix}_p99_out1"] = np.nan
            metrics[f"{prefix}_p99_out2"] = np.nan
            metrics[f"{prefix}_exact3_out1"] = np.nan
            metrics[f"{prefix}_exact3_out2"] = np.nan

    for key in ("innovation_raw_log", "tracking_error_raw_log"):
        if bundle.get(key) is not None:
            arr = np.asarray(bundle[key], float)
            prefix = key.replace("_raw_log", "_raw")
            metrics[f"{prefix}_p99_out1"] = float(np.percentile(np.abs(arr[:, 0]), 99))
            metrics[f"{prefix}_p99_out2"] = float(np.percentile(np.abs(arr[:, 1]), 99))
            metrics[f"{prefix}_gt3_out1"] = float(np.mean(np.abs(arr[:, 0]) > 3.0))
            metrics[f"{prefix}_gt3_out2"] = float(np.mean(np.abs(arr[:, 1]) > 3.0))
        else:
            prefix = key.replace("_raw_log", "_raw")
            metrics[f"{prefix}_p99_out1"] = np.nan
            metrics[f"{prefix}_p99_out2"] = np.nan
            metrics[f"{prefix}_gt3_out1"] = np.nan
            metrics[f"{prefix}_gt3_out2"] = np.nan

    if bundle.get("rho_log") is not None:
        rho = np.asarray(bundle["rho_log"], float)
        metrics["rho_mean"] = float(np.mean(rho))
        metrics["rho_eq1_frac"] = float(np.mean(rho >= 0.999999))
    else:
        metrics["rho_mean"] = np.nan
        metrics["rho_eq1_frac"] = np.nan

    if bundle.get("deadband_active_log") is not None:
        metrics["deadband_frac"] = float(np.mean(np.asarray(bundle["deadband_active_log"], float) > 0.5))
    else:
        metrics["deadband_frac"] = np.nan

    if bundle.get("delta_u_res_exec_log") is not None:
        du = np.asarray(bundle["delta_u_res_exec_log"], float)
        metrics["residual_norm_mean"] = float(np.mean(np.linalg.norm(du, axis=1)))
        metrics["residual_norm_tail"] = float(np.mean(np.linalg.norm(du[tail], axis=1)))
    else:
        metrics["residual_norm_mean"] = np.nan
        metrics["residual_norm_tail"] = np.nan

    if baseline_bundle is not None:
        baseline_err_scaled = np.asarray(baseline_bundle["delta_y_storage"], float)
        baseline_y_err_phys = y_array(baseline_bundle) - setpoint_phys(baseline_bundle)
        metrics["tail_scaled_mae_delta_vs_baseline"] = float(
            np.mean(np.abs(y_err_scaled[tail])) - np.mean(np.abs(baseline_err_scaled[tail]))
        )
        metrics["tail_phys_mae_delta_vs_baseline"] = float(
            np.mean(np.abs(y_err_phys[tail])) - np.mean(np.abs(baseline_y_err_phys[tail]))
        )
    else:
        metrics["tail_scaled_mae_delta_vs_baseline"] = np.nan
        metrics["tail_phys_mae_delta_vs_baseline"] = np.nan

    return metrics


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    values = np.asarray(values, float)
    if values.size < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def nanmean_pair(values: list[float]) -> float:
    arr = np.asarray(values, float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def plot_reward_curves(baseline_bundle: dict, runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = {"baseline": "#444444", "legacy": "#3A7CA5", "refreshed": "#D66A4E"}
    baseline_rewards = moving_average(np.asarray(baseline_bundle["avg_rewards"], float))

    for ax, family in zip(axs, ("residual", "matrix")):
        legacy = runs[f"{family}_legacy"]
        refreshed = runs[f"{family}_refreshed"]
        ax.plot(baseline_rewards, color=colors["baseline"], linewidth=2.0, label="Baseline MPC")
        ax.plot(moving_average(np.asarray(legacy["bundle"]["avg_rewards"], float)), color=colors["legacy"], linewidth=1.8, label=f"{family.title()} legacy")
        ax.plot(moving_average(np.asarray(refreshed["bundle"]["avg_rewards"], float)), color=colors["refreshed"], linewidth=1.8, label=f"{family.title()} refreshed")
        ax.set_title(f"{family.title()} average reward by sub-episode")
        ax.set_ylabel("avg reward")
        ax.legend(loc="lower right", fontsize=8)
    axs[-1].set_xlabel("Sub-episode")
    fig.savefig(FIG_ROOT / "polymer_change_reward_curves.png", dpi=220)
    plt.close(fig)


def plot_tail_traces(baseline_bundle: dict, runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"baseline": "#444444", "legacy": "#3A7CA5", "refreshed": "#D66A4E"}
    tail_n = 400

    for row_idx, family in enumerate(("residual", "matrix")):
        baseline_y = y_array(baseline_bundle)[-tail_n:]
        baseline_sp = setpoint_phys(baseline_bundle)[-tail_n:]
        legacy_y = y_array(runs[f"{family}_legacy"]["bundle"])[-tail_n:]
        refreshed_y = y_array(runs[f"{family}_refreshed"]["bundle"])[-tail_n:]
        time = np.arange(tail_n)
        for out_idx, label in enumerate(OUTPUT_LABELS):
            ax = axs[row_idx, out_idx]
            ax.plot(time, baseline_sp[:, out_idx], color="black", linestyle="--", linewidth=1.2, label="setpoint")
            ax.plot(time, baseline_y[:, out_idx], color=colors["baseline"], linewidth=1.8, label="baseline MPC")
            ax.plot(time, legacy_y[:, out_idx], color=colors["legacy"], linewidth=1.4, label="legacy")
            ax.plot(time, refreshed_y[:, out_idx], color=colors["refreshed"], linewidth=1.4, label="refreshed")
            ax.set_title(f"{family.title()} tail trace: {label}")
            if out_idx == 0:
                ax.set_ylabel("physical output")
            if row_idx == 1:
                ax.set_xlabel("Step")
            if row_idx == 0 and out_idx == 1:
                ax.legend(loc="best", fontsize=8)
    fig.savefig(FIG_ROOT / "polymer_change_tail_traces.png", dpi=220)
    plt.close(fig)


def plot_performance_bars(summary_rows: list[dict]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    order = ["baseline", "legacy", "refreshed"]
    colors = {"baseline": "#444444", "legacy": "#3A7CA5", "refreshed": "#D66A4E"}

    for row_idx, family in enumerate(("residual", "matrix")):
        rows = [row for row in summary_rows if row["family"] == family or row["family"] == "baseline"]
        rows = sorted(rows, key=lambda row: order.index(row["variant"]))
        x = np.arange(len(rows))
        axs[row_idx, 0].bar(x, [row["phys_mae_tail_mean"] for row in rows], color=[colors[row["variant"]] for row in rows])
        axs[row_idx, 0].set_title(f"{family.title()} tail physical MAE (mean across outputs)")
        axs[row_idx, 0].set_xticks(x)
        axs[row_idx, 0].set_xticklabels([row["variant"] for row in rows])
        axs[row_idx, 0].set_ylabel("MAE")

        axs[row_idx, 1].bar(x, [row["reward_final20_mean"] for row in rows], color=[colors[row["variant"]] for row in rows])
        axs[row_idx, 1].set_title(f"{family.title()} final-20 sub-episode reward")
        axs[row_idx, 1].set_xticks(x)
        axs[row_idx, 1].set_xticklabels([row["variant"] for row in rows])
        axs[row_idx, 1].set_ylabel("avg reward")

    fig.savefig(FIG_ROOT / "polymer_change_performance_bars.png", dpi=220)
    plt.close(fig)


def plot_state_conditioning(state_rows: list[dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    colors = {"legacy actual": "#3A7CA5", "refreshed actual": "#D66A4E", "refreshed fixed counterfactual": "#7B8C9B"}
    families = ["residual", "matrix"]
    labels = ["legacy actual", "refreshed actual", "refreshed fixed counterfactual"]

    for ax, family in zip(axs, families):
        rows = [row for row in state_rows if row["family"] == family]
        x = np.arange(len(labels))
        full_vals = [next(row for row in rows if row["view"] == label)["full_span_med"] for label in labels]
        late_vals = [next(row for row in rows if row["view"] == label)["late_span_med"] for label in labels]
        width = 0.35
        ax.bar(x - width / 2, full_vals, width=width, color=[colors[label] for label in labels], label="full 5-95% span")
        ax.bar(x + width / 2, late_vals, width=width, color=[colors[label] for label in labels], alpha=0.45, label="late-window 5-95% span")
        ax.set_title(f"{family.title()} policy-visible physical xhat span")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel("median span")
        ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG_ROOT / "polymer_change_state_conditioning.png", dpi=220)
    plt.close(fig)


def plot_feature_diagnostics(summary_rows: list[dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    residual_rows = [row for row in summary_rows if row["family"] == "residual" and row["variant"] in {"legacy", "refreshed"}]
    residual_rows = sorted(residual_rows, key=lambda row: row["variant"])
    x = np.arange(len(residual_rows))
    axs[0].bar(x - 0.22, [row["tracking_error_exact3_mean"] for row in residual_rows], width=0.22, label="tracking exact |3| frac")
    axs[0].bar(x, [row["innovation_exact3_mean"] for row in residual_rows], width=0.22, label="innovation exact |3| frac")
    axs[0].bar(x + 0.22, [row["rho_eq1_frac"] for row in residual_rows], width=0.22, label="rho=1 frac")
    axs[0].plot(x + 0.22, [row["deadband_frac"] for row in residual_rows], color="black", marker="o", linewidth=1.2, label="deadband frac")
    axs[0].set_title("Residual supervisory diagnostics")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([row["variant"] for row in residual_rows])
    axs[0].set_ylabel("fraction")
    axs[0].legend(fontsize=8, loc="upper right")

    matrix_rows = [row for row in summary_rows if row["family"] == "matrix" and row["variant"] in {"legacy", "refreshed"}]
    matrix_rows = sorted(matrix_rows, key=lambda row: row["variant"])
    x = np.arange(len(matrix_rows))
    axs[1].bar(x - 0.18, [row["tracking_error_p99_mean"] for row in matrix_rows], width=0.18, label="tracking p99 |transformed|")
    axs[1].bar(x, [row["innovation_p99_mean"] for row in matrix_rows], width=0.18, label="innovation p99 |transformed|")
    axs[1].bar(x + 0.18, [row["tracking_error_raw_p99_mean"] for row in matrix_rows], width=0.18, label="tracking p99 |raw|")
    axs[1].set_title("Matrix mismatch-feature dynamic range")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([row["variant"] for row in matrix_rows])
    axs[1].set_ylabel("magnitude")
    axs[1].legend(fontsize=8, loc="upper right")
    fig.savefig(FIG_ROOT / "polymer_change_feature_diagnostics.png", dpi=220)
    plt.close(fig)


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def pct_change(new_value: float, old_value: float) -> str:
    if abs(old_value) < 1e-12:
        return "n/a"
    return f"{100.0 * (new_value - old_value) / old_value:+.1f}%"


def write_markdown(
    config_rows: list[dict],
    summary_rows: list[dict],
    state_rows: list[dict],
) -> None:
    baseline = next(row for row in summary_rows if row["family"] == "baseline")
    residual_legacy = next(row for row in summary_rows if row["family"] == "residual" and row["variant"] == "legacy")
    residual_refreshed = next(row for row in summary_rows if row["family"] == "residual" and row["variant"] == "refreshed")
    matrix_legacy = next(row for row in summary_rows if row["family"] == "matrix" and row["variant"] == "legacy")
    matrix_refreshed = next(row for row in summary_rows if row["family"] == "matrix" and row["variant"] == "refreshed")

    residual_state = [row for row in state_rows if row["family"] == "residual"]
    matrix_state = [row for row in state_rows if row["family"] == "matrix"]

    lines = [
        "# Polymer Residual And Matrix Change-Impact Report",
        "",
        "Date: 2026-04-20",
        "",
        "This report compares the latest polymer residual and matrix runs against the closest pre-change reference runs and the disturbance baseline MPC.",
        "",
        "Important scope note:",
        "",
        "- the latest saved polymer runs analyzed here were generated before the later observer-default rollback",
        "- so the refreshed runs in this report still use `observer_update_alignment=\"current_measurement_corrector\"`",
        "- the report therefore answers the question \"did the changes affect the runs you actually executed?\" rather than \"what would happen under the final defaults after rollback?\"",
        "",
        "## Runs Compared",
        "",
        "| Family | Variant | Saved run |",
        "| --- | --- | --- |",
    ]
    for row in config_rows:
        lines.append(f"| {row['family']} | {row['variant']} | `{row['run_path']}` |")

    lines += [
        "",
        "## Config Comparison",
        "",
        "| Family | Variant | base_state_norm_mode | mismatch_feature_transform_mode | observer_update_alignment | rho_mapping_mode | deadband |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in config_rows:
        lines.append(
            f"| {row['family']} | {row['variant']} | `{row['base_state_norm_mode']}` | `{row['mismatch_feature_transform_mode']}` | "
            f"`{row['observer_update_alignment']}` | `{row['rho_mapping_mode']}` | `{row['residual_zero_deadband_enabled']}` |"
        )

    lines += [
        "",
        "## Performance Summary",
        "",
        "| Family | Variant | Tail physical MAE mean | Tail scaled MAE mean | Final-20 avg reward | Tail MAE vs baseline |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        if row["family"] == "baseline":
            lines.append(
                f"| baseline | baseline | {fmt(row['phys_mae_tail_mean'])} | {fmt(row['scaled_mae_tail_mean'])} | {fmt(row['reward_final20_mean'])} | n/a |"
            )
        else:
            lines.append(
                f"| {row['family']} | {row['variant']} | {fmt(row['phys_mae_tail_mean'])} | {fmt(row['scaled_mae_tail_mean'])} | "
                f"{fmt(row['reward_final20_mean'])} | {fmt(row['tail_phys_mae_delta_vs_baseline'])} |"
            )

    lines += [
        "",
        "## State-Conditioning Summary",
        "",
        "| Family | View | full-span median | late-span median | late/full ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in state_rows:
        lines.append(
            f"| {row['family']} | {row['view']} | {fmt(row['full_span_med'])} | {fmt(row['late_span_med'])} | {fmt(row['late_full_ratio_med'])} |"
        )

    lines += [
        "",
        "## Main Findings",
        "",
        f"1. The refreshed residual run did make a material difference. Tail physical MAE mean improved from `{fmt(residual_legacy['phys_mae_tail_mean'])}` to `{fmt(residual_refreshed['phys_mae_tail_mean'])}` ({pct_change(residual_refreshed['phys_mae_tail_mean'], residual_legacy['phys_mae_tail_mean'])}), and final-20 reward improved from `{fmt(residual_legacy['reward_final20_mean'])}` to `{fmt(residual_refreshed['reward_final20_mean'])}` ({pct_change(residual_refreshed['reward_final20_mean'], residual_legacy['reward_final20_mean'])}). The refreshed run also reduced `rho=1` fraction from `{fmt(residual_legacy['rho_eq1_frac'])}` to `{fmt(residual_refreshed['rho_eq1_frac'])}` and activated the new deadband on `{fmt(residual_refreshed['deadband_frac'])}` of steps.",
        f"2. The refreshed matrix run also changed behavior, but the gain is smaller and more mixed. Tail physical MAE mean improved from `{fmt(matrix_legacy['phys_mae_tail_mean'])}` to `{fmt(matrix_refreshed['phys_mae_tail_mean'])}` ({pct_change(matrix_refreshed['phys_mae_tail_mean'], matrix_legacy['phys_mae_tail_mean'])}), and final-20 reward improved from `{fmt(matrix_legacy['reward_final20_mean'])}` to `{fmt(matrix_refreshed['reward_final20_mean'])}` ({pct_change(matrix_refreshed['reward_final20_mean'], matrix_legacy['reward_final20_mean'])}). Output 1 improved more clearly than output 2.",
        f"3. The new polymer state conditioning definitely changed what the policy saw. On the refreshed residual trajectory, the median full-span of the policy-visible physical `xhat` block is `{fmt(next(row for row in residual_state if row['view'] == 'refreshed actual')['full_span_med'])}` under running normalization versus `{fmt(next(row for row in residual_state if row['view'] == 'refreshed fixed counterfactual')['full_span_med'])}` under a fixed-minmax counterfactual. On the refreshed matrix trajectory, the same comparison is `{fmt(next(row for row in matrix_state if row['view'] == 'refreshed actual')['full_span_med'])}` versus `{fmt(next(row for row in matrix_state if row['view'] == 'refreshed fixed counterfactual')['full_span_med'])}`.",
        f"4. The residual mismatch features no longer pile up at the hard clip. In the legacy residual run, transformed tracking hit exact `|3|` on `{fmt(residual_legacy['tracking_error_exact3_mean'])}` of samples on average across outputs. In the refreshed residual run, exact-`|3|` mass is essentially zero while the raw tracking p99 is still very large at `{fmt(residual_refreshed['tracking_error_raw_p99_mean'])}`. That means the new transform is exposing severity information instead of flattening it.",
        f"5. The matrix refreshed run also exposes much more mismatch-feature dynamic range than the legacy run. The mean transformed tracking p99 rises from `{fmt(matrix_legacy['tracking_error_p99_mean'])}` to `{fmt(matrix_refreshed['tracking_error_p99_mean'])}`, and the raw tracking p99 in the refreshed run is `{fmt(matrix_refreshed['tracking_error_raw_p99_mean'])}`. So the changes absolutely affected the matrix policy input, even though the closed-loop improvement is modest rather than dramatic.",
        "",
        "## Figures",
        "",
        "- [Reward curves](./polymer_change_impact/figures/polymer_change_reward_curves.png)",
        "- [Tail traces](./polymer_change_impact/figures/polymer_change_tail_traces.png)",
        "- [Performance bars](./polymer_change_impact/figures/polymer_change_performance_bars.png)",
        "- [State conditioning](./polymer_change_impact/figures/polymer_change_state_conditioning.png)",
        "- [Feature diagnostics](./polymer_change_impact/figures/polymer_change_feature_diagnostics.png)",
        "",
        "## Data Tables",
        "",
        "- [Config summary](./polymer_change_impact/data/config_summary.csv)",
        "- [Performance summary](./polymer_change_impact/data/performance_summary.csv)",
        "- [State conditioning summary](./polymer_change_impact/data/state_conditioning_summary.csv)",
    ]

    (REPO_ROOT / "report" / "polymer_change_impact_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    baseline_bundle = load_pickle(BASELINE_PATH)
    minmax = load_pickle(POLYMER_MINMAX_PATH)
    min_s = np.asarray(minmax["min_s"], float)
    max_s = np.asarray(minmax["max_s"], float)

    runs: dict[str, dict] = {}
    config_rows: list[dict] = [
        {
            "family": "baseline",
            "variant": "baseline",
            "run_path": str(BASELINE_PATH.relative_to(REPO_ROOT)),
            "base_state_norm_mode": "n/a",
            "mismatch_feature_transform_mode": "n/a",
            "observer_update_alignment": "n/a",
            "rho_mapping_mode": "n/a",
            "residual_zero_deadband_enabled": "n/a",
        }
    ]
    summary_rows: list[dict] = [
        {
            "family": "baseline",
            "variant": "baseline",
            "run_path": str(BASELINE_PATH.relative_to(REPO_ROOT)),
            **compute_metrics(baseline_bundle),
            "innovation_p99_mean": np.nan,
            "tracking_error_p99_mean": np.nan,
            "innovation_raw_p99_mean": np.nan,
            "tracking_error_raw_p99_mean": np.nan,
            "innovation_exact3_mean": np.nan,
            "tracking_error_exact3_mean": np.nan,
        }
    ]
    state_rows: list[dict] = []

    for spec in RUN_SPECS:
        bundle = load_pickle(spec.path)
        variant = "refreshed" if spec.is_refreshed else "legacy"
        key = f"{spec.family}_{variant}"
        runs[key] = {"spec": spec, "bundle": bundle}

        metrics = compute_metrics(bundle, baseline_bundle=baseline_bundle)
        metrics["innovation_p99_mean"] = nanmean_pair([metrics["innovation_p99_out1"], metrics["innovation_p99_out2"]])
        metrics["tracking_error_p99_mean"] = nanmean_pair([metrics["tracking_error_p99_out1"], metrics["tracking_error_p99_out2"]])
        metrics["innovation_raw_p99_mean"] = nanmean_pair([metrics["innovation_raw_p99_out1"], metrics["innovation_raw_p99_out2"]])
        metrics["tracking_error_raw_p99_mean"] = nanmean_pair([metrics["tracking_error_raw_p99_out1"], metrics["tracking_error_raw_p99_out2"]])
        metrics["innovation_exact3_mean"] = nanmean_pair([metrics["innovation_exact3_out1"], metrics["innovation_exact3_out2"]])
        metrics["tracking_error_exact3_mean"] = nanmean_pair([metrics["tracking_error_exact3_out1"], metrics["tracking_error_exact3_out2"]])
        summary_rows.append(
            {
                "family": spec.family,
                "variant": variant,
                "run_path": str(spec.path.relative_to(REPO_ROOT)),
                **metrics,
            }
        )

        config_rows.append(
            {
                "family": spec.family,
                "variant": variant,
                "run_path": str(spec.path.relative_to(REPO_ROOT)),
                "base_state_norm_mode": str(bundle.get("base_state_norm_mode", "legacy_fixed_minmax")),
                "mismatch_feature_transform_mode": str(bundle.get("mismatch_feature_transform_mode", "legacy_hard_clip")),
                "observer_update_alignment": str(bundle.get("observer_update_alignment", "legacy_previous_measurement")),
                "rho_mapping_mode": str(bundle.get("rho_mapping_mode", "legacy_clipped_linear")),
                "residual_zero_deadband_enabled": bool(bundle.get("residual_zero_deadband_enabled", False)),
            }
        )

        actual_state = state_span_summary(bundle, min_s=min_s, max_s=max_s, mode_override=bundle.get("base_state_norm_mode", "fixed_minmax"))
        state_rows.append({"family": spec.family, "view": f"{variant} actual", **actual_state})
        if spec.is_refreshed:
            fixed_counterfactual = state_span_summary(bundle, min_s=min_s, max_s=max_s, mode_override="fixed_minmax")
            state_rows.append({"family": spec.family, "view": "refreshed fixed counterfactual", **fixed_counterfactual})

    save_csv(
        DATA_ROOT / "config_summary.csv",
        config_rows,
        ["family", "variant", "run_path", "base_state_norm_mode", "mismatch_feature_transform_mode", "observer_update_alignment", "rho_mapping_mode", "residual_zero_deadband_enabled"],
    )
    save_csv(
        DATA_ROOT / "performance_summary.csv",
        summary_rows,
        [
            "family",
            "variant",
            "run_path",
            "scaled_mae_out1",
            "scaled_mae_out2",
            "scaled_mae_mean",
            "scaled_mae_tail_mean",
            "phys_mae_out1",
            "phys_mae_out2",
            "phys_mae_mean",
            "phys_mae_tail_out1",
            "phys_mae_tail_out2",
            "phys_mae_tail_mean",
            "u_move_mean",
            "u_move_tail",
            "reward_final20_mean",
            "reward_best",
            "reward_last",
            "innovation_p99_mean",
            "tracking_error_p99_mean",
            "innovation_raw_p99_mean",
            "tracking_error_raw_p99_mean",
            "innovation_exact3_mean",
            "tracking_error_exact3_mean",
            "rho_mean",
            "rho_eq1_frac",
            "deadband_frac",
            "residual_norm_mean",
            "residual_norm_tail",
            "tail_scaled_mae_delta_vs_baseline",
            "tail_phys_mae_delta_vs_baseline",
        ],
    )
    save_csv(
        DATA_ROOT / "state_conditioning_summary.csv",
        state_rows,
        ["family", "view", "full_span_med", "late_span_med", "late_full_ratio_med"],
    )

    plot_reward_curves(baseline_bundle=baseline_bundle, runs=runs)
    plot_tail_traces(baseline_bundle=baseline_bundle, runs=runs)
    plot_performance_bars(summary_rows=summary_rows)
    plot_state_conditioning(state_rows=state_rows)
    plot_feature_diagnostics(summary_rows=summary_rows)
    write_markdown(config_rows=config_rows, summary_rows=summary_rows, state_rows=state_rows)


if __name__ == "__main__":
    main()
