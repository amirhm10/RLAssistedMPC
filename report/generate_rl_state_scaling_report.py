from __future__ import annotations

import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from systems.distillation.notebook_params import get_distillation_notebook_defaults
from systems.polymer.notebook_params import get_polymer_notebook_defaults
from utils.helpers import apply_min_max
from utils.observation_conditioning import transform_mismatch_feature
from utils.residual_authority import project_residual_action


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "report" / "rl_state_scaling_diagnostics"
FIG_ROOT = REPORT_ROOT / "figures"
DATA_ROOT = REPORT_ROOT / "data"

DISTILLATION_BASELINE_PATH = REPO_ROOT / r"Distillation/Results/distillation_baseline_disturb_fluctuation_unified/20260413_085608/input_data.pkl"
DISTILLATION_LATEST_RESIDUAL_TD3_PATH = REPO_ROOT / r"Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260420_180906/input_data.pkl"
DISTILLATION_PREVIOUS_RESIDUAL_TD3_PATH = REPO_ROOT / r"Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260417_181426/input_data.pkl"
DISTILLATION_RESIDUAL_SAC_PATH = REPO_ROOT / r"Distillation/Results/distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified/20260415_191909/input_data.pkl"
POLYMER_RESIDUAL_TD3_PATH = REPO_ROOT / r"Polymer/Results/td3_residual_disturb/20260413_004620/input_data.pkl"


@dataclass(frozen=True)
class RunSpec:
    system: str
    label: str
    rel_path: str
    minmax_rel_path: str

    @property
    def run_path(self) -> Path:
        return REPO_ROOT / self.rel_path

    @property
    def minmax_path(self) -> Path:
        return REPO_ROOT / self.minmax_rel_path


RUN_SPECS = (
    RunSpec(
        system="distillation",
        label="Horizon DDQN\nstandard",
        rel_path=r"Distillation/Results/distillation_horizon_disturb_fluctuation_standard_unified/20260416_192434/input_data.pkl",
        minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="distillation",
        label="Matrix SAC\nstandard",
        rel_path=r"Distillation/Results/distillation_matrix_sac_disturb_fluctuation_standard_unified/20260415_104840/input_data.pkl",
        minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="distillation",
        label="Weights SAC\nstandard",
        rel_path=r"Distillation/Results/distillation_weights_sac_disturb_fluctuation_standard_unified/20260416_192555/input_data.pkl",
        minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="distillation",
        label="Residual TD3\nmismatch",
        rel_path=r"Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260420_180906/input_data.pkl",
        minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="distillation",
        label="Residual SAC\nmismatch",
        rel_path=r"Distillation/Results/distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified/20260415_191909/input_data.pkl",
        minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="distillation",
        label="Reid TD3\nmismatch",
        rel_path=r"Distillation/Results/distillation_reidentification_td3_disturb_fluctuation_mismatch_unified/20260418_115020/input_data.pkl",
        minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="polymer",
        label="Horizon DDQN\nmismatch",
        rel_path=r"Polymer/Results/horizon_disturb_unified/20260407_175020/input_data.pkl",
        minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="polymer",
        label="Dueling DDQN\nstandard",
        rel_path=r"Polymer/Results/dueling_horizon_disturb_unified/20260417_035217/input_data.pkl",
        minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="polymer",
        label="Matrix TD3\nmismatch",
        rel_path=r"Polymer/Results/td3_multipliers_disturb/20260411_011134/input_data.pkl",
        minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="polymer",
        label="Weights TD3\nstandard",
        rel_path=r"Polymer/Results/td3_weights_disturb/20260417_031734/input_data.pkl",
        minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="polymer",
        label="Residual TD3\nmismatch",
        rel_path=r"Polymer/Results/td3_residual_disturb/20260413_004620/input_data.pkl",
        minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
    ),
    RunSpec(
        system="polymer",
        label="Reid TD3\nmismatch",
        rel_path=r"Polymer/Results/td3_reidentification_disturb/20260415_120803/input_data.pkl",
        minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
    ),
)


DISTILLATION_STATE_LABELS = [f"x{i}" for i in range(1, 6)] + ["d1", "d2"]
POLYMER_STATE_LABELS = [f"x{i}" for i in range(1, 8)] + ["d1", "d2"]


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def setpoint_scaled_to_physical(bundle: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle["n_inputs"])
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp = np.asarray(bundle["y_sp"], float)
    return (y_sp + y_ss_scaled) * (data_max[n_inputs:] - data_min[n_inputs:]) + data_min[n_inputs:]


def get_last_eval_window(bundle: dict) -> tuple[int, int]:
    test_train = bundle.get("test_train_dict")
    n_steps = int(bundle["nFE"])
    if isinstance(test_train, dict) and test_train:
        eval_starts = sorted(int(k) for k, v in test_train.items() if bool(v))
        if eval_starts:
            start = eval_starts[-1]
            all_starts = sorted(int(k) for k in test_train.keys())
            later = [step for step in all_starts if step > start]
            end = later[0] if later else n_steps
            return start, end
    return max(0, n_steps - 400), n_steps


def max_abs_tracking_per_step(tracking: np.ndarray) -> np.ndarray:
    tracking = np.asarray(tracking, float)
    return np.max(np.abs(tracking), axis=1)


def detect_setpoint_change_segments(y_sp: np.ndarray) -> list[tuple[int, int]]:
    change_idx = np.flatnonzero(np.any(np.abs(np.diff(y_sp, axis=0)) > 1e-12, axis=1)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(y_sp)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s > 5]


def late_segment_mask(y_sp: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(y_sp), dtype=bool)
    for start, end in detect_setpoint_change_segments(y_sp):
        segment_len = end - start
        window = max(10, min(30, segment_len // 4))
        mask[max(start, end - window) : end] = True
    return mask


def scale_states(xhatdhat: np.ndarray, min_s: np.ndarray, max_s: np.ndarray) -> np.ndarray:
    span = np.maximum(max_s - min_s, 1e-12)
    return 2.0 * ((xhatdhat.T - min_s) / span) - 1.0


def quantile_span(arr: np.ndarray) -> np.ndarray:
    return np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)


def segment_peak_to_late(arr_by_time: np.ndarray, y_sp: np.ndarray) -> np.ndarray:
    ratios = []
    for start, end in detect_setpoint_change_segments(y_sp):
        segment_len = end - start
        early_window = max(10, min(30, segment_len // 4))
        late_window = max(10, min(30, segment_len // 4))
        early = np.abs(arr_by_time[start : min(end, start + early_window), :])
        late = np.abs(arr_by_time[max(start, end - late_window) : end, :])
        early_peak = np.max(early, axis=0)
        late_level = np.median(late, axis=0)
        ratios.append(early_peak / np.maximum(late_level, 1e-12))
    return np.median(np.vstack(ratios), axis=0)


def save_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_heatmap(ax, data: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    return im


def plot_system_heatmaps(system: str, rows: list[dict], state_labels: list[str]) -> None:
    run_labels = [row["label"] for row in rows]
    full_span = np.vstack([row["full_span_ratio"] for row in rows])
    late_ratio = np.vstack([row["late_full_ratio"] for row in rows])
    outside_frac = np.vstack([row["outside_frac"] for row in rows])

    fig, axs = plt.subplots(3, 1, figsize=(1.3 * len(state_labels) + 4.5, 1.3 * len(run_labels) + 6.5), constrained_layout=True)
    add_heatmap(axs[0], full_span, run_labels, state_labels, f"{system.title()}: 5-95% span / min-max width", cmap="YlGnBu", vmin=0.0, vmax=max(1.2, float(np.nanmax(full_span))))
    add_heatmap(axs[1], late_ratio, run_labels, state_labels, f"{system.title()}: late-window span / full span", cmap="viridis", vmin=0.0, vmax=max(1.2, float(np.nanmax(late_ratio))))
    add_heatmap(axs[2], outside_frac, run_labels, state_labels, f"{system.title()}: fraction outside saved min-max box", cmap="magma", vmin=0.0, vmax=max(0.05, float(np.nanmax(outside_frac))))
    fig.suptitle(f"{system.title()} RL-state scaling diagnostics across saved runs", fontsize=14)
    fig.savefig(FIG_ROOT / f"{system}_state_scaling_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_mismatch_summary(rows: list[dict]) -> None:
    mismatch_rows = [row for row in rows if row["state_mode"] == "mismatch"]
    systems = ["distillation", "polymer"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    for ax, system in zip(axs, systems):
        sys_rows = [row for row in mismatch_rows if row["system"] == system]
        labels = [row["label"].replace("\n", " ") for row in sys_rows]
        x = np.arange(len(labels))
        tracking = np.array([row["tracking_clip_frac"] for row in sys_rows], dtype=float)
        innovation = np.array([row["innovation_clip_frac"] for row in sys_rows], dtype=float)
        rho = np.array([row["rho_one_frac"] if row["rho_one_frac"] is not None else np.nan for row in sys_rows], dtype=float)
        width = 0.25
        ax.bar(x - width, tracking, width, label="tracking clip frac")
        ax.bar(x, innovation, width, label="innovation clip frac")
        if np.any(np.isfinite(rho)):
            ax.bar(x + width, np.nan_to_num(rho, nan=0.0), width, label="rho=1 frac")
        ax.set_title(f"{system.title()} mismatch-feature saturation")
        ax.set_ylabel("Fraction of steps")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, max(1.0, float(np.nanmax([tracking.max(), innovation.max(), np.nanmax(rho) if np.any(np.isfinite(rho)) else 0.0])) * 1.08))
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(FIG_ROOT / "mismatch_feature_saturation.png", dpi=220)
    plt.close(fig)


def plot_representative_trace(spec: RunSpec, label_prefix: str, state_labels: list[str]) -> None:
    bundle = load_pickle(spec.run_path)
    minmax = load_pickle(spec.minmax_path)
    xhatdhat = np.asarray(bundle["xhatdhat"][:, :-1], float)
    y_sp = np.asarray(bundle["y_sp"], float)
    scaled = scale_states(xhatdhat, np.asarray(minmax["min_s"], float), np.asarray(minmax["max_s"], float))
    n_outputs = int(bundle["n_outputs"])
    n_phys = xhatdhat.shape[0] - n_outputs

    window = min(1200, scaled.shape[0])
    time = np.arange(window)
    output_labels = bundle["system_metadata"]["output_labels"]

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    axs[0].plot(time, scaled[:window, :n_phys], linewidth=1.0)
    axs[0].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    axs[0].axhline(-1.0, color="k", linestyle="--", linewidth=0.8)
    axs[0].set_ylabel("scaled xhat")
    axs[0].set_title(f"{label_prefix}: representative scaled observer states")
    axs[0].legend(state_labels[:n_phys], ncols=min(n_phys, 5), fontsize=8, loc="upper right")

    axs[1].plot(time, scaled[:window, n_phys:], linewidth=1.0)
    axs[1].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    axs[1].axhline(-1.0, color="k", linestyle="--", linewidth=0.8)
    axs[1].set_ylabel("scaled dhat")
    axs[1].legend(state_labels[n_phys:], ncols=n_outputs, fontsize=8, loc="upper right")

    innovation = bundle.get("innovation_log")
    tracking = bundle.get("tracking_error_log")
    if innovation is not None:
        axs[2].plot(time, np.asarray(innovation[:window], float), linewidth=1.0)
        axs[2].axhline(3.0, color="k", linestyle="--", linewidth=0.8)
        axs[2].axhline(-3.0, color="k", linestyle="--", linewidth=0.8)
        axs[2].set_ylabel("innovation")
        axs[2].legend(output_labels, fontsize=8, loc="upper right")
    if tracking is not None:
        axs[3].plot(time, np.asarray(tracking[:window], float), linewidth=1.0)
        axs[3].axhline(3.0, color="k", linestyle="--", linewidth=0.8)
        axs[3].axhline(-3.0, color="k", linestyle="--", linewidth=0.8)
        axs[3].set_ylabel("tracking err")
        axs[3].legend(output_labels, fontsize=8, loc="upper right")
    axs[3].set_xlabel("Step")
    fig.savefig(FIG_ROOT / f"{spec.system}_{label_prefix.lower().replace(' ', '_')}_trace.png", dpi=220)
    plt.close(fig)


def plot_minmax_widths() -> None:
    distill_minmax = load_pickle(REPO_ROOT / r"Distillation/Data/min_max_states.pickle")
    polymer_minmax = load_pickle(REPO_ROOT / r"Polymer/Data/min_max_states.pickle")
    distill_width = np.asarray(distill_minmax["max_s"], float) - np.asarray(distill_minmax["min_s"], float)
    polymer_width = np.asarray(polymer_minmax["max_s"], float) - np.asarray(polymer_minmax["min_s"], float)

    fig, axs = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axs[0].bar(np.arange(distill_width.size), distill_width, color="#3A7CA5")
    axs[0].set_title("Distillation saved min-max widths by augmented state")
    axs[0].set_xticks(np.arange(distill_width.size))
    axs[0].set_xticklabels(DISTILLATION_STATE_LABELS)
    axs[0].set_ylabel("width")

    axs[1].bar(np.arange(polymer_width.size), polymer_width, color="#D66A4E")
    axs[1].set_title("Polymer saved min-max widths by augmented state")
    axs[1].set_xticks(np.arange(polymer_width.size))
    axs[1].set_xticklabels(POLYMER_STATE_LABELS)
    axs[1].set_ylabel("width")
    axs[1].set_xlabel("Augmented state")
    fig.savefig(FIG_ROOT / "saved_minmax_widths.png", dpi=220)
    plt.close(fig)


def compute_y_prev_scaled(bundle: dict) -> np.ndarray:
    y = np.asarray(bundle["y"][:-1], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle["n_inputs"])
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    return (y - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12) - y_ss_scaled


def reconstruct_raw_mismatch_features(bundle: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    if bundle.get("innovation_raw_log") is not None and bundle.get("tracking_error_raw_log") is not None:
        return (
            np.asarray(bundle["innovation_raw_log"], float),
            np.asarray(bundle["tracking_error_raw_log"], float),
        )
    if bundle.get("innovation_scale_ref") is None or bundle.get("tracking_scale_log") is None:
        return None, None
    y_prev_scaled = compute_y_prev_scaled(bundle)
    yhat = np.asarray(bundle["yhat"], float).T
    y_sp = np.asarray(bundle["y_sp"], float)
    innovation_scale_ref = np.asarray(bundle["innovation_scale_ref"], float).reshape(1, -1)
    tracking_scale_now = np.asarray(bundle["tracking_scale_log"], float)
    innovation_raw = (y_prev_scaled - yhat) / np.maximum(innovation_scale_ref, 1e-12)
    tracking_raw = (y_prev_scaled - y_sp) / np.maximum(tracking_scale_now, 1e-12)
    return innovation_raw, tracking_raw


def running_vecnormalize_feature(values: np.ndarray, clip_obs: float = 10.0, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, float).reshape(-1)
    normed = np.zeros_like(values, dtype=float)
    mean = 0.0
    m2 = 0.0
    count = 0.0
    for idx, x in enumerate(values):
        count += 1.0
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
        var = m2 / max(count, 1.0)
        normed[idx] = np.clip((x - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs)
    return normed


def transform_feature(values: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(values, float).reshape(-1)
    if kind == "hard_clip":
        return transform_mismatch_feature(values, mode="hard_clip", mismatch_clip=3.0)
    if kind == "soft_tanh":
        return transform_mismatch_feature(values, mode="soft_tanh", tanh_scale=3.0)
    if kind == "signed_log":
        return transform_mismatch_feature(values, mode="signed_log")
    if kind == "vecnorm":
        return running_vecnormalize_feature(values, clip_obs=10.0)
    raise ValueError(f"Unknown transform kind: {kind}")


def plot_vecnormalize_comparison(cache: list[dict]) -> None:
    selected = [
        next(item for item in cache if item["spec"].system == "distillation" and "Residual TD3" in item["spec"].label),
        next(item for item in cache if item["spec"].system == "polymer" and "Residual TD3" in item["spec"].label),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for row_idx, item in enumerate(selected):
        spec = item["spec"]
        bundle = item["bundle"]
        minmax = item["minmax"]
        xhatdhat = np.asarray(bundle["xhatdhat"][:, :-1], float)
        y_sp = np.asarray(bundle["y_sp"], float)
        scaled = scale_states(xhatdhat, np.asarray(minmax["min_s"], float), np.asarray(minmax["max_s"], float))
        n_outputs = int(bundle["n_outputs"])
        n_phys = xhatdhat.shape[0] - n_outputs
        phys_ratios = segment_peak_to_late(xhatdhat[:n_phys, :].T, y_sp)
        state_idx = int(np.argmax(phys_ratios))
        fixed_scaled = scaled[:, state_idx]
        vecnorm_scaled = running_vecnormalize_feature(xhatdhat[state_idx, :], clip_obs=10.0)

        full_steps = min(1200, fixed_scaled.shape[0])
        late_mask = late_segment_mask(y_sp)
        late_indices = np.flatnonzero(late_mask)
        if late_indices.size == 0:
            late_start = max(0, fixed_scaled.shape[0] - 200)
            late_end = fixed_scaled.shape[0]
        else:
            late_end = int(late_indices[-1]) + 1
            late_start = max(0, late_end - 200)

        axs[row_idx, 0].plot(np.arange(full_steps), xhatdhat[state_idx, :full_steps], color="#4C5866", linewidth=1.0)
        axs[row_idx, 0].set_title(f"{spec.system.title()}: raw state {state_idx + 1}")
        axs[row_idx, 0].set_ylabel("raw xhat")
        axs[row_idx, 0].set_xlabel("Step")

        axs[row_idx, 1].plot(np.arange(full_steps), fixed_scaled[:full_steps], label="fixed min-max", linewidth=1.1)
        axs[row_idx, 1].plot(np.arange(full_steps), vecnorm_scaled[:full_steps], label="VecNormalize-like", linewidth=1.1)
        axs[row_idx, 1].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
        axs[row_idx, 1].axhline(-1.0, color="k", linestyle="--", linewidth=0.8)
        axs[row_idx, 1].set_title(f"{spec.system.title()}: transformed state {state_idx + 1}")
        axs[row_idx, 1].set_ylabel("normalized value")
        axs[row_idx, 1].set_xlabel("Step")
        axs[row_idx, 1].legend(fontsize=8, loc="upper right")

        zoom_x = np.arange(late_start, late_end)
        axs[row_idx, 2].plot(zoom_x, fixed_scaled[late_start:late_end], label="fixed min-max", linewidth=1.1)
        axs[row_idx, 2].plot(zoom_x, vecnorm_scaled[late_start:late_end], label="VecNormalize-like", linewidth=1.1)
        axs[row_idx, 2].set_title(
            f"{spec.system.title()}: late-window zoom\nfixed late/full={np.median(quantile_span(scaled[late_mask, :n_phys]) / np.maximum(quantile_span(scaled[:, :n_phys]), 1e-12)):.3f}"
        )
        axs[row_idx, 2].set_ylabel("normalized value")
        axs[row_idx, 2].set_xlabel("Step")
        axs[row_idx, 2].legend(fontsize=8, loc="upper right")
    fig.suptitle("VecNormalize-style running normalization vs fixed min-max on representative observer states", fontsize=14)
    fig.savefig(FIG_ROOT / "vecnormalize_project_comparison.png", dpi=220)
    plt.close(fig)


def plot_normalization_sensitivity_by_state(cache: list[dict]) -> None:
    selected = [
        next(item for item in cache if item["spec"].run_path == DISTILLATION_LATEST_RESIDUAL_TD3_PATH),
        next(item for item in cache if item["spec"].run_path == POLYMER_RESIDUAL_TD3_PATH),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col_idx, item in enumerate(selected):
        bundle = item["bundle"]
        minmax = item["minmax"]
        system = item["spec"].system
        xhatdhat = np.asarray(bundle["xhatdhat"][:, :-1], float)
        y_sp = np.asarray(bundle["y_sp"], float)
        n_outputs = int(bundle["n_outputs"])
        n_phys = xhatdhat.shape[0] - n_outputs
        labels = (DISTILLATION_STATE_LABELS if system == "distillation" else POLYMER_STATE_LABELS)[:n_phys]
        widths = np.asarray(minmax["max_s"], float)[:n_phys] - np.asarray(minmax["min_s"], float)[:n_phys]
        late_mask = late_segment_mask(y_sp)
        late_std = np.std(xhatdhat[:n_phys, :][:, late_mask], axis=1)

        axs[0, col_idx].bar(np.arange(n_phys), widths, color="#4477AA")
        axs[0, col_idx].set_title(f"{system.title()}: saved physical-state widths")
        axs[0, col_idx].set_ylabel("saved width")
        axs[0, col_idx].set_xticks(np.arange(n_phys))
        axs[0, col_idx].set_xticklabels(labels)

        axs[1, col_idx].bar(np.arange(n_phys) - 0.18, 2.0 / np.maximum(widths, 1e-12), width=0.36, label="fixed min-max slope 2/W", color="#CC6677")
        axs[1, col_idx].bar(np.arange(n_phys) + 0.18, 1.0 / np.maximum(late_std, 1e-12), width=0.36, label="late z-score slope 1/sigma_late", color="#228833")
        axs[1, col_idx].set_title(f"{system.title()}: local normalization sensitivity")
        axs[1, col_idx].set_ylabel("normalized units per raw unit")
        axs[1, col_idx].set_yscale("log")
        axs[1, col_idx].set_xticks(np.arange(n_phys))
        axs[1, col_idx].set_xticklabels(labels)
        axs[1, col_idx].legend(fontsize=8, loc="upper right")
    fig.suptitle("Why wide saved min-max boxes flatten states and running z-scores do not", fontsize=14)
    fig.savefig(FIG_ROOT / "normalization_sensitivity_by_state.png", dpi=220)
    plt.close(fig)


def plot_raw_exceedance_and_transform_examples(raw_rows: list[dict]) -> None:
    by_system = {"distillation": [], "polymer": []}
    for row in raw_rows:
        by_system[row["system"]].append(row)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    thresholds = ("innovation_gt3_frac", "innovation_gt10_frac", "tracking_gt3_frac", "tracking_gt10_frac")
    threshold_labels = ["innov >3", "innov >10", "track >3", "track >10"]
    for ax, system in zip(axs, ("distillation", "polymer")):
        rows = by_system[system]
        labels = [row["label"].replace("\n", " ") for row in rows]
        x = np.arange(len(rows))
        width = 0.18
        for offset_idx, (key, label) in enumerate(zip(thresholds, threshold_labels)):
            ax.bar(x + (offset_idx - 1.5) * width, [row[key] for row in rows], width, label=label)
        ax.set_title(f"{system.title()}: raw mismatch-feature exceedance without hard clip")
        ax.set_ylabel("Fraction of entries")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, ncols=2, loc="upper right")
    fig.savefig(FIG_ROOT / "raw_mismatch_feature_exceedance.png", dpi=220)
    plt.close(fig)

    representative = {
        "distillation": next(row for row in raw_rows if row["label"].startswith("Residual TD3") and row["system"] == "distillation"),
        "polymer": next(row for row in raw_rows if row["label"].startswith("Residual TD3") and row["system"] == "polymer"),
    }
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row_idx, system in enumerate(("distillation", "polymer")):
        row = representative[system]
        for col_idx, (feature_name, title_stub) in enumerate((("innovation_raw", "Innovation"), ("tracking_raw", "Tracking error"))):
            arr = np.asarray(row[feature_name], float)
            channel = int(np.argmax(np.percentile(np.abs(arr), 99, axis=0)))
            raw = arr[:, channel]
            clipped = np.clip(raw, -3.0, 3.0)
            soft = 3.0 * np.tanh(raw / 3.0)
            steps = min(1200, raw.shape[0])
            ax = axs[row_idx, col_idx]
            ax.plot(np.arange(steps), raw[:steps], color="#A0A4A8", linewidth=0.9, label="raw")
            ax.plot(np.arange(steps), clipped[:steps], color="#C44E52", linewidth=1.0, label="hard clip ±3")
            ax.plot(np.arange(steps), soft[:steps], color="#4C72B0", linewidth=1.0, label="soft tanh squash")
            ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8)
            ax.axhline(-3.0, color="k", linestyle="--", linewidth=0.8)
            ax.set_title(f"{system.title()}: {title_stub.lower()} channel {channel + 1}")
            ax.set_ylabel("normalized feature")
            ax.set_xlabel("Step")
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Raw mismatch features vs hard clip and soft squash alternatives", fontsize=14)
    fig.savefig(FIG_ROOT / "raw_vs_clipped_transform_examples.png", dpi=220)
    plt.close(fig)


def plot_transform_curves() -> None:
    x = np.linspace(-15.0, 15.0, 2000)
    hard_clip = np.clip(x, -3.0, 3.0)
    soft_tanh = 3.0 * np.tanh(x / 3.0)
    soft_log = np.sign(x) * np.log1p(np.abs(x))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(x, x, label="no clip", linewidth=1.2)
    ax.plot(x, hard_clip, label="hard clip ±3", linewidth=1.2)
    ax.plot(x, soft_tanh, label="soft tanh squash", linewidth=1.2)
    ax.plot(x, soft_log, label="signed log1p", linewidth=1.2)
    ax.set_title("Candidate transforms for mismatch features")
    ax.set_xlabel("raw normalized innovation / tracking value")
    ax.set_ylabel("feature passed to policy")
    ax.legend()
    fig.savefig(FIG_ROOT / "mismatch_feature_transform_curves.png", dpi=220)
    plt.close(fig)


def plot_mismatch_transform_timeseries(raw_rows: list[dict]) -> None:
    selected = [
        next(row for row in raw_rows if row["run_path"] == str(DISTILLATION_LATEST_RESIDUAL_TD3_PATH.relative_to(REPO_ROOT))),
        next(row for row in raw_rows if row["run_path"] == str(POLYMER_RESIDUAL_TD3_PATH.relative_to(REPO_ROOT))),
    ]
    transform_specs = (
        ("hard_clip", "hard clip +/-3", "#C44E52"),
        ("soft_tanh", "soft tanh", "#4C72B0"),
        ("signed_log", "signed log1p", "#55A868"),
        ("vecnorm", "VecNormalize-like", "#8172B2"),
    )
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for row_idx, row in enumerate(selected):
        system = row["system"]
        bundle = load_pickle(REPO_ROOT / row["run_path"])
        start, end = get_last_eval_window(bundle)
        tail_start = max(start, end - 60)
        for col_offset, (feature_key, title_stub) in enumerate((("tracking_raw", "Tracking"), ("innovation_raw", "Innovation"))):
            arr = np.asarray(row[feature_key], float)
            channel = int(np.argmax(np.percentile(np.abs(arr), 99, axis=0)))
            raw = arr[:, channel]
            transformed = {label: transform_feature(raw, kind) for kind, label, _ in transform_specs}

            ax_full = axs[row_idx, 2 * col_offset]
            full_x = np.arange(start, end) - start
            for _, label, color in transform_specs:
                ax_full.plot(full_x, transformed[label][start:end], label=label, color=color, linewidth=1.0)
            ax_full.set_title(f"{system.title()}: {title_stub.lower()} channel {channel + 1}\nlast evaluation episode")
            ax_full.set_ylabel("transformed feature")
            ax_full.set_xlabel("Step in evaluation episode")
            if row_idx == 0 and col_offset == 0:
                ax_full.legend(fontsize=8, loc="upper right")

            ax_tail = axs[row_idx, 2 * col_offset + 1]
            tail_x = np.arange(tail_start, end) - tail_start
            for _, label, color in transform_specs:
                ax_tail.plot(tail_x, transformed[label][tail_start:end], label=label, color=color, linewidth=1.0)
            ax_tail.set_title(f"{system.title()}: {title_stub.lower()} channel {channel + 1}\nfinal 60-step zoom")
            ax_tail.set_ylabel("transformed feature")
            ax_tail.set_xlabel("Final-step index")
    fig.suptitle("How transform choices behave on real mismatch features", fontsize=14)
    fig.savefig(FIG_ROOT / "mismatch_transform_timeseries.png", dpi=220)
    plt.close(fig)


def plot_rho_candidate_mappings() -> None:
    x = np.linspace(0.0, 6.0, 600)
    floor = 0.2
    rho_current = np.clip(x, 0.0, 1.0)
    rho_exp = 1.0 - np.exp(-0.55 * x)
    rho_sigmoid = 1.0 / (1.0 + np.exp(-1.4 * (x - 1.5)))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axs[0].plot(x, rho_current, label="current clipped linear rho", linewidth=1.2)
    axs[0].plot(x, rho_exp, label="exponential gate", linewidth=1.2)
    axs[0].plot(x, rho_sigmoid, label="sigmoid gate", linewidth=1.2)
    axs[0].axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    axs[0].set_title("Candidate rho maps from raw tracking magnitude")
    axs[0].set_xlabel("raw tracking magnitude")
    axs[0].set_ylabel("rho")
    axs[0].legend(fontsize=8, loc="lower right")

    axs[1].plot(x, floor + (1.0 - floor) * rho_current, label="current rho_eff", linewidth=1.2)
    axs[1].plot(x, floor + (1.0 - floor) * rho_exp, label="exponential rho_eff", linewidth=1.2)
    axs[1].plot(x, floor + (1.0 - floor) * rho_sigmoid, label="sigmoid rho_eff", linewidth=1.2)
    axs[1].axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    axs[1].set_title("How different rho maps change residual authority")
    axs[1].set_xlabel("raw tracking magnitude")
    axs[1].set_ylabel("authority factor")
    axs[1].legend(fontsize=8, loc="lower right")
    fig.savefig(FIG_ROOT / "rho_candidate_mappings.png", dpi=220)
    plt.close(fig)


def plot_rho_diagnostics(rho_rows: list[dict]) -> None:
    x = np.linspace(0.0, 3.0, 400)
    distill_rho = np.clip(x, 0.0, 1.0)
    distill_rho_eff = 0.2 + (1.0 - 0.2) * distill_rho
    polymer_rho = np.clip(x, 0.0, 1.0)
    polymer_rho_eff = 0.15 + (1.0 - 0.15) * polymer_rho

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axs[0].plot(x, distill_rho, label="distillation rho", linewidth=1.2)
    axs[0].plot(x, distill_rho_eff, label="distillation rho_eff", linewidth=1.2)
    axs[0].plot(x, polymer_rho, label="polymer rho", linewidth=1.2)
    axs[0].plot(x, polymer_rho_eff, label="polymer rho_eff", linewidth=1.2)
    axs[0].axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    axs[0].set_title("Current rho mapping from max |tracking_error|")
    axs[0].set_xlabel("max |tracking_error feature| at step")
    axs[0].set_ylabel("authority factor")
    axs[0].legend(fontsize=8, loc="lower right")

    labels = [row["label"].replace("\n", " ") for row in rho_rows]
    xpos = np.arange(len(labels))
    width = 0.25
    axs[1].bar(xpos - width, [row["rho_one_frac"] for row in rho_rows], width, label="rho = 1 frac")
    axs[1].bar(xpos, [row["authority_projection_frac"] for row in rho_rows], width, label="authority projection frac")
    axs[1].bar(xpos + width, [row["median_exec_raw_ratio"] for row in rho_rows], width, label="median ||exec||/||raw||")
    axs[1].set_title("Empirical rho saturation and residual attenuation")
    axs[1].set_ylabel("Fraction / ratio")
    axs[1].set_xticks(xpos)
    axs[1].set_xticklabels(labels, rotation=15, ha="right")
    axs[1].set_ylim(0.0, 1.05)
    axs[1].legend(fontsize=8, loc="upper right")
    fig.savefig(FIG_ROOT / "rho_authority_diagnostics.png", dpi=220)
    plt.close(fig)


def _scale_input_series_phys_to_abs(bundle: dict, values_phys: np.ndarray) -> np.ndarray:
    values_phys = np.asarray(values_phys, float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle["n_inputs"])
    return (values_phys - data_min[:n_inputs]) / np.maximum(data_max[:n_inputs] - data_min[:n_inputs], 1e-12)


def replay_residual_refresh(bundle: dict) -> dict | None:
    if bundle.get("a_res_raw_log") is None or bundle.get("u_base") is None or bundle.get("delta_u_storage") is None:
        return None
    innovation_raw, tracking_raw = reconstruct_raw_mismatch_features(bundle)
    if innovation_raw is None or tracking_raw is None:
        return None
    config_snapshot = dict(bundle.get("config_snapshot") or {})
    if "b_min" not in config_snapshot or "b_max" not in config_snapshot:
        return None

    n_inputs = int(bundle["n_inputs"])
    ss_scaled_inputs = apply_min_max(
        np.asarray(bundle["steady_states"]["ss_inputs"], float),
        np.asarray(bundle["data_min"], float)[:n_inputs],
        np.asarray(bundle["data_max"], float)[:n_inputs],
    )
    u_min_scaled_abs = np.asarray(config_snapshot["b_min"], float).reshape(-1) + ss_scaled_inputs
    u_max_scaled_abs = np.asarray(config_snapshot["b_max"], float).reshape(-1) + ss_scaled_inputs
    u_base_scaled = _scale_input_series_phys_to_abs(bundle, bundle["u_base"])
    u_applied_scaled = _scale_input_series_phys_to_abs(bundle, bundle["u"])
    scaled_current_input = u_applied_scaled - np.asarray(bundle["delta_u_storage"], float)
    low_coef = np.asarray(bundle.get("low_coef", bundle.get("residual_low_coef")), float).reshape(-1)
    high_coef = np.asarray(bundle.get("high_coef", bundle.get("residual_high_coef")), float).reshape(-1)
    authority_beta_res = np.asarray(bundle["authority_beta_res"], float).reshape(-1)
    authority_du0_res = np.asarray(bundle["authority_du0_res"], float).reshape(-1)

    n_steps = int(bundle["nFE"])
    new_delta_exec = np.zeros((n_steps, n_inputs), dtype=float)
    new_rho = np.zeros(n_steps, dtype=float)
    new_rho_raw = np.zeros(n_steps, dtype=float)
    new_rho_eff = np.zeros(n_steps, dtype=float)
    new_deadband = np.zeros(n_steps, dtype=int)
    new_proj_authority = np.zeros(n_steps, dtype=int)
    new_proj_headroom = np.zeros(n_steps, dtype=int)

    tracking_feat = bundle.get("tracking_error_log")
    for idx in range(n_steps):
        projection = project_residual_action(
            action_raw=np.asarray(bundle["a_res_raw_log"], float)[idx, :],
            low_coef=low_coef,
            high_coef=high_coef,
            u_base=u_base_scaled[idx, :],
            scaled_current_input=scaled_current_input[idx, :],
            u_min_scaled_abs=u_min_scaled_abs,
            u_max_scaled_abs=u_max_scaled_abs,
            apply_authority=bool(bundle.get("state_mode", "standard") == "mismatch"),
            authority_use_rho=bool(bundle.get("authority_use_rho", bundle.get("use_rho_authority", True))),
            tracking_error_feat=None if tracking_feat is None else np.asarray(tracking_feat, float)[idx, :],
            tracking_error_raw=np.asarray(tracking_raw, float)[idx, :],
            innovation_raw=np.asarray(innovation_raw, float)[idx, :],
            authority_beta_res=authority_beta_res,
            authority_du0_res=authority_du0_res,
            authority_rho_floor=float(bundle.get("authority_rho_floor", 0.15)),
            authority_rho_power=float(bundle.get("authority_rho_power", 1.0)),
            rho_mapping_mode="exp_raw_tracking",
            authority_rho_k=0.55,
            residual_zero_deadband_enabled=True,
            residual_zero_tracking_raw_threshold=0.1,
            residual_zero_innovation_raw_threshold=0.1,
        )
        new_delta_exec[idx, :] = np.asarray(projection["delta_u_res_exec"], float)
        new_rho[idx] = float(projection["rho"]) if projection["rho"] is not None else 0.0
        new_rho_raw[idx] = float(projection["rho_raw"]) if projection["rho_raw"] is not None else 0.0
        new_rho_eff[idx] = float(projection["rho_eff"]) if projection["rho_eff"] is not None else 0.0
        new_deadband[idx] = int(projection["deadband_active"])
        new_proj_authority[idx] = int(projection["projection_due_to_authority"])
        new_proj_headroom[idx] = int(projection["projection_due_to_headroom"])

    return {
        "delta_u_res_exec_new": new_delta_exec,
        "rho_new": new_rho,
        "rho_raw_new": new_rho_raw,
        "rho_eff_new": new_rho_eff,
        "deadband_active_new": new_deadband,
        "projection_due_to_authority_new": new_proj_authority,
        "projection_due_to_headroom_new": new_proj_headroom,
    }


def plot_residual_refresh_offline_replay(cache: list[dict]) -> list[dict]:
    rows = []
    selected = [
        next(item for item in cache if item["spec"].system == "distillation" and "Residual TD3" in item["spec"].label),
        next(item for item in cache if item["spec"].system == "polymer" and "Residual TD3" in item["spec"].label),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.8), constrained_layout=True)
    for row_idx, item in enumerate(selected):
        spec = item["spec"]
        bundle = item["bundle"]
        replay = replay_residual_refresh(bundle)
        if replay is None:
            continue
        if spec.system == "distillation":
            start, end = get_last_eval_window(bundle)
        else:
            start, end = (0, min(int(bundle["nFE"]), 1200))
        time = np.arange(start, end)
        legacy_exec = np.asarray(bundle["delta_u_res_exec_log"], float)[start:end, :]
        legacy_rho = np.zeros(end - start, dtype=float) if bundle.get("rho_log") is None else np.asarray(bundle["rho_log"], float)[start:end]
        new_exec = np.asarray(replay["delta_u_res_exec_new"], float)[start:end, :]
        new_rho = np.asarray(replay["rho_new"], float)[start:end]
        new_deadband = np.asarray(replay["deadband_active_new"], float)[start:end]
        tracking_raw = np.asarray(reconstruct_raw_mismatch_features(bundle)[1], float)[start:end, :]

        axs[row_idx, 0].plot(time, np.linalg.norm(legacy_exec, axis=1), label="legacy")
        axs[row_idx, 0].plot(time, np.linalg.norm(new_exec, axis=1), linestyle="--", label="new replay")
        axs[row_idx, 0].set_title(f"{spec.system.title()}: residual norm")
        axs[row_idx, 0].set_ylabel("||delta_u_res||")
        axs[row_idx, 1].plot(time, legacy_rho, label="legacy rho")
        axs[row_idx, 1].plot(time, new_rho, linestyle="--", label="new rho")
        axs[row_idx, 1].set_title("Legacy vs new rho")
        axs[row_idx, 2].plot(time, np.max(np.abs(tracking_raw), axis=1), color="#355070")
        axs[row_idx, 2].axhline(0.1, color="#B56576", linestyle="--")
        axs[row_idx, 2].set_title("max |tracking_raw|")
        axs[row_idx, 3].step(time, new_deadband, where="post", color="#2D6A4F")
        axs[row_idx, 3].set_ylim(-0.05, 1.05)
        axs[row_idx, 3].set_title("new deadband active")
        for col_idx in range(4):
            axs[row_idx, col_idx].set_xlabel("step")
        axs[row_idx, 0].legend(loc="best", fontsize=8)
        axs[row_idx, 1].legend(loc="best", fontsize=8)

        tail_start = max(start, end - min(60, end - start))
        legacy_tail = legacy_exec[tail_start - start : end - start, :]
        new_tail = new_exec[tail_start - start : end - start, :]
        rows.append(
            {
                "system": spec.system,
                "label": spec.label.replace("\n", " "),
                "run_path": str(spec.run_path.relative_to(REPO_ROOT)),
                "legacy_rho_one_frac": float(np.mean(legacy_rho >= 0.999999)),
                "new_rho_one_frac": float(np.mean(new_rho >= 0.999999)),
                "new_deadband_frac": float(np.mean(new_deadband >= 0.5)),
                "legacy_tail_residual_norm": float(np.mean(np.linalg.norm(legacy_tail, axis=1))),
                "new_tail_residual_norm": float(np.mean(np.linalg.norm(new_tail, axis=1))),
                "legacy_authority_frac": float(np.mean(np.asarray(bundle.get("projection_due_to_authority_log"), float)[start:end])) if bundle.get("projection_due_to_authority_log") is not None else np.nan,
                "new_authority_frac": float(np.mean(np.asarray(replay["projection_due_to_authority_new"], float)[start:end])),
            }
        )

    fig.suptitle("Offline replay of residual refresh: saved trajectory, new rho + deadband", fontsize=14)
    fig.savefig(FIG_ROOT / "residual_refresh_offline_replay.png", dpi=220)
    plt.close(fig)
    return rows


def build_distillation_residual_eval_rows() -> list[dict]:
    baseline_bundle = load_pickle(DISTILLATION_BASELINE_PATH)
    baseline_y = np.asarray(baseline_bundle["y"], float)
    baseline_u = np.asarray(baseline_bundle["u"], float)
    baseline_sp = setpoint_scaled_to_physical(baseline_bundle)
    start, end = get_last_eval_window(baseline_bundle)
    tail_start = max(start, end - 60)

    specs = (
        ("Baseline MPC", DISTILLATION_BASELINE_PATH, False),
        ("Residual TD3 2026-04-17", DISTILLATION_PREVIOUS_RESIDUAL_TD3_PATH, True),
        ("Residual TD3 2026-04-20", DISTILLATION_LATEST_RESIDUAL_TD3_PATH, True),
        ("Residual SAC 2026-04-15", DISTILLATION_RESIDUAL_SAC_PATH, True),
    )
    rows: list[dict] = []
    for label, path, is_residual in specs:
        bundle = load_pickle(path)
        y = np.asarray(bundle["y"], float)
        u = np.asarray(bundle["u"], float)
        sp = setpoint_scaled_to_physical(bundle)
        row = {
            "label": label,
            "run_path": str(path.relative_to(REPO_ROOT)),
            "eval_mae_y1": float(np.mean(np.abs(y[start + 1 : end + 1, 0] - sp[start:end, 0]))),
            "eval_mae_y2": float(np.mean(np.abs(y[start + 1 : end + 1, 1] - sp[start:end, 1]))),
            "tail_mae_y1": float(np.mean(np.abs(y[tail_start + 1 : end + 1, 0] - sp[tail_start:end, 0]))),
            "tail_mae_y2": float(np.mean(np.abs(y[tail_start + 1 : end + 1, 1] - sp[tail_start:end, 1]))),
            "tail_mean_output2_bias_vs_sp": float(np.mean(y[tail_start + 1 : end + 1, 1] - sp[tail_start:end, 1])),
            "tail_mean_output2_bias_vs_baseline": float(np.mean(y[tail_start + 1 : end + 1, 1] - baseline_y[tail_start + 1 : end + 1, 1])),
            "tail_mean_abs_u_gap_vs_baseline_1": float(np.mean(np.abs(u[tail_start:end, 0] - baseline_u[tail_start:end, 0]))),
            "tail_mean_abs_u_gap_vs_baseline_2": float(np.mean(np.abs(u[tail_start:end, 1] - baseline_u[tail_start:end, 1]))),
        }
        if is_residual:
            u_base = np.asarray(bundle["u_base"], float)
            rho = np.asarray(bundle["rho_log"], float)
            tracking = np.asarray(bundle["tracking_error_log"], float)
            delta_u_exec = np.asarray(bundle["delta_u_res_exec_log"], float)
            row.update(
                {
                    "tail_mean_rho": float(np.mean(rho[tail_start:end])),
                    "tail_rho_one_frac": float(np.mean(rho[tail_start:end] >= 0.999999)),
                    "tail_mean_residual_norm": float(np.mean(np.linalg.norm(delta_u_exec[tail_start:end], axis=1))),
                    "tail_mean_abs_u_minus_ubase_1": float(np.mean(np.abs(u[tail_start:end, 0] - u_base[tail_start:end, 0]))),
                    "tail_mean_abs_u_minus_ubase_2": float(np.mean(np.abs(u[tail_start:end, 1] - u_base[tail_start:end, 1]))),
                    "tail_mean_abs_ubase_minus_baseline_1": float(np.mean(np.abs(u_base[tail_start:end, 0] - baseline_u[tail_start:end, 0]))),
                    "tail_mean_abs_ubase_minus_baseline_2": float(np.mean(np.abs(u_base[tail_start:end, 1] - baseline_u[tail_start:end, 1]))),
                    "tail_mean_max_abs_tracking": float(np.mean(max_abs_tracking_per_step(tracking[tail_start:end]))),
                }
            )
        else:
            row.update(
                {
                    "tail_mean_rho": 0.0,
                    "tail_rho_one_frac": 0.0,
                    "tail_mean_residual_norm": 0.0,
                    "tail_mean_abs_u_minus_ubase_1": 0.0,
                    "tail_mean_abs_u_minus_ubase_2": 0.0,
                    "tail_mean_abs_ubase_minus_baseline_1": 0.0,
                    "tail_mean_abs_ubase_minus_baseline_2": 0.0,
                    "tail_mean_max_abs_tracking": 0.0,
                }
            )
        rows.append(row)
    return rows


def plot_distillation_latest_residual_eval_episode() -> None:
    baseline = load_pickle(DISTILLATION_BASELINE_PATH)
    latest = load_pickle(DISTILLATION_LATEST_RESIDUAL_TD3_PATH)
    start, end = get_last_eval_window(latest)
    x = np.arange(start, end) - start
    y_sp_phys = setpoint_scaled_to_physical(latest)
    y_base = np.asarray(baseline["y"], float)
    y_latest = np.asarray(latest["y"], float)
    u_base = np.asarray(latest["u_base"], float)
    u_latest = np.asarray(latest["u"], float)
    u_baseline = np.asarray(baseline["u"], float)
    rho = np.asarray(latest["rho_log"], float)
    tracking = np.asarray(latest["tracking_error_log"], float)
    delta_u_exec = np.asarray(latest["delta_u_res_exec_log"], float)

    fig, axs = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    output_labels = latest["system_metadata"]["output_labels"]
    input_labels = latest["system_metadata"]["input_labels"]
    for idx in range(2):
        ax = axs[0, idx]
        ax.plot(x, y_sp_phys[start:end, idx], color="k", linestyle="--", linewidth=1.0, label="setpoint")
        ax.plot(x, y_base[start + 1 : end + 1, idx], color="#55A868", linewidth=1.0, label="baseline MPC")
        ax.plot(x, y_latest[start + 1 : end + 1, idx], color="#C44E52", linewidth=1.0, label="latest residual TD3")
        ax.set_title(f"{output_labels[idx]} during the last evaluation episode")
        ax.set_ylabel("physical output")
        ax.set_xlabel("Step in evaluation episode")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    for idx in range(2):
        ax = axs[1, idx]
        ax.plot(x, u_baseline[start:end, idx], color="#55A868", linewidth=1.0, label="baseline MPC")
        ax.plot(x, u_base[start:end, idx], color="#4C72B0", linewidth=1.0, label="residual run u_base")
        ax.plot(x, u_latest[start:end, idx], color="#C44E52", linewidth=1.0, label="residual run applied u")
        ax.set_title(f"{input_labels[idx]} during the last evaluation episode")
        ax.set_ylabel("physical input")
        ax.set_xlabel("Step in evaluation episode")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axs[2, 0].plot(x, rho[start:end], label="rho", color="#8172B2", linewidth=1.1)
    axs[2, 0].plot(x, max_abs_tracking_per_step(tracking[start:end]), label="max |tracking|", color="#CC6677", linewidth=1.1)
    axs[2, 0].set_title("Latest residual TD3 mismatch diagnostics")
    axs[2, 0].set_ylabel("feature magnitude")
    axs[2, 0].set_xlabel("Step in evaluation episode")
    axs[2, 0].legend(fontsize=8, loc="upper right")

    axs[2, 1].plot(x, np.linalg.norm(delta_u_exec[start:end], axis=1), label="||u - u_base||", color="#C44E52", linewidth=1.1)
    axs[2, 1].plot(x, np.linalg.norm(u_base[start:end] - u_baseline[start:end], axis=1), label="||u_base - u_baseline||", color="#4C72B0", linewidth=1.1)
    axs[2, 1].set_title("Direct residual action vs shifted MPC trajectory")
    axs[2, 1].set_ylabel("input-gap norm")
    axs[2, 1].set_xlabel("Step in evaluation episode")
    axs[2, 1].legend(fontsize=8, loc="upper right")
    fig.savefig(FIG_ROOT / "distillation_latest_residual_eval_episode.png", dpi=220)
    plt.close(fig)

    tail_start = max(start, end - 60)
    tail_x = np.arange(tail_start, end) - tail_start
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 7.5), constrained_layout=True)
    axs[0, 0].plot(tail_x, y_sp_phys[tail_start:end, 1], color="k", linestyle="--", linewidth=1.0, label="setpoint")
    axs[0, 0].plot(tail_x, y_base[tail_start + 1 : end + 1, 1], color="#55A868", linewidth=1.0, label="baseline MPC")
    axs[0, 0].plot(tail_x, y_latest[tail_start + 1 : end + 1, 1], color="#C44E52", linewidth=1.0, label="latest residual TD3")
    axs[0, 0].set_title(f"Tail zoom on {output_labels[1]}")
    axs[0, 0].set_ylabel("physical output")
    axs[0, 0].set_xlabel("Final-step index")
    axs[0, 0].legend(fontsize=8, loc="upper right")

    axs[0, 1].plot(tail_x, y_latest[tail_start + 1 : end + 1, 1] - y_sp_phys[tail_start:end, 1], color="#C44E52", linewidth=1.0, label="latest residual TD3")
    axs[0, 1].plot(tail_x, y_base[tail_start + 1 : end + 1, 1] - y_sp_phys[tail_start:end, 1], color="#55A868", linewidth=1.0, label="baseline MPC")
    axs[0, 1].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    axs[0, 1].set_title("Tail output-2 bias relative to the setpoint")
    axs[0, 1].set_ylabel("output bias")
    axs[0, 1].set_xlabel("Final-step index")
    axs[0, 1].legend(fontsize=8, loc="upper right")

    axs[1, 0].plot(tail_x, rho[tail_start:end], color="#8172B2", linewidth=1.0, label="rho")
    axs[1, 0].plot(tail_x, max_abs_tracking_per_step(tracking[tail_start:end]), color="#CC6677", linewidth=1.0, label="max |tracking|")
    axs[1, 0].set_title("Tail mismatch diagnostics")
    axs[1, 0].set_ylabel("feature magnitude")
    axs[1, 0].set_xlabel("Final-step index")
    axs[1, 0].legend(fontsize=8, loc="upper right")

    axs[1, 1].plot(tail_x, np.linalg.norm(delta_u_exec[tail_start:end], axis=1), color="#C44E52", linewidth=1.0, label="||u - u_base||")
    axs[1, 1].plot(tail_x, np.linalg.norm(u_base[tail_start:end] - u_baseline[tail_start:end], axis=1), color="#4C72B0", linewidth=1.0, label="||u_base - u_baseline||")
    axs[1, 1].set_title("Tail input-gap decomposition")
    axs[1, 1].set_ylabel("input-gap norm")
    axs[1, 1].set_xlabel("Final-step index")
    axs[1, 1].legend(fontsize=8, loc="upper right")
    fig.savefig(FIG_ROOT / "distillation_latest_residual_tail_diagnostics.png", dpi=220)
    plt.close(fig)


def plot_distillation_residual_eval_summary(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axs[0, 0].bar(x - width / 2, [row["tail_mae_y1"] for row in rows], width, label="output 1")
    axs[0, 0].bar(x + width / 2, [row["tail_mae_y2"] for row in rows], width, label="output 2")
    axs[0, 0].set_title("Final 60-step mean absolute tracking error")
    axs[0, 0].set_ylabel("MAE")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, rotation=15, ha="right")
    axs[0, 0].legend(fontsize=8, loc="upper right")

    axs[0, 1].bar(x - width / 2, [row["eval_mae_y1"] for row in rows], width, label="output 1")
    axs[0, 1].bar(x + width / 2, [row["eval_mae_y2"] for row in rows], width, label="output 2")
    axs[0, 1].set_title("Whole evaluation-episode mean absolute tracking error")
    axs[0, 1].set_ylabel("MAE")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels, rotation=15, ha="right")
    axs[0, 1].legend(fontsize=8, loc="upper right")

    axs[1, 0].bar(x - width / 2, [row["tail_mean_rho"] for row in rows], width, label="tail mean rho")
    axs[1, 0].bar(x + width / 2, [row["tail_mean_residual_norm"] for row in rows], width, label="tail mean ||u-u_base||")
    axs[1, 0].set_title("Residual activity near the end of the evaluation episode")
    axs[1, 0].set_ylabel("mean feature / norm")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels, rotation=15, ha="right")
    axs[1, 0].legend(fontsize=8, loc="upper right")

    axs[1, 1].bar(x - width / 2, [row["tail_mean_abs_u_minus_ubase_1"] + row["tail_mean_abs_u_minus_ubase_2"] for row in rows], width, label="direct residual gap")
    axs[1, 1].bar(x + width / 2, [row["tail_mean_abs_ubase_minus_baseline_1"] + row["tail_mean_abs_ubase_minus_baseline_2"] for row in rows], width, label="shifted MPC gap")
    axs[1, 1].set_title("How much of the input difference comes from the residual itself")
    axs[1, 1].set_ylabel("sum of tail mean absolute gaps")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, rotation=15, ha="right")
    axs[1, 1].legend(fontsize=8, loc="upper right")
    fig.savefig(FIG_ROOT / "distillation_residual_eval_summary.png", dpi=220)
    plt.close(fig)


def build_unified_method_defaults_rows() -> list[dict]:
    method_specs = (
        ("horizon_standard", "Horizon", {"changes_horizon": 1, "changes_model": 0, "changes_weights": 0, "changes_direct_input": 0, "online_identification": 0, "bounded_action_set": 1, "method_specific_guard": 0}),
        ("matrix", "Matrix", {"changes_horizon": 0, "changes_model": 1, "changes_weights": 0, "changes_direct_input": 0, "online_identification": 0, "bounded_action_set": 1, "method_specific_guard": 0}),
        ("structured_matrix", "Structured Matrix", {"changes_horizon": 0, "changes_model": 1, "changes_weights": 0, "changes_direct_input": 0, "online_identification": 0, "bounded_action_set": 1, "method_specific_guard": 1}),
        ("weights", "Weights", {"changes_horizon": 0, "changes_model": 0, "changes_weights": 1, "changes_direct_input": 0, "online_identification": 0, "bounded_action_set": 1, "method_specific_guard": 0}),
        ("residual", "Residual", {"changes_horizon": 0, "changes_model": 0, "changes_weights": 0, "changes_direct_input": 1, "online_identification": 0, "bounded_action_set": 1, "method_specific_guard": 1}),
        ("reidentification", "Reidentification", {"changes_horizon": 0, "changes_model": 1, "changes_weights": 0, "changes_direct_input": 0, "online_identification": 1, "bounded_action_set": 1, "method_specific_guard": 1}),
    )
    getters = {
        "polymer": get_polymer_notebook_defaults,
        "distillation": get_distillation_notebook_defaults,
    }
    rows: list[dict] = []
    for family_key, family_label, flags in method_specs:
        for system_name, getter in getters.items():
            cfg = getter(family_key)
            state_mode = str(cfg.get("state_mode", "standard")).lower()
            uses_rho = int(
                family_key == "residual"
                and state_mode == "mismatch"
                and bool(cfg.get("authority_use_rho", cfg.get("use_rho_authority", False)))
            )
            rows.append(
                {
                    "system": system_name,
                    "family_key": family_key,
                    "family": family_label,
                    "default_state_mode": state_mode,
                    "mismatch_default": int(state_mode == "mismatch"),
                    "rho_default_active": uses_rho,
                    "guardrail_type": (
                        "grid"
                        if family_key == "horizon_standard"
                        else "model_box"
                        if family_key in {"matrix", "structured_matrix"}
                        else "weight_box"
                        if family_key == "weights"
                        else "residual_projection"
                        if family_key == "residual"
                        else "candidate_guard"
                    ),
                    **flags,
                }
            )
    return rows


def build_combined_default_rows() -> list[dict]:
    getters = {
        "polymer": get_polymer_notebook_defaults,
        "distillation": get_distillation_notebook_defaults,
    }
    rows: list[dict] = []
    for system_name, getter in getters.items():
        cfg = getter("combined")
        residual_state_mode = str(cfg.get("residual_state_mode", "standard")).lower()
        rows.append(
            {
                "system": system_name,
                "horizon_mismatch_default": int(str(cfg.get("horizon_state_mode", "standard")).lower() == "mismatch"),
                "matrix_mismatch_default": int(str(cfg.get("matrix_state_mode", "standard")).lower() == "mismatch"),
                "weights_mismatch_default": int(str(cfg.get("weights_state_mode", "standard")).lower() == "mismatch"),
                "residual_mismatch_default": int(residual_state_mode == "mismatch"),
                "rho_default_active": int(
                    residual_state_mode == "mismatch"
                    and bool(cfg.get("authority_use_rho", cfg.get("use_rho_authority", False)))
                ),
                "residual_agent_kind": str(cfg.get("residual_agent_kind", "td3")).lower(),
            }
        )
    return rows


def build_method_parameter_rows() -> list[dict]:
    rows: list[dict] = []
    for system_name, getter in {
        "polymer": get_polymer_notebook_defaults,
        "distillation": get_distillation_notebook_defaults,
    }.items():
        residual_cfg = getter("residual")
        reward_cfg = residual_cfg["reward"]
        ctrl = residual_cfg["controller"]
        low = np.asarray(ctrl["low_coef"], float)
        high = np.asarray(ctrl["high_coef"], float)
        beta = np.asarray(residual_cfg["authority_beta_res"], float)
        du0 = np.asarray(residual_cfg["authority_du0_res"], float)
        for idx in range(low.size):
            rows.append(
                {
                    "system": system_name,
                    "parameter_group": "residual_span",
                    "channel": idx + 1,
                    "value": float(high[idx] - low[idx]),
                }
            )
            rows.append(
                {
                    "system": system_name,
                    "parameter_group": "authority_beta_res",
                    "channel": idx + 1,
                    "value": float(beta[idx]),
                }
            )
            rows.append(
                {
                    "system": system_name,
                    "parameter_group": "authority_du0_res",
                    "channel": idx + 1,
                    "value": float(du0[idx]),
                }
            )
        for idx, value in enumerate(np.asarray(reward_cfg["k_rel"], float), start=1):
            rows.append(
                {
                    "system": system_name,
                    "parameter_group": "k_rel",
                    "channel": idx,
                    "value": float(value),
                }
            )
        for idx, value in enumerate(np.asarray(reward_cfg["band_floor_phys"], float), start=1):
            rows.append(
                {
                    "system": system_name,
                    "parameter_group": "band_floor_phys",
                    "channel": idx,
                    "value": float(value),
                }
            )
        rows.append(
            {
                "system": system_name,
                "parameter_group": "authority_rho_floor",
                "channel": 0,
                "value": float(residual_cfg["authority_rho_floor"]),
            }
        )
        rows.append(
            {
                "system": system_name,
                "parameter_group": "authority_rho_power",
                "channel": 0,
                "value": float(residual_cfg["authority_rho_power"]),
            }
        )
        rows.append(
            {
                "system": system_name,
                "parameter_group": "mismatch_clip",
                "channel": 0,
                "value": float(ctrl["mismatch_clip"]),
            }
        )
        rows.append(
            {
                "system": system_name,
                "parameter_group": "tracking_eta_tol",
                "channel": 0,
                "value": float(ctrl["tracking_eta_tol"]),
            }
        )
    return rows


def plot_unified_method_feature_matrix(method_rows: list[dict]) -> None:
    feature_cols = [
        ("changes_horizon", "changes\nhorizon"),
        ("changes_model", "changes\nmodel"),
        ("changes_weights", "changes\nweights"),
        ("changes_direct_input", "changes\ndirect input"),
        ("online_identification", "online\nID"),
        ("bounded_action_set", "bounded\naction set"),
        ("method_specific_guard", "family-\nspecific guard"),
        ("mismatch_default", "default\nmismatch"),
        ("rho_default_active", "default\nrho active"),
    ]
    ordered_rows = []
    families = ["Horizon", "Matrix", "Structured Matrix", "Weights", "Residual", "Reidentification"]
    systems = ["polymer", "distillation"]
    for family in families:
        for system_name in systems:
            ordered_rows.append(next(row for row in method_rows if row["family"] == family and row["system"] == system_name))
    data = np.asarray([[row[key] for key, _ in feature_cols] for row in ordered_rows], float)
    row_labels = [f"{row['system'][:4].title()} {row['family']}" for row in ordered_rows]

    fig, ax = plt.subplots(figsize=(12.5, 8.0), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Shared RL-assisted MPC method surface: same family code, different default semantics")
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels([label for _, label in feature_cols], rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "Y" if data[i, j] >= 0.5 else "", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(FIG_ROOT / "unified_method_feature_matrix.png", dpi=220)
    plt.close(fig)


def plot_combined_default_modes(combined_rows: list[dict]) -> None:
    cols = [
        ("horizon_mismatch_default", "horizon\nmismatch"),
        ("matrix_mismatch_default", "matrix\nmismatch"),
        ("weights_mismatch_default", "weights\nmismatch"),
        ("residual_mismatch_default", "residual\nmismatch"),
        ("rho_default_active", "residual\nrho active"),
    ]
    systems = ["polymer", "distillation"]
    ordered = [next(row for row in combined_rows if row["system"] == system_name) for system_name in systems]
    data = np.asarray([[row[key] for key, _ in cols] for row in ordered], float)
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=1.0)
    ax.set_title("Combined supervisor defaults are already branch-specific, not globally unified")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _, label in cols], rotation=0)
    ax.set_yticks(np.arange(len(systems)))
    ax.set_yticklabels([name.title() for name in systems])
    for i, row in enumerate(ordered):
        labels = [
            "mismatch" if row["horizon_mismatch_default"] else "standard",
            "mismatch" if row["matrix_mismatch_default"] else "standard",
            "mismatch" if row["weights_mismatch_default"] else "standard",
            "mismatch" if row["residual_mismatch_default"] else "standard",
            "on" if row["rho_default_active"] else "off",
        ]
        for j, text in enumerate(labels):
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.savefig(FIG_ROOT / "combined_default_modes.png", dpi=220)
    plt.close(fig)


def plot_residual_method_parameter_comparison(parameter_rows: list[dict]) -> None:
    def get_values(group: str, system_name: str) -> list[float]:
        return [row["value"] for row in parameter_rows if row["parameter_group"] == group and row["system"] == system_name]

    systems = ["polymer", "distillation"]
    colors = {"polymer": "#D66A4E", "distillation": "#3A7CA5"}

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for ax, group, title in [
        (axs[0, 0], "residual_span", "Residual action span by input"),
        (axs[0, 1], "authority_beta_res", "Residual authority beta by input"),
        (axs[1, 0], "authority_du0_res", "Residual authority du0 by input"),
    ]:
        x = np.arange(2)
        width = 0.35
        for offset, system_name in zip((-0.175, 0.175), systems):
            ax.bar(x + offset, get_values(group, system_name), width=width, color=colors[system_name], label=system_name.title())
        ax.set_xticks(x)
        ax.set_xticklabels(["input 1", "input 2"])
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    x = np.arange(len(systems))
    axs[1, 1].bar(x - 0.16, [get_values("authority_rho_floor", s)[0] for s in systems], width=0.32, label="rho_floor")
    axs[1, 1].bar(x + 0.16, [get_values("authority_rho_power", s)[0] for s in systems], width=0.32, label="rho_power")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([s.title() for s in systems])
    axs[1, 1].set_title("Residual scalar authority defaults")
    axs[1, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Residual-family defaults differ substantially across polymer and distillation", fontsize=14)
    fig.savefig(FIG_ROOT / "residual_method_parameter_comparison.png", dpi=220)
    plt.close(fig)


def plot_mismatch_scale_parameter_comparison(parameter_rows: list[dict]) -> None:
    systems = ["polymer", "distillation"]
    colors = {"polymer": "#D66A4E", "distillation": "#3A7CA5"}

    def values(group: str, system_name: str) -> list[float]:
        return [row["value"] for row in parameter_rows if row["parameter_group"] == group and row["system"] == system_name]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for ax, group, title in [
        (axs[0], "k_rel", "Reward mismatch scale k_rel"),
        (axs[1], "band_floor_phys", "Reward mismatch floor band_floor_phys"),
    ]:
        x = np.arange(2)
        width = 0.35
        for offset, system_name in zip((-0.175, 0.175), systems):
            ax.bar(x + offset, values(group, system_name), width=width, color=colors[system_name], label=system_name.title())
        ax.set_xticks(x)
        ax.set_xticklabels(["output 1", "output 2"])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Mismatch scales are system-specific even though the mismatch-state API is shared", fontsize=14)
    fig.savefig(FIG_ROOT / "mismatch_scale_parameter_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    summary_rows: list[dict] = []
    state_rows: list[dict] = []
    raw_feature_rows: list[dict] = []
    rho_rows: list[dict] = []
    residual_refresh_rows: list[dict] = []
    distillation_eval_rows = build_distillation_residual_eval_rows()
    method_default_rows = build_unified_method_defaults_rows()
    combined_default_rows = build_combined_default_rows()
    method_parameter_rows = build_method_parameter_rows()
    cache: list[dict] = []

    system_grouped: dict[str, list[dict]] = {"distillation": [], "polymer": []}

    for spec in RUN_SPECS:
        bundle = load_pickle(spec.run_path)
        minmax = load_pickle(spec.minmax_path)
        cache.append({"spec": spec, "bundle": bundle, "minmax": minmax})

        xhatdhat = np.asarray(bundle["xhatdhat"][:, :-1], float)
        y_sp = np.asarray(bundle["y_sp"], float)
        scaled = scale_states(xhatdhat, np.asarray(minmax["min_s"], float), np.asarray(minmax["max_s"], float))
        late_mask = late_segment_mask(y_sp)

        n_states = xhatdhat.shape[0]
        n_outputs = int(bundle["n_outputs"])
        n_phys = n_states - n_outputs
        state_labels = DISTILLATION_STATE_LABELS if spec.system == "distillation" else POLYMER_STATE_LABELS

        full_span_ratio = quantile_span(scaled) / np.maximum(2.0, 1e-12)
        late_span_ratio = quantile_span(scaled[late_mask]) / np.maximum(2.0, 1e-12)
        late_full_ratio = late_span_ratio / np.maximum(full_span_ratio, 1e-12)
        outside_frac = np.mean(
            (xhatdhat < np.asarray(minmax["min_s"], float)[:, None]) | (xhatdhat > np.asarray(minmax["max_s"], float)[:, None]),
            axis=1,
        )

        phys_peak_to_late = segment_peak_to_late(xhatdhat[:n_phys, :].T, y_sp)
        dhat_peak_to_late = segment_peak_to_late(xhatdhat[n_phys:, :].T, y_sp)

        innovation = bundle.get("innovation_log")
        innovation_clip_frac = None
        innovation_peak_to_late = None
        if innovation is not None:
            innovation = np.asarray(innovation, float)
            innovation_clip_frac = float(np.mean(np.abs(innovation) >= 2.999999))
            innovation_peak_to_late = float(np.median(segment_peak_to_late(innovation, y_sp)))

        tracking = bundle.get("tracking_error_log")
        tracking_clip_frac = None
        tracking_peak_to_late = None
        if tracking is not None:
            tracking = np.asarray(tracking, float)
            tracking_clip_frac = float(np.mean(np.abs(tracking) >= 2.999999))
            tracking_peak_to_late = float(np.median(segment_peak_to_late(tracking, y_sp)))

        rho = bundle.get("rho_log")
        rho_one_frac = None if rho is None else float(np.mean(np.asarray(rho, float) >= 0.999999))

        run_row = {
            "system": spec.system,
            "label": spec.label,
            "run_path": str(spec.run_path.relative_to(REPO_ROOT)),
            "state_mode": str(bundle.get("state_mode")),
            "algorithm": str(bundle.get("algorithm")),
            "method_family": str(bundle.get("method_family")),
            "n_states": n_states,
            "n_phys": n_phys,
            "phys_late_full_med": float(np.median(late_full_ratio[:n_phys])),
            "dhat_late_full_med": float(np.median(late_full_ratio[n_phys:])),
            "phys_peak_to_late_med": float(np.median(phys_peak_to_late)),
            "dhat_peak_to_late_med": float(np.median(dhat_peak_to_late)),
            "outside_max": float(np.max(outside_frac)),
            "innovation_clip_frac": innovation_clip_frac,
            "tracking_clip_frac": tracking_clip_frac,
            "rho_one_frac": rho_one_frac,
            "innovation_peak_to_late_med": innovation_peak_to_late,
            "tracking_peak_to_late_med": tracking_peak_to_late,
            "full_span_ratio": full_span_ratio,
            "late_full_ratio": late_full_ratio,
            "outside_frac": outside_frac,
        }
        system_grouped[spec.system].append(run_row)

        summary_rows.append(
            {
                k: v
                for k, v in run_row.items()
                if k not in {"full_span_ratio", "late_full_ratio", "outside_frac"}
            }
        )

        innovation_raw, tracking_raw = reconstruct_raw_mismatch_features(bundle)
        if innovation_raw is not None and tracking_raw is not None:
            raw_feature_rows.append(
                {
                    "system": spec.system,
                    "label": spec.label,
                    "run_path": str(spec.run_path.relative_to(REPO_ROOT)),
                    "innovation_raw": innovation_raw,
                    "tracking_raw": tracking_raw,
                    "innovation_gt1_frac": float(np.mean(np.abs(innovation_raw) > 1.0)),
                    "innovation_gt3_frac": float(np.mean(np.abs(innovation_raw) > 3.0)),
                    "innovation_gt10_frac": float(np.mean(np.abs(innovation_raw) > 10.0)),
                    "tracking_gt1_frac": float(np.mean(np.abs(tracking_raw) > 1.0)),
                    "tracking_gt3_frac": float(np.mean(np.abs(tracking_raw) > 3.0)),
                    "tracking_gt10_frac": float(np.mean(np.abs(tracking_raw) > 10.0)),
                    "innovation_p99_ch1": float(np.percentile(np.abs(innovation_raw[:, 0]), 99)),
                    "innovation_p99_ch2": float(np.percentile(np.abs(innovation_raw[:, 1]), 99)),
                    "tracking_p99_ch1": float(np.percentile(np.abs(tracking_raw[:, 0]), 99)),
                    "tracking_p99_ch2": float(np.percentile(np.abs(tracking_raw[:, 1]), 99)),
                }
            )

        if bundle.get("rho_log") is not None:
            raw_res = np.asarray(bundle["delta_u_res_raw_log"], float)
            exec_res = np.asarray(bundle["delta_u_res_exec_log"], float)
            raw_norm = np.linalg.norm(raw_res, axis=1)
            exec_norm = np.linalg.norm(exec_res, axis=1)
            valid = raw_norm > 1e-9
            ratio = exec_norm[valid] / raw_norm[valid]
            rho_rows.append(
                {
                    "system": spec.system,
                    "label": spec.label,
                    "run_path": str(spec.run_path.relative_to(REPO_ROOT)),
                    "rho_one_frac": float(np.mean(np.asarray(bundle["rho_log"], float) >= 0.999999)),
                    "authority_projection_frac": float(np.mean(np.asarray(bundle["projection_due_to_authority_log"], float))),
                    "median_exec_raw_ratio": float(np.median(ratio)) if ratio.size else 0.0,
                    "p95_exec_raw_ratio": float(np.percentile(ratio, 95)) if ratio.size else 0.0,
                }
            )

        for idx, state_label in enumerate(state_labels):
            state_rows.append(
                {
                    "system": spec.system,
                    "run_label": spec.label.replace("\n", " "),
                    "state": state_label,
                    "state_index": idx,
                    "state_mode": bundle.get("state_mode"),
                    "full_span_ratio": float(full_span_ratio[idx]),
                    "late_full_ratio": float(late_full_ratio[idx]),
                    "outside_frac": float(outside_frac[idx]),
                    "saved_min": float(np.asarray(minmax["min_s"], float)[idx]),
                    "saved_max": float(np.asarray(minmax["max_s"], float)[idx]),
                }
            )

    save_csv(
        DATA_ROOT / "run_summary.csv",
        summary_rows,
        [
            "system",
            "label",
            "run_path",
            "state_mode",
            "algorithm",
            "method_family",
            "n_states",
            "n_phys",
            "phys_late_full_med",
            "dhat_late_full_med",
            "phys_peak_to_late_med",
            "dhat_peak_to_late_med",
            "outside_max",
            "innovation_clip_frac",
            "tracking_clip_frac",
            "rho_one_frac",
            "innovation_peak_to_late_med",
            "tracking_peak_to_late_med",
        ],
    )
    save_csv(
        DATA_ROOT / "state_summary.csv",
        state_rows,
        [
            "system",
            "run_label",
            "state",
            "state_index",
            "state_mode",
            "full_span_ratio",
            "late_full_ratio",
            "outside_frac",
            "saved_min",
            "saved_max",
        ],
    )
    save_csv(
        DATA_ROOT / "raw_mismatch_feature_summary.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k not in {"innovation_raw", "tracking_raw"}
            }
            for row in raw_feature_rows
        ],
        [
            "system",
            "label",
            "run_path",
            "innovation_gt1_frac",
            "innovation_gt3_frac",
            "innovation_gt10_frac",
            "tracking_gt1_frac",
            "tracking_gt3_frac",
            "tracking_gt10_frac",
            "innovation_p99_ch1",
            "innovation_p99_ch2",
            "tracking_p99_ch1",
            "tracking_p99_ch2",
        ],
    )
    save_csv(
        DATA_ROOT / "rho_authority_summary.csv",
        rho_rows,
        [
            "system",
            "label",
            "run_path",
            "rho_one_frac",
            "authority_projection_frac",
            "median_exec_raw_ratio",
            "p95_exec_raw_ratio",
        ],
    )
    residual_refresh_rows = plot_residual_refresh_offline_replay(cache)
    save_csv(
        DATA_ROOT / "residual_refresh_offline_summary.csv",
        residual_refresh_rows,
        [
            "system",
            "label",
            "run_path",
            "legacy_rho_one_frac",
            "new_rho_one_frac",
            "new_deadband_frac",
            "legacy_tail_residual_norm",
            "new_tail_residual_norm",
            "legacy_authority_frac",
            "new_authority_frac",
        ],
    )
    save_csv(
        DATA_ROOT / "distillation_residual_eval_summary.csv",
        distillation_eval_rows,
        [
            "label",
            "run_path",
            "eval_mae_y1",
            "eval_mae_y2",
            "tail_mae_y1",
            "tail_mae_y2",
            "tail_mean_output2_bias_vs_sp",
            "tail_mean_output2_bias_vs_baseline",
            "tail_mean_abs_u_gap_vs_baseline_1",
            "tail_mean_abs_u_gap_vs_baseline_2",
            "tail_mean_rho",
            "tail_rho_one_frac",
            "tail_mean_residual_norm",
            "tail_mean_abs_u_minus_ubase_1",
            "tail_mean_abs_u_minus_ubase_2",
            "tail_mean_abs_ubase_minus_baseline_1",
            "tail_mean_abs_ubase_minus_baseline_2",
            "tail_mean_max_abs_tracking",
        ],
    )
    save_csv(
        DATA_ROOT / "unified_method_defaults_summary.csv",
        method_default_rows,
        [
            "system",
            "family_key",
            "family",
            "default_state_mode",
            "mismatch_default",
            "rho_default_active",
            "guardrail_type",
            "changes_horizon",
            "changes_model",
            "changes_weights",
            "changes_direct_input",
            "online_identification",
            "bounded_action_set",
            "method_specific_guard",
        ],
    )
    save_csv(
        DATA_ROOT / "combined_default_modes_summary.csv",
        combined_default_rows,
        [
            "system",
            "horizon_mismatch_default",
            "matrix_mismatch_default",
            "weights_mismatch_default",
            "residual_mismatch_default",
            "rho_default_active",
            "residual_agent_kind",
        ],
    )
    save_csv(
        DATA_ROOT / "method_parameter_summary.csv",
        method_parameter_rows,
        [
            "system",
            "parameter_group",
            "channel",
            "value",
        ],
    )

    plot_minmax_widths()
    plot_system_heatmaps("distillation", system_grouped["distillation"], DISTILLATION_STATE_LABELS)
    plot_system_heatmaps("polymer", system_grouped["polymer"], POLYMER_STATE_LABELS)
    plot_mismatch_summary(summary_rows)
    plot_unified_method_feature_matrix(method_default_rows)
    plot_combined_default_modes(combined_default_rows)
    plot_residual_method_parameter_comparison(method_parameter_rows)
    plot_mismatch_scale_parameter_comparison(method_parameter_rows)
    plot_vecnormalize_comparison(cache)
    plot_normalization_sensitivity_by_state(cache)
    plot_raw_exceedance_and_transform_examples(raw_feature_rows)
    plot_transform_curves()
    plot_mismatch_transform_timeseries(raw_feature_rows)
    plot_rho_candidate_mappings()
    plot_rho_diagnostics(rho_rows)
    plot_distillation_latest_residual_eval_episode()
    plot_distillation_residual_eval_summary(distillation_eval_rows)
    plot_representative_trace(
        RunSpec(
            system="distillation",
            label="Residual TD3 mismatch",
            rel_path=r"Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260420_180906/input_data.pkl",
            minmax_rel_path=r"Distillation/Data/min_max_states.pickle",
        ),
        label_prefix="Residual TD3 mismatch",
        state_labels=DISTILLATION_STATE_LABELS,
    )
    plot_representative_trace(
        RunSpec(
            system="polymer",
            label="Residual TD3 mismatch",
            rel_path=r"Polymer/Results/td3_residual_disturb/20260413_004620/input_data.pkl",
            minmax_rel_path=r"Polymer/Data/min_max_states.pickle",
        ),
        label_prefix="Residual TD3 mismatch",
        state_labels=POLYMER_STATE_LABELS,
    )


if __name__ == "__main__":
    main()
