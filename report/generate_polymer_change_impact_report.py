from __future__ import annotations

import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from systems.distillation.notebook_params import get_distillation_notebook_defaults
from systems.polymer.notebook_params import get_polymer_notebook_defaults

REPORT_ROOT = REPO_ROOT / "report" / "polymer_change_impact"
FIG_ROOT = REPORT_ROOT / "figures"
DATA_ROOT = REPORT_ROOT / "data"

BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
POLYMER_MINMAX_PATH = REPO_ROOT / "Polymer" / "Data" / "min_max_states.pickle"

OUTPUT_LABELS = ["eta", "T"]
FAMILY_TITLES = {
    "baseline": "Baseline MPC",
    "residual": "Residual",
    "matrix": "Matrix",
    "structured": "Structured Matrix",
    "reidentification": "Reidentification",
}
FAMILY_COLORS = {
    "baseline": "#4A4A4A",
    "residual": "#1F77B4",
    "matrix": "#F28E2B",
    "structured": "#59A14F",
    "reidentification": "#E15759",
}
VARIANT_HATCH = {
    "baseline": "",
    "legacy": "",
    "refreshed_current_observer": "//",
    "refreshed_legacy_observer": "xx",
}


@dataclass(frozen=True)
class RunSpec:
    family: str
    variant_key: str
    variant_label: str
    run_label: str
    path: Path
    is_refreshed: bool

    @property
    def run_id(self) -> str:
        return f"{self.family}:{self.variant_key}"

    @property
    def short_label(self) -> str:
        base = {
            "residual": "Residual",
            "matrix": "Matrix",
            "structured": "Structured",
            "reidentification": "ReID",
        }[self.family]
        variant = {
            "legacy": "legacy",
            "refreshed_current_observer": "refresh current",
            "refreshed_legacy_observer": "refresh legacy",
        }[self.variant_key]
        return f"{base}\n{variant}"


RUN_SPECS = [
    RunSpec(
        family="residual",
        variant_key="legacy",
        variant_label="legacy",
        run_label="Residual legacy",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260413_004620" / "input_data.pkl",
        is_refreshed=False,
    ),
    RunSpec(
        family="residual",
        variant_key="refreshed_current_observer",
        variant_label="refreshed (current-observer run)",
        run_label="Residual refreshed",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260420_225631" / "input_data.pkl",
        is_refreshed=True,
    ),
    RunSpec(
        family="matrix",
        variant_key="legacy",
        variant_label="legacy",
        run_label="Matrix legacy",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260411_011134" / "input_data.pkl",
        is_refreshed=False,
    ),
    RunSpec(
        family="matrix",
        variant_key="refreshed_current_observer",
        variant_label="refreshed (current-observer run)",
        run_label="Matrix refreshed current-observer",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260420_215528" / "input_data.pkl",
        is_refreshed=True,
    ),
    RunSpec(
        family="matrix",
        variant_key="refreshed_legacy_observer",
        variant_label="refreshed (legacy-observer run)",
        run_label="Matrix refreshed legacy-observer",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260420_234944" / "input_data.pkl",
        is_refreshed=True,
    ),
    RunSpec(
        family="structured",
        variant_key="legacy",
        variant_label="legacy",
        run_label="Structured matrix legacy",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260409_193654" / "input_data.pkl",
        is_refreshed=False,
    ),
    RunSpec(
        family="structured",
        variant_key="refreshed_legacy_observer",
        variant_label="refreshed (legacy-observer run)",
        run_label="Structured matrix refreshed legacy-observer",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260420_235100" / "input_data.pkl",
        is_refreshed=True,
    ),
    RunSpec(
        family="reidentification",
        variant_key="legacy",
        variant_label="legacy",
        run_label="Reidentification legacy",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_reidentification_disturb" / "20260415_120803" / "input_data.pkl",
        is_refreshed=False,
    ),
    RunSpec(
        family="reidentification",
        variant_key="refreshed_legacy_observer",
        variant_label="refreshed (legacy-observer run)",
        run_label="Reidentification refreshed legacy-observer",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_reidentification_disturb" / "20260420_234346" / "input_data.pkl",
        is_refreshed=True,
    ),
]


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


def detect_setpoint_change_segments(y_sp: np.ndarray) -> list[tuple[int, int]]:
    change_idx = np.flatnonzero(np.any(np.abs(np.diff(y_sp, axis=0)) > 1e-12, axis=1)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(y_sp)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s > 5]


def late_segment_mask(y_sp: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(y_sp), dtype=bool)
    segments = detect_setpoint_change_segments(y_sp)
    if not segments:
        mask[max(0, len(y_sp) - min(40, len(y_sp))) :] = True
        return mask
    for start, end in segments:
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
    if not np.any(late_mask):
        late_mask[:] = True
    full_span = quantile_span(states)
    late_span = quantile_span(states[late_mask])
    return {
        "full_span_med": float(np.median(full_span)),
        "late_span_med": float(np.median(late_span)),
        "late_full_ratio_med": float(np.median(late_span / np.maximum(full_span, 1e-12))),
    }


def exact_clip_fraction(arr: np.ndarray, clip_value: float = 3.0) -> np.ndarray:
    return np.mean(np.isclose(np.abs(arr), clip_value, atol=1e-8), axis=0)


def nanmean_pair(values: list[float]) -> float:
    arr = np.asarray(values, float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def percentile_stats(arr: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(arr, float)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99))


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
        prefix = key.replace("_log", "")
        if bundle.get(key) is not None:
            arr = np.asarray(bundle[key], float)
            metrics[f"{prefix}_p99_out1"] = float(np.percentile(np.abs(arr[:, 0]), 99))
            metrics[f"{prefix}_p99_out2"] = float(np.percentile(np.abs(arr[:, 1]), 99))
            metrics[f"{prefix}_exact3_out1"] = float(exact_clip_fraction(arr)[0])
            metrics[f"{prefix}_exact3_out2"] = float(exact_clip_fraction(arr)[1])
        else:
            metrics[f"{prefix}_p99_out1"] = np.nan
            metrics[f"{prefix}_p99_out2"] = np.nan
            metrics[f"{prefix}_exact3_out1"] = np.nan
            metrics[f"{prefix}_exact3_out2"] = np.nan

    for key in ("innovation_raw_log", "tracking_error_raw_log"):
        prefix = key.replace("_raw_log", "_raw")
        if bundle.get(key) is not None:
            arr = np.asarray(bundle[key], float)
            metrics[f"{prefix}_p99_out1"] = float(np.percentile(np.abs(arr[:, 0]), 99))
            metrics[f"{prefix}_p99_out2"] = float(np.percentile(np.abs(arr[:, 1]), 99))
            metrics[f"{prefix}_gt3_out1"] = float(np.mean(np.abs(arr[:, 0]) > 3.0))
            metrics[f"{prefix}_gt3_out2"] = float(np.mean(np.abs(arr[:, 1]) > 3.0))
        else:
            metrics[f"{prefix}_p99_out1"] = np.nan
            metrics[f"{prefix}_p99_out2"] = np.nan
            metrics[f"{prefix}_gt3_out1"] = np.nan
            metrics[f"{prefix}_gt3_out2"] = np.nan

    metrics["innovation_p99_mean"] = nanmean_pair([metrics["innovation_p99_out1"], metrics["innovation_p99_out2"]])
    metrics["tracking_error_p99_mean"] = nanmean_pair([metrics["tracking_error_p99_out1"], metrics["tracking_error_p99_out2"]])
    metrics["innovation_raw_p99_mean"] = nanmean_pair([metrics["innovation_raw_p99_out1"], metrics["innovation_raw_p99_out2"]])
    metrics["tracking_error_raw_p99_mean"] = nanmean_pair([metrics["tracking_error_raw_p99_out1"], metrics["tracking_error_raw_p99_out2"]])
    metrics["innovation_exact3_mean"] = nanmean_pair([metrics["innovation_exact3_out1"], metrics["innovation_exact3_out2"]])
    metrics["tracking_error_exact3_mean"] = nanmean_pair([metrics["tracking_error_exact3_out1"], metrics["tracking_error_exact3_out2"]])

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

    if bundle.get("mapped_multiplier_log") is not None:
        mapped = np.asarray(bundle["mapped_multiplier_log"], float)
        metrics["multiplier_abs_mean"] = float(np.mean(np.abs(mapped)))
        metrics["multiplier_abs_p95"] = float(np.percentile(np.abs(mapped), 95))
        metrics["multiplier_step_norm_p95"] = float(np.percentile(np.linalg.norm(mapped, axis=1), 95))
    else:
        metrics["multiplier_abs_mean"] = np.nan
        metrics["multiplier_abs_p95"] = np.nan
        metrics["multiplier_step_norm_p95"] = np.nan

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


def compute_reid_metrics(bundle: dict) -> dict:
    if bundle.get("id_update_event_log") is None:
        return {
            "candidate_valid_frac": np.nan,
            "fallback_frac": np.nan,
            "update_event_frac": np.nan,
            "update_success_frac": np.nan,
            "condition_median": np.nan,
            "condition_p95": np.nan,
            "condition_p99": np.nan,
            "residual_norm_median": np.nan,
            "residual_norm_p95": np.nan,
            "residual_norm_p99": np.nan,
            "residual_ratio_median": np.nan,
            "residual_ratio_p95": np.nan,
            "residual_ratio_p99": np.nan,
            "eta_A_requested_p95": np.nan,
            "eta_A_applied_p95": np.nan,
            "eta_B_requested_p95": np.nan,
            "eta_B_applied_p95": np.nan,
            "blend_A_p50": np.nan,
            "blend_B_p50": np.nan,
            "source0_frac": np.nan,
            "source1_frac": np.nan,
            "source2_frac": np.nan,
        }

    condition_log = np.asarray(bundle.get("id_condition_number_log", []), float)
    residual_norm_log = np.asarray(bundle.get("id_residual_norm_log", []), float)
    residual_ratio_log = bundle.get("id_residual_ratio_full_log")
    if residual_ratio_log is None:
        residual_ratio_median = np.nan
        residual_ratio_p95 = np.nan
        residual_ratio_p99 = np.nan
    else:
        residual_ratio_median, residual_ratio_p95, residual_ratio_p99 = percentile_stats(residual_ratio_log)
    condition_median, condition_p95, condition_p99 = percentile_stats(condition_log)
    residual_norm_median, residual_norm_p95, residual_norm_p99 = percentile_stats(residual_norm_log)

    source_codes = np.asarray(bundle.get("id_source_code_log", []), int)

    def pct95(name: str) -> float:
        values = bundle.get(name)
        if values is None:
            return np.nan
        return float(np.percentile(np.asarray(values, float), 95))

    def pct50(name: str) -> float:
        values = bundle.get(name)
        if values is None:
            return np.nan
        return float(np.percentile(np.asarray(values, float), 50))

    return {
        "candidate_valid_frac": float(np.mean(np.asarray(bundle["id_candidate_valid_log"], float))),
        "fallback_frac": float(np.mean(np.asarray(bundle["id_fallback_log"], float))),
        "update_event_frac": float(np.mean(np.asarray(bundle["id_update_event_log"], float))),
        "update_success_frac": float(np.mean(np.asarray(bundle["id_update_success_log"], float))),
        "condition_median": condition_median,
        "condition_p95": condition_p95,
        "condition_p99": condition_p99,
        "residual_norm_median": residual_norm_median,
        "residual_norm_p95": residual_norm_p95,
        "residual_norm_p99": residual_norm_p99,
        "residual_ratio_median": residual_ratio_median,
        "residual_ratio_p95": residual_ratio_p95,
        "residual_ratio_p99": residual_ratio_p99,
        "eta_A_requested_p95": pct95("eta_A_requested_log"),
        "eta_A_applied_p95": pct95("eta_A_log"),
        "eta_B_requested_p95": pct95("eta_B_requested_log"),
        "eta_B_applied_p95": pct95("eta_B_log"),
        "blend_A_p50": pct50("blend_validity_scale_A_log"),
        "blend_B_p50": pct50("blend_validity_scale_B_log"),
        "source0_frac": float(np.mean(source_codes == 0)) if source_codes.size else np.nan,
        "source1_frac": float(np.mean(source_codes == 1)) if source_codes.size else np.nan,
        "source2_frac": float(np.mean(source_codes == 2)) if source_codes.size else np.nan,
    }


def extract_reid_default_surface(system_name: str, defaults: dict) -> dict:
    reid = dict(defaults["reidentification"])
    theta_low_b = float(reid["theta_low_B"])
    theta_high_b = float(reid["theta_high_B"])
    guard_enabled = all(
        key in reid
        for key in (
            "guard_validation_fraction",
            "guard_min_validation_samples",
            "guard_min_train_samples",
            "max_theta_clipped_fraction",
            "max_condition_number",
            "max_validation_residual_ratio",
            "max_full_residual_ratio",
        )
    )
    return {
        "label": f"{system_name} current",
        "system": system_name,
        "id_window": int(reid["id_window"]),
        "id_update_period": int(reid["id_update_period"]),
        "rank_B": int(reid["rank_B"]),
        "lambda_prev_B": float(reid["lambda_prev_B"]),
        "lambda_0_B": float(reid["lambda_0_B"]),
        "theta_B_halfwidth": float(max(abs(theta_low_b), abs(theta_high_b))),
        "freeze_identification_during_warm_start": bool(reid.get("freeze_identification_during_warm_start", False)),
        "guard_enabled": bool(guard_enabled),
        "candidate_guard_mode": str(reid.get("candidate_guard_mode", "n/a")),
        "blend_validity_mode": str(reid.get("blend_validity_mode", "n/a")),
        "force_eta_constant": reid.get("force_eta_constant"),
        "has_small_fixed_eta_isolation": bool(reid.get("force_eta_constant") is not None),
        "normalize_blend_extras": bool(reid.get("normalize_blend_extras", False)),
        "blend_extra_clip": float(reid.get("blend_extra_clip", np.nan)),
        "blend_residual_scale": float(reid.get("blend_residual_scale", np.nan)),
        "eta_tau_B": float(reid.get("eta_tau_B", np.nan)),
        "guard_validation_fraction": float(reid.get("guard_validation_fraction", np.nan)),
        "guard_min_validation_samples": int(reid.get("guard_min_validation_samples", 0)) if reid.get("guard_min_validation_samples") is not None else np.nan,
        "guard_min_train_samples": int(reid.get("guard_min_train_samples", 0)) if reid.get("guard_min_train_samples") is not None else np.nan,
        "max_theta_clipped_fraction": float(reid.get("max_theta_clipped_fraction", np.nan)),
        "max_condition_number": float(reid.get("max_condition_number", np.nan)),
        "max_validation_residual_ratio": float(reid.get("max_validation_residual_ratio", np.nan)),
        "max_full_residual_ratio": float(reid.get("max_full_residual_ratio", np.nan)),
    }


def build_reid_extension_rows() -> list[dict]:
    polymer_defaults = get_polymer_notebook_defaults("reidentification")
    distillation_defaults = get_distillation_notebook_defaults("reidentification")
    polymer_current = extract_reid_default_surface("polymer", polymer_defaults)
    distillation_current = extract_reid_default_surface("distillation", distillation_defaults)

    polymer_next_step = dict(polymer_current)
    polymer_next_step.update(
        {
            "label": "polymer next-step",
            "id_window": 160,
            "id_update_period": 20,
            "rank_B": 1,
            "lambda_prev_B": 5.0e-1,
            "lambda_0_B": 5.0e-3,
            "theta_B_halfwidth": 0.04,
            "freeze_identification_during_warm_start": True,
            "guard_enabled": True,
            "candidate_guard_mode": "fro_validation_clip",
            "blend_validity_mode": "off (first) / diagnostic_fade later",
            "force_eta_constant": [0.05, 0.05],
            "has_small_fixed_eta_isolation": True,
            "normalize_blend_extras": False,
            "blend_extra_clip": 1.0e6,
            "blend_residual_scale": 1.0e6,
            "eta_tau_B": 1.0,
            "guard_validation_fraction": 0.25,
            "guard_min_validation_samples": 32,
            "guard_min_train_samples": 80,
            "max_theta_clipped_fraction": 0.05,
            "max_condition_number": 3.0e4,
            "max_validation_residual_ratio": 0.995,
            "max_full_residual_ratio": 1.0,
        }
    )
    return [polymer_current, distillation_current, polymer_next_step]


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def pct_change(new_value: float, old_value: float) -> str:
    if old_value != old_value or abs(old_value) < 1e-12 or new_value != new_value:
        return "n/a"
    return f"{100.0 * (new_value - old_value) / old_value:+.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def image_lines(filename: str, alt: str) -> list[str]:
    return [f"![{alt}](./polymer_change_impact/figures/{filename})", ""]


def add_bar_annotations(ax, values: list[float]) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-12)
    for idx, value in enumerate(values):
        if value != value:
            continue
        ax.text(idx, value + 0.02 * span, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)


def plot_performance_overview(summary_rows: list[dict], state_rows: list[dict], spec_map: dict[str, RunSpec]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    baseline_row = next(row for row in summary_rows if row["family"] == "baseline")
    run_rows = [row for row in summary_rows if row["family"] != "baseline"]
    labels = ["Baseline\nMPC"] + [spec_map[row["run_id"]].short_label for row in run_rows]
    colors = [FAMILY_COLORS["baseline"]] + [FAMILY_COLORS[row["family"]] for row in run_rows]
    hatches = [""] + [VARIANT_HATCH[row["variant_key"]] for row in run_rows]

    mae_values = [baseline_row["phys_mae_tail_mean"]] + [row["phys_mae_tail_mean"] for row in run_rows]
    reward_values = [baseline_row["reward_final20_mean"]] + [row["reward_final20_mean"] for row in run_rows]
    delta_values = [0.0] + [row["tail_phys_mae_delta_vs_baseline"] for row in run_rows]

    ax = axs[0, 0]
    x = np.arange(len(labels))
    bars = ax.bar(x, mae_values, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_title("Tail physical MAE mean across saved polymer runs")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axs[0, 1]
    bars = ax.bar(x, reward_values, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_title("Final-20 sub-episode reward across saved polymer runs")
    ax.set_ylabel("avg reward")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axs[1, 0]
    bars = ax.bar(x, delta_values, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Tail physical MAE delta vs baseline MPC")
    ax.set_ylabel("MAE delta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axs[1, 1]
    refreshed_ids = [run_id for run_id, spec in spec_map.items() if spec.is_refreshed]
    refreshed_labels = []
    gains = []
    gain_colors = []
    gain_hatches = []
    for run_id in refreshed_ids:
        actual = next(row for row in state_rows if row["run_id"] == run_id and row["view"] == "actual")
        fixed = next(row for row in state_rows if row["run_id"] == run_id and row["view"] == "fixed counterfactual")
        refreshed_labels.append(spec_map[run_id].short_label)
        gains.append(actual["full_span_med"] / max(fixed["full_span_med"], 1e-12))
        gain_colors.append(FAMILY_COLORS[spec_map[run_id].family])
        gain_hatches.append(VARIANT_HATCH[spec_map[run_id].variant_key])
    x_gain = np.arange(len(refreshed_labels))
    bars = ax.bar(x_gain, gains, color=gain_colors)
    for bar, hatch in zip(bars, gain_hatches):
        bar.set_hatch(hatch)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Running normalization full-span gain vs fixed-minmax counterfactual")
    ax.set_ylabel("gain")
    ax.set_xticks(x_gain)
    ax.set_xticklabels(refreshed_labels, rotation=25, ha="right")

    fig.savefig(FIG_ROOT / "polymer_change_performance_overview.png", dpi=220)
    plt.close(fig)


def plot_family_tail_traces(baseline_bundle: dict, runs: dict[str, dict], spec_map: dict[str, RunSpec]) -> None:
    fig, axs = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    tail_n = 400
    baseline_y = y_array(baseline_bundle)[-tail_n:]
    baseline_sp = setpoint_phys(baseline_bundle)[-tail_n:]
    time = np.arange(tail_n)

    family_rows = [
        ("residual", ["residual:legacy", "residual:refreshed_current_observer"]),
        ("matrix", ["matrix:legacy", "matrix:refreshed_current_observer", "matrix:refreshed_legacy_observer"]),
        ("structured", ["structured:legacy", "structured:refreshed_legacy_observer"]),
        ("reidentification", ["reidentification:legacy", "reidentification:refreshed_legacy_observer"]),
    ]
    line_colors = {
        "baseline": "#4A4A4A",
        "legacy": "#1F77B4",
        "refreshed_current_observer": "#F28E2B",
        "refreshed_legacy_observer": "#59A14F",
    }

    for row_idx, (family, run_ids) in enumerate(family_rows):
        for out_idx, output_label in enumerate(OUTPUT_LABELS):
            ax = axs[row_idx, out_idx]
            ax.plot(time, baseline_sp[:, out_idx], color="black", linestyle="--", linewidth=1.2, label="setpoint")
            ax.plot(time, baseline_y[:, out_idx], color=line_colors["baseline"], linewidth=1.8, label="baseline MPC")
            for run_id in run_ids:
                spec = spec_map[run_id]
                run_y = y_array(runs[run_id]["bundle"])[-tail_n:]
                ax.plot(
                    time,
                    run_y[:, out_idx],
                    color=line_colors[spec.variant_key],
                    linewidth=1.4,
                    label=spec.variant_label,
                )
            ax.set_title(f"{FAMILY_TITLES[family]} tail trace: {output_label}")
            if out_idx == 0:
                ax.set_ylabel("physical output")
            if row_idx == len(family_rows) - 1:
                ax.set_xlabel("Step")
            if row_idx == 0 and out_idx == 1:
                ax.legend(loc="best", fontsize=8)

    fig.savefig(FIG_ROOT / "polymer_change_family_tail_traces.png", dpi=220)
    plt.close(fig)


def plot_state_conditioning(state_rows: list[dict], spec_map: dict[str, RunSpec]) -> None:
    refreshed_ids = [run_id for run_id, spec in spec_map.items() if spec.is_refreshed]
    labels = []
    full_gain = []
    late_gain = []
    colors = []
    hatches = []

    for run_id in refreshed_ids:
        actual = next(row for row in state_rows if row["run_id"] == run_id and row["view"] == "actual")
        fixed = next(row for row in state_rows if row["run_id"] == run_id and row["view"] == "fixed counterfactual")
        labels.append(spec_map[run_id].short_label)
        full_gain.append(actual["full_span_med"] / max(fixed["full_span_med"], 1e-12))
        late_gain.append(actual["late_span_med"] / max(fixed["late_span_med"], 1e-12))
        colors.append(FAMILY_COLORS[spec_map[run_id].family])
        hatches.append(VARIANT_HATCH[spec_map[run_id].variant_key])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(labels))

    bars = axs[0].bar(x, full_gain, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axs[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axs[0].set_title("Physical xhat full-span gain from running normalization")
    axs[0].set_ylabel("actual / fixed-minmax")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=25, ha="right")

    bars = axs[1].bar(x, late_gain, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axs[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axs[1].set_title("Physical xhat late-window span gain from running normalization")
    axs[1].set_ylabel("actual / fixed-minmax")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=25, ha="right")

    fig.savefig(FIG_ROOT / "polymer_change_state_conditioning.png", dpi=220)
    plt.close(fig)


def plot_feature_diagnostics(summary_rows: list[dict], spec_map: dict[str, RunSpec]) -> None:
    run_rows = [row for row in summary_rows if row["family"] != "baseline"]
    labels = [spec_map[row["run_id"]].short_label for row in run_rows]
    colors = [FAMILY_COLORS[row["family"]] for row in run_rows]
    hatches = [VARIANT_HATCH[row["variant_key"]] for row in run_rows]
    x = np.arange(len(run_rows))

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    panels = [
        ("innovation_exact3_mean", "Innovation exact |3| fraction", False),
        ("tracking_error_exact3_mean", "Tracking exact |3| fraction", False),
        ("innovation_raw_p99_mean", "Innovation raw p99 magnitude", True),
        ("tracking_error_raw_p99_mean", "Tracking raw p99 magnitude", True),
    ]

    for ax, (key, title, log_scale) in zip(axs.ravel(), panels):
        bars = ax.bar(x, [row[key] for row in run_rows], color=colors)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("value")

    fig.savefig(FIG_ROOT / "polymer_change_feature_diagnostics.png", dpi=220)
    plt.close(fig)


def plot_reid_health(reid_rows: list[dict]) -> None:
    legacy = next(row for row in reid_rows if row["variant_key"] == "legacy")
    refreshed = next(row for row in reid_rows if row["variant_key"] == "refreshed_legacy_observer")
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)

    labels = ["legacy", "refreshed"]
    colors = [FAMILY_COLORS["reidentification"], "#F28E2B"]
    x = np.arange(len(labels))

    frac_keys = [
        ("update_event_frac", "update event"),
        ("candidate_valid_frac", "candidate valid"),
        ("update_success_frac", "update success"),
        ("fallback_frac", "fallback"),
    ]
    width = 0.18
    for idx, (key, title) in enumerate(frac_keys):
        axs[0, 0].bar(
            x + (idx - 1.5) * width,
            [legacy[key], refreshed[key]],
            width=width,
            label=title,
        )
    axs[0, 0].set_title("Reidentification event fractions")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].set_ylabel("fraction")
    axs[0, 0].legend(fontsize=8, loc="upper right")

    cond_keys = [("condition_median", "median"), ("condition_p95", "p95"), ("condition_p99", "p99")]
    for idx, (key, title) in enumerate(cond_keys):
        axs[0, 1].bar(
            x + (idx - 1.0) * 0.24,
            [legacy[key], refreshed[key]],
            width=0.24,
            label=title,
        )
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_title("Condition number distribution")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].set_ylabel("condition number")
    axs[0, 1].legend(fontsize=8, loc="upper right")

    ratio_keys = [("residual_ratio_median", "median"), ("residual_ratio_p95", "p95"), ("residual_ratio_p99", "p99")]
    for idx, (key, title) in enumerate(ratio_keys):
        axs[0, 2].bar(
            x + (idx - 1.0) * 0.24,
            [legacy[key], refreshed[key]],
            width=0.24,
            label=title,
        )
    axs[0, 2].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axs[0, 2].set_title("Residual-ratio distribution (candidate / incumbent)")
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels(labels)
    axs[0, 2].set_ylabel("ratio")
    axs[0, 2].legend(fontsize=8, loc="upper right")

    eta_labels = ["A req p95", "A app p95", "B req p95", "B app p95"]
    legacy_eta = [
        legacy["eta_A_requested_p95"],
        legacy["eta_A_applied_p95"],
        legacy["eta_B_requested_p95"],
        legacy["eta_B_applied_p95"],
    ]
    refreshed_eta = [
        refreshed["eta_A_requested_p95"],
        refreshed["eta_A_applied_p95"],
        refreshed["eta_B_requested_p95"],
        refreshed["eta_B_applied_p95"],
    ]
    eta_x = np.arange(len(eta_labels))
    axs[1, 0].bar(eta_x - 0.18, legacy_eta, width=0.36, label="legacy")
    axs[1, 0].bar(eta_x + 0.18, refreshed_eta, width=0.36, label="refreshed")
    axs[1, 0].set_title("Eta requested vs applied p95")
    axs[1, 0].set_xticks(eta_x)
    axs[1, 0].set_xticklabels(eta_labels, rotation=15, ha="right")
    axs[1, 0].set_ylabel("eta")
    axs[1, 0].legend(fontsize=8, loc="upper right")

    norm_keys = [("residual_norm_median", "median"), ("residual_norm_p95", "p95"), ("residual_norm_p99", "p99")]
    for idx, (key, title) in enumerate(norm_keys):
        axs[1, 1].bar(
            x + (idx - 1.0) * 0.24,
            [legacy[key], refreshed[key]],
            width=0.24,
            label=title,
        )
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("Identification residual norm distribution")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].set_ylabel("residual norm")
    axs[1, 1].legend(fontsize=8, loc="upper right")

    axs[1, 2].bar(
        x,
        [legacy["source0_frac"], refreshed["source0_frac"]],
        label="source 0: no update event",
    )
    axs[1, 2].bar(
        x,
        [legacy["source1_frac"], refreshed["source1_frac"]],
        bottom=[legacy["source0_frac"], refreshed["source0_frac"]],
        label="source 1: candidate accepted",
    )
    axs[1, 2].bar(
        x,
        [legacy["source2_frac"], refreshed["source2_frac"]],
        bottom=[
            legacy["source0_frac"] + legacy["source1_frac"],
            refreshed["source0_frac"] + refreshed["source1_frac"],
        ],
        label="source 2: fallback branch",
    )
    axs[1, 2].set_title("Reidentification source-code fractions")
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels(labels)
    axs[1, 2].set_ylabel("fraction")
    axs[1, 2].legend(fontsize=7, loc="upper right")

    fig.savefig(FIG_ROOT / "polymer_reid_identification_health.png", dpi=220)
    plt.close(fig)


def plot_reid_extension_comparison(extension_rows: list[dict]) -> None:
    labels = [row["label"].replace(" ", "\n") for row in extension_rows]
    colors = ["#E15759", "#4E79A7", "#59A14F"]
    x = np.arange(len(labels))
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    width = 0.35
    axs[0, 0].bar(x - width / 2, [row["id_window"] for row in extension_rows], width=width, label="id_window")
    axs[0, 0].bar(x + width / 2, [row["id_update_period"] for row in extension_rows], width=width, label="id_update_period")
    axs[0, 0].set_title("Identification cadence")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].set_ylabel("steps")
    axs[0, 0].legend(fontsize=8, loc="upper right")

    axs[0, 1].bar(x - width / 2, [row["rank_B"] for row in extension_rows], width=width, label="rank_B")
    axs[0, 1].bar(x + width / 2, [row["theta_B_halfwidth"] for row in extension_rows], width=width, label="theta_B halfwidth")
    axs[0, 1].set_title("B-side structural restriction")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].legend(fontsize=8, loc="upper right")

    axs[1, 0].bar(x - width / 2, [row["lambda_prev_B"] for row in extension_rows], width=width, label="lambda_prev_B")
    axs[1, 0].bar(x + width / 2, [row["lambda_0_B"] for row in extension_rows], width=width, label="lambda_0_B")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("B-side regularization")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].legend(fontsize=8, loc="upper right")

    flag_names = [
        ("freeze_identification_during_warm_start", "freeze warm-start"),
        ("guard_enabled", "validation guard"),
        ("has_small_fixed_eta_isolation", "fixed small eta test"),
    ]
    flag_width = 0.22
    for idx, (key, title) in enumerate(flag_names):
        axs[1, 1].bar(
            x + (idx - 1.0) * flag_width,
            [1.0 if row[key] else 0.0 for row in extension_rows],
            width=flag_width,
            label=title,
        )
    axs[1, 1].set_title("Borrowable hardening levers")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].set_ylim(0.0, 1.15)
    axs[1, 1].set_ylabel("enabled")
    axs[1, 1].legend(fontsize=8, loc="upper right")

    fig.savefig(FIG_ROOT / "polymer_reid_extension_comparison.png", dpi=220)
    plt.close(fig)


def write_markdown(
    config_rows: list[dict],
    summary_rows: list[dict],
    state_rows: list[dict],
    reid_rows: list[dict],
    extension_rows: list[dict],
    spec_map: dict[str, RunSpec],
) -> None:
    summary_by_id = {row["run_id"]: row for row in summary_rows if row["family"] != "baseline"}
    baseline_row = next(row for row in summary_rows if row["family"] == "baseline")

    residual_legacy = summary_by_id["residual:legacy"]
    residual_refresh = summary_by_id["residual:refreshed_current_observer"]
    matrix_legacy = summary_by_id["matrix:legacy"]
    matrix_refresh_current = summary_by_id["matrix:refreshed_current_observer"]
    matrix_refresh_legacy = summary_by_id["matrix:refreshed_legacy_observer"]
    structured_legacy = summary_by_id["structured:legacy"]
    structured_refresh = summary_by_id["structured:refreshed_legacy_observer"]
    reid_legacy = summary_by_id["reidentification:legacy"]
    reid_refresh = summary_by_id["reidentification:refreshed_legacy_observer"]
    reid_legacy_health = next(row for row in reid_rows if row["run_id"] == "reidentification:legacy")
    reid_refresh_health = next(row for row in reid_rows if row["run_id"] == "reidentification:refreshed_legacy_observer")

    polymer_extension = next(row for row in extension_rows if row["label"] == "polymer next-step")
    polymer_current_ext = next(row for row in extension_rows if row["label"] == "polymer current")
    distillation_current_ext = next(row for row in extension_rows if row["label"] == "distillation current")

    lines: list[str] = [
        "# Polymer Change-Impact Report",
        "",
        "Date: 2026-04-21",
        "",
        "This report is now self-contained: the comparison tables, figures, explanations, and source notes are inside the report instead of being listed as external links only.",
        "",
        "It covers the polymer runs that are relevant to the recent method-scoped conditioning changes:",
        "",
        "- residual legacy vs refreshed",
        "- matrix legacy vs two refreshed runs",
        "- structured-matrix legacy vs refreshed",
        "- reidentification legacy vs refreshed",
        "- the disturbance baseline MPC for reference",
        "",
        "Two scope notes matter:",
        "",
        "- the first refreshed residual and first refreshed matrix runs were executed before the later observer rollback, so they still used `observer_update_alignment=\"current_measurement_corrector\"`",
        "- the later matrix, structured-matrix, and reidentification reruns used the final default `observer_update_alignment=\"legacy_previous_measurement\"`",
        "",
        "## Conditioning Mathematics",
        "",
        "The new polymer state-conditioning path changes the policy input in two places:",
        "",
        "1. Physical `xhat` block:",
        "   fixed min-max uses `z = 2 (x - x_min) / (x_max - x_min) - 1`",
        "   running normalization uses `z_t = clip((x_t - mu_t) / sqrt(var_t + eps), -c, c)`",
        "",
        "2. Mismatch extras (`innovation`, `tracking_error`):",
        "   legacy path used hard clipping near `+/-3`",
        "   new path uses `signed_log(e) = sign(e) log(1 + |e|)`",
        "",
        "For polymer, the motivation is that fixed min-max gives a very small local slope when the saved width is huge. Running z-score makes the local slope proportional to `1 / sigma_t`, which is exactly the idea used by Stable Baselines3 `VecNormalize` [SB3].",
        "",
        "For reidentification, the numerical issue is different. The identification window is solving a noisy regression problem. When the window is not informative enough, the information matrix becomes ill-conditioned, parameter updates become noise-sensitive, and a guard will reject almost every candidate. That is an inference from our logs, and it is consistent with the identification literature on persistency of excitation and regularized least squares [Mu2022] [Binette2016] [Wang2022] [Hochstenbach2011] [Lim2016].",
        "",
        "## Run Set And Configs",
        "",
    ]

    config_table_rows = []
    for row in config_rows:
        config_table_rows.append(
            [
                row["family"],
                row["variant_label"],
                f"`{row['run_path']}`",
                f"`{row['base_state_norm_mode']}`",
                f"`{row['mismatch_feature_transform_mode']}`",
                f"`{row['observer_update_alignment']}`",
                f"`{row['rho_mapping_mode']}`",
                f"`{row['candidate_guard_mode']}`",
                f"`{row['blend_validity_mode']}`",
            ]
        )
    lines += markdown_table(
        [
            "Family",
            "Variant",
            "Saved run",
            "base_state_norm_mode",
            "mismatch_transform",
            "observer_alignment",
            "rho_mapping",
            "candidate_guard",
            "blend_validity",
        ],
        config_table_rows,
    )
    lines += [
        "",
        "## Performance Summary",
        "",
    ]

    perf_table_rows = []
    perf_order = ["baseline"] + [spec.run_id for spec in RUN_SPECS]
    for key in perf_order:
        if key == "baseline":
            row = baseline_row
            perf_table_rows.append(
                [
                    "baseline",
                    "baseline",
                    fmt(row["phys_mae_tail_mean"]),
                    fmt(row["scaled_mae_tail_mean"]),
                    fmt(row["reward_final20_mean"]),
                    "n/a",
                    "n/a",
                    "n/a",
                ]
            )
            continue
        row = summary_by_id[key]
        perf_table_rows.append(
            [
                row["family"],
                spec_map[key].variant_label,
                fmt(row["phys_mae_tail_mean"]),
                fmt(row["scaled_mae_tail_mean"]),
                fmt(row["reward_final20_mean"]),
                fmt(row["tail_phys_mae_delta_vs_baseline"]),
                fmt(row["tracking_error_raw_p99_mean"]),
                fmt(row["tracking_error_exact3_mean"]),
            ]
        )
    lines += markdown_table(
        [
            "Family",
            "Variant",
            "Tail phys MAE mean",
            "Tail scaled MAE mean",
            "Final-20 reward",
            "Tail MAE delta vs baseline",
            "Tracking raw p99",
            "Tracking exact abs3 frac",
        ],
        perf_table_rows,
    )
    lines += [
        "",
        "The top-level result is mixed rather than uniform:",
        "",
        f"- Residual improved clearly: tail physical MAE mean moved from `{fmt(residual_legacy['phys_mae_tail_mean'])}` to `{fmt(residual_refresh['phys_mae_tail_mean'])}` ({pct_change(residual_refresh['phys_mae_tail_mean'], residual_legacy['phys_mae_tail_mean'])}), and final-20 reward moved from `{fmt(residual_legacy['reward_final20_mean'])}` to `{fmt(residual_refresh['reward_final20_mean'])}` ({pct_change(residual_refresh['reward_final20_mean'], residual_legacy['reward_final20_mean'])}).",
        f"- Matrix improved in the first refreshed run, but not robustly in the second. The current-observer rerun reached `{fmt(matrix_refresh_current['phys_mae_tail_mean'])}` tail physical MAE mean, while the later legacy-observer rerun moved back to `{fmt(matrix_refresh_legacy['phys_mae_tail_mean'])}`. Reward stayed better than legacy in both refreshed runs, but the tail-tracking gain was not stable.",
        f"- Structured matrix changed the policy input and improved reward modestly, but not tail MAE. Tail physical MAE mean stayed essentially flat, from `{fmt(structured_legacy['phys_mae_tail_mean'])}` to `{fmt(structured_refresh['phys_mae_tail_mean'])}`, while final-20 reward improved from `{fmt(structured_legacy['reward_final20_mean'])}` to `{fmt(structured_refresh['reward_final20_mean'])}` ({pct_change(structured_refresh['reward_final20_mean'], structured_legacy['reward_final20_mean'])}).",
        f"- Reidentification is the one family that did not benefit. Tail physical MAE mean worsened from `{fmt(reid_legacy['phys_mae_tail_mean'])}` to `{fmt(reid_refresh['phys_mae_tail_mean'])}`, and the refreshed run ended slightly worse than baseline MPC by `{fmt(reid_refresh['tail_phys_mae_delta_vs_baseline'])}`.",
        "",
    ]
    lines += image_lines("polymer_change_performance_overview.png", "Polymer performance overview across residual, matrix, structured matrix, and reidentification runs")
    lines += [
        "The overview figure shows two separate effects:",
        "",
        "- the policy input changed a lot in the refreshed polymer runs, because the normalized physical `xhat` block became much wider than the fixed-minmax counterfactual",
        "- the control benefit is family-dependent: residual benefits the most, matrix and structured matrix benefit partly, and reidentification does not",
        "",
        "## Family Tail Traces",
        "",
    ]
    lines += image_lines("polymer_change_family_tail_traces.png", "Tail traces for residual, matrix, structured matrix, and reidentification polymer runs")
    lines += [
        "The tail traces make the family-level behavior easier to see:",
        "",
        "- residual refreshed stays visibly tighter to the setpoint than residual legacy",
        "- matrix current-observer refresh is the strongest matrix run, while the later legacy-observer refresh gives back part of that tail-tracking gain",
        "- structured matrix refreshed is not a no-op, but its gain is milder than residual: reward improves, while tail MAE stays nearly unchanged",
        "- reidentification refreshed does not settle better than legacy, and it does not beat baseline MPC in the tail",
        "",
        "## State Conditioning",
        "",
    ]

    state_table_rows = []
    for run_id in [spec.run_id for spec in RUN_SPECS if spec.is_refreshed]:
        actual = next(row for row in state_rows if row["run_id"] == run_id and row["view"] == "actual")
        fixed = next(row for row in state_rows if row["run_id"] == run_id and row["view"] == "fixed counterfactual")
        state_table_rows.append(
            [
                spec_map[run_id].run_label,
                fmt(actual["full_span_med"]),
                fmt(fixed["full_span_med"]),
                fmt(actual["full_span_med"] / max(fixed["full_span_med"], 1e-12)),
                fmt(actual["late_span_med"]),
                fmt(fixed["late_span_med"]),
                fmt(actual["late_span_med"] / max(fixed["late_span_med"], 1e-12)),
            ]
        )
    lines += markdown_table(
        [
            "Run",
            "Actual full-span med",
            "Fixed CF full-span med",
            "Full-span gain",
            "Actual late-span med",
            "Fixed CF late-span med",
            "Late-span gain",
        ],
        state_table_rows,
    )
    lines += [
        "",
    ]
    lines += image_lines("polymer_change_state_conditioning.png", "Running-normalization gain on polymer physical xhat policy span")
    lines += [
        "This figure confirms that the observation-conditioning change is real, not cosmetic. The refreshed polymer runs all present a much wider physical `xhat` signal to the policy than the fixed-minmax counterfactual on the same saved trajectory. That is why it was reasonable to expect an effect from the new defaults, and the residual family is the clearest case where the better state spread translated into better closed-loop performance.",
        "",
        "## Mismatch-Feature Diagnostics",
        "",
    ]
    lines += image_lines("polymer_change_feature_diagnostics.png", "Polymer mismatch feature diagnostics across legacy and refreshed runs")
    lines += [
        "The feature-diagnostic figure shows why the transform change matters:",
        "",
        f"- legacy residual tracking piled up at exact `|3|` on `{fmt(residual_legacy['tracking_error_exact3_mean'])}` of samples on average across outputs",
        f"- refreshed residual tracking exact-`|3|` mass is effectively zero, even though raw tracking p99 stays huge at `{fmt(residual_refresh['tracking_error_raw_p99_mean'])}`",
        f"- the same pattern appears in the matrix, structured-matrix, and reidentification refreshed runs: the raw mismatch magnitude is still large, but the transform is no longer flattening it into a single clipped bucket",
        "",
        "So the transform change clearly improved what the policy can distinguish. The remaining question is whether the downstream family can exploit that richer mismatch information. Residual does. Reidentification currently does not.",
        "",
        "## Reidentification: Why It Is Not Working",
        "",
    ]

    reid_table_rows = []
    for row in reid_rows:
        reid_table_rows.append(
            [
                spec_map[row["run_id"]].variant_label,
                fmt(row["candidate_valid_frac"]),
                fmt(row["update_event_frac"]),
                fmt(row["update_success_frac"]),
                fmt(row["fallback_frac"]),
                fmt(row["condition_median"], digits=1),
                fmt(row["condition_p95"], digits=1),
                fmt(row["residual_ratio_median"]),
                fmt(row["residual_ratio_p95"]),
                fmt(row["eta_A_requested_p95"]),
                fmt(row["eta_A_applied_p95"]),
            ]
        )
    lines += markdown_table(
        [
            "Variant",
            "Candidate valid frac",
            "Update event frac",
            "Update success frac",
            "Fallback frac",
            "Cond median",
            "Cond p95",
            "Residual ratio median",
            "Residual ratio p95",
            "eta_A req p95",
            "eta_A app p95",
        ],
        reid_table_rows,
    )
    lines += [
        "",
    ]
    lines += image_lines("polymer_reid_identification_health.png", "Polymer reidentification health diagnostics")
    lines += [
        "The reidentification failure diagnosis is strong:",
        "",
        "The update math makes the failure mode explicit. The shared reidentification path is solving a regularized batch problem of the form",
        "",
        "`theta_hat = argmin_theta ||Y - Phi theta||_2^2 + lambda_prev ||theta - theta_prev||_2^2 + lambda_0 ||theta||_2^2`",
        "",
        "on each accepted identification window. The information content of that window is carried by the Gram matrix `G = Phi^T Phi`. If `G` is poorly conditioned, then small data noise produces large parameter movement; this is why the report tracks the condition number `kappa(G) = sigma_max(G) / sigma_min(G)` and the candidate residual ratios. In practice, a useful candidate should satisfy `r_val = ||e_val,cand|| / ||e_val,prev|| < 1` and `r_full = ||e_full,cand|| / ||e_full,prev|| <= 1` while staying inside the parameter and conditioning guards.",
        "",
        f"- Updates are attempted often enough. The update-event fraction is `{fmt(reid_refresh_health['update_event_frac'])}` in the refreshed run, essentially the same as legacy.",
        f"- But almost none of those attempts survive the guard. Candidate-valid fraction is only `{fmt(reid_refresh_health['candidate_valid_frac'])}`, and update-success fraction is only `{fmt(reid_refresh_health['update_success_frac'])}`.",
        f"- The regression windows are numerically bad. The refreshed run has a median condition number of `{fmt(reid_refresh_health['condition_median'], digits=1)}` and p95 of `{fmt(reid_refresh_health['condition_p95'], digits=1)}`.",
        f"- Even when a candidate is formed, it usually does not improve the fit enough. The refreshed residual-ratio median is `{fmt(reid_refresh_health['residual_ratio_median'])}`, which is effectively one, and p95 is `{fmt(reid_refresh_health['residual_ratio_p95'])}`, which means many candidate windows are much worse than the incumbent model.",
        f"- The RL agent is still requesting aggressive identification authority. In the refreshed run, `eta_A` requested p95 is `{fmt(reid_refresh_health['eta_A_requested_p95'])}` and applied p95 is `{fmt(reid_refresh_health['eta_A_applied_p95'])}`; for `eta_B`, those are `{fmt(reid_refresh_health['eta_B_requested_p95'])}` and `{fmt(reid_refresh_health['eta_B_applied_p95'])}`. Because `blend_validity_mode` is off, there is almost no moderation from the validity layer.",
        "",
        "So the main problem is not that the RL state is blind anymore. The main problem is that the online identification layer almost never produces a trustworthy candidate model. The policy is asking for identification action, but the identification engine mostly rejects or falls back, and when it does evaluate a candidate the window is poorly conditioned and often not actually better.",
        "",
        "That interpretation is consistent with the literature:",
        "",
        "- persistency of excitation is the condition that makes the regression informative enough to uniquely determine parameters [Mu2022]",
        "- in process-control settings, online re-identification may be impossible without enough excitation [Binette2016]",
        "- adaptive MPC papers therefore use information-matrix tests to decide whether a data window is informative enough to trigger a model update [Wang2022]",
        "- when the least-squares problem is ill-conditioned, regularization is a standard fix because it reduces sensitivity to noise and numerical error [Hochstenbach2011] [Lim2016]",
        "",
        "The result in this repository matches that story closely. The refreshed reidentification run still sees the mismatch better than before, but the identification subproblem is not healthy enough to convert that information into good model updates.",
        "",
        "## How To Extend Polymer Reidentification",
        "",
        "The next polymer reidentification work should not start from scratch. The shared runner already contains several hardening levers that were added for the distillation branch and can be reused here.",
        "",
    ]

    extension_table_rows = []
    for row in extension_rows:
        extension_table_rows.append(
            [
                row["label"],
                str(row["id_window"]),
                str(row["id_update_period"]),
                str(row["rank_B"]),
                fmt(row["theta_B_halfwidth"], digits=3),
                f"{row['lambda_prev_B']:.1e}",
                f"{row['lambda_0_B']:.1e}",
                "yes" if row["freeze_identification_during_warm_start"] else "no",
                "yes" if row["guard_enabled"] else "no",
                str(row["candidate_guard_mode"]),
                str(row["blend_validity_mode"]),
                str(row["force_eta_constant"]),
            ]
        )
    lines += markdown_table(
        [
            "Surface",
            "id_window",
            "id_update_period",
            "rank_B",
            "theta_B halfwidth",
            "lambda_prev_B",
            "lambda_0_B",
            "freeze warm-start",
            "validation guard",
            "guard mode",
            "blend mode",
            "force_eta_constant",
        ],
        extension_table_rows,
    )
    lines += [
        "",
    ]
    lines += image_lines("polymer_reid_extension_comparison.png", "Polymer versus distillation reidentification hardening levers and polymer next-step recommendation")
    lines += [
        "This comparison suggests a practical three-layer extension strategy for polymer reidentification:",
        "",
        f"1. Borrow the hardening that already exists in the distillation path. The biggest immediate differences are cadence and warm-start protection: polymer still runs at `{polymer_current_ext['id_window']}/{polymer_current_ext['id_update_period']}`, while distillation moved to `{distillation_current_ext['id_window']}/{distillation_current_ext['id_update_period']}` and enables `freeze_identification_during_warm_start=True`. The distillation diagnostics in this repo showed that the slower `160/20` cadence did not cause collapse by itself; warm-start hidden drift and excessive blend authority were the real problems.",
        f"2. Make the polymer B-side more conservative. Distillation already tightened the B side to `rank_B={distillation_current_ext['rank_B']}`, `lambda_prev_B={distillation_current_ext['lambda_prev_B']:.1e}`, `lambda_0_B={distillation_current_ext['lambda_0_B']:.1e}`, and `theta_B` halfwidth `{distillation_current_ext['theta_B_halfwidth']:.3f}`. Polymer is still at `rank_B={polymer_current_ext['rank_B']}`, `lambda_prev_B={polymer_current_ext['lambda_prev_B']:.1e}`, `lambda_0_B={polymer_current_ext['lambda_0_B']:.1e}`, and halfwidth `{polymer_current_ext['theta_B_halfwidth']:.3f}`. Given how rarely polymer accepts candidate updates, reducing B flexibility is a reasonable next ablation.",
        f"3. Turn on the validation-aware guard layer before trusting the RL blend again. The shared runner already supports `guard_validation_fraction`, minimum train/validation sample thresholds, clipped-theta checks, and residual-ratio / condition-number limits. Polymer is not using any of them now, but the recommended next-step surface in the table enables them with the same thresholds already tested on distillation.",
        "",
        "The one distillation idea that should be borrowed only as a diagnostic, not as the final method, is `force_eta_constant`. In the distillation branch, a small fixed `eta` was useful to separate \"identification quality\" from \"RL blend authority.\" That same isolation test is worthwhile in polymer: if polymer still fails with a tiny fixed blend such as `[0.05, 0.05]`, the bottleneck is the identification engine itself, not the policy.",
        "",
        "The one distillation idea that should not be copied blindly is `blend_validity_mode`. Distillation explored validity-aware fading, but the current active defaults are still `off`. For polymer, the correct order is:",
        "",
        "- first make candidates valid more often with cadence, warm-start freeze, B-side restriction, and validation guards",
        "- then use `diagnostic_fade` only as a second-stage ablation if candidate quality improves but aggressive blend authority still hurts control",
        "",
        "## Literature-Backed Reidentification Roadmap",
        "",
        "The literature suggests three extensions beyond the current repo surface:",
        "",
        "1. Informative-window gating and explicit excitation.",
        "   Persistence of excitation is the condition that makes identification informative. The adaptive MPC literature now includes PE conditions directly in the controller or in the update trigger, rather than passively hoping the closed-loop trajectory contains enough information [Berberich2022] [Lu2023]. For polymer, that means the update trigger should depend not only on buffer length but also on an information or singular-value test. When the test fails, the safest next step is not to update the model.",
        "",
        "2. Active exploration when identification is starved.",
        "   Dual/adaptive MPC literature treats control as both a performance input and an identification input [Parsi2020]. In this project that does not mean abandoning RL-assisted MPC. It means adding a small, bounded probing policy or excitation schedule that is activated only when the identification window is uninformative and the plant is in a safe region. This is more targeted than letting the blend policy indirectly create excitation.",
        "",
        "3. More robust online estimators.",
        "   The polymer logs look like a noisy, ill-conditioned regression problem. The recursive identification literature suggests regularized recursive least-squares or total least-squares variants when both regressors and outputs are noisy [Lim2016]. The current batch ridge solve is a reasonable baseline, but if polymer remains condition-limited after the guard/cadence changes, a recursive regularized estimator is the next algorithmic step worth testing.",
        "",
        "A concrete borrowing path from distillation is therefore:",
        "",
        "- Step 1: move polymer to the distillation-style hardening surface without changing the RL policy structure: slower `id_window/id_update_period`, `freeze_identification_during_warm_start=True`, smaller `rank_B`, tighter `theta_B` bounds, and the existing validation/conditioning guard thresholds.",
        "- Step 2: run a fixed-small-eta study exactly as an isolation tool. If a tiny constant `eta` still fails, the identification engine is the blocker. If a tiny constant `eta` works but learned `eta` does not, then the next work is blend supervision rather than estimator redesign.",
        "- Step 3: add informative-window gating before candidate construction. Concretely, require the buffer Gram matrix or its smallest singular value to clear a threshold before solving the batch update, so the policy cannot keep pushing on a statistically empty window.",
        "- Step 4: only after steps 1-3, consider validity-aware blend fading or bounded excitation triggers. Those are second-stage tools, not substitutes for a non-informative window.",
        "",
        "A practical order for polymer is therefore:",
        "",
        "- Stage A: enable distillation-style hardening in the current shared runner",
        "- Stage B: run a fixed-small-eta isolation study to separate blend problems from identification problems",
        "- Stage C: add informative-window gating and, if needed, bounded excitation triggers",
        "- Stage D: only after A-C, test stronger blend policies or new recursive estimators",
        "",
        "## Conclusions",
        "",
        "From the saved polymer runs, the recent changes did matter, but not in one uniform way:",
        "",
        "- yes, the observation-conditioning and mismatch-transform changes are clearly changing the policy-visible state in polymer",
        "- yes, those changes helped the residual family materially",
        "- yes, they affected matrix and structured matrix, but matrix is sensitive to the observer choice and structured matrix is mostly reward-level improvement rather than a clear tail-MAE win",
        "- no, the same changes did not fix polymer reidentification, because that family is currently bottlenecked by candidate-model quality, conditioning, and validity, not only by RL-state scaling",
        "",
        "The immediate project implication is that polymer residual remains the strongest beneficiary of the new conditioning path, matrix and structured matrix are secondary candidates for further tuning, and polymer reidentification needs identification-layer changes next: distillation-style hardening first, then informative-window gating, and then stronger online estimation methods if the candidate-valid fraction remains near zero.",
        "",
        "## Sources",
        "",
        "- [SB3] Stable Baselines3 `VecNormalize` implementation and running-mean/running-variance observation normalization.",
        "- [Mu2022] Mu et al., *Persistence of excitation for identifying switched linear systems*, Automatica 2022.",
        "- [Binette2016] Binette and Srinivasan, *On the Use of Nonlinear Model Predictive Control without Parameter Adaptation for Batch Processes*, Processes 2016.",
        "- [Wang2022] *Offset-free ARX-based adaptive model predictive control applied to a nonlinear process*, ISA Transactions 2022.",
        "- [Berberich2022] *Forward-looking persistent excitation in model predictive control*, Automatica 2022.",
        "- [Lu2023] *Robust adaptive model predictive control with persistent excitation conditions*, Automatica 2023.",
        "- [Parsi2020] *Active exploration in adaptive model predictive control*, ETH Zurich research collection 2020.",
        "- [Hochstenbach2011] Hochstenbach and Reichel, *Fractional Tikhonov regularization for linear discrete ill-posed problems*, BIT Numerical Mathematics 2011.",
        "- [Lim2016] Lim and Pang, *l1-regularized recursive total least squares based sparse system identification for the error-in-variables*, SpringerPlus 2016.",
        "",
        "[SB3]: https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/common/vec_env/vec_normalize.html",
        "[Mu2022]: https://doi.org/10.1016/j.automatica.2021.110142",
        "[Binette2016]: https://www.mdpi.com/2227-9717/4/3/27",
        "[Wang2022]: https://www.sciencedirect.com/science/article/abs/pii/S0019057821002937",
        "[Berberich2022]: https://doi.org/10.1016/j.automatica.2021.110033",
        "[Lu2023]: https://doi.org/10.1016/j.automatica.2023.110959",
        "[Parsi2020]: https://www.research-collection.ethz.ch/handle/20.500.11850/461407",
        "[Hochstenbach2011]: https://doi.org/10.1007/s10543-011-0313-9",
        "[Lim2016]: https://doi.org/10.1186/s40064-016-3120-6",
    ]

    (REPO_ROOT / "report" / "polymer_change_impact_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    baseline_bundle = load_pickle(BASELINE_PATH)
    minmax = load_pickle(POLYMER_MINMAX_PATH)
    min_s = np.asarray(minmax["min_s"], float)
    max_s = np.asarray(minmax["max_s"], float)

    spec_map = {spec.run_id: spec for spec in RUN_SPECS}
    extension_rows = build_reid_extension_rows()
    runs: dict[str, dict] = {}
    config_rows: list[dict] = [
        {
            "run_id": "baseline",
            "family": "baseline",
            "variant_key": "baseline",
            "variant_label": "baseline",
            "run_path": repo_rel(BASELINE_PATH),
            "base_state_norm_mode": "n/a",
            "mismatch_feature_transform_mode": "n/a",
            "observer_update_alignment": "n/a",
            "rho_mapping_mode": "n/a",
            "candidate_guard_mode": "n/a",
            "blend_validity_mode": "n/a",
        }
    ]
    summary_rows: list[dict] = [
        {
            "run_id": "baseline",
            "family": "baseline",
            "variant_key": "baseline",
            "variant_label": "baseline",
            "run_path": repo_rel(BASELINE_PATH),
            **compute_metrics(baseline_bundle),
        }
    ]
    state_rows: list[dict] = []
    reid_rows: list[dict] = []

    for spec in RUN_SPECS:
        bundle = load_pickle(spec.path)
        runs[spec.run_id] = {"spec": spec, "bundle": bundle}

        metrics = compute_metrics(bundle, baseline_bundle=baseline_bundle)
        summary_rows.append(
            {
                "run_id": spec.run_id,
                "family": spec.family,
                "variant_key": spec.variant_key,
                "variant_label": spec.variant_label,
                "run_path": repo_rel(spec.path),
                **metrics,
            }
        )

        config_rows.append(
            {
                "run_id": spec.run_id,
                "family": spec.family,
                "variant_key": spec.variant_key,
                "variant_label": spec.variant_label,
                "run_path": repo_rel(spec.path),
                "base_state_norm_mode": str(bundle.get("base_state_norm_mode", "legacy_fixed_minmax")),
                "mismatch_feature_transform_mode": str(bundle.get("mismatch_feature_transform_mode", "legacy_hard_clip")),
                "observer_update_alignment": str(bundle.get("observer_update_alignment", "legacy_previous_measurement")),
                "rho_mapping_mode": str(bundle.get("rho_mapping_mode", "legacy_clipped_linear")),
                "candidate_guard_mode": str(bundle.get("candidate_guard_mode", "n/a")),
                "blend_validity_mode": str(bundle.get("blend_validity_mode", "n/a")),
            }
        )

        actual_state = state_span_summary(
            bundle,
            min_s=min_s,
            max_s=max_s,
            mode_override=bundle.get("base_state_norm_mode", "fixed_minmax"),
        )
        state_rows.append(
            {
                "run_id": spec.run_id,
                "family": spec.family,
                "variant_key": spec.variant_key,
                "view": "actual",
                **actual_state,
            }
        )
        if str(bundle.get("base_state_norm_mode", "fixed_minmax")).strip().lower() != "fixed_minmax":
            fixed_state = state_span_summary(bundle, min_s=min_s, max_s=max_s, mode_override="fixed_minmax")
            state_rows.append(
                {
                    "run_id": spec.run_id,
                    "family": spec.family,
                    "variant_key": spec.variant_key,
                    "view": "fixed counterfactual",
                    **fixed_state,
                }
            )

        if spec.family == "reidentification":
            reid_rows.append(
                {
                    "run_id": spec.run_id,
                    "family": spec.family,
                    "variant_key": spec.variant_key,
                    "variant_label": spec.variant_label,
                    **compute_reid_metrics(bundle),
                }
            )

    save_csv(
        DATA_ROOT / "config_summary.csv",
        config_rows,
        [
            "run_id",
            "family",
            "variant_key",
            "variant_label",
            "run_path",
            "base_state_norm_mode",
            "mismatch_feature_transform_mode",
            "observer_update_alignment",
            "rho_mapping_mode",
            "candidate_guard_mode",
            "blend_validity_mode",
        ],
    )
    save_csv(
        DATA_ROOT / "performance_summary.csv",
        summary_rows,
        [
            "run_id",
            "family",
            "variant_key",
            "variant_label",
            "run_path",
            "scaled_mae_mean",
            "scaled_mae_tail_mean",
            "phys_mae_mean",
            "phys_mae_tail_mean",
            "reward_final20_mean",
            "tail_phys_mae_delta_vs_baseline",
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
            "multiplier_abs_mean",
            "multiplier_abs_p95",
            "multiplier_step_norm_p95",
        ],
    )
    save_csv(
        DATA_ROOT / "state_conditioning_summary.csv",
        state_rows,
        ["run_id", "family", "variant_key", "view", "full_span_med", "late_span_med", "late_full_ratio_med"],
    )
    save_csv(
        DATA_ROOT / "reidentification_health_summary.csv",
        reid_rows,
        [
            "run_id",
            "family",
            "variant_key",
            "variant_label",
            "candidate_valid_frac",
            "fallback_frac",
            "update_event_frac",
            "update_success_frac",
            "condition_median",
            "condition_p95",
            "condition_p99",
            "residual_norm_median",
            "residual_norm_p95",
            "residual_norm_p99",
            "residual_ratio_median",
            "residual_ratio_p95",
            "residual_ratio_p99",
            "eta_A_requested_p95",
            "eta_A_applied_p95",
            "eta_B_requested_p95",
            "eta_B_applied_p95",
            "blend_A_p50",
            "blend_B_p50",
            "source0_frac",
            "source1_frac",
            "source2_frac",
        ],
    )
    save_csv(
        DATA_ROOT / "reidentification_extension_plan_summary.csv",
        extension_rows,
        [
            "label",
            "system",
            "id_window",
            "id_update_period",
            "rank_B",
            "lambda_prev_B",
            "lambda_0_B",
            "theta_B_halfwidth",
            "freeze_identification_during_warm_start",
            "guard_enabled",
            "candidate_guard_mode",
            "blend_validity_mode",
            "force_eta_constant",
            "has_small_fixed_eta_isolation",
            "normalize_blend_extras",
            "blend_extra_clip",
            "blend_residual_scale",
            "eta_tau_B",
            "guard_validation_fraction",
            "guard_min_validation_samples",
            "guard_min_train_samples",
            "max_theta_clipped_fraction",
            "max_condition_number",
            "max_validation_residual_ratio",
            "max_full_residual_ratio",
        ],
    )

    plot_performance_overview(summary_rows=summary_rows, state_rows=state_rows, spec_map=spec_map)
    plot_family_tail_traces(baseline_bundle=baseline_bundle, runs=runs, spec_map=spec_map)
    plot_state_conditioning(state_rows=state_rows, spec_map=spec_map)
    plot_feature_diagnostics(summary_rows=summary_rows, spec_map=spec_map)
    plot_reid_health(reid_rows=reid_rows)
    plot_reid_extension_comparison(extension_rows=extension_rows)
    write_markdown(
        config_rows=config_rows,
        summary_rows=summary_rows,
        state_rows=state_rows,
        reid_rows=reid_rows,
        extension_rows=extension_rows,
        spec_map=spec_map,
    )


if __name__ == "__main__":
    main()
