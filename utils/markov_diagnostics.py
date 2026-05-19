from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


MARKOV_STAGE_BUNDLE_KEYS = (
    "A",
    "B",
    "C",
    "A_aug",
    "B_aug",
    "C_aug",
    "u_sequence_nominal_log",
    "u_sequence_requested_log",
    "u_sequence_ls_log",
    "u_sequence_executed_log",
    "u0_nominal_log",
    "u0_requested_log",
    "u0_ls_log",
    "u0_executed_log",
    "u0_requested_minus_nominal_log",
    "u0_ls_minus_nominal_log",
    "u0_executed_minus_nominal_log",
    "u0_requested_minus_nominal_norm_log",
    "u0_ls_minus_nominal_norm_log",
    "u0_executed_minus_nominal_norm_log",
    "u_sequence_requested_minus_nominal_norm_log",
    "u_sequence_ls_minus_nominal_norm_log",
    "u_sequence_executed_minus_nominal_norm_log",
    "nominal_cost_log",
    "requested_candidate_native_cost_log",
    "requested_candidate_nominal_cost_log",
    "requested_cost_margin_log",
    "requested_cost_guard_pass_log",
    "requested_gain_drift_log",
    "requested_prediction_score_log",
    "ls_candidate_native_cost_log",
    "ls_candidate_nominal_cost_log",
    "ls_cost_margin_log",
    "ls_cost_guard_pass_log",
    "ls_gain_drift_log",
    "ls_prediction_score_log",
    "executed_candidate_native_cost_log",
    "executed_candidate_nominal_cost_log",
    "executed_cost_margin_log",
    "executed_cost_guard_pass_log",
    "executed_gain_drift_log",
    "executed_prediction_score_log",
    "rl_actor_raw_action_log",
    "rl_requested_raw_action_log",
    "rl_executed_raw_action_log",
    "td3_priority_phase_log",
    "td3_authority_scale_log",
    "td3_probation_active_log",
    "td3_probation_trigger_log",
)


def _as_array(bundle: dict, key: str, dtype=float):
    value = bundle.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _safe_scalar(arr: np.ndarray | None, idx: int):
    if arr is None:
        return np.nan
    value = arr[idx]
    if np.ndim(value) != 0:
        return np.nan
    return float(value)


def _safe_int(arr: np.ndarray | None, idx: int):
    if arr is None:
        return 0
    return int(arr[idx])


def _safe_row_norm(arr: np.ndarray | None, idx: int):
    if arr is None:
        return np.nan
    row = np.asarray(arr[idx], float).reshape(-1)
    if row.size == 0 or np.all(~np.isfinite(row)):
        return np.nan
    return float(np.linalg.norm(row[np.isfinite(row)]))


def _safe_row_component(arr: np.ndarray | None, idx: int, comp: int):
    if arr is None:
        return np.nan
    row = np.asarray(arr[idx], float).reshape(-1)
    if comp >= row.size:
        return np.nan
    return float(row[comp])


def build_markov_stage_diagnostics_rows(bundle: dict) -> list[dict]:
    nFE = int(bundle.get("nFE", 0))
    action_source_names = dict(bundle.get("rl_action_source_names") or {})
    u0_nominal = _as_array(bundle, "u0_nominal_log", dtype=float)
    n_inputs = int(u0_nominal.shape[1]) if u0_nominal is not None and u0_nominal.ndim == 2 else int(
        np.asarray(bundle.get("u"), float).shape[1]
    )

    action_source = _as_array(bundle, "rl_action_source_log", dtype=int)
    accepted = _as_array(bundle, "accepted_log", dtype=int)
    fallback = _as_array(bundle, "fallback_log", dtype=int)
    rl_actor_raw = _as_array(bundle, "rl_actor_raw_action_log", dtype=float)
    rl_requested_raw = _as_array(bundle, "rl_requested_raw_action_log", dtype=float)
    rl_executed_raw = _as_array(bundle, "rl_executed_raw_action_log", dtype=float)
    rl_requested_z = _as_array(bundle, "rl_requested_z_log", dtype=float)
    rl_ls_z = _as_array(bundle, "rl_ls_z_log", dtype=float)
    z_executed = _as_array(bundle, "z_executed_log", dtype=float)
    requested_gain_drift = _as_array(bundle, "requested_gain_drift_log", dtype=float)
    ls_gain_drift = _as_array(bundle, "ls_gain_drift_log", dtype=float)
    executed_gain_drift = _as_array(bundle, "executed_gain_drift_log", dtype=float)
    requested_prediction_score = _as_array(bundle, "requested_prediction_score_log", dtype=float)
    ls_prediction_score = _as_array(bundle, "ls_prediction_score_log", dtype=float)
    executed_prediction_score = _as_array(bundle, "executed_prediction_score_log", dtype=float)
    nominal_cost = _as_array(bundle, "nominal_cost_log", dtype=float)
    requested_candidate_native_cost = _as_array(bundle, "requested_candidate_native_cost_log", dtype=float)
    requested_candidate_nominal_cost = _as_array(bundle, "requested_candidate_nominal_cost_log", dtype=float)
    requested_cost_margin = _as_array(bundle, "requested_cost_margin_log", dtype=float)
    requested_cost_guard_pass = _as_array(bundle, "requested_cost_guard_pass_log", dtype=int)
    ls_candidate_native_cost = _as_array(bundle, "ls_candidate_native_cost_log", dtype=float)
    ls_candidate_nominal_cost = _as_array(bundle, "ls_candidate_nominal_cost_log", dtype=float)
    ls_cost_margin = _as_array(bundle, "ls_cost_margin_log", dtype=float)
    ls_cost_guard_pass = _as_array(bundle, "ls_cost_guard_pass_log", dtype=int)
    executed_candidate_native_cost = _as_array(bundle, "executed_candidate_native_cost_log", dtype=float)
    executed_candidate_nominal_cost = _as_array(bundle, "executed_candidate_nominal_cost_log", dtype=float)
    executed_cost_margin = _as_array(bundle, "executed_cost_margin_log", dtype=float)
    executed_cost_guard_pass = _as_array(bundle, "executed_cost_guard_pass_log", dtype=int)
    u0_requested_minus_nominal = _as_array(bundle, "u0_requested_minus_nominal_log", dtype=float)
    u0_ls_minus_nominal = _as_array(bundle, "u0_ls_minus_nominal_log", dtype=float)
    u0_executed_minus_nominal = _as_array(bundle, "u0_executed_minus_nominal_log", dtype=float)
    u0_requested_minus_nominal_norm = _as_array(bundle, "u0_requested_minus_nominal_norm_log", dtype=float)
    u0_ls_minus_nominal_norm = _as_array(bundle, "u0_ls_minus_nominal_norm_log", dtype=float)
    u0_executed_minus_nominal_norm = _as_array(bundle, "u0_executed_minus_nominal_norm_log", dtype=float)
    u_sequence_requested_minus_nominal_norm = _as_array(
        bundle, "u_sequence_requested_minus_nominal_norm_log", dtype=float
    )
    u_sequence_ls_minus_nominal_norm = _as_array(bundle, "u_sequence_ls_minus_nominal_norm_log", dtype=float)
    u_sequence_executed_minus_nominal_norm = _as_array(
        bundle, "u_sequence_executed_minus_nominal_norm_log", dtype=float
    )
    td3_priority_phase = _as_array(bundle, "td3_priority_phase_log", dtype=int)
    td3_authority_scale = _as_array(bundle, "td3_authority_scale_log", dtype=float)
    td3_probation_active = _as_array(bundle, "td3_probation_active_log", dtype=int)
    td3_probation_trigger = _as_array(bundle, "td3_probation_trigger_log", dtype=int)

    rows = []
    for step in range(nFE):
        row = {
            "step": step,
            "accepted": _safe_int(accepted, step),
            "fallback_used": _safe_int(fallback, step),
            "action_source": _safe_int(action_source, step),
            "action_source_name": action_source_names.get(_safe_int(action_source, step), "unknown"),
            "td3_priority_phase": _safe_int(td3_priority_phase, step),
            "td3_authority_scale": _safe_scalar(td3_authority_scale, step),
            "td3_probation_active": _safe_int(td3_probation_active, step),
            "td3_probation_trigger": _safe_int(td3_probation_trigger, step),
            "actor_raw_action_norm": _safe_row_norm(rl_actor_raw, step),
            "requested_raw_action_norm": _safe_row_norm(rl_requested_raw, step),
            "executed_raw_action_norm": _safe_row_norm(rl_executed_raw, step),
            "requested_z_norm": _safe_row_norm(rl_requested_z, step),
            "ls_z_norm": _safe_row_norm(rl_ls_z, step),
            "executed_z_norm": _safe_row_norm(z_executed, step),
            "requested_gain_drift": _safe_scalar(requested_gain_drift, step),
            "ls_gain_drift": _safe_scalar(ls_gain_drift, step),
            "executed_gain_drift": _safe_scalar(executed_gain_drift, step),
            "requested_prediction_score": _safe_scalar(requested_prediction_score, step),
            "ls_prediction_score": _safe_scalar(ls_prediction_score, step),
            "executed_prediction_score": _safe_scalar(executed_prediction_score, step),
            "nominal_cost": _safe_scalar(nominal_cost, step),
            "requested_candidate_native_cost": _safe_scalar(requested_candidate_native_cost, step),
            "requested_candidate_nominal_cost": _safe_scalar(requested_candidate_nominal_cost, step),
            "requested_cost_margin": _safe_scalar(requested_cost_margin, step),
            "requested_cost_guard_pass": _safe_int(requested_cost_guard_pass, step),
            "ls_candidate_native_cost": _safe_scalar(ls_candidate_native_cost, step),
            "ls_candidate_nominal_cost": _safe_scalar(ls_candidate_nominal_cost, step),
            "ls_cost_margin": _safe_scalar(ls_cost_margin, step),
            "ls_cost_guard_pass": _safe_int(ls_cost_guard_pass, step),
            "executed_candidate_native_cost": _safe_scalar(executed_candidate_native_cost, step),
            "executed_candidate_nominal_cost": _safe_scalar(executed_candidate_nominal_cost, step),
            "executed_cost_margin": _safe_scalar(executed_cost_margin, step),
            "executed_cost_guard_pass": _safe_int(executed_cost_guard_pass, step),
            "requested_first_move_diff_norm": _safe_scalar(u0_requested_minus_nominal_norm, step),
            "ls_first_move_diff_norm": _safe_scalar(u0_ls_minus_nominal_norm, step),
            "executed_first_move_diff_norm": _safe_scalar(u0_executed_minus_nominal_norm, step),
            "requested_full_sequence_diff_norm": _safe_scalar(u_sequence_requested_minus_nominal_norm, step),
            "ls_full_sequence_diff_norm": _safe_scalar(u_sequence_ls_minus_nominal_norm, step),
            "executed_full_sequence_diff_norm": _safe_scalar(u_sequence_executed_minus_nominal_norm, step),
        }
        for comp in range(n_inputs):
            row[f"requested_first_move_diff_u{comp + 1}"] = _safe_row_component(u0_requested_minus_nominal, step, comp)
            row[f"ls_first_move_diff_u{comp + 1}"] = _safe_row_component(u0_ls_minus_nominal, step, comp)
            row[f"executed_first_move_diff_u{comp + 1}"] = _safe_row_component(u0_executed_minus_nominal, step, comp)
        rows.append(row)
    return rows


def write_markov_stage_diagnostics_csv(bundle: dict, out_path: str | Path):
    rows = build_markov_stage_diagnostics_rows(bundle)
    if not rows:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
