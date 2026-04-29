from __future__ import annotations

import numpy as np

from utils.multiplier_release_schedule import PHASE_FULL, PHASE_NOMINAL, PHASE_PROTECTED, PHASE_RAMP
from utils.multiplier_sensitivity import build_markov_matrix


ACCEPTANCE_REASON_DISABLED = 0
ACCEPTANCE_REASON_ACCEPTED = 1
ACCEPTANCE_REASON_REJECTED_COST = 2
ACCEPTANCE_REASON_CANDIDATE_SOLVE_FAILED = 3

DUAL_COST_SHADOW_REASON_DISABLED = 0
DUAL_COST_SHADOW_REASON_EVALUATED = 1
DUAL_COST_SHADOW_REASON_CANDIDATE_SOLVE_FAILED = 2

USEFULNESS_GATE_REASON_DISABLED = 0
USEFULNESS_GATE_REASON_ACCEPTED = 1
USEFULNESS_GATE_REASON_CANDIDATE_SOLVE_FAILED = 2
USEFULNESS_GATE_REASON_REJECTED_NOMINAL_SAFETY = 3
USEFULNESS_GATE_REASON_REJECTED_CANDIDATE_USEFULNESS = 4
USEFULNESS_GATE_REASON_REJECTED_GAIN_DRIFT = 5


def run_mpc_acceptance_gate(
    *,
    mpc_obj,
    solve_fn,
    acceptance_cfg,
    A_candidate,
    B_candidate,
    A_nominal,
    B_nominal,
    y_sp,
    u_prev_dev,
    x0_model,
    initial_guess,
    bounds,
    step_idx,
):
    """Select candidate or nominal MPC with a strict nominal-model cost gate."""
    cfg = dict(acceptance_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    fallback_on_candidate_solve_failure = bool(cfg.get("fallback_on_candidate_solve_failure", True))
    relative_tolerance = float(cfg.get("relative_tolerance", 0.0))
    absolute_tolerance = float(cfg.get("absolute_tolerance", 1e-8))

    core = _evaluate_dual_cost_core(
        mpc_obj=mpc_obj,
        solve_fn=solve_fn,
        A_candidate=A_candidate,
        B_candidate=B_candidate,
        A_nominal=A_nominal,
        B_nominal=B_nominal,
        y_sp=y_sp,
        u_prev_dev=u_prev_dev,
        x0_model=x0_model,
        initial_guess=initial_guess,
        bounds=bounds,
        step_idx=step_idx,
        fallback_on_candidate_solve_failure=fallback_on_candidate_solve_failure,
    )
    candidate_sol = core["candidate_sol"]
    nominal_sol = core["nominal_sol"]
    nominal_cost = core["nominal_cost"]
    candidate_solve_failed = bool(core["candidate_solve_failed"])

    if not enabled:
        mpc_obj.A = np.asarray(A_candidate, float)
        mpc_obj.B = np.asarray(B_candidate, float)
        return {
            "sol": candidate_sol,
            "accepted": True,
            "fallback_active": False,
            "reason_code": ACCEPTANCE_REASON_DISABLED,
            "candidate_cost_on_nominal": np.nan,
            "candidate_cost_native": float(core["candidate_cost_native"]),
            "nominal_cost": np.nan,
            "cost_margin": np.nan,
            "threshold": np.nan,
        }

    if candidate_solve_failed:
        mpc_obj.A = np.asarray(A_nominal, float)
        mpc_obj.B = np.asarray(B_nominal, float)
        return {
            "sol": nominal_sol,
            "accepted": False,
            "fallback_active": True,
            "reason_code": ACCEPTANCE_REASON_CANDIDATE_SOLVE_FAILED,
            "candidate_cost_on_nominal": np.nan,
            "candidate_cost_native": np.nan,
            "nominal_cost": nominal_cost,
            "cost_margin": np.nan,
            "threshold": np.nan,
        }

    candidate_cost_on_nominal = float(core["candidate_cost_on_nominal"])
    threshold = (1.0 + relative_tolerance) * nominal_cost + absolute_tolerance
    accepted = bool(candidate_cost_on_nominal <= threshold)
    if accepted:
        mpc_obj.A = np.asarray(A_candidate, float)
        mpc_obj.B = np.asarray(B_candidate, float)
        selected_sol = candidate_sol
        reason_code = ACCEPTANCE_REASON_ACCEPTED
    else:
        mpc_obj.A = np.asarray(A_nominal, float)
        mpc_obj.B = np.asarray(B_nominal, float)
        selected_sol = nominal_sol
        reason_code = ACCEPTANCE_REASON_REJECTED_COST

    return {
        "sol": selected_sol,
        "accepted": accepted,
        "fallback_active": not accepted,
        "reason_code": int(reason_code),
        "candidate_cost_on_nominal": candidate_cost_on_nominal,
        "candidate_cost_native": float(core["candidate_cost_native"]),
        "nominal_cost": nominal_cost,
        "cost_margin": candidate_cost_on_nominal - nominal_cost,
        "threshold": threshold,
    }


def run_mpc_dual_cost_shadow(
    *,
    mpc_obj,
    solve_fn,
    shadow_cfg,
    A_candidate,
    B_candidate,
    A_nominal,
    B_nominal,
    y_sp,
    u_prev_dev,
    x0_model,
    initial_guess,
    bounds,
    step_idx,
):
    """Evaluate dual-cost shadow diagnostics without cost-based fallback."""
    cfg = dict(shadow_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    fallback_on_candidate_solve_failure = bool(cfg.get("fallback_on_candidate_solve_failure", True))
    relative_tolerance = float(cfg.get("relative_tolerance", 1e-4))
    absolute_tolerance = float(cfg.get("absolute_tolerance", 1e-8))
    benefit_tolerance = float(cfg.get("benefit_tolerance", 0.0))

    if not enabled:
        mpc_obj.A = np.asarray(A_candidate, float)
        mpc_obj.B = np.asarray(B_candidate, float)
        return {
            "sol": None,
            "executed_candidate": False,
            "fallback_active": False,
            "reason_code": DUAL_COST_SHADOW_REASON_DISABLED,
            "candidate_cost_native": np.nan,
            "nominal_cost": np.nan,
            "candidate_cost_on_nominal": np.nan,
            "nominal_cost_on_candidate": np.nan,
            "nominal_penalty": np.nan,
            "safe_threshold": np.nan,
            "candidate_advantage": np.nan,
            "safe_pass": False,
            "benefit_pass": False,
            "dual_pass": False,
        }

    core = _evaluate_dual_cost_core(
        mpc_obj=mpc_obj,
        solve_fn=solve_fn,
        A_candidate=A_candidate,
        B_candidate=B_candidate,
        A_nominal=A_nominal,
        B_nominal=B_nominal,
        y_sp=y_sp,
        u_prev_dev=u_prev_dev,
        x0_model=x0_model,
        initial_guess=initial_guess,
        bounds=bounds,
        step_idx=step_idx,
        fallback_on_candidate_solve_failure=fallback_on_candidate_solve_failure,
    )
    nominal_sol = core["nominal_sol"]
    candidate_sol = core["candidate_sol"]
    nominal_cost = float(core["nominal_cost"])
    candidate_cost_native = float(core["candidate_cost_native"])
    candidate_cost_on_nominal = float(core["candidate_cost_on_nominal"])
    nominal_cost_on_candidate = float(core["nominal_cost_on_candidate"])
    nominal_penalty = float(core["nominal_penalty"])
    candidate_advantage = float(core["candidate_advantage"])
    candidate_solve_failed = bool(core["candidate_solve_failed"])
    safe_threshold = (
        relative_tolerance * nominal_cost + absolute_tolerance if np.isfinite(nominal_cost) else np.nan
    )
    safe_pass = bool(np.isfinite(nominal_penalty) and np.isfinite(safe_threshold) and nominal_penalty <= safe_threshold)
    benefit_pass = bool(np.isfinite(candidate_advantage) and candidate_advantage >= benefit_tolerance)
    dual_pass = bool(safe_pass and benefit_pass)

    if candidate_solve_failed:
        mpc_obj.A = np.asarray(A_nominal, float)
        mpc_obj.B = np.asarray(B_nominal, float)
        return {
            "sol": nominal_sol,
            "executed_candidate": False,
            "fallback_active": True,
            "reason_code": DUAL_COST_SHADOW_REASON_CANDIDATE_SOLVE_FAILED,
            "candidate_cost_native": candidate_cost_native,
            "nominal_cost": nominal_cost,
            "candidate_cost_on_nominal": candidate_cost_on_nominal,
            "nominal_cost_on_candidate": nominal_cost_on_candidate,
            "nominal_penalty": nominal_penalty,
            "safe_threshold": safe_threshold,
            "candidate_advantage": candidate_advantage,
            "safe_pass": safe_pass,
            "benefit_pass": benefit_pass,
            "dual_pass": dual_pass,
        }

    mpc_obj.A = np.asarray(A_candidate, float)
    mpc_obj.B = np.asarray(B_candidate, float)
    return {
        "sol": candidate_sol,
        "executed_candidate": True,
        "fallback_active": False,
        "reason_code": DUAL_COST_SHADOW_REASON_EVALUATED,
        "candidate_cost_native": candidate_cost_native,
        "nominal_cost": nominal_cost,
        "candidate_cost_on_nominal": candidate_cost_on_nominal,
        "nominal_cost_on_candidate": nominal_cost_on_candidate,
        "nominal_penalty": nominal_penalty,
        "safe_threshold": safe_threshold,
        "candidate_advantage": candidate_advantage,
        "safe_pass": safe_pass,
        "benefit_pass": benefit_pass,
        "dual_pass": dual_pass,
    }


def run_mpc_usefulness_gate(
    *,
    mpc_obj,
    solve_fn,
    gate_cfg,
    A_candidate,
    B_candidate,
    A_nominal,
    B_nominal,
    C_matrix,
    predict_h,
    theta_B_candidate,
    b_labels,
    phase_code,
    y_sp,
    u_prev_dev,
    x0_model,
    initial_guess,
    bounds,
    step_idx,
):
    """Execute a hard Step 3D usefulness gate on the Step 2-clipped candidate model."""
    cfg = dict(gate_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    fallback_on_candidate_solve_failure = bool(cfg.get("fallback_on_candidate_solve_failure", True))
    relative_tolerance = float(cfg.get("relative_tolerance", 1e-4))
    absolute_tolerance = float(cfg.get("absolute_tolerance", 1e-8))
    benefit_tolerance = float(cfg.get("benefit_tolerance", 0.0))
    b_penalty_lambda = float(cfg.get("b_penalty_lambda", 0.02))
    b_label_weights = dict(cfg.get("b_label_weights", {}))
    safe_scales = dict(cfg.get("safe_threshold_scales_by_phase", {}))
    benefit_offsets = dict(cfg.get("benefit_threshold_offsets_by_phase", {}))
    gain_thresholds = dict(cfg.get("gain_drift_thresholds_by_phase", {}))

    if not enabled:
        return {
            "sol": None,
            "executed_candidate": False,
            "fallback_active": False,
            "reason_code": USEFULNESS_GATE_REASON_DISABLED,
            "candidate_cost_native": np.nan,
            "nominal_cost": np.nan,
            "candidate_cost_on_nominal": np.nan,
            "nominal_cost_on_candidate": np.nan,
            "nominal_penalty": np.nan,
            "safe_threshold": np.nan,
            "candidate_advantage": np.nan,
            "b_authority_distance": np.nan,
            "b_penalty": np.nan,
            "benefit_threshold": np.nan,
            "gain_drift": np.nan,
            "gain_drift_threshold": np.nan,
            "safe_pass": False,
            "benefit_pass": False,
            "gain_pass": False,
            "gate_pass": False,
        }

    phase_key = _phase_key_from_code(phase_code)
    safe_scale = float(safe_scales.get(phase_key, 1.0))
    benefit_offset = float(benefit_offsets.get(phase_key, 0.0))
    gain_threshold = float(gain_thresholds.get(phase_key, np.inf))

    core = _evaluate_dual_cost_core(
        mpc_obj=mpc_obj,
        solve_fn=solve_fn,
        A_candidate=A_candidate,
        B_candidate=B_candidate,
        A_nominal=A_nominal,
        B_nominal=B_nominal,
        y_sp=y_sp,
        u_prev_dev=u_prev_dev,
        x0_model=x0_model,
        initial_guess=initial_guess,
        bounds=bounds,
        step_idx=step_idx,
        fallback_on_candidate_solve_failure=fallback_on_candidate_solve_failure,
    )
    nominal_sol = core["nominal_sol"]
    candidate_sol = core["candidate_sol"]
    nominal_cost = float(core["nominal_cost"])
    candidate_cost_native = float(core["candidate_cost_native"])
    candidate_cost_on_nominal = float(core["candidate_cost_on_nominal"])
    nominal_cost_on_candidate = float(core["nominal_cost_on_candidate"])
    nominal_penalty = float(core["nominal_penalty"])
    candidate_advantage = float(core["candidate_advantage"])
    candidate_solve_failed = bool(core["candidate_solve_failed"])

    safe_threshold_base = relative_tolerance * nominal_cost + absolute_tolerance if np.isfinite(nominal_cost) else np.nan
    safe_threshold = safe_threshold_base * safe_scale if np.isfinite(safe_threshold_base) else np.nan
    b_authority_distance = _b_authority_distance(theta_B_candidate, b_labels, b_label_weights)
    b_penalty = b_penalty_lambda * b_authority_distance if np.isfinite(b_authority_distance) else np.nan
    benefit_threshold = (
        benefit_tolerance + benefit_offset + b_penalty if np.isfinite(b_penalty) else np.nan
    )
    gain_drift = _gain_drift_ratio(A_candidate, B_candidate, A_nominal, B_nominal, C_matrix, predict_h)

    safe_pass = bool(np.isfinite(nominal_penalty) and np.isfinite(safe_threshold) and nominal_penalty <= safe_threshold)
    benefit_pass = bool(
        np.isfinite(candidate_advantage) and np.isfinite(benefit_threshold) and candidate_advantage >= benefit_threshold
    )
    gain_pass = bool(np.isfinite(gain_drift) and np.isfinite(gain_threshold) and gain_drift <= gain_threshold)
    gate_pass = bool(safe_pass and benefit_pass and gain_pass)

    if candidate_solve_failed:
        reason_code = USEFULNESS_GATE_REASON_CANDIDATE_SOLVE_FAILED
    elif not safe_pass:
        reason_code = USEFULNESS_GATE_REASON_REJECTED_NOMINAL_SAFETY
    elif not benefit_pass:
        reason_code = USEFULNESS_GATE_REASON_REJECTED_CANDIDATE_USEFULNESS
    elif not gain_pass:
        reason_code = USEFULNESS_GATE_REASON_REJECTED_GAIN_DRIFT
    else:
        reason_code = USEFULNESS_GATE_REASON_ACCEPTED

    if gate_pass:
        mpc_obj.A = np.asarray(A_candidate, float)
        mpc_obj.B = np.asarray(B_candidate, float)
        selected_sol = candidate_sol
    else:
        mpc_obj.A = np.asarray(A_nominal, float)
        mpc_obj.B = np.asarray(B_nominal, float)
        selected_sol = nominal_sol

    return {
        "sol": selected_sol,
        "executed_candidate": gate_pass,
        "fallback_active": not gate_pass,
        "reason_code": int(reason_code),
        "candidate_cost_native": candidate_cost_native,
        "nominal_cost": nominal_cost,
        "candidate_cost_on_nominal": candidate_cost_on_nominal,
        "nominal_cost_on_candidate": nominal_cost_on_candidate,
        "nominal_penalty": nominal_penalty,
        "safe_threshold": safe_threshold,
        "candidate_advantage": candidate_advantage,
        "b_authority_distance": b_authority_distance,
        "b_penalty": b_penalty,
        "benefit_threshold": benefit_threshold,
        "gain_drift": gain_drift,
        "gain_drift_threshold": gain_threshold,
        "safe_pass": safe_pass,
        "benefit_pass": benefit_pass,
        "gain_pass": gain_pass,
        "gate_pass": gate_pass,
    }


def _evaluate_dual_cost_core(
    *,
    mpc_obj,
    solve_fn,
    A_candidate,
    B_candidate,
    A_nominal,
    B_nominal,
    y_sp,
    u_prev_dev,
    x0_model,
    initial_guess,
    bounds,
    step_idx,
    fallback_on_candidate_solve_failure,
):
    A_candidate = np.asarray(A_candidate, float)
    B_candidate = np.asarray(B_candidate, float)
    A_nominal = np.asarray(A_nominal, float)
    B_nominal = np.asarray(B_nominal, float)

    def _solve_with_model(A_model, B_model):
        mpc_obj.A = np.asarray(A_model, float)
        mpc_obj.B = np.asarray(B_model, float)
        return solve_fn(
            mpc_obj=mpc_obj,
            y_sp=y_sp,
            u_prev_dev=u_prev_dev,
            x0_model=x0_model,
            initial_guess=initial_guess,
            bounds=bounds,
            step_idx=step_idx,
        )

    def _eval_cost(A_model, B_model, sol):
        if sol is None:
            return np.nan
        mpc_obj.A = np.asarray(A_model, float)
        mpc_obj.B = np.asarray(B_model, float)
        return float(mpc_obj.mpc_opt_fun(sol.x, y_sp, u_prev_dev, x0_model))

    nominal_sol = _solve_with_model(A_nominal, B_nominal)
    nominal_cost = _safe_fun(nominal_sol)

    candidate_model_finite = bool(np.all(np.isfinite(A_candidate)) and np.all(np.isfinite(B_candidate)))
    candidate_sol = None
    candidate_solve_failed = False
    if not candidate_model_finite:
        candidate_solve_failed = True
        if not fallback_on_candidate_solve_failure:
            raise RuntimeError(f"Candidate MPC model became non-finite at step {step_idx}.")
    else:
        try:
            candidate_sol = _solve_with_model(A_candidate, B_candidate)
        except RuntimeError:
            candidate_solve_failed = True
            if not fallback_on_candidate_solve_failure:
                raise

    candidate_cost_native = _safe_fun(candidate_sol)
    candidate_cost_on_nominal = _eval_cost(A_nominal, B_nominal, candidate_sol)
    nominal_cost_on_candidate = _eval_cost(A_candidate, B_candidate, nominal_sol) if candidate_model_finite else np.nan
    nominal_penalty = (
        candidate_cost_on_nominal - nominal_cost
        if np.isfinite(candidate_cost_on_nominal) and np.isfinite(nominal_cost)
        else np.nan
    )
    candidate_advantage = (
        nominal_cost_on_candidate - candidate_cost_native
        if np.isfinite(nominal_cost_on_candidate) and np.isfinite(candidate_cost_native)
        else np.nan
    )

    return {
        "nominal_sol": nominal_sol,
        "nominal_cost": nominal_cost,
        "candidate_sol": candidate_sol,
        "candidate_cost_native": candidate_cost_native,
        "candidate_cost_on_nominal": candidate_cost_on_nominal,
        "nominal_cost_on_candidate": nominal_cost_on_candidate,
        "nominal_penalty": nominal_penalty,
        "candidate_advantage": candidate_advantage,
        "candidate_solve_failed": candidate_solve_failed,
    }


def _phase_key_from_code(phase_code):
    phase_code = int(phase_code)
    if phase_code in {PHASE_NOMINAL, PHASE_PROTECTED}:
        return "protected"
    if phase_code == PHASE_RAMP:
        return "ramp"
    if phase_code in {PHASE_FULL}:
        return "full"
    return "full"


def _b_authority_distance(theta_B_candidate, b_labels, b_label_weights):
    theta_B_candidate = np.asarray(theta_B_candidate, float).reshape(-1)
    weights = np.asarray(
        [float(b_label_weights.get(str(label), 1.0)) for label in tuple(str(label) for label in b_labels)],
        float,
    )
    if theta_B_candidate.shape != weights.shape:
        raise ValueError("theta_B_candidate and b_labels must define the same length.")
    return float(np.linalg.norm(weights * (theta_B_candidate - 1.0), ord=2))


def _gain_drift_ratio(A_candidate, B_candidate, A_nominal, B_nominal, C_matrix, predict_h):
    A_candidate = np.asarray(A_candidate, float)
    B_candidate = np.asarray(B_candidate, float)
    A_nominal = np.asarray(A_nominal, float)
    B_nominal = np.asarray(B_nominal, float)
    C_matrix = np.asarray(C_matrix, float)
    if not (
        np.all(np.isfinite(A_candidate))
        and np.all(np.isfinite(B_candidate))
        and np.all(np.isfinite(A_nominal))
        and np.all(np.isfinite(B_nominal))
        and np.all(np.isfinite(C_matrix))
    ):
        return np.nan
    G_candidate = build_markov_matrix(A_candidate, B_candidate, C_matrix, int(predict_h))
    G_nominal = build_markov_matrix(A_nominal, B_nominal, C_matrix, int(predict_h))
    denom = float(np.linalg.norm(G_nominal, ord="fro")) + 1e-12
    return float(np.linalg.norm(G_candidate - G_nominal, ord="fro") / denom)


def _safe_fun(sol):
    if sol is None:
        return np.nan
    return float(getattr(sol, "fun", np.nan))


__all__ = [
    "ACCEPTANCE_REASON_ACCEPTED",
    "ACCEPTANCE_REASON_CANDIDATE_SOLVE_FAILED",
    "ACCEPTANCE_REASON_DISABLED",
    "ACCEPTANCE_REASON_REJECTED_COST",
    "DUAL_COST_SHADOW_REASON_CANDIDATE_SOLVE_FAILED",
    "DUAL_COST_SHADOW_REASON_DISABLED",
    "DUAL_COST_SHADOW_REASON_EVALUATED",
    "USEFULNESS_GATE_REASON_ACCEPTED",
    "USEFULNESS_GATE_REASON_CANDIDATE_SOLVE_FAILED",
    "USEFULNESS_GATE_REASON_DISABLED",
    "USEFULNESS_GATE_REASON_REJECTED_CANDIDATE_USEFULNESS",
    "USEFULNESS_GATE_REASON_REJECTED_GAIN_DRIFT",
    "USEFULNESS_GATE_REASON_REJECTED_NOMINAL_SAFETY",
    "run_mpc_acceptance_gate",
    "run_mpc_dual_cost_shadow",
    "run_mpc_usefulness_gate",
]
