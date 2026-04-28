from __future__ import annotations

import numpy as np


ACCEPTANCE_REASON_DISABLED = 0
ACCEPTANCE_REASON_ACCEPTED = 1
ACCEPTANCE_REASON_REJECTED_COST = 2
ACCEPTANCE_REASON_CANDIDATE_SOLVE_FAILED = 3

DUAL_COST_SHADOW_REASON_DISABLED = 0
DUAL_COST_SHADOW_REASON_EVALUATED = 1
DUAL_COST_SHADOW_REASON_CANDIDATE_SOLVE_FAILED = 2


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

    candidate_sol = None
    candidate_solve_failed = False
    if not (np.all(np.isfinite(A_candidate)) and np.all(np.isfinite(B_candidate))):
        candidate_solve_failed = True
        if (not enabled) or (not fallback_on_candidate_solve_failure):
            raise RuntimeError(f"Candidate MPC model became non-finite at step {step_idx}.")
    else:
        try:
            candidate_sol = _solve_with_model(A_candidate, B_candidate)
        except RuntimeError:
            candidate_solve_failed = True
            if (not enabled) or (not fallback_on_candidate_solve_failure):
                raise

    if not enabled:
        mpc_obj.A = A_candidate
        mpc_obj.B = B_candidate
        return {
            "sol": candidate_sol,
            "accepted": True,
            "fallback_active": False,
            "reason_code": ACCEPTANCE_REASON_DISABLED,
            "candidate_cost_on_nominal": np.nan,
            "candidate_cost_native": _safe_fun(candidate_sol),
            "nominal_cost": np.nan,
            "cost_margin": np.nan,
            "threshold": np.nan,
        }

    nominal_sol = _solve_with_model(A_nominal, B_nominal)
    nominal_cost = _safe_fun(nominal_sol)

    if candidate_solve_failed:
        mpc_obj.A = A_nominal
        mpc_obj.B = B_nominal
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

    mpc_obj.A = A_nominal
    mpc_obj.B = B_nominal
    candidate_cost_on_nominal = float(mpc_obj.mpc_opt_fun(candidate_sol.x, y_sp, u_prev_dev, x0_model))
    threshold = (1.0 + relative_tolerance) * nominal_cost + absolute_tolerance
    accepted = bool(candidate_cost_on_nominal <= threshold)
    if accepted:
        mpc_obj.A = A_candidate
        mpc_obj.B = B_candidate
        selected_sol = candidate_sol
        reason_code = ACCEPTANCE_REASON_ACCEPTED
    else:
        mpc_obj.A = A_nominal
        mpc_obj.B = B_nominal
        selected_sol = nominal_sol
        reason_code = ACCEPTANCE_REASON_REJECTED_COST

    return {
        "sol": selected_sol,
        "accepted": accepted,
        "fallback_active": not accepted,
        "reason_code": int(reason_code),
        "candidate_cost_on_nominal": candidate_cost_on_nominal,
        "candidate_cost_native": _safe_fun(candidate_sol),
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

    if not enabled:
        mpc_obj.A = A_candidate
        mpc_obj.B = B_candidate
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
    safe_threshold = (
        relative_tolerance * nominal_cost + absolute_tolerance
        if np.isfinite(nominal_cost)
        else np.nan
    )
    candidate_advantage = (
        nominal_cost_on_candidate - candidate_cost_native
        if np.isfinite(nominal_cost_on_candidate) and np.isfinite(candidate_cost_native)
        else np.nan
    )
    safe_pass = bool(np.isfinite(nominal_penalty) and np.isfinite(safe_threshold) and nominal_penalty <= safe_threshold)
    benefit_pass = bool(np.isfinite(candidate_advantage) and candidate_advantage >= benefit_tolerance)
    dual_pass = bool(safe_pass and benefit_pass)

    if candidate_solve_failed:
        mpc_obj.A = A_nominal
        mpc_obj.B = B_nominal
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

    mpc_obj.A = A_candidate
    mpc_obj.B = B_candidate
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
    "run_mpc_acceptance_gate",
    "run_mpc_dual_cost_shadow",
]
