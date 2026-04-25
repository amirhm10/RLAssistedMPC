from __future__ import annotations

import numpy as np


ACCEPTANCE_REASON_DISABLED = 0
ACCEPTANCE_REASON_ACCEPTED = 1
ACCEPTANCE_REASON_REJECTED_COST = 2
ACCEPTANCE_REASON_CANDIDATE_SOLVE_FAILED = 3


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


def _safe_fun(sol):
    if sol is None:
        return np.nan
    return float(getattr(sol, "fun", np.nan))


__all__ = [
    "ACCEPTANCE_REASON_ACCEPTED",
    "ACCEPTANCE_REASON_CANDIDATE_SOLVE_FAILED",
    "ACCEPTANCE_REASON_DISABLED",
    "ACCEPTANCE_REASON_REJECTED_COST",
    "run_mpc_acceptance_gate",
]
