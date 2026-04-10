import numpy as np
import scipy.optimize as spo


FALLBACK_SOURCE_CODES = {
    "assisted_tight": 0,
    "nominal_tight": 1,
    "nominal_full": 2,
}


def build_tightened_input_bounds(b_min, b_max, tightening_frac):
    b_min = np.asarray(b_min, float).reshape(-1)
    b_max = np.asarray(b_max, float).reshape(-1)
    if b_min.shape != b_max.shape:
        raise ValueError("b_min and b_max must have the same shape.")
    if np.any(~np.isfinite(b_min)) or np.any(~np.isfinite(b_max)):
        raise ValueError("Input bounds must be finite.")
    if np.any(b_max <= b_min):
        raise ValueError("Each upper input bound must be greater than the lower bound.")

    frac = float(max(0.0, tightening_frac))
    span = b_max - b_min
    margin = np.clip(frac * span, 0.0, 0.49 * span)
    b_min_tight = b_min + margin
    b_max_tight = b_max - margin
    bad = b_min_tight >= b_max_tight
    if np.any(bad):
        b_min_tight[bad] = b_min[bad]
        b_max_tight[bad] = b_max[bad]
        margin[bad] = 0.0
    return {
        "b_min": b_min,
        "b_max": b_max,
        "b_min_tight": b_min_tight,
        "b_max_tight": b_max_tight,
        "margin": margin,
        "tightening_frac": frac,
    }


def repeat_bounds_for_horizon(b_min, b_max, horizon):
    b_min = np.asarray(b_min, float).reshape(-1)
    b_max = np.asarray(b_max, float).reshape(-1)
    horizon = int(horizon)
    return tuple(
        (float(b_min[j]), float(b_max[j]))
        for _ in range(horizon)
        for j in range(b_min.size)
    )


def build_probe_input_sequence(u_probe, horizon):
    u_probe = np.asarray(u_probe, float).reshape(-1)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("Probe horizon must be positive.")
    return np.tile(u_probe.reshape(1, -1), (horizon, 1))


def rollout_prediction(A, B, C, x0, u_sequence):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    x0 = np.asarray(x0, float).reshape(-1)
    u_sequence = np.asarray(u_sequence, float)
    if u_sequence.ndim != 2:
        raise ValueError("u_sequence must be 2D with shape (horizon, n_inputs).")
    horizon = int(u_sequence.shape[0])
    x_pred = np.zeros((A.shape[0], horizon + 1), dtype=float)
    x_pred[:, 0] = x0
    for j in range(horizon):
        x_pred[:, j + 1] = A @ x_pred[:, j] + B @ u_sequence[j, :]
    y_pred = (C @ x_pred).T
    return {
        "x_pred": x_pred,
        "y_pred": y_pred,
    }


def prediction_deviation_metric(A_nom, B_nom, A_cand, B_cand, C, x0, u_probe, horizon):
    probe_u = build_probe_input_sequence(u_probe, horizon)
    nominal = rollout_prediction(A_nom, B_nom, C, x0, probe_u)
    candidate = rollout_prediction(A_cand, B_cand, C, x0, probe_u)
    diff = np.asarray(candidate["y_pred"][1:, :] - nominal["y_pred"][1:, :], float)
    if diff.size == 0:
        max_dev = 0.0
        mean_dev = 0.0
    else:
        row_norms = np.linalg.norm(diff, axis=1)
        max_dev = float(np.max(row_norms))
        mean_dev = float(np.mean(row_norms))
    return {
        "max_dev": max_dev,
        "mean_dev": mean_dev,
        "nominal_y_pred": nominal["y_pred"],
        "candidate_y_pred": candidate["y_pred"],
        "diff_y_pred": diff,
    }


def _relative_frobenius(candidate, nominal):
    candidate = np.asarray(candidate, float)
    nominal = np.asarray(nominal, float)
    denom = float(np.linalg.norm(nominal, ord="fro"))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(candidate - nominal, ord="fro") / denom)


def validate_prediction_candidate(
    *,
    A_nom,
    B_nom,
    A_candidate,
    B_candidate,
    C,
    x0,
    u_probe,
    low_bounds=None,
    high_bounds=None,
    mapped_action=None,
    enable_accept_norm_test=True,
    eps_A_norm_frac=0.05,
    eps_B_norm_frac=0.05,
    enable_accept_prediction_test=True,
    prediction_check_horizon=2,
    eps_y_pred_scaled=0.10,
):
    A_candidate = np.asarray(A_candidate, float)
    B_candidate = np.asarray(B_candidate, float)
    finite_ok = bool(np.all(np.isfinite(A_candidate)) and np.all(np.isfinite(B_candidate)))

    bounds_ok = True
    if low_bounds is not None and high_bounds is not None and mapped_action is not None:
        low_bounds = np.asarray(low_bounds, float).reshape(-1)
        high_bounds = np.asarray(high_bounds, float).reshape(-1)
        mapped_action = np.asarray(mapped_action, float).reshape(-1)
        bounds_ok = bool(
            mapped_action.shape == low_bounds.shape
            and np.all(np.isfinite(mapped_action))
            and np.all(mapped_action >= (low_bounds - 1e-12))
            and np.all(mapped_action <= (high_bounds + 1e-12))
        )

    A_rel = _relative_frobenius(A_candidate, A_nom)
    B_rel = _relative_frobenius(B_candidate, B_nom)
    norm_ok = True
    if enable_accept_norm_test:
        norm_ok = bool(A_rel <= float(eps_A_norm_frac) and B_rel <= float(eps_B_norm_frac))

    pred_info = {
        "max_dev": 0.0,
        "mean_dev": 0.0,
        "nominal_y_pred": None,
        "candidate_y_pred": None,
        "diff_y_pred": None,
    }
    prediction_ok = True
    if finite_ok and enable_accept_prediction_test:
        pred_info = prediction_deviation_metric(
            A_nom=A_nom,
            B_nom=B_nom,
            A_cand=A_candidate,
            B_cand=B_candidate,
            C=C,
            x0=x0,
            u_probe=u_probe,
            horizon=prediction_check_horizon,
        )
        prediction_ok = bool(pred_info["max_dev"] <= float(eps_y_pred_scaled))
    elif enable_accept_prediction_test and not finite_ok:
        prediction_ok = False

    accepted = bool(finite_ok and bounds_ok and norm_ok and prediction_ok)
    return {
        "accepted": accepted,
        "finite_ok": finite_ok,
        "bounds_ok": bounds_ok,
        "norm_ok": norm_ok,
        "prediction_ok": prediction_ok,
        "A_rel": A_rel,
        "B_rel": B_rel,
        "prediction_max_dev": float(pred_info["max_dev"]),
        "prediction_mean_dev": float(pred_info["mean_dev"]),
        "nominal_y_pred": pred_info["nominal_y_pred"],
        "candidate_y_pred": pred_info["candidate_y_pred"],
        "diff_y_pred": pred_info["diff_y_pred"],
    }


def attempt_mpc_solve(mpc_obj, y_sp, u_prev_dev, x0_model, initial_guess, bounds, constraints=None):
    constraints = [] if constraints is None else constraints
    try:
        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
            np.asarray(initial_guess, float),
            bounds=bounds,
            constraints=constraints,
        )
    except Exception as exc:
        return {
            "success": False,
            "sol": None,
            "message": str(exc),
        }

    success = bool(
        sol is not None
        and bool(getattr(sol, "success", True))
        and getattr(sol, "x", None) is not None
        and np.all(np.isfinite(np.asarray(sol.x, float)))
        and np.isfinite(float(getattr(sol, "fun", 0.0)))
    )
    return {
        "success": success,
        "sol": sol,
        "message": str(getattr(sol, "message", "")),
    }


def solve_prediction_mpc_with_fallback(
    *,
    mpc_obj,
    y_sp,
    u_prev_dev,
    x0_model,
    initial_guess,
    A_nom,
    B_nom,
    A_assisted,
    B_assisted,
    candidate_accepted,
    tightened_bounds,
    original_bounds,
    constraints=None,
    enable_solver_fallback=True,
    compute_nominal_reference=True,
):
    nominal_reference = None
    if compute_nominal_reference:
        mpc_obj.A = np.asarray(A_nom, float)
        mpc_obj.B = np.asarray(B_nom, float)
        nominal_reference = attempt_mpc_solve(
            mpc_obj=mpc_obj,
            y_sp=y_sp,
            u_prev_dev=u_prev_dev,
            x0_model=x0_model,
            initial_guess=initial_guess,
            bounds=tightened_bounds,
            constraints=constraints,
        )

    attempts = []
    if candidate_accepted:
        attempts.append(("assisted_tight", A_assisted, B_assisted, tightened_bounds))
    attempts.append(("nominal_tight", A_nom, B_nom, tightened_bounds))
    if enable_solver_fallback:
        attempts.append(("nominal_full", A_nom, B_nom, original_bounds))

    seen = set()
    chosen = None
    attempt_results = {}
    for source, A_use, B_use, bounds in attempts:
        key = (source, tuple(np.asarray(bounds, float).reshape(-1))) if bounds is not None else (source, None)
        if key in seen:
            continue
        seen.add(key)
        mpc_obj.A = np.asarray(A_use, float)
        mpc_obj.B = np.asarray(B_use, float)
        result = attempt_mpc_solve(
            mpc_obj=mpc_obj,
            y_sp=y_sp,
            u_prev_dev=u_prev_dev,
            x0_model=x0_model,
            initial_guess=initial_guess,
            bounds=bounds,
            constraints=constraints,
        )
        attempt_results[source] = result
        if result["success"]:
            chosen = (source, result, np.asarray(A_use, float), np.asarray(B_use, float), bounds)
            break

    if chosen is None:
        raise RuntimeError("All MPC solve attempts failed for the robust matrix-prediction step.")

    source, result, A_used, B_used, bounds_used = chosen
    nominal_first_move = None
    if nominal_reference is not None and nominal_reference["success"]:
        nominal_first_move = np.asarray(nominal_reference["sol"].x, float)[: np.asarray(B_nom, float).shape[1]]

    first_move = np.asarray(result["sol"].x, float)[: np.asarray(B_used, float).shape[1]]
    if nominal_first_move is None:
        first_move_delta = np.full(first_move.shape, np.nan, dtype=float)
    else:
        first_move_delta = np.asarray(first_move - nominal_first_move, float)

    return {
        "source": source,
        "source_code": int(FALLBACK_SOURCE_CODES[source]),
        "sol": result["sol"],
        "A_used": A_used,
        "B_used": B_used,
        "bounds_used": bounds_used,
        "nominal_reference": nominal_reference,
        "attempt_results": attempt_results,
        "first_move_delta_vs_nominal_tight": first_move_delta,
    }


def sweep_prediction_multiplier_sensitivity(
    *,
    build_prediction_model,
    theta_center,
    parameter_names,
    sweep_values,
    A_nom,
    B_nom,
    C,
    x0,
    u_probe,
    prediction_horizon,
    first_move_solver=None,
):
    theta_center = np.asarray(theta_center, float).reshape(-1)
    parameter_names = list(parameter_names)
    if theta_center.size != len(parameter_names):
        raise ValueError("theta_center and parameter_names must have the same length.")

    rows = []
    for idx, name in enumerate(parameter_names):
        for value in np.asarray(sweep_values, float).reshape(-1):
            theta = theta_center.copy()
            theta[idx] = float(value)
            A_pred, B_pred, meta = build_prediction_model(theta)
            pred = prediction_deviation_metric(
                A_nom=A_nom,
                B_nom=B_nom,
                A_cand=A_pred,
                B_cand=B_pred,
                C=C,
                x0=x0,
                u_probe=u_probe,
                horizon=prediction_horizon,
            )
            row = {
                "parameter": name,
                "index": int(idx),
                "value": float(value),
                "A_rel": _relative_frobenius(A_pred, A_nom),
                "B_rel": _relative_frobenius(B_pred, B_nom),
                "prediction_max_dev": float(pred["max_dev"]),
                "prediction_mean_dev": float(pred["mean_dev"]),
                "meta": meta,
            }
            if first_move_solver is not None:
                row["first_move_delta"] = np.asarray(first_move_solver(A_pred, B_pred), float)
            rows.append(row)
    return rows
