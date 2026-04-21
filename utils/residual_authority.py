import numpy as np


def map_to_bounds(action, low, high):
    action = np.asarray(action, float).reshape(-1)
    low = np.asarray(low, float).reshape(-1)
    high = np.asarray(high, float).reshape(-1)
    return low + ((action + 1.0) / 2.0) * (high - low)


def map_from_bounds(value, low, high):
    value = np.asarray(value, float).reshape(-1)
    low = np.asarray(low, float).reshape(-1)
    high = np.asarray(high, float).reshape(-1)
    return 2.0 * (value - low) / np.maximum(high - low, 1e-12) - 1.0


def compute_residual_rho(*, tracking_values, rho_mapping_mode="clipped_linear", authority_rho_k=0.55):
    tracking_values = np.asarray(tracking_values, float).reshape(-1)
    max_abs_tracking = float(np.max(np.abs(tracking_values))) if tracking_values.size > 0 else 0.0
    mapping_mode = str(rho_mapping_mode or "clipped_linear").strip().lower()
    if mapping_mode == "exp_raw_tracking":
        rho = float(1.0 - np.exp(-float(authority_rho_k) * max_abs_tracking))
    elif mapping_mode == "clipped_linear":
        rho = float(np.clip(max_abs_tracking, 0.0, 1.0))
    else:
        raise ValueError("rho_mapping_mode must be 'clipped_linear' or 'exp_raw_tracking'.")
    return {
        "max_abs_tracking": max_abs_tracking,
        "rho": float(np.clip(rho, 0.0, 1.0)),
        "rho_mapping_mode": mapping_mode,
    }


def project_residual_action(
    *,
    action_raw,
    low_coef,
    high_coef,
    u_base,
    scaled_current_input,
    u_min_scaled_abs,
    u_max_scaled_abs,
    apply_authority,
    authority_use_rho,
    tracking_error_feat=None,
    tracking_error_raw=None,
    innovation_raw=None,
    authority_beta_res=None,
    authority_du0_res=None,
    authority_rho_floor=0.15,
    authority_rho_power=1.0,
    rho_mapping_mode="clipped_linear",
    authority_rho_k=0.55,
    residual_zero_deadband_enabled=False,
    residual_zero_tracking_raw_threshold=0.1,
    residual_zero_innovation_raw_threshold=0.1,
    tol=1e-9,
):
    action_raw = np.asarray(action_raw, float).reshape(-1)
    low_coef = np.asarray(low_coef, float).reshape(-1)
    high_coef = np.asarray(high_coef, float).reshape(-1)
    u_base = np.asarray(u_base, float).reshape(-1)
    scaled_current_input = np.asarray(scaled_current_input, float).reshape(-1)
    u_min_scaled_abs = np.asarray(u_min_scaled_abs, float).reshape(-1)
    u_max_scaled_abs = np.asarray(u_max_scaled_abs, float).reshape(-1)

    delta_u_res_raw = map_to_bounds(action_raw, low_coef, high_coef).astype(np.float32)
    low_headroom = (u_min_scaled_abs - u_base).astype(np.float32)
    high_headroom = (u_max_scaled_abs - u_base).astype(np.float32)

    rho_raw = None
    rho = None
    rho_eff = None
    authority_low = None
    authority_high = None
    deadband_active = False
    projection_due_to_deadband = False
    rho_raw_source = None
    max_abs_tracking_raw = None
    max_abs_innovation_raw = None

    if apply_authority:
        if tracking_error_feat is None and tracking_error_raw is None:
            raise ValueError("Residual authority projection requires tracking_error_feat or tracking_error_raw.")
        if authority_beta_res is None or authority_du0_res is None:
            raise ValueError("Residual authority projection requires authority_beta_res and authority_du0_res.")
        tracking_error_feat = None if tracking_error_feat is None else np.asarray(tracking_error_feat, float).reshape(-1)
        tracking_error_raw = None if tracking_error_raw is None else np.asarray(tracking_error_raw, float).reshape(-1)
        innovation_raw = None if innovation_raw is None else np.asarray(innovation_raw, float).reshape(-1)
        authority_beta_res = np.asarray(authority_beta_res, float).reshape(-1)
        authority_du0_res = np.asarray(authority_du0_res, float).reshape(-1)
        if authority_beta_res.size != delta_u_res_raw.size or authority_du0_res.size != delta_u_res_raw.size:
            raise ValueError("authority_beta_res and authority_du0_res must match residual action dimension.")
        rho_source = tracking_error_raw if tracking_error_raw is not None else tracking_error_feat
        rho_raw_source = "tracking_error_raw" if tracking_error_raw is not None else "tracking_error_feat"
        rho_payload = compute_residual_rho(
            tracking_values=rho_source,
            rho_mapping_mode=rho_mapping_mode,
            authority_rho_k=authority_rho_k,
        )
        max_abs_tracking_raw = float(rho_payload["max_abs_tracking"])
        rho_raw = max_abs_tracking_raw
        rho = float(rho_payload["rho"])
        if authority_use_rho:
            rho_eff = float(np.clip(authority_rho_floor, 0.0, 1.0) + (1.0 - np.clip(authority_rho_floor, 0.0, 1.0)) * (rho ** float(authority_rho_power)))
        else:
            rho_eff = 1.0
        max_abs_innovation_raw = (
            float(np.max(np.abs(innovation_raw))) if innovation_raw is not None and innovation_raw.size > 0 else 0.0
        )
        deadband_active = bool(
            residual_zero_deadband_enabled
            and max_abs_tracking_raw <= float(residual_zero_tracking_raw_threshold)
            and max_abs_innovation_raw <= float(residual_zero_innovation_raw_threshold)
        )
        delta_u_res_candidate = np.zeros_like(delta_u_res_raw) if deadband_active else delta_u_res_raw
        delta_u_mpc = (u_base - scaled_current_input).astype(np.float32)
        mag = (rho_eff * authority_beta_res) * (np.abs(delta_u_mpc) + authority_du0_res)
        authority_low = (-mag).astype(np.float32)
        authority_high = mag.astype(np.float32)
        final_low = np.maximum(authority_low, low_headroom)
        final_high = np.minimum(authority_high, high_headroom)
    else:
        delta_u_res_candidate = delta_u_res_raw
        final_low = np.maximum(low_coef, low_headroom).astype(np.float32)
        final_high = np.minimum(high_coef, high_headroom).astype(np.float32)

    bad = final_low > final_high
    if np.any(bad):
        final_low = final_low.copy()
        final_high = final_high.copy()
        final_low[bad] = 0.0
        final_high[bad] = 0.0

    delta_u_res_exec = np.clip(delta_u_res_candidate, final_low, final_high).astype(np.float32)
    u_applied_scaled_abs = np.clip(u_base + delta_u_res_exec, u_min_scaled_abs, u_max_scaled_abs).astype(np.float32)
    delta_u_res_exec = (u_applied_scaled_abs - u_base).astype(np.float32)

    action_exec = np.clip(map_from_bounds(delta_u_res_exec, low_coef, high_coef), -1.0, 1.0).astype(np.float32)
    projection_active = bool(np.any(np.abs(delta_u_res_exec - delta_u_res_raw) > tol))
    projection_due_to_deadband = bool(deadband_active and np.any(np.abs(delta_u_res_raw) > tol))
    projection_due_to_authority = bool(
        apply_authority
        and authority_low is not None
        and authority_high is not None
        and np.any((delta_u_res_candidate < authority_low - tol) | (delta_u_res_candidate > authority_high + tol))
    )
    projection_due_to_headroom = bool(
        np.any((delta_u_res_candidate < low_headroom - tol) | (delta_u_res_candidate > high_headroom + tol))
    )

    return {
        "a_raw": action_raw.astype(np.float32),
        "a_exec": action_exec,
        "delta_u_res_raw": delta_u_res_raw,
        "delta_u_res_exec": delta_u_res_exec,
        "rho_raw": rho_raw,
        "rho": rho,
        "rho_eff": rho_eff,
        "rho_raw_source": rho_raw_source,
        "rho_mapping_mode": None if rho is None else str(rho_mapping_mode),
        "max_abs_tracking_raw": max_abs_tracking_raw,
        "max_abs_innovation_raw": max_abs_innovation_raw,
        "authority_low": None if authority_low is None else authority_low.astype(np.float32),
        "authority_high": None if authority_high is None else authority_high.astype(np.float32),
        "low_headroom": low_headroom,
        "high_headroom": high_headroom,
        "final_low": final_low.astype(np.float32),
        "final_high": final_high.astype(np.float32),
        "u_applied_scaled_abs": u_applied_scaled_abs,
        "deadband_active": deadband_active,
        "projection_active": projection_active,
        "projection_due_to_deadband": projection_due_to_deadband,
        "projection_due_to_authority": projection_due_to_authority,
        "projection_due_to_headroom": projection_due_to_headroom,
    }
