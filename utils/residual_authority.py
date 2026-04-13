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
    authority_beta_res=None,
    authority_du0_res=None,
    authority_rho_floor=0.15,
    authority_rho_power=1.0,
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

    if apply_authority:
        if tracking_error_feat is None:
            raise ValueError("Residual authority projection requires tracking_error_feat.")
        if authority_beta_res is None or authority_du0_res is None:
            raise ValueError("Residual authority projection requires authority_beta_res and authority_du0_res.")
        tracking_error_feat = np.asarray(tracking_error_feat, float).reshape(-1)
        authority_beta_res = np.asarray(authority_beta_res, float).reshape(-1)
        authority_du0_res = np.asarray(authority_du0_res, float).reshape(-1)
        if authority_beta_res.size != delta_u_res_raw.size or authority_du0_res.size != delta_u_res_raw.size:
            raise ValueError("authority_beta_res and authority_du0_res must match residual action dimension.")
        track_norm = np.abs(tracking_error_feat)
        rho_raw = float(np.max(track_norm)) if track_norm.size > 0 else 0.0
        rho = float(np.clip(rho_raw, 0.0, 1.0))
        if authority_use_rho:
            rho_eff = float(np.clip(authority_rho_floor, 0.0, 1.0) + (1.0 - np.clip(authority_rho_floor, 0.0, 1.0)) * (rho ** float(authority_rho_power)))
        else:
            rho_eff = 1.0
        delta_u_mpc = (u_base - scaled_current_input).astype(np.float32)
        mag = (rho_eff * authority_beta_res) * (np.abs(delta_u_mpc) + authority_du0_res)
        authority_low = (-mag).astype(np.float32)
        authority_high = mag.astype(np.float32)
        final_low = np.maximum(authority_low, low_headroom)
        final_high = np.minimum(authority_high, high_headroom)
    else:
        final_low = np.maximum(low_coef, low_headroom).astype(np.float32)
        final_high = np.minimum(high_coef, high_headroom).astype(np.float32)

    bad = final_low > final_high
    if np.any(bad):
        final_low = final_low.copy()
        final_high = final_high.copy()
        final_low[bad] = 0.0
        final_high[bad] = 0.0

    delta_u_res_exec = np.clip(delta_u_res_raw, final_low, final_high).astype(np.float32)
    u_applied_scaled_abs = np.clip(u_base + delta_u_res_exec, u_min_scaled_abs, u_max_scaled_abs).astype(np.float32)
    delta_u_res_exec = (u_applied_scaled_abs - u_base).astype(np.float32)

    action_exec = np.clip(map_from_bounds(delta_u_res_exec, low_coef, high_coef), -1.0, 1.0).astype(np.float32)
    projection_active = bool(np.any(np.abs(delta_u_res_exec - delta_u_res_raw) > tol))
    projection_due_to_authority = bool(
        apply_authority
        and authority_low is not None
        and authority_high is not None
        and np.any((delta_u_res_raw < authority_low - tol) | (delta_u_res_raw > authority_high + tol))
    )
    projection_due_to_headroom = bool(
        np.any((delta_u_res_raw < low_headroom - tol) | (delta_u_res_raw > high_headroom + tol))
    )

    return {
        "a_raw": action_raw.astype(np.float32),
        "a_exec": action_exec,
        "delta_u_res_raw": delta_u_res_raw,
        "delta_u_res_exec": delta_u_res_exec,
        "rho_raw": rho_raw,
        "rho": rho,
        "rho_eff": rho_eff,
        "authority_low": None if authority_low is None else authority_low.astype(np.float32),
        "authority_high": None if authority_high is None else authority_high.astype(np.float32),
        "low_headroom": low_headroom,
        "high_headroom": high_headroom,
        "final_low": final_low.astype(np.float32),
        "final_high": final_high.astype(np.float32),
        "u_applied_scaled_abs": u_applied_scaled_abs,
        "projection_active": projection_active,
        "projection_due_to_authority": projection_due_to_authority,
        "projection_due_to_headroom": projection_due_to_headroom,
    }
