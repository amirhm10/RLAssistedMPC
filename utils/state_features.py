import numpy as np

from utils.helpers import apply_rl_scaled, reverse_min_max


def default_mismatch_scale(min_max_dict, floor=1e-3):
    """
    Deprecated compatibility helper.

    New mismatch-v2 logic should use reward-band-derived scales instead of the
    setpoint-span heuristic kept here for older bundles or untouched codepaths.
    """
    y_sp_min = np.asarray(min_max_dict["y_sp_min"], float)
    y_sp_max = np.asarray(min_max_dict["y_sp_max"], float)
    scale = np.maximum(np.abs(y_sp_min), np.abs(y_sp_max))
    return np.maximum(scale, float(floor)).astype(np.float32)


def compute_band_scaled_from_reward(y_sp_phys, data_min, data_max, n_inputs, k_rel, band_floor_phys):
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    y_sp_phys = np.asarray(y_sp_phys, float)
    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    dy = np.maximum(data_max[int(n_inputs):] - data_min[int(n_inputs):], 1e-12)
    band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor_phys)
    return (band_phys / dy).astype(np.float32)


def compute_band_ref_scaled(y_sp_scenario, steady_states, data_min, data_max, n_inputs, k_rel, band_floor_phys):
    y_sp_scenario = np.asarray(y_sp_scenario, float)
    if y_sp_scenario.ndim == 1:
        y_sp_scenario = y_sp_scenario.reshape(1, -1)
    y_ss_phys = np.asarray(steady_states["y_ss"], float)
    y_ss_scaled = apply_min_max_vector(y_ss_phys, data_min[int(n_inputs):], data_max[int(n_inputs):])
    y_sp_phys = reverse_min_max(y_sp_scenario + y_ss_scaled, data_min[int(n_inputs):], data_max[int(n_inputs):])
    band_scaled = compute_band_scaled_from_reward(
        y_sp_phys=y_sp_phys,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
        k_rel=k_rel,
        band_floor_phys=band_floor_phys,
    )
    return np.median(np.asarray(band_scaled, float), axis=0).astype(np.float32)


def apply_min_max_vector(values, vmin, vmax):
    values = np.asarray(values, float)
    vmin = np.asarray(vmin, float)
    vmax = np.asarray(vmax, float)
    return ((values - vmin) / np.maximum(vmax - vmin, 1e-12)).astype(np.float32)


def resolve_mismatch_settings(
    *,
    state_mode,
    mismatch_cfg,
    reward_params,
    y_sp_scenario,
    steady_states,
    data_min,
    data_max,
    n_inputs,
):
    state_mode = str(state_mode).lower()
    mismatch_cfg = dict(mismatch_cfg or {})
    mismatch_clip = mismatch_cfg.get("mismatch_clip", 3.0)
    if state_mode != "mismatch":
        return {
            "mismatch_clip": mismatch_clip,
            "innovation_scale_mode": mismatch_cfg.get("innovation_scale_mode", "band_ref"),
            "innovation_scale_ref": None,
            "tracking_scale_mode": mismatch_cfg.get("tracking_scale_mode", "eta_band"),
            "tracking_eta_tol": float(mismatch_cfg.get("tracking_eta_tol", 0.3)),
            "tracking_scale_floor_mode": mismatch_cfg.get("tracking_scale_floor_mode", "half_eta_band_ref"),
            "tracking_scale_floor": None,
            "band_ref_scaled": None,
            "append_rho_to_state": bool(mismatch_cfg.get("append_rho_to_state", False)),
            "k_rel": None,
            "band_floor_phys": None,
        }

    k_rel = np.asarray(reward_params.get("k_rel"), float)
    band_floor_phys = np.asarray(reward_params.get("band_floor_phys"), float)
    band_ref_scaled = compute_band_ref_scaled(
        y_sp_scenario=y_sp_scenario,
        steady_states=steady_states,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
        k_rel=k_rel,
        band_floor_phys=band_floor_phys,
    )
    innovation_scale_mode = str(mismatch_cfg.get("innovation_scale_mode", "band_ref")).lower()
    innovation_scale_ref = mismatch_cfg.get("innovation_scale_ref")
    if innovation_scale_ref is None or innovation_scale_mode == "band_ref":
        innovation_scale_ref = band_ref_scaled.copy()
    innovation_scale_ref = np.asarray(innovation_scale_ref, float).reshape(-1)
    if innovation_scale_ref.shape[0] != band_ref_scaled.shape[0]:
        raise ValueError("innovation_scale_ref must match the number of outputs.")

    tracking_eta_tol = float(mismatch_cfg.get("tracking_eta_tol", 0.3))
    tracking_scale_floor_mode = str(mismatch_cfg.get("tracking_scale_floor_mode", "half_eta_band_ref")).lower()
    tracking_scale_floor = mismatch_cfg.get("tracking_scale_floor")
    if tracking_scale_floor is None:
        if tracking_scale_floor_mode == "half_eta_band_ref":
            tracking_scale_floor = 0.5 * tracking_eta_tol * band_ref_scaled
        else:
            tracking_scale_floor = band_ref_scaled.copy()
    tracking_scale_floor = np.asarray(tracking_scale_floor, float).reshape(-1)
    if tracking_scale_floor.shape[0] != band_ref_scaled.shape[0]:
        raise ValueError("tracking_scale_floor must match the number of outputs.")

    return {
        "mismatch_clip": mismatch_clip,
        "innovation_scale_mode": innovation_scale_mode,
        "innovation_scale_ref": innovation_scale_ref.astype(np.float32),
        "tracking_scale_mode": str(mismatch_cfg.get("tracking_scale_mode", "eta_band")).lower(),
        "tracking_eta_tol": tracking_eta_tol,
        "tracking_scale_floor_mode": tracking_scale_floor_mode,
        "tracking_scale_floor": tracking_scale_floor.astype(np.float32),
        "band_ref_scaled": band_ref_scaled.astype(np.float32),
        "append_rho_to_state": bool(mismatch_cfg.get("append_rho_to_state", False)),
        "k_rel": k_rel.astype(np.float32),
        "band_floor_phys": band_floor_phys.astype(np.float32),
    }


def compute_tracking_scale_now(
    *,
    y_sp_phys,
    data_min,
    data_max,
    n_inputs,
    k_rel,
    band_floor_phys,
    tracking_eta_tol,
    tracking_scale_floor,
):
    band_scaled_now = compute_band_scaled_from_reward(
        y_sp_phys=y_sp_phys,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
        k_rel=k_rel,
        band_floor_phys=band_floor_phys,
    )
    tracking_scale_floor = np.asarray(tracking_scale_floor, float)
    tracking_scale_now = np.maximum(float(tracking_eta_tol) * band_scaled_now, tracking_scale_floor)
    return band_scaled_now.astype(np.float32), tracking_scale_now.astype(np.float32)


def get_rl_state_dim(base_aug_dim, n_outputs, n_inputs, state_mode, append_rho_to_state=False):
    state_mode = str(state_mode).lower()
    base_dim = int(base_aug_dim) + int(n_outputs) + int(n_inputs)
    if state_mode == "standard":
        return base_dim
    if state_mode == "mismatch":
        return base_dim + 2 * int(n_outputs) + (1 if append_rho_to_state else 0)
    raise ValueError("state_mode must be 'standard' or 'mismatch'.")


def build_rl_state(
    min_max_dict,
    x_d_states,
    y_sp,
    u,
    state_mode,
    y_prev_scaled=None,
    yhat_pred=None,
    innovation_scale_ref=None,
    tracking_scale_now=None,
    mismatch_clip=3.0,
    append_rho_to_state=False,
    rho_value=None,
):
    state_mode = str(state_mode).lower()
    base_state = np.asarray(apply_rl_scaled(min_max_dict, x_d_states, y_sp, u), np.float32)

    if state_mode == "standard":
        return base_state, {
            "innovation": None,
            "tracking_error": None,
            "innovation_scale_ref": None,
            "tracking_scale_now": None,
            "rho_feature": None,
        }

    if state_mode != "mismatch":
        raise ValueError("state_mode must be 'standard' or 'mismatch'.")

    if y_prev_scaled is None or yhat_pred is None:
        raise ValueError("mismatch state mode requires y_prev_scaled and yhat_pred.")
    if innovation_scale_ref is None:
        raise ValueError("mismatch state mode requires innovation_scale_ref.")
    if tracking_scale_now is None:
        raise ValueError("mismatch state mode requires tracking_scale_now.")

    y_prev_scaled = np.asarray(y_prev_scaled, float).reshape(-1)
    yhat_pred = np.asarray(yhat_pred, float).reshape(-1)
    y_sp = np.asarray(y_sp, float).reshape(-1)
    innovation_scale_ref = np.asarray(innovation_scale_ref, float).reshape(-1)
    tracking_scale_now = np.asarray(tracking_scale_now, float).reshape(-1)
    if innovation_scale_ref.shape[0] != y_sp.shape[0]:
        raise ValueError("innovation_scale_ref must match the number of outputs.")
    if tracking_scale_now.shape[0] != y_sp.shape[0]:
        raise ValueError("tracking_scale_now must match the number of outputs.")

    innovation = (y_prev_scaled - yhat_pred) / np.maximum(innovation_scale_ref, 1e-12)
    tracking_error = (y_prev_scaled - y_sp) / np.maximum(tracking_scale_now, 1e-12)
    if mismatch_clip is not None:
        mismatch_clip = float(mismatch_clip)
        innovation = np.clip(innovation, -mismatch_clip, mismatch_clip)
        tracking_error = np.clip(tracking_error, -mismatch_clip, mismatch_clip)
    innovation = innovation.astype(np.float32)
    tracking_error = tracking_error.astype(np.float32)

    state_parts = [base_state, innovation, tracking_error]
    rho_feature = None
    if append_rho_to_state:
        if rho_value is None:
            raise ValueError("append_rho_to_state=True requires rho_value.")
        rho_feature = np.asarray([float(rho_value)], np.float32)
        state_parts.append(rho_feature)

    state = np.concatenate(state_parts, axis=0).astype(np.float32)
    return state, {
        "innovation": innovation,
        "tracking_error": tracking_error,
        "innovation_scale_ref": innovation_scale_ref.astype(np.float32),
        "tracking_scale_now": tracking_scale_now.astype(np.float32),
        "rho_feature": rho_feature,
    }
