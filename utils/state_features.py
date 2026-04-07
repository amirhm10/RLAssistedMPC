import numpy as np

from utils.helpers import apply_rl_scaled


def default_mismatch_scale(min_max_dict, floor=1e-3):
    y_sp_min = np.asarray(min_max_dict["y_sp_min"], float)
    y_sp_max = np.asarray(min_max_dict["y_sp_max"], float)
    scale = np.maximum(np.abs(y_sp_min), np.abs(y_sp_max))
    return np.maximum(scale, float(floor)).astype(np.float32)


def get_rl_state_dim(base_aug_dim, n_outputs, n_inputs, state_mode):
    state_mode = str(state_mode).lower()
    if state_mode == "standard":
        return int(base_aug_dim) + int(n_outputs) + int(n_inputs)
    if state_mode == "mismatch":
        return int(base_aug_dim) + 3 * int(n_outputs) + int(n_inputs)
    raise ValueError("state_mode must be 'standard' or 'mismatch'.")


def build_rl_state(
    min_max_dict,
    x_d_states,
    y_sp,
    u,
    state_mode,
    y_prev_scaled=None,
    yhat_pred=None,
    mismatch_scale=None,
    mismatch_clip=3.0,
):
    state_mode = str(state_mode).lower()
    base_state = np.asarray(apply_rl_scaled(min_max_dict, x_d_states, y_sp, u), np.float32)

    if state_mode == "standard":
        return base_state, {"innovation": None, "tracking_error": None}

    if state_mode != "mismatch":
        raise ValueError("state_mode must be 'standard' or 'mismatch'.")

    if y_prev_scaled is None or yhat_pred is None:
        raise ValueError("mismatch state mode requires y_prev_scaled and yhat_pred.")
    if mismatch_scale is None:
        raise ValueError("mismatch state mode requires mismatch_scale.")

    y_prev_scaled = np.asarray(y_prev_scaled, float)
    yhat_pred = np.asarray(yhat_pred, float)
    y_sp = np.asarray(y_sp, float)
    mismatch_scale = np.asarray(mismatch_scale, float)
    if mismatch_scale.ndim != 1 or mismatch_scale.shape[0] != y_sp.shape[0]:
        raise ValueError("mismatch_scale must be a 1D vector matching the number of outputs.")

    innovation = (y_prev_scaled - yhat_pred) / mismatch_scale
    tracking_error = (y_prev_scaled - y_sp) / mismatch_scale
    if mismatch_clip is not None:
        mismatch_clip = float(mismatch_clip)
        innovation = np.clip(innovation, -mismatch_clip, mismatch_clip)
        tracking_error = np.clip(tracking_error, -mismatch_clip, mismatch_clip)
    innovation = innovation.astype(np.float32)
    tracking_error = tracking_error.astype(np.float32)

    state = np.concatenate([base_state, innovation, tracking_error], axis=0).astype(np.float32)
    return state, {"innovation": innovation, "tracking_error": tracking_error}
