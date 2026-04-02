import numpy as np

from utils.helpers import apply_rl_scaled


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
    clip_value=2.0,
):
    state_mode = str(state_mode).lower()
    base_state = np.asarray(apply_rl_scaled(min_max_dict, x_d_states, y_sp, u), np.float32)

    if state_mode == "standard":
        return base_state, {"innovation": None, "tracking_error": None}

    if state_mode != "mismatch":
        raise ValueError("state_mode must be 'standard' or 'mismatch'.")

    if y_prev_scaled is None or yhat_pred is None:
        raise ValueError("mismatch state mode requires y_prev_scaled and yhat_pred.")

    y_prev_scaled = np.asarray(y_prev_scaled, float)
    yhat_pred = np.asarray(yhat_pred, float)
    y_sp = np.asarray(y_sp, float)

    innovation = np.clip(y_prev_scaled - yhat_pred, -clip_value, clip_value).astype(np.float32)
    tracking_error = np.clip(y_prev_scaled - y_sp, -clip_value, clip_value).astype(np.float32)

    state = np.concatenate([base_state, innovation, tracking_error], axis=0).astype(np.float32)
    return state, {"innovation": innovation, "tracking_error": tracking_error}
