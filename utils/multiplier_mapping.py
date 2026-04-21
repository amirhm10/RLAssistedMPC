import numpy as np


def _as_float_array(value, name):
    arr = np.asarray(value, float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain infs or NaNs.")
    return arr


def _resolve_nominal(low, high, nominal):
    low = _as_float_array(low, "low")
    high = _as_float_array(high, "high")
    if low.shape != high.shape:
        raise ValueError("low and high must have the same shape.")
    if np.any(high <= low):
        raise ValueError("high must be strictly greater than low.")

    nominal_arr = np.asarray(nominal, float)
    if nominal_arr.ndim == 0:
        nominal_arr = np.full(low.shape, float(nominal_arr), dtype=float)
    else:
        nominal_arr = _as_float_array(nominal_arr, "nominal")
        if nominal_arr.shape != low.shape:
            raise ValueError("nominal must be scalar or have the same shape as low/high.")

    if np.any(nominal_arr <= low) or np.any(nominal_arr >= high):
        raise ValueError("nominal must lie strictly inside the low/high bounds.")
    return low, high, nominal_arr


def map_centered_action_to_bounds(action, low, high, nominal=1.0):
    low, high, nominal_arr = _resolve_nominal(low, high, nominal)
    action = _as_float_array(action, "action")
    if action.shape != low.shape:
        raise ValueError("action and bounds must have the same shape.")
    action = np.clip(action, -1.0, 1.0)
    lower_span = nominal_arr - low
    upper_span = high - nominal_arr
    return np.where(
        action <= 0.0,
        nominal_arr + action * lower_span,
        nominal_arr + action * upper_span,
    )


def map_centered_bounds_to_action(value, low, high, nominal=1.0):
    low, high, nominal_arr = _resolve_nominal(low, high, nominal)
    value = _as_float_array(value, "value")
    if value.shape != low.shape:
        raise ValueError("value and bounds must have the same shape.")
    value = np.clip(value, low, high)
    lower_span = nominal_arr - low
    upper_span = high - nominal_arr
    return np.where(
        value <= nominal_arr,
        (value - nominal_arr) / np.maximum(lower_span, 1e-12),
        (value - nominal_arr) / np.maximum(upper_span, 1e-12),
    )


__all__ = [
    "map_centered_action_to_bounds",
    "map_centered_bounds_to_action",
]
