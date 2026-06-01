from __future__ import annotations

import numpy as np


def validate_action_vector(action, action_dim, name, require_finite=True) -> np.ndarray:
    """Return a strict 1-D float32 action vector."""
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = int(action_dim)
    if arr.size != expected:
        raise ValueError(f"{name} has size {arr.size}, expected {expected}.")
    if bool(require_finite) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_float_scalar(value, name):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar-like, got shape {np.asarray(value).shape}.")
    return float(arr[0])


def compute_conservative_score(
    q1,
    q2,
    action,
    supervisor_action=None,
    previous_action=None,
    uncertainty_weight=0.0,
    supervisor_weight=0.0,
    previous_weight=0.0,
) -> float:
    q1_value = _as_float_scalar(q1, "q1")
    q2_value = _as_float_scalar(q2, "q2")
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_arr.size <= 0:
        raise ValueError("action must be a nonempty vector.")

    score = min(q1_value, q2_value)
    score -= float(max(0.0, uncertainty_weight)) * abs(q1_value - q2_value)

    if supervisor_action is not None and supervisor_weight != 0.0:
        supervisor_arr = np.asarray(supervisor_action, dtype=np.float64).reshape(-1)
        if supervisor_arr.size != action_arr.size:
            raise ValueError(
                f"supervisor_action has size {supervisor_arr.size}, expected {action_arr.size}."
            )
        score -= float(max(0.0, supervisor_weight)) * float(np.sum((action_arr - supervisor_arr) ** 2))

    if previous_action is not None and previous_weight != 0.0:
        previous_arr = np.asarray(previous_action, dtype=np.float64).reshape(-1)
        if previous_arr.size != action_arr.size:
            raise ValueError(f"previous_action has size {previous_arr.size}, expected {action_arr.size}.")
        score -= float(max(0.0, previous_weight)) * float(np.sum((action_arr - previous_arr) ** 2))

    return float(score)


def choose_policy_or_supervisor(
    score_policy,
    score_supervisor,
    margin=0.0,
    default_to_supervisor=True,
) -> str:
    policy_score = float(score_policy)
    supervisor_score = float(score_supervisor)
    if not np.isfinite(policy_score):
        return "supervisor"
    if not np.isfinite(supervisor_score):
        return "policy"

    threshold = supervisor_score + float(margin)
    if bool(default_to_supervisor):
        return "policy" if policy_score > threshold else "supervisor"
    return "policy" if policy_score >= threshold else "supervisor"
