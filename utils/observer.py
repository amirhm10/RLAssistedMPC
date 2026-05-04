import warnings

import control
import numpy as np
from scipy import signal


def compute_observer_gain(A, C, desired_poles):
    """
    Compute the observer gain for the augmented model.

    This mirrors the notebook implementation while suppressing the pole
    placement warnings that currently clutter notebook output.
    """

    A = np.asarray(A, float)
    C = np.asarray(C, float)
    desired_poles = np.asarray(desired_poles, float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs_gain_calc = signal.place_poles(A.T, C.T, desired_poles, method="KNV0")

    L = np.squeeze(obs_gain_calc.gain_matrix).T

    observability_matrix = control.obsv(A, C)
    rank = np.linalg.matrix_rank(observability_matrix)
    if rank != A.shape[0]:
        warnings.warn("Augmented system is not fully observable for the requested poles.")

    return L


def maybe_refresh_observer_model(
    *,
    enabled,
    A_candidate,
    B_candidate,
    A_current,
    B_current,
    L_current,
    C,
    poles,
    rtol=1e-9,
    atol=1e-12,
):
    """
    Refresh the observer model/gain only when the executed model changes.

    When refresh is disabled, or when the candidate model matches the current
    observer model, the current observer state-space pair and gain are kept.
    If pole placement fails for the new executed model, the previous observer
    model is retained so the rollout can continue safely.
    """

    A_candidate = np.asarray(A_candidate, float)
    B_candidate = np.asarray(B_candidate, float)
    A_current = np.asarray(A_current, float)
    B_current = np.asarray(B_current, float)
    L_current = np.asarray(L_current, float)
    C = np.asarray(C, float)
    poles = np.asarray(poles, float)

    changed = not (
        A_candidate.shape == A_current.shape
        and B_candidate.shape == B_current.shape
        and np.allclose(A_candidate, A_current, rtol=rtol, atol=atol)
        and np.allclose(B_candidate, B_current, rtol=rtol, atol=atol)
    )
    if (not enabled) or (not changed):
        return {
            "A": A_current,
            "B": B_current,
            "L": L_current,
            "event": bool(enabled and changed),
            "success": False,
            "reason": "disabled" if not enabled else "unchanged",
        }

    if not (np.all(np.isfinite(A_candidate)) and np.all(np.isfinite(B_candidate))):
        return {
            "A": A_current,
            "B": B_current,
            "L": L_current,
            "event": True,
            "success": False,
            "reason": "nonfinite_model",
        }

    try:
        L_candidate = compute_observer_gain(A_candidate, C, poles)
    except Exception as exc:
        return {
            "A": A_current,
            "B": B_current,
            "L": L_current,
            "event": True,
            "success": False,
            "reason": f"gain_failure:{exc}",
        }

    L_candidate = np.asarray(L_candidate, float)
    if not np.all(np.isfinite(L_candidate)):
        return {
            "A": A_current,
            "B": B_current,
            "L": L_current,
            "event": True,
            "success": False,
            "reason": "nonfinite_gain",
        }

    return {
        "A": A_candidate,
        "B": B_candidate,
        "L": L_candidate,
        "event": True,
        "success": True,
        "reason": "updated",
    }
