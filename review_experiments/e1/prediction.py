"""Held-out future-output diagnostics using decision-time linear predictors."""
from __future__ import annotations

import numpy as np

from utils.helpers import apply_min_max, reverse_min_max
from utils.markov_runner import (
    apply_markov_correction, build_toeplitz_from_markov, free_response, raw_action_to_z,
)


def prediction_probe(snapshot, decision, trace, context):
    """Compare nominal, accepted-LS/zero and RL-proposed forced responses.

    All three predictors use the SAME initial observer state and observed future
    input sequence within one branch. Coefficients come only from the sampled
    decision; future outputs are used only to calculate errors. These are
    retrospective open-loop forecasts with known inputs, not MPC planned-input
    forecasts or a claim that RL changes the nominal free response/observer.
    """
    if context == "combined":
        A, C = snapshot["A_base"], snapshot["C_aug"]
        initial = snapshot["xhatdhat"][:, int(snapshot["i"])].copy()
        minimum, maximum = snapshot["data_min"], snapshot["data_max"]
        steady_input, steady_output = snapshot["ss_scaled_inputs"], snapshot["y_ss_scaled"]
        bound = snapshot["markov_z_bound"]
    else:
        A, C, initial = snapshot["A"], snapshot["C"], snapshot["x_model"].copy()
        cfg = snapshot["ctx"]
        minimum, maximum = cfg["data_min"], cfg["data_max"]
        steady_input, steady_output = cfg["ss_scaled_inputs"], cfg["y_ss_scaled"]
        bound = snapshot["config"]["z_bound"]
    blocks, bases = snapshot["m_blocks"], snapshot["basis_blocks"]
    length = min(8, len(blocks), len(trace["u"]))
    if length < 1:
        raise ValueError("No future control/output sample available for prediction validation.")
    inputs = trace["u"].shape[1]
    deviations = apply_min_max(trace["u"][:length], minimum[:inputs], maximum[:inputs]) - steady_input
    free = free_response(A, C, initial, length)
    correction = {
        "nominal": np.zeros(np.asarray(bases).shape[0]),
        "supervisor_ls_or_zero": raw_action_to_z(decision.supervisor_action, bound),
        "rl_proposed": raw_action_to_z(decision.policy_action, bound),
    }
    forecasts, errors = {}, {}
    for name, parameters in correction.items():
        corrected = apply_markov_correction(blocks, bases, parameters)[:length]
        dynamic = build_toeplitz_from_markov(corrected, length, length, hold_last=False)
        scaled = (free + dynamic @ deviations.reshape(-1)).reshape(length, C.shape[0])
        physical = reverse_min_max(scaled + steady_output, minimum[inputs:], maximum[inputs:])
        forecasts[name] = physical
        difference = physical - trace["y"][:length]
        errors[name] = {str(horizon): {
            "rmse_physical": np.sqrt(np.mean(difference[:horizon] ** 2, axis=0)),
            "max_abs_error_physical": np.max(np.abs(difference[:horizon]), axis=0),
        } for horizon in (1, 4, 8) if horizon <= length}
    return {"horizon": length, "metrics": errors,
            "scope": "decision-time predictor; common observed future inputs; held-out future outputs; shared nominal free response"}, forecasts
