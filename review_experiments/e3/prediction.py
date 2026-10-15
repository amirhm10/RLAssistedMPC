"""Known-input forecasts with frozen correction coefficients and nominal free response."""
from __future__ import annotations

import numpy as np

from utils.helpers import apply_min_max, reverse_min_max
from utils.markov_runner import (
    apply_markov_correction, build_toeplitz_from_markov, compute_markov_blocks,
    free_response, make_markov_basis,
)
from .common import PREDICTORS


def forecasts(model, x0, u_physical):
    """Use no output measurements; each row predicts y[k+h] from xhat[k].

    The actual future input sequence is shared by all predictors. This is a
    retrospective known-input model diagnostic, not a planned-input MPC forecast.
    """
    A, B, C = (np.asarray(model[key], float) for key in ("A", "B", "C"))
    inputs = np.asarray(u_physical, float)
    h, nu = inputs.shape
    if B.shape[1] != nu or h < 1:
        raise ValueError("Forecast inputs are incompatible with the captured predictor.")
    minimum, maximum = np.asarray(model["data_min"]), np.asarray(model["data_max"])
    if np.any(maximum <= minimum):
        raise ValueError("Identification scaling must have strictly positive ranges.")
    deviations = apply_min_max(inputs, minimum[:nu], maximum[:nu]) - model["ss_scaled_inputs"]
    blocks = compute_markov_blocks(A, B, C, h)
    basis, _ = make_markov_basis(blocks, str(model["basis_family"]))
    free = free_response(A, C, x0, h)
    result, failures = {}, {}
    for index, name in enumerate(PREDICTORS):
        z = np.asarray(model["corrections"])[index]
        try:
            if not np.all(np.isfinite(z)):
                raise FloatingPointError("Nonfinite captured correction proposal.")
            with np.errstate(over="raise", invalid="raise"):
                corrected = apply_markov_correction(blocks, basis, z)
                dynamic = build_toeplitz_from_markov(corrected, h, h, hold_last=False)
                scaled = (free + dynamic @ deviations.reshape(-1)).reshape(h, C.shape[0])
                physical = reverse_min_max(scaled + model["y_ss_scaled"], minimum[nu:], maximum[nu:])
                if not np.all(np.isfinite(physical)):
                    raise FloatingPointError("Nonfinite forecast.")
                result[name] = physical
        except (ValueError, FloatingPointError) as error:
            result[name] = np.full((h, C.shape[0]), np.nan)
            failures[name] = str(error)
    return result, failures


def observer_history(model, y_physical, u_physical):
    """Causal state initialization for independent nominal data; no correction refit."""
    A, B, C, L = (np.asarray(model[key], float) for key in ("A", "B", "C", "L"))
    nu = B.shape[1]
    minimum, maximum = np.asarray(model["data_min"]), np.asarray(model["data_max"])
    y = apply_min_max(y_physical, minimum[nu:], maximum[nu:]) - model["y_ss_scaled"]
    u = apply_min_max(u_physical, minimum[:nu], maximum[:nu]) - model["ss_scaled_inputs"]
    if len(y) != len(u) + 1:
        raise ValueError("Nominal output history must contain y[0] and one output after every input.")
    mode = str(model["observer_mode"])
    states = np.zeros((len(y), A.shape[0]))
    for k, move in enumerate(u):
        predicted = A @ states[k] + B @ move
        if mode in {"predictor_corrector_current", "current_measurement_corrector"}:
            states[k + 1] = predicted + L @ (y[k + 1] - C @ predicted)
        elif mode == "legacy_previous_measurement":
            states[k + 1] = predicted + L @ (y[k] - C @ states[k])
        else:
            raise ValueError(f"Unsupported captured observer convention: {mode}.")
        if bool(model.get("observer_float32", False)):
            states[k + 1] = states[k + 1].astype(np.float32)
    return states


def score_predictions(predicted, measured):
    """Lead-specific residuals are retained; aggregation happens across origins."""
    measured = np.asarray(measured, float)
    return {name: np.asarray(value, float) - measured for name, value in predicted.items()}
