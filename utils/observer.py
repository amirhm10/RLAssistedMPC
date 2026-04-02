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
