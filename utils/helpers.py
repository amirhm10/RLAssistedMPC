import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

from Simulation.mpc import augment_state_space


def resolve_repo_root(start=None):
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / "utils").exists() and (candidate / "Simulation").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the provided start path.")


# -------------
#  Discrete action space: prediction and control horizons
# -------------
def build_horizon_recipes(predict_grid: List[int], control_grid: List[int]) -> List[tuple]:
    """
    Returns a list of (Hp, Hc) with Hc <= Hp
    Order defines action index (0 ... n-1)
    """
    recipes = []
    for Hp in predict_grid:
        for Hc in control_grid:
            if Hc <= Hp:
                recipes.append((Hp, Hc))
    if not recipes:
        raise ValueError("No valid Horizon recipes found")
    return recipes

def action_to_horizons(horizon_recipes: List[tuple], a_idx: int):
    return horizon_recipes[a_idx]

# ------------
# utilities
# ------------
def apply_min_max(data, min_val, max_val):
    """
    Applies min and max values to data.
    :param data:
    :param min_val:
    :param max_val:
    :return: scaled data
    """
    data = (data - min_val) / (max_val - min_val)
    return data


def reverse_min_max(scaled_data, min_val, max_val):
    """
    Reverses min-max scaling to recover the original data.

    Parameters:
    -----------
    scaled_data : array-like
        Data that has been scaled to the [0, 1] range.
    min_val : float or array-like
        The minimum value(s) used in the original scaling.
    max_val : float or array-like
        The maximum value(s) used in the original scaling.

    Returns:
    --------
    original_data : array-like
        The data mapped back to its original scale.
    """
    original_data = scaled_data * (max_val - min_val) + min_val
    return original_data


def apply_rl_scaled(min_max_dict, x_d_states, y_sp, u):
    """
    This function will apply RL scaling for the neural networks
    :param min_max_dict:
    :param state:
    :return: rl scaled of the state
    """

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

    y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

    u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

    states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))

    return states

# ------------
# Load system data
# ------------
def resolve_data_dir(data_dir="Data"):
    path = Path(data_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_pickle_path(data_dir, preferred_name, fallbacks=()):
    data_dir = resolve_data_dir(data_dir)
    candidates = [preferred_name, *fallbacks]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of {candidates} in {data_dir}.")


def load_and_prepare_system_data(
    steady_states,
    setpoint_y,
    u_min,
    u_max,
    data_dir="Data",
    n_inputs=2,
    system_dict_filename="system_dict.pickle",
    scaling_factor_filename="scaling_factor.pickle",
    min_max_states_filename="min_max_states.pickle",
):
    """
    Loads system matrices, scaling factors, and min-max state info from files,
    augments the state space, and applies min-max scaling to the steady states
    and setpoint. Returns a dictionary with the processed data.

    Parameters:
        data_dir (str): Directory where the data files are stored. Defaults to 'Data'.
        steady_states (dict): Dictionary containing:
            - 'y_ss': steady-state outputs.
            - 'ss_inputs': steady-state inputs.
            This is required.
        u_min, u_max (float): Min-max inputs respectively.
        setpoint_y (np.ndarray): 2D array for the output setpoint.
        n_inputs (int): Number of input channels. Defaults to 2.

    Returns:
        dict: A dictionary containing:
            - 'A', 'B', 'C': original system matrices.
            - 'A_aug', 'B_aug', 'C_aug': augmented system matrices.
            - 'data_min', 'data_max': scaling factor arrays.
            - 'min_max_states': dictionary loaded from the min-max states file.
            - 'y_ss_scaled': scaled steady-state outputs.
            - 'y_sp_scaled': scaled setpoint outputs.
            - 'y_sp_scaled_deviation': deviation of setpoint from steady-state.
            - 'u_ss_scaled': scaled steady-state inputs.
            - 'b_min', 'b_max': scaled control bounds (in deviation form).
            - 'min_max_dict': a dictionary combining state bounds and setpoint/input bounds.
    """

    # Load the system matrices dictionary (A, B, C)
    system_dict_path = _resolve_pickle_path(data_dir, system_dict_filename, fallbacks=("system_dict",))
    with open(system_dict_path, 'rb') as file:
        system_dict = pickle.load(file)

    A = system_dict['A']
    B = system_dict['B']
    C = system_dict['C']

    # Augment the state space
    A_aug, B_aug, C_aug = augment_state_space(A, B, C)

    # Load scaling factors (min and max)
    scaling_factor_path = _resolve_pickle_path(data_dir, scaling_factor_filename)
    with open(scaling_factor_path, 'rb') as file:
        scaling_factor = pickle.load(file)
    data_min = scaling_factor["min"]
    data_max = scaling_factor["max"]

    # Load the min-max states dictionary
    min_max_states_path = _resolve_pickle_path(data_dir, min_max_states_filename)
    with open(min_max_states_path, 'rb') as file:
        min_max_states = pickle.load(file)

    # Scale the steady-state outputs and setpoint outputs (using apply_min_max)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled = apply_min_max(setpoint_y, data_min[n_inputs:], data_max[n_inputs:])

    # Compute the deviation (setpoint - steady-state)
    y_sp_scaled_deviation = y_sp_scaled - y_ss_scaled

    # Scale the steady-state inputs
    u_ss_scaled = apply_min_max(steady_states['ss_inputs'], data_min[:n_inputs], data_max[:n_inputs])

    # Apply scaling to the bounds and subtract the steady-state inputs to get deviations
    b_min = apply_min_max(u_min, data_min[:n_inputs], data_max[:n_inputs]) - u_ss_scaled
    b_max = apply_min_max(u_max, data_min[:n_inputs], data_max[:n_inputs]) - u_ss_scaled

    # Create a dictionary combining the scaled state and control bounds
    min_max_dict = {
        "x_max": min_max_states["max_s"],
        "x_min": min_max_states["min_s"],
        "y_sp_min": y_sp_scaled_deviation[0],
        "y_sp_max": y_sp_scaled_deviation[1],
        "u_max": b_max,
        "u_min": b_min
    }

    return {
        "A": A,
        "B": B,
        "C": C,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "data_min": data_min,
        "data_max": data_max,
        "min_max_states": min_max_states,
        "y_ss_scaled": y_ss_scaled,
        "y_sp_scaled": y_sp_scaled,
        "y_sp_scaled_deviation": y_sp_scaled_deviation,
        "u_ss_scaled": u_ss_scaled,
        "b_min": b_min,
        "b_max": b_max,
        "min_max_dict": min_max_dict
    }


def build_polymer_disturbance_schedule(qi, qs, ha):
    return {
        "qi": np.asarray(qi, float),
        "qs": np.asarray(qs, float),
        "ha": np.asarray(ha, float),
    }


def disturbance_profile_from_schedule(disturbance_schedule, disturbance_labels=None):
    if disturbance_schedule is None:
        return None
    if isinstance(disturbance_schedule, dict):
        return {key: np.asarray(value, float) for key, value in disturbance_schedule.items()}

    disturbance_labels = list(disturbance_labels or [])
    series = np.asarray(disturbance_schedule, float)
    if series.ndim == 1:
        label = disturbance_labels[0] if disturbance_labels else "d1"
        return {label: series}
    if series.ndim != 2:
        raise ValueError("disturbance_schedule must be a 1D/2D array or a dictionary of named series.")
    return {
        disturbance_labels[idx] if idx < len(disturbance_labels) else f"d{idx + 1}": series[:, idx]
        for idx in range(series.shape[1])
    }


def disturbance_step_at(disturbance_schedule, idx):
    if disturbance_schedule is None:
        return None
    if isinstance(disturbance_schedule, dict):
        step = {}
        for key, value in disturbance_schedule.items():
            arr = np.asarray(value, float)
            step[key] = float(arr[idx]) if arr.ndim > 0 else float(arr)
        return step
    arr = np.asarray(disturbance_schedule, float)
    if arr.ndim == 1:
        return np.array([arr[idx]], dtype=float)
    if arr.ndim == 2:
        return np.asarray(arr[idx], dtype=float)
    raise ValueError("disturbance_schedule must be a 1D/2D array or a dictionary.")


def step_system_with_disturbance(system, idx=None, disturbance_schedule=None, system_stepper=None):
    disturbance_step = disturbance_step_at(disturbance_schedule, idx) if idx is not None else None

    if system_stepper is not None:
        system_stepper(system, disturbance_step)
        return disturbance_step

    if disturbance_step is None:
        system.step()
        return None

    if isinstance(disturbance_step, dict):
        system.step()
        for key, value in disturbance_step.items():
            setattr(system, key, float(value))
        return disturbance_step

    system.step(disturbances=np.atleast_1d(np.asarray(disturbance_step, float)))
    return disturbance_step


def generate_setpoints_training_rl_gradually(y_sp_scenario, n_tests, set_points_len, warm_start, test_cycle,
                                             nominal_qi, nominal_qs, nominal_ha,
                                             qi_change, qs_change, ha_change):
    # For each scenario, create a block of size (set_points_len, n_outputs)
    blocks = [np.full((set_points_len, y_sp_scenario.shape[1]), scenario)
              for scenario in y_sp_scenario]

    # Concatenate the blocks to form one cycle
    cycle = np.concatenate(blocks, axis=0)
    # Repeat the cycle 'repetitions' times
    y_sp = np.concatenate([cycle] * n_tests, axis=0)

    # Test train scenario
    test_cycle = test_cycle * int(n_tests / len(test_cycle))
    # Try making everything trainable but te end cycle should be only for testing
    test_cycle[-1] = True

    time_in_sub_episodes = set_points_len * len(y_sp_scenario)

    nFE = int(y_sp.shape[0])
    idxs_setpoints = np.arange(time_in_sub_episodes - 1, nFE, time_in_sub_episodes)
    idxs_tests = np.arange(0, nFE, time_in_sub_episodes)
    sub_episodes_changes = np.arange(1, len(idxs_setpoints) + 1)
    sub_episodes_changes_dict = {}
    test_train_dict = {}
    for i in range(len(idxs_setpoints)):
        sub_episodes_changes_dict[idxs_setpoints[i]] = sub_episodes_changes[i]
    for i in range(len(idxs_tests)):
        test_train_dict[idxs_tests[i]] = test_cycle[i]
    warm_start = list(test_train_dict.keys())[warm_start]

    qi = np.linspace(nominal_qi, nominal_qi * qi_change, nFE)
    qs = np.linspace(nominal_qs, nominal_qs * qs_change, nFE)
    ha = np.linspace(nominal_ha, nominal_ha * ha_change, int(nFE / 2))
    ha = np.hstack((ha, np.tile(nominal_ha * ha_change, int(nFE/ 2))))

    return y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, warm_start, qi, qs, ha
