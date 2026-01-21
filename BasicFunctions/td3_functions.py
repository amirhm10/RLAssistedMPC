import scipy.optimize as spo
import torch
import numpy as np
import pickle
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import control as ct

import os
import pickle
from Simulation.mpc import augment_state_space, apply_min_max


def load_and_prepare_system_data(steady_states, setpoint_y, u_min, u_max, data_dir='Data', n_inputs=2):
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

    # Ensure the full data directory exists
    full_data_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.exists(full_data_dir):
        os.makedirs(full_data_dir)

    # Load the system matrices dictionary (A, B, C)
    system_dict_path = os.path.join(full_data_dir, "system_dict")
    with open(system_dict_path, 'rb') as file:
        system_dict = pickle.load(file)

    A = system_dict['A']
    B = system_dict['B']
    C = system_dict['C']

    # Augment the state space
    A_aug, B_aug, C_aug = augment_state_space(A, B, C)

    # Load scaling factors (min and max)
    scaling_factor_path = os.path.join(full_data_dir, "scaling_factor.pickle")
    with open(scaling_factor_path, 'rb') as file:
        scaling_factor = pickle.load(file)
    data_min = scaling_factor["min"]
    data_max = scaling_factor["max"]

    # Load the min-max states dictionary
    min_max_states_path = os.path.join(full_data_dir, "min_max_states.pickle")
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


def print_accuracy(replay_buffer, agent, n_samples=1000, device="cpu"):
    entire_states, entire_actions = replay_buffer.complete_states_and_actions(n_samples, device)
    entire_accuracy = r2_score(entire_actions.detach().cpu().numpy(),
                               agent.actor(entire_states).detach().cpu().numpy())
    accuracy_input1 = r2_score(entire_actions.detach().cpu().numpy()[:, 0],
                               agent.actor(entire_states).detach().cpu().numpy()[:, 0])
    accuracy_input2 = r2_score(entire_actions.detach().cpu().numpy()[:, 1],
                               agent.actor(entire_states).detach().cpu().numpy()[:, 1])
    print(f"Agent r2 score for the predicted inputs compare to MPC inputs: {entire_accuracy:6f}")
    print(f"Agent r2 score for the predicted input 1 compare to MPC input 1: {accuracy_input1:6f}")
    print(f"Agent r2 score for the predicted input 1 compare to MPC input 2: {accuracy_input2:6f}")


def optimize_sample(i, MPC_obj, y_sp, u, x0_model, IC_opt, bnds, cons):
    sol = spo.minimize(
        lambda x: MPC_obj.mpc_opt_fun(x, y_sp, u, x0_model),
        IC_opt,
        bounds=bnds, constraints=cons
    )

    return sol.x[:MPC_obj.B.shape[1]]


def exponential_decay_bonus(percentage_error, A=1000, B=0.5):
    return A * np.exp(-B * percentage_error)

def apply_min_max_reward(data, min_val, max_val):
    """
    Applies min and max values to data.
    :param data:
    :param min_val:
    :param max_val:
    :return: scaled data
    """
    data = 2 * (data - min_val) / (max_val - min_val) - 1
    return data

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


def filling_the_buffer(
        min_max_dict,
        A, B, C,
        MPC_obj,
        mpc_pretrain_samples_numbers,
        Q_penalty, R_penalty,
        agent,
        IC_opt, bnds, cons, y_ss_scaled, data_min, data_max,
        chunk_size=10000):
    """
    Fill the replay buffer in batches to optimize the performance and manage memory

    Parameters:
        - min_max_dict: Dictionary containing min and max values for states, actions, and setpoints.
        - A, B, C: System matrices.
        - MPC_obj: Instance of MpcSolver.
        - mpc_pretrain_samples_numbers: Total number of pretraining samples to generate.
        - Q_penalty, R_penalty: Penalty matrices for the objective function.
        - agent: Agent instance containing the replay buffer.
        - u_ss: Steady-state input.
        - IC_opt: Initial guess for the optimizer.
        - bnds: Bounds for the optimizer.
        - cons: Constraints for the optimizer.
        - chunk_size: Number of samples to process in each chunk.
    """

    y_ss_scaled = apply_min_max(y_ss_scaled, data_min[2:], data_max[2:])

    num_full_chunks = mpc_pretrain_samples_numbers // chunk_size
    remaining_samples = mpc_pretrain_samples_numbers % chunk_size

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    # mu = 0
    # sigma = (x_max - x_min)/10.0

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    for chunk in range(num_full_chunks):
        print(f"Processing chunk {chunk + 1}/{num_full_chunks}")

        # x_d_states = np.random.normal(
        #     mu, sigma, size=(chunk_size, A.shape[0])
        # )

        x_d_states = np.random.uniform(
            low=x_min,
            high=x_max,
            size=(chunk_size, A.shape[0])
        )

        x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

        y_sp = np.random.uniform(
            low=y_sp_min,
            high=y_sp_max,
            size=(chunk_size, C.shape[0])
        )

        y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

        # u is in deviation form because u_min and u_max is in deviation form
        u = np.random.uniform(
            low=u_min,
            high=u_max,
            size=(chunk_size, B.shape[1])
        )

        u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

        # Perform parallel optimization for the current chunk
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(optimize_sample)(
                i + chunk * chunk_size,
                MPC_obj,
                y_sp[i, :],
                u[i, :],
                x_d_states[i, :],
                IC_opt,
                bnds,
                cons
            )
            for i in range(chunk_size)
        )

        u_mpc = np.array(results)

        next_x_d_states = np.dot(A, x_d_states.T) + np.dot(B, u_mpc.T)
        y_pred = np.dot(C, next_x_d_states)

        next_x_d_states_scaled = 2 * ((next_x_d_states.T - x_min) / (x_max - x_min)) - 1
        u_mpc_scaled = 2 * ((u_mpc - u_min) / (u_max - u_min)) - 1

        rewards = np.zeros(chunk_size)
        for k in range(chunk_size):
            rewards[k] = (-1.0 * (
                    (y_pred[:, k] - y_sp[k, :]).T @ Q_penalty @ (y_pred[:, k] - y_sp[k, :]) +
                    (u[k, :] - u_mpc[k, :]).T @ R_penalty @ (u[k, :] - u_mpc[k, :])
            ))


        actions = u_mpc_scaled.copy()

        states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
        next_states = np.hstack((next_x_d_states_scaled, y_sp_scaled, u_mpc_scaled))

        agent.replay_buffer.pretrain_add(states, actions, rewards, next_states)

    print("Replay buffer has been filled with generated samples.")


def add_steady_state_samples(
        min_max_dict,
        A, B, C,
        MPC_obj,
        steady_state_samples_numbers,
        Q_penalty, R_penalty,
        agent,
        IC_opt, bnds, cons, y_ss_scaled, data_min, data_max,
        chunk_size=10000):
    """
    Fill the replay buffer in batches to optimize the performance and manage memory

    Parameters:
        - min_max_dict: Dictionary containing min and max values for states, actions, and setpoints.
        - A, B, C: System matrices.
        - MPC_obj: Instance of MpcSolver.
        - mpc_pretrain_samples_numbers: Total number of pretraining samples to generate.
        - Q_penalty, R_penalty: Penalty matrices for the objective function.
        - agent: Agent instance containing the replay buffer.
        - u_ss: Steady-state input.
        - IC_opt: Initial guess for the optimizer.
        - bnds: Bounds for the optimizer.
        - cons: Constraints for the optimizer.
        - chunk_size: Number of samples to process in each chunk.
    """

    y_ss_scaled = apply_min_max(y_ss_scaled, data_min[2:], data_max[2:])

    num_full_chunks = steady_state_samples_numbers // chunk_size

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    mu = 0
    sigma = (x_max - x_min) / 10.0e12

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    for chunk in range(num_full_chunks):
        print(f"Processing chunk {chunk + 1}/{num_full_chunks}")

        x_d_states = np.random.normal(
            mu, sigma, size=(chunk_size, A.shape[0])
        )

        x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

        y_sp = np.random.uniform(
            low=0,
            high=0,
            size=(chunk_size, C.shape[0])
        )

        y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

        # u is in deviation form because u_min and u_max is in deviation form
        u = np.random.uniform(
            low=0,
            high=1e-08,
            size=(chunk_size, B.shape[1])
        )

        u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

        # Perform parallel optimization for the current chunk
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(optimize_sample)(
                i + chunk * chunk_size,
                MPC_obj,
                y_sp[i, :],
                u[i, :],
                x_d_states[i, :],
                IC_opt,
                bnds,
                cons
            )
            for i in range(chunk_size)
        )

        u_mpc = np.array(results)

        next_x_d_states = np.dot(A, x_d_states.T) + np.dot(B, u_mpc.T)
        y_pred = np.dot(C, next_x_d_states)

        next_x_d_states_scaled = 2 * ((next_x_d_states.T - x_min) / (x_max - x_min)) - 1
        u_mpc_scaled = 2 * ((u_mpc - u_min) / (u_max - u_min)) - 1

        rewards = np.zeros(chunk_size)
        for k in range(chunk_size):
            rewards[k] = (-1.0 * (
                    (y_pred[:, k] - y_sp[k, :]).T @ Q_penalty @ (y_pred[:, k] - y_sp[k, :]) +
                    (u[k, :] - u_mpc[k, :]).T @ R_penalty @ (u[k, :] - u_mpc[k, :])
            ))

        actions = u_mpc_scaled.copy()

        states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
        next_states = np.hstack((next_x_d_states_scaled, y_sp_scaled, u_mpc_scaled))

        agent.replay_buffer.pretrain_add(states, actions, rewards, next_states)

    print("Replay buffer has been filled up with the steady_state values.")
