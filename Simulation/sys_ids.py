import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from Simulation.system_functions import PolymerCSTR
# import matlab.engine
# import matlab
from BasicFunctions.bs_fns import apply_min_max, reverse_min_max


def generate_step_test_data(u_start, step_value,
                            initial_duration=40,
                            step_duration=200,
                            step_index=0):
    """
    Generate test data with a step change.

    Parameters:
    -----------
    u_start : array-like of shape (n_inputs,)
        The starting input values.
    step_value : float
        The value to add to u_start on the specified input.
    initial_duration : int, optional
        Number of time steps to hold the initial value (default is 40).
    step_duration : int, optional
        Number of time steps to hold the stepped value (default is 200).
    step_index : int, optional
        The index of the inputs to which the step is applied (default is 0).

    Returns:
    --------
    test_data : np.ndarray
        An array of shape ((initial_duration + step_duration), n_inputs)
        containing the test data with the step change applied.
    """
    # Create an array for the initial duration using the starting values
    initial_array = np.full((initial_duration, len(u_start)), u_start)

    # Copy the starting value and apply the step change to the specified channel
    stepped_input = np.array(u_start, copy=True)
    stepped_input[step_index] += step_value

    # Create an array for the stepped duration with the modified input
    step_array = np.full((step_duration, len(u_start)), stepped_input)

    # Concatenate the two arrays to form the complete test data
    test_data = np.concatenate((initial_array, step_array), axis=0)

    return test_data


def simulate_system(system, input_sequence):
    """
    Simulate the system with the given input sequence.

    Parameters:
    -----------
    system : object
        The system to be simulated. It must have a method `step()` and
        an attribute `current_input` that can be set.
    input_sequence : array-like
        A sequence (e.g. numpy array) of inputs to apply at each time step.

    Returns:
    --------
    results : dict
        A dictionary with keys:
          - 'inputs': all applied inputs (including the initial condition)
          - 'outputs': the outputs recorded after each step
   """

    # Initialize lists to store simulation data.
    # Record the initial output (and state, if available)
    outputs = [system.current_output]

    # Loop over each input step.
    for inp in input_sequence:
        system.current_input = inp
        system.step()  # Advance the simulation one time step.

        outputs.append(system.current_output)

    results = {
        'inputs': np.array(input_sequence),
        'outputs': np.array(outputs)
    }

    return results


def plot_results(time, outputs, inputs, output_labels=None, input_labels=None):
    """
    Plot system outputs and inputs.

    Parameters:
    -----------
    time : array-like
        Time points corresponding to the simulation data.
    outputs : numpy.ndarray
        Array of outputs with shape (n_points, n_outputs).
    inputs : numpy.ndarray
        Array of inputs with shape (n_points-1, n_inputs) (if the first output was recorded before applying any input).
    output_labels : list of str
        Labels for the output channels.
    input_labels : list of str
        Labels for the input channels.
    """
    if input_labels is None:
        input_labels = ['Input 1', 'Input 2']
    if output_labels is None:
        output_labels = ['Output 1', 'Output 2']
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

    # Plot outputs
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, outputs[:, 0], 'b-', lw=2, label=output_labels[0])
    plt.ylabel(output_labels[0])
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(time, outputs[:, 1], 'b-', lw=2, label=output_labels[1])
    plt.ylabel(output_labels[1])
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Plot inputs (using step plots)
    plt.figure(figsize=(10, 8))
    time_input = time[:-1]  # assuming one input per step (applied before each step)
    plt.subplot(2, 1, 1)
    plt.step(time_input, inputs[:, 0], 'k-', lw=2, label=input_labels[0])
    plt.ylabel(input_labels[0])
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.step(time_input, inputs[:, 1], 'k-', lw=2, label=input_labels[1])
    plt.ylabel(input_labels[1])
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def save_simulation_data(data, filename, column_names):
    """
    Save simulation data (e.g. concatenated inputs and outputs) to a CSV file.

    Parameters:
    -----------
    data : numpy.ndarray
        Data array to save.
    filename : str
        Path to the CSV file.
    column_names : list of str
        Column headers for the CSV.
    """
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(filename, index=False)


def run_cstr_experiment(
    step_value,
    step_channel,
    save_filename,
    system_params,
    system_design_params,
    system_steady_state_inputs,
    delta_t,
    data_dir=None,
):
    # Instantiate the reactor
    cstr = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)

    # Retrieve initial input from the reactor.
    u_start = cstr.current_input

    # Generate step test input data
    step_data = generate_step_test_data(u_start, step_value, step_index=step_channel)

    # Run the simulation
    results = simulate_system(cstr, step_data)

    # Create a time vector
    n_points = results['outputs'].shape[0]
    time = np.linspace(0, n_points * delta_t, n_points)

    # Save the combined data (here we concatenate the input and the outputs excluding the initial output)
    # Adjust the column names as appropriate.
    data_to_save = np.concatenate((results['inputs'], results['outputs'][1:]), axis=1)
    data_dir = data_dir or os.path.join(os.getcwd(), 'Data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    save_path = os.path.join(data_dir, save_filename)
    column_names = ["Qc", "Qm", "Etha", "T"]
    save_simulation_data(data_to_save, save_path, column_names)

    # Plot the results
    plot_results(time, results['outputs'], results['inputs'],
                 output_labels=["Etha", "T"],
                 input_labels=["Qc", "Qm"])

    return results


def scaling_min_max_factors(file_paths):
    """
    Reads CSV data files, applies the scaling transformation, and returns min max values
    :param file_paths:
    :return: min_max_factors
    """
    data_min = []
    data_max = []
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        # Find the maximum and minimum and append to the lists
        data_min.append(df.min())
        data_max.append(df.max())

    return np.min(data_min, axis=0), np.max(data_max, axis=0)


def apply_deviation_form_scaled(steady_states, file_paths, data_min, data_max):
    """
    Reads CSV data files, applies the deviation transformation, and returns
    the resulting DataFrames.

    Parameters
    ----------
    steady_states : dict
        A dictionary that provides steady state data.
        'ss_inputs' and 'y_ss' which are used to form the steady state vector.
    file_paths : dict
        A dictionary mapping a key (e.g., "Qc", "Qm") to a file path.

    Returns
    -------
    deviations : dict
        A dictionary mapping each key to its deviation-form DataFrame.
    """
    # Construct the full steady state vector
    u_ss = steady_states['ss_inputs']  # e.g., steady inputs
    y_ss = steady_states['y_ss']  # e.g., steady outputs
    ss = np.concatenate((u_ss, y_ss), axis=0)

    ss_scaled = apply_min_max(ss, data_min, data_max)

    deviations = {}
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        # Subtract the steady state vector from all columns
        deviations[key] = apply_min_max(df, data_min, data_max) - ss_scaled
    return deviations


def data_time28_63_dict(df, mode=0, sampling_period=0.5, interactive=True):
    """
    Processes a deviation-form DataFrame to extract transfer function parameters.
    The function assumes that the input column is at the position indicated by 'mode'
    and that the last two columns correspond to the system outputs.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame in deviation form (steady state subtracted).
    mode : int, optional
        The index of the input column on which the test was performed (default is 0).
    sampling_period : float, optional
        The sampling period (in hours) used to create the time vector (default is 0.5).
    interactive : bool, optional
        If True, the function will prompt the user for time values (default is True).

    Returns
    -------
    transfer_functions_dict : dict
        A dictionary where each key is an output channel and the value is another
        dictionary containing the transfer function parameters: Time 63, Time 28, kp,
        taup, and theta.
    data : pd.DataFrame
        The subset of the original data starting just before the step change.
    """
    # Determine the input column to use (for detecting the step change)
    input_col = df.columns[mode]
    constant_value = df[input_col].iloc[0]

    # Find the first row where the input deviates from the constant value.
    mask = df[input_col] != constant_value
    if mask.any():
        index_change = df.index[mask][0] - 1
    else:
        index_change = 0

    # Extract the data starting at (or just before) the step change
    data = df.iloc[index_change:].reset_index(drop=True)
    point_numbers = len(data)
    # Create a time vector
    time_plot = np.arange(point_numbers) * sampling_period

    transfer_functions_dict = {}

    # Process each output column (assumed to be the last two columns of the DataFrame)
    for output in df.columns[-2:]:
        while True:
            if interactive:
                user_input = input(
                    f"Enter 'time_28' and 'time_63' (in hours) for {output} separated by a comma (or type 'Done' to finish): "
                )
            else:
                raise ValueError("Non-interactive mode not implemented. Set interactive=True.")

            if user_input.lower() == 'done':
                break

            try:
                time_28, time_63 = map(float, user_input.split(','))
            except ValueError:
                print("Invalid input. Please enter two numeric values separated by a comma.")
                continue

            # Calculate the change in output and input over the test interval
            delta_y = data[output].iloc[-1] - data[output].iloc[0]
            delta_u = data[input_col].iloc[-1] - data[input_col].iloc[0]
            kp = delta_y / delta_u if delta_u != 0 else np.nan

            # Determine the 28% and 63% response levels
            y_28 = data[output].iloc[0] + 0.28 * delta_y
            y_63 = data[output].iloc[0] + 0.63 * delta_y

            # Plot the response along with horizontal lines at y_28 and y_63
            plt.figure(figsize=(8, 6))
            plt.plot(time_plot, data[output], label=f'Response of {output}')
            plt.hlines(y=y_28, xmin=time_plot[0], xmax=time_plot[-1],
                       colors="red", label="y_28 (28% response)")
            plt.hlines(y=y_63, xmin=time_plot[0], xmax=time_plot[-1],
                       colors="yellow", label="y_63 (63% response)")
            plt.scatter([time_28], [y_28], color="red", zorder=5)
            plt.scatter([time_63], [y_63], color="yellow", zorder=5)
            plt.xlabel("Time (hour)")
            plt.ylabel(output)
            plt.title(f"Time Response for {output}")
            plt.legend()
            plt.xlim([0, time_plot[-1]])
            plt.show()

            # Confirm with the user if the chosen times are correct.
            correct_input = input("Are the times correct? Type 'yes' to confirm, or 'no' to re-enter values: ")
            if correct_input.lower() == 'yes':
                # Optionally, convert days to hours:
                time_63_hours = time_63
                time_28_hours = time_28
                delta_t = time_63_hours - time_28_hours
                taup = 1.5 * delta_t
                theta = time_28_hours - delta_t  # adjusted formula for theta
                transfer_functions_dict[output] = {
                    "Time 63 (hrs)": time_63_hours,
                    "Time 28 (hrs)": time_28_hours,
                    "kp": kp,
                    "taup": taup,
                    "theta": theta
                }
                break

    print(f'\nTransfer Function details for input mode "{input_col}":')
    for key, value in transfer_functions_dict.items():
        print(f'{key}: {value}')

    return transfer_functions_dict, data


def state_space_form_using_matlab(u1_dict, u2_dict, delay_list, data_u1, data_u2, sampling_time = 0.5):
    # Start MATLAB engine
    input_name1, input_name2 = data_u1.columns[0], data_u1.columns[1]
    output_name1, output_name2 = data_u1.columns[2], data_u1.columns[3]
    delta_u1 = data_u1[input_name1].iloc[1] - data_u1[input_name1].iloc[0]
    delta_u2 = data_u2[input_name2].iloc[1] - data_u2[input_name2].iloc[0]
    delta_u = [delta_u1, delta_u2]
    end_time = (data_u1.shape[0]-1) * sampling_time
    eng = matlab.engine.start_matlab()
    # Create num, den, and delay variables
    num = (
        f'num = {{{u1_dict[output_name1]["kp"]}, {u2_dict[output_name1]["kp"]}; '
        f'{u1_dict[output_name2]["kp"]}, {u2_dict[output_name2]["kp"]}}};')
    den = (
        f'den = {{[{u1_dict[output_name1]["taup"]}, 1], [{u2_dict[output_name1]["taup"]}, 1]; '
        f'[{u1_dict[output_name2]["taup"]}, 1], [{u2_dict[output_name2]["taup"]}, 1]}};')
    delay = f'delay = [{delay_list[0]}, {delay_list[1]}; {delay_list[2]}, {delay_list[3]}];'

    eng.eval(num, nargout=0)
    eng.eval(den, nargout=0)
    eng.eval(delay, nargout=0)
    eng.eval("tf_system = tf(num, den, 'IODelay', delay, 'TimeUnit', 'hours');", nargout=0)

    # Convert the transfer function to a state-space model
    eng.eval("ss_system = ss(tf_system);", nargout=0)

    # Discretize the state-space model with the specified sampling time
    Ts = sampling_time  # Sampling time in hours
    eng.workspace['end_time'] = end_time
    eng.workspace['Ts'] = Ts
    eng.eval("mimo_ss_dis = c2d(ss_system, Ts);", nargout=0)

    # Set up options for the step response
    eng.eval("opt = stepDataOptions('InputOffset', 0, 'StepAmplitude', " + str(delta_u) + ");", nargout=0)

    # Define the time vector for simulation
    eng.eval("t = 0:Ts:end_time;", nargout=0)

    # Absorb the delay into the discretized state-space model
    eng.eval("mimo_ss_dis_ab_delay = absorbDelay(mimo_ss_dis);", nargout=0)

    # Optionally, simulate or visualize the response
    eng.eval("step(mimo_ss_dis_ab_delay, t, opt);", nargout=0)

    # Run step response and fetch results
    eng.eval("t = 0:Ts:end_time;", nargout=0)
    eng.eval("opt = stepDataOptions('InputOffset', 0, 'StepAmplitude', " + str(delta_u) + ");", nargout=0)
    eng.eval("mimo_ss_dis_ab_delay = absorbDelay(mimo_ss_dis);", nargout=0)
    eng.eval("[y_dis_model_ss_ab_delay, tOut] = step(mimo_ss_dis_ab_delay, t, opt);", nargout=0)

    # Retrieve data
    y_dis_model_ss_ab_delay = eng.workspace['y_dis_model_ss_ab_delay']
    tOut = eng.workspace['tOut']

    # Convert MATLAB matrix to numpy array
    y_dis_model_ss_ab_delay = np.array(y_dis_model_ss_ab_delay)
    tOut = np.array(tOut)

    A_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.A", nargout=1))
    B_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.B", nargout=1))
    C_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.C", nargout=1))
    D_matrix = np.array(eng.eval("mimo_ss_dis_ab_delay.D", nargout=1))

    return A_matrix, B_matrix, C_matrix, D_matrix, y_dis_model_ss_ab_delay, tOut


def plot_results_statespace(tOut, y_dis_model_ss_ab_delay, data_u1, data_u2):
    input_name1, input_name2 = data_u1.columns[0], data_u1.columns[1]
    output_name1, output_name2 = data_u1.columns[2], data_u1.columns[3]

    y1_actual_ref_dev = data_u1[output_name1]
    y2_actual_ref_dev = data_u1[output_name2]

    y1_actual_reb_dev = data_u2[output_name1]
    y2_actual_reb_dev = data_u2[output_name2]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a figure with a 2x2 grid of axes

    # Plot for Tray 24 Composition from Reflux
    axs[0, 0].plot(tOut, y1_actual_ref_dev, 'r-', linewidth=2, label="Actual system response")
    axs[0, 0].plot(tOut, y_dis_model_ss_ab_delay[:, :, 0][:, 0], 'b--', linewidth=2, label="Step response")
    axs[0, 0].set(xlabel='Time (hr)', ylabel=output_name1, title=f'Step in {input_name1}')
    axs[0, 0].tick_params(direction='in', length=6, width=1)

    # Plot for Tray 85 Temperature from Reflux
    axs[1, 0].plot(tOut, y2_actual_ref_dev, 'r-', linewidth=2, label="Actual system response")
    axs[1, 0].plot(tOut, y_dis_model_ss_ab_delay[:, :, 0][:, 1], 'b--', linewidth=2, label="Step response")
    axs[1, 0].set(xlabel='Time (hr)', ylabel=output_name2)

    # Plot for Tray 24 Composition from Reboiler
    axs[0, 1].plot(tOut, y1_actual_reb_dev, 'r-', linewidth=2, label="Actual system response")
    axs[0, 1].plot(tOut, y_dis_model_ss_ab_delay[:, :, 1][:, 0], 'b--', linewidth=2, label="Step response")
    axs[0, 1].set(xlabel='Time (hr)', ylabel=output_name1, title=f'Step in {input_name2}')

    # Plot for Tray 85 Temperature from Reboiler
    axs[1, 1].plot(tOut, y2_actual_reb_dev, 'r-', linewidth=2, label="Actual system response")
    axs[1, 1].plot(tOut, y_dis_model_ss_ab_delay[:, :, 1][:, 1], 'b--', linewidth=2, label="Step response")
    axs[1, 1].set(xlabel='Time (hr)', ylabel=output_name2)

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    plt.show()


def generate_step_test_sequence(u_start, step_delta, initial_duration=40, step_duration=200):
    """
    Generate a generic step-test input sequence from a steady input vector and
    an additive step vector.
    """

    u_start = np.asarray(u_start, dtype=float)
    step_delta = np.asarray(step_delta, dtype=float)
    if step_delta.shape != u_start.shape:
        raise ValueError("step_delta must have the same shape as u_start.")

    initial_array = np.full((int(initial_duration), len(u_start)), u_start, dtype=float)
    step_array = np.full((int(step_duration), len(u_start)), u_start + step_delta, dtype=float)
    return np.concatenate((initial_array, step_array), axis=0)


def extract_fopdt_2863_auto(df, input_idx=0, Ts=1.0, limit=None, pre_win=10, post_win=10, plot=True):
    """
    Estimate FOPDT channel parameters from one step-test dataframe using the
    28/63 percent heuristic without interactive prompts.

    The function supports mild inverse-response behavior by allowing the 28/63
    crossing search to start after an early extremum.
    """

    cols = list(df.columns)
    if len(cols) < 4:
        raise ValueError("Expected step-test data with at least 4 columns: 2 inputs and 2 outputs.")

    input_idx = int(input_idx)
    input_col = cols[input_idx]
    output_cols = cols[-2:]

    u = df[input_col].to_numpy(dtype=float)
    base = float(np.median(u[: max(int(pre_win), 5)]))
    du = np.abs(u - base)
    noise = float(np.median(np.abs(du[: max(int(pre_win), 20)] - np.median(du[: max(int(pre_win), 20)]))))
    threshold = max(1e-9, 5.0 * noise)
    k0 = int(np.argmax(du > threshold))
    k0 = max(k0 - 1, 0)

    data = df.iloc[k0:].reset_index(drop=True)
    t = np.arange(len(data), dtype=float) * float(Ts)
    results = {}

    denom = np.log(0.72 / 0.37)
    log072 = np.log(0.72)

    def crossing_time_after(y, target, start_idx):
        if start_idx >= len(y) - 1:
            return np.nan
        sign = np.sign(y - target)
        sign[:start_idx] = sign[start_idx]
        crossings = np.where(np.diff(sign) != 0)[0]
        if crossings.size == 0:
            return np.nan
        k = int(crossings[0])
        y0_local = float(y[k])
        y1_local = float(y[k + 1])
        if y1_local == y0_local:
            return float(t[k])
        frac = (float(target) - y0_local) / (y1_local - y0_local)
        return float(t[k] + float(Ts) * frac)

    for output_name in output_cols:
        y = data[output_name].to_numpy(dtype=float)
        u_seg = data[input_col].to_numpy(dtype=float)

        k_pre = min(int(pre_win), max(1, len(y) // 10))
        k_post = min(int(post_win), max(1, len(y) // 10))
        y0 = float(np.mean(y[:k_pre]))
        yF = float(np.mean(y[-k_post:]))
        u0 = float(np.mean(u_seg[:k_pre]))
        uF = float(np.mean(u_seg[-k_post:]))

        dY = yF - y0
        dU = uF - u0 if abs(uF - u0) > 0 else np.nan
        kp = dY / dU if np.isfinite(dU) else np.nan

        final_sign = np.sign(dY) if dY != 0 else 0.0
        search_end = max(int(0.4 * len(y)), k_pre + 5)
        if final_sign >= 0:
            k_ext = int(np.argmin(y[:search_end]))
        else:
            k_ext = int(np.argmax(y[:search_end]))

        inverse = False
        if final_sign != 0:
            early_move = np.sign(float(y[min(k_ext, len(y) - 1)]) - y0)
            inverse = (early_move != 0) and (early_move != final_sign)

        if inverse:
            y_ext = float(y[k_ext])
            M = abs((y_ext - y0) / dY) if dY != 0 else 0.0
            start_idx = k_ext
        else:
            M = 0.0
            start_idx = 0

        y28 = y0 + 0.28 * dY
        y63 = y0 + 0.63 * dY

        t28 = crossing_time_after(y, y28, start_idx)
        t63 = crossing_time_after(y, y63, start_idx)

        if (not np.isfinite(t28)) or (not np.isfinite(t63)) or (t63 <= t28):
            tau = np.nan
            theta = np.nan
            tz_est = np.nan
        else:
            tau = (t63 - t28) / denom
            theta = t28 + tau * log072
            tz_est = M * tau if inverse else 0.0

        results[output_name] = {
            "kp": float(kp) if np.isfinite(kp) else np.nan,
            "t28": float(t28) if np.isfinite(t28) else np.nan,
            "t63": float(t63) if np.isfinite(t63) else np.nan,
            "taup": float(tau) if np.isfinite(tau) else np.nan,
            "theta": float(max(0.0, theta)) if np.isfinite(theta) else np.nan,
            "inverse": bool(inverse),
            "t_ext": float(t[k_ext]) if inverse else None,
            "M": float(M),
            "Tz_est": float(tz_est) if np.isfinite(tz_est) else np.nan,
            "y0": y0,
            "yF": yF,
        }

        if plot:
            plt.figure(figsize=(7.5, 5.5))
            plt.plot(t, y, label=f"{output_name} response")
            plt.axhline(y28, linestyle="--", label="28% level")
            plt.axhline(y63, linestyle="--", label="63% level")
            if inverse:
                plt.axvline(t[k_ext], linestyle=":", label="inverse extremum")
            if np.isfinite(t28):
                plt.axvline(t28, linestyle="--", label="t28")
                plt.scatter([t28], [y28], zorder=5)
            if np.isfinite(t63):
                plt.axvline(t63, linestyle="--", label="t63")
                plt.scatter([t63], [y63], zorder=5)
            if limit is not None:
                plt.xlim([0.0, t[min(int(limit), len(t) - 1)]])
            plt.xlabel("Time (h)")
            plt.ylabel(output_name)
            plt.title(f"28/63 identification for {output_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return results, data


def _quantize_delay_steps(theta, Ts, quantization="round"):
    theta = max(0.0, float(theta))
    Ts = float(Ts)
    if Ts <= 0.0:
        raise ValueError("Ts must be positive.")
    samples = theta / Ts
    if quantization == "ceil":
        return int(np.ceil(samples))
    if quantization == "floor":
        return int(np.floor(samples))
    return int(np.rint(samples))


def build_discrete_fopdt_channel(kp, taup, delay_steps, Ts):
    """
    Build the exact discrete first-order part of one FOPDT channel and expose
    its scalar coefficients.
    """

    taup = float(max(float(taup), 1e-9))
    kp = float(kp)
    delay_steps = int(max(int(delay_steps), 0))
    Ts = float(Ts)
    if Ts <= 0.0:
        raise ValueError("Ts must be positive.")
    a = float(np.exp(-Ts / taup))
    return {
        "a": a,
        "b": float(kp * (1.0 - a)),
        "delay_steps": delay_steps,
        "taup": taup,
        "kp": kp,
    }


def build_delay_chain_matrices(delay_steps):
    """
    Return the delay-chain state matrix and input vector for a pure shift
    register with integer-sample delay.
    """

    delay_steps = int(max(int(delay_steps), 0))
    if delay_steps == 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0, 1), dtype=float)

    A_delay = np.zeros((delay_steps, delay_steps), dtype=float)
    if delay_steps > 1:
        A_delay[1:, :-1] = np.eye(delay_steps - 1)
    B_delay = np.zeros((delay_steps, 1), dtype=float)
    B_delay[0, 0] = 1.0
    return A_delay, B_delay


def build_mimo_state_space_from_fopdt_python(channel_fits, input_names, output_names, Ts, quantization="round"):
    """
    Build one global discrete-time MIMO realization by stacking one first-order
    dynamic state and an optional delay chain for each input-output channel.
    """

    input_names = list(input_names)
    output_names = list(output_names)
    Ts = float(Ts)
    if Ts <= 0.0:
        raise ValueError("Ts must be positive.")

    channel_specs = []
    total_states = 0
    n_inputs = len(input_names)
    n_outputs = len(output_names)

    for output_idx, output_name in enumerate(output_names):
        for input_idx, input_name in enumerate(input_names):
            fit = channel_fits[(input_name, output_name)]
            kp = float(fit["kp"])
            taup = max(float(fit["taup"]), 1e-9)
            theta = max(0.0, float(fit["theta"]))
            delay_steps = _quantize_delay_steps(theta, Ts, quantization=quantization)
            a = float(np.exp(-Ts / taup))
            b = float(kp * (1.0 - a))
            size = 1 + delay_steps
            channel_specs.append(
                {
                    "input_name": input_name,
                    "output_name": output_name,
                    "input_idx": input_idx,
                    "output_idx": output_idx,
                    "a": a,
                    "b": b,
                    "delay_steps": delay_steps,
                    "taup": taup,
                    "theta": theta,
                    "kp": kp,
                    "start": total_states,
                    "size": size,
                }
            )
            total_states += size

    A = np.zeros((total_states, total_states), dtype=float)
    B = np.zeros((total_states, n_inputs), dtype=float)
    C = np.zeros((n_outputs, total_states), dtype=float)
    D = np.zeros((n_outputs, n_inputs), dtype=float)
    delay_matrix = np.zeros((n_outputs, n_inputs), dtype=int)

    for spec in channel_specs:
        start = int(spec["start"])
        stop = start + int(spec["size"])
        delay_steps = int(spec["delay_steps"])
        state_slice = slice(start, stop)

        A_block = np.zeros((spec["size"], spec["size"]), dtype=float)
        A_block[0, 0] = spec["a"]
        if delay_steps == 0:
            B[start, spec["input_idx"]] = spec["b"]
        else:
            A_block[0, -1] = spec["b"]
            if delay_steps > 1:
                A_block[2:, 1:-1] = np.eye(delay_steps - 1)
            B[start + 1, spec["input_idx"]] = 1.0

        A[state_slice, state_slice] = A_block
        C[spec["output_idx"], start] = 1.0
        delay_matrix[spec["output_idx"], spec["input_idx"]] = delay_steps

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "delay_steps": delay_matrix,
        "channel_specs": channel_specs,
        "input_names": input_names,
        "output_names": output_names,
        "Ts": Ts,
        "quantization": str(quantization),
    }


def simulate_discrete_state_space_model(A, B, C, D, input_sequence, x0=None):
    """
    Simulate the discrete state-space model with one output sample recorded
    before the first input move and one after each step.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    input_sequence = np.asarray(input_sequence, dtype=float)

    if input_sequence.ndim != 2:
        raise ValueError("input_sequence must have shape (n_steps, n_inputs).")

    n_states = A.shape[0]
    x = np.zeros(n_states, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    states = [x.copy()]
    outputs = [C @ x]

    for u in input_sequence:
        x = A @ x + B @ u
        y = C @ x + D @ u
        states.append(x.copy())
        outputs.append(np.asarray(y, dtype=float).copy())

    return {
        "states": np.asarray(states, dtype=float),
        "outputs": np.asarray(outputs, dtype=float),
        "inputs": input_sequence.copy(),
    }


def plot_results_statespace_python(validation_cases, output_labels, Ts, title_prefix="Identified vs nonlinear", show=True):
    """
    Plot nonlinear and identified step responses for a list of validation
    cases. Each case must provide `name`, `measured_outputs`, and
    `predicted_outputs`.
    """

    output_labels = list(output_labels)
    validation_cases = list(validation_cases)
    if not validation_cases:
        raise ValueError("validation_cases must not be empty.")

    n_outputs = len(output_labels)
    n_cases = len(validation_cases)
    fig, axs = plt.subplots(
        n_outputs,
        n_cases,
        figsize=(5.5 * n_cases, 3.8 * n_outputs),
        squeeze=False,
        constrained_layout=True,
    )

    def fit_percent(y_sim, y_meas):
        y_sim = np.asarray(y_sim, dtype=float).ravel()
        y_meas = np.asarray(y_meas, dtype=float).ravel()
        denom = np.linalg.norm(y_meas - np.mean(y_meas))
        if denom < 1e-12:
            return 0.0
        return 100.0 * (1.0 - np.linalg.norm(y_sim - y_meas) / denom)

    for col, case in enumerate(validation_cases):
        measured_outputs = np.asarray(case["measured_outputs"], dtype=float)
        predicted_outputs = np.asarray(case["predicted_outputs"], dtype=float)
        if measured_outputs.shape != predicted_outputs.shape:
            raise ValueError("Measured and predicted outputs must have the same shape per validation case.")

        t = np.arange(measured_outputs.shape[0], dtype=float) * float(Ts)
        for row, output_label in enumerate(output_labels):
            fit = fit_percent(predicted_outputs[:, row], measured_outputs[:, row])
            ax = axs[row, col]
            ax.plot(t, measured_outputs[:, row], "r-", linewidth=2, label="Nonlinear")
            ax.plot(t, predicted_outputs[:, row], "b--", linewidth=2, label="Identified")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel(output_label)
            ax.set_title(f"{title_prefix}\n{case['name']} | FIT={fit:.1f}%")
            ax.grid(True)
            if row == 0 and col == 0:
                ax.legend()

    if show:
        plt.show()
    return fig, axs
