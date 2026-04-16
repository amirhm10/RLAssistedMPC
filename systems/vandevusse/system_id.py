from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from Simulation.sys_ids import (
    generate_step_test_sequence,
    save_simulation_data,
    scaling_min_max_factors,
    simulate_discrete_state_space_model,
)

from .config import VANDEVUSSE_SYSTEM_ID_CSV_COLUMNS
from .data_io import ensure_vandevusse_directories
from .labels import VANDEVUSSE_SYSTEM_METADATA
from .plant import build_vandevusse_system, vandevusse_system_stepper


def simulate_vandevusse_system(system, input_sequence, disturbance_schedule=None):
    input_sequence = np.asarray(input_sequence, dtype=float)
    outputs = [np.asarray(system.current_output, dtype=float).copy()]
    states = [np.asarray(system.current_state, dtype=float).copy()]

    if disturbance_schedule is not None and len(disturbance_schedule) != len(input_sequence):
        raise ValueError("disturbance_schedule must have the same length as input_sequence.")

    for idx, inp in enumerate(input_sequence):
        system.current_input = np.asarray(inp, dtype=float).copy()
        disturbance_step = None if disturbance_schedule is None else disturbance_schedule[idx]
        vandevusse_system_stepper(system, disturbance_step=disturbance_step)
        states.append(np.asarray(system.current_state, dtype=float).copy())
        outputs.append(np.asarray(system.current_output, dtype=float).copy())

    return {
        "inputs": input_sequence.copy(),
        "outputs": np.asarray(outputs, dtype=float),
        "states": np.asarray(states, dtype=float),
        "steady_state": np.asarray(system.steady_trajectory, dtype=float).copy(),
        "y_ss": np.asarray(system.y_ss, dtype=float).copy(),
    }


def build_vandevusse_step_test_inputs(ss_inputs, step_tests, delta_t, initial_hold_hours, step_hold_hours, input_bounds=None):
    ss_inputs = np.asarray(ss_inputs, dtype=float)
    initial_steps = int(round(float(initial_hold_hours) / float(delta_t)))
    step_steps = int(round(float(step_hold_hours) / float(delta_t)))
    built_tests = []

    u_min = None
    u_max = None
    if input_bounds is not None:
        u_min = np.asarray(input_bounds["u_min"], dtype=float)
        u_max = np.asarray(input_bounds["u_max"], dtype=float)

    for step_cfg in step_tests:
        step_delta = np.asarray(step_cfg["step_delta"], dtype=float)
        stepped_inputs = ss_inputs + step_delta
        if u_min is not None and np.any(stepped_inputs < u_min):
            raise ValueError(f"Step test {step_cfg['name']} violates lower input bounds.")
        if u_max is not None and np.any(stepped_inputs > u_max):
            raise ValueError(f"Step test {step_cfg['name']} violates upper input bounds.")

        built_tests.append(
            {
                "name": str(step_cfg["name"]),
                "save_filename": str(step_cfg["save_filename"]),
                "input_index": step_cfg.get("input_index"),
                "step_delta": step_delta.copy(),
                "input_sequence": generate_step_test_sequence(
                    ss_inputs,
                    step_delta,
                    initial_duration=initial_steps,
                    step_duration=step_steps,
                ),
                "initial_steps": initial_steps,
                "step_steps": step_steps,
            }
        )

    return built_tests


def plot_vandevusse_step_test(results, step_name, delta_t, result_dir=None, show=True):
    time = np.arange(results["outputs"].shape[0], dtype=float) * float(delta_t)
    input_time = time[:-1]
    output_labels = list(VANDEVUSSE_SYSTEM_METADATA["output_labels"])
    input_labels = list(VANDEVUSSE_SYSTEM_METADATA["input_labels"])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axs[0, 0].plot(time, results["outputs"][:, 0], linewidth=2)
    axs[0, 0].set_ylabel(output_labels[0])
    axs[0, 0].set_title(f"{step_name}: output responses")
    axs[0, 0].grid(True)

    axs[1, 0].plot(time, results["outputs"][:, 1], linewidth=2)
    axs[1, 0].set_xlabel("Time (h)")
    axs[1, 0].set_ylabel(output_labels[1])
    axs[1, 0].grid(True)

    axs[0, 1].step(input_time, results["inputs"][:, 0], where="post", linewidth=2)
    axs[0, 1].set_ylabel(input_labels[0])
    axs[0, 1].set_title(f"{step_name}: inputs")
    axs[0, 1].grid(True)

    axs[1, 1].step(input_time, results["inputs"][:, 1], where="post", linewidth=2)
    axs[1, 1].set_xlabel("Time (h)")
    axs[1, 1].set_ylabel(input_labels[1])
    axs[1, 1].grid(True)

    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(result_dir / f"{step_name}_step_test.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axs


def run_vandevusse_step_test_experiment(
    step_cfg,
    system_params,
    design_params,
    ss_inputs,
    delta_t,
    data_dir,
    result_dir=None,
    show_plot=False,
):
    system = build_vandevusse_system(
        params=system_params,
        design_params=design_params,
        ss_inputs=ss_inputs,
        delta_t=delta_t,
        deviation_form=False,
    )
    results = simulate_vandevusse_system(system, step_cfg["input_sequence"])

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / step_cfg["save_filename"]
    data_to_save = np.concatenate((results["inputs"], results["outputs"][1:]), axis=1)
    save_simulation_data(data_to_save, save_path, VANDEVUSSE_SYSTEM_ID_CSV_COLUMNS)

    if result_dir is not None:
        plot_vandevusse_step_test(
            results,
            step_name=step_cfg["name"],
            delta_t=delta_t,
            result_dir=result_dir,
            show=show_plot,
        )

    results["csv_path"] = save_path
    results["time"] = np.arange(results["outputs"].shape[0], dtype=float) * float(delta_t)
    return results


def solve_vandevusse_nominal_steady_state(system_params, design_params, ss_inputs, delta_t):
    plant = build_vandevusse_system(
        params=system_params,
        design_params=design_params,
        ss_inputs=ss_inputs,
        delta_t=delta_t,
        deviation_form=False,
    )
    return {
        "x_ss": np.asarray(plant.steady_trajectory, dtype=float).copy(),
        "ss_inputs": np.asarray(plant.ss_inputs, dtype=float).copy(),
        "y_ss": np.asarray(plant.y_ss, dtype=float).copy(),
    }


def _perturbation_scale(nominal_value, rel_step, abs_floor):
    return max(abs(float(nominal_value)) * float(rel_step), float(abs_floor))


def linearize_vandevusse_continuous(
    system_params,
    design_params,
    x_ss,
    u_ss,
    linearization_cfg,
    delta_t,
):
    x_ss = np.asarray(x_ss, dtype=float)
    u_ss = np.asarray(u_ss, dtype=float)

    plant = build_vandevusse_system(
        params=system_params,
        design_params=design_params,
        ss_inputs=u_ss,
        delta_t=delta_t,
        deviation_form=False,
    )

    def f(x, u):
        return np.asarray(plant.odes(0.0, np.asarray(x, dtype=float), np.asarray(u, dtype=float)), dtype=float)

    n_states = x_ss.size
    n_inputs = u_ss.size
    A_c = np.zeros((n_states, n_states), dtype=float)
    B_c = np.zeros((n_states, n_inputs), dtype=float)
    state_eps = np.zeros(n_states, dtype=float)
    input_eps = np.zeros(n_inputs, dtype=float)

    for idx in range(n_states):
        eps = _perturbation_scale(
            x_ss[idx],
            linearization_cfg["state_eps_rel"],
            linearization_cfg["state_eps_abs"],
        )
        state_eps[idx] = eps
        dx = np.zeros(n_states, dtype=float)
        dx[idx] = eps
        f_plus = f(x_ss + dx, u_ss)
        f_minus = f(x_ss - dx, u_ss)
        A_c[:, idx] = (f_plus - f_minus) / (2.0 * eps)

    for idx in range(n_inputs):
        eps = _perturbation_scale(
            u_ss[idx],
            linearization_cfg["input_eps_rel"],
            linearization_cfg["input_eps_abs"],
        )
        input_eps[idx] = eps
        du = np.zeros(n_inputs, dtype=float)
        du[idx] = eps
        f_plus = f(x_ss, u_ss + du)
        f_minus = f(x_ss, u_ss - du)
        B_c[:, idx] = (f_plus - f_minus) / (2.0 * eps)

    C_c = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
    D_c = np.zeros((2, 2), dtype=float)
    return {
        "A_c": A_c,
        "B_c": B_c,
        "C_c": C_c,
        "D_c": D_c,
        "state_eps": state_eps,
        "input_eps": input_eps,
    }


def discretize_vandevusse_linear_model(A_c, B_c, C_c, D_c, delta_t, method="zoh"):
    A_d, B_d, C_d, D_d, _ = signal.cont2discrete(
        (np.asarray(A_c, dtype=float), np.asarray(B_c, dtype=float), np.asarray(C_c, dtype=float), np.asarray(D_c, dtype=float)),
        float(delta_t),
        method=str(method),
    )
    return {
        "A_d": np.asarray(A_d, dtype=float),
        "B_d": np.asarray(B_d, dtype=float),
        "C_d": np.asarray(C_d, dtype=float),
        "D_d": np.asarray(D_d, dtype=float),
    }


def build_vandevusse_nominal_linear_model(system_params, design_params, ss_inputs, delta_t, linearization_cfg):
    steady_states = solve_vandevusse_nominal_steady_state(
        system_params=system_params,
        design_params=design_params,
        ss_inputs=ss_inputs,
        delta_t=delta_t,
    )
    continuous_model = linearize_vandevusse_continuous(
        system_params=system_params,
        design_params=design_params,
        x_ss=steady_states["x_ss"],
        u_ss=steady_states["ss_inputs"],
        linearization_cfg=linearization_cfg,
        delta_t=delta_t,
    )
    discrete_model = discretize_vandevusse_linear_model(
        continuous_model["A_c"],
        continuous_model["B_c"],
        continuous_model["C_c"],
        continuous_model["D_c"],
        delta_t=delta_t,
        method=linearization_cfg["discretization_method"],
    )
    system_dict = {
        "A": discrete_model["A_d"],
        "B": discrete_model["B_d"],
        "C": discrete_model["C_d"],
        "D": discrete_model["D_d"],
    }
    return {
        "steady_states": steady_states,
        "continuous_model": continuous_model,
        "discrete_model": discrete_model,
        "system_dict": system_dict,
    }


def apply_vandevusse_deviation_form(steady_states, file_paths):
    ss_inputs = np.asarray(steady_states["ss_inputs"], dtype=float)
    y_ss = np.asarray(steady_states["y_ss"], dtype=float)
    ss = np.concatenate((ss_inputs, y_ss), axis=0)

    deviations = {}
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        deviations[key] = df - ss
    return deviations


def simulate_vandevusse_linear_model(system_dict, input_deviation_sequence, x0=None):
    return simulate_discrete_state_space_model(
        np.asarray(system_dict["A"], dtype=float),
        np.asarray(system_dict["B"], dtype=float),
        np.asarray(system_dict["C"], dtype=float),
        np.asarray(system_dict["D"], dtype=float),
        np.asarray(input_deviation_sequence, dtype=float),
        x0=x0,
    )


def _fit_percent(y_sim, y_meas):
    y_sim = np.asarray(y_sim, dtype=float).ravel()
    y_meas = np.asarray(y_meas, dtype=float).ravel()
    denom = np.linalg.norm(y_meas - np.mean(y_meas))
    if denom < 1e-12:
        return 0.0
    return 100.0 * (1.0 - np.linalg.norm(y_sim - y_meas) / denom)


def _rmse(y_sim, y_meas):
    y_sim = np.asarray(y_sim, dtype=float).ravel()
    y_meas = np.asarray(y_meas, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_sim - y_meas) ** 2)))


def plot_vandevusse_linear_validation(validation_cases, output_labels, delta_t, result_dir=None, show=True):
    output_labels = list(output_labels)
    validation_cases = list(validation_cases)
    fig, axs = plt.subplots(
        len(output_labels),
        len(validation_cases),
        figsize=(5.5 * len(validation_cases), 4.0 * len(output_labels)),
        squeeze=False,
        constrained_layout=True,
    )

    for col, case in enumerate(validation_cases):
        time = np.arange(case["measured_abs"].shape[0], dtype=float) * float(delta_t)
        for row, label in enumerate(output_labels):
            metrics_key = case["metric_keys"][row]
            metrics = case["metrics"][metrics_key]
            ax = axs[row, col]
            ax.plot(time, case["measured_abs"][:, row], "r-", linewidth=2, label="Nonlinear")
            ax.plot(time, case["predicted_abs"][:, row], "b--", linewidth=2, label="Linearized")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel(label)
            ax.set_title(f"{case['name']} | FIT={metrics['fit_percent']:.1f}% | RMSE={metrics['rmse']:.4g}")
            ax.grid(True)
            if row == 0 and col == 0:
                ax.legend()

    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(result_dir / "vandevusse_linearized_vs_nonlinear.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axs


def validate_vandevusse_linearized_model(
    system_dict,
    absolute_dfs,
    deviation_dfs,
    step_tests,
    steady_states,
    delta_t,
    result_dir=None,
    show_plot=True,
):
    y_ss = np.asarray(steady_states["y_ss"], dtype=float)
    validation_cases = []
    validation_metrics = {}
    output_names = list(VANDEVUSSE_SYSTEM_ID_CSV_COLUMNS[2:])

    for step_cfg in step_tests:
        abs_df = absolute_dfs[step_cfg["name"]]
        dev_df = deviation_dfs[step_cfg["name"]]
        input_dev = dev_df.iloc[:, :2].to_numpy(dtype=float)
        measured_dev = dev_df.iloc[:, 2:].to_numpy(dtype=float)
        measured_abs = abs_df.iloc[:, 2:].to_numpy(dtype=float)

        simulated = simulate_vandevusse_linear_model(system_dict, input_dev)
        predicted_dev = simulated["outputs"][1:]
        predicted_abs = predicted_dev + y_ss

        metrics = {}
        for idx, output_name in enumerate(output_names):
            metrics[output_name] = {
                "fit_percent": float(_fit_percent(predicted_dev[:, idx], measured_dev[:, idx])),
                "rmse": float(_rmse(predicted_dev[:, idx], measured_dev[:, idx])),
            }

        validation_metrics[step_cfg["name"]] = metrics
        validation_cases.append(
            {
                "name": step_cfg["name"],
                "measured_abs": measured_abs,
                "predicted_abs": predicted_abs,
                "metrics": metrics,
                "metric_keys": output_names,
            }
        )

    fig, axs = plot_vandevusse_linear_validation(
        validation_cases,
        output_labels=VANDEVUSSE_SYSTEM_METADATA["output_labels"],
        delta_t=delta_t,
        result_dir=result_dir,
        show=show_plot,
    )
    return {
        "cases": validation_cases,
        "metrics_by_test": validation_metrics,
        "figure": fig,
        "axes": axs,
    }


def compute_vandevusse_min_max_states(step_results_by_name, steady_states):
    x_ss = np.asarray(steady_states["x_ss"], dtype=float)
    y_ss = np.asarray(steady_states["y_ss"], dtype=float)
    samples = []

    for results in step_results_by_name.values():
        state_dev = np.asarray(results["states"], dtype=float) - x_ss
        output_dev = np.asarray(results["outputs"], dtype=float) - y_ss
        samples.append(np.hstack((state_dev, output_dev)))

    if not samples:
        raise ValueError("step_results_by_name must contain at least one simulated experiment.")

    stacked = np.vstack(samples)
    return {
        "min_s": np.min(stacked, axis=0),
        "max_s": np.max(stacked, axis=0),
    }


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return obj


def save_vandevusse_identification_artifacts(
    repo_root,
    data_dir,
    result_dir,
    system_dict,
    scaling_factor,
    min_max_states,
    metadata,
    save_metadata_json=True,
):
    data_dir, result_dir = ensure_vandevusse_directories(repo_root, data_override=data_dir, result_override=result_dir)

    artifact_paths = {
        "system_dict": data_dir / "system_dict.pickle",
        "scaling_factor": data_dir / "scaling_factor.pickle",
        "min_max_states": data_dir / "min_max_states.pickle",
        "metadata_pickle": data_dir / "system_id_metadata.pickle",
    }

    with open(artifact_paths["system_dict"], "wb") as handle:
        pickle.dump(system_dict, handle)
    with open(artifact_paths["scaling_factor"], "wb") as handle:
        pickle.dump(scaling_factor, handle)
    with open(artifact_paths["min_max_states"], "wb") as handle:
        pickle.dump(min_max_states, handle)
    with open(artifact_paths["metadata_pickle"], "wb") as handle:
        pickle.dump(metadata, handle)

    if save_metadata_json:
        artifact_paths["metadata_json"] = data_dir / "system_id_metadata.json"
        artifact_paths["metadata_json"].write_text(
            json.dumps(_to_serializable(metadata), indent=2),
            encoding="utf-8",
        )

    return artifact_paths


__all__ = [
    "apply_vandevusse_deviation_form",
    "build_vandevusse_nominal_linear_model",
    "build_vandevusse_step_test_inputs",
    "compute_vandevusse_min_max_states",
    "discretize_vandevusse_linear_model",
    "linearize_vandevusse_continuous",
    "run_vandevusse_step_test_experiment",
    "save_vandevusse_identification_artifacts",
    "scaling_min_max_factors",
    "simulate_vandevusse_linear_model",
    "simulate_vandevusse_system",
    "solve_vandevusse_nominal_steady_state",
    "validate_vandevusse_linearized_model",
]
