from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Simulation.sys_ids import (
    apply_deviation_form_scaled,
    build_mimo_state_space_from_fopdt_python,
    extract_fopdt_2863_auto,
    generate_step_test_sequence,
    plot_results_statespace_python,
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
                "input_index": step_cfg["input_index"],
                "fit_use": bool(step_cfg["fit_use"]),
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


def aggregate_fopdt_channel_fits(fit_results_by_test, input_names, output_names):
    aggregated = {}
    for input_idx, input_name in enumerate(input_names):
        for output_name in output_names:
            samples = []
            source_tests = []
            inverse_flags = []
            for test_name, fit_bundle in fit_results_by_test.items():
                if int(fit_bundle["input_index"]) != int(input_idx):
                    continue
                fit_entry = fit_bundle["channel_fits"][output_name]
                values = np.array([fit_entry["kp"], fit_entry["taup"], fit_entry["theta"]], dtype=float)
                if not np.all(np.isfinite(values)):
                    continue
                samples.append(values)
                source_tests.append(test_name)
                inverse_flags.append(bool(fit_entry.get("inverse", False)))

            if not samples:
                raise ValueError(f"No finite FOPDT fits were found for input '{input_name}' and output '{output_name}'.")

            samples_arr = np.asarray(samples, dtype=float)
            med = np.median(samples_arr, axis=0)
            aggregated[(input_name, output_name)] = {
                "kp": float(med[0]),
                "taup": float(med[1]),
                "theta": float(med[2]),
                "sample_count": int(len(samples)),
                "source_tests": list(source_tests),
                "inverse_detected": bool(any(inverse_flags)),
            }
    return aggregated


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


def build_vandevusse_identification_model(
    deviation_dfs,
    step_tests,
    delta_t,
    pre_window_steps,
    post_window_steps,
    plot=True,
    quantization="round",
):
    input_names = list(VANDEVUSSE_SYSTEM_ID_CSV_COLUMNS[:2])
    output_names = list(VANDEVUSSE_SYSTEM_ID_CSV_COLUMNS[2:])

    fit_results_by_test = {}
    for step_cfg in step_tests:
        if not step_cfg["fit_use"]:
            continue
        df = deviation_dfs[step_cfg["name"]]
        channel_fits, fit_df = extract_fopdt_2863_auto(
            df,
            input_idx=step_cfg["input_index"],
            Ts=delta_t,
            pre_win=pre_window_steps,
            post_win=post_window_steps,
            plot=plot,
        )
        fit_results_by_test[step_cfg["name"]] = {
            "input_index": int(step_cfg["input_index"]),
            "channel_fits": channel_fits,
            "fit_rows": int(len(fit_df)),
        }

    aggregated_fits = aggregate_fopdt_channel_fits(fit_results_by_test, input_names=input_names, output_names=output_names)
    state_space = build_mimo_state_space_from_fopdt_python(
        aggregated_fits,
        input_names=input_names,
        output_names=output_names,
        Ts=delta_t,
        quantization=quantization,
    )

    system_dict = {
        "A": np.asarray(state_space["A"], dtype=float),
        "B": np.asarray(state_space["B"], dtype=float),
        "C": np.asarray(state_space["C"], dtype=float),
        "D": np.asarray(state_space["D"], dtype=float),
    }
    return {
        "system_dict": system_dict,
        "fit_results_by_test": fit_results_by_test,
        "aggregated_fits": aggregated_fits,
        "state_space": state_space,
    }


def validate_vandevusse_identified_model(system_dict, deviation_dfs, step_tests, delta_t, result_dir=None, show_plot=True):
    validation_cases = []
    for step_cfg in step_tests:
        df = deviation_dfs[step_cfg["name"]]
        input_sequence = df.iloc[:, :2].to_numpy(dtype=float)
        measured_outputs = df.iloc[:, 2:].to_numpy(dtype=float)
        simulated = simulate_discrete_state_space_model(
            system_dict["A"],
            system_dict["B"],
            system_dict["C"],
            system_dict["D"],
            input_sequence,
        )
        validation_cases.append(
            {
                "name": step_cfg["name"],
                "measured_outputs": measured_outputs,
                "predicted_outputs": simulated["outputs"][1:],
            }
        )

    fig, axs = plot_results_statespace_python(
        validation_cases,
        output_labels=VANDEVUSSE_SYSTEM_METADATA["output_labels"],
        Ts=delta_t,
        title_prefix="Van de Vusse identified vs nonlinear",
        show=show_plot,
    )
    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(result_dir / "vandevusse_identified_vs_nonlinear.png", dpi=200, bbox_inches="tight")
    if not show_plot:
        plt.close(fig)
    return {"cases": validation_cases, "figure": fig, "axes": axs}


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
    "aggregate_fopdt_channel_fits",
    "apply_deviation_form_scaled",
    "build_vandevusse_identification_model",
    "build_vandevusse_step_test_inputs",
    "compute_vandevusse_min_max_states",
    "run_vandevusse_step_test_experiment",
    "save_vandevusse_identification_artifacts",
    "scaling_min_max_factors",
    "simulate_vandevusse_system",
    "validate_vandevusse_identified_model",
]
