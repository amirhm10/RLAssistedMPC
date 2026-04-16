from __future__ import annotations

from pathlib import Path

import control
import numpy as np

from Simulation.mpc import MpcSolverGeneral
from utils.helpers import apply_min_max, resolve_repo_root
from utils.mpc_baseline_runner import run_offsetfree_mpc
from utils.observer import compute_observer_gain

from .data_io import canonical_baseline_path, load_vandevusse_system_data, resolve_vandevusse_data_dir, resolve_vandevusse_result_dir
from .labels import VANDEVUSSE_SYSTEM_METADATA
from .plant import build_vandevusse_system, vandevusse_system_stepper
from .scenarios import build_vandevusse_disturbance_schedule, canonical_disturbance_profile, validate_run_profile


def _build_quadratic_reward(Q_out, R_in):
    Q_out = np.asarray(Q_out, dtype=float).reshape(-1)
    R_in = np.asarray(R_in, dtype=float).reshape(-1)

    def reward_fn(delta_y, delta_u, y_sp_phys=None):
        del y_sp_phys
        delta_y = np.asarray(delta_y, dtype=float).reshape(-1)
        delta_u = np.asarray(delta_u, dtype=float).reshape(-1)
        return -float(np.sum(Q_out * delta_y ** 2) + np.sum(R_in * delta_u ** 2))

    return {
        "reward_kind": "quadratic_mpc",
        "Q_out": Q_out.copy(),
        "R_in": R_in.copy(),
    }, reward_fn


def resolve_vandevusse_observer_gain(A_aug, C_aug, default_poles, fallback_poles):
    A_aug = np.asarray(A_aug, dtype=float)
    C_aug = np.asarray(C_aug, dtype=float)
    default_poles = np.asarray(default_poles, dtype=float).reshape(-1)
    fallback_poles = np.asarray(fallback_poles, dtype=float).reshape(-1)

    n_states = int(A_aug.shape[0])
    if default_poles.size != n_states or fallback_poles.size != n_states:
        raise ValueError("Observer pole vectors must match the augmented state dimension.")

    observability_matrix = control.obsv(A_aug, C_aug)
    rank = int(np.linalg.matrix_rank(observability_matrix))
    if rank != n_states:
        raise ValueError(
            f"Van de Vusse augmented model is not observable (rank {rank} < {n_states}); cannot place observer poles."
        )

    def _attempt(poles):
        L = np.asarray(compute_observer_gain(A_aug, C_aug, poles), dtype=float)
        if not np.all(np.isfinite(L)):
            raise ValueError("Observer gain contains non-finite values.")
        eigvals = np.linalg.eigvals(A_aug - L @ C_aug)
        if not np.all(np.isfinite(eigvals)):
            raise ValueError("Observer error dynamics eigenvalues are non-finite.")
        spectral_radius = float(np.max(np.abs(eigvals)))
        if spectral_radius >= 1.0:
            raise ValueError(
                f"Observer error dynamics are unstable or marginally stable (spectral radius {spectral_radius:.6f})."
            )
        return L, eigvals, spectral_radius

    default_error = None
    try:
        L, eigvals, spectral_radius = _attempt(default_poles)
        return {
            "L": L,
            "poles_requested": default_poles.copy(),
            "poles_used": default_poles.copy(),
            "used_fallback": False,
            "observer_error_eigs": eigvals,
            "observer_error_spectral_radius": spectral_radius,
        }
    except Exception as exc:
        default_error = exc

    try:
        L, eigvals, spectral_radius = _attempt(fallback_poles)
        return {
            "L": L,
            "poles_requested": default_poles.copy(),
            "poles_used": fallback_poles.copy(),
            "used_fallback": True,
            "observer_error_eigs": eigvals,
            "observer_error_spectral_radius": spectral_radius,
        }
    except Exception as exc:
        raise RuntimeError(
            "Could not place a stable Van de Vusse observer with either the default or fallback poles. "
            f"Default error: {default_error}. Fallback error: {exc}."
        ) from exc


def prepare_vandevusse_offset_free_mpc_runtime(
    repo_root,
    baseline_cfg,
    run_mode=None,
    disturbance_profile=None,
):
    repo_root = resolve_repo_root(repo_root or Path.cwd())
    baseline_cfg = dict(baseline_cfg)

    run_mode = str(baseline_cfg.get("run_mode") if run_mode is None else run_mode).lower()
    disturbance_profile_name = str(
        baseline_cfg.get("disturbance_profile") if disturbance_profile is None else disturbance_profile
    ).lower()
    validate_run_profile(run_mode, disturbance_profile_name)
    disturbance_profile_name = canonical_disturbance_profile(run_mode, disturbance_profile_name)

    sys = dict(baseline_cfg["system_setup"])
    ctrl = dict(baseline_cfg["controller"])
    run_profiles = dict(baseline_cfg["run_profiles"])
    run_profile = dict(run_profiles[(run_mode, disturbance_profile_name)])

    data_dir = resolve_vandevusse_data_dir(repo_root, override=baseline_cfg.get("data_dir_override"))
    result_dir = resolve_vandevusse_result_dir(repo_root, override=baseline_cfg.get("results_dir_override"))
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    system = build_vandevusse_system(
        params=np.asarray(sys["system_params"], dtype=float).copy(),
        design_params=np.asarray(sys["design_params"], dtype=float).copy(),
        ss_inputs=np.asarray(sys["ss_inputs"], dtype=float).copy(),
        delta_t=float(sys["delta_t_hours"]),
        deviation_form=False,
    )
    steady_states = {
        "x_ss": np.asarray(system.steady_trajectory, dtype=float).copy(),
        "y_ss": np.asarray(system.y_ss, dtype=float).copy(),
        "ss_inputs": np.asarray(system.ss_inputs, dtype=float).copy(),
    }

    setpoint_range_phys = np.asarray(sys["setpoint_range_phys"], dtype=float).copy()
    u_min = np.asarray(sys["input_bounds"]["u_min"], dtype=float).copy()
    u_max = np.asarray(sys["input_bounds"]["u_max"], dtype=float).copy()
    system_data = load_vandevusse_system_data(
        repo_root=repo_root,
        steady_states=steady_states,
        setpoint_y=setpoint_range_phys,
        u_min=u_min,
        u_max=u_max,
        data_override=data_dir,
    )

    A_aug = np.asarray(system_data["A_aug"], dtype=float)
    B_aug = np.asarray(system_data["B_aug"], dtype=float)
    C_aug = np.asarray(system_data["C_aug"], dtype=float)
    data_min = np.asarray(system_data["data_min"], dtype=float)
    data_max = np.asarray(system_data["data_max"], dtype=float)
    n_inputs = int(B_aug.shape[1])

    y_sp_scenario_phys = np.asarray(sys["baseline_setpoints_phys"], dtype=float).copy()
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled

    observer_info = resolve_vandevusse_observer_gain(
        A_aug=A_aug,
        C_aug=C_aug,
        default_poles=np.asarray(sys["observer_poles_default"], dtype=float),
        fallback_poles=np.asarray(sys["observer_poles_fallback"], dtype=float),
    )

    Q_out = np.asarray(ctrl["Q_out"], dtype=float).reshape(-1)
    R_in = np.asarray(ctrl["R_in"], dtype=float).reshape(-1)
    if Q_out.size != 2 or R_in.size != 2:
        raise ValueError("Van de Vusse baseline MPC expects two-output Q_out and two-input R_in vectors.")

    reward_params, reward_fn = _build_quadratic_reward(Q_out=Q_out, R_in=R_in)
    mpc_obj = MpcSolverGeneral(
        A_aug,
        B_aug,
        C_aug,
        Q_out=Q_out,
        R_in=R_in,
        NP=int(ctrl["predict_h"]),
        NC=int(ctrl["cont_h"]),
    )

    n_tests = int(run_profile["n_tests"] if baseline_cfg.get("n_tests_override") is None else baseline_cfg["n_tests_override"])
    set_points_len = int(
        run_profile["set_points_len"]
        if baseline_cfg.get("set_points_len_override") is None
        else baseline_cfg["set_points_len_override"]
    )
    warm_start = int(run_profile["warm_start"] if baseline_cfg.get("warm_start_override") is None else baseline_cfg["warm_start_override"])
    test_cycle = list(
        run_profile["test_cycle"] if baseline_cfg.get("test_cycle_override") is None else baseline_cfg["test_cycle_override"]
    )
    plot_start_episode = int(
        run_profile.get("plot_start_episode", 1)
        if baseline_cfg.get("plot_start_episode_override") is None
        else baseline_cfg["plot_start_episode_override"]
    )
    total_steps = int(n_tests * set_points_len * len(y_sp_scenario_phys))

    disturbance_schedule = build_vandevusse_disturbance_schedule(
        run_mode=run_mode,
        disturbance_profile=disturbance_profile_name,
        total_steps=total_steps,
        design_params=np.asarray(sys["design_params"], dtype=float).copy(),
        block_values={
            "c_A0": np.asarray(sys["disturbance_block_values"]["c_A0"], dtype=float).copy(),
            "T_in": np.asarray(sys["disturbance_block_values"]["T_in"], dtype=float).copy(),
            "block_length": int(set_points_len),
        },
    )

    baseline_save_path = (
        Path(baseline_cfg["baseline_save_path_override"]).expanduser()
        if baseline_cfg.get("baseline_save_path_override")
        else canonical_baseline_path(
            repo_root=repo_root,
            run_mode=run_mode,
            disturbance_profile=disturbance_profile_name,
            data_override=data_dir,
        )
    )
    result_prefix = baseline_cfg.get("result_prefix_override") or f"vandevusse_baseline_{run_mode}_{disturbance_profile_name}_unified"

    mpc_cfg = {
        "run_mode": run_mode,
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "warm_start": warm_start,
        "test_cycle": test_cycle,
        "predict_h": int(ctrl["predict_h"]),
        "cont_h": int(ctrl["cont_h"]),
        "use_shifted_mpc_warm_start": bool(ctrl.get("use_shifted_mpc_warm_start", False)),
        "Q1_penalty": float(Q_out[0]),
        "Q2_penalty": float(Q_out[1]),
        "R1_penalty": float(R_in[0]),
        "R2_penalty": float(R_in[1]),
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
        "b_min": np.asarray(system_data["b_min"], dtype=float),
        "b_max": np.asarray(system_data["b_max"], dtype=float),
    }
    runtime_ctx = {
        "system": system,
        "MPC_obj": mpc_obj,
        "steady_states": steady_states,
        "data_min": data_min,
        "data_max": data_max,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "y_sp_scenario": y_sp_scenario,
        "reward_fn": reward_fn,
        "L": np.asarray(observer_info["L"], dtype=float),
        "poles": np.asarray(observer_info["poles_used"], dtype=float),
        "system_metadata": VANDEVUSSE_SYSTEM_METADATA,
        "disturbance_schedule": disturbance_schedule,
        "system_stepper": vandevusse_system_stepper,
        "disturbance_labels": VANDEVUSSE_SYSTEM_METADATA["disturbance_labels"],
    }

    return {
        "repo_root": Path(repo_root),
        "data_dir": data_dir,
        "result_dir": result_dir,
        "run_mode": run_mode,
        "disturbance_profile_name": disturbance_profile_name,
        "baseline_save_path": baseline_save_path,
        "result_prefix": result_prefix,
        "plot_start_episode": plot_start_episode,
        "system_data": system_data,
        "steady_states": steady_states,
        "y_sp_scenario_phys": y_sp_scenario_phys,
        "observer_info": observer_info,
        "reward_params": reward_params,
        "controller_params": {
            "predict_h": int(ctrl["predict_h"]),
            "cont_h": int(ctrl["cont_h"]),
            "Q_out": Q_out.copy(),
            "R_in": R_in.copy(),
            "use_shifted_mpc_warm_start": bool(ctrl.get("use_shifted_mpc_warm_start", False)),
        },
        "mpc_cfg": mpc_cfg,
        "runtime_ctx": runtime_ctx,
    }


def run_vandevusse_offset_free_mpc(prepared_runtime):
    prepared_runtime = dict(prepared_runtime)
    result_bundle = run_offsetfree_mpc(
        mpc_cfg=dict(prepared_runtime["mpc_cfg"]),
        runtime_ctx=dict(prepared_runtime["runtime_ctx"]),
    )
    result_bundle["system_metadata"] = dict(VANDEVUSSE_SYSTEM_METADATA)
    result_bundle["reward_params"] = dict(prepared_runtime["reward_params"])
    result_bundle["observer_poles_requested"] = np.asarray(
        prepared_runtime["observer_info"]["poles_requested"], dtype=float
    )
    result_bundle["observer_poles_used"] = np.asarray(prepared_runtime["observer_info"]["poles_used"], dtype=float)
    result_bundle["observer_used_fallback"] = bool(prepared_runtime["observer_info"]["used_fallback"])
    result_bundle["observer_error_eigs"] = np.asarray(prepared_runtime["observer_info"]["observer_error_eigs"], dtype=complex)
    result_bundle["observer_error_spectral_radius"] = float(
        prepared_runtime["observer_info"]["observer_error_spectral_radius"]
    )
    result_bundle["disturbance_profile_name"] = prepared_runtime["disturbance_profile_name"]
    result_bundle["baseline_save_path"] = str(prepared_runtime["baseline_save_path"])
    result_bundle["controller_params"] = {
        key: np.asarray(value, dtype=float).copy() if isinstance(value, np.ndarray) else value
        for key, value in prepared_runtime["controller_params"].items()
    }
    result_bundle["setpoint_scenario_phys"] = np.asarray(prepared_runtime["y_sp_scenario_phys"], dtype=float).copy()
    result_bundle["data_dir"] = str(prepared_runtime["data_dir"])
    result_bundle["result_dir"] = str(prepared_runtime["result_dir"])
    return result_bundle


__all__ = [
    "prepare_vandevusse_offset_free_mpc_runtime",
    "resolve_vandevusse_observer_gain",
    "run_vandevusse_offset_free_mpc",
]
