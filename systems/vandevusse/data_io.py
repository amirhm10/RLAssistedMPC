from pathlib import Path

from utils.helpers import load_and_prepare_system_data

from .config import VANDEVUSSE_DATA_SUBDIR, VANDEVUSSE_RESULT_SUBDIR


def resolve_vandevusse_data_dir(repo_root, override=None):
    path = Path(override) if override else Path(repo_root) / VANDEVUSSE_DATA_SUBDIR
    return path.resolve()


def resolve_vandevusse_result_dir(repo_root, override=None):
    path = Path(override) if override else Path(repo_root) / VANDEVUSSE_RESULT_SUBDIR
    return path.resolve()


def ensure_vandevusse_directories(repo_root, data_override=None, result_override=None):
    data_dir = resolve_vandevusse_data_dir(repo_root, override=data_override)
    result_dir = resolve_vandevusse_result_dir(repo_root, override=result_override)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, result_dir


def canonical_baseline_filename(run_mode, disturbance_profile):
    run_mode = str(run_mode).lower()
    disturbance_profile = str(disturbance_profile).lower()
    if run_mode == "nominal":
        return "mpc_results_nominal.pickle"
    if disturbance_profile == "ca0_blocks":
        return "mpc_results_disturb_ca0_blocks.pickle"
    raise ValueError("Van de Vusse disturbance baseline must use disturbance_profile='ca0_blocks'.")


def canonical_baseline_path(repo_root, run_mode, disturbance_profile, data_override=None):
    return resolve_vandevusse_data_dir(repo_root, override=data_override) / canonical_baseline_filename(
        run_mode,
        disturbance_profile,
    )


def load_vandevusse_system_data(
    repo_root,
    steady_states,
    setpoint_y,
    u_min,
    u_max,
    n_inputs=2,
    data_override=None,
):
    data_dir = resolve_vandevusse_data_dir(repo_root, override=data_override)
    try:
        return load_and_prepare_system_data(
            steady_states=steady_states,
            setpoint_y=setpoint_y,
            u_min=u_min,
            u_max=u_max,
            data_dir=data_dir,
            n_inputs=n_inputs,
            system_dict_filename="system_dict.pickle",
            scaling_factor_filename="scaling_factor.pickle",
            min_max_states_filename="min_max_states.pickle",
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing Van de Vusse system-identification artifacts in {data_dir}. "
            "Run vandevusse_systemIdentification_unified.ipynb first."
        ) from exc
