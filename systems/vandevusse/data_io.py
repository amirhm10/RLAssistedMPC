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
