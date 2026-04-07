import shutil
from pathlib import Path

from utils.helpers import load_and_prepare_system_data

from .config import ARCHIVED_DISTILLATION_ROOT, DISTILLATION_DATA_SUBDIR, DISTILLATION_RESULT_SUBDIR


LEGACY_FILE_MAP = {
    "system_dict.pickle": "system_dict.pickle",
    "scaling_factor.pickle": "scaling_factor.pickle",
    "min_max_states.pickle": "min_max_states.pickle",
    "Reflux.csv": "Reflux.csv",
    "Reboiler.csv": "Reboiler.csv",
    "mpc_results_200.pickle": "mpc_results_nominal.pickle",
    "mpc_results_dist_ramp.pickle": "mpc_results_disturb_ramp.pickle",
    "mpc_result_dist_ramp_noise.pickle": "mpc_results_disturb_fluctuation.pickle",
}


def resolve_distillation_data_dir(repo_root, override=None):
    path = Path(override) if override else Path(repo_root) / DISTILLATION_DATA_SUBDIR
    return path.resolve()


def resolve_distillation_result_dir(repo_root, override=None):
    path = Path(override) if override else Path(repo_root) / DISTILLATION_RESULT_SUBDIR
    return path.resolve()


def ensure_distillation_directories(repo_root, data_override=None, result_override=None):
    data_dir = resolve_distillation_data_dir(repo_root, override=data_override)
    result_dir = resolve_distillation_result_dir(repo_root, override=result_override)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, result_dir


def copy_legacy_distillation_data(repo_root, overwrite=False):
    repo_root = Path(repo_root)
    src_candidates = [
        repo_root / DISTILLATION_DATA_SUBDIR,
        repo_root / "Data" / "distillation",
        repo_root / ARCHIVED_DISTILLATION_ROOT / "Data",
    ]
    data_dir, result_dir = ensure_distillation_directories(repo_root)
    copied = []
    for src_name, dst_name in LEGACY_FILE_MAP.items():
        for src_dir in src_candidates:
            src = src_dir / src_name
            if not src.exists():
                continue
            dst = data_dir / dst_name
            if dst.exists() and not overwrite:
                break
            shutil.copy2(src, dst)
            copied.append((src, dst))
            break
    return {"data_dir": data_dir, "result_dir": result_dir, "copied": copied}


def canonical_baseline_filename(run_mode, disturbance_profile):
    run_mode = str(run_mode).lower()
    disturbance_profile = str(disturbance_profile).lower()
    if run_mode == "nominal":
        return "mpc_results_nominal.pickle"
    if disturbance_profile == "ramp":
        return "mpc_results_disturb_ramp.pickle"
    if disturbance_profile == "fluctuation":
        return "mpc_results_disturb_fluctuation.pickle"
    raise ValueError("Disturbance distillation baseline must use 'ramp' or 'fluctuation'.")


def canonical_baseline_path(repo_root, run_mode, disturbance_profile, data_override=None):
    return resolve_distillation_data_dir(repo_root, override=data_override) / canonical_baseline_filename(
        run_mode,
        disturbance_profile,
    )


def load_distillation_system_data(
    repo_root,
    steady_states,
    setpoint_y,
    u_min,
    u_max,
    n_inputs=2,
    data_override=None,
):
    data_dir = resolve_distillation_data_dir(repo_root, override=data_override)
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
