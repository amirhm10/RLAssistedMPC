import shutil
from pathlib import Path

from utils.helpers import load_and_prepare_system_data

from .config import POLYMER_DATA_SUBDIR, POLYMER_RESULT_SUBDIR


LEGACY_ROOT_DATA = Path("Data")
LEGACY_FILE_NAMES = (
    "system_dict",
    "system_dict.pickle",
    "scaling_factor.pickle",
    "min_max_states.pickle",
    "Qc.csv",
    "Qm.csv",
)


def resolve_polymer_data_dir(repo_root, override=None):
    path = Path(override) if override else Path(repo_root) / POLYMER_DATA_SUBDIR
    return path.resolve()


def resolve_polymer_result_dir(repo_root, override=None):
    path = Path(override) if override else Path(repo_root) / POLYMER_RESULT_SUBDIR
    return path.resolve()


def ensure_polymer_directories(repo_root, data_override=None, result_override=None):
    data_dir = resolve_polymer_data_dir(repo_root, override=data_override)
    result_dir = resolve_polymer_result_dir(repo_root, override=result_override)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, result_dir


def copy_legacy_polymer_data(repo_root, overwrite=False):
    repo_root = Path(repo_root)
    src_dir = repo_root / LEGACY_ROOT_DATA
    data_dir, result_dir = ensure_polymer_directories(repo_root)
    copied = []

    for name in LEGACY_FILE_NAMES:
        src = src_dir / name
        dst = data_dir / name
        if not src.exists():
            continue
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied.append((src, dst))

    for src in src_dir.glob("mpc_results*.pickle"):
        dst = data_dir / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied.append((src, dst))

    return {"data_dir": data_dir, "result_dir": result_dir, "copied": copied}


def canonical_baseline_path(repo_root, run_mode, data_override=None):
    run_mode = str(run_mode).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("run_mode must be 'nominal' or 'disturb'.")
    filename = "mpc_results_nominal.pickle" if run_mode == "nominal" else "mpc_results_dist.pickle"
    return resolve_polymer_data_dir(repo_root, override=data_override) / filename


def load_polymer_system_data(repo_root, steady_states, setpoint_y, u_min, u_max, n_inputs=2, data_override=None):
    data_dir = resolve_polymer_data_dir(repo_root, override=data_override)
    return load_and_prepare_system_data(
        steady_states=steady_states,
        setpoint_y=setpoint_y,
        u_min=u_min,
        u_max=u_max,
        data_dir=data_dir,
        n_inputs=n_inputs,
        system_dict_filename="system_dict",
        scaling_factor_filename="scaling_factor.pickle",
        min_max_states_filename="min_max_states.pickle",
    )
