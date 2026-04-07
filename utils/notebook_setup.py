from pathlib import Path

from systems.distillation.config import resolve_aspen_paths
from systems.distillation.data_io import (
    copy_legacy_distillation_data,
    resolve_distillation_data_dir,
    resolve_distillation_result_dir,
)
from systems.distillation.scenarios import canonical_disturbance_profile, validate_run_profile
from systems.polymer.data_io import copy_legacy_polymer_data, resolve_polymer_data_dir, resolve_polymer_result_dir
from utils.helpers import resolve_repo_root


def prepare_polymer_notebook_env(data_dir_override=None, results_dir_override=None):
    repo_root = resolve_repo_root(Path.cwd())
    copy_legacy_polymer_data(repo_root)
    data_dir = resolve_polymer_data_dir(repo_root, override=data_dir_override)
    result_dir = resolve_polymer_result_dir(repo_root, override=results_dir_override)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, data_dir, result_dir


def prepare_distillation_notebook_env(
    run_mode,
    disturbance_profile,
    family,
    aspen_preset="default",
    dyn_path_override=None,
    snaps_path_override=None,
    aspen_root_override=None,
    data_dir_override=None,
    results_dir_override=None,
):
    repo_root = resolve_repo_root(Path.cwd())
    validate_run_profile(run_mode, disturbance_profile)
    disturbance_profile = canonical_disturbance_profile(run_mode, disturbance_profile)

    copy_legacy_distillation_data(repo_root)
    data_dir = resolve_distillation_data_dir(repo_root) if data_dir_override is None else Path(data_dir_override).resolve()
    result_dir = (
        resolve_distillation_result_dir(repo_root) if results_dir_override is None else Path(results_dir_override).resolve()
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dyn_path, snaps_path, aspen_source = resolve_aspen_paths(
        family=family,
        disturbance_profile=disturbance_profile,
        aspen_preset=aspen_preset,
        dyn_path_override=dyn_path_override,
        snaps_path_override=snaps_path_override,
        aspen_root=aspen_root_override,
    )
    return repo_root, data_dir, result_dir, disturbance_profile, dyn_path, snaps_path, aspen_source


def print_notebook_summary(title, items):
    print(title)
    for key, value in items.items():
        print(f"  {key:<20}: {value}")


def print_grouped_notebook_summary(title, groups):
    print(title)
    for group_name, items in groups.items():
        print(f"\n[{group_name}]")
        for key, value in items.items():
            print(f"  {key:<20}: {value}")
