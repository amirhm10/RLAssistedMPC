from pathlib import Path

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
