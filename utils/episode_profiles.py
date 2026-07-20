from __future__ import annotations

from copy import deepcopy

import numpy as np


EPISODE_PROFILE_RESULT_KEYS = (
    "training_profile_name",
    "experiment_phase_windows",
    "phase_switch_step",
    "phase_switch_episode",
    "exploration_freeze_step",
    "exploration_freeze_settings",
    "fouling_active",
    "fouled_ha",
    "phase2_learning_enabled",
    "phase2_final_episode_evaluation_only",
    "phase2_episode_status",
    "phase1_setpoints_phys",
    "phase2_setpoints_phys",
    "phase1_disturbance_end",
    "phase2_disturbance_end",
    "profile_metric_windows",
)


def validate_episode_bundle(
    episode_bundle,
    *,
    expected_n_tests,
    expected_n_outputs,
):
    """Validate and copy a step-level episode schedule supplied by an entrypoint."""

    if not isinstance(episode_bundle, dict):
        raise TypeError("runtime_ctx['episode_bundle'] must be a dictionary.")

    required = (
        "y_sp",
        "nFE",
        "sub_episodes_changes_dict",
        "test_train_dict",
        "warm_start_step",
        "time_in_sub_episodes",
    )
    missing = [key for key in required if key not in episode_bundle]
    if missing:
        raise KeyError(f"runtime_ctx['episode_bundle'] is missing required fields: {missing}.")

    copied = deepcopy(episode_bundle)
    y_sp = np.asarray(copied["y_sp"], float)
    nFE = int(copied["nFE"])
    episode_steps = int(copied["time_in_sub_episodes"])
    expected_n_tests = int(expected_n_tests)
    expected_n_outputs = int(expected_n_outputs)

    if y_sp.shape != (nFE, expected_n_outputs):
        raise ValueError(
            "runtime_ctx['episode_bundle']['y_sp'] must have shape "
            f"({nFE}, {expected_n_outputs}), received {y_sp.shape}."
        )
    if not np.all(np.isfinite(y_sp)):
        raise ValueError("runtime_ctx['episode_bundle']['y_sp'] must contain only finite values.")
    if episode_steps <= 0 or nFE != expected_n_tests * episode_steps:
        raise ValueError(
            "runtime_ctx['episode_bundle'] length must equal "
            "expected_n_tests * time_in_sub_episodes."
        )

    episode_ends = dict(copied["sub_episodes_changes_dict"])
    episode_starts = dict(copied["test_train_dict"])
    if len(episode_ends) != expected_n_tests or len(episode_starts) != expected_n_tests:
        raise ValueError(
            "runtime_ctx['episode_bundle'] must contain one episode-end and one test/train entry "
            "for every episode."
        )

    expected_starts = np.arange(0, nFE, episode_steps, dtype=int)
    expected_ends = expected_starts + episode_steps - 1
    if sorted(int(key) for key in episode_starts) != expected_starts.tolist():
        raise ValueError("runtime_ctx['episode_bundle']['test_train_dict'] has invalid episode starts.")
    if sorted(int(key) for key in episode_ends) != expected_ends.tolist():
        raise ValueError(
            "runtime_ctx['episode_bundle']['sub_episodes_changes_dict'] has invalid episode ends."
        )

    copied["y_sp"] = y_sp.copy()
    copied["nFE"] = nFE
    copied["time_in_sub_episodes"] = episode_steps
    copied["sub_episodes_changes_dict"] = {
        int(key): int(value) for key, value in episode_ends.items()
    }
    copied["sub_episode_changes_dict"] = dict(copied["sub_episodes_changes_dict"])
    copied["test_train_dict"] = {
        int(key): bool(value) for key, value in episode_starts.items()
    }
    copied["warm_start_step"] = int(copied["warm_start_step"])

    for key in ("qi", "qs", "ha"):
        values = copied.get(key)
        if values is None:
            copied[key] = np.zeros(nFE, dtype=float)
            continue
        values = np.asarray(values, float).reshape(-1)
        if values.shape != (nFE,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"runtime_ctx['episode_bundle'][{key!r}] must be a finite length-{nFE} array."
            )
        copied[key] = values.copy()

    disturbance_schedule = copied.get("disturbance_schedule")
    if disturbance_schedule is not None:
        disturbance_schedule = np.asarray(disturbance_schedule, float)
        if disturbance_schedule.shape[0] != nFE or not np.all(np.isfinite(disturbance_schedule)):
            raise ValueError(
                "runtime_ctx['episode_bundle']['disturbance_schedule'] must be finite and have "
                f"{nFE} rows."
            )
        copied["disturbance_schedule"] = disturbance_schedule.copy()

    return copied


def resolve_episode_bundle(
    runtime_ctx,
    *,
    fallback_builder,
    expected_n_tests,
    expected_n_outputs,
):
    supplied = runtime_ctx.get("episode_bundle")
    if supplied is None:
        return fallback_builder()
    return validate_episode_bundle(
        supplied,
        expected_n_tests=expected_n_tests,
        expected_n_outputs=expected_n_outputs,
    )


def episode_profile_result_fields(profile_bundle):
    """Return portable experiment-profile metadata for result persistence."""

    return {
        key: deepcopy(profile_bundle[key])
        for key in EPISODE_PROFILE_RESULT_KEYS
        if key in profile_bundle
    }
