"""Explicitly non-scientific, compressed-clock integration checks."""
from contextlib import contextmanager
from copy import deepcopy
import importlib

import numpy as np

from .runtime import RUNNER_TARGETS


@contextmanager
def compressed_smoke_clock(category, stride=200):
    """Retain episode order/events, but shorten dwell for technical tests only.

    First build the authoritative, unmodified 300 x 800-step Scenario 2 profile.
    A process-local adapter then compresses the episode clock. No study defaults,
    source files, or training equations are changed. These runs cannot provide
    scientific performance evidence and are labelled smoke throughout.
    """
    from utils.episode_profiles import validate_episode_bundle
    module = importlib.import_module(RUNNER_TARGETS[category][0])
    original = module.resolve_episode_bundle

    def resolve(*args, **kwargs):
        bundle = deepcopy(original(*args, **kwargs))
        original_nfe = int(bundle["nFE"])
        if original_nfe % stride or int(bundle["time_in_sub_episodes"]) % stride:
            raise ValueError("Smoke stride must divide the original episode clock.")
        for key, value in list(bundle.items()):
            if isinstance(value, np.ndarray) and value.shape and value.shape[0] == original_nfe:
                bundle[key] = value[::stride].copy()
        for key in ("sub_episodes_changes_dict", "sub_episode_changes_dict"):
            bundle[key] = {(int(step) + 1) // stride - 1: episode for step, episode in bundle[key].items()}
        bundle["test_train_dict"] = {int(step) // stride: flag for step, flag in bundle["test_train_dict"].items()}
        for key in ("nFE", "time_in_sub_episodes", "warm_start_step", "phase_switch_step", "exploration_freeze_step"):
            if key in bundle:
                bundle[key] = int(bundle[key]) // stride
        for window in bundle.get("experiment_phase_windows", []):
            for key in ("step_start", "step_end_exclusive"):
                window[key] = int(window[key]) // stride
        settings = bundle.get("exploration_freeze_settings", {})
        if "start_step" in settings:
            settings["start_step"] = int(settings["start_step"]) // stride
        bundle["e1_smoke_clock_stride"] = stride
        return validate_episode_bundle(bundle, expected_n_tests=kwargs["expected_n_tests"],
                                       expected_n_outputs=kwargs["expected_n_outputs"])

    module.resolve_episode_bundle = resolve
    try:
        yield
    finally:
        module.resolve_episode_bundle = original
