from __future__ import annotations

import numpy as np

from .config import VANDEVUSSE_BASELINE_CA0_BLOCKS, VANDEVUSSE_BASELINE_DISTURBANCE_PROFILES, VANDEVUSSE_BASELINE_TIN_BLOCKS


def canonical_disturbance_profile(run_mode, disturbance_profile):
    run_mode = str(run_mode).lower()
    disturbance_profile = str(disturbance_profile).lower()
    if run_mode == "nominal":
        return "none"
    if disturbance_profile not in VANDEVUSSE_BASELINE_DISTURBANCE_PROFILES:
        raise ValueError(
            f"Van de Vusse disturbance runs must use one of {VANDEVUSSE_BASELINE_DISTURBANCE_PROFILES}."
        )
    if disturbance_profile == "none":
        raise ValueError("Van de Vusse disturbance runs must use disturbance_profile='ca0_blocks'.")
    return disturbance_profile


def validate_run_profile(run_mode, disturbance_profile):
    run_mode = str(run_mode).lower()
    disturbance_profile = str(disturbance_profile).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("run_mode must be 'nominal' or 'disturb'.")
    if disturbance_profile not in VANDEVUSSE_BASELINE_DISTURBANCE_PROFILES:
        raise ValueError(
            f"disturbance_profile must be one of {VANDEVUSSE_BASELINE_DISTURBANCE_PROFILES}."
        )
    if run_mode == "nominal" and disturbance_profile != "none":
        raise ValueError("Nominal Van de Vusse runs must use disturbance_profile='none'.")
    if run_mode == "disturb" and disturbance_profile != "ca0_blocks":
        raise ValueError("Disturbance Van de Vusse runs must use disturbance_profile='ca0_blocks'.")


def _build_piecewise_constant_cycle(values, block_length, total_steps):
    values = np.asarray(values, dtype=float).reshape(-1)
    block_length = int(block_length)
    total_steps = int(total_steps)
    if block_length <= 0:
        raise ValueError("block_length must be positive.")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    cycle = np.repeat(values, block_length)
    repeats = int(np.ceil(total_steps / len(cycle)))
    return np.tile(cycle, repeats)[:total_steps]


def build_vandevusse_disturbance_schedule(
    run_mode,
    disturbance_profile,
    total_steps,
    design_params,
    block_values=None,
):
    validate_run_profile(run_mode, disturbance_profile)
    if str(run_mode).lower() == "nominal":
        return None

    design_params = np.asarray(design_params, dtype=float).reshape(-1)
    if design_params.size < 2:
        raise ValueError("design_params must contain at least [c_A0, T_in].")

    cfg = dict(block_values or {})
    c_a0_values = np.asarray(cfg.get("c_A0", VANDEVUSSE_BASELINE_CA0_BLOCKS), dtype=float).reshape(-1)
    t_in_values = np.asarray(cfg.get("T_in", VANDEVUSSE_BASELINE_TIN_BLOCKS), dtype=float).reshape(-1)
    if c_a0_values.size != t_in_values.size:
        raise ValueError("c_A0 and T_in disturbance block arrays must have the same length.")

    block_length = int(cfg.get("block_length", max(1, int(total_steps) // int(c_a0_values.size))))
    total_steps = int(total_steps)
    return {
        "c_A0": _build_piecewise_constant_cycle(c_a0_values, block_length, total_steps),
        "T_in": _build_piecewise_constant_cycle(t_in_values, block_length, total_steps),
    }


__all__ = [
    "build_vandevusse_disturbance_schedule",
    "canonical_disturbance_profile",
    "validate_run_profile",
]
