from __future__ import annotations

import numpy as np

from utils.multiplier_mapping import map_centered_bounds_to_action


PHASE_DISABLED = 0
PHASE_NOMINAL = 1
PHASE_PROTECTED = 2
PHASE_RAMP = 3
PHASE_FULL = 4


def extract_suggested_bounds_from_diagnostic(diagnostic_result, labels):
    """Return compact advisory bounds from a Step 1 diagnostic result."""
    labels = tuple(str(label) for label in labels)
    if diagnostic_result is None:
        return None

    rows = diagnostic_result.get("suggested_bounds") if isinstance(diagnostic_result, dict) else None
    if rows is None:
        return None

    by_label = {str(row.get("coordinate_label", row.get("coordinate"))): row for row in rows}
    low = []
    high = []
    missing = []
    for label in labels:
        row = by_label.get(label)
        if row is None:
            missing.append(label)
            continue
        low.append(float(row["suggested_low"]))
        high.append(float(row["suggested_high"]))

    if missing:
        raise ValueError(f"Step 1 diagnostic suggested_bounds is missing coordinates: {missing}")

    return {"labels": labels, "low": np.asarray(low, float), "high": np.asarray(high, float)}


def build_release_authority_schedule(
    *,
    config,
    labels,
    wide_low,
    wide_high,
    warm_start_step,
    action_freeze_end_step=None,
    time_in_sub_episodes=1,
    n_steps=None,
):
    """Build phase-dependent release bounds for advisory multiplier caps."""
    cfg = dict(config or {})
    labels = tuple(str(label) for label in labels)
    wide_low = _as_1d_float(wide_low, "wide_low")
    wide_high = _as_1d_float(wide_high, "wide_high")
    _validate_wide_bounds(wide_low, wide_high, labels)

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "labels": labels,
            "wide_low": wide_low,
            "wide_high": wide_high,
            "diagnostic_low": wide_low.copy(),
            "diagnostic_high": wide_high.copy(),
            "nominal_until_step": int(warm_start_step),
            "release_start_step": int(warm_start_step) + 1,
            "protected_end_step": int(warm_start_step),
            "ramp_start_step": int(warm_start_step) + 1,
            "ramp_end_step": int(warm_start_step),
            "store_executed_action_in_replay": bool(cfg.get("store_executed_action_in_replay", True)),
            "log_policy_and_executed_multipliers": bool(cfg.get("log_policy_and_executed_multipliers", True)),
        }

    fail_if_missing = bool(cfg.get("fail_if_diagnostic_missing", True))
    advisory_bounds = cfg.get("advisory_bounds")
    if bool(cfg.get("use_offline_diagnostic_bounds", True)):
        diag_low, diag_high = _coerce_advisory_bounds(
            advisory_bounds=advisory_bounds,
            labels=labels,
            wide_low=wide_low,
            wide_high=wide_high,
            fail_if_missing=fail_if_missing,
        )
    else:
        diag_low, diag_high = wide_low.copy(), wide_high.copy()

    subepisode_len = max(1, int(time_in_sub_episodes))
    protected_steps = max(0, int(cfg.get("protected_live_subepisodes", 15))) * subepisode_len
    ramp_steps = max(0, int(cfg.get("authority_ramp_subepisodes", 30))) * subepisode_len
    nominal_until_step = max(int(warm_start_step), int(action_freeze_end_step if action_freeze_end_step is not None else warm_start_step))
    release_start_step = nominal_until_step + 1
    protected_end_step = release_start_step + protected_steps - 1 if protected_steps > 0 else nominal_until_step
    ramp_start_step = protected_end_step + 1
    ramp_end_step = ramp_start_step + ramp_steps - 1 if ramp_steps > 0 else protected_end_step

    if n_steps is not None:
        n_steps = int(n_steps)

    return {
        "enabled": True,
        "labels": labels,
        "wide_low": wide_low,
        "wide_high": wide_high,
        "diagnostic_low": diag_low,
        "diagnostic_high": diag_high,
        "nominal_until_step": nominal_until_step,
        "release_start_step": release_start_step,
        "protected_end_step": protected_end_step,
        "ramp_start_step": ramp_start_step,
        "ramp_end_step": ramp_end_step,
        "protected_steps": protected_steps,
        "ramp_steps": ramp_steps,
        "n_steps": n_steps,
        "store_executed_action_in_replay": bool(cfg.get("store_executed_action_in_replay", True)),
        "log_policy_and_executed_multipliers": bool(cfg.get("log_policy_and_executed_multipliers", True)),
    }


def effective_bounds_for_step(schedule, step_idx):
    """Return effective low/high multiplier bounds and phase metadata for a step."""
    step_idx = int(step_idx)
    wide_low = np.asarray(schedule["wide_low"], float)
    wide_high = np.asarray(schedule["wide_high"], float)
    if not bool(schedule.get("enabled", False)):
        return wide_low.copy(), wide_high.copy(), PHASE_DISABLED, 1.0

    if step_idx <= int(schedule["nominal_until_step"]):
        ones = np.ones_like(wide_low, dtype=float)
        return ones, ones.copy(), PHASE_NOMINAL, 0.0

    diag_low = np.asarray(schedule["diagnostic_low"], float)
    diag_high = np.asarray(schedule["diagnostic_high"], float)
    if step_idx <= int(schedule["protected_end_step"]):
        return diag_low.copy(), diag_high.copy(), PHASE_PROTECTED, 0.0

    ramp_end = int(schedule["ramp_end_step"])
    ramp_start = int(schedule["ramp_start_step"])
    if ramp_start <= step_idx <= ramp_end and ramp_end >= ramp_start:
        denom = max(1, ramp_end - ramp_start)
        ramp_fraction = float(np.clip((step_idx - ramp_start) / denom, 0.0, 1.0))
        low = _log_interpolate(diag_low, wide_low, ramp_fraction)
        high = _log_interpolate(diag_high, wide_high, ramp_fraction)
        return low, high, PHASE_RAMP, ramp_fraction

    return wide_low.copy(), wide_high.copy(), PHASE_FULL, 1.0


def clip_multipliers_to_release_bounds(multipliers, schedule, step_idx):
    """Clip requested multipliers to Step 2 bounds and return trace metadata."""
    multipliers = _as_1d_float(multipliers, "multipliers")
    low, high, phase_code, ramp_fraction = effective_bounds_for_step(schedule, step_idx)
    if multipliers.shape != low.shape:
        raise ValueError("multipliers and release bounds must have the same shape.")
    executed = np.clip(multipliers, low, high)
    clip_mask = np.abs(executed - multipliers) > 1e-12
    return {
        "multipliers": executed,
        "low": low,
        "high": high,
        "phase_code": int(phase_code),
        "ramp_fraction": float(ramp_fraction),
        "clip_mask": clip_mask,
        "clip_fraction": float(np.mean(clip_mask)) if clip_mask.size else 0.0,
        "guard_active": bool(phase_code in {PHASE_NOMINAL, PHASE_PROTECTED, PHASE_RAMP}),
    }


def map_effective_multipliers_to_raw_action(effective_multipliers, wide_low, wide_high):
    """Map executed multipliers back to the original wide raw action space."""
    return map_centered_bounds_to_action(effective_multipliers, wide_low, wide_high, nominal=1.0)


def _coerce_advisory_bounds(*, advisory_bounds, labels, wide_low, wide_high, fail_if_missing):
    if advisory_bounds is None:
        if fail_if_missing:
            raise RuntimeError(
                "Step 2 release-protected advisory caps are enabled, but Step 1 diagnostic bounds were not provided."
            )
        return wide_low.copy(), wide_high.copy()

    if isinstance(advisory_bounds, dict) and {"labels", "low", "high"}.issubset(advisory_bounds):
        source_labels = tuple(str(label) for label in advisory_bounds["labels"])
        source_low = _as_1d_float(advisory_bounds["low"], "advisory_low")
        source_high = _as_1d_float(advisory_bounds["high"], "advisory_high")
        if len(source_labels) != source_low.size or source_low.shape != source_high.shape:
            raise ValueError("Compact advisory bounds must have matching labels, low, and high lengths.")
        by_label = {
            label: (float(source_low[idx]), float(source_high[idx]))
            for idx, label in enumerate(source_labels)
        }
    else:
        rows = advisory_bounds.get("suggested_bounds") if isinstance(advisory_bounds, dict) else advisory_bounds
        by_label = {
            str(row.get("coordinate_label", row.get("coordinate"))): (
                float(row["suggested_low"]),
                float(row["suggested_high"]),
            )
            for row in rows
        }

    diag_low = wide_low.copy()
    diag_high = wide_high.copy()
    missing = []
    for idx, label in enumerate(labels):
        pair = by_label.get(label)
        if pair is None:
            missing.append(label)
            continue
        diag_low[idx] = float(pair[0])
        diag_high[idx] = float(pair[1])

    if missing and fail_if_missing:
        raise RuntimeError(f"Step 2 advisory bounds are missing coordinates: {missing}")

    if np.any(diag_low <= 0.0) or np.any(diag_high <= 0.0):
        raise ValueError("Step 2 advisory bounds must be positive for log-space ramping.")
    if np.any(diag_low > 1.0) or np.any(diag_high < 1.0):
        raise ValueError("Step 2 advisory bounds must contain the nominal multiplier 1.0.")

    diag_low = np.maximum(diag_low, wide_low)
    diag_high = np.minimum(diag_high, wide_high)
    if np.any(diag_low > diag_high):
        raise ValueError("Step 2 advisory bounds became invalid after clipping to wide bounds.")
    return diag_low, diag_high


def _as_1d_float(value, name):
    arr = np.asarray(value, float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain infs or NaNs.")
    return arr.astype(float, copy=True)


def _validate_wide_bounds(wide_low, wide_high, labels):
    if wide_low.shape != wide_high.shape:
        raise ValueError("wide_low and wide_high must have matching shapes.")
    if len(labels) != wide_low.size:
        raise ValueError("labels must have the same length as the multiplier bounds.")
    if np.any(wide_low <= 0.0):
        raise ValueError("Wide lower bounds must be positive.")
    if np.any(wide_high <= wide_low):
        raise ValueError("Wide upper bounds must be greater than wide lower bounds.")
    if np.any(wide_low >= 1.0) or np.any(wide_high <= 1.0):
        raise ValueError("Wide bounds must strictly contain the nominal multiplier 1.0.")


def _log_interpolate(start, end, fraction):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return np.exp((1.0 - fraction) * np.log(start) + fraction * np.log(end))


__all__ = [
    "PHASE_DISABLED",
    "PHASE_NOMINAL",
    "PHASE_PROTECTED",
    "PHASE_RAMP",
    "PHASE_FULL",
    "build_release_authority_schedule",
    "clip_multipliers_to_release_bounds",
    "effective_bounds_for_step",
    "extract_suggested_bounds_from_diagnostic",
    "map_effective_multipliers_to_raw_action",
]
