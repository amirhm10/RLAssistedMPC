import numpy as np

from .config import DISTILLATION_NOMINAL_CONDITIONS


def canonical_disturbance_profile(run_mode, disturbance_profile):
    run_mode = str(run_mode).lower()
    disturbance_profile = str(disturbance_profile).lower()
    if run_mode == "nominal":
        return "none"
    if disturbance_profile not in {"ramp", "fluctuation"}:
        raise ValueError("Distillation disturbance runs must use 'ramp' or 'fluctuation'.")
    return disturbance_profile


def validate_run_profile(run_mode, disturbance_profile):
    run_mode = str(run_mode).lower()
    disturbance_profile = str(disturbance_profile).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("run_mode must be 'nominal' or 'disturb'.")
    if disturbance_profile not in {"none", "ramp", "fluctuation"}:
        raise ValueError("disturbance_profile must be 'none', 'ramp', or 'fluctuation'.")
    if run_mode == "nominal" and disturbance_profile != "none":
        raise ValueError("Nominal distillation runs must use DISTURBANCE_PROFILE='none'.")
    if run_mode == "disturb" and disturbance_profile == "none":
        raise ValueError("Disturbance distillation runs must choose 'ramp' or 'fluctuation'.")


def generate_feed_ramp(total_steps, nominal_feed=float(DISTILLATION_NOMINAL_CONDITIONS[0]), target_feed=154000.0):
    return np.linspace(float(nominal_feed), float(target_feed), int(total_steps), dtype=float)


def generate_feed_fluctuation(
    total_steps,
    nominal_feed=float(DISTILLATION_NOMINAL_CONDITIONS[0]),
    slow_horizon_range=(5000, 10000),
    slow_std=2000.0,
    slow_offset_bounds=(-2500.0, 2500.0),
    fast_std=0.0,
    seed=42,
):
    total_steps = int(total_steps)
    nominal_feed = float(nominal_feed)
    rng = np.random.RandomState(int(seed))
    seq = np.empty(total_steps, dtype=float)
    current = nominal_feed
    idx = 0

    while idx < total_steps:
        horizon = int(rng.randint(int(slow_horizon_range[0]), int(slow_horizon_range[1])))
        horizon = min(horizon, total_steps - idx)

        if slow_offset_bounds is not None:
            lo_off, hi_off = map(float, slow_offset_bounds)
            offset = float(rng.randn() * float(slow_std))
            while offset < lo_off or offset > hi_off:
                offset = float(rng.randn() * float(slow_std))
            target = nominal_feed + offset
        else:
            drift = float(rng.randn() * float(slow_std))
            target = current + drift

        ramp = np.linspace(current, target, horizon, dtype=float)
        noise = rng.randn(horizon) * float(fast_std)
        seq[idx : idx + horizon] = ramp + noise
        current = target
        idx += horizon
    return seq


def build_distillation_disturbance_schedule(run_mode, disturbance_profile, total_steps, nominal_feed=None, seed=42):
    validate_run_profile(run_mode, disturbance_profile)
    if str(run_mode).lower() == "nominal":
        return None
    nominal_feed = float(DISTILLATION_NOMINAL_CONDITIONS[0] if nominal_feed is None else nominal_feed)
    profile = canonical_disturbance_profile(run_mode, disturbance_profile)
    if profile == "ramp":
        return generate_feed_ramp(total_steps=total_steps, nominal_feed=nominal_feed)
    return generate_feed_fluctuation(total_steps=total_steps, nominal_feed=nominal_feed, seed=seed)
