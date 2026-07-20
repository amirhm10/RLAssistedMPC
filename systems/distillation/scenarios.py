import numpy as np

from utils.helpers import apply_min_max

from .config import DISTILLATION_NOMINAL_CONDITIONS, DISTILLATION_RL_SETPOINTS_PHYS


DISTILLATION_LEGACY_TRAINING_PROFILE = "legacy_200"
DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE = "temperature_flip_200_100"

TEMPERATURE_FLIP_PHASE1_EPISODES = 200
TEMPERATURE_FLIP_PHASE2_EPISODES = 100
TEMPERATURE_FLIP_TOTAL_EPISODES = 300
TEMPERATURE_FLIP_SETPOINT_HOLD_STEPS = 200

TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS = np.asarray(
    DISTILLATION_RL_SETPOINTS_PHYS,
    dtype=float,
).copy()
TEMPERATURE_FLIP_PHASE2_SETPOINTS_PHYS = TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS.copy()
TEMPERATURE_FLIP_PHASE2_SETPOINTS_PHYS[:, 1] = TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS[::-1, 1]


def canonical_distillation_training_profile(profile_name):
    name = str(profile_name or DISTILLATION_LEGACY_TRAINING_PROFILE).strip().lower()
    aliases = {
        "legacy": DISTILLATION_LEGACY_TRAINING_PROFILE,
        "legacy_fluctuation_200": DISTILLATION_LEGACY_TRAINING_PROFILE,
        DISTILLATION_LEGACY_TRAINING_PROFILE: DISTILLATION_LEGACY_TRAINING_PROFILE,
        "temperature_flip": DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        "robustness": DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        "two_phase_200_100": DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE: DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
    }
    if name not in aliases:
        raise ValueError(
            "Unknown distillation training profile. Expected one of "
            f"{sorted(aliases)}, received {profile_name!r}."
        )
    return aliases[name]


def default_distillation_profile_episode_count(profile_name):
    profile_name = canonical_distillation_training_profile(profile_name)
    if profile_name == DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE:
        return TEMPERATURE_FLIP_TOTAL_EPISODES
    return TEMPERATURE_FLIP_PHASE1_EPISODES


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
    seq, _ = _generate_feed_fluctuation_segment(
        total_steps=total_steps,
        nominal_feed=nominal_feed,
        current_feed=nominal_feed,
        rng=rng,
        slow_horizon_range=slow_horizon_range,
        slow_std=slow_std,
        slow_offset_bounds=slow_offset_bounds,
        fast_std=fast_std,
    )
    return seq


def _generate_feed_fluctuation_segment(
    *,
    total_steps,
    nominal_feed,
    current_feed,
    rng,
    slow_horizon_range=(5000, 10000),
    slow_std=2000.0,
    slow_offset_bounds=(-2500.0, 2500.0),
    fast_std=0.0,
):
    """Generate one fluctuation segment while retaining caller-owned RNG state."""

    total_steps = int(total_steps)
    nominal_feed = float(nominal_feed)
    seq = np.empty(total_steps, dtype=float)
    current = float(current_feed)
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
    return seq, current


def build_distillation_disturbance_schedule(run_mode, disturbance_profile, total_steps, nominal_feed=None, seed=42):
    validate_run_profile(run_mode, disturbance_profile)
    if str(run_mode).lower() == "nominal":
        return None
    nominal_feed = float(DISTILLATION_NOMINAL_CONDITIONS[0] if nominal_feed is None else nominal_feed)
    profile = canonical_disturbance_profile(run_mode, disturbance_profile)
    if profile == "ramp":
        return generate_feed_ramp(total_steps=total_steps, nominal_feed=nominal_feed)
    return generate_feed_fluctuation(total_steps=total_steps, nominal_feed=nominal_feed, seed=seed)


def _episode_bookkeeping(n_episodes, episode_steps, warm_start, test_cycle):
    n_episodes = int(n_episodes)
    episode_steps = int(episode_steps)
    warm_start = int(warm_start)
    if n_episodes <= 0 or episode_steps <= 0:
        raise ValueError("n_episodes and episode_steps must be positive.")
    if warm_start < 0 or warm_start >= n_episodes:
        raise ValueError("warm_start must identify an episode before the end of the run.")

    test_pattern = [bool(value) for value in list(test_cycle)]
    if not test_pattern:
        raise ValueError("test_cycle must contain at least one boolean value.")
    test_flags = [test_pattern[idx % len(test_pattern)] for idx in range(n_episodes)]
    test_flags[-1] = True

    n_steps = n_episodes * episode_steps
    episode_starts = np.arange(0, n_steps, episode_steps, dtype=int)
    episode_ends = episode_starts + episode_steps - 1
    return {
        "nFE": int(n_steps),
        "time_in_sub_episodes": int(episode_steps),
        "sub_episodes_changes_dict": {
            int(step): int(idx + 1) for idx, step in enumerate(episode_ends)
        },
        "test_train_dict": {
            int(step): bool(test_flags[idx]) for idx, step in enumerate(episode_starts)
        },
        "warm_start_step": int(episode_starts[warm_start]),
    }


def _scale_physical_setpoints(setpoints_phys, *, steady_outputs, data_min, data_max, n_inputs):
    setpoints_phys = np.asarray(setpoints_phys, float)
    steady_outputs = np.asarray(steady_outputs, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    n_inputs = int(n_inputs)
    scaled_ss = apply_min_max(steady_outputs, data_min[n_inputs:], data_max[n_inputs:])
    return apply_min_max(setpoints_phys, data_min[n_inputs:], data_max[n_inputs:]) - scaled_ss


def _repeat_setpoint_cycle(setpoints, n_episodes, set_points_len):
    setpoints = np.asarray(setpoints, float)
    blocks = [
        np.repeat(row.reshape(1, -1), int(set_points_len), axis=0)
        for row in setpoints
    ]
    return np.concatenate([np.concatenate(blocks, axis=0)] * int(n_episodes), axis=0)


def build_distillation_training_profile(
    *,
    profile_name,
    run_mode,
    disturbance_profile,
    y_sp_scenario_phys,
    n_tests,
    set_points_len,
    warm_start,
    test_cycle,
    steady_outputs,
    data_min,
    data_max,
    n_inputs,
    nominal_feed=None,
    seed=42,
):
    """Build the canonical distillation setpoint, disturbance, and phase schedule."""

    profile_name = canonical_distillation_training_profile(profile_name)
    run_mode = str(run_mode).lower()
    disturbance_profile = canonical_disturbance_profile(run_mode, disturbance_profile)
    validate_run_profile(run_mode, disturbance_profile)

    y_sp_scenario_phys = np.asarray(y_sp_scenario_phys, float)
    if y_sp_scenario_phys.shape != TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS.shape:
        raise ValueError("Distillation training profiles require two targets and two outputs.")

    if profile_name == DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE:
        if run_mode != "disturb" or disturbance_profile != "fluctuation":
            raise ValueError(
                f"{DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE!r} requires "
                "run_mode='disturb' and disturbance_profile='fluctuation'."
            )
        if int(n_tests) != TEMPERATURE_FLIP_TOTAL_EPISODES:
            raise ValueError(
                f"{DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE!r} requires n_tests="
                f"{TEMPERATURE_FLIP_TOTAL_EPISODES}, received {n_tests}."
            )
        if int(set_points_len) != TEMPERATURE_FLIP_SETPOINT_HOLD_STEPS:
            raise ValueError(
                f"{DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE!r} requires set_points_len="
                f"{TEMPERATURE_FLIP_SETPOINT_HOLD_STEPS}, received {set_points_len}."
            )
        if not np.allclose(
            y_sp_scenario_phys,
            TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "The temperature-flip profile requires the canonical Phase-1 targets "
                f"{TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS.tolist()}."
            )

    n_tests = int(n_tests)
    set_points_len = int(set_points_len)
    episode_steps = set_points_len * y_sp_scenario_phys.shape[0]
    bookkeeping = _episode_bookkeeping(n_tests, episode_steps, warm_start, test_cycle)
    scaled_phase1 = _scale_physical_setpoints(
        y_sp_scenario_phys,
        steady_outputs=steady_outputs,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
    )

    nominal_feed = float(
        DISTILLATION_NOMINAL_CONDITIONS[0] if nominal_feed is None else nominal_feed
    )
    if profile_name == DISTILLATION_LEGACY_TRAINING_PROFILE:
        y_sp = _repeat_setpoint_cycle(scaled_phase1, n_tests, set_points_len)
        disturbance_schedule = build_distillation_disturbance_schedule(
            run_mode,
            disturbance_profile,
            bookkeeping["nFE"],
            nominal_feed=nominal_feed,
            seed=seed,
        )
        return {
            **bookkeeping,
            "y_sp": y_sp,
            "sub_episode_changes_dict": dict(bookkeeping["sub_episodes_changes_dict"]),
            "qi": np.zeros(bookkeeping["nFE"], dtype=float),
            "qs": np.zeros(bookkeeping["nFE"], dtype=float),
            "ha": np.zeros(bookkeeping["nFE"], dtype=float),
            "disturbance_schedule": disturbance_schedule,
            "training_profile_name": DISTILLATION_LEGACY_TRAINING_PROFILE,
            "experiment_phase_windows": [],
            "phase_switch_step": None,
            "phase_switch_episode": None,
            "exploration_freeze_step": None,
            "exploration_freeze_settings": {"enabled": False},
            "phase2_learning_enabled": False,
            "phase1_setpoints_phys": y_sp_scenario_phys.copy(),
        }

    phase1_steps = TEMPERATURE_FLIP_PHASE1_EPISODES * episode_steps
    phase2_steps = TEMPERATURE_FLIP_PHASE2_EPISODES * episode_steps
    scaled_phase2 = _scale_physical_setpoints(
        TEMPERATURE_FLIP_PHASE2_SETPOINTS_PHYS,
        steady_outputs=steady_outputs,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
    )
    y_phase1 = _repeat_setpoint_cycle(
        scaled_phase1,
        TEMPERATURE_FLIP_PHASE1_EPISODES,
        set_points_len,
    )
    y_phase2 = _repeat_setpoint_cycle(
        scaled_phase2,
        TEMPERATURE_FLIP_PHASE2_EPISODES,
        set_points_len,
    )

    rng = np.random.RandomState(int(seed))
    feed_phase1, current_feed = _generate_feed_fluctuation_segment(
        total_steps=phase1_steps,
        nominal_feed=nominal_feed,
        current_feed=nominal_feed,
        rng=rng,
    )
    feed_phase2, _ = _generate_feed_fluctuation_segment(
        total_steps=phase2_steps,
        nominal_feed=nominal_feed,
        current_feed=current_feed,
        rng=rng,
    )
    disturbance_schedule = np.concatenate([feed_phase1, feed_phase2])
    total_steps = int(bookkeeping["nFE"])
    if disturbance_schedule.shape != (total_steps,):
        raise RuntimeError("Distillation two-phase disturbance schedule length is inconsistent.")

    phase_windows = [
        {
            "name": "legacy_training",
            "phase_id": 1,
            "episode_start": 1,
            "episode_end": TEMPERATURE_FLIP_PHASE1_EPISODES,
            "step_start": 0,
            "step_end_exclusive": phase1_steps,
            "learning_enabled": True,
        },
        {
            "name": "temperature_flip_adaptation",
            "phase_id": 2,
            "episode_start": TEMPERATURE_FLIP_PHASE1_EPISODES + 1,
            "episode_end": TEMPERATURE_FLIP_TOTAL_EPISODES,
            "step_start": phase1_steps,
            "step_end_exclusive": total_steps,
            "learning_enabled": True,
            "final_episode_evaluation_only": True,
        },
    ]
    metric_windows = {
        "phase1_tail_episodes_191_200": (191, 200),
        "phase2_entry_episodes_201_210": (201, 210),
        "phase2_tail_episodes_291_300": (291, 300),
    }
    return {
        **bookkeeping,
        "y_sp": np.vstack([y_phase1, y_phase2]),
        "sub_episode_changes_dict": dict(bookkeeping["sub_episodes_changes_dict"]),
        "qi": np.zeros(total_steps, dtype=float),
        "qs": np.zeros(total_steps, dtype=float),
        "ha": np.zeros(total_steps, dtype=float),
        "disturbance_schedule": disturbance_schedule,
        "training_profile_name": DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        "experiment_phase_windows": phase_windows,
        "phase_switch_step": phase1_steps,
        "phase_switch_episode": TEMPERATURE_FLIP_PHASE1_EPISODES + 1,
        "exploration_freeze_step": phase1_steps,
        "exploration_freeze_settings": {
            "enabled": True,
            "freeze_amplitude_only": True,
            "start_step": phase1_steps,
            "start_episode": TEMPERATURE_FLIP_PHASE1_EPISODES + 1,
            "last_training_episode": TEMPERATURE_FLIP_TOTAL_EPISODES - 1,
            "evaluation_episode": TEMPERATURE_FLIP_TOTAL_EPISODES,
        },
        "phase2_learning_enabled": True,
        "phase2_final_episode_evaluation_only": True,
        "phase2_episode_status": {
            "learning_episode_start": TEMPERATURE_FLIP_PHASE1_EPISODES + 1,
            "learning_episode_end": TEMPERATURE_FLIP_TOTAL_EPISODES - 1,
            "evaluation_only_episodes": [TEMPERATURE_FLIP_TOTAL_EPISODES],
        },
        "phase1_setpoints_phys": TEMPERATURE_FLIP_PHASE1_SETPOINTS_PHYS.copy(),
        "phase2_setpoints_phys": TEMPERATURE_FLIP_PHASE2_SETPOINTS_PHYS.copy(),
        "phase1_disturbance_end": {"feed": float(feed_phase1[-1])},
        "phase2_disturbance_end": {"feed": float(feed_phase2[-1])},
        "profile_metric_windows": metric_windows,
    }
