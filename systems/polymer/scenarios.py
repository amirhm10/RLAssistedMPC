from __future__ import annotations

from copy import deepcopy

import numpy as np

from utils.helpers import apply_min_max, generate_setpoints_training_rl_gradually


POLYMER_LEGACY_TRAINING_PROFILE = "legacy_gradual_200"
POLYMER_ROBUSTNESS_TRAINING_PROFILE = "robustness_200_100"

ROBUSTNESS_PHASE1_EPISODES = 200
ROBUSTNESS_PHASE2_EPISODES = 100
ROBUSTNESS_TOTAL_EPISODES = ROBUSTNESS_PHASE1_EPISODES + ROBUSTNESS_PHASE2_EPISODES
ROBUSTNESS_SETPOINT_HOLD_STEPS = 400

ROBUSTNESS_PHASE2_SETPOINTS_PHYS = np.asarray(
    [
        [4.0, 321.5],
        [3.3, 324.5],
    ],
    dtype=float,
)

ROBUSTNESS_PHASE2_QI_FINAL = 108.0 * 0.95
ROBUSTNESS_PHASE2_QS_FINAL = 459.0 * 1.05
ROBUSTNESS_FOULED_HA = 1.05e6 * 0.85


def canonical_polymer_training_profile(profile_name: str | None) -> str:
    name = str(profile_name or POLYMER_LEGACY_TRAINING_PROFILE).strip().lower()
    aliases = {
        "legacy": POLYMER_LEGACY_TRAINING_PROFILE,
        "legacy_200": POLYMER_LEGACY_TRAINING_PROFILE,
        POLYMER_LEGACY_TRAINING_PROFILE: POLYMER_LEGACY_TRAINING_PROFILE,
        "robustness": POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        "two_phase": POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        "two_phase_200_100": POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        POLYMER_ROBUSTNESS_TRAINING_PROFILE: POLYMER_ROBUSTNESS_TRAINING_PROFILE,
    }
    if name not in aliases:
        raise ValueError(
            "Unknown polymer training profile. Expected one of "
            f"{sorted(aliases)}, received {profile_name!r}."
        )
    return aliases[name]


def default_polymer_profile_episode_count(profile_name: str | None) -> int:
    profile_name = canonical_polymer_training_profile(profile_name)
    if profile_name == POLYMER_ROBUSTNESS_TRAINING_PROFILE:
        return ROBUSTNESS_TOTAL_EPISODES
    return ROBUSTNESS_PHASE1_EPISODES


def _episode_bookkeeping(n_episodes: int, episode_steps: int, warm_start: int, test_cycle) -> dict:
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


def _scale_physical_setpoints(
    setpoints_phys,
    *,
    steady_outputs,
    data_min,
    data_max,
    n_inputs: int,
) -> np.ndarray:
    setpoints_phys = np.asarray(setpoints_phys, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    steady_outputs = np.asarray(steady_outputs, float)
    n_inputs = int(n_inputs)
    scaled_ss = apply_min_max(steady_outputs, data_min[n_inputs:], data_max[n_inputs:])
    return apply_min_max(setpoints_phys, data_min[n_inputs:], data_max[n_inputs:]) - scaled_ss


def _repeat_setpoint_cycle(setpoints, n_episodes: int, set_points_len: int) -> np.ndarray:
    setpoints = np.asarray(setpoints, float)
    blocks = [np.repeat(row.reshape(1, -1), int(set_points_len), axis=0) for row in setpoints]
    return np.concatenate([np.concatenate(blocks, axis=0)] * int(n_episodes), axis=0)


def _legacy_episode_bundle(
    *,
    y_sp_scenario,
    n_tests,
    set_points_len,
    warm_start,
    test_cycle,
    nominal_qi,
    nominal_qs,
    nominal_ha,
    qi_change,
    qs_change,
    ha_change,
) -> dict:
    (
        y_sp,
        nFE,
        sub_episodes_changes_dict,
        time_in_sub_episodes,
        test_train_dict,
        warm_start_step,
        qi,
        qs,
        ha,
    ) = generate_setpoints_training_rl_gradually(
        np.asarray(y_sp_scenario, float),
        int(n_tests),
        int(set_points_len),
        int(warm_start),
        list(test_cycle),
        float(nominal_qi),
        float(nominal_qs),
        float(nominal_ha),
        float(qi_change),
        float(qs_change),
        float(ha_change),
    )
    return {
        "y_sp": np.asarray(y_sp, float),
        "nFE": int(nFE),
        "sub_episodes_changes_dict": dict(sub_episodes_changes_dict),
        "sub_episode_changes_dict": dict(sub_episodes_changes_dict),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "test_train_dict": dict(test_train_dict),
        "warm_start_step": int(warm_start_step),
        "qi": np.asarray(qi, float),
        "qs": np.asarray(qs, float),
        "ha": np.asarray(ha, float),
        "training_profile_name": POLYMER_LEGACY_TRAINING_PROFILE,
        "experiment_phase_windows": [],
        "phase_switch_step": None,
        "phase_switch_episode": None,
        "exploration_freeze_step": None,
        "exploration_freeze_settings": {"enabled": False},
        "fouling_active": False,
        "phase2_learning_enabled": False,
    }


def build_polymer_training_profile(
    *,
    profile_name,
    y_sp_scenario,
    n_tests,
    set_points_len,
    warm_start,
    test_cycle,
    nominal_qi,
    nominal_qs,
    nominal_ha,
    qi_change,
    qs_change,
    ha_change,
    steady_outputs,
    data_min,
    data_max,
    n_inputs: int,
) -> dict:
    """Build the canonical polymer step-level training and disturbance profile."""

    profile_name = canonical_polymer_training_profile(profile_name)
    if profile_name == POLYMER_LEGACY_TRAINING_PROFILE:
        return _legacy_episode_bundle(
            y_sp_scenario=y_sp_scenario,
            n_tests=n_tests,
            set_points_len=set_points_len,
            warm_start=warm_start,
            test_cycle=test_cycle,
            nominal_qi=nominal_qi,
            nominal_qs=nominal_qs,
            nominal_ha=nominal_ha,
            qi_change=qi_change,
            qs_change=qs_change,
            ha_change=ha_change,
        )

    if int(n_tests) != ROBUSTNESS_TOTAL_EPISODES:
        raise ValueError(
            f"{POLYMER_ROBUSTNESS_TRAINING_PROFILE!r} requires n_tests="
            f"{ROBUSTNESS_TOTAL_EPISODES}; received {n_tests}. Select the legacy profile "
            "for arbitrary one-off episode counts."
        )
    if int(set_points_len) != ROBUSTNESS_SETPOINT_HOLD_STEPS:
        raise ValueError(
            f"{POLYMER_ROBUSTNESS_TRAINING_PROFILE!r} requires set_points_len="
            f"{ROBUSTNESS_SETPOINT_HOLD_STEPS}; received {set_points_len}."
        )

    phase1 = _legacy_episode_bundle(
        y_sp_scenario=y_sp_scenario,
        n_tests=ROBUSTNESS_PHASE1_EPISODES,
        set_points_len=set_points_len,
        warm_start=warm_start,
        test_cycle=test_cycle,
        nominal_qi=nominal_qi,
        nominal_qs=nominal_qs,
        nominal_ha=nominal_ha,
        qi_change=qi_change,
        qs_change=qs_change,
        ha_change=ha_change,
    )
    phase1_steps = int(phase1["nFE"])
    episode_steps = int(phase1["time_in_sub_episodes"])
    phase2_steps = ROBUSTNESS_PHASE2_EPISODES * episode_steps

    phase2_scaled = _scale_physical_setpoints(
        ROBUSTNESS_PHASE2_SETPOINTS_PHYS,
        steady_outputs=steady_outputs,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
    )
    y_phase2 = _repeat_setpoint_cycle(
        phase2_scaled,
        n_episodes=ROBUSTNESS_PHASE2_EPISODES,
        set_points_len=set_points_len,
    )

    qi_phase2 = np.linspace(float(phase1["qi"][-1]), ROBUSTNESS_PHASE2_QI_FINAL, phase2_steps)
    qs_phase2 = np.linspace(float(phase1["qs"][-1]), ROBUSTNESS_PHASE2_QS_FINAL, phase2_steps)
    ha_phase2 = np.full(phase2_steps, ROBUSTNESS_FOULED_HA, dtype=float)
    bookkeeping = _episode_bookkeeping(
        ROBUSTNESS_TOTAL_EPISODES,
        episode_steps,
        warm_start,
        test_cycle,
    )
    total_steps = int(bookkeeping["nFE"])

    phase_windows = [
        {
            "name": "legacy_training",
            "phase_id": 1,
            "episode_start": 1,
            "episode_end": ROBUSTNESS_PHASE1_EPISODES,
            "step_start": 0,
            "step_end_exclusive": phase1_steps,
            "learning_enabled": True,
        },
        {
            "name": "robustness_adaptation",
            "phase_id": 2,
            "episode_start": ROBUSTNESS_PHASE1_EPISODES + 1,
            "episode_end": ROBUSTNESS_TOTAL_EPISODES,
            "step_start": phase1_steps,
            "step_end_exclusive": total_steps,
            "learning_enabled": True,
            "final_episode_evaluation_only": True,
        },
    ]
    return {
        "y_sp": np.vstack([phase1["y_sp"], y_phase2]),
        "nFE": total_steps,
        "sub_episodes_changes_dict": bookkeeping["sub_episodes_changes_dict"],
        "sub_episode_changes_dict": bookkeeping["sub_episodes_changes_dict"],
        "time_in_sub_episodes": episode_steps,
        "test_train_dict": bookkeeping["test_train_dict"],
        "warm_start_step": bookkeeping["warm_start_step"],
        "qi": np.concatenate([phase1["qi"], qi_phase2]),
        "qs": np.concatenate([phase1["qs"], qs_phase2]),
        "ha": np.concatenate([phase1["ha"], ha_phase2]),
        "training_profile_name": POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        "experiment_phase_windows": deepcopy(phase_windows),
        "phase_switch_step": phase1_steps,
        "phase_switch_episode": ROBUSTNESS_PHASE1_EPISODES + 1,
        "exploration_freeze_step": phase1_steps,
        "exploration_freeze_settings": {
            "enabled": True,
            "freeze_amplitude_only": True,
            "start_step": phase1_steps,
            "start_episode": ROBUSTNESS_PHASE1_EPISODES + 1,
            "last_training_episode": ROBUSTNESS_TOTAL_EPISODES - 1,
            "evaluation_episode": ROBUSTNESS_TOTAL_EPISODES,
        },
        "fouling_active": True,
        "fouled_ha": float(ROBUSTNESS_FOULED_HA),
        "phase2_learning_enabled": True,
        "phase2_final_episode_evaluation_only": True,
        "phase2_episode_status": {
            "learning_episode_start": ROBUSTNESS_PHASE1_EPISODES + 1,
            "learning_episode_end": ROBUSTNESS_TOTAL_EPISODES - 1,
            "evaluation_only_episodes": [ROBUSTNESS_TOTAL_EPISODES],
        },
        "phase1_setpoints_phys": np.asarray(
            [[4.5, 324.0], [3.4, 321.0]], dtype=float
        ),
        "phase2_setpoints_phys": ROBUSTNESS_PHASE2_SETPOINTS_PHYS.copy(),
        "phase1_disturbance_end": {
            "qi": float(phase1["qi"][-1]),
            "qs": float(phase1["qs"][-1]),
            "ha": float(phase1["ha"][-1]),
        },
        "phase2_disturbance_end": {
            "qi": float(qi_phase2[-1]),
            "qs": float(qs_phase2[-1]),
            "ha": float(ha_phase2[-1]),
        },
    }


def polymer_profile_result_fields(profile_bundle: dict) -> dict:
    keys = (
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
    )
    return {key: deepcopy(profile_bundle[key]) for key in keys if key in profile_bundle}
