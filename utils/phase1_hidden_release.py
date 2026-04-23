from __future__ import annotations

from typing import Any, Dict

import numpy as np


ACTION_SOURCE_WARM_START_BASELINE = 0
ACTION_SOURCE_PHASE1_HIDDEN_BASELINE = 1
ACTION_SOURCE_POLICY_TRAIN_LIVE = 2
ACTION_SOURCE_POLICY_EVAL_LIVE = 3


def build_test_mode_log(test_train_dict, n_steps):
    n_steps = int(max(0, n_steps))
    log = np.zeros(n_steps, dtype=bool)
    current = False
    for step in range(n_steps):
        if step in test_train_dict:
            current = bool(test_train_dict[step])
        log[step] = current
    return log


def build_phase1_schedule(
    *,
    agent_kind,
    warm_start_step,
    time_in_sub_episodes,
    n_steps,
    test_train_dict,
    action_freeze_subepisodes,
    actor_freeze_subepisodes,
    batch_size,
    initial_buffer_size=0,
    base_actor_freeze=0,
    push_start_step=0,
    train_start_step=0,
):
    n_steps = int(max(0, n_steps))
    warm_start_step = int(warm_start_step)
    time_in_sub_episodes = int(max(1, time_in_sub_episodes))
    action_freeze_subepisodes = int(max(0, action_freeze_subepisodes))
    actor_freeze_subepisodes = int(max(0, actor_freeze_subepisodes))
    batch_size = int(max(1, batch_size))
    initial_buffer_size = int(max(0, initial_buffer_size))
    base_actor_freeze = int(max(0, base_actor_freeze))
    push_start_step = int(push_start_step)
    train_start_step = int(train_start_step)

    enabled = (
        str(agent_kind).strip().lower() == "td3"
        and action_freeze_subepisodes > 0
        and actor_freeze_subepisodes > 0
    )

    action_freeze_start_step = warm_start_step + 1
    action_freeze_end_step = (
        warm_start_step + action_freeze_subepisodes * time_in_sub_episodes if enabled else warm_start_step
    )
    actor_freeze_end_step = (
        warm_start_step + actor_freeze_subepisodes * time_in_sub_episodes if enabled else warm_start_step
    )
    first_live_action_step = action_freeze_end_step + 1

    hidden_window_active_log = np.zeros(n_steps, dtype=int)
    if enabled and n_steps > 0:
        active_start = max(0, action_freeze_start_step)
        active_end = min(action_freeze_end_step, n_steps - 1)
        if active_end >= active_start:
            hidden_window_active_log[active_start : active_end + 1] = 1

    test_mode_log = build_test_mode_log(test_train_dict, n_steps)
    buffer_size = initial_buffer_size
    actor_freeze_train_steps = 0
    for step in range(n_steps):
        if not test_mode_log[step] and step >= push_start_step:
            buffer_size += 1
        critic_update_happens = (not test_mode_log[step]) and step >= train_start_step and buffer_size >= batch_size
        if critic_update_happens and enabled and step <= actor_freeze_end_step:
            actor_freeze_train_steps += 1

    return {
        "enabled": bool(enabled),
        "action_freeze_subepisodes": int(action_freeze_subepisodes),
        "actor_freeze_subepisodes": int(actor_freeze_subepisodes),
        "action_freeze_start_step": int(action_freeze_start_step),
        "action_freeze_end_step": int(action_freeze_end_step),
        "actor_freeze_end_step": int(actor_freeze_end_step),
        "first_live_action_step": int(first_live_action_step),
        "actor_freeze_train_steps": int(actor_freeze_train_steps),
        "effective_actor_freeze": int(max(base_actor_freeze, actor_freeze_train_steps)),
        "hidden_window_active_log": hidden_window_active_log,
        "test_mode_log": test_mode_log.astype(int),
    }


def resolve_phase1_action_source(step_idx, warm_start_step, hidden_window_active, is_test):
    step_idx = int(step_idx)
    warm_start_step = int(warm_start_step)
    if step_idx <= warm_start_step:
        return ACTION_SOURCE_WARM_START_BASELINE
    if hidden_window_active:
        return ACTION_SOURCE_PHASE1_HIDDEN_BASELINE
    if is_test:
        return ACTION_SOURCE_POLICY_EVAL_LIVE
    return ACTION_SOURCE_POLICY_TRAIN_LIVE


def record_phase1_train_step(traces, env_step, train_meta):
    if train_meta is None:
        return
    env_step = int(env_step)
    if bool(train_meta.get("critic_updated", False)):
        traces["critic_update_env_step_trace"].append(env_step)
    if bool(train_meta.get("actor_slot", False)):
        traces["actor_update_slot_env_step_trace"].append(env_step)
        if bool(train_meta.get("actor_updated", False)):
            traces["actor_update_applied_env_step_trace"].append(env_step)
        else:
            traces["actor_update_blocked_env_step_trace"].append(env_step)


def build_phase1_bundle_fields(
    phase1,
    *,
    policy_action_raw_log,
    executed_action_raw_log,
    action_source_log,
    traces,
    prefix="",
):
    prefix = str(prefix)
    return {
        f"{prefix}phase1_enabled": bool(phase1["enabled"]),
        f"{prefix}phase1_action_freeze_subepisodes": int(phase1["action_freeze_subepisodes"]),
        f"{prefix}phase1_actor_freeze_subepisodes": int(phase1["actor_freeze_subepisodes"]),
        f"{prefix}phase1_action_freeze_start_step": int(phase1["action_freeze_start_step"]),
        f"{prefix}phase1_action_freeze_end_step": int(phase1["action_freeze_end_step"]),
        f"{prefix}phase1_first_live_action_step": int(phase1["first_live_action_step"]),
        f"{prefix}phase1_actor_freeze_train_steps": int(phase1["actor_freeze_train_steps"]),
        f"{prefix}phase1_effective_actor_freeze": int(phase1["effective_actor_freeze"]),
        f"{prefix}phase1_hidden_window_active_log": np.asarray(phase1["hidden_window_active_log"], int),
        f"{prefix}phase1_action_source_log": np.asarray(action_source_log, int),
        f"{prefix}policy_action_raw_log": np.asarray(policy_action_raw_log, float),
        f"{prefix}executed_action_raw_log": np.asarray(executed_action_raw_log, float),
        f"{prefix}critic_update_env_step_trace": np.asarray(traces["critic_update_env_step_trace"], int),
        f"{prefix}actor_update_slot_env_step_trace": np.asarray(traces["actor_update_slot_env_step_trace"], int),
        f"{prefix}actor_update_applied_env_step_trace": np.asarray(traces["actor_update_applied_env_step_trace"], int),
        f"{prefix}actor_update_blocked_env_step_trace": np.asarray(traces["actor_update_blocked_env_step_trace"], int),
    }


def init_phase1_train_traces() -> Dict[str, list[int]]:
    return {
        "critic_update_env_step_trace": [],
        "actor_update_slot_env_step_trace": [],
        "actor_update_applied_env_step_trace": [],
        "actor_update_blocked_env_step_trace": [],
    }


__all__ = [
    "ACTION_SOURCE_PHASE1_HIDDEN_BASELINE",
    "ACTION_SOURCE_POLICY_EVAL_LIVE",
    "ACTION_SOURCE_POLICY_TRAIN_LIVE",
    "ACTION_SOURCE_WARM_START_BASELINE",
    "build_phase1_bundle_fields",
    "build_phase1_schedule",
    "build_test_mode_log",
    "init_phase1_train_traces",
    "record_phase1_train_step",
    "resolve_phase1_action_source",
]
