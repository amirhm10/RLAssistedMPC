from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normalized_exponential_tail(progress: float, sharpness: float = 5.0) -> float:
    progress = float(min(1.0, max(0.0, progress)))
    denom = 1.0 - math.exp(-sharpness)
    if denom <= 0.0:
        return 1.0 - progress
    num = math.exp(-sharpness * progress) - math.exp(-sharpness)
    return float(max(0.0, num / denom))


def build_behavioral_cloning_schedule(
    *,
    config: dict[str, Any] | None,
    warm_start_step: int,
    time_in_sub_episodes: int,
    n_steps: int,
) -> dict[str, Any]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    target_mode = str(cfg.get("target_mode", "nominal_only")).strip().lower()
    lambda_bc_start = float(cfg.get("lambda_bc_start", 0.0))
    lambda_bc_end = float(cfg.get("lambda_bc_end", 0.0))
    decay_mode = str(cfg.get("decay_mode", "exp")).strip().lower()
    active_subepisodes = int(max(0, cfg.get("active_subepisodes", 0)))
    start_after_warm_start = bool(cfg.get("start_after_warm_start", True))
    log_diagnostics = bool(cfg.get("log_diagnostics", True))

    if target_mode != "nominal_only":
        raise ValueError("behavioral_cloning target_mode must be 'nominal_only' for this implementation.")
    if decay_mode not in {"constant", "linear", "exp"}:
        raise ValueError("behavioral_cloning decay_mode must be 'constant', 'linear', or 'exp'.")

    n_steps = int(max(0, n_steps))
    time_in_sub_episodes = int(max(1, time_in_sub_episodes))
    active_steps = int(active_subepisodes * time_in_sub_episodes)
    start_step = int(warm_start_step) + 1 if start_after_warm_start else 0
    end_step = start_step + active_steps - 1
    active_enabled = bool(enabled and active_steps > 0 and lambda_bc_start > 0.0)

    active_log = np.zeros(n_steps, dtype=int)
    if active_enabled and n_steps > 0:
        lo = max(0, start_step)
        hi = min(end_step, n_steps - 1)
        if hi >= lo:
            active_log[lo : hi + 1] = 1

    return {
        "enabled": bool(active_enabled),
        "target_mode": target_mode,
        "lambda_bc_start": float(lambda_bc_start),
        "lambda_bc_end": float(lambda_bc_end),
        "decay_mode": decay_mode,
        "active_subepisodes": int(active_subepisodes),
        "start_after_warm_start": bool(start_after_warm_start),
        "log_diagnostics": bool(log_diagnostics),
        "active_steps": int(active_steps),
        "start_step": int(start_step),
        "end_step": int(end_step if active_enabled else start_step - 1),
        "active_log": active_log,
    }


def resolve_behavioral_cloning_context(
    schedule: dict[str, Any],
    *,
    step_idx: int,
    nominal_target_action,
) -> dict[str, Any] | None:
    if not schedule.get("enabled", False):
        return None

    step_idx = int(step_idx)
    if step_idx < int(schedule["start_step"]) or step_idx > int(schedule["end_step"]):
        return None

    active_steps = int(schedule["active_steps"])
    if active_steps <= 1:
        progress = 1.0
    else:
        progress = float(step_idx - int(schedule["start_step"])) / float(active_steps - 1)
    progress = min(1.0, max(0.0, progress))

    start = float(schedule["lambda_bc_start"])
    end = float(schedule["lambda_bc_end"])
    decay_mode = str(schedule["decay_mode"]).lower()
    if decay_mode == "constant":
        weight = start
    elif decay_mode == "linear":
        weight = start + (end - start) * progress
    else:
        tail = _normalized_exponential_tail(progress)
        weight = end + (start - end) * tail

    return {
        "active": True,
        "weight": float(max(0.0, weight)),
        "target_mode": str(schedule["target_mode"]),
        "target_action": np.asarray(nominal_target_action, float).reshape(-1),
        "progress": float(progress),
    }


def init_behavioral_cloning_logs(n_steps: int):
    n_steps = int(max(0, n_steps))
    return {
        "bc_active_log": np.zeros(n_steps, dtype=int),
        "bc_weight_log": np.zeros(n_steps, dtype=float),
        "bc_loss_log": np.full(n_steps, np.nan, dtype=float),
        "bc_actor_target_distance_log": np.full(n_steps, np.nan, dtype=float),
        "bc_policy_nominal_distance_log": np.zeros(n_steps, dtype=float),
    }


def record_behavioral_cloning_step(logs, *, step_idx: int, bc_context, policy_action, nominal_target_action, train_meta):
    step_idx = int(step_idx)
    nominal = np.asarray(nominal_target_action, float).reshape(-1)
    policy = np.asarray(policy_action, float).reshape(-1)
    logs["bc_policy_nominal_distance_log"][step_idx] = float(np.linalg.norm(policy - nominal))
    if bc_context is None:
        return
    logs["bc_active_log"][step_idx] = int(bool(bc_context.get("active", False)))
    logs["bc_weight_log"][step_idx] = float(bc_context.get("weight", 0.0))
    if isinstance(train_meta, dict):
        bc_loss = train_meta.get("bc_loss")
        bc_distance = train_meta.get("bc_actor_target_distance")
        if bc_loss is not None:
            logs["bc_loss_log"][step_idx] = float(bc_loss)
        if bc_distance is not None:
            logs["bc_actor_target_distance_log"][step_idx] = float(bc_distance)


def build_behavioral_cloning_bundle_fields(schedule, logs):
    return {
        "behavioral_cloning": dict(schedule),
        "behavioral_cloning_enabled": bool(schedule.get("enabled", False)),
        "bc_active_log": np.asarray(logs["bc_active_log"], int),
        "bc_weight_log": np.asarray(logs["bc_weight_log"], float),
        "bc_loss_log": np.asarray(logs["bc_loss_log"], float),
        "bc_actor_target_distance_log": np.asarray(logs["bc_actor_target_distance_log"], float),
        "bc_policy_nominal_distance_log": np.asarray(logs["bc_policy_nominal_distance_log"], float),
    }


__all__ = [
    "build_behavioral_cloning_bundle_fields",
    "build_behavioral_cloning_schedule",
    "init_behavioral_cloning_logs",
    "record_behavioral_cloning_step",
    "resolve_behavioral_cloning_context",
]
