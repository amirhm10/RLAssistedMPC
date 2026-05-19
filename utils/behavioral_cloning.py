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
    start_step_override: int | None = None,
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
    coordinate_weights_cfg = cfg.get("coordinate_weights")
    label_weight_overrides_cfg = cfg.get("label_weight_overrides")
    action_gap_tolerance = float(cfg.get("action_gap_tolerance", 0.0))
    tail_anchor_cfg = cfg.get("tail_anchor")

    if target_mode not in {"nominal_only", "executed_action", "ls_action"}:
        raise ValueError(
            "behavioral_cloning target_mode must be 'nominal_only', 'executed_action', or 'ls_action'."
        )
    if decay_mode not in {"constant", "linear", "exp"}:
        raise ValueError("behavioral_cloning decay_mode must be 'constant', 'linear', or 'exp'.")
    if not math.isfinite(action_gap_tolerance) or action_gap_tolerance < 0.0:
        raise ValueError("behavioral_cloning action_gap_tolerance must be finite and non-negative.")
    if coordinate_weights_cfg is not None:
        coordinate_weights_cfg = np.asarray(coordinate_weights_cfg, float).reshape(-1)
        if coordinate_weights_cfg.size == 0:
            raise ValueError("behavioral_cloning coordinate_weights must not be empty when provided.")
        if not np.all(np.isfinite(coordinate_weights_cfg)) or np.any(coordinate_weights_cfg < 0.0):
            raise ValueError("behavioral_cloning coordinate_weights must be finite and non-negative.")
    if label_weight_overrides_cfg is None:
        label_weight_overrides = {}
    else:
        if not isinstance(label_weight_overrides_cfg, dict):
            raise ValueError("behavioral_cloning label_weight_overrides must be a dict when provided.")
        label_weight_overrides = {}
        for key, value in label_weight_overrides_cfg.items():
            label = str(key)
            weight = float(value)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("behavioral_cloning label_weight_overrides values must be finite and non-negative.")
            label_weight_overrides[label] = weight

    if tail_anchor_cfg is None:
        tail_anchor = {
            "enabled": False,
            "weight": 0.0,
            "action_gap_tolerance": 0.0,
            "require_ls_target": True,
            "activate_on_score_deficit": True,
            "activate_on_negative_requested_score": False,
            "start_after_main_window": True,
        }
    else:
        if not isinstance(tail_anchor_cfg, dict):
            raise ValueError("behavioral_cloning tail_anchor must be a dict when provided.")
        tail_anchor = {
            "enabled": bool(tail_anchor_cfg.get("enabled", False)),
            "weight": float(tail_anchor_cfg.get("weight", 0.0)),
            "action_gap_tolerance": float(tail_anchor_cfg.get("action_gap_tolerance", action_gap_tolerance)),
            "require_ls_target": bool(tail_anchor_cfg.get("require_ls_target", True)),
            "activate_on_score_deficit": bool(tail_anchor_cfg.get("activate_on_score_deficit", True)),
            "activate_on_negative_requested_score": bool(
                tail_anchor_cfg.get("activate_on_negative_requested_score", False)
            ),
            "start_after_main_window": bool(tail_anchor_cfg.get("start_after_main_window", True)),
        }
        if not math.isfinite(tail_anchor["weight"]) or tail_anchor["weight"] < 0.0:
            raise ValueError("behavioral_cloning tail_anchor weight must be finite and non-negative.")
        if not math.isfinite(tail_anchor["action_gap_tolerance"]) or tail_anchor["action_gap_tolerance"] < 0.0:
            raise ValueError(
                "behavioral_cloning tail_anchor action_gap_tolerance must be finite and non-negative."
            )

    n_steps = int(max(0, n_steps))
    time_in_sub_episodes = int(max(1, time_in_sub_episodes))
    active_steps = int(active_subepisodes * time_in_sub_episodes)
    if start_step_override is None:
        start_step = int(warm_start_step) + 1 if start_after_warm_start else 0
    else:
        start_step = int(start_step_override)
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
        "coordinate_weights": None if coordinate_weights_cfg is None else np.asarray(coordinate_weights_cfg, float).copy(),
        "label_weight_overrides": dict(label_weight_overrides),
        "action_gap_tolerance": float(action_gap_tolerance),
        "active_steps": int(active_steps),
        "start_step": int(start_step),
        "end_step": int(end_step if active_enabled else start_step - 1),
        "active_log": active_log,
        "tail_anchor": dict(tail_anchor),
    }


def _resolve_coordinate_weights(schedule: dict[str, Any], *, target_action, action_labels=None) -> np.ndarray:
    target_action = np.asarray(target_action, float).reshape(-1)
    weight_vec = np.ones_like(target_action, dtype=float)

    coordinate_weights_cfg = schedule.get("coordinate_weights")
    if coordinate_weights_cfg is not None:
        coord = np.asarray(coordinate_weights_cfg, float).reshape(-1)
        if coord.size != target_action.size:
            raise ValueError(
                f"behavioral_cloning coordinate_weights size {coord.size} does not match action size {target_action.size}."
            )
        weight_vec = coord.copy()

    label_weight_overrides = dict(schedule.get("label_weight_overrides", {}) or {})
    if label_weight_overrides:
        if action_labels is None:
            raise ValueError("behavioral_cloning label_weight_overrides require action_labels.")
        labels = tuple(str(label) for label in action_labels)
        if len(labels) != target_action.size:
            raise ValueError(
                f"behavioral_cloning action_labels length {len(labels)} does not match action size {target_action.size}."
            )
        for label, weight in label_weight_overrides.items():
            if label not in labels:
                raise KeyError(f"behavioral_cloning label_weight_overrides label not found: {label}")
            weight_vec[labels.index(label)] = float(weight)

    if not np.all(np.isfinite(weight_vec)) or np.any(weight_vec < 0.0):
        raise ValueError("Resolved behavioral-cloning coordinate weights must be finite and non-negative.")
    return weight_vec


def resolve_behavioral_cloning_context(
    schedule: dict[str, Any],
    *,
    step_idx: int,
    target_action=None,
    nominal_target_action=None,
    action_labels=None,
    policy_action=None,
    tail_meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if target_action is None:
        if nominal_target_action is None:
            raise ValueError("behavioral_cloning requires target_action or nominal_target_action.")
        target_action = nominal_target_action

    step_idx = int(step_idx)
    if schedule.get("enabled", False) and step_idx >= int(schedule["start_step"]) and step_idx <= int(schedule["end_step"]):
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
            "target_action": np.asarray(target_action, float).reshape(-1),
            "coordinate_weights": _resolve_coordinate_weights(
                schedule,
                target_action=target_action,
                action_labels=action_labels,
            ),
            "progress": float(progress),
            "phase": "main_window",
        }

    tail_anchor = dict(schedule.get("tail_anchor", {}) or {})
    if not bool(tail_anchor.get("enabled", False)):
        return None
    if bool(tail_anchor.get("start_after_main_window", True)) and step_idx <= int(schedule["end_step"]):
        return None
    if policy_action is None:
        return None

    tail_meta = dict(tail_meta or {})
    if bool(tail_anchor.get("require_ls_target", True)) and not bool(tail_meta.get("target_is_ls", False)):
        return None

    requested_score = tail_meta.get("requested_score")
    ls_score = tail_meta.get("ls_score")
    if bool(tail_anchor.get("activate_on_score_deficit", True)):
        if requested_score is None or ls_score is None:
            return None
        if (not np.isfinite(float(requested_score))) or (not np.isfinite(float(ls_score))):
            return None
        if not (float(requested_score) < float(ls_score)):
            return None
    if bool(tail_anchor.get("activate_on_negative_requested_score", False)):
        if requested_score is None or (not np.isfinite(float(requested_score))) or float(requested_score) >= 0.0:
            return None

    policy = np.asarray(policy_action, float).reshape(-1)
    target = np.asarray(target_action, float).reshape(-1)
    if policy.shape != target.shape:
        raise ValueError(
            f"behavioral_cloning policy_action size {policy.size} does not match target_action size {target.size}."
        )
    gap = float(np.linalg.norm(policy - target))
    if gap <= float(tail_anchor.get("action_gap_tolerance", 0.0)):
        return None

    return {
        "active": True,
        "weight": float(max(0.0, tail_anchor.get("weight", 0.0))),
        "target_mode": str(schedule["target_mode"]),
        "target_action": target,
        "coordinate_weights": _resolve_coordinate_weights(
            schedule,
            target_action=target_action,
            action_labels=action_labels,
        ),
        "progress": 1.0,
        "phase": "tail_anchor",
        "tail_gap": gap,
    }


def init_behavioral_cloning_logs(n_steps: int):
    n_steps = int(max(0, n_steps))
    return {
        "bc_active_log": np.zeros(n_steps, dtype=int),
        "bc_weight_log": np.zeros(n_steps, dtype=float),
        "bc_loss_log": np.full(n_steps, np.nan, dtype=float),
        "bc_actor_target_distance_log": np.full(n_steps, np.nan, dtype=float),
        "bc_policy_target_distance_log": np.zeros(n_steps, dtype=float),
        "bc_policy_nominal_distance_log": np.zeros(n_steps, dtype=float),
    }


def record_behavioral_cloning_step(
    logs,
    *,
    step_idx: int,
    bc_context,
    policy_action,
    target_action=None,
    nominal_target_action=None,
    target_mode=None,
    train_meta,
):
    step_idx = int(step_idx)
    if target_action is None:
        if nominal_target_action is None:
            raise ValueError("behavioral_cloning requires target_action or nominal_target_action.")
        target_action = nominal_target_action
    policy = np.asarray(policy_action, float).reshape(-1)
    target = np.asarray(target_action, float).reshape(-1)
    current_target_mode = (
        str(bc_context.get("target_mode"))
        if isinstance(bc_context, dict) and bc_context.get("target_mode") is not None
        else str(target_mode if target_mode is not None else "nominal_only")
    ).strip().lower()
    target_distance = float(np.linalg.norm(policy - target))
    logs["bc_policy_target_distance_log"][step_idx] = target_distance
    if current_target_mode == "nominal_only":
        logs["bc_policy_nominal_distance_log"][step_idx] = target_distance
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
        "bc_policy_target_distance_log": np.asarray(logs["bc_policy_target_distance_log"], float),
        "bc_policy_nominal_distance_log": np.asarray(logs["bc_policy_nominal_distance_log"], float),
    }


__all__ = [
    "build_behavioral_cloning_bundle_fields",
    "build_behavioral_cloning_schedule",
    "init_behavioral_cloning_logs",
    "record_behavioral_cloning_step",
    "resolve_behavioral_cloning_context",
]
