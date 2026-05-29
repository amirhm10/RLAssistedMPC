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
    release_gate_cfg = cfg.get("release_gate")
    handoff_cfg = cfg.get("handoff")

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

    if release_gate_cfg is None:
        release_gate = {
            "enabled": False,
            "diagnostic_only": False,
            "window_subepisodes": 1,
            "mean_action_gap_max": 0.25,
            "max_coordinate_gap_max": 0.20,
            "min_window_fraction": 1.0,
        }
    else:
        if not isinstance(release_gate_cfg, dict):
            raise ValueError("behavioral_cloning release_gate must be a dict when provided.")
        release_gate = {
            "enabled": bool(release_gate_cfg.get("enabled", False)),
            "diagnostic_only": bool(release_gate_cfg.get("diagnostic_only", False)),
            "window_subepisodes": int(max(1, release_gate_cfg.get("window_subepisodes", 1))),
            "mean_action_gap_max": float(release_gate_cfg.get("mean_action_gap_max", 0.25)),
            "max_coordinate_gap_max": float(release_gate_cfg.get("max_coordinate_gap_max", 0.20)),
            "min_window_fraction": float(release_gate_cfg.get("min_window_fraction", 1.0)),
        }
        if (
            not math.isfinite(release_gate["mean_action_gap_max"])
            or release_gate["mean_action_gap_max"] < 0.0
        ):
            raise ValueError("behavioral_cloning release_gate mean_action_gap_max must be finite and non-negative.")
        if (
            not math.isfinite(release_gate["max_coordinate_gap_max"])
            or release_gate["max_coordinate_gap_max"] < 0.0
        ):
            raise ValueError("behavioral_cloning release_gate max_coordinate_gap_max must be finite and non-negative.")
        if (
            not math.isfinite(release_gate["min_window_fraction"])
            or release_gate["min_window_fraction"] <= 0.0
            or release_gate["min_window_fraction"] > 1.0
        ):
            raise ValueError("behavioral_cloning release_gate min_window_fraction must be in (0, 1].")

    if handoff_cfg is None:
        handoff = {
            "enabled": False,
            "mode": "raw_action_blend",
            "start_authority": 1.0,
            "end_authority": 1.0,
            "active_subepisodes": 0,
        }
    else:
        if not isinstance(handoff_cfg, dict):
            raise ValueError("behavioral_cloning handoff must be a dict when provided.")
        handoff = {
            "enabled": bool(handoff_cfg.get("enabled", False)),
            "mode": str(handoff_cfg.get("mode", "raw_action_blend")).strip().lower(),
            "start_authority": float(handoff_cfg.get("start_authority", 1.0)),
            "end_authority": float(handoff_cfg.get("end_authority", 1.0)),
            "active_subepisodes": int(max(0, handoff_cfg.get("active_subepisodes", active_subepisodes))),
        }
    if handoff["mode"] != "raw_action_blend":
        raise ValueError("behavioral_cloning handoff mode must be 'raw_action_blend'.")
    for key in ("start_authority", "end_authority"):
        if not math.isfinite(handoff[key]) or handoff[key] < 0.0 or handoff[key] > 1.0:
            raise ValueError(f"behavioral_cloning handoff {key} must be in [0, 1].")

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
        "configured_enabled": bool(enabled),
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
        "release_gate": dict(release_gate),
        "release_gate_window_steps": int(release_gate["window_subepisodes"] * time_in_sub_episodes),
        "handoff": {
            **dict(handoff),
            "start_step": int(start_step),
            "time_in_sub_episodes": int(time_in_sub_episodes),
            "active_steps": int(handoff["active_subepisodes"] * time_in_sub_episodes),
        },
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


def resolve_bc_handoff_authority(schedule, *, step_idx: int) -> dict[str, Any]:
    """Return the BC-handoff authority for the current environment step."""
    handoff = dict(schedule.get("handoff", {}) or {})
    enabled = bool(handoff.get("enabled", False))
    start = float(handoff.get("start_authority", 1.0))
    end = float(handoff.get("end_authority", 1.0))
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "authority": 1.0,
            "progress": 1.0,
            "handoff_episode": 0,
        }

    step_idx = int(step_idx)
    start_step = int(handoff.get("start_step", schedule.get("start_step", 0)))
    time_in_sub = int(max(1, handoff.get("time_in_sub_episodes", 1)))
    active_subepisodes = int(max(1, handoff.get("active_subepisodes", 1)))
    if step_idx < start_step:
        return {
            "enabled": True,
            "active": False,
            "authority": 0.0,
            "progress": 0.0,
            "handoff_episode": 0,
        }
    rel_step = max(0, step_idx - start_step)
    episode = int(rel_step // time_in_sub) + 1

    if episode >= active_subepisodes:
        progress = 1.0
    else:
        progress = 0.0 if active_subepisodes <= 1 else float((episode - 1) / float(active_subepisodes - 1))
    authority = start + progress * (end - start)
    if step_idx >= start_step + active_subepisodes * time_in_sub:
        authority = end
        progress = 1.0

    return {
        "enabled": True,
        "active": bool(step_idx < start_step + active_subepisodes * time_in_sub),
        "authority": float(np.clip(authority, 0.0, 1.0)),
        "progress": float(np.clip(progress, 0.0, 1.0)),
        "handoff_episode": int(min(max(episode, 1), active_subepisodes)),
    }


def apply_bc_handoff_action(td3_action, safe_action, authority):
    """Blend from safe action to TD3 raw action and clip to the actor range."""
    td3 = np.asarray(td3_action, float).reshape(-1)
    safe = np.asarray(safe_action, float).reshape(-1)
    if td3.shape != safe.shape:
        raise ValueError("td3_action and safe_action must have the same shape for BC handoff.")
    alpha = float(np.clip(authority, 0.0, 1.0))
    return np.clip(safe + alpha * (td3 - safe), -1.0, 1.0)


def init_bc_handoff_logs(n_steps: int, action_dim: int):
    n_steps = int(max(0, n_steps))
    action_dim = int(max(1, action_dim))
    return {
        "bc_handoff_authority_log": np.ones(n_steps, dtype=float),
        "bc_handoff_safe_action_log": np.zeros((n_steps, action_dim), dtype=float),
        "bc_handoff_td3_action_log": np.zeros((n_steps, action_dim), dtype=float),
        "bc_handoff_executed_action_log": np.zeros((n_steps, action_dim), dtype=float),
    }


def record_bc_handoff_step(logs, *, step_idx: int, authority_info, safe_action, td3_action, executed_action):
    step_idx = int(step_idx)
    logs["bc_handoff_authority_log"][step_idx] = float(authority_info.get("authority", 1.0))
    logs["bc_handoff_safe_action_log"][step_idx, :] = np.asarray(safe_action, float).reshape(-1)
    logs["bc_handoff_td3_action_log"][step_idx, :] = np.asarray(td3_action, float).reshape(-1)
    logs["bc_handoff_executed_action_log"][step_idx, :] = np.asarray(executed_action, float).reshape(-1)


def build_bc_handoff_bundle_fields(schedule, logs, *, prefix=""):
    prefix = str(prefix)
    return {
        f"{prefix}bc_handoff": dict(schedule.get("handoff", {}) or {}),
        f"{prefix}bc_handoff_enabled": bool(dict(schedule.get("handoff", {}) or {}).get("enabled", False)),
        **{f"{prefix}{key}": value for key, value in logs.items()},
    }


def init_protected_bc_release_gate(schedule, n_steps: int):
    n_steps = int(max(0, n_steps))
    release_gate = dict(schedule.get("release_gate", {}) or {})
    window_steps = int(max(1, schedule.get("release_gate_window_steps", 1)))
    bc_configured = bool(schedule.get("configured_enabled", schedule.get("enabled", False)))
    enabled = bool(bc_configured and release_gate.get("enabled", False))
    diagnostic_only = bool(release_gate.get("diagnostic_only", False))
    return {
        "state": {
            "enabled": enabled,
            "diagnostic_only": diagnostic_only,
            "live_blocking_enabled": bool(enabled and not diagnostic_only),
            "released": not enabled,
            "release_step": -1,
            "window_steps": window_steps,
            "mean_action_gap_max": float(release_gate.get("mean_action_gap_max", 0.25)),
            "max_coordinate_gap_max": float(release_gate.get("max_coordinate_gap_max", 0.20)),
            "min_window_fraction": float(release_gate.get("min_window_fraction", 1.0)),
        },
        "logs": {
            "release_gate_action_gap_norm_log": np.full(n_steps, np.nan, dtype=float),
            "release_gate_max_coordinate_gap_log": np.full(n_steps, np.nan, dtype=float),
            "release_gate_rolling_mean_gap_log": np.full(n_steps, np.nan, dtype=float),
            "release_gate_rolling_max_coordinate_gap_log": np.full(n_steps, np.nan, dtype=float),
            "release_gate_window_count_log": np.zeros(n_steps, dtype=int),
            "release_gate_pass_log": np.zeros(n_steps, dtype=int),
            "release_gate_released_log": np.zeros(n_steps, dtype=int),
            "release_gate_blocked_log": np.zeros(n_steps, dtype=int),
            "release_gate_release_step_log": np.full(n_steps, -1, dtype=int),
        },
    }


def update_protected_bc_release_gate(
    gate_state,
    gate_logs,
    *,
    step_idx: int,
    warm_start_step: int,
    policy_action,
    target_action,
):
    step_idx = int(step_idx)
    warm_start_step = int(warm_start_step)
    policy = np.asarray(policy_action, float).reshape(-1)
    target = np.asarray(target_action, float).reshape(-1)
    if policy.shape != target.shape:
        raise ValueError(
            f"release-gate policy action size {policy.size} does not match target action size {target.size}."
        )
    gap_vec = policy - target
    gap_norm = float(np.linalg.norm(gap_vec))
    max_coord_gap = float(np.max(np.abs(gap_vec))) if gap_vec.size else 0.0
    gate_logs["release_gate_action_gap_norm_log"][step_idx] = gap_norm
    gate_logs["release_gate_max_coordinate_gap_log"][step_idx] = max_coord_gap

    enabled = bool(gate_state.get("enabled", False))
    if not enabled:
        gate_logs["release_gate_released_log"][step_idx] = int(step_idx > warm_start_step)
        gate_logs["release_gate_release_step_log"][step_idx] = int(max(0, warm_start_step + 1))
        return {
            "enabled": False,
            "diagnostic_only": False,
            "live_blocking_enabled": False,
            "released": bool(step_idx > warm_start_step),
            "blocked": False,
            "passed": bool(step_idx > warm_start_step),
            "live_released": bool(step_idx > warm_start_step),
            "live_blocked": False,
            "gap_norm": gap_norm,
            "max_coordinate_gap": max_coord_gap,
            "rolling_mean_gap": gap_norm,
            "rolling_max_coordinate_gap": max_coord_gap,
            "window_count": 1,
            "release_step": int(max(0, warm_start_step + 1)),
        }

    window_steps = int(max(1, gate_state.get("window_steps", 1)))
    diagnostic_only = bool(gate_state.get("diagnostic_only", False))
    start = max(0, step_idx - window_steps + 1)
    norm_window = np.asarray(gate_logs["release_gate_action_gap_norm_log"][start : step_idx + 1], float)
    coord_window = np.asarray(gate_logs["release_gate_max_coordinate_gap_log"][start : step_idx + 1], float)
    mask = np.isfinite(norm_window) & np.isfinite(coord_window)
    count = int(np.sum(mask))
    min_count = int(math.ceil(float(gate_state.get("min_window_fraction", 1.0)) * window_steps))
    rolling_mean = float(np.mean(norm_window[mask])) if count else float("nan")
    rolling_max_coord = float(np.max(coord_window[mask])) if count else float("nan")
    passed = bool(
        step_idx > warm_start_step
        and count >= min_count
        and np.isfinite(rolling_mean)
        and np.isfinite(rolling_max_coord)
        and rolling_mean <= float(gate_state["mean_action_gap_max"])
        and rolling_max_coord <= float(gate_state["max_coordinate_gap_max"])
    )
    if passed and not bool(gate_state.get("released", False)):
        gate_state["released"] = True
        gate_state["release_step"] = step_idx
    released = bool(gate_state.get("released", False) and step_idx > warm_start_step)
    blocked = bool(step_idx > warm_start_step and not released)

    gate_logs["release_gate_rolling_mean_gap_log"][step_idx] = rolling_mean
    gate_logs["release_gate_rolling_max_coordinate_gap_log"][step_idx] = rolling_max_coord
    gate_logs["release_gate_window_count_log"][step_idx] = count
    gate_logs["release_gate_pass_log"][step_idx] = int(passed)
    gate_logs["release_gate_released_log"][step_idx] = int(released)
    gate_logs["release_gate_blocked_log"][step_idx] = int(blocked)
    gate_logs["release_gate_release_step_log"][step_idx] = int(gate_state.get("release_step", -1))
    live_released = bool(step_idx > warm_start_step) if diagnostic_only else released
    live_blocked = False if diagnostic_only else blocked
    return {
        "enabled": True,
        "diagnostic_only": diagnostic_only,
        "live_blocking_enabled": bool(not diagnostic_only),
        "released": released,
        "blocked": blocked,
        "passed": passed,
        "live_released": live_released,
        "live_blocked": live_blocked,
        "gap_norm": gap_norm,
        "max_coordinate_gap": max_coord_gap,
        "rolling_mean_gap": rolling_mean,
        "rolling_max_coordinate_gap": rolling_max_coord,
        "window_count": count,
        "release_step": int(gate_state.get("release_step", -1)),
    }


def build_protected_bc_release_gate_bundle_fields(gate_bundle, *, prefix=""):
    prefix = str(prefix)
    if not gate_bundle:
        return {}
    state = dict(gate_bundle.get("state", {}) or {})
    logs = dict(gate_bundle.get("logs", {}) or {})
    return {
        f"{prefix}protected_bc_release_gate": dict(state),
        f"{prefix}protected_bc_release_gate_enabled": bool(state.get("enabled", False)),
        f"{prefix}protected_bc_release_gate_diagnostic_only": bool(state.get("diagnostic_only", False)),
        f"{prefix}protected_bc_release_gate_live_blocking_enabled": bool(
            state.get("live_blocking_enabled", state.get("enabled", False))
        ),
        f"{prefix}protected_bc_release_gate_release_step": int(state.get("release_step", -1)),
        f"{prefix}release_gate_action_gap_norm_log": np.asarray(
            logs.get("release_gate_action_gap_norm_log", []), float
        ),
        f"{prefix}release_gate_max_coordinate_gap_log": np.asarray(
            logs.get("release_gate_max_coordinate_gap_log", []), float
        ),
        f"{prefix}release_gate_rolling_mean_gap_log": np.asarray(
            logs.get("release_gate_rolling_mean_gap_log", []), float
        ),
        f"{prefix}release_gate_rolling_max_coordinate_gap_log": np.asarray(
            logs.get("release_gate_rolling_max_coordinate_gap_log", []), float
        ),
        f"{prefix}release_gate_window_count_log": np.asarray(
            logs.get("release_gate_window_count_log", []), int
        ),
        f"{prefix}release_gate_pass_log": np.asarray(logs.get("release_gate_pass_log", []), int),
        f"{prefix}release_gate_released_log": np.asarray(logs.get("release_gate_released_log", []), int),
        f"{prefix}release_gate_blocked_log": np.asarray(logs.get("release_gate_blocked_log", []), int),
        f"{prefix}release_gate_release_step_log": np.asarray(
            logs.get("release_gate_release_step_log", []), int
        ),
    }


__all__ = [
    "apply_bc_handoff_action",
    "build_bc_handoff_bundle_fields",
    "build_protected_bc_release_gate_bundle_fields",
    "build_behavioral_cloning_bundle_fields",
    "build_behavioral_cloning_schedule",
    "init_bc_handoff_logs",
    "init_behavioral_cloning_logs",
    "init_protected_bc_release_gate",
    "record_bc_handoff_step",
    "record_behavioral_cloning_step",
    "resolve_bc_handoff_authority",
    "resolve_behavioral_cloning_context",
    "update_protected_bc_release_gate",
]
