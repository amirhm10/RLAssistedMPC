import numpy as np


HORIZON_SAFETY_REASON_CODES = {
    "accepted": 0,
    "warm_default": 1,
    "release_projection": 2,
    "cooldown_default": 3,
    "invalid_request_default": 4,
}


def _pair(action_idx, horizon_recipes, default_action):
    try:
        idx = int(action_idx)
    except (TypeError, ValueError):
        idx = int(default_action)
    if idx < 0 or idx >= len(horizon_recipes):
        idx = int(default_action)
    return idx, tuple(int(v) for v in horizon_recipes[idx])


def _window_bounds(config, post_warm_episode):
    release_filter = dict(config.get("release_filter", {}) or {})
    if not bool(config.get("enabled", False)) or not bool(release_filter.get("enabled", False)):
        return None
    if int(post_warm_episode) <= 0:
        return None

    protected = dict(release_filter.get("protected", {}) or {})
    protected_subepisodes = int(max(0, protected.get("subepisodes", 0)))
    if int(post_warm_episode) <= protected_subepisodes:
        return {
            "predict_min": int(protected.get("predict_min", 4)),
            "predict_max": int(protected.get("predict_max", 12)),
            "control_min": int(protected.get("control_min", 2)),
            "control_max": int(protected.get("control_max", 6)),
        }

    ramp = dict(release_filter.get("ramp", {}) or {})
    ramp_end = int(max(protected_subepisodes, ramp.get("end_subepisode", protected_subepisodes)))
    if int(post_warm_episode) <= ramp_end:
        return {
            "predict_min": int(ramp.get("predict_min", 3)),
            "predict_max": int(ramp.get("predict_max", 18)),
            "control_min": int(ramp.get("control_min", 1)),
            "control_max": int(ramp.get("control_max", 10)),
        }
    return None


def _allowed_actions(horizon_recipes, bounds):
    if bounds is None:
        return tuple(range(len(horizon_recipes)))
    allowed = []
    for idx, recipe in enumerate(horizon_recipes):
        hp, hc = (int(recipe[0]), int(recipe[1]))
        if (
            bounds["predict_min"] <= hp <= bounds["predict_max"]
            and bounds["control_min"] <= hc <= bounds["control_max"]
        ):
            allowed.append(idx)
    return tuple(allowed)


def _nearest_allowed_action(horizon_recipes, requested_pair, allowed_actions, default_action):
    if not allowed_actions:
        return int(default_action)
    default_pair = tuple(int(v) for v in horizon_recipes[int(default_action)])
    req_hp, req_hc = requested_pair
    scored = []
    for idx in allowed_actions:
        hp, hc = (int(horizon_recipes[idx][0]), int(horizon_recipes[idx][1]))
        requested_dist = abs(hp - req_hp) + abs(hc - req_hc)
        default_dist = abs(hp - default_pair[0]) + abs(hc - default_pair[1])
        scored.append((requested_dist, default_dist, int(idx)))
    return int(min(scored)[2])


def resolve_horizon_safety(
    config,
    *,
    step_idx,
    warm_start_step,
    time_in_sub_episodes,
    requested_action,
    horizon_recipes,
    default_action,
    cooldown_until_subepisode=0,
):
    cfg = dict(config or {})
    recipes = [tuple(int(v) for v in recipe) for recipe in horizon_recipes]
    default_action = int(default_action)
    step_idx = int(step_idx)
    warm_start_step = int(warm_start_step)
    time_in_sub = int(max(1, time_in_sub_episodes))
    current_subepisode = int(step_idx // time_in_sub) + 1
    post_warm_episode = 0 if step_idx <= warm_start_step else int((step_idx - warm_start_step - 1) // time_in_sub) + 1

    requested_idx, requested_pair = _pair(requested_action, recipes, default_action)
    requested_invalid = int(requested_action) != requested_idx if isinstance(requested_action, (int, np.integer)) else False
    cooldown_active = bool(
        cfg.get("enabled", False)
        and post_warm_episode > 0
        and current_subepisode <= int(cooldown_until_subepisode)
    )
    bounds = _window_bounds(cfg, post_warm_episode)
    allowed_actions = _allowed_actions(recipes, bounds)

    reason = "accepted"
    if step_idx <= warm_start_step:
        executed_idx = default_action
        reason = "warm_default"
    elif cooldown_active:
        executed_idx = default_action
        reason = "cooldown_default"
    elif requested_invalid:
        executed_idx = default_action
        reason = "invalid_request_default"
    elif requested_idx not in allowed_actions:
        executed_idx = _nearest_allowed_action(recipes, requested_pair, allowed_actions, default_action)
        reason = "release_projection"
    else:
        executed_idx = requested_idx

    executed_pair = tuple(int(v) for v in recipes[int(executed_idx)])
    return {
        "requested_action": int(requested_idx),
        "executed_action": int(executed_idx),
        "requested_pair": requested_pair,
        "executed_pair": executed_pair,
        "reason": reason,
        "reason_code": int(HORIZON_SAFETY_REASON_CODES[reason]),
        "projection_active": bool(int(executed_idx) != int(requested_idx)),
        "cooldown_active": bool(cooldown_active),
        "post_warm_episode": int(post_warm_episode),
        "current_subepisode": int(current_subepisode),
        "allowed_action_count": int(len(allowed_actions)),
        "allowed_predict_min": float("nan") if bounds is None else float(bounds["predict_min"]),
        "allowed_predict_max": float("nan") if bounds is None else float(bounds["predict_max"]),
        "allowed_control_min": float("nan") if bounds is None else float(bounds["control_min"]),
        "allowed_control_max": float("nan") if bounds is None else float(bounds["control_max"]),
    }


def init_horizon_safety_logs(n_steps):
    n_steps = int(max(0, n_steps))
    return {
        "horizon_requested_action_log": np.full(n_steps, -1, dtype=int),
        "horizon_executed_action_log": np.full(n_steps, -1, dtype=int),
        "horizon_requested_trace_log": np.zeros((n_steps, 2), dtype=int),
        "horizon_executed_trace_log": np.zeros((n_steps, 2), dtype=int),
        "horizon_safety_reason_log": np.zeros(n_steps, dtype=int),
        "horizon_projection_active_log": np.zeros(n_steps, dtype=int),
        "horizon_cooldown_active_log": np.zeros(n_steps, dtype=int),
        "horizon_post_warm_episode_log": np.zeros(n_steps, dtype=int),
        "horizon_allowed_action_count_log": np.zeros(n_steps, dtype=int),
        "horizon_allowed_predict_min_log": np.full(n_steps, np.nan, dtype=float),
        "horizon_allowed_predict_max_log": np.full(n_steps, np.nan, dtype=float),
        "horizon_allowed_control_min_log": np.full(n_steps, np.nan, dtype=float),
        "horizon_allowed_control_max_log": np.full(n_steps, np.nan, dtype=float),
    }


def record_horizon_safety_step(logs, *, step_idx, safety_info):
    step_idx = int(step_idx)
    logs["horizon_requested_action_log"][step_idx] = int(safety_info["requested_action"])
    logs["horizon_executed_action_log"][step_idx] = int(safety_info["executed_action"])
    logs["horizon_requested_trace_log"][step_idx, :] = np.asarray(safety_info["requested_pair"], int)
    logs["horizon_executed_trace_log"][step_idx, :] = np.asarray(safety_info["executed_pair"], int)
    logs["horizon_safety_reason_log"][step_idx] = int(safety_info["reason_code"])
    logs["horizon_projection_active_log"][step_idx] = int(bool(safety_info["projection_active"]))
    logs["horizon_cooldown_active_log"][step_idx] = int(bool(safety_info["cooldown_active"]))
    logs["horizon_post_warm_episode_log"][step_idx] = int(safety_info["post_warm_episode"])
    logs["horizon_allowed_action_count_log"][step_idx] = int(safety_info["allowed_action_count"])
    logs["horizon_allowed_predict_min_log"][step_idx] = float(safety_info["allowed_predict_min"])
    logs["horizon_allowed_predict_max_log"][step_idx] = float(safety_info["allowed_predict_max"])
    logs["horizon_allowed_control_min_log"][step_idx] = float(safety_info["allowed_control_min"])
    logs["horizon_allowed_control_max_log"][step_idx] = float(safety_info["allowed_control_max"])


def build_horizon_safety_bundle_fields(config, logs, *, prefix=""):
    prefix = str(prefix)
    return {
        f"{prefix}horizon_safety": dict(config or {}),
        f"{prefix}horizon_safety_enabled": bool((config or {}).get("enabled", False)),
        f"{prefix}horizon_safety_reason_codes": dict(HORIZON_SAFETY_REASON_CODES),
        **{f"{prefix}{key}": value for key, value in logs.items()},
    }


__all__ = [
    "HORIZON_SAFETY_REASON_CODES",
    "build_horizon_safety_bundle_fields",
    "init_horizon_safety_logs",
    "record_horizon_safety_step",
    "resolve_horizon_safety",
]
