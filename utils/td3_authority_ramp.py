from __future__ import annotations

import numpy as np


def resolve_td3_authority_ramp(config, *, step_idx: int, warm_start_step: int, time_in_sub_episodes: int):
    """Resolve controlled live-authority status and scalar cap for a TD3 action."""
    cfg = config if isinstance(config, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    step_idx = int(step_idx)
    warm_start_step = int(warm_start_step)
    time_in_sub = int(max(1, time_in_sub_episodes))
    live_enabled = bool(enabled and step_idx > warm_start_step)
    if not live_enabled:
        return {
            "enabled": enabled,
            "live_enabled": False,
            "post_warm_episode": 0,
            "progress": 0.0,
            "cap": float("nan"),
        }

    post_warm_step = max(0, step_idx - warm_start_step - 1)
    post_warm_episode = int(post_warm_step // time_in_sub) + 1
    protected = int(max(0, cfg.get("protected_subepisodes", 0)))
    ramp = int(max(0, cfg.get("ramp_subepisodes", cfg.get("authority_ramp_subepisodes", 1))))
    start_cap = float(cfg.get("start_cap", cfg.get("initial_cap", cfg.get("protected_cap", 0.0))))
    end_cap = float(cfg.get("end_cap", cfg.get("final_cap", cfg.get("full_cap", start_cap))))

    if post_warm_episode <= protected:
        progress = 0.0
        cap = start_cap
    else:
        ramp_episode = max(1, post_warm_episode - protected)
        if ramp <= 1:
            progress = 1.0
        else:
            progress = float(np.clip((ramp_episode - 1) / float(ramp - 1), 0.0, 1.0))
        cap = start_cap + progress * (end_cap - start_cap)

    return {
        "enabled": enabled,
        "live_enabled": True,
        "post_warm_episode": int(post_warm_episode),
        "progress": float(progress),
        "cap": float(max(0.0, cap)),
    }


def apply_symmetric_deviation_cap(value, *, center, cap, low=None, high=None, active=True, tol=1.0e-12):
    """Clip value to center +/- cap, then optional absolute low/high bounds."""
    value = np.asarray(value, float).reshape(-1)
    center = np.asarray(center, float).reshape(-1)
    if value.shape != center.shape:
        raise ValueError("value and center must have the same shape for authority-ramp clipping.")

    if not bool(active) or not np.isfinite(cap):
        clipped = value.copy()
    else:
        cap = float(max(0.0, cap))
        clipped = np.clip(value, center - cap, center + cap)

    if low is not None:
        clipped = np.maximum(clipped, np.asarray(low, float).reshape(-1))
    if high is not None:
        clipped = np.minimum(clipped, np.asarray(high, float).reshape(-1))

    return clipped, {
        "projection_active": bool(np.any(np.abs(clipped - value) > float(tol))),
        "delta_norm": float(np.linalg.norm(clipped - value)),
        "value_norm": float(np.linalg.norm(value)),
        "clipped_norm": float(np.linalg.norm(clipped)),
    }


def init_td3_authority_ramp_logs(n_steps: int, action_dim: int):
    n_steps = int(max(0, n_steps))
    action_dim = int(max(1, action_dim))
    return {
        "td3_authority_ramp_live_log": np.zeros(n_steps, dtype=int),
        "td3_authority_ramp_gate_override_log": np.zeros(n_steps, dtype=int),
        "td3_authority_ramp_cap_log": np.full(n_steps, np.nan, dtype=float),
        "td3_authority_ramp_progress_log": np.full(n_steps, np.nan, dtype=float),
        "td3_authority_ramp_projection_active_log": np.zeros(n_steps, dtype=int),
        "td3_authority_ramp_delta_norm_log": np.full(n_steps, np.nan, dtype=float),
        "td3_authority_ramp_preclip_action_log": np.zeros((n_steps, action_dim), dtype=float),
        "td3_authority_ramp_postclip_action_log": np.zeros((n_steps, action_dim), dtype=float),
    }


def record_td3_authority_ramp_step(
    logs,
    *,
    step_idx: int,
    ramp_info,
    preclip_action,
    postclip_action,
    projection_active=False,
    delta_norm=np.nan,
    gate_override=False,
):
    step_idx = int(step_idx)
    preclip_action = np.asarray(preclip_action, float).reshape(-1)
    postclip_action = np.asarray(postclip_action, float).reshape(-1)
    logs["td3_authority_ramp_live_log"][step_idx] = int(bool(ramp_info.get("live_enabled", False)))
    logs["td3_authority_ramp_gate_override_log"][step_idx] = int(bool(gate_override))
    logs["td3_authority_ramp_cap_log"][step_idx] = float(ramp_info.get("cap", np.nan))
    logs["td3_authority_ramp_progress_log"][step_idx] = float(ramp_info.get("progress", np.nan))
    logs["td3_authority_ramp_projection_active_log"][step_idx] = int(bool(projection_active))
    logs["td3_authority_ramp_delta_norm_log"][step_idx] = float(delta_norm)
    logs["td3_authority_ramp_preclip_action_log"][step_idx, :] = preclip_action
    logs["td3_authority_ramp_postclip_action_log"][step_idx, :] = postclip_action


def build_td3_authority_ramp_bundle_fields(config, logs, *, prefix=""):
    prefix = str(prefix)
    return {
        f"{prefix}td3_authority_ramp": dict(config if isinstance(config, dict) else {}),
        f"{prefix}td3_authority_ramp_enabled": bool(
            config.get("enabled", False) if isinstance(config, dict) else False
        ),
        **{f"{prefix}{key}": value for key, value in logs.items()},
    }


__all__ = [
    "apply_symmetric_deviation_cap",
    "build_td3_authority_ramp_bundle_fields",
    "init_td3_authority_ramp_logs",
    "record_td3_authority_ramp_step",
    "resolve_td3_authority_ramp",
]
