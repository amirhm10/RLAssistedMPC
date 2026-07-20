from __future__ import annotations

from copy import deepcopy


def maybe_freeze_agent_exploration(agent, *, environment_step: int, freeze_step: int | None):
    """Freeze an agent's exploration annealing once at an environment-step boundary."""

    if agent is None or freeze_step is None or int(environment_step) < int(freeze_step):
        return None
    existing = getattr(agent, "exploration_freeze_info", None)
    if existing is not None:
        return deepcopy(existing)
    freeze_fn = getattr(agent, "freeze_exploration", None)
    if not callable(freeze_fn):
        return None
    info = dict(freeze_fn() or {})
    info["environment_step"] = int(environment_step)
    agent.exploration_freeze_info = deepcopy(info)
    return info


def effective_agent_exploration_value(agent, *, test: bool = False) -> float:
    if agent is None or bool(test):
        return 0.0
    resolver = getattr(agent, "effective_exploration_schedule_value", None)
    if callable(resolver):
        return float(resolver(eval_mode=False))
    return float(getattr(agent, "last_exploration_value", 0.0))


def exploration_freeze_result_fields(agent, effective_log) -> dict:
    return {
        "exploration_freeze_info": deepcopy(getattr(agent, "exploration_freeze_info", None)),
        "effective_exploration_step_log": effective_log,
    }
