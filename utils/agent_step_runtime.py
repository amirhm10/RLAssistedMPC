from dataclasses import dataclass

import numpy as np

from utils.phase1_hidden_release import record_phase1_train_step, resolve_phase1_action_source


@dataclass
class HorizonStepDecision:
    action: int
    last_action: int | None
    decision_taken: int
    source: int


@dataclass
class ContinuousStepDecision:
    action: np.ndarray
    policy_action: np.ndarray | None
    source: int
    phase1_hidden_active: bool
    nonfinite_fallback_used: bool = False


def select_horizon_action(
    *,
    agent,
    state,
    step: int,
    warm_start_step: int,
    decision_interval: int,
    default_action: int,
    last_action: int | None,
    test: bool,
) -> HorizonStepDecision:
    """Select a discrete horizon action using the single-agent DQN semantics."""
    if step <= warm_start_step:
        return HorizonStepDecision(
            action=int(default_action),
            last_action=last_action,
            decision_taken=0,
            source=0,
        )

    if (step % int(decision_interval) == 0) or (last_action is None):
        state_f32 = np.asarray(state, np.float32)
        if test:
            action = int(agent.act_eval(state_f32))
            source = 3
        else:
            action = int(agent.take_action(state_f32, eval_mode=False))
            source = 2
        return HorizonStepDecision(
            action=action,
            last_action=action,
            decision_taken=1,
            source=source,
        )

    return HorizonStepDecision(
        action=int(last_action),
        last_action=last_action,
        decision_taken=0,
        source=4,
    )


def replay_train_horizon_agent(
    *,
    agent,
    state,
    action: int,
    reward: float,
    next_state,
    done: float,
    step: int,
    test: bool,
    replay_start_step: int,
    train_start_step: int,
) -> dict:
    """Push/train a discrete horizon agent using the single-agent horizon gates."""
    pushed = False
    trained = False
    train_meta = None
    if not test:
        if step > replay_start_step:
            agent.push(
                np.asarray(state, np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, np.float32),
                float(done),
            )
            pushed = True
        if step >= train_start_step:
            train_meta = agent.train_step()
            trained = True
    return {"pushed": pushed, "trained": trained, "train_meta": train_meta}


def select_continuous_action(
    *,
    agent,
    state,
    step: int,
    warm_start_step: int,
    test: bool,
    baseline_action,
    phase1=None,
    action_dim: int | None = None,
    nonfinite_fallback: bool = False,
) -> ContinuousStepDecision:
    """Select a TD3/SAC action using the single-agent continuous semantics."""
    baseline = np.asarray(baseline_action, float).reshape(-1)
    hidden_active = bool(
        phase1 is not None
        and phase1.get("enabled", False)
        and bool(phase1["hidden_window_active_log"][step])
    )
    policy_action = None

    if step > warm_start_step:
        if phase1 is not None:
            policy_action = np.asarray(agent.act_eval(state), float).reshape(-1)
            if not np.all(np.isfinite(policy_action)):
                policy_action = baseline.copy()
        if hidden_active:
            action = baseline.copy()
        elif not test:
            action = np.asarray(agent.take_action(state, explore=True), float).reshape(-1)
        else:
            action = (
                policy_action.copy()
                if policy_action is not None
                else np.asarray(agent.act_eval(state), float).reshape(-1)
            )
    else:
        action = baseline.copy()
        if phase1 is not None:
            policy_action = baseline.copy()

    if action_dim is not None and action.size != int(action_dim):
        raise ValueError(f"Continuous action has size {action.size}, expected {int(action_dim)}.")
    nonfinite_fallback_used = False
    if nonfinite_fallback and not np.all(np.isfinite(action)):
        action = baseline.copy()
        nonfinite_fallback_used = True

    source = resolve_phase1_action_source(step, warm_start_step, hidden_active, test)
    return ContinuousStepDecision(
        action=np.asarray(action, float).reshape(-1),
        policy_action=policy_action,
        source=int(source),
        phase1_hidden_active=hidden_active,
        nonfinite_fallback_used=nonfinite_fallback_used,
    )


def replay_train_continuous_agent(
    *,
    agent,
    state,
    action,
    reward: float,
    next_state,
    done: float,
    step: int,
    test: bool,
    train_start_step: int,
    phase1_train_traces=None,
    bc_context=None,
) -> dict:
    """Push/train a TD3/SAC agent using the single-agent continuous gates."""
    pushed = False
    trained = False
    train_meta = None
    if not test:
        agent.push(
            np.asarray(state, np.float32),
            np.asarray(action, np.float32),
            float(reward),
            np.asarray(next_state, np.float32),
            float(done),
        )
        pushed = True
        if step >= train_start_step:
            train_meta = agent.train_step(bc_context=bc_context)
            trained = True
            if phase1_train_traces is not None:
                record_phase1_train_step(phase1_train_traces, step, train_meta)
    return {"pushed": pushed, "trained": trained, "train_meta": train_meta}
