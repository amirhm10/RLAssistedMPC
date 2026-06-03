from dataclasses import dataclass

import numpy as np

from DQN.supervisor_gated_dqn_agent import (
    SOURCE_HELD,
    SOURCE_NAMES as DISCRETE_SUPERVISOR_SOURCE_NAMES,
    SOURCE_SUPERVISOR,
    SOURCE_WARM_START,
)
from utils.phase1_hidden_release import (
    ACTION_SOURCE_HELD_INTERVAL,
    ACTION_SOURCE_PHASE1_HIDDEN_BASELINE,
    ACTION_SOURCE_POLICY_EVAL_LIVE,
    ACTION_SOURCE_POLICY_TRAIN_LIVE,
    ACTION_SOURCE_WARM_START_BASELINE,
    record_phase1_train_step,
    resolve_phase1_action_source,
)


@dataclass
class HorizonStepDecision:
    action: int
    last_action: int | None
    decision_taken: int
    source: int


@dataclass
class SupervisorGatedHorizonStepDecision:
    action: int
    last_action: int | None
    decision_taken: int
    source: int
    policy_action: int
    supervisor_action: int
    previous_action: int
    selected_source: int
    score_policy: float
    score_supervisor: float
    advantage_policy_supervisor: float
    q_policy: float
    q_supervisor: float


@dataclass
class ContinuousStepDecision:
    action: np.ndarray
    last_action: np.ndarray | None
    last_action_test: bool | None
    decision_taken: int
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
    post_warm_action_freeze_steps: int = 0,
) -> HorizonStepDecision:
    """Select a discrete horizon action using the single-agent DQN semantics."""
    warm_start_step = int(warm_start_step)
    step = int(step)
    decision_interval = int(max(1, decision_interval))
    post_warm_action_freeze_steps = int(max(0, post_warm_action_freeze_steps))
    action_freeze_end_step = warm_start_step + post_warm_action_freeze_steps
    if step <= warm_start_step:
        return HorizonStepDecision(
            action=int(default_action),
            last_action=last_action,
            decision_taken=0,
            source=ACTION_SOURCE_WARM_START_BASELINE,
        )
    if post_warm_action_freeze_steps > 0 and step <= action_freeze_end_step:
        return HorizonStepDecision(
            action=int(default_action),
            last_action=None,
            decision_taken=0,
            source=ACTION_SOURCE_PHASE1_HIDDEN_BASELINE,
        )

    if (step % decision_interval == 0) or (last_action is None):
        state_f32 = np.asarray(state, np.float32)
        if test:
            action = int(agent.act_eval(state_f32))
            source = ACTION_SOURCE_POLICY_EVAL_LIVE
        else:
            action = int(agent.take_action(state_f32, eval_mode=False))
            source = ACTION_SOURCE_POLICY_TRAIN_LIVE
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
        source=ACTION_SOURCE_HELD_INTERVAL,
    )


def select_supervisor_gated_horizon_action(
    *,
    agent,
    state,
    step: int,
    warm_start_step: int,
    decision_interval: int,
    default_action: int,
    last_action: int | None,
    test: bool,
    post_warm_action_freeze_steps: int = 0,
    supervisor_action: int | None = None,
) -> SupervisorGatedHorizonStepDecision:
    """Select a discrete horizon action with an SG-DQN gate against a supervisor."""
    warm_start_step = int(warm_start_step)
    step = int(step)
    decision_interval = int(max(1, decision_interval))
    post_warm_action_freeze_steps = int(max(0, post_warm_action_freeze_steps))
    action_freeze_end_step = warm_start_step + post_warm_action_freeze_steps
    supervisor_idx = int(default_action if supervisor_action is None else supervisor_action)
    previous_idx = int(supervisor_idx if last_action is None else last_action)

    if step <= warm_start_step:
        return SupervisorGatedHorizonStepDecision(
            action=int(supervisor_idx),
            last_action=last_action,
            decision_taken=0,
            source=ACTION_SOURCE_WARM_START_BASELINE,
            policy_action=int(supervisor_idx),
            supervisor_action=int(supervisor_idx),
            previous_action=int(previous_idx),
            selected_source=SOURCE_WARM_START,
            score_policy=float("nan"),
            score_supervisor=float("nan"),
            advantage_policy_supervisor=float("nan"),
            q_policy=float("nan"),
            q_supervisor=float("nan"),
        )
    if post_warm_action_freeze_steps > 0 and step <= action_freeze_end_step:
        return SupervisorGatedHorizonStepDecision(
            action=int(supervisor_idx),
            last_action=None,
            decision_taken=0,
            source=ACTION_SOURCE_PHASE1_HIDDEN_BASELINE,
            policy_action=int(supervisor_idx),
            supervisor_action=int(supervisor_idx),
            previous_action=int(previous_idx),
            selected_source=SOURCE_SUPERVISOR,
            score_policy=float("nan"),
            score_supervisor=float("nan"),
            advantage_policy_supervisor=float("nan"),
            q_policy=float("nan"),
            q_supervisor=float("nan"),
        )

    if (step % decision_interval == 0) or (last_action is None):
        sg_decision = agent.select_action_with_supervisor(
            np.asarray(state, np.float32),
            supervisor_action=int(supervisor_idx),
            previous_action=int(previous_idx),
            explore=not bool(test),
            test=bool(test),
        )
        source = ACTION_SOURCE_POLICY_EVAL_LIVE if test else ACTION_SOURCE_POLICY_TRAIN_LIVE
        return SupervisorGatedHorizonStepDecision(
            action=int(sg_decision.action),
            last_action=int(sg_decision.action),
            decision_taken=1,
            source=source,
            policy_action=int(sg_decision.policy_action),
            supervisor_action=int(sg_decision.supervisor_action),
            previous_action=int(previous_idx),
            selected_source=int(sg_decision.selected_source),
            score_policy=float(sg_decision.score_policy),
            score_supervisor=float(sg_decision.score_supervisor),
            advantage_policy_supervisor=float(sg_decision.advantage_policy_supervisor),
            q_policy=float(sg_decision.q_policy),
            q_supervisor=float(sg_decision.q_supervisor),
        )

    return SupervisorGatedHorizonStepDecision(
        action=int(last_action),
        last_action=last_action,
        decision_taken=0,
        source=ACTION_SOURCE_HELD_INTERVAL,
        policy_action=int(last_action),
        supervisor_action=int(supervisor_idx),
        previous_action=int(previous_idx),
        selected_source=SOURCE_HELD,
        score_policy=float("nan"),
        score_supervisor=float("nan"),
        advantage_policy_supervisor=float("nan"),
        q_policy=float("nan"),
        q_supervisor=float("nan"),
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


def replay_train_supervisor_gated_horizon_agent(
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
    decision: SupervisorGatedHorizonStepDecision,
) -> dict:
    """Push/train an SG-DQN horizon agent using executed-action replay."""
    pushed = False
    trained = False
    train_meta = None
    if not test:
        if step > replay_start_step:
            agent.push_supervised(
                np.asarray(state, np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, np.float32),
                float(done),
                policy_action=int(decision.policy_action),
                supervisor_action=int(decision.supervisor_action),
                previous_action=int(decision.previous_action),
                selected_source=int(decision.selected_source),
                score_policy=float(decision.score_policy),
                score_supervisor=float(decision.score_supervisor),
                advantage_policy_supervisor=float(decision.advantage_policy_supervisor),
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
    decision_interval: int = 1,
    last_action=None,
    last_action_test: bool | None = None,
    first_live_action_step: int | None = None,
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

    decision_interval = int(max(1, decision_interval))
    first_live = (
        int(first_live_action_step)
        if first_live_action_step is not None
        else int(phase1.get("first_live_action_step", int(warm_start_step) + 1))
        if phase1 is not None
        else int(warm_start_step) + 1
    )
    last = None if last_action is None else np.asarray(last_action, float).reshape(-1)
    decision_taken = 0
    held_action_available = (
        last is not None
        and last.size == baseline.size
        and np.all(np.isfinite(last))
        and last_action_test is not None
        and bool(last_action_test) == bool(test)
    )

    if step > warm_start_step:
        live_offset = max(0, int(step) - first_live)
        should_decide = (live_offset % decision_interval == 0) or not held_action_available
        if hidden_active and phase1 is not None:
            policy_action = np.asarray(agent.act_eval(state), float).reshape(-1)
            if not np.all(np.isfinite(policy_action)):
                policy_action = baseline.copy()
        if hidden_active:
            action = baseline.copy()
            last = None
            last_action_test = None
        elif not test:
            if should_decide:
                if phase1 is not None:
                    policy_action = np.asarray(agent.act_eval(state), float).reshape(-1)
                    if not np.all(np.isfinite(policy_action)):
                        policy_action = baseline.copy()
                action = np.asarray(agent.take_action(state, explore=True), float).reshape(-1)
                decision_taken = 1
                last = action.copy()
                last_action_test = False
            else:
                action = last.copy()
        else:
            if should_decide:
                if policy_action is None:
                    policy_action = np.asarray(agent.act_eval(state), float).reshape(-1)
                    if not np.all(np.isfinite(policy_action)):
                        policy_action = baseline.copy()
                action = (
                    policy_action.copy()
                    if policy_action is not None
                    else np.asarray(agent.act_eval(state), float).reshape(-1)
                )
                decision_taken = 1
                last = action.copy()
                last_action_test = True
            else:
                action = last.copy()
    else:
        action = baseline.copy()
        last = None
        last_action_test = None
        if phase1 is not None:
            policy_action = baseline.copy()

    if action_dim is not None and action.size != int(action_dim):
        raise ValueError(f"Continuous action has size {action.size}, expected {int(action_dim)}.")
    nonfinite_fallback_used = False
    if nonfinite_fallback and not np.all(np.isfinite(action)):
        action = baseline.copy()
        last = None
        last_action_test = None
        decision_taken = 0
        nonfinite_fallback_used = True

    source = resolve_phase1_action_source(step, warm_start_step, hidden_active, test)
    if source in {2, 3} and not decision_taken:
        source = ACTION_SOURCE_HELD_INTERVAL
    return ContinuousStepDecision(
        action=np.asarray(action, float).reshape(-1),
        last_action=None if last is None else np.asarray(last, float).reshape(-1),
        last_action_test=last_action_test,
        decision_taken=int(decision_taken),
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
