from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from DQN.dqn_agent import DQNAgent
from DQN.replay_buffer import PERRecentReplayBuffer
from utils.noisy_layers import mean_module_abs_sigma


SOURCE_WARM_START = 0
SOURCE_SUPERVISOR = 1
SOURCE_POLICY = 2
SOURCE_HELD = 3
SOURCE_FALLBACK = 4

SOURCE_NAMES = {
    SOURCE_WARM_START: "warm_start",
    SOURCE_SUPERVISOR: "supervisor",
    SOURCE_POLICY: "policy",
    SOURCE_HELD: "held",
    SOURCE_FALLBACK: "fallback",
}


@dataclass
class DiscreteSupervisorGateConfig:
    advantage_margin: float = 0.0
    default_to_supervisor: bool = True
    min_train_steps_before_policy_gate: int = 0


@dataclass
class DiscreteSupervisorGatedDecision:
    action: int
    policy_action: int
    supervisor_action: int
    selected_source: int
    score_policy: float
    score_supervisor: float
    advantage_policy_supervisor: float
    q_policy: float
    q_supervisor: float


def _validate_action_index(action, action_dim: int, name: str) -> int:
    try:
        idx = int(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer action index.") from exc
    if idx < 0 or idx >= int(action_dim):
        raise ValueError(f"{name}={idx} is outside [0, {int(action_dim) - 1}].")
    return idx


class DiscreteSupervisorPERRecentReplayBuffer(PERRecentReplayBuffer):
    """PER/recent replay with scalar discrete supervisor-gate metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_actions = np.full((self.capacity,), -1, dtype=np.int64)
        self.supervisor_actions = np.full((self.capacity,), -1, dtype=np.int64)
        self.previous_actions = np.full((self.capacity,), -1, dtype=np.int64)
        self.selected_sources = np.zeros((self.capacity,), dtype=np.int32)
        self.score_policy = np.full((self.capacity,), np.nan, dtype=np.float32)
        self.score_supervisor = np.full((self.capacity,), np.nan, dtype=np.float32)
        self.advantage_policy_supervisor = np.full((self.capacity,), np.nan, dtype=np.float32)

    def push_supervised(
        self,
        s,
        executed_action,
        r,
        ns,
        done,
        p0=None,
        discount_n=None,
        n_actual=1,
        behavior_prob=None,
        behavior_logprob=None,
        policy_action=None,
        supervisor_action=None,
        previous_action=None,
        selected_source=SOURCE_SUPERVISOR,
        score_policy=np.nan,
        score_supervisor=np.nan,
        advantage_policy_supervisor=np.nan,
    ):
        idx = int(self.ptr)
        executed_idx = int(executed_action)
        policy_idx = executed_idx if policy_action is None else int(policy_action)
        supervisor_idx = executed_idx if supervisor_action is None else int(supervisor_action)
        previous_idx = executed_idx if previous_action is None else int(previous_action)

        super().push(
            s,
            executed_idx,
            r,
            ns,
            done,
            p0=p0,
            discount_n=discount_n,
            n_actual=n_actual,
            behavior_prob=behavior_prob,
            behavior_logprob=behavior_logprob,
        )
        self.policy_actions[idx] = policy_idx
        self.supervisor_actions[idx] = supervisor_idx
        self.previous_actions[idx] = previous_idx
        self.selected_sources[idx] = int(selected_source)
        self.score_policy[idx] = float(score_policy)
        self.score_supervisor[idx] = float(score_supervisor)
        self.advantage_policy_supervisor[idx] = float(advantage_policy_supervisor)

    def sample_supervised(
        self,
        batch_size: int,
        device="cpu",
        frac_per: float | None = None,
        frac_recent: float | None = None,
        recent_window: int | None = None,
    ) -> dict:
        sample = super().sample(
            batch_size,
            device=device,
            frac_per=frac_per,
            frac_recent=frac_recent,
            recent_window=recent_window,
        )
        s, a, r, ns, done, discounts, n_actual, idx, is_w = sample
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "next_states": ns,
            "dones": done,
            "discounts": discounts,
            "n_actual": n_actual,
            "indices": idx,
            "is_w": is_w,
            "policy_actions": torch.from_numpy(self.policy_actions[idx]).to(device),
            "supervisor_actions": torch.from_numpy(self.supervisor_actions[idx]).to(device),
            "previous_actions": torch.from_numpy(self.previous_actions[idx]).to(device),
            "selected_sources": torch.from_numpy(self.selected_sources[idx]).to(device),
            "score_policy": torch.from_numpy(self.score_policy[idx]).to(device),
            "score_supervisor": torch.from_numpy(self.score_supervisor[idx]).to(device),
            "advantage_policy_supervisor": torch.from_numpy(
                self.advantage_policy_supervisor[idx]
            ).to(device),
        }

    def export_snapshot(self, ordered: bool = True) -> dict:
        snapshot = super().export_snapshot(ordered=ordered)
        snapshot["source_names"] = dict(SOURCE_NAMES)
        if self.size <= 0:
            return snapshot
        idx = self._ordered_indices() if ordered else np.arange(self.size, dtype=np.int64)
        snapshot.update(
            {
                "policy_actions": self.policy_actions[idx].copy(),
                "supervisor_actions": self.supervisor_actions[idx].copy(),
                "previous_actions": self.previous_actions[idx].copy(),
                "selected_sources": self.selected_sources[idx].copy(),
                "score_policy": self.score_policy[idx].copy(),
                "score_supervisor": self.score_supervisor[idx].copy(),
                "advantage_policy_supervisor": self.advantage_policy_supervisor[idx].copy(),
            }
        )
        return snapshot


class DiscreteSupervisorGatedMixin:
    """Shared pure-value supervisor gate for discrete Q agents."""

    @staticmethod
    def _coerce_gate_config(config) -> DiscreteSupervisorGateConfig:
        if config is None:
            return DiscreteSupervisorGateConfig()
        if isinstance(config, DiscreteSupervisorGateConfig):
            return config
        if isinstance(config, dict):
            return DiscreteSupervisorGateConfig(**config)
        raise TypeError(
            "supervisor_gate_config must be None, a dict, or DiscreteSupervisorGateConfig."
        )

    def _init_discrete_supervisor_gate(self, supervisor_gate_config=None) -> None:
        if self.multistep_mode not in {"one_step", "n_step"} or (
            self.multistep_mode == "n_step" and int(self.n_step) != 1
        ):
            raise NotImplementedError(
                "Supervisor-gated DQN v1 supports multistep_mode='one_step' and "
                "multistep_mode='n_step' only when n_step == 1."
            )

        self.supervisor_gate_config = self._coerce_gate_config(supervisor_gate_config)
        base_buffer = self.buffer
        self.buffer = DiscreteSupervisorPERRecentReplayBuffer(
            base_buffer.capacity,
            base_buffer.state_dim,
            default_discount=base_buffer.default_discount,
            eps=base_buffer.eps,
            alpha=base_buffer.alpha,
            beta_start=base_buffer.beta_start,
            beta_end=base_buffer.beta_end,
            beta_steps=base_buffer.beta_steps,
            frac_per=base_buffer.frac_per,
            frac_recent=base_buffer.frac_recent,
            recent_window=base_buffer.recent_window,
        )

        self.sg_score_policy_trace = []
        self.sg_score_supervisor_trace = []
        self.sg_advantage_trace = []
        self.sg_selected_source_trace = []
        self.sg_q_policy_trace = []
        self.sg_q_supervisor_trace = []

    def _state_tensor(self, state) -> torch.Tensor:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.view(1, -1)
        if state_t.ndim != 2 or state_t.shape[0] != 1:
            raise ValueError(
                "Supervisor-gated DQN expects a single 1-D state or one-row 2-D state."
            )
        return state_t

    @torch.no_grad()
    def _q_values_np(self, state) -> np.ndarray:
        self._set_eval_noise()
        q_values = self.online(self._state_tensor(state)).view(-1)
        return q_values.detach().cpu().numpy().astype(float, copy=True)

    @torch.no_grad()
    def evaluate_action_score(self, state, action) -> dict:
        action_idx = _validate_action_index(action, self.action_dim, "action")
        q_values = self._q_values_np(state)
        q_value = float(q_values[action_idx])
        return {"score": q_value, "q": q_value}

    @torch.no_grad()
    def _sample_policy_action_for_gate(self, state, explore: bool, test: bool) -> int:
        self.steps += 1
        state_t = self._state_tensor(state)

        if bool(test) or not bool(explore):
            self._set_eval_noise()
            self.last_epsilon = 0.0
            self.last_exploration_value = 0.0
            self.last_behavior_prob = 1.0
            self.last_behavior_logprob = 0.0
            q_values = self.online(state_t)
            return int(q_values.argmax(dim=1).item())

        if self.exploration_mode == "epsilon":
            self._set_eval_noise()
            epsilon = float(self.effective_exploration_schedule_value())
            self.last_epsilon = epsilon
            self.last_exploration_value = epsilon
            q_values = self.online(state_t)
            greedy_action = int(q_values.argmax(dim=1).item())
            if random.random() < epsilon:
                action = random.randrange(self.action_dim)
            else:
                action = greedy_action
            prob = epsilon / self.action_dim
            if int(action) == greedy_action:
                prob += 1.0 - epsilon
            self.last_behavior_prob = float(prob)
            self.last_behavior_logprob = float(np.log(max(prob, 1.0e-12)))
            return int(action)

        self._set_training_noise()
        self.last_epsilon = 0.0
        self.last_exploration_value = mean_module_abs_sigma(self.online)
        self.last_behavior_prob = 1.0
        self.last_behavior_logprob = 0.0
        q_values = self.online(state_t)
        return int(q_values.argmax(dim=1).item())

    @torch.no_grad()
    def select_action_with_supervisor(
        self,
        state,
        supervisor_action,
        previous_action=None,
        explore=False,
        test=False,
    ) -> DiscreteSupervisorGatedDecision:
        supervisor_idx = _validate_action_index(
            supervisor_action,
            self.action_dim,
            "supervisor_action",
        )
        if previous_action is None:
            previous_idx = supervisor_idx
        else:
            previous_idx = _validate_action_index(
                previous_action,
                self.action_dim,
                "previous_action",
            )
        del previous_idx

        policy_idx = self._sample_policy_action_for_gate(state, explore=explore, test=test)
        policy_idx = _validate_action_index(policy_idx, self.action_dim, "policy_action")
        policy_score = self.evaluate_action_score(state, policy_idx)
        supervisor_score = self.evaluate_action_score(state, supervisor_idx)
        advantage = float(policy_score["score"] - supervisor_score["score"])

        cfg = self.supervisor_gate_config
        threshold = float(supervisor_score["score"]) + float(cfg.advantage_margin)
        if bool(cfg.default_to_supervisor):
            choose_policy = float(policy_score["score"]) > threshold
        else:
            choose_policy = float(policy_score["score"]) >= threshold
        if self.train_steps < int(cfg.min_train_steps_before_policy_gate):
            choose_policy = False

        if choose_policy:
            action = int(policy_idx)
            selected_source = SOURCE_POLICY
        else:
            action = int(supervisor_idx)
            selected_source = SOURCE_SUPERVISOR

        self.sg_score_policy_trace.append(float(policy_score["score"]))
        self.sg_score_supervisor_trace.append(float(supervisor_score["score"]))
        self.sg_advantage_trace.append(float(advantage))
        self.sg_selected_source_trace.append(float(selected_source))
        self.sg_q_policy_trace.append(float(policy_score["q"]))
        self.sg_q_supervisor_trace.append(float(supervisor_score["q"]))

        return DiscreteSupervisorGatedDecision(
            action=int(action),
            policy_action=int(policy_idx),
            supervisor_action=int(supervisor_idx),
            selected_source=int(selected_source),
            score_policy=float(policy_score["score"]),
            score_supervisor=float(supervisor_score["score"]),
            advantage_policy_supervisor=float(advantage),
            q_policy=float(policy_score["q"]),
            q_supervisor=float(supervisor_score["q"]),
        )

    def push_supervised(
        self,
        state,
        executed_action,
        reward,
        next_state,
        done,
        policy_action=None,
        supervisor_action=None,
        previous_action=None,
        selected_source=SOURCE_SUPERVISOR,
        score_policy=np.nan,
        score_supervisor=np.nan,
        advantage_policy_supervisor=np.nan,
    ) -> None:
        executed_idx = _validate_action_index(executed_action, self.action_dim, "executed_action")
        policy_idx = executed_idx if policy_action is None else _validate_action_index(
            policy_action,
            self.action_dim,
            "policy_action",
        )
        supervisor_idx = executed_idx if supervisor_action is None else _validate_action_index(
            supervisor_action,
            self.action_dim,
            "supervisor_action",
        )
        previous_idx = executed_idx if previous_action is None else _validate_action_index(
            previous_action,
            self.action_dim,
            "previous_action",
        )

        self.buffer.push_supervised(
            np.asarray(state, np.float32),
            int(executed_idx),
            float(reward),
            np.asarray(next_state, np.float32),
            bool(done),
            n_actual=1,
            behavior_prob=self.last_behavior_prob,
            behavior_logprob=self.last_behavior_logprob,
            policy_action=int(policy_idx),
            supervisor_action=int(supervisor_idx),
            previous_action=int(previous_idx),
            selected_source=int(selected_source),
            score_policy=float(score_policy),
            score_supervisor=float(score_supervisor),
            advantage_policy_supervisor=float(advantage_policy_supervisor),
        )


class SupervisorGatedDQNAgent(DiscreteSupervisorGatedMixin, DQNAgent):
    """DQN extension that gates a discrete policy action against a supervisor action."""

    def __init__(self, *args, supervisor_gate_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_discrete_supervisor_gate(supervisor_gate_config)


__all__ = [
    "DiscreteSupervisorGateConfig",
    "DiscreteSupervisorGatedDecision",
    "DiscreteSupervisorGatedMixin",
    "DiscreteSupervisorPERRecentReplayBuffer",
    "SOURCE_FALLBACK",
    "SOURCE_HELD",
    "SOURCE_NAMES",
    "SOURCE_POLICY",
    "SOURCE_SUPERVISOR",
    "SOURCE_WARM_START",
    "SupervisorGatedDQNAgent",
]
