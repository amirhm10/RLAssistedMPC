from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from TD3Agent.agent import TD3Agent, col, hard_update, soft_update
from TD3Agent.supervisor_replay_buffer import (
    SOURCE_FALLBACK,
    SOURCE_POLICY,
    SOURCE_SUPERVISOR,
    SupervisorPERRecentReplayBuffer,
)
from utils.supervisor_gated_action import (
    choose_policy_or_supervisor,
    compute_conservative_score,
    validate_action_vector,
)


@dataclass
class SupervisorGateConfig:
    score_uncertainty_weight: float = 0.0
    score_previous_action_weight: float = 0.0
    score_supervisor_action_weight: float = 0.0
    advantage_margin: float = 0.0
    default_to_supervisor: bool = True
    actor_q_mode: str = "mean"
    supervisor_bc_weight: float = 0.0
    supervisor_bc_temperature: float = 1.0
    smooth_action_weight: float = 0.0
    detach_supervisor_weight: bool = True
    enable_supervisor_actor_loss: bool = True
    min_train_steps_before_policy_gate: int = 0


@dataclass
class SupervisorGatedDecision:
    action: np.ndarray
    policy_action: np.ndarray
    supervisor_action: np.ndarray
    selected_source: int
    score_policy: float
    score_supervisor: float
    advantage_policy_supervisor: float
    q1_policy: float
    q2_policy: float
    q1_supervisor: float
    q2_supervisor: float
    q_gap_policy: float
    q_gap_supervisor: float


def _nanmean_tensor(value: torch.Tensor) -> float:
    finite = torch.isfinite(value)
    if not bool(finite.any()):
        return float("nan")
    return float(value[finite].mean().item())


class SupervisorGatedTD3Agent(TD3Agent):
    """TD3 extension that gates a policy candidate against a supervisor action."""

    def __init__(self, *args, supervisor_gate_config: SupervisorGateConfig | dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.multistep_mode != "one_step":
            raise NotImplementedError(
                "SupervisorGatedTD3Agent v1 supports only multistep_mode='one_step'."
            )

        self.supervisor_gate_config = self._coerce_gate_config(supervisor_gate_config)
        if self.supervisor_gate_config.actor_q_mode not in {"min", "max", "mean", "q1"}:
            raise ValueError("actor_q_mode must be one of 'min', 'max', 'mean', or 'q1'.")

        base_buffer = self.buffer
        self.buffer = SupervisorPERRecentReplayBuffer(
            base_buffer.capacity,
            base_buffer.state_dim,
            base_buffer.action_dim,
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
        self.sg_q_gap_policy_trace = []
        self.sg_q_gap_supervisor_trace = []
        self.sg_weight_trace = []
        self.sg_bc_loss_trace = []
        self.sg_smooth_loss_trace = []

    @staticmethod
    def _coerce_gate_config(config) -> SupervisorGateConfig:
        if config is None:
            return SupervisorGateConfig()
        if isinstance(config, SupervisorGateConfig):
            return config
        if isinstance(config, dict):
            return SupervisorGateConfig(**config)
        raise TypeError("supervisor_gate_config must be None, a dict, or SupervisorGateConfig.")

    def _state_tensor(self, state) -> torch.Tensor:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.view(1, -1)
        if state_t.ndim != 2:
            raise ValueError(f"state must be 1-D or 2-D, got shape {tuple(state_t.shape)}.")
        return state_t

    def _clip_action(self, action) -> np.ndarray:
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -self.max_action, self.max_action)

    def _validate_normalized_action(self, action, name, require_finite=True) -> np.ndarray:
        arr = validate_action_vector(action, self.buffer.action_dim, name, require_finite=require_finite)
        if require_finite and np.max(np.abs(arr)) > self.max_action + 1e-5:
            raise ValueError(
                f"{name} must be in normalized TD3 action bounds "
                f"[-{self.max_action}, {self.max_action}]."
            )
        return arr

    @torch.no_grad()
    def evaluate_action_score(
        self,
        state,
        action,
        supervisor_action=None,
        previous_action=None,
    ) -> dict:
        action_arr = self._validate_normalized_action(action, "action")
        supervisor_arr = None
        previous_arr = None
        if supervisor_action is not None:
            supervisor_arr = self._validate_normalized_action(supervisor_action, "supervisor_action")
        if previous_action is not None:
            previous_arr = self._validate_normalized_action(previous_action, "previous_action")

        state_t = self._state_tensor(state)
        if state_t.shape[0] != 1:
            raise ValueError("evaluate_action_score expects a single state.")
        action_t = torch.as_tensor(action_arr, dtype=torch.float32, device=self.device).view(1, -1)
        q1, q2 = self.critic.forward(state_t, action_t)
        q1_value = float(q1.view(-1)[0].item())
        q2_value = float(q2.view(-1)[0].item())
        cfg = self.supervisor_gate_config
        score = compute_conservative_score(
            q1_value,
            q2_value,
            action_arr,
            supervisor_action=supervisor_arr,
            previous_action=previous_arr,
            uncertainty_weight=cfg.score_uncertainty_weight,
            supervisor_weight=cfg.score_supervisor_action_weight,
            previous_weight=cfg.score_previous_action_weight,
        )
        return {
            "score": float(score),
            "q1": q1_value,
            "q2": q2_value,
            "q_gap": abs(q1_value - q2_value),
        }

    @torch.no_grad()
    def select_action_with_supervisor(
        self,
        state,
        supervisor_action,
        previous_action=None,
        explore=False,
        test=False,
    ) -> SupervisorGatedDecision:
        self.steps += 1
        supervisor_arr = self._validate_normalized_action(supervisor_action, "supervisor_action")
        previous_arr = None
        if previous_action is not None:
            previous_arr = self._validate_normalized_action(previous_action, "previous_action")

        state_t = self._state_tensor(state)
        if state_t.shape[0] != 1:
            raise ValueError("select_action_with_supervisor expects a single state.")

        clean_policy = self.actor(state_t).view(-1).detach().cpu().numpy()
        policy_arr = clean_policy.copy()
        self.last_exploration_value = 0.0
        self.last_param_noise_scale = 0.0
        if bool(explore) and not bool(test):
            if self.exploration_mode == "gaussian":
                sigma = float(self.effective_exploration_schedule_value())
                noise = np.random.randn(*policy_arr.shape) * sigma
                policy_arr = policy_arr + noise
                self.last_exploration_value = float(np.mean(np.abs(noise)))
            else:
                self._resample_param_noise_actor()
                policy_arr = self.perturbed_actor(state_t).view(-1).detach().cpu().numpy()
                self.last_exploration_value = float(np.mean(np.abs(policy_arr - clean_policy)))
        policy_arr = self._clip_action(policy_arr)

        if policy_arr.size != self.buffer.action_dim or not np.all(np.isfinite(policy_arr)):
            self._record_action_diagnostics(supervisor_arr, clean_action=clean_policy)
            return SupervisorGatedDecision(
                action=supervisor_arr.copy(),
                policy_action=supervisor_arr.copy(),
                supervisor_action=supervisor_arr.copy(),
                selected_source=SOURCE_FALLBACK,
                score_policy=float("nan"),
                score_supervisor=float("nan"),
                advantage_policy_supervisor=float("nan"),
                q1_policy=float("nan"),
                q2_policy=float("nan"),
                q1_supervisor=float("nan"),
                q2_supervisor=float("nan"),
                q_gap_policy=float("nan"),
                q_gap_supervisor=float("nan"),
            )

        policy_score = self.evaluate_action_score(
            state_t,
            policy_arr,
            supervisor_action=supervisor_arr,
            previous_action=previous_arr,
        )
        supervisor_score = self.evaluate_action_score(
            state_t,
            supervisor_arr,
            supervisor_action=supervisor_arr,
            previous_action=previous_arr,
        )
        advantage = float(policy_score["score"] - supervisor_score["score"])
        choice = choose_policy_or_supervisor(
            policy_score["score"],
            supervisor_score["score"],
            margin=self.supervisor_gate_config.advantage_margin,
            default_to_supervisor=self.supervisor_gate_config.default_to_supervisor,
        )
        if self.train_steps < int(self.supervisor_gate_config.min_train_steps_before_policy_gate):
            choice = "supervisor"

        if choice == "policy":
            action = policy_arr.copy()
            selected_source = SOURCE_POLICY
        else:
            action = supervisor_arr.copy()
            selected_source = SOURCE_SUPERVISOR
        self._record_action_diagnostics(action, clean_action=clean_policy if bool(explore) else None)
        return SupervisorGatedDecision(
            action=action,
            policy_action=policy_arr.copy(),
            supervisor_action=supervisor_arr.copy(),
            selected_source=int(selected_source),
            score_policy=float(policy_score["score"]),
            score_supervisor=float(supervisor_score["score"]),
            advantage_policy_supervisor=advantage,
            q1_policy=float(policy_score["q1"]),
            q2_policy=float(policy_score["q2"]),
            q1_supervisor=float(supervisor_score["q1"]),
            q2_supervisor=float(supervisor_score["q2"]),
            q_gap_policy=float(policy_score["q_gap"]),
            q_gap_supervisor=float(supervisor_score["q_gap"]),
        )

    def push_supervised(
        self,
        s,
        executed_action,
        r,
        ns,
        done,
        policy_action=None,
        supervisor_action=None,
        previous_action=None,
        selected_source=SOURCE_SUPERVISOR,
        score_policy=np.nan,
        score_supervisor=np.nan,
        advantage_policy_supervisor=np.nan,
        p0=None,
        discount_n=None,
        n_actual=1,
    ):
        self.buffer.push(
            s,
            executed_action,
            r,
            ns,
            bool(done),
            p0=p0,
            discount_n=discount_n,
            n_actual=n_actual,
            policy_action=policy_action,
            supervisor_action=supervisor_action,
            previous_action=previous_action,
            selected_source=selected_source,
            score_policy=score_policy,
            score_supervisor=score_supervisor,
            advantage_policy_supervisor=advantage_policy_supervisor,
        )

    def _tensor_scores(self, states, policy_actions, supervisor_actions, previous_actions):
        cfg = self.supervisor_gate_config
        q1_policy, q2_policy = self.critic.forward(states, policy_actions)
        q1_supervisor, q2_supervisor = self.critic.forward(states, supervisor_actions.detach())
        policy_gap = torch.abs(q1_policy - q2_policy)
        supervisor_gap = torch.abs(q1_supervisor - q2_supervisor)
        score_policy = torch.min(q1_policy, q2_policy) - float(cfg.score_uncertainty_weight) * policy_gap
        score_supervisor = (
            torch.min(q1_supervisor, q2_supervisor)
            - float(cfg.score_uncertainty_weight) * supervisor_gap
        )
        if cfg.score_previous_action_weight != 0.0:
            prev = previous_actions.detach()
            score_policy = score_policy - float(cfg.score_previous_action_weight) * torch.sum(
                (policy_actions - prev) ** 2,
                dim=1,
                keepdim=True,
            )
            score_supervisor = score_supervisor - float(cfg.score_previous_action_weight) * torch.sum(
                (supervisor_actions.detach() - prev) ** 2,
                dim=1,
                keepdim=True,
            )
        if cfg.score_supervisor_action_weight != 0.0:
            sup = supervisor_actions.detach()
            score_policy = score_policy - float(cfg.score_supervisor_action_weight) * torch.sum(
                (policy_actions - sup) ** 2,
                dim=1,
                keepdim=True,
            )
        return {
            "q1_policy": q1_policy,
            "q2_policy": q2_policy,
            "q1_supervisor": q1_supervisor,
            "q2_supervisor": q2_supervisor,
            "score_policy": score_policy,
            "score_supervisor": score_supervisor,
            "policy_gap": policy_gap,
            "supervisor_gap": supervisor_gap,
        }

    def train_step(self, bc_context: Optional[dict] = None) -> Optional[dict]:
        if self.multistep_mode != "one_step":
            raise NotImplementedError(
                "SupervisorGatedTD3Agent v1 supports only multistep_mode='one_step'."
            )
        if len(self.buffer) < self.batch_size:
            return None
        train_index_before = int(self.train_steps)
        batch = self.buffer.sample_supervised(self.batch_size, device=self.device)

        s = batch["states"].to(self.device, non_blocking=True).float()
        a = batch["actions"].to(self.device, non_blocking=True).float()
        r = col(batch["rewards"].to(self.device, non_blocking=True).float())
        ns = batch["next_states"].to(self.device, non_blocking=True).float()
        done = col(batch["dones"].to(self.device, non_blocking=True).float())
        discount_n = col(batch["discounts"].to(self.device, non_blocking=True).float())
        n_actual = col(batch["n_actual"].to(self.device, non_blocking=True).float())
        is_w = col(batch["is_w"].to(self.device, non_blocking=True).float())
        priority_index = batch["indices"]

        supervisor_actions = batch["supervisor_actions"].to(self.device, non_blocking=True).float()
        previous_actions = batch["previous_actions"].to(self.device, non_blocking=True).float()
        score_policy_batch = batch["score_policy"].to(self.device, non_blocking=True).float()
        score_supervisor_batch = batch["score_supervisor"].to(self.device, non_blocking=True).float()
        advantage_batch = batch["advantage_policy_supervisor"].to(self.device, non_blocking=True).float()
        selected_sources = batch["selected_sources"].to(self.device, non_blocking=True).float()

        with torch.no_grad():
            base_next = self.actor_target(ns)
            noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
            noise.clamp_(-self.noise_clip, self.noise_clip)
            next_action = (base_next + noise).clip(-self.max_action, self.max_action)
            q_next = self.critic_target.combined_forward(ns, next_action, mode=self.target_combine)
            bootstrap_q = q_next * (1.0 - done)
            y = r + discount_n * bootstrap_q

        q1, q2 = self.critic.forward(s, a)
        q1 = col(q1)
        q2 = col(q2)
        td1 = (y - q1).detach().abs().view(-1)
        td2 = (y - q2).detach().abs().view(-1)
        td = 0.5 * (td1 + td2)
        l1 = self.loss_fn_critic(q1, y)
        l2 = self.loss_fn_critic(q2, y)
        critic_loss = (is_w * (l1 + l2)).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        self.critic_losses.append(float(critic_loss.item()))
        self.critic_q1_trace.append(float(q1.mean().item()))
        self.critic_q2_trace.append(float(q2.mean().item()))
        self.critic_q_gap_trace.append(float((q1 - q2).abs().mean().item()))
        self.reward_n_mean_trace.append(float(r.mean().item()))
        self.discount_n_mean_trace.append(float(discount_n.mean().item()))
        self.bootstrap_q_mean_trace.append(float(bootstrap_q.mean().item()))
        self.n_actual_mean_trace.append(float(n_actual.mean().item()))
        self.truncated_fraction_trace.append(float((n_actual < float(self.n_step)).float().mean().item()))

        actor_slot = bool(self.total_it % self.policy_delay == 0)
        actor_updated = False
        actor_loss_value = None
        bc_active = False
        bc_weight = 0.0
        bc_loss_value = None
        bc_actor_target_distance = None
        sg_weight_value = None
        sg_bc_loss_value = None
        sg_smooth_loss_value = None
        sg_q_policy_value = float("nan")
        sg_q_supervisor_value = float("nan")
        sg_q_gap_policy_value = float("nan")
        sg_q_gap_supervisor_value = float("nan")

        if actor_slot:
            cfg = self.supervisor_gate_config
            curr = self.actor(s)
            q_for_actor = self.critic.combined_forward(s, curr, mode=cfg.actor_q_mode)
            actor_loss = -torch.mean(q_for_actor)

            tensor_scores = self._tensor_scores(s, curr, supervisor_actions, previous_actions)
            advantage = tensor_scores["score_policy"] - tensor_scores["score_supervisor"]
            temperature = max(1.0e-6, float(cfg.supervisor_bc_temperature))
            supervisor_weight = torch.sigmoid((float(cfg.advantage_margin) - advantage) / temperature)
            if cfg.detach_supervisor_weight:
                supervisor_weight = supervisor_weight.detach()
            sg_weight_value = float(supervisor_weight.mean().item())
            sg_q_policy_value = float(
                (0.5 * (tensor_scores["q1_policy"] + tensor_scores["q2_policy"])).mean().item()
            )
            sg_q_supervisor_value = float(
                (0.5 * (tensor_scores["q1_supervisor"] + tensor_scores["q2_supervisor"])).mean().item()
            )
            sg_q_gap_policy_value = float(tensor_scores["policy_gap"].mean().item())
            sg_q_gap_supervisor_value = float(tensor_scores["supervisor_gap"].mean().item())

            if bool(cfg.enable_supervisor_actor_loss) and cfg.supervisor_bc_weight > 0.0:
                sg_bc_penalty = torch.sum((curr - supervisor_actions.detach()) ** 2, dim=1, keepdim=True)
                sg_bc_loss = torch.mean(supervisor_weight * sg_bc_penalty)
                actor_loss = actor_loss + float(cfg.supervisor_bc_weight) * sg_bc_loss
                sg_bc_loss_value = float(sg_bc_loss.item())
            if cfg.smooth_action_weight > 0.0:
                sg_smooth_loss = torch.mean(torch.sum((curr - previous_actions.detach()) ** 2, dim=1))
                actor_loss = actor_loss + float(cfg.smooth_action_weight) * sg_smooth_loss
                sg_smooth_loss_value = float(sg_smooth_loss.item())

            if bc_context is not None and bool(bc_context.get("active", False)):
                bc_active = True
                bc_weight = float(max(0.0, bc_context.get("weight", 0.0)))
                target_action = torch.as_tensor(
                    np.asarray(bc_context.get("target_action"), np.float32),
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, -1)
                target_action = target_action.expand_as(curr)
                err_sq = (curr - target_action) ** 2
                coord_weights = bc_context.get("coordinate_weights")
                if coord_weights is None:
                    bc_penalty = torch.mean(err_sq, dim=1)
                else:
                    coord_weights_t = torch.as_tensor(
                        np.asarray(coord_weights, np.float32),
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, -1)
                    if coord_weights_t.shape[1] != curr.shape[1]:
                        raise ValueError(
                            f"behavioral-cloning coordinate_weights size {coord_weights_t.shape[1]} "
                            f"does not match action dimension {curr.shape[1]}."
                        )
                    denom = torch.clamp(coord_weights_t.sum(dim=1), min=1e-8)
                    bc_penalty = torch.sum(err_sq * coord_weights_t, dim=1) / denom
                bc_loss = torch.mean(bc_penalty)
                actor_loss = actor_loss + bc_weight * bc_loss
                bc_loss_value = float(bc_loss.item())
                bc_actor_target_distance = float(torch.norm(curr - target_action, dim=1).mean().item())
            actor_loss_value = float(actor_loss.item())

            if self.total_it >= self.actor_freeze:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()
                actor_updated = True
            self.actor_losses.append(actor_loss_value)

            if self.target_update == "soft":
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)
            else:
                if self.train_steps % self.hard_update_interval == 0:
                    hard_update(self.actor_target, self.actor)
                    hard_update(self.critic_target, self.critic)

        self.bc_active_trace.append(float(bc_active))
        self.bc_weight_trace.append(float(bc_weight))
        self.bc_loss_trace.append(np.nan if bc_loss_value is None else float(bc_loss_value))
        self.bc_actor_target_distance_trace.append(
            np.nan if bc_actor_target_distance is None else float(bc_actor_target_distance)
        )

        sg_score_policy_value = _nanmean_tensor(score_policy_batch)
        sg_score_supervisor_value = _nanmean_tensor(score_supervisor_batch)
        sg_advantage_value = _nanmean_tensor(advantage_batch)
        sg_selected_source_value = float(selected_sources.mean().item())
        self.sg_score_policy_trace.append(sg_score_policy_value)
        self.sg_score_supervisor_trace.append(sg_score_supervisor_value)
        self.sg_advantage_trace.append(sg_advantage_value)
        self.sg_selected_source_trace.append(sg_selected_source_value)
        self.sg_q_policy_trace.append(sg_q_policy_value)
        self.sg_q_supervisor_trace.append(sg_q_supervisor_value)
        self.sg_q_gap_policy_trace.append(sg_q_gap_policy_value)
        self.sg_q_gap_supervisor_trace.append(sg_q_gap_supervisor_value)
        self.sg_weight_trace.append(np.nan if sg_weight_value is None else float(sg_weight_value))
        self.sg_bc_loss_trace.append(np.nan if sg_bc_loss_value is None else float(sg_bc_loss_value))
        self.sg_smooth_loss_trace.append(
            np.nan if sg_smooth_loss_value is None else float(sg_smooth_loss_value)
        )

        self.total_it += 1
        self.train_steps += 1
        if hasattr(self.buffer, "update_priorities"):
            self.buffer.update_priorities(priority_index, td)

        return {
            "critic_updated": True,
            "actor_slot": actor_slot,
            "actor_updated": actor_updated,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "bc_active": bool(bc_active),
            "bc_weight": float(bc_weight),
            "bc_loss": bc_loss_value,
            "bc_actor_target_distance": bc_actor_target_distance,
            "train_index_before": train_index_before,
            "train_index_after": int(self.train_steps),
            "sg_score_policy": sg_score_policy_value,
            "sg_score_supervisor": sg_score_supervisor_value,
            "sg_advantage": sg_advantage_value,
            "sg_selected_source": sg_selected_source_value,
            "sg_supervisor_weight": sg_weight_value,
            "sg_bc_loss": sg_bc_loss_value,
            "sg_smooth_loss": sg_smooth_loss_value,
            "sg_q_policy": sg_q_policy_value,
            "sg_q_supervisor": sg_q_supervisor_value,
            "sg_q_gap_policy": sg_q_gap_policy_value,
            "sg_q_gap_supervisor": sg_q_gap_supervisor_value,
        }
