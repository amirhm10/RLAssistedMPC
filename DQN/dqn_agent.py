import math
import random
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQN.qnetwork import DiscreteQNetwork
from DQN.replay_buffer import PERRecentReplayBuffer
from utils.nstep import NStepAccumulator
from utils.nstep_targets import build_discrete_retrace_targets, build_truncated_lambda_returns
from utils.noisy_layers import mean_module_abs_sigma


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


def make_loss_fn(loss_type: str):
    loss_type = str(loss_type).lower()
    if loss_type == "huber":
        return nn.SmoothL1Loss(reduction="none")
    if loss_type == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError("loss_type must be 'huber' or 'mse'.")


@dataclass
class EpsilonSchedule:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    mode: Literal["linear", "exp", "cosine"] = "linear"
    decay_rate: float = 0.99995

    def value(self, step: int) -> float:
        if self.mode == "linear":
            frac = min(1.0, step / max(1, self.eps_decay_steps))
            return self.eps_start + frac * (self.eps_end - self.eps_start)
        if self.mode == "exp":
            return self.eps_end + (self.eps_start - self.eps_end) * (self.decay_rate ** step)
        if self.mode == "cosine":
            frac = min(1.0, step / max(1, self.eps_decay_steps))
            return self.eps_end + 0.5 * (self.eps_start - self.eps_end) * (1 + math.cos(math.pi * frac))
        raise ValueError("Unsupported epsilon schedule mode.")


def _sequence_discount_powers(seq_len: int, gamma: float, device: torch.device) -> torch.Tensor:
    return torch.pow(torch.full((seq_len,), float(gamma), dtype=torch.float32, device=device), torch.arange(seq_len, device=device, dtype=torch.float32))


def _sequence_endpoint_stats(rewards, dones, bootstrap_values, mask, gamma: float):
    seq_len = rewards.shape[1]
    powers = _sequence_discount_powers(seq_len, gamma, rewards.device).view(1, -1)
    reward_prefix = (rewards * mask * powers).sum(dim=1)
    lengths = mask.sum(dim=1).clamp_min(1.0)
    last_idx = (lengths.long() - 1).clamp(min=0)
    last_done = dones.gather(1, last_idx.view(-1, 1)).squeeze(1)
    endpoint_discount = torch.pow(torch.full_like(lengths, float(gamma)), lengths) * (1.0 - last_done)
    endpoint_bootstrap = bootstrap_values.gather(1, last_idx.view(-1, 1)).squeeze(1) * (1.0 - last_done)
    truncated_fraction = (lengths < float(seq_len)).float().mean()
    return reward_prefix, endpoint_discount, endpoint_bootstrap, lengths, truncated_fraction


class DQNAgent(nn.Module):
    """
    Clean Double DQN baseline with optional advanced multistep targets.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: List[int],
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 128,
        buffer_size: int = 50_000,
        n_step: int = 1,
        multistep_mode: Literal["one_step", "n_step", "lambda", "retrace"] = "one_step",
        lambda_value: float = 0.9,
        grad_clip_norm: float = 10.0,
        double_dqn: bool = True,
        target_update: Literal["soft", "hard"] = "soft",
        tau: float = 0.01,
        hard_update_interval: int = 10_000,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        eps_schedule: Optional[EpsilonSchedule] = None,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_rate: float = 0.99,
        eps_decay_steps: int = 100_000,
        eps_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
        exploration_mode: Literal["epsilon", "noisy"] = "epsilon",
        loss_type: Literal["huber", "mse"] = "huber",
        noisy_sigma_init: float = 0.5,
        device: Optional[torch.device] = None,
        target_combine: Optional[str] = None,
    ):
        super().__init__()

        del target_combine
        self.device = device if device is not None else get_device()
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.n_step = int(n_step)
        self.sequence_len = max(1, int(n_step))
        self.multistep_mode = str(multistep_mode).lower()
        self.lambda_value = float(lambda_value)
        self.grad_clip_norm = grad_clip_norm
        self.double_dqn = bool(double_dqn)
        self.target_update = str(target_update).lower()
        self.tau = float(tau)
        self.hard_update_interval = int(hard_update_interval)
        self.exploration_mode = str(exploration_mode).lower()
        self.loss_type = str(loss_type).lower()
        self.noisy_sigma_init = float(noisy_sigma_init)
        self.action_dim = int(action_dim)
        if self.exploration_mode not in {"epsilon", "noisy"}:
            raise ValueError("exploration_mode must be 'epsilon' or 'noisy'.")
        if self.multistep_mode not in {"one_step", "n_step", "lambda", "retrace"}:
            raise ValueError("multistep_mode must be one of 'one_step', 'n_step', 'lambda', or 'retrace'.")
        if self.multistep_mode == "retrace" and self.exploration_mode != "epsilon":
            raise ValueError("Exact Retrace is only supported with exploration_mode='epsilon'.")

        use_noisy = self.exploration_mode == "noisy"
        self.online = DiscreteQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dim,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
            use_noisy=use_noisy,
            noisy_sigma_init=self.noisy_sigma_init,
        ).to(self.device)
        self.target = DiscreteQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dim,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
            use_noisy=use_noisy,
            noisy_sigma_init=self.noisy_sigma_init,
        ).to(self.device)
        hard_update(self.target, self.online)
        self.online.set_noise_enabled(use_noisy)
        self.target.set_noise_enabled(use_noisy)

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = make_loss_fn(self.loss_type)
        self.buffer = PERRecentReplayBuffer(buffer_size, state_dim, default_discount=self.gamma)
        self.nstep_accumulator = NStepAccumulator(gamma=self.gamma, n_step=self.n_step)

        self.steps = 0
        self.train_steps = 0
        self.last_epsilon = float(eps_start)
        self.last_exploration_value = 0.0
        self.last_behavior_prob = 1.0
        self.last_behavior_logprob = 0.0
        self.eps_schedule = eps_schedule if eps_schedule is not None else EpsilonSchedule(
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            mode=eps_decay_mode,
            decay_rate=eps_decay_rate,
        )

        self.loss_history = []
        self.exploration_trace = []
        self.epsilon_trace = []
        self.avg_td_error_trace = []
        self.avg_max_q_trace = []
        self.avg_chosen_q_trace = []
        self.noisy_sigma_trace = []
        self.reward_n_mean_trace = []
        self.discount_n_mean_trace = []
        self.bootstrap_q_mean_trace = []
        self.n_actual_mean_trace = []
        self.truncated_fraction_trace = []
        self.lambda_return_mean_trace = []
        self.offpolicy_rho_mean_trace = []
        self.offpolicy_c_mean_trace = []
        self.behavior_logprob_mean_trace = []
        self.retrace_c_clip_fraction_trace = []

    def _set_eval_noise(self):
        self.online.set_noise_enabled(False)
        self.target.set_noise_enabled(False)

    def _set_training_noise(self):
        enabled = self.exploration_mode == "noisy"
        self.online.set_noise_enabled(enabled)
        self.target.set_noise_enabled(enabled)
        if enabled:
            self.online.reset_noise()
            self.target.reset_noise()

    @torch.no_grad()
    def _greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.online(state_tensor)
        return int(q_values.argmax(dim=1).item())

    @torch.no_grad()
    def take_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        self.steps += 1
        if eval_mode:
            return self.act_eval(state)

        if self.exploration_mode == "epsilon":
            self._set_eval_noise()
            epsilon = float(self.eps_schedule.value(self.steps))
            self.last_epsilon = epsilon
            self.last_exploration_value = epsilon
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.online(state_tensor)
            greedy_action = int(q_values.argmax(dim=1).item())
            if random.random() < epsilon:
                action = random.randrange(self.action_dim)
            else:
                action = greedy_action
            prob = epsilon / self.action_dim
            if action == greedy_action:
                prob += 1.0 - epsilon
            self.last_behavior_prob = float(prob)
            self.last_behavior_logprob = float(np.log(max(prob, 1e-12)))
            return int(action)

        self._set_training_noise()
        self.last_epsilon = 0.0
        self.last_exploration_value = mean_module_abs_sigma(self.online)
        self.last_behavior_prob = 1.0
        self.last_behavior_logprob = 0.0
        return self._greedy_action(state)

    def push(self, s, a, r, ns, done):
        if self.multistep_mode == "n_step":
            ready = self.nstep_accumulator.append(s, a, r, ns, bool(done))
            for transition in ready:
                self.buffer.push(
                    transition.state,
                    transition.action,
                    transition.reward_n,
                    transition.next_state_n,
                    transition.done_n,
                    discount_n=transition.discount_n,
                    n_actual=transition.n_actual,
                )
            return

        self.buffer.push(
            s,
            a,
            r,
            ns,
            bool(done),
            behavior_prob=self.last_behavior_prob,
            behavior_logprob=self.last_behavior_logprob,
        )

    @torch.no_grad()
    def act_eval(self, state: np.ndarray) -> int:
        self._set_eval_noise()
        return self._greedy_action(state)

    def _update_target(self):
        self.train_steps += 1
        if self.target_update == "soft":
            soft_update(self.target, self.online, tau=self.tau)
        elif self.train_steps % self.hard_update_interval == 0:
            hard_update(self.target, self.online)

    def _train_endpoint_mode(self) -> float:
        s, a_idx, r, ns, done, discount_n, n_actual, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
        s = s.float()
        a = a_idx.long()
        r = r.float()
        ns = ns.float()
        done = done.float()
        discount_n = discount_n.float()
        n_actual = n_actual.float()
        is_w = is_w.float()

        q_online = self.online(s)
        q_sa = q_online.gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(ns).argmax(dim=1)
                q_target = self.target(ns)
                q_next = q_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
            else:
                q_next = self.target(ns).max(dim=1).values
            bootstrap_q = q_next * (1.0 - done)
            y = r + discount_n * bootstrap_q

        td = y - q_sa
        loss = (is_w * self.loss_fn(q_sa, y)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.buffer.update_priorities(idx, td.abs())
        self._update_target()

        self.loss_history.append(float(loss.item()))
        self.exploration_trace.append(float(self.last_exploration_value))
        self.epsilon_trace.append(float(self.last_epsilon))
        self.avg_td_error_trace.append(float(td.abs().mean().item()))
        self.avg_max_q_trace.append(float(q_online.max(dim=1).values.mean().item()))
        self.avg_chosen_q_trace.append(float(q_sa.mean().item()))
        self.noisy_sigma_trace.append(float(mean_module_abs_sigma(self.online)))
        self.reward_n_mean_trace.append(float(r.mean().item()))
        self.discount_n_mean_trace.append(float(discount_n.mean().item()))
        self.bootstrap_q_mean_trace.append(float(bootstrap_q.mean().item()))
        self.n_actual_mean_trace.append(float(n_actual.mean().item()))
        self.truncated_fraction_trace.append(float((n_actual < float(self.n_step)).float().mean().item()))
        return float(loss.item())

    def _train_lambda_mode(self) -> float:
        seq = self.buffer.sample_sequence(self.batch_size, self.sequence_len, device=self.device)
        states = seq["states"].float()
        actions = seq["actions"].long()
        rewards = seq["rewards"].float()
        next_states = seq["next_states"].float()
        dones = seq["dones"].float()
        mask = seq["mask"].float()
        is_w = seq["is_w"].float()
        start_indices = seq["start_indices"]

        batch_size, seq_len, state_dim = states.shape
        states_flat = states.view(batch_size * seq_len, state_dim)
        next_states_flat = next_states.view(batch_size * seq_len, state_dim)

        q_online_all = self.online(states_flat).view(batch_size, seq_len, self.action_dim)
        q_first = q_online_all[:, 0, :].gather(1, actions[:, :1]).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_states_flat).view(batch_size, seq_len, self.action_dim).argmax(dim=2)
                next_q_target = self.target(next_states_flat).view(batch_size, seq_len, self.action_dim)
                bootstrap_values = next_q_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1) * (1.0 - dones)
            else:
                next_q_target = self.target(next_states_flat).view(batch_size, seq_len, self.action_dim)
                bootstrap_values = next_q_target.max(dim=2).values * (1.0 - dones)
            targets = build_truncated_lambda_returns(
                rewards=rewards,
                dones=dones,
                bootstrap_values=bootstrap_values,
                mask=mask,
                gamma=self.gamma,
                lambda_value=self.lambda_value,
            )

        td = targets - q_first
        loss = (is_w * self.loss_fn(q_first, targets)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.buffer.update_priorities(start_indices, td.abs())
        self._update_target()

        reward_prefix, endpoint_discount, endpoint_bootstrap, lengths, truncated_fraction = _sequence_endpoint_stats(
            rewards, dones, bootstrap_values, mask, self.gamma
        )
        self.loss_history.append(float(loss.item()))
        self.exploration_trace.append(float(self.last_exploration_value))
        self.epsilon_trace.append(float(self.last_epsilon))
        self.avg_td_error_trace.append(float(td.abs().mean().item()))
        self.avg_max_q_trace.append(float(q_online_all[:, 0, :].max(dim=1).values.mean().item()))
        self.avg_chosen_q_trace.append(float(q_first.mean().item()))
        self.noisy_sigma_trace.append(float(mean_module_abs_sigma(self.online)))
        self.reward_n_mean_trace.append(float(reward_prefix.mean().item()))
        self.discount_n_mean_trace.append(float(endpoint_discount.mean().item()))
        self.bootstrap_q_mean_trace.append(float(endpoint_bootstrap.mean().item()))
        self.n_actual_mean_trace.append(float(lengths.mean().item()))
        self.truncated_fraction_trace.append(float(truncated_fraction.item()))
        self.lambda_return_mean_trace.append(float(targets.mean().item()))
        return float(loss.item())

    def _train_retrace_mode(self) -> float:
        seq = self.buffer.sample_sequence(self.batch_size, self.sequence_len, device=self.device)
        states = seq["states"].float()
        actions = seq["actions"].long()
        rewards = seq["rewards"].float()
        next_states = seq["next_states"].float()
        dones = seq["dones"].float()
        behavior_prob = seq["behavior_prob"].float()
        behavior_logprob = seq["behavior_logprob"].float()
        mask = seq["mask"].float()
        is_w = seq["is_w"].float()
        start_indices = seq["start_indices"]

        batch_size, seq_len, state_dim = states.shape
        states_flat = states.view(batch_size * seq_len, state_dim)
        next_states_flat = next_states.view(batch_size * seq_len, state_dim)

        q_online_all = self.online(states_flat).view(batch_size, seq_len, self.action_dim)
        q_first = q_online_all[:, 0, :].gather(1, actions[:, :1]).squeeze(1)

        with torch.no_grad():
            online_greedy = q_online_all.detach().argmax(dim=2)
            target_q_current = self.target(states_flat).view(batch_size, seq_len, self.action_dim)
            q_taken_target = target_q_current.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            next_actions = self.online(next_states_flat).view(batch_size, seq_len, self.action_dim).argmax(dim=2)
            next_q_target = self.target(next_states_flat).view(batch_size, seq_len, self.action_dim)
            bootstrap_values = next_q_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1) * (1.0 - dones)
            target_action_prob = (actions == online_greedy).float()
            targets, rho, c = build_discrete_retrace_targets(
                rewards=rewards,
                dones=dones,
                bootstrap_values=bootstrap_values,
                q_taken=q_taken_target,
                target_action_prob=target_action_prob,
                behavior_prob=behavior_prob,
                mask=mask,
                gamma=self.gamma,
                lambda_value=self.lambda_value,
            )

        td = targets - q_first
        loss = (is_w * self.loss_fn(q_first, targets)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.buffer.update_priorities(start_indices, td.abs())
        self._update_target()

        reward_prefix, endpoint_discount, endpoint_bootstrap, lengths, truncated_fraction = _sequence_endpoint_stats(
            rewards, dones, bootstrap_values, mask, self.gamma
        )
        valid_mask = mask > 0.0
        clipped_fraction = torch.where(valid_mask, (rho > 1.0).float(), torch.zeros_like(rho)).sum() / valid_mask.sum().clamp_min(1.0)

        self.loss_history.append(float(loss.item()))
        self.exploration_trace.append(float(self.last_exploration_value))
        self.epsilon_trace.append(float(self.last_epsilon))
        self.avg_td_error_trace.append(float(td.abs().mean().item()))
        self.avg_max_q_trace.append(float(q_online_all[:, 0, :].max(dim=1).values.mean().item()))
        self.avg_chosen_q_trace.append(float(q_first.mean().item()))
        self.noisy_sigma_trace.append(float(mean_module_abs_sigma(self.online)))
        self.reward_n_mean_trace.append(float(reward_prefix.mean().item()))
        self.discount_n_mean_trace.append(float(endpoint_discount.mean().item()))
        self.bootstrap_q_mean_trace.append(float(endpoint_bootstrap.mean().item()))
        self.n_actual_mean_trace.append(float(lengths.mean().item()))
        self.truncated_fraction_trace.append(float(truncated_fraction.item()))
        self.offpolicy_rho_mean_trace.append(float(torch.where(valid_mask, rho, torch.zeros_like(rho)).sum().item() / valid_mask.sum().clamp_min(1.0).item()))
        self.offpolicy_c_mean_trace.append(float(torch.where(valid_mask, c, torch.zeros_like(c)).sum().item() / valid_mask.sum().clamp_min(1.0).item()))
        self.behavior_logprob_mean_trace.append(float(torch.where(valid_mask, behavior_logprob, torch.zeros_like(behavior_logprob)).sum().item() / valid_mask.sum().clamp_min(1.0).item()))
        self.retrace_c_clip_fraction_trace.append(float(clipped_fraction.item()))
        self.lambda_return_mean_trace.append(float(targets.mean().item()))
        return float(loss.item())

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        self._set_training_noise()
        if self.multistep_mode in {"one_step", "n_step"}:
            return self._train_endpoint_mode()
        if self.multistep_mode == "lambda":
            return self._train_lambda_mode()
        return self._train_retrace_mode()
