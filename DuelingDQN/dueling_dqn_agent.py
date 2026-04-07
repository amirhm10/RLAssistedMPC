import math
import random
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DuelingDQN.qnetwork import DuelingQNetwork
from DuelingDQN.replay_buffer import PERRecentReplayBuffer


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)


@dataclass
class EpsilonSchedule:
    eps_start: float = 0.30
    eps_end: float = 0.01
    eps_decay_steps: int = 15_000
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
            return self.eps_end + 0.5 * (self.eps_start - self.eps_end) * (1.0 + math.cos(math.pi * frac))
        raise ValueError(f"Unsupported epsilon schedule mode: {self.mode}")


class DuelingDQNAgent(nn.Module):
    """
    Dueling Double DQN agent for discrete supervisory horizon selection.

    Defaults are chosen for sparse, structured supervisory MPC decisions rather
    than high-frequency game-like interaction:
    - linear epsilon decay is easier to reason about across limited decision counts
    - hard target updates give stable targets in discrete Q-learning
    - PERRecentReplayBuffer is reused for continuity with the current repo
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: List[int],
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 128,
        buffer_size: int = 40_000,
        grad_clip_norm: float = 10.0,
        double_dqn: bool = True,
        target_update: Literal["soft", "hard"] = "hard",
        tau: float = 0.005,
        hard_update_interval: int = 2_000,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        eps_schedule: Optional[EpsilonSchedule] = None,
        eps_start: float = 0.30,
        eps_end: float = 0.01,
        eps_decay_rate: float = 0.99995,
        eps_decay_steps: int = 15_000,
        eps_decay_mode: Literal["linear", "exp", "cosine"] = "linear",
        replay_frac_per: float = 0.6,
        replay_frac_recent: float = 0.2,
        replay_recent_window: int = 5_000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device if device is not None else get_device()
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.grad_clip_norm = grad_clip_norm
        self.double_dqn = bool(double_dqn)
        self.target_update = str(target_update).lower()
        self.tau = float(tau)
        self.hard_update_interval = int(hard_update_interval)
        self.action_dim = int(action_dim)
        self.replay_frac_per = float(replay_frac_per)
        self.replay_frac_recent = float(replay_frac_recent)
        self.replay_recent_window = int(replay_recent_window)

        self.online = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dim,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
        ).to(self.device)
        self.target = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dim,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
        ).to(self.device)
        hard_update(self.target, self.online)

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.buffer = PERRecentReplayBuffer(buffer_size, state_dim)

        self.steps = 0
        self.train_steps = 0
        self.last_epsilon = float(eps_start)
        self.eps_schedule = eps_schedule if eps_schedule is not None else EpsilonSchedule(
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            mode=eps_decay_mode,
            decay_rate=eps_decay_rate,
        )

        self.loss_history = []
        self.epsilon_trace = []
        self.avg_td_error_trace = []
        self.avg_max_q_trace = []
        self.avg_value_trace = []
        self.avg_advantage_spread_trace = []

    @torch.no_grad()
    def _greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.online(state_tensor)
        return int(q_values.argmax(dim=1).item())

    @torch.no_grad()
    def take_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        self.steps += 1
        epsilon = 0.0 if eval_mode else float(self.eps_schedule.value(self.steps))
        self.last_epsilon = epsilon

        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        return self._greedy_action(state)

    @torch.no_grad()
    def act_eval(self, state: np.ndarray) -> int:
        return self._greedy_action(state)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        state, action_idx, reward, next_state, done, sample_idx, is_w = self.buffer.sample(
            self.batch_size,
            device=self.device,
            frac_per=self.replay_frac_per,
            frac_recent=self.replay_frac_recent,
            recent_window=self.replay_recent_window,
        )
        state = state.float()
        action = action_idx.long()
        reward = reward.float()
        next_state = next_state.float()
        done = done.float()
        is_w = is_w.float()

        q_values, values, advantages = self.online.forward_with_streams(state)
        q_sa = q_values.gather(1, action.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_state).argmax(dim=1)
                target_q_values = self.target(next_state)
                next_q = target_q_values.gather(1, next_actions.view(-1, 1)).squeeze(1)
            else:
                next_q = self.target(next_state).max(dim=1).values
            y = reward + self.gamma * next_q * (1.0 - done)

        td_error = y - q_sa
        per_sample_loss = self.loss_fn(q_sa, y)
        loss = (is_w * per_sample_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if hasattr(self.buffer, "update_priorities"):
            self.buffer.update_priorities(sample_idx, td_error.abs())

        self.train_steps += 1
        if self.target_update == "soft":
            soft_update(self.target, self.online, tau=self.tau)
        else:
            if self.train_steps % self.hard_update_interval == 0:
                hard_update(self.target, self.online)

        advantage_spread = advantages.max(dim=1).values - advantages.min(dim=1).values
        self.loss_history.append(float(loss.item()))
        self.epsilon_trace.append(float(self.last_epsilon))
        self.avg_td_error_trace.append(float(td_error.abs().mean().item()))
        self.avg_max_q_trace.append(float(q_values.max(dim=1).values.mean().item()))
        self.avg_value_trace.append(float(values.mean().item()))
        self.avg_advantage_spread_trace.append(float(advantage_spread.mean().item()))

        return float(loss.item())
