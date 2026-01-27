import torch
import torch.nn as nn
from typing import List, Literal, Optional
from DQN.qnetwork import TwinDiscreteQNetwork
from dataclasses import dataclass
import torch.optim as optim
import numpy as np
import random
from DQN.replay_buffer import PERRecentReplayBuffer
import math


# ----------------
# Utilities
# ----------------
def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())

@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)

@dataclass
class EpsilonSchedule:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000  # For linear and cosine annealing
    mode: Literal["linear", "exp", "cosine"] = "linear"    # linear | exp | cosine
    decay_rate: float = 0.99995 # for exp decay

    def value(self, step: int) -> float:
        if self.mode == "linear":
            # Linear decay
            t = min(1.0, step / max(1, self.eps_decay_steps))
            return self.eps_start + t * (self.eps_end - self.eps_start)

        if self.mode == "exp":
            return self.eps_end + (self.eps_start - self.eps_end) * (self.decay_rate ** step)

        if self.mode == "cosine":
            t = min(1.0, step / max(1, self.eps_decay_steps))
            return self.eps_end + 0.5 * (self.eps_start - self.eps_end) * (1 + math.cos(math.pi * t))



class DQNAgent(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: List[int],
            # Learning and optimization params
            gamma: float = 0.99,
            lr: float = 1e-3,
            batch_size: int = 128,
            buffer_size: int = 50_000,
            grad_clip_norm: float = 10.0,
            # targets and double network params
            double_dqn: bool = True,
            target_update: Literal["soft", "hard"] = "soft",
            tau: float = 0.01,
            hard_update_interval: int = 10_000,
            target_combine: Literal["q1", "min", "max", "mean"] = "q1",
            # architecture
            activation: str = "relu",
            use_layer_norm: bool = False,
            dropout: float = 0.0,
            # exploration
            eps_schedule: Optional[EpsilonSchedule] = None,
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            eps_decay_rate: float = 0.99,
            eps_decay_steps: int = 100_000,
            eps_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
            # device
            device: Optional[torch.device] = None,
    ):

        super(DQNAgent, self).__init__()

        self.device = device if device is not None else get_device()

        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.double_dqn = double_dqn
        self.target_update = target_update
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.target_combine = target_combine

        self.online = TwinDiscreteQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dim,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
        ).to(self.device)

        self.target = TwinDiscreteQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dim,
            activation=activation,
            use_layernorm=use_layer_norm,
            dropout=dropout,
        ).to(self.device)

        hard_update(self.target, self.online)

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none") # Huber Loss

        # Buffer initialization
        self.buffer = PERRecentReplayBuffer(buffer_size, state_dim)

        self.action_dim = action_dim
        self.steps = 0
        self.train_steps = 0
        self.eps_schedule = eps_schedule if eps_schedule is not None else EpsilonSchedule(
            eps_start=eps_start, eps_end=eps_end,
            eps_decay_steps=eps_decay_steps, mode=eps_decay_mode,
            decay_rate=eps_decay_rate,
        )


    # ------------ action -----------------
    @torch.no_grad()
    def take_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        # move one step
        self.steps += 1

        eps = 0.0 if eval_mode else self.eps_schedule.value(self.steps)
        # probability of exploring
        if random.random() < eps:
            return random.randrange(self.action_dim)
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q1 = self.online.q1_forward(s)
        return int(q1.argmax(dim=1).item())

    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    @torch.no_grad()
    def act_eval(self, state: np.ndarray) -> int:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q1 = self.target.q1_forward(s)

        return int(q1.argmax(dim=1).item())


    # ---------- training --------------------
    def train_step(self) -> Optional[float]:
        # check if the buffer has enough info
        if len(self.buffer) < self.batch_size:
            return None

        s, a_idx, r, ns, done, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
        s = s.to(self.device, non_blocking=True).float()    # [B, S]
        a = a_idx.to(self.device, non_blocking=True).long()     # [B]
        r = r.to(self.device, non_blocking=True).float()    # [B]
        ns = ns.to(self.device, non_blocking=True).float()   # [B, S]
        done = done.to(self.device, non_blocking=True).float()  # [B]
        is_w = is_w.to(self.device, non_blocking=True).float()  # [B]

        # Q_online(s, a)
        q1_online, _ = self.online(s)   # [B, A]
        q_sa = q1_online.gather(1, a.view(-1, 1)).squeeze(1)    # [B]

        with torch.no_grad():
            if self.double_dqn:
                # Select the action by only target q1, evaluate with target (combined)
                a_star = self.online.q1_forward(ns).argmax(dim=1)   # [B]
                q_t = self.target.combined_forward(ns, mode=self.target_combine)    # [B, A]
                q_next = q_t.gather(1, a_star.view(-1,1)).squeeze(1)    # [B]
            else:
                # max over target (combined)
                q_t = self.target.combined_forward(ns, mode=self.target_combine)    # [B, A]
                q_next = q_t.max(dim=1).values  # [B]

            y = r + self.gamma * q_next * (1 - done)    # [B]

        # -------- PER importance weighted huber loss --------
        td = y - q_sa
        per_sample = self.loss_fn(q_sa, y)
        loss = (is_w * per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # --------- PER: update priorities from |TD| ----------
        if hasattr(self.buffer, "update_priorities"):
            self.buffer.update_priorities(idx, td.abs())

        self.train_steps += 1
        if self.target_update == "soft":
            soft_update(self.target, self.online, tau=self.tau)
        else:
            if self.train_steps % self.hard_update_interval == 0:
                hard_update(self.target, self.online)

        return float(loss.item())
