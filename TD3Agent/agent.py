import copy
import math
import pickle
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# importing nets and buffer
from TD3Agent.critic import Critic
from TD3Agent.actor import Actor
from TD3Agent.replay_buffer import PERRecentReplayBuffer
from utils.nstep import NStepAccumulator
from utils.nstep_targets import build_truncated_lambda_returns

import os
from datetime import datetime


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


# exploration schedule for Gaussian action noise
@dataclass
class GaussianNoiseSchedule:
    std_start: float = 0.2
    std_end: float = 0.02
    mode: Literal["linear", "exp", "cosine"] = "exp"
    decay_steps: int = 200_000
    decay_rate: float = 0.99995

    def value(self, step: int) -> float:
        if self.mode == "linear":
            t = min(1.0, step / max(1, self.decay_steps))
            return self.std_start + (self.std_end - self.std_start) * t
        if self.mode == "exp":
            return self.std_end + (self.std_start - self.std_end) * (self.decay_rate ** step)
        if self.mode == "cosine":
            t = min(1.0, step / max(1, self.decay_steps))
            return self.std_end + 0.5 * (self.std_start - self.std_end) * (1 + math.cos(math.pi * t))
        raise ValueError("mode must be 'linear' | 'exp' | 'cosine'")


def col(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x.view(-1, 1)


def make_loss_fn(loss_type: str):
    loss_type = str(loss_type).lower()
    if loss_type == "huber":
        return nn.SmoothL1Loss(reduction="none")
    if loss_type == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError("loss_type must be 'huber' or 'mse'.")


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


class TD3Agent(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            actor_hidden: List[int],
            critic_hidden: List[int],
            # learning
            gamma: float = 0.99,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-3,
            batch_size: int = 256,
            n_step: int = 1,
            multistep_mode: Literal["one_step", "n_step", "lambda"] = "one_step",
            lambda_value: float = 0.9,
            grad_clip_norm: Optional[float] = 10.0,
            # TD3
            policy_delay: int = 2,
            target_policy_smoothing_noise_std: float = 0.2,
            noise_clip: float = 0.5,
            max_action: float = 1.0,
            # targets update
            target_update: Literal["soft", "hard"] = "soft",
            tau: float = 0.005,
            hard_update_interval: int = 10_000,
            target_combine: Literal["min", "max", "mean", "q1"] = "min",
            # architecture of the actor and critic
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
            squash: str = "tanh",
            # exploration
            exploration_schedule: Optional[GaussianNoiseSchedule] = None,
            std_start: float = 1.0,
            std_end: float = 0.05,
            std_decay_rate: float = 0.99,
            std_decay_steps: int = 100_000,
            std_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
            exploration_mode: Literal["gaussian", "param_noise"] = "gaussian",
            param_noise_std_start: float = 0.2,
            param_noise_std_end: float = 0.02,
            param_noise_decay_rate: float = 0.99995,
            param_noise_decay_steps: int = 100_000,
            param_noise_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
            param_noise_resample_interval: int = 1,
            loss_type: Literal["huber", "mse"] = "huber",
            # buffer
            buffer_size: int = 40_000,
            replay_frac_per: float = 0.5,
            replay_frac_recent: float = 0.2,
            replay_recent_window: int = 1_000,
            replay_alpha: float = 0.6,
            replay_beta_start: float = 0.4,
            replay_beta_end: float = 1.0,
            replay_beta_steps: int = 50_000,
            # device/opt
            device: Optional[torch.device] = None,
            use_adamw: bool = True,
            # actor freeze
            actor_freeze: int = 0,
    ):
        super(TD3Agent, self).__init__()
        self.device = device if device is not None else get_device()

        # --- hparams ---
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_step = int(n_step)
        self.sequence_len = max(1, int(n_step))
        self.multistep_mode = str(multistep_mode).lower()
        self.lambda_value = float(lambda_value)
        self.grad_clip_norm = grad_clip_norm
        self.policy_delay = policy_delay
        self.t_std = target_policy_smoothing_noise_std
        self.noise_clip = noise_clip
        self.max_action = float(max_action)
        self.target_update = target_update
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.target_combine = target_combine
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.exploration_mode = str(exploration_mode).lower()
        self.loss_type = str(loss_type).lower()
        self.param_noise_resample_interval = max(1, int(param_noise_resample_interval))
        self.replay_frac_per = float(replay_frac_per)
        self.replay_frac_recent = float(replay_frac_recent)
        self.replay_recent_window = int(replay_recent_window)
        self.replay_alpha = float(replay_alpha)
        self.replay_beta_start = float(replay_beta_start)
        self.replay_beta_end = float(replay_beta_end)
        self.replay_beta_steps = int(replay_beta_steps)
        if self.exploration_mode not in {"gaussian", "param_noise"}:
            raise ValueError("exploration_mode must be 'gaussian' or 'param_noise'.")
        if self.multistep_mode not in {"one_step", "n_step", "lambda"}:
            raise ValueError("multistep_mode must be 'one_step', 'n_step', or 'lambda'.")

        self.steps = 0
        self.train_steps = 0
        self.total_it = 0
        self.last_exploration_value = 0.0
        self.last_param_noise_scale = 0.0

        self.actor_freeze = actor_freeze

        # --- actors and critic networks ---
        self.actor = Actor(state_dim, action_dim, actor_hidden,
                           activation=activation, use_layernorm=use_layernorm,
                           dropout=dropout, max_action=max_action, squash=squash).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, actor_hidden,
                                  activation=activation, use_layernorm=use_layernorm,
                                  dropout=dropout, max_action=max_action, squash=squash).to(self.device)
        self.critic = Critic(state_dim, action_dim, critic_hidden,
                             activation=activation, use_layernorm=use_layernorm,
                             dropout=dropout).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden,
                                    activation=activation, use_layernorm=use_layernorm,
                                    dropout=dropout).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # --- optimizers and loss function ---
        if use_adamw:
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0.0)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss_fn_critic = make_loss_fn(self.loss_type)

        # Buffer initialization: the active TD3 path always uses the mixed PER/recent buffer.
        self.buffer = PERRecentReplayBuffer(
            buffer_size,
            state_dim,
            action_dim,
            default_discount=self.gamma,
            alpha=self.replay_alpha,
            beta_start=self.replay_beta_start,
            beta_end=self.replay_beta_end,
            beta_steps=self.replay_beta_steps,
            frac_per=self.replay_frac_per,
            frac_recent=self.replay_frac_recent,
            recent_window=self.replay_recent_window,
        )
        self.nstep_accumulator = NStepAccumulator(gamma=self.gamma, n_step=self.n_step)

        # logs
        self.actor_losses, self.critic_losses = [], []
        self.critic_q1_trace, self.critic_q2_trace, self.critic_q_gap_trace = [], [], []
        self.exploration_trace = []
        self.exploration_magnitude_trace = []
        self.param_noise_scale_trace = []
        self.action_saturation_trace = []
        self.reward_n_mean_trace = []
        self.discount_n_mean_trace = []
        self.bootstrap_q_mean_trace = []
        self.n_actual_mean_trace = []
        self.truncated_fraction_trace = []
        self.lambda_return_mean_trace = []

        # decay scheduler
        self.expl_sched = exploration_schedule if exploration_schedule is not None else GaussianNoiseSchedule(
            std_start=std_start, std_end=std_end,
            decay_steps=std_decay_steps, mode=std_decay_mode,
            decay_rate=std_decay_rate,
        )
        self.param_noise_sched = GaussianNoiseSchedule(
            std_start=param_noise_std_start,
            std_end=param_noise_std_end,
            decay_steps=param_noise_decay_steps,
            mode=param_noise_decay_mode,
            decay_rate=param_noise_decay_rate,
        )
        self.perturbed_actor = None


    # -------- interactions ------
    @torch.no_grad()
    def act_eval(self, state: np.ndarray, sigma_eval: float = 0.0) -> np.ndarray:
        del sigma_eval
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.actor(s)
        return a.clamp(-self.max_action, self.max_action).cpu().numpy()

    @torch.no_grad()
    def take_action(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        self.steps += 1
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        clean_action = self.actor(s).detach().cpu().numpy()
        action = clean_action.copy()
        self.last_exploration_value = 0.0
        self.last_param_noise_scale = 0.0
        if explore:
            if self.exploration_mode == "gaussian":
                self._expl_sigma = self.expl_sched.value(self.steps)
                noise = np.random.randn(*action.shape) * self._expl_sigma
                action = action + noise
                self.last_exploration_value = float(np.mean(np.abs(noise)))
            else:
                self._resample_param_noise_actor()
                perturbed_action = self.perturbed_actor(s).detach().cpu().numpy()
                action = perturbed_action
                self.last_exploration_value = float(np.mean(np.abs(action - clean_action)))
        action = np.clip(action, -self.max_action, self.max_action)
        self._record_action_diagnostics(action, clean_action=clean_action if explore else None)
        return action


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
        self.buffer.push(s, a, r, ns, bool(done))

    def flush_nstep(self):
        if self.multistep_mode != "n_step":
            return 0
        flushed = self.nstep_accumulator.flush()
        for transition in flushed:
            self.buffer.push(
                transition.state,
                transition.action,
                transition.reward_n,
                transition.next_state_n,
                transition.done_n,
                discount_n=transition.discount_n,
                n_actual=transition.n_actual,
            )
        return len(flushed)

    def set_actor_lr(self, new_lr: float):
        for g in self.actor_optimizer.param_groups:
            g['lr'] = new_lr
        if self.total_it % 800 == 1:
            print(f"Actor learning rate changed to: {new_lr:.2e}")

    def _resample_param_noise_actor(self, force: bool = False):
        if self.exploration_mode != "param_noise":
            return
        if (not force) and self.perturbed_actor is not None and (self.steps % self.param_noise_resample_interval != 1):
            return
        self.perturbed_actor = copy.deepcopy(self.actor).to(self.device)
        self.perturbed_actor.eval()
        sigma = float(self.param_noise_sched.value(self.steps))
        with torch.no_grad():
            for param in self.perturbed_actor.parameters():
                param.add_(torch.randn_like(param) * sigma)
        self.last_param_noise_scale = sigma

    def _record_action_diagnostics(self, action, clean_action=None):
        action = np.asarray(action, float)
        if clean_action is not None:
            clean_action = np.asarray(clean_action, float)
            self.last_exploration_value = float(np.mean(np.abs(action - clean_action)))
        sat = float(np.mean(np.abs(action) >= (self.max_action - 1e-6)))
        self.action_saturation_trace.append(sat)
        self.exploration_trace.append(float(self.last_exploration_value))
        self.exploration_magnitude_trace.append(float(self.last_exploration_value))
        self.param_noise_scale_trace.append(float(self.last_param_noise_scale))


    # ------ training ------
    def train_step(self) -> Optional[dict]:
        # check if the buffer has enough info
        if len(self.buffer) < self.batch_size:
            return None
        train_index_before = int(self.train_steps)

        if self.multistep_mode == "lambda":
            seq = self.buffer.sample_sequence(self.batch_size, self.sequence_len, device=self.device)
            states = seq["states"].float()
            actions = seq["actions"].float()
            rewards = seq["rewards"].float()
            next_states = seq["next_states"].float()
            dones = seq["dones"].float()
            mask = seq["mask"].float()
            start_indices = seq["start_indices"]
            is_w = col(seq["is_w"].float())

            batch_size = states.shape[0]
            seq_len = states.shape[1]
            state_dim = states.shape[2]
            s = states[:, 0, :]
            a = actions[:, 0, :]

            next_states_flat = next_states.view(batch_size * seq_len, state_dim)
            with torch.no_grad():
                base_next = self.actor_target(next_states_flat)
                noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
                noise.clamp_(-self.noise_clip, self.noise_clip)
                next_action_flat = (base_next + noise).clip(-self.max_action, self.max_action)
                q_next_all = self.critic_target.combined_forward(next_states_flat, next_action_flat, mode=self.target_combine)
                bootstrap_values = q_next_all.view(batch_size, seq_len) * (1.0 - dones)
                targets = build_truncated_lambda_returns(
                    rewards=rewards,
                    dones=dones,
                    bootstrap_values=bootstrap_values,
                    mask=mask,
                    gamma=self.gamma,
                    lambda_value=self.lambda_value,
                )
                y = col(targets)

            q1, q2 = self.critic(s, a)
            q1 = col(q1)
            q2 = col(q2)
            td1 = (y - q1).detach().abs().view(-1)
            td2 = (y - q2).detach().abs().view(-1)
            td = 0.5 * (td1 + td2)
            l1 = self.loss_fn_critic(q1, y)
            l2 = self.loss_fn_critic(q2, y)
            critic_loss = (is_w * (l1 + l2)).mean()

            reward_prefix, endpoint_discount, endpoint_bootstrap, lengths, truncated_fraction = _sequence_endpoint_stats(
                rewards, dones, bootstrap_values, mask, self.gamma
            )
            reward_trace_mean = float(reward_prefix.mean().item())
            discount_trace_mean = float(endpoint_discount.mean().item())
            bootstrap_trace_mean = float(endpoint_bootstrap.mean().item())
            n_actual_mean = float(lengths.mean().item())
            truncated_fraction_value = float(truncated_fraction.item())
            lambda_return_mean = float(targets.mean().item())
            priority_index = start_indices
        else:
            s, a, r, ns, done, discount_n, n_actual, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
            is_w = col(is_w.to(self.device, non_blocking=True).float())

            s = s.to(self.device, non_blocking=True).float()
            a = a.to(self.device, non_blocking=True).float()
            r = col(r.to(self.device, non_blocking=True).float())
            ns = ns.to(self.device, non_blocking=True).float()
            done = col(done.to(self.device, non_blocking=True).float())
            discount_n = col(discount_n.to(self.device, non_blocking=True).float())
            n_actual = col(n_actual.to(self.device, non_blocking=True).float())

            with torch.no_grad():
                base_next = self.actor_target(ns)
                noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
                noise.clamp_(-self.noise_clip, self.noise_clip)
                next_action = (base_next + noise).clip(-self.max_action, self.max_action)
                q_next = self.critic_target.combined_forward(ns, next_action, mode=self.target_combine)
                bootstrap_q = q_next * (1.0 - done)
                y = r + discount_n * bootstrap_q

            q1, q2 = self.critic(s, a)
            q1 = col(q1)
            q2 = col(q2)
            td1 = (y - q1).detach().abs().view(-1)
            td2 = (y - q2).detach().abs().view(-1)
            td = 0.5 * (td1 + td2)
            l1 = self.loss_fn_critic(q1, y)
            l2 = self.loss_fn_critic(q2, y)
            critic_loss = (is_w * (l1 + l2)).mean()

            reward_trace_mean = float(r.mean().item())
            discount_trace_mean = float(discount_n.mean().item())
            bootstrap_trace_mean = float(bootstrap_q.mean().item())
            n_actual_mean = float(n_actual.mean().item())
            truncated_fraction_value = float((n_actual < float(self.n_step)).float().mean().item())
            lambda_return_mean = None
            priority_index = idx

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        self.critic_losses.append(float(critic_loss.item()))
        self.critic_q1_trace.append(float(q1.mean().item()))
        self.critic_q2_trace.append(float(q2.mean().item()))
        self.critic_q_gap_trace.append(float((q1 - q2).abs().mean().item()))
        self.reward_n_mean_trace.append(reward_trace_mean)
        self.discount_n_mean_trace.append(discount_trace_mean)
        self.bootstrap_q_mean_trace.append(bootstrap_trace_mean)
        self.n_actual_mean_trace.append(n_actual_mean)
        self.truncated_fraction_trace.append(truncated_fraction_value)
        if lambda_return_mean is not None:
            self.lambda_return_mean_trace.append(lambda_return_mean)

        actor_slot = bool(self.total_it % self.policy_delay == 0)
        actor_updated = False
        actor_loss_value = None

        # ------ Delayed actor + target update -------
        if actor_slot:
            curr = self.actor(s)
            q_for_actor = self.critic.q1_forward(s, curr)
            actor_loss = -torch.mean(q_for_actor)
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

        self.total_it += 1
        self.train_steps += 1

        # --------- PER: update priorities from twin-critic |TD| ----------
        if hasattr(self.buffer, "update_priorities"):
            self.buffer.update_priorities(priority_index, td)

        return {
            "critic_updated": True,
            "actor_slot": actor_slot,
            "actor_updated": actor_updated,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "train_index_before": train_index_before,
            "train_index_after": int(self.train_steps),
        }

    def load(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f)

        self.actor.load_state_dict(d['actor_state_dict'])
        self.critic.load_state_dict(d['critic_state_dict'])
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # re-init optimizers then load states
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=0)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=0)

        print(f"Agent loaded successfully from: {path}")

    def save(
            self,
            directory: str,
            prefix: str = "td3",
            include_optim: bool = False,
    ) -> str:
        """
        Save a checkpoint to `directory` with filename based on current time.
        Returns the full path to the saved pickle.

        Notes:
        - Your existing `load(...)` only reads 'actor_state_dict' and 'critic_state_dict'.
          Extra keys saved here are harmless and simply ignored by that loader.
        - Set `include_optim=True` if you later add a loader that restores optimizers.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"{prefix}_{timestamp}.pkl")

        payload = {
            # what your current `load(...)` expects:
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),

            # nice-to-have extras (ignored by your current loader):
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "hparams": {
                "gamma": self.gamma,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "batch_size": self.batch_size,
                "grad_clip_norm": self.grad_clip_norm,
                "policy_delay": self.policy_delay,
                "target_update": self.target_update,
                "tau": self.tau,
                "hard_update_interval": self.hard_update_interval,
                "target_combine": self.target_combine,
                "t_std": self.t_std,
                "noise_clip": self.noise_clip,
                "max_action": self.max_action,
                "actor_freeze": self.actor_freeze,
                "exploration_mode": self.exploration_mode,
                "loss_type": self.loss_type,
                "param_noise_resample_interval": self.param_noise_resample_interval,
                "n_step": self.n_step,
                "multistep_mode": self.multistep_mode,
                "lambda_value": self.lambda_value,
                "replay_frac_per": self.replay_frac_per,
                "replay_frac_recent": self.replay_frac_recent,
                "replay_recent_window": self.replay_recent_window,
                "replay_alpha": self.replay_alpha,
                "replay_beta_start": self.replay_beta_start,
                "replay_beta_end": self.replay_beta_end,
                "replay_beta_steps": self.replay_beta_steps,
                "steps": self.steps,
                "train_steps": self.train_steps,
                "total_it": self.total_it,
            },
        }

        if include_optim:
            payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"Saved TD3 checkpoint to: {path}")
        return path

