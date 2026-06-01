from __future__ import annotations

import math
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from TD7Agent.actor import TD7Actor
from TD7Agent.critic import TD7Critic
from TD7Agent.encoder import TD7Encoder
from TD7Agent.replay_buffer import HybridLAPReplayBuffer


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_global_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def col(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x.view(-1, 1)


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
        raise ValueError("mode must be 'linear', 'exp', or 'cosine'.")


def lap_huber_loss(td_abs: torch.Tensor, min_priority: float = 1.0) -> torch.Tensor:
    min_priority = float(min_priority)
    loss = torch.where(
        td_abs < min_priority,
        0.5 * td_abs.pow(2),
        min_priority * td_abs,
    )
    return loss.sum(dim=1).mean()


class TD7Agent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden: List[int],
        critic_hidden: List[int],
        encoder_hidden: Optional[List[int]] = None,
        zs_dim: int = 256,
        gamma: float = 0.99,
        actor_lr: float = 3.0e-4,
        critic_lr: float = 3.0e-4,
        encoder_lr: float = 3.0e-4,
        batch_size: int = 256,
        grad_clip_norm: Optional[float] = 10.0,
        policy_delay: int = 2,
        target_update_rate: int = 250,
        target_policy_smoothing_noise_std: float = 0.2,
        noise_clip: float = 0.5,
        max_action: float = 1.0,
        actor_activation: str = "relu",
        critic_activation: str = "elu",
        encoder_activation: str = "elu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        buffer_size: int = 50_000,
        replay_frac_priority: float = 0.4,
        replay_frac_recent: float = 0.3,
        replay_recent_window: int = 1_000,
        replay_alpha: float = 0.4,
        min_priority: float = 1.0,
        std_start: float = 0.2,
        std_end: float = 0.02,
        std_decay_rate: float = 0.99995,
        std_decay_steps: int = 100_000,
        std_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
        bc_lambda_scale: float = 1.0,
        actor_freeze: int = 0,
        device: Optional[torch.device] = None,
        use_adamw: bool = False,
        seed: Optional[int] = None,
        use_checkpoints: bool = False,
    ):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            set_global_seeds(self.seed)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.zs_dim = int(zs_dim)
        self.gamma = float(gamma)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.encoder_lr = float(encoder_lr)
        self.batch_size = int(batch_size)
        self.grad_clip_norm = grad_clip_norm
        self.policy_delay = int(policy_delay)
        self.target_update_rate = int(target_update_rate)
        self.t_std = float(target_policy_smoothing_noise_std)
        self.noise_clip = float(noise_clip)
        self.max_action = float(max_action)
        self.min_priority = float(min_priority)
        self.bc_lambda_scale = float(bc_lambda_scale)
        self.actor_freeze = int(actor_freeze)
        self.use_checkpoints = bool(use_checkpoints)
        self.use_adamw = bool(use_adamw)
        self.n_step = 1
        self.multistep_mode = "one_step"
        self.lambda_value = None

        encoder_hidden = list(actor_hidden if encoder_hidden is None else encoder_hidden)
        self.actor = TD7Actor(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=list(actor_hidden),
            activation=actor_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            max_action=max_action,
        ).to(self.device)
        self.actor_target = TD7Actor(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=list(actor_hidden),
            activation=actor_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            max_action=max_action,
        ).to(self.device)
        self.checkpoint_actor = TD7Actor(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=list(actor_hidden),
            activation=actor_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            max_action=max_action,
        ).to(self.device)

        self.critic = TD7Critic(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=list(critic_hidden),
            activation=critic_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)
        self.critic_target = TD7Critic(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=list(critic_hidden),
            activation=critic_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)

        self.encoder = TD7Encoder(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=encoder_hidden,
            activation=encoder_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)
        self.fixed_encoder = TD7Encoder(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=encoder_hidden,
            activation=encoder_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)
        self.fixed_encoder_target = TD7Encoder(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=encoder_hidden,
            activation=encoder_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)
        self.checkpoint_encoder = TD7Encoder(
            self.state_dim,
            self.action_dim,
            zs_dim=self.zs_dim,
            hidden_dims=encoder_hidden,
            activation=encoder_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.checkpoint_actor, self.actor)
        hard_update(self.critic_target, self.critic)
        hard_update(self.fixed_encoder, self.encoder)
        hard_update(self.fixed_encoder_target, self.fixed_encoder)
        hard_update(self.checkpoint_encoder, self.fixed_encoder)
        set_requires_grad(self.fixed_encoder, False)
        set_requires_grad(self.fixed_encoder_target, False)
        set_requires_grad(self.checkpoint_encoder, False)
        set_requires_grad(self.checkpoint_actor, False)

        optimizer_cls = optim.AdamW if use_adamw else optim.Adam
        self.actor_optimizer = optimizer_cls(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optimizer_cls(self.critic.parameters(), lr=self.critic_lr)
        self.encoder_optimizer = optimizer_cls(self.encoder.parameters(), lr=self.encoder_lr)

        self.buffer = HybridLAPReplayBuffer(
            buffer_size,
            self.state_dim,
            self.action_dim,
            default_discount=self.gamma,
            alpha=replay_alpha,
            min_priority=min_priority,
            frac_priority=replay_frac_priority,
            frac_recent=replay_frac_recent,
            recent_window=replay_recent_window,
        )

        self.expl_sched = GaussianNoiseSchedule(
            std_start=std_start,
            std_end=std_end,
            decay_steps=std_decay_steps,
            decay_rate=std_decay_rate,
            mode=std_decay_mode,
        )
        self.steps = 0
        self.train_steps = 0
        self.total_it = 0
        self.last_exploration_value = 0.0
        self.last_param_noise_scale = 0.0
        self.checkpoint_update_count = 0

        self.q_value_min = float("inf")
        self.q_value_max = float("-inf")
        self.q_target_min = 0.0
        self.q_target_max = 0.0

        self.actor_losses, self.critic_losses, self.encoder_losses = [], [], []
        self.critic_q1_trace, self.critic_q2_trace, self.critic_q_gap_trace = [], [], []
        self.q_value_min_trace, self.q_value_max_trace = [], []
        self.q_target_min_trace, self.q_target_max_trace = [], []
        self.priority_mean_trace, self.priority_max_trace, self.priority_min_trace = [], [], []
        self.exploration_trace, self.exploration_magnitude_trace = [], []
        self.param_noise_scale_trace = []
        self.action_saturation_trace = []
        self.reward_n_mean_trace, self.discount_n_mean_trace = [], []
        self.bootstrap_q_mean_trace, self.n_actual_mean_trace = [], []
        self.truncated_fraction_trace = []
        self.bc_active_trace, self.bc_weight_trace = [], []
        self.bc_loss_trace, self.bc_actor_target_distance_trace = [], []
        self.checkpoint_active_trace, self.checkpoint_update_trace = [], []

    def _as_batch_state(self, state) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return s.view(1, -1) if s.ndim == 1 else s

    def _record_action_diagnostics(self, action, clean_action=None):
        action = np.asarray(action, float)
        if clean_action is not None:
            clean_action = np.asarray(clean_action, float)
            self.last_exploration_value = float(np.mean(np.abs(action - clean_action)))
        sat = float(np.mean(np.abs(action) >= (self.max_action - 1.0e-6)))
        self.action_saturation_trace.append(sat)
        self.exploration_trace.append(float(self.last_exploration_value))
        self.exploration_magnitude_trace.append(float(self.last_exploration_value))
        self.param_noise_scale_trace.append(float(self.last_param_noise_scale))

    @torch.no_grad()
    def act_eval(self, state: np.ndarray, sigma_eval: float = 0.0, use_checkpoint: Optional[bool] = None) -> np.ndarray:
        del sigma_eval
        s = self._as_batch_state(state)
        use_checkpoint = self.use_checkpoints if use_checkpoint is None else bool(use_checkpoint)
        if use_checkpoint:
            zs = self.checkpoint_encoder.zs(s)
            action = self.checkpoint_actor(s, zs)
        else:
            zs = self.fixed_encoder.zs(s)
            action = self.actor(s, zs)
        return action.clamp(-self.max_action, self.max_action).cpu().numpy().reshape(-1)

    @torch.no_grad()
    def take_action(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        self.steps += 1
        clean_action = self.act_eval(state, use_checkpoint=False)
        action = clean_action.copy()
        self.last_exploration_value = 0.0
        self.last_param_noise_scale = 0.0
        if explore:
            sigma = float(self.expl_sched.value(self.steps))
            noise = np.random.randn(*action.shape) * sigma
            action = action + noise
            self.last_exploration_value = float(np.mean(np.abs(noise)))
        action = np.clip(action, -self.max_action, self.max_action)
        self._record_action_diagnostics(action, clean_action=clean_action if explore else None)
        return action

    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, bool(done))

    def flush_nstep(self):
        return 0

    def _resolve_bc_loss(self, curr: torch.Tensor, bc_context: Optional[dict]):
        if bc_context is None or not bool(bc_context.get("active", False)):
            return False, 0.0, None, None, None
        bc_weight = float(max(0.0, bc_context.get("weight", 0.0))) * self.bc_lambda_scale
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
            denom = torch.clamp(coord_weights_t.sum(dim=1), min=1.0e-8)
            bc_penalty = torch.sum(err_sq * coord_weights_t, dim=1) / denom
        bc_loss = torch.mean(bc_penalty)
        bc_distance = float(torch.norm(curr - target_action, dim=1).mean().item())
        return True, bc_weight, bc_loss, float(bc_loss.item()), bc_distance

    def train_step(self, bc_context: Optional[dict] = None) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None
        train_index_before = int(self.train_steps)
        self.train_steps += 1
        self.total_it += 1

        s, a, r, ns, done, discount_n, n_actual, idx = self.buffer.sample(self.batch_size, device=self.device)
        s = s.float()
        a = a.float()
        r = col(r.float())
        ns = ns.float()
        done = col(done.float())
        discount_n = col(discount_n.float())
        n_actual = col(n_actual.float())

        with torch.no_grad():
            next_zs_target = self.encoder.zs(ns)
        zs_current = self.encoder.zs(s)
        pred_next_zs = self.encoder.zsa(zs_current, a)
        encoder_loss = nn.functional.mse_loss(pred_next_zs, next_zs_target)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
        self.encoder_optimizer.step()
        self.encoder_losses.append(float(encoder_loss.item()))

        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(ns)
            noise = torch.empty_like(a).normal_(0.0, self.t_std)
            noise.clamp_(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(ns, fixed_target_zs) + noise).clamp(-self.max_action, self.max_action)
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            q_next = self.critic_target.combined_forward(
                ns,
                next_action,
                fixed_target_zsa,
                fixed_target_zs,
                mode="min",
            )
            clipped_next = q_next.clamp(self.q_target_min, self.q_target_max)
            bootstrap_q = clipped_next * (1.0 - done)
            y = r + discount_n * bootstrap_q
            self.q_value_min = min(self.q_value_min, float(y.min().item()))
            self.q_value_max = max(self.q_value_max, float(y.max().item()))

        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(s)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, a)
        q1, q2 = self.critic(s, a, fixed_zsa, fixed_zs)
        q1 = col(q1)
        q2 = col(q2)
        td1 = (y - q1).abs()
        td2 = (y - q2).abs()
        td_pair = torch.cat([td1, td2], dim=1)
        critic_loss = lap_huber_loss(td_pair, min_priority=self.min_priority)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        td_priority = torch.max(td1.detach(), td2.detach()).view(-1)
        self.buffer.update_priorities(idx, td_priority)
        priority_stats = self.buffer.priority_stats()

        self.critic_losses.append(float(critic_loss.item()))
        self.critic_q1_trace.append(float(q1.mean().item()))
        self.critic_q2_trace.append(float(q2.mean().item()))
        self.critic_q_gap_trace.append(float((q1 - q2).abs().mean().item()))
        self.q_value_min_trace.append(float(self.q_value_min))
        self.q_value_max_trace.append(float(self.q_value_max))
        self.q_target_min_trace.append(float(self.q_target_min))
        self.q_target_max_trace.append(float(self.q_target_max))
        self.priority_mean_trace.append(priority_stats["priority_mean"])
        self.priority_max_trace.append(priority_stats["priority_max"])
        self.priority_min_trace.append(priority_stats["priority_min"])
        self.reward_n_mean_trace.append(float(r.mean().item()))
        self.discount_n_mean_trace.append(float(discount_n.mean().item()))
        self.bootstrap_q_mean_trace.append(float(bootstrap_q.mean().item()))
        self.n_actual_mean_trace.append(float(n_actual.mean().item()))
        self.truncated_fraction_trace.append(0.0)

        actor_slot = bool(self.train_steps % self.policy_delay == 0)
        actor_updated = False
        actor_loss_value = None
        bc_active = False
        bc_weight = 0.0
        bc_loss_value = None
        bc_actor_target_distance = None

        if actor_slot:
            with torch.no_grad():
                actor_zs = self.fixed_encoder.zs(s)
            curr = self.actor(s, actor_zs)
            actor_zsa = self.fixed_encoder.zsa(actor_zs, curr)
            q_for_actor = self.critic.combined_forward(s, curr, actor_zsa, actor_zs, mode="mean")
            actor_loss = -torch.mean(q_for_actor)
            bc_active, bc_weight, bc_loss, bc_loss_value, bc_actor_target_distance = self._resolve_bc_loss(
                curr,
                bc_context,
            )
            if bc_loss is not None:
                actor_loss = actor_loss + bc_weight * bc_loss
            actor_loss_value = float(actor_loss.item())
            if self.train_steps >= self.actor_freeze:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()
                actor_updated = True
            self.actor_losses.append(actor_loss_value)

        target_updated = False
        if self.train_steps % self.target_update_rate == 0:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)
            set_requires_grad(self.fixed_encoder, True)
            hard_update(self.fixed_encoder, self.encoder)
            set_requires_grad(self.fixed_encoder, False)
            set_requires_grad(self.fixed_encoder_target, True)
            hard_update(self.fixed_encoder_target, self.fixed_encoder)
            set_requires_grad(self.fixed_encoder_target, False)
            self.q_target_min = self.q_value_min if np.isfinite(self.q_value_min) else 0.0
            self.q_target_max = self.q_value_max if np.isfinite(self.q_value_max) else 0.0
            self.buffer.reset_max_priority()
            target_updated = True

        self.bc_active_trace.append(float(bc_active))
        self.bc_weight_trace.append(float(bc_weight))
        self.bc_loss_trace.append(np.nan if bc_loss_value is None else float(bc_loss_value))
        self.bc_actor_target_distance_trace.append(
            np.nan if bc_actor_target_distance is None else float(bc_actor_target_distance)
        )
        self.checkpoint_active_trace.append(float(self.use_checkpoints))
        self.checkpoint_update_trace.append(float(self.checkpoint_update_count))

        return {
            "critic_updated": True,
            "encoder_updated": True,
            "actor_slot": actor_slot,
            "actor_updated": actor_updated,
            "target_updated": target_updated,
            "critic_loss": float(critic_loss.item()),
            "encoder_loss": float(encoder_loss.item()),
            "actor_loss": actor_loss_value,
            "bc_active": bool(bc_active),
            "bc_weight": float(bc_weight),
            "bc_loss": bc_loss_value,
            "bc_actor_target_distance": bc_actor_target_distance,
            "train_index_before": train_index_before,
            "train_index_after": int(self.train_steps),
        }

    def update_checkpoint(self) -> None:
        hard_update(self.checkpoint_actor, self.actor)
        set_requires_grad(self.checkpoint_encoder, True)
        hard_update(self.checkpoint_encoder, self.fixed_encoder)
        set_requires_grad(self.checkpoint_encoder, False)
        self.checkpoint_update_count += 1

    def save(self, directory: str, prefix: str = "td7", include_optim: bool = False) -> str:
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"{prefix}_{timestamp}.pkl")
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "checkpoint_actor_state_dict": self.checkpoint_actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "fixed_encoder_state_dict": self.fixed_encoder.state_dict(),
            "fixed_encoder_target_state_dict": self.fixed_encoder_target.state_dict(),
            "checkpoint_encoder_state_dict": self.checkpoint_encoder.state_dict(),
            "hparams": {
                "gamma": self.gamma,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "encoder_lr": self.encoder_lr,
                "batch_size": self.batch_size,
                "policy_delay": self.policy_delay,
                "target_update_rate": self.target_update_rate,
                "t_std": self.t_std,
                "noise_clip": self.noise_clip,
                "max_action": self.max_action,
                "zs_dim": self.zs_dim,
                "min_priority": self.min_priority,
                "bc_lambda_scale": self.bc_lambda_scale,
                "actor_freeze": self.actor_freeze,
                "n_step": self.n_step,
                "multistep_mode": self.multistep_mode,
                "lambda_value": self.lambda_value,
                "seed": self.seed,
                "steps": self.steps,
                "train_steps": self.train_steps,
                "total_it": self.total_it,
                "q_value_min": self.q_value_min,
                "q_value_max": self.q_value_max,
                "q_target_min": self.q_target_min,
                "q_target_max": self.q_target_max,
                "checkpoint_update_count": self.checkpoint_update_count,
                "use_checkpoints": self.use_checkpoints,
            },
        }
        if include_optim:
            payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
            payload["encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved TD7 checkpoint to: {path}")
        return path

    def load(self, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        self.encoder.load_state_dict(payload["encoder_state_dict"])
        if "actor_target_state_dict" in payload:
            self.actor_target.load_state_dict(payload["actor_target_state_dict"])
        else:
            hard_update(self.actor_target, self.actor)
        if "critic_target_state_dict" in payload:
            self.critic_target.load_state_dict(payload["critic_target_state_dict"])
        else:
            hard_update(self.critic_target, self.critic)
        for key, module, fallback in (
            ("fixed_encoder_state_dict", self.fixed_encoder, self.encoder),
            ("fixed_encoder_target_state_dict", self.fixed_encoder_target, self.fixed_encoder),
            ("checkpoint_encoder_state_dict", self.checkpoint_encoder, self.fixed_encoder),
            ("checkpoint_actor_state_dict", self.checkpoint_actor, self.actor),
        ):
            if key in payload:
                module.load_state_dict(payload[key])
            else:
                hard_update(module, fallback)
        hparams = payload.get("hparams", {})
        self.steps = int(hparams.get("steps", self.steps))
        self.train_steps = int(hparams.get("train_steps", self.train_steps))
        self.total_it = int(hparams.get("total_it", self.total_it))
        self.q_value_min = float(hparams.get("q_value_min", self.q_value_min))
        self.q_value_max = float(hparams.get("q_value_max", self.q_value_max))
        self.q_target_min = float(hparams.get("q_target_min", self.q_target_min))
        self.q_target_max = float(hparams.get("q_target_max", self.q_target_max))
        self.checkpoint_update_count = int(hparams.get("checkpoint_update_count", self.checkpoint_update_count))
        self.use_checkpoints = bool(hparams.get("use_checkpoints", self.use_checkpoints))
        set_requires_grad(self.fixed_encoder, False)
        set_requires_grad(self.fixed_encoder_target, False)
        set_requires_grad(self.checkpoint_encoder, False)
        set_requires_grad(self.checkpoint_actor, False)
        print(f"TD7 agent loaded successfully from: {path}")
