import torch
import torch.nn as nn
from typing import List, Literal, Optional
from dataclasses import dataclass
import torch.optim as optim
import numpy as np
import pickle
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# importing nets and buffer
from TD3Agent.critic import Critic
from SACAgent.gaussian_actor import GaussianActor
from TD3Agent.replay_buffer import PERRecentReplayBuffer
from utils.nstep import NStepAccumulator
from utils.nstep_targets import build_endpoint_bootstrap_target, build_truncated_lambda_returns

import os
from datetime import datetime


# ----------------
# Utilities
# ----------------
def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def col(x: torch.Tensor) -> torch.Tensor:
    """
    Enforce column shape [B, 1].
    """
    if x.ndim == 1:
        return x.view(-1, 1)
    return x


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

@torch.no_grad()
def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class SACAgent(nn.Module):
    """
    Soft Actor Critic Agent
    """
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
            alpha_lr: float = 1e-4,
            batch_size: int = 256,
            n_step: int = 1,
            multistep_mode: Literal["one_step", "n_step", "sac_n", "lambda"] = "one_step",
            lambda_value: float = 0.9,
            grad_clip_norm: Optional[float] = 10.0,
            # entropy/ temperature
            init_alpha: float = 0.2,
            learn_alpha: bool = True,
            target_entropy: Optional[float] = None,
            # targets update
            target_update: Literal["soft", "hard"] = "soft",
            tau: float = 0.005,
            hard_update_interval: int = 10_000,
            # architecture of the actor and critic
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
            max_action: float = 1.0,
            # buffer
            buffer_size: int = 40_000,
            replay_frac_per: float = 0.5,
            replay_frac_recent: float = 0.2,
            replay_recent_window: int = 1_000,
            replay_alpha: float = 0.6,
            replay_beta_start: float = 0.4,
            replay_beta_end: float = 1.0,
            replay_beta_steps: int = 50_000,
            # device
            device: Optional[torch.device] = None,
            use_adamw: bool = True,
            actor_freeze: int = 0,
            loss_type: Literal["huber", "mse"] = "huber",
    ):
        super(SACAgent, self).__init__()
        self.device = device if device is not None else get_device()

        # hparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_step = int(n_step)
        self.sequence_len = max(1, int(n_step))
        self.multistep_mode = str(multistep_mode).lower()
        self.lambda_value = float(lambda_value)
        self.grad_clip_norm = grad_clip_norm
        self.target_update = target_update
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.max_action = float(max_action)
        self.replay_frac_per = float(replay_frac_per)
        self.replay_frac_recent = float(replay_frac_recent)
        self.replay_recent_window = int(replay_recent_window)
        self.replay_alpha = float(replay_alpha)
        self.replay_beta_start = float(replay_beta_start)
        self.replay_beta_end = float(replay_beta_end)
        self.replay_beta_steps = int(replay_beta_steps)
        self.actor_freeze = int(actor_freeze)
        self.loss_type = str(loss_type).lower()
        if self.multistep_mode not in {"one_step", "n_step", "sac_n", "lambda"}:
            raise ValueError("multistep_mode must be 'one_step', 'n_step', 'sac_n', or 'lambda'.")
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.alpha_lr = float(alpha_lr)
        self.init_alpha = float(init_alpha)

        # ---- critic (double Q) ----
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)

        self.critic_target = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)

        hard_update(self.critic_target, self.critic)

        self.loss_fn_critic = make_loss_fn(self.loss_type)

        # ---- actor (Gaussian policy) ----
        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden,
            activation=activation,
            use_layer_norm=use_layernorm,
            dropout=dropout,
            max_action=max_action,
        ).to(self.device)

        # ---- temperature (alpha) ----
        self.learn_alpha = learn_alpha
        # log_alpha will be parameterized so alpha = exp(log_alpha) > 0
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )

        # --- optimizers and loss function ---
        if use_adamw:
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0.0)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=alpha_lr, weight_decay=0.0)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        if target_entropy is None:
            # default heuristic
            self.target_entropy = - float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # ---- replay buffer ----
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

        # ---- counters / logs ----
        self.train_steps = 0
        self.total_it = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.alphas = []
        self.entropy_trace = []
        self.mean_log_prob_trace = []
        self.reward_n_mean_trace = []
        self.discount_n_mean_trace = []
        self.bootstrap_q_mean_trace = []
        self.n_actual_mean_trace = []
        self.truncated_fraction_trace = []
        self.lambda_return_mean_trace = []
        self.target_logprob_mean_trace = []
        self.bc_active_trace = []
        self.bc_weight_trace = []
        self.bc_loss_trace = []
        self.bc_actor_target_distance_trace = []

    # ---- interactions ----
    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Map a state to an action
        - eval_mode=True: deterministic mean
        - eval_mode=False: stochastic
        """
        self.actor.eval()
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        if eval_mode:
            a = self.actor.deterministic_action(s)
        else:
            a, log_prob, mean_a = self.actor.sample(s)
        self.actor.train()
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def take_action(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        """
        In same style with TD3
        """
        return self.act(state, eval_mode=not explore)

    @torch.no_grad()
    def act_eval(self, state: np.ndarray, sigma_eval: float = 0.0) -> np.ndarray:
        del sigma_eval
        a = self.act(state, eval_mode=True)
        return a

    def push(self, s, a, r, ns, done):
        """
        store an experience into the buffer
        """
        if self.multistep_mode in {"n_step", "sac_n"}:
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
        if self.multistep_mode not in {"n_step", "sac_n"}:
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


    # ---- train step ----
    def train_step(self, bc_context: dict | None = None):
        """
        One SAC update:
            1) sample batch
            2) critic update
            3) actor update
            4) alpha update
            5) target update
        """
        # check if the buffer has enough info
        if len(self.buffer) < self.batch_size:
            return None

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
                next_action_flat, logp_next_flat, _ = self.actor.sample(next_states_flat)
                q_next_flat = self.critic_target.combined_forward(next_states_flat, next_action_flat, mode="min")
                q_next_flat = col(q_next_flat)
                alpha = self.log_alpha.exp()
                bootstrap_values = (q_next_flat - alpha * logp_next_flat).view(batch_size, seq_len) * (1.0 - dones)
                targets = build_truncated_lambda_returns(
                    rewards=rewards,
                    dones=dones,
                    bootstrap_values=bootstrap_values,
                    mask=mask,
                    gamma=self.gamma,
                    lambda_value=self.lambda_value,
                )
                y = col(targets)
                target_logprob_seq = logp_next_flat.view(batch_size, seq_len)

            q1, q2 = self.critic(s, a)
            q1 = col(q1)
            q2 = col(q2)
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
            target_logprob_mean = float(target_logprob_seq.mean().item())
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
                next_action, logp_next, _ = self.actor.sample(ns)
                q_next = self.critic_target.combined_forward(ns, next_action, mode="min")
                q_next = col(q_next)
                alpha = self.log_alpha.exp()
                bootstrap_q = (q_next - alpha * logp_next) * (1.0 - done)
                if self.multistep_mode == "sac_n":
                    y = build_endpoint_bootstrap_target(r, discount_n, bootstrap_q)
                else:
                    y = r + discount_n * bootstrap_q

            q1, q2 = self.critic(s, a)
            q1 = col(q1)
            q2 = col(q2)
            l1 = self.loss_fn_critic(q1, y)
            l2 = self.loss_fn_critic(q2, y)
            critic_loss = (is_w * (l1 + l2)).mean()

            reward_trace_mean = float(r.mean().item())
            discount_trace_mean = float(discount_n.mean().item())
            bootstrap_trace_mean = float(bootstrap_q.mean().item())
            n_actual_mean = float(n_actual.mean().item())
            truncated_fraction_value = float((n_actual < float(self.n_step)).float().mean().item())
            lambda_return_mean = None
            target_logprob_mean = float(logp_next.mean().item())
            priority_index = idx

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # TD errors for PER priority
        if priority_index is not None and hasattr(self.buffer, "update_priorities"):
            td1 = (y - q1).detach().abs().view(-1)
            td2 = (y - q2).detach().abs().view(-1)
            td_errors = 0.5 * (td1 + td2)
            self.buffer.update_priorities(priority_index, td_errors)

        # ---- actor update ----
        # sample action from current policy for states s
        new_actions, logp_new, mean_new = self.actor.sample(s)
        # Q(s, a_new) using critic 1
        q_new = self.critic.combined_forward(s, new_actions, mode="q1")
        q_new = col(q_new)

        alpha = self.log_alpha.exp().detach()
        # policy loss: E[alpha * log_pi(a|s) - Q(s,a)]
        actor_loss = (alpha * logp_new - q_new).mean()
        bc_active = False
        bc_weight = 0.0
        bc_loss_value = None
        bc_actor_target_distance = None
        if bc_context is not None and bool(bc_context.get("active", False)):
            bc_active = True
            bc_weight = float(max(0.0, bc_context.get("weight", 0.0)))
            target_action = torch.as_tensor(
                np.asarray(bc_context.get("target_action"), np.float32),
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
            target_action = target_action.expand_as(mean_new)
            err_sq = (mean_new - target_action) ** 2
            coord_weights = bc_context.get("coordinate_weights")
            if coord_weights is None:
                bc_penalty = torch.mean(err_sq, dim=1)
            else:
                coord_weights_t = torch.as_tensor(
                    np.asarray(coord_weights, np.float32),
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, -1)
                if coord_weights_t.shape[1] != mean_new.shape[1]:
                    raise ValueError(
                        f"behavioral-cloning coordinate_weights size {coord_weights_t.shape[1]} "
                        f"does not match action dimension {mean_new.shape[1]}."
                    )
                denom = torch.clamp(coord_weights_t.sum(dim=1), min=1e-8)
                bc_penalty = torch.sum(err_sq * coord_weights_t, dim=1) / denom
            bc_loss = torch.mean(bc_penalty)
            actor_loss = actor_loss + bc_weight * bc_loss
            bc_loss_value = float(bc_loss.item())
            bc_actor_target_distance = float(torch.norm(mean_new - target_action, dim=1).mean().item())

        actor_updated = False
        if self.train_steps >= getattr(self, "actor_freeze", 0):
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            actor_updated = True

        # ---- temperature (alpha) update ----
        if self.learn_alpha:
            # L(alpha) = E[-alpha * (log_pi + H_target)]
            alpha_loss = -(self.log_alpha.exp() * (logp_new + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # ----- target nets update -----
        if self.target_update == "soft":
            soft_update(self.critic_target, self.critic, self.tau)
        else:
            if self.train_steps % self.hard_update_interval == 0:
                hard_update(self.critic_target, self.critic)

        self.train_steps += 1
        self.total_it += 1

        # logging
        self.actor_losses.append(float(actor_loss.item()))
        self.critic_losses.append(float(critic_loss.item()))
        self.alpha_losses.append(float(alpha_loss.item()))
        self.alphas.append(float(self.log_alpha.exp().item()))
        self.mean_log_prob_trace.append(float(logp_new.mean().item()))
        self.entropy_trace.append(float((-logp_new).mean().item()))
        self.reward_n_mean_trace.append(reward_trace_mean)
        self.discount_n_mean_trace.append(discount_trace_mean)
        self.bootstrap_q_mean_trace.append(bootstrap_trace_mean)
        self.n_actual_mean_trace.append(n_actual_mean)
        self.truncated_fraction_trace.append(truncated_fraction_value)
        self.target_logprob_mean_trace.append(target_logprob_mean)
        if lambda_return_mean is not None:
            self.lambda_return_mean_trace.append(lambda_return_mean)
        self.bc_active_trace.append(float(bc_active))
        self.bc_weight_trace.append(float(bc_weight))
        self.bc_loss_trace.append(np.nan if bc_loss_value is None else float(bc_loss_value))
        self.bc_actor_target_distance_trace.append(
            np.nan if bc_actor_target_distance is None else float(bc_actor_target_distance)
        )

        return {
            "critic_updated": True,
            "actor_slot": True,
            "actor_updated": bool(actor_updated),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "bc_active": bool(bc_active),
            "bc_weight": float(bc_weight),
            "bc_loss": bc_loss_value,
            "bc_actor_target_distance": bc_actor_target_distance,
            "train_index_before": int(self.train_steps - 1),
            "train_index_after": int(self.train_steps),
        }

    def save(
            self,
            directory: str,
            prefix: str = None,
            include_optim: bool = False,):
        """
        Save a checkpoint to `directory` with filename
        f"{prefix}_{timestamp}.pkl".
        """
        os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"{prefix}_{timestamp}.pkl")

        payload = {}

        if hasattr(self, "actor"):
            payload["actor_state_dict"] = self.actor.state_dict()
        if hasattr(self, "critic"):
            payload["critic_state_dict"] = self.critic.state_dict()

        # --- save targets if they exist ---
        if hasattr(self, "actor_target"):
            payload["actor_target_state_dict"] = self.actor_target.state_dict()
        if hasattr(self, "critic_target"):
            payload["critic_target_state_dict"] = self.critic_target.state_dict()

        # --- hyperparams ---
        hparams = {}
        for name in [
            "gamma",
            "actor_lr",
            "critic_lr",
            "batch_size",
            "grad_clip_norm",
            "loss_type",
            "policy_delay",
            "target_update",
            "tau",
            "hard_update_interval",
            "target_combine",
            "t_std",
            "noise_clip",
            "max_action",
            "actor_freeze",
            "steps",
            "train_steps",
            "total_it",
            # SAC-specific
            "alpha",
            "alpha_lr",
            "target_entropy",
            "n_step",
            "multistep_mode",
            "lambda_value",
            "replay_frac_per",
            "replay_frac_recent",
            "replay_recent_window",
            "replay_alpha",
            "replay_beta_start",
            "replay_beta_end",
            "replay_beta_steps",
        ]:
            if hasattr(self, name):
                val = getattr(self, name)
                if hasattr(val, "item"):
                    val = float(val.item())
                hparams[name] = val

        if hasattr(self, "log_alpha"):
            # store as a scalar float
            hparams["log_alpha"] = float(self.log_alpha.detach().cpu().item())

        payload["hparams"] = hparams

        # --- optimizer states ---
        if include_optim:
            if hasattr(self, "actor_optimizer"):
                payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            if hasattr(self, "critic_optimizer"):
                payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
            # SAC-specific: alpha_optimizer
            if hasattr(self, "alpha_optimizer"):
                payload["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"Saved checkpoint to: {path}")

    def load(self, path: str, load_optim: bool = False, load_alpha: bool = True):

        with open(path, "rb") as f:
            d = pickle.load(f)

        # ---- main nets ----
        if "actor_state_dict" in d and hasattr(self, "actor"):
            self.actor.load_state_dict(d["actor_state_dict"])
        else:
            print("[load] WARNING: no 'actor_state_dict' or no self.actor")

        if "critic_state_dict" in d and hasattr(self, "critic"):
            self.critic.load_state_dict(d["critic_state_dict"])
        else:
            print("[load] WARNING: no 'critic_state_dict' or no self.critic")

        # ---- targets (if class has them) ----
        if hasattr(self, "actor_target"):
            if "actor_target_state_dict" in d:
                self.actor_target.load_state_dict(d["actor_target_state_dict"])
            else:
                hard_update(self.actor_target, self.actor)

        if hasattr(self, "critic_target"):
            if "critic_target_state_dict" in d:
                self.critic_target.load_state_dict(d["critic_target_state_dict"])
            else:
                hard_update(self.critic_target, self.critic)

        # ---- hyperparams from checkpoint, if any ----
        hparams = d.get("hparams", {})

        actor_lr = getattr(self, "actor_lr", None)
        critic_lr = getattr(self, "critic_lr", None)

        if actor_lr is None:
            actor_lr = hparams.get("actor_lr", 3e-4)
            self.actor_lr = actor_lr
        if critic_lr is None:
            critic_lr = hparams.get("critic_lr", 3e-4)
            self.critic_lr = critic_lr

        # ---- (re)init optimizers ----
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=0.0
        )

        if load_optim:
            if "actor_optimizer_state_dict" in d:
                self.actor_optimizer.load_state_dict(d["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in d:
                self.critic_optimizer.load_state_dict(d["critic_optimizer_state_dict"])

            # SAC-specific: alpha optimizer if present
            if hasattr(self, "alpha_optimizer") and "alpha_optimizer_state_dict" in d:
                self.alpha_optimizer.load_state_dict(d["alpha_optimizer_state_dict"])

        # ---- SAC-specific alpha stuff (safe for TD3; just checks) ----
        if "alpha" in hparams and hasattr(self, "alpha"):
            self.alpha = hparams["alpha"]

        if load_alpha and "log_alpha" in hparams and hasattr(self, "log_alpha"):
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(hparams["log_alpha"], device=self.log_alpha.device))

        print(f"Agent loaded successfully from: {path}")
