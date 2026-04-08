from __future__ import annotations

import numpy as np
import torch

from utils.sequence_sampling import (
    build_sequence_index_batch,
    ordered_ring_indices,
    sample_hybrid_start_positions,
)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, default_discount: float = 1.0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.default_discount = float(default_discount)
        self.ptr = 0
        self.size = 0
        self.current_episode_id = 0

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.zeros((self.capacity,), dtype=np.float32)
        self.n_actual = np.ones((self.capacity,), dtype=np.int32)
        self.episode_ids = np.zeros((self.capacity,), dtype=np.int64)
        self.behavior_prob = np.ones((self.capacity,), dtype=np.float32)
        self.behavior_logprob = np.zeros((self.capacity,), dtype=np.float32)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
        discount_n: float | None = None,
        n_actual: int = 1,
        behavior_prob: float | None = None,
        behavior_logprob: float | None = None,
    ):
        i = self.ptr
        done = bool(done)
        self.states[i] = s
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_states[i] = s2
        self.dones[i] = float(done)
        self.discounts[i] = self.default_discount if discount_n is None else float(discount_n)
        self.n_actual[i] = int(n_actual)
        self.episode_ids[i] = int(self.current_episode_id)
        if behavior_prob is None:
            behavior_prob = 1.0
        if behavior_logprob is None:
            behavior_logprob = float(np.log(max(float(behavior_prob), 1e-12)))
        self.behavior_prob[i] = float(behavior_prob)
        self.behavior_logprob[i] = float(behavior_logprob)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if done:
            self.current_episode_id += 1

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
            torch.from_numpy(self.discounts[idx]),
            torch.from_numpy(self.n_actual[idx]),
        )

    def _ordered_indices(self) -> np.ndarray:
        return ordered_ring_indices(self.size, self.ptr, self.capacity)

    def sample_sequence(self, batch_size: int, seq_len: int, device: str | torch.device = "cpu"):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty buffer.")
        ordered_indices = self._ordered_indices()
        start_positions = np.random.choice(ordered_indices.size, size=int(batch_size), replace=True)
        batch = build_sequence_index_batch(
            ordered_indices=ordered_indices,
            episode_ids=self.episode_ids,
            dones=self.dones,
            start_positions=start_positions,
            seq_len=int(seq_len),
        )
        idx = batch.index_matrix
        return {
            "states": torch.from_numpy(self.states[idx]).to(device),
            "actions": torch.from_numpy(self.actions[idx]).to(device),
            "rewards": torch.from_numpy(self.rewards[idx]).to(device),
            "next_states": torch.from_numpy(self.next_states[idx]).to(device),
            "dones": torch.from_numpy(self.dones[idx]).to(device),
            "discounts": torch.from_numpy(self.discounts[idx]).to(device),
            "n_actual": torch.from_numpy(self.n_actual[idx]).to(device),
            "behavior_prob": torch.from_numpy(self.behavior_prob[idx]).to(device),
            "behavior_logprob": torch.from_numpy(self.behavior_logprob[idx]).to(device),
            "mask": torch.from_numpy(batch.mask).to(device),
            "lengths": torch.from_numpy(batch.lengths).to(device),
            "start_indices": batch.physical_start_indices,
            "is_w": torch.from_numpy(batch.is_weights).to(device),
        }

    def __len__(self):
        return self.size


class PERRecentReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        default_discount: float = 1.0,
        eps: float = 1e-6,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 300_000,
    ):
        super().__init__(capacity=capacity, state_dim=state_dim, default_discount=default_discount)
        self.step_counter = 0
        self.birth_step = np.zeros(self.capacity, dtype=np.int64)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_steps = int(beta_steps)
        self.beta_t = 0
        self._max_priority = 1.0

    def _beta(self):
        frac = min(1.0, self.beta_t / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(
        self,
        s,
        a,
        r,
        ns,
        done,
        p0=None,
        discount_n=None,
        n_actual=1,
        behavior_prob: float | None = None,
        behavior_logprob: float | None = None,
    ):
        i = self.ptr
        super().push(
            s,
            a,
            r,
            ns,
            done,
            discount_n=discount_n,
            n_actual=n_actual,
            behavior_prob=behavior_prob,
            behavior_logprob=behavior_logprob,
        )
        pri = float(p0) if (p0 is not None and p0 > 0) else self._max_priority
        self.priorities[i] = pri
        self._max_priority = max(self._max_priority, pri)
        self.birth_step[i] = self.step_counter
        self.step_counter += 1

    def sample(
        self,
        batch_size: int,
        device="cpu",
        frac_per: float = 0.6,
        frac_recent: float = 0.2,
        recent_window=5000,
    ):
        assert self.size > 0
        k_per = int(batch_size * frac_per)
        k_recent = int(batch_size * frac_recent)
        k_uniform = int(batch_size) - k_per - k_recent

        idx_all = np.arange(self.size)
        recent_window = min(self.size, int(recent_window))
        if recent_window <= 0:
            recent_window = self.size
        cutoff = np.partition(self.birth_step[: self.size], -recent_window)[-recent_window]
        pool_recent = idx_all[self.birth_step[: self.size] >= cutoff]
        if pool_recent.size == 0:
            pool_recent = idx_all

        idx_recent = np.random.choice(pool_recent, size=k_recent, replace=(pool_recent.size < k_recent))
        idx_uniform = np.random.choice(self.size, size=k_uniform, replace=True)

        if k_per > 0:
            pr = np.maximum(self.priorities[: self.size], self.eps)
            probs = pr ** self.alpha
            probs /= probs.sum()
            idx_per = np.random.choice(self.size, size=k_per, replace=True, p=probs)
            beta = self._beta()
            self.beta_t += 1
            w = (self.size * probs[idx_per]) ** (-beta)
            w /= np.maximum(w.max(), 1e-12)
            isw_per = w.astype(np.float32)
        else:
            idx_per = np.array([], dtype=np.int64)
            isw_per = np.array([], dtype=np.float32)

        idx = np.concatenate([idx_per, idx_recent, idx_uniform])
        is_w = np.concatenate([isw_per, np.ones(k_recent + k_uniform, dtype=np.float32)], axis=0)

        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
            torch.from_numpy(self.discounts[idx]).to(device),
            torch.from_numpy(self.n_actual[idx]).to(device),
            idx,
            torch.from_numpy(is_w).to(device),
        )

    def sample_sequence(
        self,
        batch_size: int,
        seq_len: int,
        device="cpu",
        frac_per: float = 0.6,
        frac_recent: float = 0.2,
        recent_window: int = 5000,
    ):
        ordered_indices = self._ordered_indices()
        ordered_priorities = self.priorities[ordered_indices]
        beta = self._beta()
        self.beta_t += 1
        start_positions, is_weights = sample_hybrid_start_positions(
            size=ordered_indices.size,
            batch_size=int(batch_size),
            recent_window=int(recent_window),
            recent_frac=float(frac_recent),
            per_frac=float(frac_per),
            priorities_ordered=ordered_priorities,
            alpha=self.alpha,
            beta=beta,
            eps=self.eps,
        )
        batch = build_sequence_index_batch(
            ordered_indices=ordered_indices,
            episode_ids=self.episode_ids,
            dones=self.dones,
            start_positions=start_positions,
            seq_len=int(seq_len),
            is_weights=is_weights,
        )
        idx = batch.index_matrix
        return {
            "states": torch.from_numpy(self.states[idx]).to(device),
            "actions": torch.from_numpy(self.actions[idx]).to(device),
            "rewards": torch.from_numpy(self.rewards[idx]).to(device),
            "next_states": torch.from_numpy(self.next_states[idx]).to(device),
            "dones": torch.from_numpy(self.dones[idx]).to(device),
            "discounts": torch.from_numpy(self.discounts[idx]).to(device),
            "n_actual": torch.from_numpy(self.n_actual[idx]).to(device),
            "behavior_prob": torch.from_numpy(self.behavior_prob[idx]).to(device),
            "behavior_logprob": torch.from_numpy(self.behavior_logprob[idx]).to(device),
            "mask": torch.from_numpy(batch.mask).to(device),
            "lengths": torch.from_numpy(batch.lengths).to(device),
            "start_indices": batch.physical_start_indices,
            "is_w": torch.from_numpy(batch.is_weights).to(device),
        }

    def update_priorities(self, idx, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td = td_errors.detach().abs().view(-1).cpu().numpy()
        else:
            td = np.abs(td_errors).reshape(-1)
        p = np.clip(td + self.eps, 1e-6, 1e6)
        self.priorities[idx] = p.astype(np.float32)
        self._max_priority = max(self._max_priority, float(p.max()))
