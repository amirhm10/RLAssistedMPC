from __future__ import annotations

import numpy as np
import torch

from utils.sequence_sampling import ordered_ring_indices


class HybridLAPReplayBuffer:
    """Hybrid priority/recent/uniform replay with TD7/LAP priority semantics."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        default_discount: float = 1.0,
        alpha: float = 0.4,
        min_priority: float = 1.0,
        frac_priority: float = 0.4,
        frac_recent: float = 0.3,
        recent_window: int = 1_000,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.default_discount = float(default_discount)
        self.alpha = float(alpha)
        self.min_priority = float(min_priority)
        self.frac_priority = float(frac_priority)
        self.frac_recent = float(frac_recent)
        self.recent_window = int(recent_window)

        self.ptr = 0
        self.size = 0
        self.step_counter = 0
        self.current_episode_id = 0
        self._max_priority = max(1.0, self.min_priority**max(self.alpha, 0.0))

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.zeros((self.capacity,), dtype=np.float32)
        self.n_actual = np.ones((self.capacity,), dtype=np.int32)
        self.episode_ids = np.zeros((self.capacity,), dtype=np.int64)
        self.birth_step = np.zeros((self.capacity,), dtype=np.int64)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self):
        return self.size

    def _ordered_indices(self) -> np.ndarray:
        return ordered_ring_indices(self.size, self.ptr, self.capacity)

    def _priority_from_td(self, td_errors) -> np.ndarray:
        td = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        priority = np.maximum(np.abs(td), self.min_priority)
        if self.alpha != 1.0:
            priority = priority ** self.alpha
        return np.clip(priority, 1.0e-6, 1.0e6).astype(np.float32)

    def push(self, s, a, r, ns, done: bool, p0=None, discount_n=None, n_actual: int = 1):
        i = self.ptr
        done = bool(done)
        self.states[i] = np.asarray(s, np.float32)
        self.actions[i] = np.asarray(a, np.float32)
        self.rewards[i] = float(r)
        self.next_states[i] = np.asarray(ns, np.float32)
        self.dones[i] = float(done)
        self.discounts[i] = self.default_discount if discount_n is None else float(discount_n)
        self.n_actual[i] = int(n_actual)
        self.episode_ids[i] = int(self.current_episode_id)
        if p0 is None:
            priority = self._max_priority
        else:
            priority = float(self._priority_from_td([p0])[0])
        self.priorities[i] = priority
        self._max_priority = max(self._max_priority, priority)
        self.birth_step[i] = self.step_counter
        self.step_counter += 1

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if done:
            self.current_episode_id += 1

    def _sample_recent(self, k: int) -> np.ndarray:
        if k <= 0:
            return np.asarray([], dtype=np.int64)
        all_idx = np.arange(self.size, dtype=np.int64)
        window = min(self.size, max(1, int(self.recent_window)))
        cutoff = np.partition(self.birth_step[: self.size], -window)[-window]
        pool = all_idx[self.birth_step[: self.size] >= cutoff]
        if pool.size == 0:
            pool = all_idx
        return np.random.choice(pool, size=k, replace=(pool.size < k))

    def _sample_priority(self, k: int) -> np.ndarray:
        if k <= 0:
            return np.asarray([], dtype=np.int64)
        priorities = np.maximum(self.priorities[: self.size], 1.0e-12)
        total = float(priorities.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.random.choice(self.size, size=k, replace=True)
        probs = priorities / total
        return np.random.choice(self.size, size=k, replace=True, p=probs)

    def sample(self, batch_size: int, device="cpu"):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty buffer.")
        batch_size = int(batch_size)
        frac_priority = min(max(self.frac_priority, 0.0), 1.0)
        frac_recent = min(max(self.frac_recent, 0.0), 1.0)
        if frac_priority + frac_recent > 1.0:
            scale = 1.0 / (frac_priority + frac_recent)
            frac_priority *= scale
            frac_recent *= scale
        k_priority = int(batch_size * frac_priority)
        k_recent = int(batch_size * frac_recent)
        k_uniform = batch_size - k_priority - k_recent

        idx_priority = self._sample_priority(k_priority)
        idx_recent = self._sample_recent(k_recent)
        idx_uniform = np.random.choice(self.size, size=k_uniform, replace=True) if k_uniform > 0 else np.asarray([], dtype=np.int64)
        idx = np.concatenate([idx_priority, idx_recent, idx_uniform]).astype(np.int64)
        if idx.size != batch_size:
            extra = np.random.choice(self.size, size=batch_size - idx.size, replace=True)
            idx = np.concatenate([idx, extra]).astype(np.int64)

        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
            torch.from_numpy(self.discounts[idx]).to(device),
            torch.from_numpy(self.n_actual[idx]).to(device),
            idx,
        )

    def update_priorities(self, idx, td_errors):
        priority = self._priority_from_td(td_errors)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if priority.size != idx.size:
            priority = np.resize(priority, idx.size).astype(np.float32)
        self.priorities[idx] = priority
        self._max_priority = max(self._max_priority, float(priority.max()))

    def reset_max_priority(self):
        if self.size > 0:
            self._max_priority = max(float(np.max(self.priorities[: self.size])), 1.0e-6)

    def priority_stats(self) -> dict:
        if self.size <= 0:
            return {"priority_mean": float("nan"), "priority_max": float("nan"), "priority_min": float("nan")}
        p = self.priorities[: self.size]
        return {
            "priority_mean": float(np.mean(p)),
            "priority_max": float(np.max(p)),
            "priority_min": float(np.min(p)),
        }

    def export_snapshot(self, ordered: bool = True) -> dict:
        if self.size <= 0:
            return {
                "buffer_type": type(self).__name__,
                "size": 0,
                "capacity": int(self.capacity),
                "ptr": int(self.ptr),
                "current_episode_id": int(self.current_episode_id),
                "step_counter": int(self.step_counter),
                "alpha": float(self.alpha),
                "min_priority": float(self.min_priority),
                "frac_priority": float(self.frac_priority),
                "frac_recent": float(self.frac_recent),
                "recent_window": int(self.recent_window),
            }
        idx = self._ordered_indices() if ordered else np.arange(self.size, dtype=np.int64)
        return {
            "buffer_type": type(self).__name__,
            "size": int(self.size),
            "capacity": int(self.capacity),
            "ptr": int(self.ptr),
            "current_episode_id": int(self.current_episode_id),
            "step_counter": int(self.step_counter),
            "ordered": bool(ordered),
            "alpha": float(self.alpha),
            "min_priority": float(self.min_priority),
            "frac_priority": float(self.frac_priority),
            "frac_recent": float(self.frac_recent),
            "recent_window": int(self.recent_window),
            "states": self.states[idx].copy(),
            "actions": self.actions[idx].copy(),
            "rewards": self.rewards[idx].copy(),
            "next_states": self.next_states[idx].copy(),
            "dones": self.dones[idx].copy(),
            "discounts": self.discounts[idx].copy(),
            "n_actual": self.n_actual[idx].copy(),
            "episode_ids": self.episode_ids[idx].copy(),
            "birth_step": self.birth_step[idx].copy(),
            "priorities": self.priorities[idx].copy(),
        }
