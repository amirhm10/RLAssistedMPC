from __future__ import annotations

import numpy as np
import torch

from TD3Agent.replay_buffer import PERRecentReplayBuffer


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


class SupervisorPERRecentReplayBuffer(PERRecentReplayBuffer):
    """Mixed PER/recent/uniform replay with supervisor-gate metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_actions = np.zeros((self.capacity, self.action_dim), np.float32)
        self.supervisor_actions = np.zeros((self.capacity, self.action_dim), np.float32)
        self.previous_actions = np.zeros((self.capacity, self.action_dim), np.float32)
        self.selected_sources = np.zeros((self.capacity,), np.int32)
        self.score_policy = np.full((self.capacity,), np.nan, np.float32)
        self.score_supervisor = np.full((self.capacity,), np.nan, np.float32)
        self.advantage_policy_supervisor = np.full((self.capacity,), np.nan, np.float32)

    def _action_or_default(self, value, default, name):
        if value is None:
            arr = np.asarray(default, dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size != self.action_dim:
            raise ValueError(f"{name} has size {arr.size}, expected {self.action_dim}.")
        return arr.astype(np.float32, copy=False)

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
        policy_action=None,
        supervisor_action=None,
        previous_action=None,
        selected_source=SOURCE_SUPERVISOR,
        score_policy=np.nan,
        score_supervisor=np.nan,
        advantage_policy_supervisor=np.nan,
    ):
        idx = int(self.ptr)
        executed_action = self._action_or_default(a, a, "action")
        policy_arr = self._action_or_default(policy_action, executed_action, "policy_action")
        supervisor_arr = self._action_or_default(supervisor_action, executed_action, "supervisor_action")
        previous_arr = self._action_or_default(previous_action, executed_action, "previous_action")

        super().push(
            s,
            executed_action,
            r,
            ns,
            done,
            p0=p0,
            discount_n=discount_n,
            n_actual=n_actual,
        )

        self.policy_actions[idx] = policy_arr
        self.supervisor_actions[idx] = supervisor_arr
        self.previous_actions[idx] = previous_arr
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
