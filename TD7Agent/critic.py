from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from TD7Agent.encoder import AvgL1Norm
from utils.helpers_net import build_network, init_layer


class _TD7QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int,
        hidden_dims: List[int],
        activation: str,
        state_action_feature_dim: int,
        use_layernorm: bool,
        dropout: float,
        prefix: str,
        norm_eps: float,
    ):
        super().__init__()
        self.norm = AvgL1Norm(norm_eps)
        self.state_action_feature = nn.Linear(int(state_dim) + int(action_dim), int(state_action_feature_dim))
        init_layer(self.state_action_feature, non_linearity="linear")
        self.q = build_network(
            in_dim=int(state_action_feature_dim) + 2 * int(zs_dim),
            hidden_dims=hidden_dims,
            out_dim=1,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix=prefix,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, zsa: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        sa_feat = self.norm(self.state_action_feature(torch.cat([state, action], dim=1)))
        return self.q(torch.cat([sa_feat, zsa, zs], dim=1))


class TD7Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hidden_dims: List[int] | None = None,
        activation: str = "elu",
        state_action_feature_dim: int | None = None,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        norm_eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = [256, 256] if hidden_dims is None else list(hidden_dims)
        feature_dim = int(state_action_feature_dim or (hidden_dims[0] if hidden_dims else zs_dim))
        self.q1 = _TD7QNetwork(
            state_dim,
            action_dim,
            zs_dim,
            hidden_dims,
            activation,
            feature_dim,
            use_layernorm,
            dropout,
            "q1",
            norm_eps,
        )
        self.q2 = _TD7QNetwork(
            state_dim,
            action_dim,
            zs_dim,
            hidden_dims,
            activation,
            feature_dim,
            use_layernorm,
            dropout,
            "q2",
            norm_eps,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, zsa: torch.Tensor, zs: torch.Tensor):
        return self.q1(state, action, zsa, zs), self.q2(state, action, zsa, zs)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor, zsa: torch.Tensor, zs: torch.Tensor):
        return self.q1(state, action, zsa, zs)

    def combined_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        zsa: torch.Tensor,
        zs: torch.Tensor,
        mode: str = "min",
    ):
        q1, q2 = self.forward(state, action, zsa, zs)
        if mode == "q1":
            return q1
        if mode == "min":
            return torch.min(q1, q2)
        if mode == "max":
            return torch.max(q1, q2)
        if mode == "mean":
            return 0.5 * (q1 + q2)
        raise ValueError("mode must be min/max/mean/q1")
