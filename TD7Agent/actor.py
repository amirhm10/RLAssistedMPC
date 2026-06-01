from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from TD7Agent.encoder import AvgL1Norm
from utils.helpers_net import build_network, init_layer


class TD7Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hidden_dims: List[int] | None = None,
        activation: str = "relu",
        state_feature_dim: int | None = None,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        max_action: float = 1.0,
        norm_eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = [256, 256] if hidden_dims is None else list(hidden_dims)
        feature_dim = int(state_feature_dim or (hidden_dims[0] if hidden_dims else zs_dim))
        self.max_action = float(max_action)
        self.norm = AvgL1Norm(norm_eps)
        self.state_feature = nn.Linear(int(state_dim), feature_dim)
        init_layer(self.state_feature, non_linearity="linear")
        self.policy = build_network(
            in_dim=feature_dim + int(zs_dim),
            hidden_dims=hidden_dims,
            out_dim=int(action_dim),
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="pi",
        )
        out = getattr(self.policy, "pi_out")
        nn.init.uniform_(out.weight, -1.0e-3, 1.0e-3)
        nn.init.uniform_(out.bias, -1.0e-3, 1.0e-3)

    def forward(self, state: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        state_feat = self.norm(self.state_feature(state))
        action = self.policy(torch.cat([state_feat, zs], dim=1))
        return torch.tanh(action) * self.max_action
