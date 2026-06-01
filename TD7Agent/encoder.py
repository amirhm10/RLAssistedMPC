from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils.helpers_net import build_network


class AvgL1Norm(nn.Module):
    def __init__(self, eps: float = 1.0e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return x / scale


class TD7Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hidden_dims: List[int] | None = None,
        activation: str = "elu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        norm_eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = [256, 256] if hidden_dims is None else list(hidden_dims)
        self.zs_dim = int(zs_dim)
        self.norm = AvgL1Norm(norm_eps)
        self.state_encoder = build_network(
            in_dim=int(state_dim),
            hidden_dims=hidden_dims,
            out_dim=self.zs_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="zs",
        )
        self.state_action_encoder = build_network(
            in_dim=self.zs_dim + int(action_dim),
            hidden_dims=hidden_dims,
            out_dim=self.zs_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="zsa",
        )

    def zs(self, state: torch.Tensor) -> torch.Tensor:
        return self.norm(self.state_encoder(state))

    def zsa(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.state_action_encoder(torch.cat([zs, action], dim=1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zs = self.zs(state)
        zsa = self.zsa(zs, action)
        return zs, zsa
