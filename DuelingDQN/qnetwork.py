from typing import List

import torch
import torch.nn as nn

from utils.helpers_net import get_activation, init_layer


def _make_mlp_block(
    in_dim: int,
    out_dim: int,
    activation: str,
    use_layernorm: bool,
    dropout: float,
    name: str,
):
    block = nn.Sequential()
    linear = nn.Linear(in_dim, out_dim)
    init_layer(linear, non_linearity=activation)
    block.add_module(f"{name}_linear", linear)
    if use_layernorm:
        block.add_module(f"{name}_norm", nn.LayerNorm(out_dim))
    block.add_module(f"{name}_act", get_activation(activation))
    if dropout > 0.0:
        block.add_module(f"{name}_dropout", nn.Dropout(dropout))
    return block


class DuelingQNetwork(nn.Module):
    """
    Standard dueling Q-network:
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.use_layernorm = bool(use_layernorm)
        self.dropout = float(dropout)

        self.encoder = nn.Sequential()
        prev_dim = self.state_dim
        for idx, hidden_dim in enumerate(self.hidden_dims):
            self.encoder.add_module(
                f"enc_{idx}",
                _make_mlp_block(
                    in_dim=prev_dim,
                    out_dim=int(hidden_dim),
                    activation=self.activation,
                    use_layernorm=self.use_layernorm,
                    dropout=self.dropout,
                    name=f"enc_{idx}",
                ),
            )
            prev_dim = int(hidden_dim)

        value_layer = nn.Linear(prev_dim, 1)
        init_layer(value_layer, non_linearity="linear")
        advantage_layer = nn.Linear(prev_dim, self.action_dim)
        init_layer(advantage_layer, non_linearity="linear")

        self.value_head = value_layer
        self.advantage_head = advantage_layer

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if len(self.encoder) == 0:
            return state
        return self.encoder(state)

    def forward_with_streams(self, state: torch.Tensor):
        features = self.encode(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values, value, advantage

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q_values, _, _ = self.forward_with_streams(state)
        return q_values

    @torch.no_grad()
    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).argmax(dim=1)
