from typing import List

import torch
import torch.nn as nn

from utils.helpers_net import get_activation, init_layer
from utils.noisy_layers import NoisyLinear, reset_module_noise, set_module_noise_enabled


def _make_linear(in_dim, out_dim, activation, use_noisy, sigma_init):
    if use_noisy:
        return NoisyLinear(in_dim, out_dim, sigma_init=sigma_init)
    layer = nn.Linear(in_dim, out_dim)
    init_layer(layer, non_linearity=activation)
    return layer


class DiscreteQNetwork(nn.Module):
    """
    Standard discrete Q-network for DDQN.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        use_noisy: bool = False,
        noisy_sigma_init: float = 0.5,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = list(hidden_dims)
        self.activation = str(activation)
        self.use_layernorm = bool(use_layernorm)
        self.dropout = float(dropout)
        self.use_noisy = bool(use_noisy)
        self.noisy_sigma_init = float(noisy_sigma_init)

        layers = []
        prev_dim = self.state_dim
        for idx, hidden_dim in enumerate(self.hidden_dims):
            layers.append(
                _make_linear(
                    in_dim=prev_dim,
                    out_dim=int(hidden_dim),
                    activation=self.activation,
                    use_noisy=self.use_noisy,
                    sigma_init=self.noisy_sigma_init,
                )
            )
            if self.use_layernorm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(get_activation(self.activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = int(hidden_dim)
        layers.append(
            _make_linear(
                in_dim=prev_dim,
                out_dim=self.action_dim,
                activation="linear",
                use_noisy=self.use_noisy,
                sigma_init=self.noisy_sigma_init,
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).argmax(dim=1)

    def reset_noise(self) -> None:
        reset_module_noise(self)

    def set_noise_enabled(self, enabled: bool) -> None:
        set_module_noise_enabled(self, enabled)
