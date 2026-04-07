import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet layer.

    Noise use is controlled explicitly through `set_noise_enabled(...)` so
    evaluation can remain deterministic even if the module stays in train mode.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)
        self.noise_enabled = True

        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.register_buffer("weight_epsilon", torch.zeros(self.out_features, self.in_features))

        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        self.register_buffer("bias_epsilon", torch.zeros(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features).to(self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features).to(self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def set_noise_enabled(self, enabled):
        self.noise_enabled = bool(enabled)

    def mean_abs_sigma(self):
        return 0.5 * (
            float(self.weight_sigma.detach().abs().mean().item())
            + float(self.bias_sigma.detach().abs().mean().item())
        )

    def forward(self, x):
        if self.noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def reset_module_noise(module):
    for submodule in module.modules():
        if submodule is module:
            continue
        if hasattr(submodule, "reset_noise"):
            submodule.reset_noise()


def set_module_noise_enabled(module, enabled):
    for submodule in module.modules():
        if submodule is module:
            continue
        if hasattr(submodule, "set_noise_enabled"):
            submodule.set_noise_enabled(enabled)


def mean_module_abs_sigma(module):
    sigmas = []
    for submodule in module.modules():
        if submodule is module:
            continue
        if hasattr(submodule, "mean_abs_sigma"):
            sigmas.append(float(submodule.mean_abs_sigma()))
    if not sigmas:
        return 0.0
    return float(sum(sigmas) / len(sigmas))
