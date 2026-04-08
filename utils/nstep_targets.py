from __future__ import annotations

import torch


def build_endpoint_bootstrap_target(reward_value, discount_value, bootstrap_value):
    return reward_value + discount_value * bootstrap_value


def build_sac_n_endpoint_target(reward_value, discount_value, soft_bootstrap_value):
    return build_endpoint_bootstrap_target(reward_value, discount_value, soft_bootstrap_value)


def build_truncated_lambda_returns(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lambda_value: float,
) -> torch.Tensor:
    rewards = rewards.float()
    dones = dones.float()
    bootstrap_values = bootstrap_values.float()
    mask = mask.float()
    gamma = float(gamma)
    lambda_value = float(lambda_value)

    batch_size, seq_len = rewards.shape
    returns = torch.zeros((batch_size,), device=rewards.device, dtype=rewards.dtype)

    for step in range(seq_len - 1, -1, -1):
        valid = mask[:, step] > 0.0
        if not torch.any(valid):
            continue
        done_step = dones[:, step] > 0.5
        next_valid = (mask[:, step + 1] > 0.0) if step + 1 < seq_len else torch.zeros_like(valid)
        last_valid = valid & (~next_valid)

        one_step = rewards[:, step] + gamma * bootstrap_values[:, step]
        lambda_mix = rewards[:, step] + gamma * ((1.0 - lambda_value) * bootstrap_values[:, step] + lambda_value * returns)
        target_step = torch.where(done_step, rewards[:, step], torch.where(last_valid, one_step, lambda_mix))
        returns = torch.where(valid, target_step, returns)

    return returns


def build_discrete_retrace_targets(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: torch.Tensor,
    q_taken: torch.Tensor,
    target_action_prob: torch.Tensor,
    behavior_prob: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lambda_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rewards = rewards.float()
    dones = dones.float()
    bootstrap_values = bootstrap_values.float()
    q_taken = q_taken.float()
    target_action_prob = target_action_prob.float()
    behavior_prob = behavior_prob.float().clamp_min(1e-12)
    mask = mask.float()
    gamma = float(gamma)
    lambda_value = float(lambda_value)

    rho = torch.where(mask > 0.0, target_action_prob / behavior_prob, torch.zeros_like(target_action_prob))
    c = lambda_value * torch.clamp(rho, max=1.0)

    batch_size, seq_len = rewards.shape
    returns = torch.zeros((batch_size,), device=rewards.device, dtype=rewards.dtype)

    for step in range(seq_len - 1, -1, -1):
        valid = mask[:, step] > 0.0
        if not torch.any(valid):
            continue
        done_step = dones[:, step] > 0.5
        next_valid = (mask[:, step + 1] > 0.0) if step + 1 < seq_len else torch.zeros_like(valid)
        last_valid = valid & (~next_valid)

        correction = torch.zeros_like(returns)
        if step + 1 < seq_len:
            correction = c[:, step + 1] * (returns - q_taken[:, step + 1])

        target_step = rewards[:, step] + gamma * (
            bootstrap_values[:, step] + torch.where(last_valid | done_step, torch.zeros_like(correction), correction)
        )
        target_step = torch.where(done_step, rewards[:, step], target_step)
        returns = torch.where(valid, target_step, returns)

    return returns, rho, c
