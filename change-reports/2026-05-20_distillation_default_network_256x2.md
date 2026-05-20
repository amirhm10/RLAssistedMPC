# Distillation Default Network 256x2

Date: 2026-05-20

## Summary

Changed the shared distillation RL network defaults from `[128, 128, 128]` back to `[256, 256]`.

## Rationale

The strongest saved distillation Markov checkpoints, including the TD3-only/no-safeguard run `20260518_091937` and the best guarded run `20260518_184548`, used two hidden layers with 256 units for both actor and critic networks. This change makes new distillation runs use that architecture by default.

## Scope

- DQN horizon and dueling horizon hidden layers now resolve to `[256, 256]`.
- TD3/SAC actor hidden layers now resolve to `[256, 256]`.
- TD3/SAC critic hidden layers now resolve to `[256, 256]`.
- Polymer defaults were already `[256, 256]` and were not changed.

## Validation

Confirmed the active distillation notebook defaults resolve to `[256, 256]` across horizon, dueling, matrix, Markov, structured matrix, weights, residual, and combined families.
