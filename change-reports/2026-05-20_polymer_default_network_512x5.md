# Polymer Default Network 512x5

Date: 2026-05-20

## Summary

Changed the polymer RL network defaults from `[256, 256]` to `[512, 512, 512, 512, 512]`.

## Scope

- DQN horizon and dueling horizon hidden layers now resolve to `[512, 512, 512, 512, 512]`.
- TD3/SAC actor hidden layers now resolve to `[512, 512, 512, 512, 512]`.
- TD3/SAC critic hidden layers now resolve to `[512, 512, 512, 512, 512]`.
- Matrix, Markov, structured matrix, reidentification, weights, residual, and combined polymer defaults inherit the new size where applicable.
- Distillation defaults were already changed separately in `change-reports/2026-05-20_distillation_default_network_512x5.md`.

## Validation

Confirm the active polymer notebook defaults resolve to `[512, 512, 512, 512, 512]` across horizon, dueling, matrix, Markov, structured matrix, reidentification, weights, residual, and combined families before launching long runs.
