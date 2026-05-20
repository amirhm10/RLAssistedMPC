# Distillation Default Network 512x5

Date: 2026-05-20

## Summary

Changed the shared distillation RL network defaults from `[256, 256]` to `[512, 512, 512, 512, 512]`.

## Rationale

This is an intentional high-capacity distillation experiment. The previous strongest Markov checkpoints used `[256, 256]`; this change tests whether much larger policies can exploit the new Markov z-safety layer and the restored `gamma = 0.99` defaults without becoming unsafe.

## Scope

- DQN horizon and dueling horizon hidden layers now resolve to `[512, 512, 512, 512, 512]`.
- TD3/SAC actor hidden layers now resolve to `[512, 512, 512, 512, 512]`.
- TD3/SAC critic hidden layers now resolve to `[512, 512, 512, 512, 512]`.
- Polymer defaults were not changed.

## Validation

Confirm the active distillation notebook defaults resolve to `[512, 512, 512, 512, 512]` across horizon, dueling, matrix, Markov, structured matrix, weights, residual, and combined families before launching long Aspen runs.
