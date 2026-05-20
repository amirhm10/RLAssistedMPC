# Distillation Default Network 128x3

Date: 2026-05-20

## Summary

- Changed shared distillation RL network defaults from `[128, 128]` to `[128, 128, 128]`.
- The update applies through `systems/distillation/notebook_params.py` to DQN horizon, dueling DQN horizon, TD3/SAC actor networks, and TD3/SAC critic networks.
- The discount factor remains unchanged at the current distillation default, `gamma = 0.99`.

## Validation

- Imported all active distillation family defaults and confirmed the resolved DQN, actor, and critic hidden layers are `[128, 128, 128]`.
- Ran `py_compile` on `systems/distillation/notebook_params.py`.
