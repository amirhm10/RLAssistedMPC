# Reduce Default RL Network Sizes

Date: 2026-04-24

## Summary

Changed notebook-facing RL network defaults to two hidden layers of 256 units.

## Scope

- Polymer DQN and dueling DQN `hidden_layers`.
- Polymer TD3 and SAC `actor_hidden` / `critic_hidden`.
- Distillation DQN and dueling DQN `hidden_layers`.
- Distillation TD3 and SAC `actor_hidden` / `critic_hidden`.

Van de Vusse currently has no RL network defaults in its notebook parameter file.

## Validation

- Confirmed all polymer and distillation notebook default families with RL agent sections resolve to `[256, 256]`.
- Compiled the edited default modules with `py_compile`.
