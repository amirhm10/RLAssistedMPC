# 2026-04-23 Phase 1 TD3 Hidden Release

## Summary

Implemented the Phase 1 TD3 hidden-release rollout across the shared polymer and distillation notebook families:

- `matrix`
- `structured_matrix`
- `weights`
- `residual`
- `reidentification`
- `combined`

The Phase 1 behavior keeps the existing warm-start learning gate, freezes the executed TD3 action for the configured post-warm-start window, keeps critic learning active during that window, and blocks TD3 actor updates until the hidden window finishes.

## Main Changes

- Added shared Phase 1 scheduling and trace helpers in `utils/phase1_hidden_release.py`.
- Extended `TD3Agent.train_step()` to return structured critic/actor update metadata for runner-side alignment.
- Updated all shared TD3 runners to:
  - freeze executed actions during the hidden window
  - log deterministic policy actions separately from executed actions
  - track critic updates, actor slots, blocked actor updates, and applied actor updates by environment step
  - persist Phase 1 metadata into the run bundle
- Added Phase 1 plotting support in `utils/plotting_core.py`:
  - `fig_phase1_release_window_actions`
  - `fig_phase1_release_window_updates`
  - `fig_combined_phase1_release_window_td3_actions`
  - `fig_combined_phase1_release_window_td3_updates`
- Added default Phase 1 config keys to polymer and distillation notebook defaults.
- Patched the unified notebooks so the new config keys are passed into the shared runners.

## Validation

- `py_compile` passed for the updated agent, runner, plotting, and notebook-parameter modules.
- `nbformat` validation passed for all edited unified notebooks.
- Static checks confirmed the new Phase 1 defaults exist in both polymer and distillation notebook default maps.
