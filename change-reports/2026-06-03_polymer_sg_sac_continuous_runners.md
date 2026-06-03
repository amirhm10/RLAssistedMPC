# Polymer SG-SAC Continuous Runners

Implemented opt-in polymer SG-SAC runner support for the continuous weights,
residual, and Markov families.

## What Changed

- Added SG-SAC wrapper entrypoints for:
  - `RL_assisted_MPC_weights_supervisor_gated_sac_critic_warm_unified.py`
  - `RL_assisted_MPC_residual_supervisor_gated_sac_critic_warm_unified.py`
  - `RL_assisted_MPC_markov_supervisor_gated_sac_critic_warm_unified.py`
- Extended shared weights, residual, and Markov runners to accept
  `agent_kind="sg_sac"` while preserving existing `td3`, `sg_td3`, `sac`, and
  `td7` paths.
- Reused the SG-TD3 execution principle:
  - warm-start supervisor execution,
  - 3-subepisode post-warm hidden supervisor execution,
  - critic-only training during the hidden window,
  - executed-action replay with SG metadata.
- Markov SG-SAC now constructs `SupervisorGatedSACAgent` from SAC replay and
  entropy settings, with the LS-or-MPC supervisor action.
- SG-SAC result bundles now expose:
  - `supervisor_gated_sac_enabled`
  - `supervisor_gated_td3_enabled`
  - `supervisor_gated_algorithm`
- Added `sg_sac` run-profile defaults for polymer weights, residual, and Markov.
- Added `NB_CONFIGURE`, `NOTEBOOK_SOURCE_OVERRIDE`, and
  `RUN_SUMMARY_TITLE_OVERRIDE` support to the standard residual exported runner.

## Runner Safety Defaults

- Weights: identity supervisor, nonfinite identity fallback, and
  `shadow_identity_mpc` diagnostics enabled. Reward probation and solve-failure
  fallback remain off.
- Residual: zero-residual supervisor, nonfinite zero fallback, and shadow rho,
  deadband, and direction-risk diagnostics enabled. Live rho authority,
  deadband execution, early-release guard, and reward probation remain off.
- Markov: LS-or-MPC supervisor, live `z_safety`, TD3 priority fallback,
  authority ramp, and handoff disabled. Previous manual safety settings are
  retained under `markov_shadow_safety` for diagnostics only.

## Validation

- `py_compile` for SG-SAC, shared runners, wrappers, exported scripts, and tests.
- `tests/test_supervisor_gated_sac.py`
- `tests/test_supervisor_gated_sac_runners.py`
- `tests/test_supervisor_gated_td3.py`
- `tests/test_supervisor_gated_markov_integration.py`
- `tests/test_supervisor_gated_residual_integration.py`

No polymer closed-loop runner was executed.
