# Polymer SG-DQN Horizon Runners

## Summary
- Added polymer SG-DQN and SG-dueling-DQN horizon entrypoints that wrap the existing polymer horizon runners.
- The wrappers use the polymer OF-MPC horizon `(Hp, Hc) = (9, 3)` as the supervisor action through the shared horizon runner.
- Added runner hook support for `NB_CONFIGURE`, `NOTEBOOK_SOURCE_OVERRIDE`, `RUN_SUMMARY_TITLE_OVERRIDE`, and `HORIZON_AGENT_CLASS_OVERRIDE`.

## Defaults
- `run_mode="disturb"` and `state_mode="mismatch"`.
- Warm start stays at 10 subepisodes with a 3-subepisode critic-warm/action-freeze release.
- SG gate uses zero advantage margin, defaults ties to the supervisor, and allows policy gates immediately after training begins.
- Exploration is epsilon-greedy with `eps_start=0.2`, `eps_end=0.02`, and `eps_decay_steps=38_000`.
- Legacy horizon safety, release filter, reward probation, and shadow default MPC are disabled in the wrappers.

## Validation
- Added tests for polymer wrapper configuration and hook exposure.
- Validation avoids executing polymer closed-loop runners.
