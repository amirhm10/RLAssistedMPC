# Polymer SG-TD3 Markov Runner

## Summary

- Added `sg_td3` support to the shared Markov runner.
- Added a polymer SG-TD3 Markov critic-warm entrypoint using an LS-or-MPC supervisor.
- The new runner executes the supervisor during warm start and a 3-subepisode critic-only window.
- Live Markov safety layers are disabled for the ablation, while shadow z-safety and priority diagnostics remain logged.

## Method

- Supervisor action: accepted LS Markov correction when available, otherwise zero Markov correction / OF-MPC.
- SG-TD3 action gate: compare the actor action with the supervisor action in normalized Markov action space.
- Replay: store the executed action plus policy, supervisor, previous-action, selected-source, score, and advantage metadata.
- Numerical fallback: if the selected policy action cannot solve corrected MPC, execute the current supervisor and log the fallback reason.

## Validation Plan

- Static compile for the Markov runner, polymer Markov entrypoints, SG-TD3 agent files, and integration tests.
- Direct execution of SG-TD3 unit tests and the new Markov integration test.
- No full polymer runtime is included in this patch unless run separately.
