# 2026-05-23 Distillation Controlled TD3 Authority Ramp

## Scope

Implemented the next diagnostic step for active distillation continuous TD3 families: weights, residual, and Markov. Horizon and dueling DQN were not changed.

## Motivation

The latest May 22 distillation report showed that protected BC prevented unsafe TD3 behavior, but the raw action-gap release gate never opened. As a result, weights and residual stayed nominal and Markov had zero post-warm TD3 source fraction.

## Changes

- Added `utils/td3_authority_ramp.py` with reusable controlled-authority helpers and logs.
- TD3 weights now treats the protected-BC release gate as diagnostic when the controlled ramp is enabled:
  - first post-warm multiplier deviation cap is `±0.05`;
  - cap ramps to `±0.25` over 30 subepisodes;
  - clipping is applied in physical multiplier space around identity `[1, 1, 1, 1]`.
- TD3 residual now uses the same diagnostic release-gate override:
  - first post-warm residual cap is `±0.005`;
  - cap ramps to the current residual bound `±0.02` over 30 subepisodes;
  - clipping is applied in scaled input-delta residual space before the existing rho/headroom projection.
- TD3 Markov now allows post-warm TD3 proposals through the existing z-safety and candidate-cost/priority checks even when the raw BC release gate remains blocked.
- Added authority-ramp diagnostics to result bundles:
  - live-enabled log;
  - release-gate override log;
  - cap/progress log;
  - projection-active and projection-delta logs;
  - preclip/postclip raw action logs.
- Distillation weight, residual, and Markov entrypoints now pass and print the new controlled-authority config.

## Validation

- `py_compile` passed for the changed helpers, runners, distillation defaults, and three distillation TD3 entrypoints.
- Config check confirmed:
  - weights authority ramp `0.05 -> 0.25`;
  - residual authority ramp `0.005 -> 0.02`;
  - Markov z-safety live release enabled.
- Pure helper check confirmed first post-warm cap resolution and physical multiplier clipping.

No full Aspen distillation simulation was run.
