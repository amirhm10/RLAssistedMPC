# Polymer Residual No-Rho Distillation Alignment

## Summary

Aligned the polymer residual TD3/TD7 runtime posture with the current distillation residual posture.

## Changes

- Disabled rho residual authority in polymer residual defaults.
- Disabled appending rho to the polymer residual RL state.
- Set polymer residual post-warm action and actor freeze windows to zero.
- Switched polymer residual behavioral cloning to the distillation-style `nominal_only` target with diagnostic release gate and raw-action handoff.
- Added the distillation-style residual authority ramp and residual safety blocks to polymer residual defaults.
- Passed the new residual authority, ramp, and safety settings through both polymer residual entrypoints.

## Validation

- Static compile of polymer residual entrypoints and polymer defaults.
- Default checks confirming polymer residual now matches distillation for `authority_use_rho`, `use_rho_authority`, `append_rho_to_state`, and `residual_authority_enabled`.

## Runtime Notes

- No polymer or distillation training run was launched.
