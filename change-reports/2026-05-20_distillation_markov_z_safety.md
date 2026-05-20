# Distillation Markov z Safety

Date: 2026-05-20

## Summary

Implemented the moderate z-safety profile for the distillation Markov TD3 workflow.

## Changes

- Changed the distillation Markov default `z_bound` from `0.05` to `0.04`.
- Added a `z_safety` configuration with dynamic effective caps:
  - protected cap `0.02`
  - ramp cap `0.03` to `0.04`
  - full cap `0.04`
  - probation cap `0.02`
  - vector 2-norm cap `0.06`
- Added shared Markov-runner safety projection for LS and TD3 z proposals before prediction scoring, MPC candidate evaluation, execution, and replay storage.
- Added uncapped z logs plus effective-cap, norm, scale, and projection-active diagnostics for requested TD3 and LS candidates.
- Clarified progress printing by reporting `TD3 executed (subepisode)` and post-warm cumulative TD3/LS/nominal source fractions instead of the ambiguous `TD3 accepted` label.

## Validation

The implementation was designed to avoid running the full Aspen distillation simulation during validation. Compile and pure helper checks should be used to confirm the code path before a long experiment run.
