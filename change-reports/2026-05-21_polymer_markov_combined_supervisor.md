# Polymer Markov-Combined Supervisor Default

Date: 2026-05-21

## What Changed

- Restored an active polymer combined entrypoint as `RL_assisted_MPC_combined_unified.py`.
- Changed the active combined model-adaptation block from the legacy scalar matrix multiplier to the Markov/dynamic-matrix lifted-response correction.
- The default active combined agent set is now horizon + Markov + weights + residual.
- Legacy matrix support remains in `utils/combined_runner.py`, but `systems/polymer/notebook_params.py` defaults it off.
- Polymer Markov z-safety is enabled by default with:
  - base `z_bound = 0.05`
  - protected cap `0.025`
  - ramp cap `0.035 -> 0.05`
  - full cap `0.05`
  - probation cap `0.025`
  - vector 2-norm cap `0.075`
- The active standalone polymer Markov entrypoint no longer overrides `z_bound` to `0.08`.

## Combined Control Order

The active combined runner applies the supervisors in this order:

1. Horizon agent selects `(Hp, Hc)`.
2. Weight agent selects the `Q/R` multipliers.
3. Markov agent selects a safety-projected lifted-response correction `z`.
4. Lifted Markov MPC solves the base move with the selected horizon and weights.
5. Residual agent applies the final input correction with the existing residual authority layer.

## Diagnostics Added

Combined result bundles now include Markov diagnostics for:

- executed/requested/LS `z`
- pre-safety requested and LS `z`
- Markov action source and source fractions
- prediction score and gain drift
- effective z cap
- requested/LS norm before and after safety projection
- coordinate clipping and vector projection flags
- TD3 priority phase, authority scale, and probation flags

`plot_combined_results` now generates Markov z, safety, projection, source, mismatch, and training diagnostic figures when the Markov block is active.

## Validation

Validation was intentionally limited to static and lightweight construction checks. A full 200-episode polymer combined rollout was not run because it is expensive.
