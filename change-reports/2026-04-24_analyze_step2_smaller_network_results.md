# Analyze Step 2 Smaller-Network Results

Date: 2026-04-24

## Summary

Updated the ongoing matrix-multiplier cap report with the latest polymer Step 2-only runs after the TD3 default network size was reduced to `[256, 256]`.

## Findings

- Scalar matrix Step 2 became safer during protected release: protected-window reward delta improved from `-0.1075` to `+0.0204`, and raw saturation dropped from `87.5%` to `57.6%`.
- Scalar matrix tail performance decreased but stayed positive: tail reward delta moved from `+0.7758` to `+0.6366`.
- Structured matrix Step 2 had mixed behavior: protected-window reward worsened from `-0.2052` to `-0.4180`, but ramp and full-run performance improved slightly.
- The latest result bundles do not explicitly store actor/critic hidden-layer sizes, so the report records the `[256, 256]` comparison as a provenance assumption based on run timing, changed defaults, and the user's run note.

## Files

- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
- `report/figures/matrix_multiplier_step2_network_size/step2_network_size_reward_delta.png`
- `report/figures/matrix_multiplier_step2_network_size/scalar_step2_network_size_saturation_clip.png`
- `report/figures/matrix_multiplier_step2_network_size/structured_step2_network_size_saturation_clip.png`
