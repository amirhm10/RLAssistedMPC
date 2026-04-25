# Analyze Step 3 Strict-Gate Results

Date: 2026-04-24

## Summary

Updated the ongoing matrix-multiplier cap report with the latest polymer Step 3 strict acceptance/fallback results.

## Findings

- Step 3 strict gating made both scalar and structured runs behave almost exactly like MPC.
- The cause was fallback dominance, not candidate MPC solve failure.
- Scalar accepted only `0.72%` of live decisions after warm start and action freeze.
- Structured accepted only `0.064%` of live decisions after warm start and action freeze.
- The strict nominal-cost rule with `relative_tolerance = 0.0` and `absolute_tolerance = 1e-8` effectively asks the candidate plan to reproduce nominal MPC.
- Offline tolerance replay suggests `relative_tolerance = 1e-4` as the next conservative polymer trial.

## Files

- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
- `report/figures/matrix_multiplier_step3_strict_gate/step3_strict_reward_delta_vs_step2.png`
- `report/figures/matrix_multiplier_step3_strict_gate/step3_strict_acceptance_by_phase.png`
- `report/figures/matrix_multiplier_step3_strict_gate/step3_tolerance_acceptance_curve.png`
