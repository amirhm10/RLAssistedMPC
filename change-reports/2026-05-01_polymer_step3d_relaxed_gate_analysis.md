# Polymer Step 3D Relaxed Gate Analysis

Date: 2026-05-01

## Summary

Confirmed that the latest polymer scalar and structured matrix reruns used the intended relaxed Step 3D setup and extended the main matrix recovery report with the resulting analysis.

## Changes

- verified from the saved `input_data.pkl` bundles that both latest polymer runs used:
  - Step 2 release-protected advisory caps enabled
  - behavioral cloning disabled
  - `gain_drift_thresholds_by_phase = {protected: 0.10, ramp: 0.15, full: 0.40}`
- added a dedicated analysis script:
  - `report/scripts/generate_polymer_step3d_relaxed_gate_update.py`
- generated a new figure set under:
  - `report/figures/matrix_multiplier_step3d_relaxed_gate_20260501/`
- updated `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md` with:
  - run-bundle references
  - quantitative relaxed-gate results
  - figure embeds
  - updated polymer recommendation

## Main Findings

- scalar matrix remained fully collapsed to nominal execution despite the relaxed full-phase gain threshold
- structured matrix opened a non-empty gate, but only weakly:
  - `211 / 160000` accepted steps
  - all acceptance in the full release phase
  - final executed multiplier distance still near zero
  - reward remained worse than MPC on average
- Step 4G remains the clear polymer execution baseline

## Validation

- executed `report/scripts/generate_polymer_step3d_relaxed_gate_update.py` in `rl-env`
- verified the generated CSV summaries and figure outputs under `report/figures/matrix_multiplier_step3d_relaxed_gate_20260501/`
