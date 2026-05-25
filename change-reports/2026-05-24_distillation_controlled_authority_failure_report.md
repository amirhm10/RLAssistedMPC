# 2026-05-24 Distillation Controlled-Authority Failure Report

## Scope

Analyzed the latest completed distillation runs from 2026-05-23 after the controlled TD3 authority-ramp change. The report compares the latest weights, residual, horizon, dueling horizon, and Markov runs against OF-MPC and the previous blocked-safe TD3 runs.

## Outputs

- Added `report/distillation_controlled_authority_failure_analysis_2026_05_24.md`.
- Added `report/scripts/analyze_distillation_controlled_authority_failure_20260524.py`.
- Generated figures and metric tables under `report/figures/distillation_controlled_authority_failure_20260524/`.

## Main Finding

The performance collapse was caused by allowing live TD3 authority while the protected-BC release gate was still blocked. Weights saturated to ramp-extreme multipliers. Markov accepted a fixed saturated z pattern despite active reward probation and negative prediction scores. Residual was the exception because rho/headroom projection kept executed corrections small.

## Validation

- Recomputed metrics from saved result bundles using current distillation reward defaults.
- Audited the generated figures visually for readability.
- No full Aspen simulation was run.
