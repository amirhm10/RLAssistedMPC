# Restore Distillation Matrix Every-Step Observer Refresh

Date: 2026-05-10

## Summary

- restored the shared distillation scalar-matrix and structured-matrix defaults to `decision_interval = 1`
- re-enabled observer redesign for distillation matrix-family runs and added an explicit per-step refresh path
- updated the shared observer helper and matrix-family runners so notebooks can request observer refresh every MPC step even when the executed assisted model repeats

## Files Changed

- `systems/distillation/notebook_params.py`
- `utils/observer.py`
- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`
