# Disable Distillation Matrix Observer Recalculation

Date: 2026-05-08

## Summary

- changed the shared distillation notebook defaults so matrix and structured-matrix runs no longer refresh the observer when the executed prediction model changes
- left the toggle in place for explicit notebook opt-in, but the default behavior is now a fixed nominal observer

## Files Changed

- `systems/distillation/notebook_params.py`
