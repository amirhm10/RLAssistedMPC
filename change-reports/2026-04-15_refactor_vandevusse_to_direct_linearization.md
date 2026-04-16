# Van de Vusse Direct-Linearization Refactor

## Summary

Refactored the active Van de Vusse system-identification workflow away from the earlier FOPDT-plus-delay-absorption path and made direct local linearization the canonical nominal-model method.

## What Changed

- Replaced the active Van de Vusse system-ID helpers in `systems/vandevusse/system_id.py` with a direct-linearization workflow:
  - steady-state solve
  - continuous-time Jacobians from centered finite differences
  - ZOH discretization
  - linear-vs-nonlinear local validation
- Removed the Van de Vusse-specific FOPDT orchestration surface from `systems/vandevusse/__init__.py`.
- Reduced the canonical validation/scaling tests to:
  - `F_step.csv`
  - `QK_step.csv`
  - `combined_validation.csv`
- Updated `systems/vandevusse/config.py` and `systems/vandevusse/notebook_params.py` so the active defaults describe direct local linearization instead of FOPDT fitting.
- Refactored `vandevusse_systemIdentification_unified.ipynb` to follow the new flow end-to-end.
- Replaced `report/08_vandevusse_system_identification.tex` with `report/09_vandevusse_direct_linearization_system_id.tex`.

## Intent

The Van de Vusse benchmark should now produce its canonical nominal model from the local linearization of the nonlinear plant at the benchmark operating point. Step tests remain only for validation, scaling, and sanity checks.
