## Summary

Made the Van de Vusse baseline observer update explicit and consistent with the actual loop order, and added a notebook-visible random search utility for observer poles.

## What Changed

- Added explicit observer update modes in `utils/mpc_baseline_runner.py`:
  - `predictor_corrector_current`
  - `legacy_previous_measurement`
- Kept the legacy mode for comparison, but the Van de Vusse baseline now defaults to `predictor_corrector_current`.
- Added manual observer-pole override support in `systems/vandevusse/baseline_mpc.py`.
- Added reusable observer-pole random-search utilities in `systems/vandevusse/pole_search.py`.
- Exposed observer update mode, manual poles, and notebook-visible search controls in `systems/vandevusse/notebook_params.py` and `vandevusse_MPCOffsetFree_unified.ipynb`.
- Added run metadata for:
  - observer update mode
  - observer poles used
  - observer spectral radius
  - simple input saturation summary

## Pole Search

The new Van de Vusse pole search:

- runs in nominal mode by default
- samples 6 augmented observer poles
- supports `uniform` and `mixed` sampling modes
- rejects invalid observer candidates automatically
- scores candidates using both tracking error and oscillation-sensitive terms

The baseline notebook now shows:

- editable search controls
- a visible top-candidate table
- a `BEST_POLES = np.array([...])` line for quick reuse

## Scope

- Polymer was not changed.
- Distillation was not changed.
- The controller architecture was not redesigned.
