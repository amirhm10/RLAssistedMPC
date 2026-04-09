# Make Observer Recalculation Robust To Non-Finite Matrix Updates

## Summary

The matrix and combined supervisors could crash when
`recalculate_observer_on_matrix_change=True` if the matrix agent produced a
non-finite action or if pole placement failed for the updated matrix model.

This patch keeps the new observer-recalculation option, but makes it robust:

- non-finite matrix actions fall back to a safe nominal or last-valid action
- observer pole placement failures no longer abort the run
- the previous observer gain is kept when recalculation fails
- the saved result bundle records how often the fallback path was used

## Files Changed

- `utils/matrix_runner.py`
- `utils/combined_runner.py`

## Behavior Change

### Matrix runner

- if the matrix action contains `NaN` or `inf`, the runner now falls back to
  the nominal raw matrix action for that step
- if observer recalculation is enabled and the updated `A/C` pair is non-finite
  or `compute_observer_gain(...)` raises, the previous `L` is retained

### Combined runner

- if the matrix branch action contains `NaN` or `inf`, the runner now falls
  back to the last valid matrix action, or the nominal matrix action if none
  exists yet
- if observer recalculation is enabled and pole placement fails, the previous
  `L` is retained

## Saved Bundle Additions

Both result bundles now include:

- `nonfinite_matrix_action_count`
- `observer_recalc_fallback_count`

These fields make it clear whether the run used the safety fallback.

## Validation

- `python -m py_compile utils/matrix_runner.py utils/combined_runner.py`
- notebook wiring audit for matrix/combined observer-toggle references
